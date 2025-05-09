"""
The Hybrid RAG approach dynamically chooses between text-only and multimodal answer strategies
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from src.embeddings.vector_store import TechnicalManualVectorStore
from src.embeddings.image_embeddings import get_clip_model
from src.embeddings.projection import create_joint_embedding
from src.llm.open_source_llm import OpenSourceLLM
import torch
from enum import Enum
from src.query.query_processor import QueryProcessor, QueryType
from src.rag.hybrid_refinement import HybridRefinement, StrategyType
import base64
import time
from tiktoken import encoding_for_model

class Strategy(Enum):
    TEXT_ONLY = "text_only"
    UNIFIED = "unified"
    TWO_STAGE = "two_stage"
    VISION = "vision"

class HybridRAG:    
    def __init__(self,
                 text_model_name: str = "text-embedding-3-small",
                 persist_directory: str = "./vector_store/openai",
                 gpt_model: str = "gpt-4o",
                 cache_dir: str = "./models"):
        self.vector_store = TechnicalManualVectorStore(persist_directory=persist_directory)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.clip_model, self.clip_processor = get_clip_model()
        
        self.text_model_name = text_model_name
        self.gpt_model = gpt_model
        
        self.llm = OpenSourceLLM(model_name=gpt_model)
        
        self.tokenizer = encoding_for_model(gpt_model)
        
        self.query_processor = QueryProcessor(openai_model=gpt_model)
        self.refinement = HybridRefinement()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.clip_model = self.clip_model.to(self.device)
    
    def _generate_text_embedding(self, text: str) -> np.ndarray:
        return create_joint_embedding(content=text, image_path=None)
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _determine_strategy(self, query: str) -> Strategy:
        query_info = self.query_processor.process_query(query)
        
        query_type = query_info["query_type"]
        complexity = query_info["complexity"]["complexity_score"]
        
        if query_type in ["factual", "comparative"] and complexity < 0.5:
            return Strategy.TEXT_ONLY
        elif "figure" in query.lower() or "image" in query.lower() or "diagram" in query.lower():
            return Strategy.TWO_STAGE
        elif complexity > 0.7 or query_type in ["procedural", "troubleshooting"]:
            return Strategy.VISION
        else:
            return Strategy.UNIFIED
    
    def retrieve_unified(self, query: str) -> Tuple[List[str], List[str], int]:
        query_embedding = self._generate_text_embedding(query)
        
        results = self.vector_store.unified_query(
            query_embedding=query_embedding,
            n_results=6  
        )
        
        text_results = [r for r in results if r["metadata"]["content_type"] == "text"]
        image_results = [r for r in results if r["metadata"]["content_type"] == "image"]
        
        text_results = text_results[:3]
        image_results = image_results[:3]
        
        sections = [result["content"] for result in text_results]
        
        figures = self._resolve_image_paths(image_results)
        
        retrieval_tokens = sum(len(section.split()) for section in sections) // 4
        retrieval_tokens += len(image_results) * 10  
        
        return sections, figures, retrieval_tokens
    
    def retrieve_two_stage(self, query: str) -> Tuple[List[str], List[str], int]:
        query_embedding = self._generate_text_embedding(query)
        
        text_results, image_results = self.vector_store.two_stage_query(
            text_embedding=query_embedding,
            n_text_results=3,
            n_image_results=3
        )
        
        sections = [result["content"] for result in text_results]
        
        enhanced_query = self._enhance_query_with_context(query, sections)
        
        enhanced_embedding = self._generate_text_embedding(enhanced_query)
        
        _, enhanced_image_results = self.vector_store.two_stage_query(
            text_embedding=enhanced_embedding,
            n_text_results=3,
            n_image_results=3
        )
        
        image_ids = set()
        combined_image_results = []
        
        for img in enhanced_image_results:
            img_id = img["id"]
            if img_id not in image_ids:
                image_ids.add(img_id)
                combined_image_results.append(img)
        
        for img in image_results:
            img_id = img["id"]
            if img_id not in image_ids:
                image_ids.add(img_id)
                combined_image_results.append(img)
        
        figures = self._resolve_image_paths(combined_image_results[:3])  
        
        retrieval_tokens = sum(len(section.split()) for section in sections) // 4
        retrieval_tokens += len(combined_image_results) * 10
        retrieval_tokens += len(enhanced_query.split()) // 4
        
        return sections, figures, retrieval_tokens
    
    def _enhance_query_with_context(self, query: str, sections: List[str]) -> str:
        context = "\n\n".join(sections)
        
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        messages = [
            {"role": "system", "content": "You are a query enhancement system. Your task is to reformulate the original query to be more specific and accurate using the retrieved context."},
            {"role": "user", "content": f"""Original query: {query}
            
Retrieved context:
{context}

Based on this context, reformulate the query to make it more specific and aligned with the information available in the context.
Focus on extracting visual aspects that might be present in images.
Keep the enhanced query concise (max 2-3 sentences)."""}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        
        return f"{query} {enhanced_query}"
    
    def _resolve_image_paths(self, image_results: List[Dict[str, Any]]) -> List[str]:
        figures = []
        for result in image_results:
            if "metadata" in result and "figure_filename" in result["metadata"]:
                figure_filename = result["metadata"]["figure_filename"]
                
                figure_path = os.path.join("./data", figure_filename.split("_figure_")[0], "figures", figure_filename)
                
                if os.path.exists(figure_path):
                    figures.append(figure_path)
                else:
                    figure_dir = os.path.join("./data", figure_filename.split("_figure_")[0], "figures")
                    if os.path.exists(figure_dir):
                        parts = figure_filename.split("_figure_")
                        if len(parts) >= 2:
                            manual_name = parts[0]
                            figure_id_parts = parts[1].split("_", 1)
                            if len(figure_id_parts) >= 1:
                                figure_number = figure_id_parts[0]
                                
                                potential_matches = [
                                    f for f in os.listdir(figure_dir) 
                                    if f.startswith(f"{manual_name}_figure_{figure_number}_") or
                                    f.startswith(f"{manual_name}_figure_{figure_number}.")
                                ]
                                
                                if potential_matches:
                                    found_path = os.path.join(figure_dir, potential_matches[0])
                                    figures.append(found_path)
                                    print(f"Found alternative file: {found_path} for {figure_filename}")
                                else:
                                    print(f"WARNING: Image file not found: {figure_path}")
                    else:
                        print(f"WARNING: Image directory not found: {figure_dir}")
            else:
                print("WARNING: Missing metadata or figure_filename in image result")
        
        return figures
    
    def retrieve(self, query: str) -> Tuple[List[str], List[str], int]:
        strategy = self._determine_strategy(query)
        
        if strategy == Strategy.TEXT_ONLY:
            text_embedding = self._generate_text_embedding(query)
            text_results = self.vector_store.query_text(
                query_embedding=text_embedding,
                n_results=3
            )
            sections = [result["content"] for result in text_results]
            figures = []
            retrieval_tokens = sum(len(section.split()) for section in sections) // 4
        
        elif strategy == Strategy.UNIFIED:
            sections, figures, retrieval_tokens = self.retrieve_unified(query)
        
        elif strategy == Strategy.TWO_STAGE:
            sections, figures, retrieval_tokens = self.retrieve_two_stage(query)
        
        else:  # Strategy.VISION
            sections, figures, retrieval_tokens = self.retrieve_unified(query)
        
        return sections, figures, retrieval_tokens, strategy
    
    def apply_cross_modal_attention(
        self, 
        query: str, 
        sections: List[str], 
        figures: List[str]
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        if not figures:
            return sections, figures, {}
        
        attention_prompt = f"""Query: {query}

Text sections:
{chr(10).join([f"[{i+1}] {section[:200]}..." for i, section in enumerate(sections)])}

The query is requesting information that may involve visual elements. I have {len(figures)} images that might be relevant.
For each section above, score its relevance to the query on a scale of 0-1.
For each potential image-text pair, score how well they would complement each other on a scale of 0-1.

Respond with a JSON object containing:
1. 'section_scores': list of relevance scores for each section
2. 'cross_modal_scores': matrix of cross-modal relevance between each section and potential image

Format example:
{{"section_scores": [0.9, 0.4, 0.7], "cross_modal_scores": [[0.8, 0.2], [0.3, 0.5], [0.6, 0.7]]}}"""
        
        messages = [
            {"role": "system", "content": "You are a cross-modal attention system that can analyze the relationship between text and images in technical documentation."},
            {"role": "user", "content": attention_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        try:
            import json
            attention_data = json.loads(response.choices[0].message.content)
            section_scores = attention_data.get("section_scores", [1.0] * len(sections))
            cross_modal_scores = attention_data.get("cross_modal_scores", [[1.0] * len(figures)] * len(sections))
            
            if len(section_scores) != len(sections):
                section_scores = [1.0] * len(sections)
            if len(cross_modal_scores) != len(sections):
                cross_modal_scores = [[1.0] * len(figures)] * len(sections)
                
            figure_scores = [0.0] * len(figures)
            for i, section_score in enumerate(section_scores):
                for j, cross_score in enumerate(cross_modal_scores[i][:len(figures)]):
                    figure_scores[j] += section_score * cross_score
                    
            section_ranking = sorted(range(len(sections)), key=lambda i: section_scores[i], reverse=True)
            figure_ranking = sorted(range(len(figures)), key=lambda i: figure_scores[i], reverse=True)
            
            reranked_sections = [sections[i] for i in section_ranking]
            reranked_figures = [figures[i] for i in figure_ranking]
            
            attention_scores = {
                "section_scores": section_scores,
                "figure_scores": figure_scores,
                "cross_modal_scores": cross_modal_scores
            }
            
            return reranked_sections, reranked_figures, attention_scores
            
        except Exception as e:
            print(f"Error in cross-modal attention: {e}")
            return sections, figures, {}
    
    def generate_answer(self, 
                       query: str, 
                       sections: List[str], 
                       figures: List[str],
                       strategy: Strategy,
                       language: str = "english") -> Tuple[str, int]:
        context = "\n\n".join(sections)
        
        language_instruction = f"Please respond in {language}" if language != "english" else ""
        
        if strategy == Strategy.TEXT_ONLY or not figures:
            messages = [
                {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context to answer questions accurately and comprehensively {language_instruction}"},
                {"role": "user", "content": f"""Based on the following technical documentation, please answer the question

Question: {query}

Relevant Documentation:
{context}

Please provide a clear, step-by-step answer if the question is procedural"""}
            ]
            
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
        else:
            messages = [
                {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context and images to answer questions accurately and comprehensively {language_instruction}"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"""Based on the following technical documentation and images, please answer the question

Question: {query}

Relevant Documentation:
{context}

Please analyze the provided images and incorporate their information into your answer.
Please provide a clear, step-by-step answer if the question is procedural"""}
                ]}
            ]
            
            for figure_path in figures:
                base64_image = self._encode_image(figure_path)
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
        
        answer = response.choices[0].message.content
        token_count = response.usage.total_tokens
        
        return answer, token_count
    
    def answer_question(self, query: str, language: str = "english") -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        
        query_info = self.query_processor.process_query(query)
        
        sections, figures, retrieval_tokens, strategy = self.retrieve(query)
        
        if len(sections) > 0 and len(figures) > 0:
            sections, figures, attention_scores = self.apply_cross_modal_attention(query, sections, figures)
        else:
            attention_scores = {}
        
        answer, generation_tokens = self.generate_answer(query, sections, figures, strategy, language)
        
        end_time = time.time()
        response_time = end_time - start_time
        total_tokens = retrieval_tokens + generation_tokens
        
        query_processing_tokens = len(query.split()) * 2 
        
        performance = {
            "strategy": strategy.value,
            "response_time": response_time,
            "total_tokens": total_tokens + query_processing_tokens,
            "retrieval_tokens": retrieval_tokens,
            "generation_tokens": generation_tokens,
            "query_type": query_info["query_type"],
            "complexity": query_info["complexity"]["complexity_score"],
            "retrieved_sections": sections,
            "retrieved_figures": figures,
            "language": language,
            "attention_scores": attention_scores,
            "models_used": {
                "embedding": {
                    "name": self.text_model_name,
                    "type": "openai",
                    "billable": True,
                    "tokens": retrieval_tokens // 2  
                },
                "image_embedding": {
                    "name": "CLIP",
                    "type": "open_source",
                    "billable": False,
                    "tokens": len(figures) * 50 
                },
                "generation": {
                    "name": self.gpt_model,
                    "type": "openai",
                    "billable": True,
                    "tokens": generation_tokens
                },
                "query_processor": {
                    "name": self.gpt_model,  
                    "type": "openai",
                    "billable": True,
                    "tokens": query_processing_tokens
                }
            },
            "billable_tokens": {
                "total": (retrieval_tokens // 2) + generation_tokens + query_processing_tokens,
                "embedding": retrieval_tokens // 2,
                "generation": generation_tokens,
                "query_processing": query_processing_tokens
            }
        }
        
        self.refinement.record_performance(
            StrategyType(strategy.value),
            performance
        )
        
        return answer, performance
    
    def __str__(self) -> str:
        return f"Enhanced Hybrid RAG (Unified Embeddings, Cross-Modal Attention)" 