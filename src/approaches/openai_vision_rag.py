"""
OpenAI Vision RAG implementation using OpenAI's text embeddings and CLIP for images
This approach uses OpenAI's text embeddings for text and CLIP for images
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
from openai import OpenAI
from src.embeddings.vector_store import TechnicalManualVectorStore
from src.embeddings.image_embeddings import get_clip_model
from src.embeddings.projection import create_joint_embedding
from src.llm.open_source_llm import OpenSourceLLM
import torch
import base64
from PIL import Image
from tiktoken import encoding_for_model
import time

class OpenAIVisionRAG:
    def __init__(self,
                 text_model_name: str = "text-embedding-3-small",
                 persist_directory: str = "./vector_store/openai",
                 gpt_model: str = "gpt-4o",
                 cache_dir: str = "./models"):
        self.vector_store = TechnicalManualVectorStore(persist_directory=persist_directory, embedding_model="openai")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.clip_model, self.clip_processor = get_clip_model()
        
        self.text_model_name = text_model_name
        self.gpt_model = gpt_model
        
        self.llm = OpenSourceLLM(model_name=gpt_model)
        
        self.tokenizer = encoding_for_model(gpt_model)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.clip_model = self.clip_model.to(self.device)
            
        test_embedding = self._generate_text_embedding("test")
        print(f"DEBUG: Model embedding dimensions: {test_embedding.shape[0]}")
        if test_embedding.shape[0] != 1024:
            print(f"WARNING: Model produces {test_embedding.shape[0]}-dimensional embeddings, but vector store expects 1024 dimensions")
    
    def _generate_text_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.text_model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _generate_unified_embedding(self, query: str, image_path: str = None) -> np.ndarray:
        return create_joint_embedding(
            content=query,
            image_path=image_path,
            target_dim=1536,
            model_type="openai"
        )
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def retrieve(self, query: str) -> Tuple[List[str], List[str], int]:
        unified_query_embedding = self._generate_unified_embedding(query)
        
        text_results, image_results = self.vector_store.true_multimodal_query(
            query_embedding=unified_query_embedding,
            n_results=6,
            text_to_image_ratio=0.5
        )
        
        sections = [result["content"] for result in text_results]
        
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
        
        retrieval_tokens = sum(len(section.split()) for section in sections) // 4
        retrieval_tokens += len(image_results) * 10
        
        return sections, figures, retrieval_tokens
    
    def generate_answer(self, 
                       query: str, 
                       sections: List[str], 
                       figures: List[str],
                       language: str = "english") -> Tuple[str, int]:
        context = "\n\n".join(sections)
        
        language_instruction = f"Please respond in {language}" if language != "english" else ""
        
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
        
        sections, figures, retrieval_tokens = self.retrieve(query)
        
        answer, generation_tokens = self.generate_answer(query, sections, figures, language)
        
        end_time = time.time()
        response_time = end_time - start_time
        total_tokens = retrieval_tokens + generation_tokens
        
        performance = {
            "response_time": response_time,
            "total_tokens": total_tokens,
            "retrieval_tokens": retrieval_tokens,
            "generation_tokens": generation_tokens,
            "retrieved_sections": sections,
            "retrieved_figures": figures,
            "language": language,
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
                }
            },
            "billable_tokens": {
                "total": (retrieval_tokens // 2) + generation_tokens,
                "embedding": retrieval_tokens // 2,
                "generation": generation_tokens
            }
        }
        
        return answer, performance
    
    def __str__(self) -> str:
        return f"OpenAI Vision RAG ({self.text_model_name} + {self.gpt_model})" 