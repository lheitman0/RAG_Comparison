"""
OpenAI + CLIP RAG approach uses OpenAI embeddings for text and CLIP for images
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
from tiktoken import encoding_for_model
import time

class OpenAIClipRAG:    
    def __init__(self,
                 openai_model: str = "text-embedding-3-small",
                 gpt_model: str = "gpt-4o",
                 persist_directory: str = "./vector_store/openai"):
        self.vector_store = TechnicalManualVectorStore(persist_directory=persist_directory, embedding_model="openai")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.clip_model, self.clip_processor = get_clip_model()
        
        self.openai_model = openai_model
        self.gpt_model = gpt_model
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.clip_model = self.clip_model.to(self.device)
    
    def _generate_unified_embedding(self, query: str) -> np.ndarray:
        return create_joint_embedding(
            content=query,
            image_path=None,
            target_dim=1536,
            model_type="openai"
        )
    
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
                    "name": self.openai_model,
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
        return f"OpenAI + CLIP RAG ({self.openai_model} + CLIP + {self.gpt_model})" 