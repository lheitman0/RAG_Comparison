"""
Open Source RAG implementation using SentenceTransformer for text and CLIP for images
This approach uses only open source models, making it free to use but potentially less accurate
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from src.embeddings.vector_store import TechnicalManualVectorStore
from src.embeddings.image_embeddings import get_clip_model
from src.embeddings.projection import create_joint_embedding
from src.llm.open_source_llm import OpenSourceLLM
import torch
from tiktoken import encoding_for_model
import time
import re

class OpenSourceRAG:
    def __init__(self,
                 text_model_name: str = "intfloat/multilingual-e5-large",
                 persist_directory: str = "./vector_store/opensource",
                 llm_model_name: str = "gpt-4o",
                 cache_dir: str = "./models"):
        self.text_vector_store = TechnicalManualVectorStore(
            persist_directory=persist_directory, 
            embedding_model="llama",
            collection_name="text"
        )
        
        self.image_vector_store = TechnicalManualVectorStore(
            persist_directory=persist_directory, 
            embedding_model="clip",
            collection_name="images"
        )
        
        print(f"Loading text model {text_model_name} from cache...")
        self.text_model = SentenceTransformer(
            text_model_name,
            cache_folder=cache_dir
        )
        
        self.clip_model, self.clip_processor = get_clip_model()
        
        self.text_model_name = text_model_name
        
        self.llm = OpenSourceLLM(model_name=llm_model_name)
        
        self.tokenizer = encoding_for_model(llm_model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.text_model = self.text_model.to(self.device)
            self.clip_model = self.clip_model.to(self.device)
            
        test_embedding = self._generate_text_embedding("test")
        print(f"DEBUG: Model embedding dimensions: {test_embedding.shape[0]}")
        if test_embedding.shape[0] != 1024:
            print(f"WARNING: Model produces {test_embedding.shape[0]}-dimensional embeddings, but vector store expects 1024 dimensions")
    
    def _generate_text_embedding(self, text: str) -> np.ndarray:
        print(f"\nDEBUG: Generating text embedding for: {text[:50]}...")
        with torch.no_grad():
            print(f"DEBUG: Device before encoding: {self.device}")
            embedding = self.text_model.encode(text, convert_to_tensor=True)
            print(f"DEBUG: Embedding device after encoding: {embedding.device}")
            
            print("DEBUG: Moving embedding to CPU")
            embedding = embedding.cpu()
            print(f"DEBUG: Final embedding device: {embedding.device}")
            print(f"DEBUG: Embedding shape: {embedding.shape}")
            
            numpy_embedding = embedding.numpy()
            print(f"DEBUG: Numpy embedding shape: {numpy_embedding.shape}")
            return numpy_embedding
    
    def _generate_image_embedding(self, text: str) -> np.ndarray:
        print(f"\nDEBUG: Generating CLIP text embedding for: {text[:50]}...")
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        print(f"DEBUG: Inputs device: {inputs['input_ids'].device}")
        
        if self.device == "cuda" or self.device == "mps":
            print(f"DEBUG: Moving inputs to {self.device}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            print(f"DEBUG: Inputs device after move: {inputs['input_ids'].device}")
        
        with torch.no_grad():
            print("DEBUG: Getting text features from CLIP model")
            text_features = self.clip_model.get_text_features(**inputs)
            print(f"DEBUG: Text features device: {text_features.device}")
            if self.device == "cuda" or self.device == "mps":
                print("DEBUG: Moving text features to CPU")
                text_features = text_features.cpu()
            print(f"DEBUG: Final text features device: {text_features.device}")
            print(f"DEBUG: Text features shape: {text_features.shape}")
            return text_features.numpy().flatten()
    
    def retrieve(self, query: str) -> Tuple[List[str], List[str], int, List[str]]:
        start_time = time.time()
        
        text_embedding = self._generate_text_embedding(query)
        
        image_embedding = self._generate_image_embedding(query)
        
        initial_results = self.text_vector_store.query_text(
            query_embedding=text_embedding,
            n_results=5
        )
        
        doc_counts = {}
        for result in initial_results:
            doc_type = result["metadata"].get("document", "")
            if doc_type:
                doc_counts[doc_type] = doc_counts.get(doc_type, 0) + 1
        
        document_filter = None
        if doc_counts:
            top_doc, top_count = max(doc_counts.items(), key=lambda x: x[1])
            if top_count >= 0.6 * len(initial_results):
                document_filter = top_doc
                print(f"Automatically detected document context: {document_filter}")
        
        section_pattern = r'(section|chapter|part)\s+(\d+(\.\d+)*)'
        section_match = re.search(section_pattern, query, re.IGNORECASE)
        section_id = section_match.group(2) if section_match else None
        
        if section_id:
            print(f"Detected section reference: {section_id}")
        
        metadata_filters = {}
        if document_filter:
            metadata_filters["document"] = document_filter
        
        if section_id:
            text_results = self.text_vector_store.query_by_section_context(
                query_embedding=text_embedding,
                section_id=section_id,
                n_results=3,
                include_siblings=True,
                include_children=True
            )
            
            image_results = self.image_vector_store.get_related_images_by_section(
                section_id=section_id,
                query_embedding=image_embedding,
                n_results=2
            )
        else:
            text_results = self.text_vector_store.query_with_metadata(
                query_embedding=text_embedding,
                n_results=3,
                metadata_filters=metadata_filters,
                content_type="text",
                boost_by_section=True
            )
            
            image_results = self.image_vector_store.query_with_metadata(
                query_embedding=image_embedding,
                n_results=2,
                metadata_filters=metadata_filters,
                content_type="image"
            )
        
        sections = []
        for i, result in enumerate(text_results):
            metadata = result["metadata"]
            
            context_header = f"SECTION {metadata.get('section_id', '')}: {metadata.get('section_title', '')}\n"
            context_header += f"FROM: {metadata.get('document', '')}\n"
            context_header += f"PATH: {metadata.get('parent_path', '')}\n"
            
            sections.append(f"{context_header}\n{result['content']}")
        
        figures = []
        captions = []
        for result in image_results:
            if "metadata" in result and "figure_filename" in result["metadata"]:
                figure_filename = result["metadata"]["figure_filename"]
                figure_dir = result["metadata"].get("figure_dir", "")
                
                caption = result["metadata"].get("semantic_caption", 
                        result["metadata"].get("caption", 
                        result["content"] if "content" in result else "No caption available"))
                
                section_context = f"From section {result['metadata'].get('section_id', '')}: {result['metadata'].get('section_title', '')}"
                
                if figure_dir and figure_filename:
                    figure_path = os.path.join(figure_dir, figure_filename)
                    if os.path.exists(figure_path):
                        figures.append(figure_path)
                        captions.append(f"Figure: {figure_filename}\n{section_context}\nDescription: {caption}")
                    else:
                        alt_path = os.path.join("data", result["metadata"].get("document", "").split("_")[0].lower() + "_manual", "figures", figure_filename)
                        if os.path.exists(alt_path):
                            figures.append(alt_path)
                            captions.append(f"Figure: {figure_filename}\n{section_context}\nDescription: {caption}")
                        else:
                            print(f"WARNING: Image file not found: {figure_path} or {alt_path}")
        
        retrieval_tokens = sum(len(section.split()) for section in sections) // 4
        
        retrieval_time = time.time() - start_time
        print(f"\nRetrieval completed in {retrieval_time:.2f} seconds")
        print(f"Found {len(sections)} relevant sections and {len(figures)} relevant figures")
        
        return sections, figures, retrieval_tokens, captions
    
    def answer_question(self, query: str, language: str = "english") -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        
        sections, figures, retrieval_tokens, captions = self.retrieve(query)
        
        if not sections:
            answer = "I couldn't find any relevant information to answer your question."
            
            performance = {
                "response_time": time.time() - start_time,
                "total_tokens": 0,
                "retrieval_tokens": 0,
                "generation_tokens": 0,
                "retrieved_sections": [],
                "retrieved_figures": []
            }
            
            return answer, performance
        
        context_sources = []
        for i, section in enumerate(sections):
            section_lines = section.split('\n')
            if len(section_lines) >= 3 and section_lines[0].startswith("SECTION"):
                section_info = section_lines[0].strip()
                document_info = section_lines[1].strip()
                context_sources.append(f"Source {i+1}: {document_info} - {section_info}")
        
        context = "\n\n".join(sections)
        
        if captions:
            figure_context = "\n\n=== RELEVANT FIGURES ===\n" + "\n\n".join(captions)
            context += figure_context
        
        if language.lower() != "english":
            language_instruction = f"Please respond in {language}"
        else:
            language_instruction = ""
        
        prompt = f"""
        Answer the following question based on the provided technical manual excerpts.
        
        QUESTION: {query}
        
        INFORMATION SOURCES:
        {' '.join(context_sources)}
        
        CONTEXT:
        {context}
        
        INSTRUCTIONS:
        1. Answer only using information from the provided context
        2. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
        3. Cite specific sections when possible (e.g., "According to Section 5.2.2.1...")
        4. Organize your answer in a clear, step-by-step format when appropriate
        5. When referring to figures, mention them explicitly
        {language_instruction}
        
        ANSWER:
        """
        
        try:
            answer = self.llm.generate(prompt)
            
            if not isinstance(answer, str):
                if answer is None:
                    answer = "I'm sorry, I couldn't generate an answer at this time."
                else:
                    answer = str(answer)
            
            prompt_tokens = len(self.tokenizer.encode(prompt))
            try:
                answer_tokens = len(self.tokenizer.encode(answer))
            except Exception:
                answer_tokens = len(answer.split()) * 4 // 3
            
            total_tokens = prompt_tokens + answer_tokens
            generation_tokens = total_tokens
            
            response_time = time.time() - start_time
            
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
                        "type": "open_source",
                        "billable": False,
                        "tokens": retrieval_tokens // 2
                    },
                    "image_embedding": {
                        "name": "CLIP",
                        "type": "open_source",
                        "billable": False,
                        "tokens": len(figures) * 50
                    },
                    "generation": {
                        "name": self.llm.model_name,
                        "type": "openai" if "gpt" in self.llm.model_name.lower() else "open_source",
                        "billable": "gpt" in self.llm.model_name.lower(),
                        "tokens": generation_tokens
                    }
                },
                "billable_tokens": {
                    "total": generation_tokens if "gpt" in self.llm.model_name.lower() else 0,
                    "embedding": 0,
                    "generation": generation_tokens if "gpt" in self.llm.model_name.lower() else 0
                }
            }
            
            return answer, performance
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(error_msg)
            
            return error_msg, {
                "response_time": time.time() - start_time,
                "error": True,
                "error_message": str(e),
                "retrieval_tokens": retrieval_tokens,
                "retrieved_sections": sections,
                "retrieved_figures": figures,
            }
    
    def __str__(self) -> str:
        return "Open Source RAG (SentenceTransformer + CLIP + Open Source LLM)"

    def generate_with_context(self, prompt: str, context: str) -> Tuple[str, int]:
        try:
            full_prompt = f"""
            Context:
            {context}
            
            Question/Instructions:
            {prompt}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful technical assistant. Use the provided context to answer questions accurately and comprehensively."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            return answer, tokens
        except Exception as e:
            print(f"Error in generate_with_context: {str(e)}")
            return f"I encountered an error: {str(e)}", 0 