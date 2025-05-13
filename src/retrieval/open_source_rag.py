"""
Open Source RAG approach using E5 text embeddings and CLIP for images
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import chromadb
from pathlib import Path
from src.utils.embeddings import embed_text, embed_image, embed_text_clip
from src.retrieval.base import RetrievalRecipe


class OpenSourceRetriever:    
    def __init__(
        self, 
        persist_directory: str = "vector_store/opensource",
        text_model: str = "huggingface/intfloat/multilingual-e5-base",
        device: Optional[str] = None
    ):
        self.persist_directory = persist_directory
        self.text_model = text_model
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.text_collection = self.client.get_collection("text_e5")
        self.images_collection = self.client.get_collection("figures_clip")
        
        try:
            self.text_clip_collection = self.client.get_collection("text_clip_extra")
            self.has_text_clip = True
        except:
            self.has_text_clip = False
            

    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        return 1.0 - distance
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        if recipe is None:
            recipe = RetrievalRecipe.open_source()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "use_text_clip": self.has_text_clip
        }
        
        ALPHA = 0.5  
        
        device_for_clip = self.device
        
        text_embedding = embed_text(self.text_model, query)
        
        text_results = self.text_collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=recipe.n_text_results,
            include=["metadatas", "documents", "distances"]
        )
        
        hits = []
        for i in range(len(text_results["ids"][0])):
            document_id = text_results["ids"][0][i]
            content = text_results["documents"][0][i]
            metadata = text_results["metadatas"][0][i]
            distance = text_results["distances"][0][i]
            
            similarity = self._convert_distance_to_similarity(distance)
            
            hits.append({
                "content": content,
                "metadata": metadata,
                "score": similarity * ALPHA, 
                "source": "text"
            })
        
        run_image_search = True
        image_embedding = None
        if user_image and os.path.exists(user_image):
            image_embedding = embed_image(user_image)
        elif run_image_search:
            image_embedding = embed_text_clip(query)

        if image_embedding is not None:
            image_results = self.images_collection.query(
                query_embeddings=[image_embedding.tolist()],
                n_results=recipe.n_image_results,
                include=["metadatas", "documents", "distances"]
            )
            
            for i in range(len(image_results["ids"][0])):
                document_id = image_results["ids"][0][i]
                content = image_results["documents"][0][i]
                metadata = image_results["metadatas"][0][i]
                distance = image_results["distances"][0][i]
                
                similarity = self._convert_distance_to_similarity(distance)
                
                self._resolve_figure_path(metadata)
                
                hits.append({
                    "content": content,
                    "metadata": metadata,
                    "score": similarity * (1 - ALPHA),  
                    "source": "image"
                })
                
            if self.has_text_clip:
                clip_text_results = self.text_clip_collection.query(
                    query_embeddings=[image_embedding.tolist()],
                    n_results=recipe.n_image_results,
                    include=["metadatas", "documents", "distances"]
                )
                
                for i in range(len(clip_text_results["ids"][0])):
                    document_id = clip_text_results["ids"][0][i]
                    content = clip_text_results["documents"][0][i]
                    metadata = clip_text_results["metadatas"][0][i]
                    distance = clip_text_results["distances"][0][i]
                    
                    similarity = self._convert_distance_to_similarity(distance)
                    
                    self._resolve_figure_path(metadata)
                    
                    hits.append({
                        "content": content,
                        "metadata": metadata,
                        "score": similarity * (1 - ALPHA),  
                        "source": "clip_text"
                    })
        
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        metrics["clip_text_hits"] = sum(1 for hit in hits if hit["source"] == "clip_text")

        metrics["retrieval_tokens"] = (
            recipe.n_text_results + (recipe.n_image_results if image_embedding is not None else 0)
        )
        
        return hits, metrics 


    def _resolve_figure_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        if "figure_path" in metadata and os.path.exists(metadata["figure_path"]):
            return metadata["figure_path"]

        if "image_path" in metadata and os.path.exists(metadata["image_path"]):
            metadata["figure_path"] = metadata["image_path"]
            return metadata["figure_path"]

        if "figure_filename" in metadata:
            fname = metadata["figure_filename"]
            candidate = os.path.join("./data", fname.split("_figure_")[0], "figures", fname)
            if os.path.exists(candidate):
                metadata["figure_path"] = candidate
                return candidate
            # Fallback flat figures dir
            alt = os.path.join("./data/figures", fname)
            if os.path.exists(alt):
                metadata["figure_path"] = alt
                return alt
        return None 