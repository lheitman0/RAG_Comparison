"""
OpenAI + CLIP RAG approach using OpenAI's text embeddings and CLIP for images
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import chromadb
from pathlib import Path
from src.utils.embeddings import embed_text, embed_image
from src.retrieval.base import RetrievalRecipe


class OpenAIClipRetriever:    
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        device: Optional[str] = None
    ):

        self.persist_directory = persist_directory
        self.text_model = text_model
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.text_collection = self.client.get_collection("text_openai")
        self.images_collection = self.client.get_collection("figures_clip")
        
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        return 1.0 - distance
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        if recipe is None:
            recipe = RetrievalRecipe.openai_clip()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0
        }
        
        ALPHA = recipe.text_to_image_ratio  
        
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
        
        if user_image and os.path.exists(user_image):
            image_embedding = embed_image(user_image)
            
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
                image_score = similarity * (1 - ALPHA)  
                
                if "figure_filename" in metadata:
                    figure_filename = metadata["figure_filename"]
                    figure_path = os.path.join("./data", figure_filename.split("_figure_")[0], 
                                              "figures", figure_filename)
                    
                    if os.path.exists(figure_path):
                        metadata["figure_path"] = figure_path
                
                hits.append({
                    "content": content,
                    "metadata": metadata,
                    "score": image_score,
                    "source": "image"
                })
        
        hits.sort(key=lambda x: x["score"], reverse=True)
        

        if len(hits) > 0:
            document_counts = {}
            for hit in hits[:10]:  
                doc = hit.get("metadata", {}).get("document", "")
                if doc:
                    document_counts[doc] = document_counts.get(doc, 0) + 1
            
            if document_counts:
                most_common_doc = max(document_counts.items(), key=lambda x: x[1])[0]
                
                for hit in hits:
                    if hit.get("metadata", {}).get("document") == most_common_doc:
                        hit["score"] *= 1.05 
            
            hits.sort(key=lambda x: x["score"], reverse=True)
        
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        
        return hits, metrics 