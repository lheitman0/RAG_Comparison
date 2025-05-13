"""
Hybrid RAG implementation with late fusion and cross-encoder reranking
OpenAI text embeddings + CLIP image embeddings + GTE cross-encoder reranking
"""

import os
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import chromadb
from tqdm import tqdm
from pathlib import Path
from src.utils.embeddings import embed_text, embed_image, embed_text_clip
from src.retrieval.base import RetrievalRecipe


class HybridRetriever:    
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16
    ):

        self.persist_directory = persist_directory
        self.text_model = text_model
        self.cross_encoder_name = cross_encoder
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.text_collection = self.client.get_collection("text_openai")
        self.images_collection = self.client.get_collection("figures_clip")
        
        self._cross_encoder = None
        
        self._fallback_cross_encoders = [
            "BAAI/bge-reranker-base",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ]
        
    
    @property
    def cross_encoder(self):
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                self._cross_encoder = None
                return self._cross_encoder

            try:
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_name,
                    device=self.device,
                )
            except Exception as e:
                for alt in self._fallback_cross_encoders:
                    try:
                        print(f"Attempting fallback cross-encoder {alt}")
                        self._cross_encoder = CrossEncoder(alt, device=self.device)
                        self.cross_encoder_name = alt  
                        break
                    except Exception as e_alt:
                        print(f" ‑ Failed: {e_alt}")
                else:
                    print("WARNING: No cross-encoder could be loaded!!!")
                    self._cross_encoder = None
        return self._cross_encoder
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        return 1.0 - distance
    
    def _resolve_figure_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        if "figure_path" in metadata and os.path.exists(metadata["figure_path"]):
            return metadata["figure_path"]

        if "image_path" in metadata and os.path.exists(metadata["image_path"]):
            metadata["figure_path"] = metadata["image_path"]
            return metadata["figure_path"]

        if "figure_filename" in metadata:
            figure_filename = metadata["figure_filename"]
            figure_path = os.path.join("./data", figure_filename.split("_figure_")[0], "figures", figure_filename)
            if os.path.exists(figure_path):
                metadata["figure_path"] = figure_path
                return figure_path

            alt_path = os.path.join("./data/figures", figure_filename)
            if os.path.exists(alt_path):
                metadata["figure_path"] = alt_path
                return alt_path

        return None
    
    def _cross_encoder_rerank(
        self, 
        query: str, 
        hits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not self.cross_encoder or not hits:
            return hits
        
        query_doc_pairs = [(query, hit["content"]) for hit in hits]
        
        batch_size = self.batch_size
        all_scores = []
        
        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i:i + batch_size]
            scores = self.cross_encoder.predict(batch)
            all_scores.extend(scores.tolist() if isinstance(scores, np.ndarray) else scores)
        
        for i, hit in enumerate(hits):
            cross_encoder_score = all_scores[i]
            
            normalized_score = 1 / (1 + np.exp(-cross_encoder_score)) 
            
            fusion_weight = 0.3  
            hit["score"] = fusion_weight * hit["score"] + (1 - fusion_weight) * normalized_score
        
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        return hits
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if recipe is None:
            recipe = RetrievalRecipe.hybrid()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "cross_encoder": self.cross_encoder_name if recipe.cross_encoder_rerank else None
        }
        
        α = recipe.text_to_image_ratio 
        
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
                "score": similarity * α, 
                "source": "text"
            })
        
        image_embedding = None
        if user_image and os.path.exists(user_image):
            image_embedding = embed_image(user_image)
        else:
            image_embedding = embed_text_clip(query)

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
            image_score = similarity * (1 - α)  
            
            self._resolve_figure_path(metadata)
            
            hits.append({
                "content": content,
                "metadata": metadata,
                "score": image_score,
                "source": "image"
            })
        
        hits.sort(key=lambda x: x["score"], reverse=True)
        

        if len(hits) > 1:
            top_doc = hits[0].get("metadata", {}).get("document", "")
            if top_doc:
                for hit in hits[1:]:
                    if hit.get("metadata", {}).get("document") == top_doc:
                        hit["score"] *= 1.05  
                hits.sort(key=lambda x: x["score"], reverse=True)
        
        initial_hits_count = len(hits)
        if recipe.cross_encoder_rerank and self.cross_encoder:
            rerank_hits = self._cross_encoder_rerank(query, hits[:40])
            
            hits = rerank_hits + hits[40:]
        
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        metrics["retrieval_tokens"] = recipe.n_text_results + recipe.n_image_results
        metrics["used_cross_encoder"] = recipe.cross_encoder_rerank and self.cross_encoder is not None
        
        return hits, metrics 