"""
Hybrid RAG implementation with late fusion and cross-encoder reranking.
This approach combines OpenAI text embeddings, CLIP image embeddings,
and GTE cross-encoder reranking for optimal retrieval.
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
    """Retrieval implementation using hybrid late fusion and cross-encoder reranking."""
    
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        # Use a well-supported cross-encoder for MS-MARCO style reranking. The previous
        # `gte-base-zh-v1-mmarco` model is not available on the Hub and causes a 404.
        # Default to `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is small (~110 M),
        # multilingual enough for our manuals, and widely used.  A short fallback
        # list is tried if the primary model cannot be downloaded (e.g. offline).
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize the Hybrid Retriever.
        
        Args:
            persist_directory: Path to the vector store directory
            text_model: Model identifier for text embeddings
            cross_encoder: Model for cross-encoder reranking
            device: Device to run models on ('cuda' or 'cpu'), auto-detected if None
            batch_size: Batch size for cross-encoder inference
        """
        self.persist_directory = persist_directory
        self.text_model = text_model
        self.cross_encoder_name = cross_encoder
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get collections
        self.text_collection = self.client.get_collection("text_openai")
        self.images_collection = self.client.get_collection("figures_clip")
        
        # Lazy-load cross-encoder model
        self._cross_encoder = None
        
        # Ordered fallback models to attempt if the requested cross-encoder cannot be loaded.
        self._fallback_cross_encoders = [
            "BAAI/bge-reranker-base",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ]
        
        print(f"Initialized HybridRetriever with collections: text_openai, figures_clip")
    
    @property
    def cross_encoder(self):
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                print("WARNING: sentence-transformers not installed; cross-encoder reranking disabled ❌")
                self._cross_encoder = None
                return self._cross_encoder

            # Try the requested model first, then fallbacks
            try:
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_name,
                    device=self.device,
                )
            except Exception as e:
                print(f"WARNING: Failed to load cross-encoder '{self.cross_encoder_name}': {e}")
                # Iterate through fallbacks
                for alt in self._fallback_cross_encoders:
                    try:
                        print(f"Attempting fallback cross-encoder '{alt}' …")
                        self._cross_encoder = CrossEncoder(alt, device=self.device)
                        self.cross_encoder_name = alt  # Update to the working model
                        break
                    except Exception as e_alt:
                        print(f" ‑ Failed: {e_alt}")
                else:
                    print("WARNING: No cross-encoder could be loaded; reranking disabled ❌")
                    self._cross_encoder = None
        return self._cross_encoder
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score."""
        return 1.0 - distance
    
    def _resolve_figure_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Resolve figure path from metadata if not already present."""
        # If a canonical figure_path already exists and is valid, keep it
        if "figure_path" in metadata and os.path.exists(metadata["figure_path"]):
            return metadata["figure_path"]

        # If image_path is directly provided (absolute path), re-use it
        if "image_path" in metadata and os.path.exists(metadata["image_path"]):
            metadata["figure_path"] = metadata["image_path"]
            return metadata["figure_path"]

        # Otherwise derive from filename if present
        if "figure_filename" in metadata:
            figure_filename = metadata["figure_filename"]
            figure_path = os.path.join("./data", figure_filename.split("_figure_")[0], "figures", figure_filename)
            if os.path.exists(figure_path):
                metadata["figure_path"] = figure_path
                return figure_path

            # Try fallback path
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
        """
        Use cross-encoder for more accurate relevance scoring.
        
        Args:
            query: User query
            hits: List of initial hits with content and metadata
        
        Returns:
            Reranked list of hits
        """
        if not self.cross_encoder or not hits:
            return hits
        
        # Prepare input pairs
        query_doc_pairs = [(query, hit["content"]) for hit in hits]
        
        # Batch inference for cross-encoder
        batch_size = self.batch_size
        all_scores = []
        
        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i:i + batch_size]
            scores = self.cross_encoder.predict(batch)
            all_scores.extend(scores.tolist() if isinstance(scores, np.ndarray) else scores)
        
        # Update scores
        for i, hit in enumerate(hits):
            # Normalize cross-encoder score to 0-1 range if needed
            # GTE returns scores that can be outside 0-1 range
            cross_encoder_score = all_scores[i]
            
            # For GTE models, scores are typically in a wider range, let's normalize them
            normalized_score = 1 / (1 + np.exp(-cross_encoder_score))  # Sigmoid
            
            # Hybrid scoring: original bi-encoder score and cross-encoder score
            # The fusion weight determines influence of each component
            fusion_weight = 0.3  # Weight for original score (0.7 for cross-encoder)
            hit["score"] = fusion_weight * hit["score"] + (1 - fusion_weight) * normalized_score
        
        # Re-sort hits by adjusted score
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        return hits
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid retrieval with late fusion.
        
        Args:
            query: User query
            user_image: Optional path to a user-provided image
            recipe: Retrieval recipe with parameters
            
        Returns:
            Tuple containing:
            - List of retrieved documents (dicts with 'content', 'metadata', 'score')
            - Dictionary of retrieval metrics
        """
        if recipe is None:
            recipe = RetrievalRecipe.hybrid()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "cross_encoder": self.cross_encoder_name if recipe.cross_encoder_rerank else None
        }
        
        α = recipe.text_to_image_ratio  # Text weight (0.7 by default)
        
        # Generate text embedding
        text_embedding = embed_text(self.text_model, query)
        
        # Query text collection
        text_results = self.text_collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=recipe.n_text_results,
            include=["metadatas", "documents", "distances"]
        )
        
        # Process text results
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
                "score": similarity * α,  # Apply text weight
                "source": "text"
            })
        
        # Determine image embedding for search
        image_embedding = None
        if user_image and os.path.exists(user_image):
            image_embedding = embed_image(user_image)
        else:
            image_embedding = embed_text_clip(query)

        # Query images collection (always)
        image_results = self.images_collection.query(
            query_embeddings=[image_embedding.tolist()],
            n_results=recipe.n_image_results,
            include=["metadatas", "documents", "distances"]
        )
        
        # Process image results
        for i in range(len(image_results["ids"][0])):
            document_id = image_results["ids"][0][i]
            content = image_results["documents"][0][i]
            metadata = image_results["metadatas"][0][i]
            distance = image_results["distances"][0][i]
            
            similarity = self._convert_distance_to_similarity(distance)
            image_score = similarity * (1 - α)  # Apply image weight
            
            # Add figure path for retrieval if available
            self._resolve_figure_path(metadata)
            
            hits.append({
                "content": content,
                "metadata": metadata,
                "score": image_score,
                "source": "image"
            })
        
        # Sort hits by score (highest first)
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply document correlation boost
        # This adds a small boost to hits from the same document as the top hit
        if len(hits) > 1:
            top_doc = hits[0].get("metadata", {}).get("document", "")
            if top_doc:
                for hit in hits[1:]:
                    if hit.get("metadata", {}).get("document") == top_doc:
                        hit["score"] *= 1.05  # Small boost for coherence
                
                # Re-sort after boosting
                hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply cross-encoder reranking if enabled
        initial_hits_count = len(hits)
        if recipe.cross_encoder_rerank and self.cross_encoder:
            # Keep top-40 results for reranking to reduce computation
            rerank_hits = self._cross_encoder_rerank(query, hits[:40])
            
            # Combine reranked hits with any remaining hits
            hits = rerank_hits + hits[40:]
        
        # Update metrics
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        metrics["retrieval_tokens"] = recipe.n_text_results + recipe.n_image_results
        metrics["used_cross_encoder"] = recipe.cross_encoder_rerank and self.cross_encoder is not None
        
        return hits, metrics 