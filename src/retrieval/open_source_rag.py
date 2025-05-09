"""
Open Source RAG implementation using E5 text embeddings and CLIP for images.
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
    """Retrieval implementation using open source models (E5 + CLIP)."""
    
    def __init__(
        self, 
        persist_directory: str = "vector_store/opensource",
        text_model: str = "huggingface/intfloat/multilingual-e5-base",
        device: Optional[str] = None
    ):
        """
        Initialize the Open Source Retriever.
        
        Args:
            persist_directory: Path to the vector store directory
            text_model: Model identifier for text embeddings
            device: Device to run models on ('cuda' or 'cpu'), auto-detected if None
        """
        self.persist_directory = persist_directory
        self.text_model = text_model
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get collections
        self.text_collection = self.client.get_collection("text_e5")
        self.images_collection = self.client.get_collection("figures_clip")
        
        # Optional CLIP text collection
        try:
            self.text_clip_collection = self.client.get_collection("text_clip_extra")
            self.has_text_clip = True
        except:
            self.has_text_clip = False
            
        print(f"Initialized OpenSourceRetriever with collections: text_e5, figures_clip"
              f"{', text_clip_extra' if self.has_text_clip else ''}")
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score."""
        return 1.0 - distance
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve relevant documents using E5 text and CLIP image embeddings.
        
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
            recipe = RetrievalRecipe.open_source()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "use_text_clip": self.has_text_clip
        }
        
        α = 0.5  # Equal weight between text and captions per optimal strategy
        
        device_for_clip = self.device
        
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
        
        # Decide whether we will run image search
        run_image_search = True  # Always run per optimal strategy

        # If user provided an image we embed it, otherwise embed query with CLIP text encoder
        image_embedding = None
        if user_image and os.path.exists(user_image):
            image_embedding = embed_image(user_image)
        elif run_image_search:
            image_embedding = embed_text_clip(query)

        if image_embedding is not None:
            # Query images collection
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
                
                # Ensure figure_path is present if possible
                self._resolve_figure_path(metadata)
                
                hits.append({
                    "content": content,
                    "metadata": metadata,
                    "score": similarity * (1 - α),  # Apply image weight
                    "source": "image"
                })
                
            # If we have CLIP text encodings, use them as well
            if self.has_text_clip:
                # We can reuse the same CLIP embedding used for images
                clip_text_results = self.text_clip_collection.query(
                    query_embeddings=[image_embedding.tolist()],
                    n_results=recipe.n_image_results,
                    include=["metadatas", "documents", "distances"]
                )
                
                # Process CLIP text results
                for i in range(len(clip_text_results["ids"][0])):
                    document_id = clip_text_results["ids"][0][i]
                    content = clip_text_results["documents"][0][i]
                    metadata = clip_text_results["metadatas"][0][i]
                    distance = clip_text_results["distances"][0][i]
                    
                    similarity = self._convert_distance_to_similarity(distance)
                    
                    # No figure path for caption hits; attempt to derive if possible
                    self._resolve_figure_path(metadata)
                    
                    hits.append({
                        "content": content,
                        "metadata": metadata,
                        "score": similarity * (1 - α),  # Equal weight to image
                        "source": "clip_text"
                    })
        
        # Sort hits by score (highest first)
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Update metrics
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        metrics["clip_text_hits"] = sum(1 for hit in hits if hit["source"] == "clip_text")

        # Approximate retrieval token usage (1 token per result as cheap proxy)
        metrics["retrieval_tokens"] = (
            recipe.n_text_results + (recipe.n_image_results if image_embedding is not None else 0)
        )
        
        return hits, metrics 

    # ------------------------------------------------------------------
    # Helper: ensure that a usable 'figure_path' key exists in metadata
    # ------------------------------------------------------------------
    def _resolve_figure_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Try to populate metadata["figure_path"] from available fields.

        Accepts either an absolute `image_path`, a pre-existing `figure_path`,
        or just a `figure_filename` (e.g. 'wifi_manual_figure_34.png') that can
        be joined with the canonical data directory structure.
        """
        # Already present & valid
        if "figure_path" in metadata and os.path.exists(metadata["figure_path"]):
            return metadata["figure_path"]

        # Direct absolute image_path
        if "image_path" in metadata and os.path.exists(metadata["image_path"]):
            metadata["figure_path"] = metadata["image_path"]
            return metadata["figure_path"]

        # Derive from filename
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