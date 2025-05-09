"""
OpenAI + CLIP RAG implementation using OpenAI's text embeddings and CLIP for images.
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
    """Retrieval implementation using OpenAI embeddings and CLIP."""
    
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        device: Optional[str] = None
    ):
        """
        Initialize the OpenAI + CLIP Retriever.
        
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
        self.text_collection = self.client.get_collection("text_openai")
        self.images_collection = self.client.get_collection("figures_clip")
        
        print(f"Initialized OpenAIClipRetriever with collections: text_openai, figures_clip")
    
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
        Retrieve relevant documents using OpenAI text and CLIP image embeddings.
        
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
            recipe = RetrievalRecipe.openai_clip()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0
        }
        
        α = recipe.text_to_image_ratio  # Text weight (0.8 by default)
        
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
        
        # If user provided an image, include image search
        if user_image and os.path.exists(user_image):
            # Generate image embedding
            image_embedding = embed_image(user_image)
            
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
                image_score = similarity * (1 - α)  # Apply image weight
                
                # Add figure path for retrieval if available
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
        
        # Sort hits by score (highest first)
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Add content boost for document coherence
        # This prioritizes results from the same document
        if len(hits) > 0:
            # Find most common document
            document_counts = {}
            for hit in hits[:10]:  # Only look at top hits
                doc = hit.get("metadata", {}).get("document", "")
                if doc:
                    document_counts[doc] = document_counts.get(doc, 0) + 1
            
            # Boost hits from most common document
            if document_counts:
                most_common_doc = max(document_counts.items(), key=lambda x: x[1])[0]
                
                for hit in hits:
                    if hit.get("metadata", {}).get("document") == most_common_doc:
                        hit["score"] *= 1.05  # Small boost
            
            # Re-sort after boosting
            hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Update metrics
        metrics["num_hits"] = len(hits)
        metrics["text_hits"] = sum(1 for hit in hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in hits if hit["source"] == "image")
        metrics["retrieval_tokens"] = sum(len(h["content"].split()) for h in hits) // 4
        
        return hits, metrics 