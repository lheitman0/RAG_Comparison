"""
OpenAI Vision RAG implementation with vision-based reranking.
This approach uses OpenAI's embeddings for retrieval and GPT-4o Vision for reranking.
"""

import os
import base64
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import json
import chromadb
from openai import OpenAI
from pathlib import Path
from src.utils.embeddings import embed_text, embed_image, embed_text_clip
from src.retrieval.base import RetrievalRecipe


class OpenAIVisionRetriever:
    """Retrieval implementation using OpenAI embeddings with Vision reranking."""
    
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        vision_model: str = "gpt-4o",
        device: Optional[str] = None
    ):
        """
        Initialize the OpenAI Vision Retriever.
        
        Args:
            persist_directory: Path to the vector store directory
            text_model: Model identifier for text embeddings
            vision_model: Model to use for vision-based reranking
            device: Device to run models on ('cuda' or 'cpu'), auto-detected if None
        """
        self.persist_directory = persist_directory
        self.text_model = text_model
        self.vision_model = vision_model
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get collections
        self.text_collection = self.chroma_client.get_collection("text_openai")
        self.images_collection = self.chroma_client.get_collection("figures_clip")
        
        print(f"Initialized OpenAIVisionRetriever with collections: text_openai, figures_clip")
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score."""
        return 1.0 - distance
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API use."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _generate_text_embedding_for_images(self, query: str) -> np.ndarray:
        """
        Generate a text embedding suitable for querying the images collection.
        This creates a 512-dimension embedding compatible with CLIP.
        
        Args:
            query: Text query
            
        Returns:
            512-dimensional embedding for image search
        """
        # For simplicity, we'll create a fake image embedding from text
        # In a production system, we would use a text-to-image model or CLIP's text encoder
        
        # Option 1: Use CLIP text encoder directly (best)
        try:
            from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
            
            model_name = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(model_name, cache_dir="./models")
            tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir="./models")
            
            # Move model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            # Encode text with CLIP
            inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy
            return text_features.squeeze(0).cpu().numpy()
        
        except Exception as e:
            print(f"Error using CLIP text encoder: {e}. Falling back to random projection.")
            
            # Fallback: Create a random projection of the text embedding
            text_embedding = embed_text(self.text_model, query)
            
            # Random projection matrix (would be better to use a learned projection)
            # In production, you'd want to use a proper dimensionality reduction
            np.random.seed(42)  # For reproducibility
            projection_matrix = np.random.randn(text_embedding.shape[0], 512)
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
            
            # Project to 512 dimensions
            image_compatible_embedding = np.dot(text_embedding, projection_matrix)
            
            # Normalize
            image_compatible_embedding = image_compatible_embedding / np.linalg.norm(image_compatible_embedding)
            
            return image_compatible_embedding
    
    def _vision_reranking(
        self, 
        query: str, 
        hits: List[Dict[str, Any]], 
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Use GPT-4o Vision to rerank results based on relevance to the query.
        
        Args:
            query: User query
            hits: List of initial hits with content and metadata
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked list of hits
        """
        # Exit early if no hits
        if not hits:
            return []
        
        # Select image hits that have figure paths
        image_hits = [
            hit for hit in hits if hit["source"] == "image" 
            and (hit.get("metadata", {}).get("figure_path") or 
                 self._resolve_figure_path(hit.get("metadata", {})))
        ]
        
        # Select top text hits
        text_hits = [hit for hit in hits if hit["source"] == "text"][:30]
        
        # Limit to 3 images for API economy
        image_hits = image_hits[:3]
        
        # If we don't have any images, just return the text hits
        if not image_hits:
            return hits[:top_k]
        
        # Prepare content for GPT-4o Vision
        context_text = ""
        for i, hit in enumerate(text_hits[:15]):  # Limit text context
            context_text += f"[Document {i+1}]: {hit['content']}\n\n"
        
        # Prepare reranking prompt
        prompt = f"""Analyze these documents and images to find the most relevant content for this query:
        
QUERY: "{query}"

CONTEXT DOCUMENTS:
{context_text}

Your task is to analyze both the text passages and the images to determine which are most relevant to the query.
Output a JSON with rankings and explanations. 

For each image, assess:
1. How relevant the image is to the query
2. What specific visual elements support answering the query
3. How well it complements the text context

The JSON format should be:
{{
  "rankings": [
    {{
      "type": "text", 
      "index": 0, 
      "relevance_score": 9.5, 
      "explanation": "Directly addresses..."
    }},
    ...
  ]
}}

Your ranking should determine which items are MOST informative for answering the specific query.
"""

        # Prepare messages for GPT-4o
        messages = [
            {"role": "system", "content": "You are a specialized retrieval system that analyzes and ranks documents and images by relevance."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Add images to the message
        for i, hit in enumerate(image_hits):
            img_path = hit.get("metadata", {}).get("figure_path")
            if not img_path:
                img_path = self._resolve_figure_path(hit.get("metadata", {}))
                
            if img_path and os.path.exists(img_path):
                try:
                    base64_image = self._encode_image(img_path)
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                except Exception as e:
                    print(f"Error encoding image {img_path}: {e}")
        
        try:
            # Call GPT-4o Vision
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Rerank hits based on the vision model's assessment
            if "rankings" in result:
                # Create a score adjustment mapping
                adjustments = {}
                for item in result["rankings"]:
                    if item["type"] == "text":
                        idx = item.get("index", 0)
                        if 0 <= idx < len(text_hits):
                            # Get the original hit
                            original_hit = text_hits[idx]
                            content_hash = hash(original_hit.get("content", ""))
                            adjustments[content_hash] = item.get("relevance_score", 5.0) / 10.0
                    
                    elif item["type"] == "image":
                        idx = item.get("index", 0)
                        if 0 <= idx < len(image_hits):
                            # Get the original hit
                            original_hit = image_hits[idx]
                            content_hash = hash(original_hit.get("content", ""))
                            adjustments[content_hash] = item.get("relevance_score", 5.0) / 10.0
                
                # Apply adjustments to all hits
                for hit in hits:
                    content_hash = hash(hit.get("content", ""))
                    if content_hash in adjustments:
                        # Scale the original score
                        hit["score"] = hit["score"] * adjustments[content_hash] * 2.0
                
                # Re-sort hits by adjusted score
                hits.sort(key=lambda x: x["score"], reverse=True)
                
            return hits[:top_k]
                
        except Exception as e:
            print(f"Error in vision reranking: {e}")
            # Fall back to original ranking
            return hits[:top_k]
    
    def _resolve_figure_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Resolve figure path from metadata if not already present."""
        # Already has valid figure_path
        if "figure_path" in metadata and os.path.exists(metadata["figure_path"]):
            return metadata["figure_path"]

        # Direct absolute image_path
        if "image_path" in metadata and os.path.exists(metadata["image_path"]):
            metadata["figure_path"] = metadata["image_path"]
            return metadata["figure_path"]

        # Derive from filename
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
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve relevant documents using OpenAI text and CLIP image embeddings,
        then rerank using GPT-4o Vision.
        
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
            recipe = RetrievalRecipe.openai_vision()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "vision_model": self.vision_model
        }
        
        α = recipe.text_to_image_ratio  # Text weight
        
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
        
        # Always include image search for this retriever, even without user image
        # Generate query embedding for images (using text)
        clip_text_embedding = embed_image(user_image) if user_image else None
        
        # Query images collection
        if clip_text_embedding is not None:
            # Use user image embedding
            image_results = self.images_collection.query(
                query_embeddings=[clip_text_embedding.tolist()],
                n_results=recipe.n_image_results,
                include=["metadatas", "documents", "distances"]
            )
        else:
            # Use text-derived image embedding compatible with CLIP dimensions
            clip_compatible_embedding = embed_text_clip(query)
            
            # Now query the images collection with the proper dimensions
            image_results = self.images_collection.query(
                query_embeddings=[clip_compatible_embedding.tolist()],
                n_results=recipe.n_image_results // 2,
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
        
        # Apply vision reranking
        initial_hits_count = len(hits)
        reranked_hits = self._vision_reranking(query, hits)
        
        # Update metrics
        metrics["num_initial_hits"] = initial_hits_count
        metrics["num_hits_after_reranking"] = len(reranked_hits)
        metrics["text_hits"] = sum(1 for hit in reranked_hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in reranked_hits if hit["source"] == "image")
        metrics["used_vision_reranking"] = True
        metrics["retrieval_tokens"] = recipe.n_text_results + recipe.n_image_results
        
        return reranked_hits, metrics 