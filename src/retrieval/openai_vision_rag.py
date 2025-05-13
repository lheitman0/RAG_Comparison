"""
OpenAI Vision RAG approach with vision-based reranking
OpenAI's embeddings for retrieval and GPT-4o Vision for reranking
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
    def __init__(
        self, 
        persist_directory: str = "vector_store/openai",
        text_model: str = "openai/text-embedding-3-small",
        vision_model: str = "gpt-4o",
        device: Optional[str] = None
    ):
        self.persist_directory = persist_directory
        self.text_model = text_model
        self.vision_model = vision_model
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        self.text_collection = self.chroma_client.get_collection("text_openai")
        self.images_collection = self.chroma_client.get_collection("figures_clip")
        
        print(f"Initialized OpenAIVisionRetriever with collections: text_openai, figures_clip")
    
    def _convert_distance_to_similarity(self, distance: float) -> float:
        return 1.0 - distance
    
    def _encode_image(self, image_path: str) -> str:
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
            
            # Fallback: Create a random projection of the text embedding, as of now the clip model doesn't fail to load so this is not needed
            text_embedding = embed_text(self.text_model, query)
            np.random.seed(42)  
            projection_matrix = np.random.randn(text_embedding.shape[0], 512)
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
            
            image_compatible_embedding = np.dot(text_embedding, projection_matrix)
            image_compatible_embedding = image_compatible_embedding / np.linalg.norm(image_compatible_embedding)
            
            return image_compatible_embedding
    
    def _vision_reranking(
        self, 
        query: str, 
        hits: List[Dict[str, Any]], 
        top_k: int = 8
    ) -> List[Dict[str, Any]]:

        if not hits:
            return []
        
        image_hits = [
            hit for hit in hits if hit["source"] == "image" 
            and (hit.get("metadata", {}).get("figure_path") or 
                 self._resolve_figure_path(hit.get("metadata", {})))
        ]
        
        text_hits = [hit for hit in hits if hit["source"] == "text"][:30]
        
        image_hits = image_hits[:3]
        
        if not image_hits:
            return hits[:top_k]
        
        context_text = ""
        for i, hit in enumerate(text_hits[:15]):  # Limit text context
            context_text += f"[Document {i+1}]: {hit['content']}\n\n"
        # COME BACK TO: I think this could be optimized, even passing the images in with the text context and the query to get an answer could be interesting
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

        messages = [
            {"role": "system", "content": "You are a specialized retrieval system that analyzes and ranks documents and images by relevance."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
        
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
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if "rankings" in result:
                adjustments = {}
                for item in result["rankings"]:
                    if item["type"] == "text":
                        idx = item.get("index", 0)
                        if 0 <= idx < len(text_hits):
                            original_hit = text_hits[idx]
                            content_hash = hash(original_hit.get("content", ""))
                            adjustments[content_hash] = item.get("relevance_score", 5.0) / 10.0
                    
                    elif item["type"] == "image":
                        idx = item.get("index", 0)
                        if 0 <= idx < len(image_hits):
                            original_hit = image_hits[idx]
                            content_hash = hash(original_hit.get("content", ""))
                            adjustments[content_hash] = item.get("relevance_score", 5.0) / 10.0
                
                for hit in hits:
                    content_hash = hash(hit.get("content", ""))
                    if content_hash in adjustments:
                        hit["score"] = hit["score"] * adjustments[content_hash] * 2.0
                
                hits.sort(key=lambda x: x["score"], reverse=True)
                
            return hits[:top_k]
                
        except Exception as e:
            print(f"Error in vision reranking: {e}")
            return hits[:top_k]
    
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
    
    def retrieve(
        self,
        query: str,
        user_image: Optional[str] = None,
        recipe: Optional[RetrievalRecipe] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if recipe is None:
            recipe = RetrievalRecipe.openai_vision()
        
        metrics = {
            "text_to_image_ratio": recipe.text_to_image_ratio,
            "n_text_results": recipe.n_text_results,
            "n_image_results": recipe.n_image_results if user_image else 0,
            "vision_model": self.vision_model
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
        
        clip_text_embedding = embed_image(user_image) if user_image else None
        
        if clip_text_embedding is not None:
            image_results = self.images_collection.query(
                query_embeddings=[clip_text_embedding.tolist()],
                n_results=recipe.n_image_results,
                include=["metadatas", "documents", "distances"]
            )
        else:
            clip_compatible_embedding = embed_text_clip(query)
            
            image_results = self.images_collection.query(
                query_embeddings=[clip_compatible_embedding.tolist()],
                n_results=recipe.n_image_results // 2,
                include=["metadatas", "documents", "distances"]
            )
        
        for i in range(len(image_results["ids"][0])):
            document_id = image_results["ids"][0][i]
            content = image_results["documents"][0][i]
            metadata = image_results["metadatas"][0][i]
            distance = image_results["distances"][0][i]
            
            similarity = self._convert_distance_to_similarity(distance)
            image_score = similarity * (1 - ALPHA)  
            self._resolve_figure_path(metadata)
            
            hits.append({
                "content": content,
                "metadata": metadata,
                "score": image_score,
                "source": "image"
            })
        
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        initial_hits_count = len(hits)
        reranked_hits = self._vision_reranking(query, hits)
        
        metrics["num_initial_hits"] = initial_hits_count
        metrics["num_hits_after_reranking"] = len(reranked_hits)
        metrics["text_hits"] = sum(1 for hit in reranked_hits if hit["source"] == "text")
        metrics["image_hits"] = sum(1 for hit in reranked_hits if hit["source"] == "image")
        metrics["used_vision_reranking"] = True
        metrics["retrieval_tokens"] = recipe.n_text_results + recipe.n_image_results
        
        return reranked_hits, metrics 