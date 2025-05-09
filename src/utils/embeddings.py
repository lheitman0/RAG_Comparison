"""
Embedding utilities for text and image embeddings
"""

import os
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image
from openai import OpenAI

# TODO: Initialize models on first call instead of import time
# For CLIP, we should implement lazy loading to avoid unnecessarily 
# loading the model if it's not going to be used

def embed_text(
    model_id: str, 
    text: str,
    max_length: int = 512
) -> np.ndarray:
    """
    Generate text embeddings using the specified model.
    
    Args:
        model_id: Identifier for the embedding model to use.
                 Format: "{provider}/{model_name}" e.g. "openai/text-embedding-3-small"
                 or "huggingface/intfloat/multilingual-e5-large"
        text: Text to embed
        max_length: Maximum token length for the text
        
    Returns:
        Numpy array containing the embedding vector
    """
    provider, *model_parts = model_id.split('/')
    model_name = '/'.join(model_parts)
    
    if provider.lower() == 'openai':
        # Use OpenAI API for embeddings
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model=model_name,
            input=text,
            dimensions=1024  # Standard dimension for our vector store
        )
        return np.array(response.data[0].embedding)
    
    elif provider.lower() == 'huggingface':
        # TODO: Initialize model on first use and cache it
        from sentence_transformers import SentenceTransformer
        
        # Check if model is already loaded
        model = SentenceTransformer(model_name, cache_folder="./models")
        
        # Encode and return the embedding
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def embed_image(
    image_path: str,
    return_tensors: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate image embeddings using CLIP.
    
    Args:
        image_path: Path to the image file
        return_tensors: If True, return PyTorch tensors; otherwise return numpy arrays
        
    Returns:
        Embedding vector for the image as numpy array or torch tensor
    """
    # TODO: Initialize CLIP model on first use
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    
    # Load model and processor (should be cached after first load)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, cache_dir="./models")
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir="./models")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embeddings = outputs.detach()
    
    # Normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    if return_tensors:
        return embeddings.squeeze(0)
    else:
        return embeddings.squeeze(0).cpu().numpy()


def get_clip_model() -> Tuple[Any, Any]:
    """
    Load CLIP model and processor.
    
    Returns:
        Tuple containing the CLIP model and processor
    """
    # TODO: Proper implementation with caching
    from transformers import CLIPProcessor, CLIPModel
    
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, cache_dir="./models")
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir="./models")
    
    return model, processor 


# ----------------------------------------------
# CLIP text embedding helper for 512-D vectors
# ----------------------------------------------

_clip_text_model = None
_clip_text_tokenizer = None

def embed_text_clip(text: str) -> np.ndarray:
    """Return a 512-dim CLIP text embedding compatible with the CLIP image encoder.
    Falls back to random projection if transformers/CLIP is unavailable.
    """
    global _clip_text_model, _clip_text_tokenizer
    try:
        from transformers import CLIPTokenizer, CLIPModel
        import torch
        import numpy as np
        model_name = "openai/clip-vit-base-patch32"
        # Lazy load
        if _clip_text_model is None:
            _clip_text_model = CLIPModel.from_pretrained(model_name, cache_dir="./models")
            _clip_text_tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir="./models")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_text_model = _clip_text_model.to(device)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = _clip_text_tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = _clip_text_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"embed_text_clip fallback: {e}")
        # Fallback random projection of OpenAI embedding (deterministic seed)
        openai_vec = embed_text("openai/text-embedding-3-small", text)
        np.random.seed(42)
        proj = np.random.randn(openai_vec.shape[0], 512)
        proj /= np.linalg.norm(proj, axis=0)
        vec = openai_vec @ proj
        vec /= np.linalg.norm(vec)
        return vec 