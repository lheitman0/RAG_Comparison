"""
Model caching module to load models once and keep them in memory.
"""
from functools import lru_cache
import torch

@lru_cache(maxsize=None)
def get_llm():
    """
    Get the TinyLlama model (loaded only once).
    
    Returns:
        HuggingFacePipeline instance
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        from langchain_community.llms import HuggingFacePipeline
    
    print("Loading TinyLlama model (first time only)...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Determine device - use MPS if on Mac with GPU support, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=True,
        device=device
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print("TinyLlama model loaded successfully!")
    return llm

@lru_cache(maxsize=None)
def get_e5_embeddings():
    """
    Get the E5 embedding model (loaded only once).
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    
    print("Loading E5 multilingual embedding model (first time only)...")
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./models"
    )
    
    print("E5 embedding model loaded successfully!")
    return embeddings

@lru_cache(maxsize=None)
def get_clip():
    """
    Get the CLIP model and processor (loaded only once).
    
    Returns:
        Tuple of (model, processor)
    """
    from transformers import CLIPProcessor, CLIPModel
    
    print("Loading CLIP model (first time only)...")
    
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Move to appropriate device if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    
    print("CLIP model loaded successfully!")
    return model, processor

@lru_cache(maxsize=None)
def get_vector_store(manual_type=None, persist_directory=None):
    """
    Get the vector store for a specific manual (loaded only once per type).
    
    Args:
        manual_type: Either 'VM_manual', 'wifi_manual', or a store name for combined
        persist_directory: Optional override for the persist directory
    
    Returns:
        TechnicalManualVectorStore instance
    """
    from src.embeddings.vector_store import TechnicalManualVectorStore
    
    # Set default persist directory based on manual type
    if persist_directory is None:
        if manual_type:
            persist_directory = f"./vector_db_opensource/{manual_type}"
        else:
            persist_directory = "./vector_db_opensource/combined"
    
    # Use store name for logging
    store_name = manual_type if manual_type else "combined store"
    
    print(f"Loading vector store for {store_name} from {persist_directory} (first time only)...")
    vector_store = TechnicalManualVectorStore(persist_directory=persist_directory)
    print(f"Vector store for {store_name} loaded successfully!")
    
    return vector_store