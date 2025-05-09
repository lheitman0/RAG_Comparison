"""
Base retrieval interface for RAG systems.
Defines a common API for all retrieval approaches.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import os
from pathlib import Path
from src.utils.chunk_merge import ContextHit, merge_adjacent_chunks
from src.utils.language_id import detect_language
from src.utils.query_distill import distill_query


class RetrievalApproach(Enum):
    """Enumeration of supported retrieval approaches."""
    
    OPEN_SOURCE = "open_source"
    OPENAI_CLIP = "openai_clip"
    OPENAI_VISION = "openai_vision"
    HYBRID = "hybrid"


class RetrievalRecipe:
    """Configuration for retrieval parameters."""
    
    def __init__(
        self,
        approach: Union[RetrievalApproach, str],
        text_to_image_ratio: float = 0.8,
        n_text_results: int = 40,
        n_image_results: int = 20,
        use_distillation: bool = True,
        cross_encoder_rerank: bool = False,
        language: Optional[str] = None
    ):
        """
        Initialize a retrieval recipe.
        
        Args:
            approach: Which retrieval approach to use
            text_to_image_ratio: Weight for text vs image results (higher = more text)
            n_text_results: Number of initial text results to retrieve
            n_image_results: Number of initial image results to retrieve
            use_distillation: Whether to apply query distillation
            cross_encoder_rerank: Whether to apply cross-encoder reranking
            language: ISO language code for queries (auto-detect if None)
        """
        # Convert string approach to enum if needed
        if isinstance(approach, str):
            approach = RetrievalApproach(approach)
            
        self.approach = approach
        self.text_to_image_ratio = text_to_image_ratio
        self.n_text_results = n_text_results
        self.n_image_results = n_image_results
        self.use_distillation = use_distillation
        self.cross_encoder_rerank = cross_encoder_rerank
        self.language = language
    
    @classmethod
    def open_source(cls) -> 'RetrievalRecipe':
        """Create a recipe for open source approach."""
        return cls(
            approach=RetrievalApproach.OPEN_SOURCE,
            text_to_image_ratio=0.5,
            n_text_results=40,
            n_image_results=20,
        )
    
    @classmethod
    def openai_clip(cls) -> 'RetrievalRecipe':
        """Create a recipe for OpenAI + CLIP approach."""
        return cls(
            approach=RetrievalApproach.OPENAI_CLIP,
            text_to_image_ratio=0.8,
            n_text_results=40,
            n_image_results=20,
        )
    
    @classmethod
    def openai_vision(cls) -> 'RetrievalRecipe':
        """Create a recipe for OpenAI + Vision rerank approach."""
        return cls(
            approach=RetrievalApproach.OPENAI_VISION,
            text_to_image_ratio=0.7,
            n_text_results=40,
            n_image_results=15,
        )
    
    @classmethod
    def hybrid(cls) -> 'RetrievalRecipe':
        """Create a recipe for hybrid late fusion approach."""
        return cls(
            approach=RetrievalApproach.HYBRID,
            text_to_image_ratio=0.7,
            n_text_results=40,
            n_image_results=20,
            cross_encoder_rerank=True,
        )


def retrieve(
    query: str,
    user_image: Optional[str] = None,
    recipe: Optional[RetrievalRecipe] = None,
    k: int = 8
) -> Tuple[List[ContextHit], Dict[str, Any]]:
    """
    Retrieve relevant documents using the specified approach.
    
    This is the main entry point for retrieval across all approaches.
    
    Args:
        query: The user query text
        user_image: Optional path to a user-provided image to include in the query
        recipe: Retrieval recipe with parameters
        k: Number of final results to return
    
    Returns:
        Tuple containing:
        - List of relevant context hits
        - Dictionary of retrieval metrics
    """
    # Default recipe if none provided
    if recipe is None:
        recipe = RetrievalRecipe.hybrid()
    
    # Detect language if not specified
    if recipe.language is None:
        recipe.language = detect_language(query)
    
    # Apply query distillation if enabled
    if recipe.use_distillation:
        distilled_query = distill_query(query, recipe.language)
    else:
        distilled_query = query
    
    # Import the appropriate retriever based on the approach
    start_time = __import__('time').time()
    
    metrics = {
        "approach": recipe.approach.value,
        "original_query": query,
        "distilled_query": distilled_query,
        "language": recipe.language,
        "k": k,
    }
    
    # Import and use the appropriate retrieval approach
    if recipe.approach == RetrievalApproach.OPEN_SOURCE:
        from src.retrieval.open_source_rag import OpenSourceRetriever
        retriever = OpenSourceRetriever()
    elif recipe.approach == RetrievalApproach.OPENAI_CLIP:
        from src.retrieval.openai_clip_rag import OpenAIClipRetriever
        retriever = OpenAIClipRetriever()
    elif recipe.approach == RetrievalApproach.OPENAI_VISION:
        from src.retrieval.openai_vision_rag import OpenAIVisionRetriever
        retriever = OpenAIVisionRetriever()
    elif recipe.approach == RetrievalApproach.HYBRID:
        from src.retrieval.hybrid_rag import HybridRetriever
        retriever = HybridRetriever()
    else:
        raise ValueError(f"Unsupported retrieval approach: {recipe.approach}")
    
    # Perform retrieval
    hits, approach_metrics = retriever.retrieve(
        query=distilled_query,
        user_image=user_image,
        recipe=recipe
    )
    
    # Merge adjacent chunks
    merged_hits = merge_adjacent_chunks(hits, max_tokens=400)
    
    # Truncate to k results
    final_hits = merged_hits[:k]
    
    # Update metrics
    metrics.update(approach_metrics)
    metrics["retrieval_time"] = __import__('time').time() - start_time
    metrics["num_raw_hits"] = len(hits)
    metrics["num_merged_hits"] = len(merged_hits)
    metrics["num_returned_hits"] = len(final_hits)
    
    return final_hits, metrics 