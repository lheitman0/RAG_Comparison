"""
Base retrieval interface for all RAG approaches
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import os
from pathlib import Path
from src.utils.chunk_merge import ContextHit, merge_adjacent_chunks
from src.utils.language_id import detect_language
from src.utils.query_distill import distill_query


class RetrievalApproach(Enum):    
    OPEN_SOURCE = "open_source"
    OPENAI_CLIP = "openai_clip"
    OPENAI_VISION = "openai_vision"
    HYBRID = "hybrid"


class RetrievalRecipe:    
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
        return cls(
            approach=RetrievalApproach.OPEN_SOURCE,
            text_to_image_ratio=0.5,
            n_text_results=40,
            n_image_results=20,
        )
    
    @classmethod
    def openai_clip(cls) -> 'RetrievalRecipe':
        return cls(
            approach=RetrievalApproach.OPENAI_CLIP,
            text_to_image_ratio=0.8,
            n_text_results=40,
            n_image_results=20,
        )
    
    @classmethod
    def openai_vision(cls) -> 'RetrievalRecipe':
        return cls(
            approach=RetrievalApproach.OPENAI_VISION,
            text_to_image_ratio=0.7,
            n_text_results=40,
            n_image_results=15,
        )
    
    @classmethod
    def hybrid(cls) -> 'RetrievalRecipe':
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
    if recipe is None:
        recipe = RetrievalRecipe.hybrid()
    
    if recipe.language is None:
        recipe.language = detect_language(query)
    
    if recipe.use_distillation:
        distilled_query = distill_query(query, recipe.language)
    else:
        distilled_query = query
    
    start_time = __import__('time').time()
    
    metrics = {
        "approach": recipe.approach.value,
        "original_query": query,
        "distilled_query": distilled_query,
        "language": recipe.language,
        "k": k,
    }
    
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
    
    hits, approach_metrics = retriever.retrieve(
        query=distilled_query,
        user_image=user_image,
        recipe=recipe
    )
    
    merged_hits = merge_adjacent_chunks(hits, max_tokens=400)
    final_hits = merged_hits[:k]
    
    metrics.update(approach_metrics)
    metrics["retrieval_time"] = __import__('time').time() - start_time
    metrics["num_raw_hits"] = len(hits)
    metrics["num_merged_hits"] = len(merged_hits)
    metrics["num_returned_hits"] = len(final_hits)
    
    return final_hits, metrics 