"""
Error tracking system for RAG evaluation.
Tracks and analyzes different types of failures in the RAG pipeline.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import json
from datetime import datetime
import os

class ErrorType(Enum):
    """Types of errors that can occur in the RAG pipeline."""
    RETRIEVAL_FAILURE = "retrieval_failure"  # Relevant content not found
    GENERATION_FAILURE = "generation_failure"  # Incorrect or incomplete answer
    HALLUCINATION = "hallucination"  # Made-up information
    CONTEXT_ERROR = "context_error"  # Wrong context used
    STRATEGY_ERROR = "strategy_error"  # Wrong strategy chosen
    TIMEOUT = "timeout"  # Operation took too long

class ErrorTracker:
    """Tracks and analyzes errors in the RAG pipeline."""
    
    def __init__(self, output_dir: str = "evaluation_results/error_analysis"):
        """
        Initialize the error tracker.
        
        Args:
            output_dir: Directory to save error analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {error_type.value: 0 for error_type in ErrorType}
    
    def track_error(self,
                   error_type: ErrorType,
                   query: str,
                   approach: str,
                   details: Dict[str, Any],
                   ground_truth: Optional[str] = None,
                   generated_answer: Optional[str] = None) -> None:
        """
        Track a new error.
        
        Args:
            error_type: Type of error
            query: The query that caused the error
            approach: The RAG approach that failed
            details: Additional details about the error
            ground_truth: Optional ground truth for comparison
            generated_answer: Optional generated answer for comparison
        """
        error = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            "query": query,
            "approach": approach,
            "details": details,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer
        }
        
        self.errors.append(error)
        self.error_counts[error_type.value] += 1
    
    def analyze_retrieval_failure(self,
                                query: str,
                                retrieved_sections: List[str],
                                relevant_sections: List[str],
                                approach: str) -> None:
        """
        Analyze a retrieval failure.
        
        Args:
            query: The query
            retrieved_sections: Sections that were retrieved
            relevant_sections: Sections that should have been retrieved
            approach: The RAG approach
        """
        details = {
            "retrieved_sections": retrieved_sections,
            "relevant_sections": relevant_sections,
            "retrieval_score": self._calculate_retrieval_score(retrieved_sections, relevant_sections)
        }
        
        self.track_error(
            ErrorType.RETRIEVAL_FAILURE,
            query,
            approach,
            details
        )
    
    def analyze_generation_failure(self,
                                 query: str,
                                 generated_answer: str,
                                 ground_truth: str,
                                 approach: str) -> None:
        """
        Analyze a generation failure.
        
        Args:
            query: The query
            generated_answer: The generated answer
            ground_truth: The correct answer
            approach: The RAG approach
        """
        details = {
            "completeness_score": self._calculate_completeness_score(generated_answer, ground_truth),
            "accuracy_score": self._calculate_accuracy_score(generated_answer, ground_truth),
            "error_details": self._identify_generation_errors(generated_answer, ground_truth)
        }
        
        self.track_error(
            ErrorType.GENERATION_FAILURE,
            query,
            approach,
            details,
            ground_truth,
            generated_answer
        )
    
    def analyze_hallucination(self,
                            query: str,
                            generated_answer: str,
                            context: List[str],
                            approach: str) -> None:
        """
        Analyze a hallucination.
        
        Args:
            query: The query
            generated_answer: The generated answer
            context: The context used for generation
            approach: The RAG approach
        """
        details = {
            "hallucination_score": self._calculate_hallucination_score(generated_answer, context),
            "hallucinated_content": self._identify_hallucinated_content(generated_answer, context),
            "context_used": context
        }
        
        self.track_error(
            ErrorType.HALLUCINATION,
            query,
            approach,
            details,
            generated_answer=generated_answer
        )
    
    def save_analysis(self) -> None:
        """Save the error analysis to files."""
        # Save all errors
        with open(os.path.join(self.output_dir, "all_errors.json"), "w") as f:
            json.dump(self.errors, f, indent=2)
        
        # Save error counts
        with open(os.path.join(self.output_dir, "error_counts.json"), "w") as f:
            json.dump(self.error_counts, f, indent=2)
        
        # Generate summary report
        summary = {
            "total_errors": len(self.errors),
            "error_distribution": self.error_counts,
            "error_trends": self._analyze_error_trends(),
            "recommendations": self._generate_recommendations()
        }
        
        with open(os.path.join(self.output_dir, "error_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_retrieval_score(self,
                                 retrieved: List[str],
                                 relevant: List[str]) -> float:
        """Calculate how well the retrieval performed."""
        if not relevant:
            return 0.0
        
        # Simple overlap-based score
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        intersection = retrieved_set.intersection(relevant_set)
        
        return len(intersection) / len(relevant_set)
    
    def _calculate_completeness_score(self,
                                    generated: str,
                                    ground_truth: str) -> float:
        """Calculate how complete the generated answer is."""
        # Simple word overlap-based score
        generated_words = set(generated.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        intersection = generated_words.intersection(ground_truth_words)
        
        return len(intersection) / len(ground_truth_words) if ground_truth_words else 0.0
    
    def _calculate_accuracy_score(self,
                                generated: str,
                                ground_truth: str) -> float:
        """Calculate how accurate the generated answer is."""
        # Simple word-based accuracy score
        generated_words = generated.lower().split()
        ground_truth_words = ground_truth.lower().split()
        
        correct = sum(1 for w1, w2 in zip(generated_words, ground_truth_words) if w1 == w2)
        return correct / len(ground_truth_words) if ground_truth_words else 0.0
    
    def _calculate_hallucination_score(self,
                                     generated: str,
                                     context: List[str]) -> float:
        """Calculate how much of the generated answer is hallucinated."""
        # Simple word-based hallucination score
        context_words = set()
        for c in context:
            context_words.update(c.lower().split())
        
        generated_words = set(generated.lower().split())
        intersection = generated_words.intersection(context_words)
        
        return 1.0 - (len(intersection) / len(generated_words)) if generated_words else 0.0
    
    def _identify_generation_errors(self,
                                  generated: str,
                                  ground_truth: str) -> List[str]:
        """Identify specific errors in the generated answer."""
        errors = []
        
        # Check for missing information
        ground_truth_words = set(ground_truth.lower().split())
        generated_words = set(generated.lower().split())
        missing = ground_truth_words - generated_words
        if missing:
            errors.append(f"Missing information: {', '.join(missing)}")
        
        # Check for incorrect information
        # This is a simplified check - in practice, you'd want more sophisticated analysis
        if len(generated_words) > len(ground_truth_words) * 1.5:
            errors.append("Answer contains excessive information")
        
        return errors
    
    def _identify_hallucinated_content(self,
                                     generated: str,
                                     context: List[str]) -> List[str]:
        """Identify specific hallucinated content in the generated answer."""
        context_words = set()
        for c in context:
            context_words.update(c.lower().split())
        
        generated_words = generated.lower().split()
        hallucinated = [word for word in generated_words if word not in context_words]
        
        return hallucinated
    
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze trends in the errors."""
        # Group errors by approach
        errors_by_approach = {}
        for error in self.errors:
            approach = error["approach"]
            if approach not in errors_by_approach:
                errors_by_approach[approach] = []
            errors_by_approach[approach].append(error)
        
        # Calculate error rates by approach
        error_rates = {}
        for approach, errors in errors_by_approach.items():
            error_rates[approach] = {
                "total_errors": len(errors),
                "error_types": {},
                "common_queries": self._find_common_queries(errors)
            }
            for error_type in ErrorType:
                count = sum(1 for e in errors if e["error_type"] == error_type.value)
                error_rates[approach]["error_types"][error_type.value] = count
        
        return error_rates
    
    def _find_common_queries(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find queries that commonly lead to errors."""
        query_counts = {}
        for error in errors:
            query = error["query"]
            if query not in query_counts:
                query_counts[query] = 0
            query_counts[query] += 1
        
        # Sort by frequency
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"query": query, "count": count} for query, count in sorted_queries[:5]]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # Analyze retrieval failures
        retrieval_failures = [e for e in self.errors if e["error_type"] == ErrorType.RETRIEVAL_FAILURE.value]
        if retrieval_failures:
            avg_score = sum(e["details"]["retrieval_score"] for e in retrieval_failures) / len(retrieval_failures)
            if avg_score < 0.5:
                recommendations.append("Improve retrieval strategy - current retrieval is missing relevant content")
        
        # Analyze generation failures
        generation_failures = [e for e in self.errors if e["error_type"] == ErrorType.GENERATION_FAILURE.value]
        if generation_failures:
            avg_completeness = sum(e["details"]["completeness_score"] for e in generation_failures) / len(generation_failures)
            if avg_completeness < 0.7:
                recommendations.append("Improve answer generation - answers are often incomplete")
        
        # Analyze hallucinations
        hallucinations = [e for e in self.errors if e["error_type"] == ErrorType.HALLUCINATION.value]
        if hallucinations:
            avg_hallucination = sum(e["details"]["hallucination_score"] for e in hallucinations) / len(hallucinations)
            if avg_hallucination > 0.3:
                recommendations.append("Reduce hallucinations - answers contain too much made-up information")
        
        return recommendations 