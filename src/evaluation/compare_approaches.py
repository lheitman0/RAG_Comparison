"""
Script to compare different RAG approaches using the test suite.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from src.approaches.open_source_rag import OpenSourceRAG
from src.approaches.openai_clip_rag import OpenAIClipRAG
from src.approaches.openai_vision_rag import OpenAIVisionRAG
from src.approaches.hybrid_rag import HybridRAG
from src.evaluation.test_suite import RAGTestSuite

def generate_comparison_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comparison report from the results of different approaches.
    
    Args:
        results: Dictionary containing results from each approach
        
    Returns:
        Dictionary containing the comparison report
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'approaches': {},
        'summary': {
            'best_text_accuracy': {'approach': None, 'value': 0},
            'best_image_accuracy': {'approach': None, 'value': 0},
            'fastest_retrieval': {'approach': None, 'value': float('inf')},
            'fastest_generation': {'approach': None, 'value': float('inf')},
            'most_efficient': {'approach': None, 'value': float('inf')},  # tokens per correct answer
            'most_comprehensive': {'approach': None, 'value': 0}  # average answer quality
        }
    }
    
    for approach_name, approach_results in results.items():
        # Calculate metrics
        avg_text_accuracy = sum(r.text_accuracy for r in approach_results) / len(approach_results)
        avg_image_accuracy = sum(r.image_accuracy for r in approach_results) / len(approach_results)
        avg_retrieval_time = sum(r.retrieval_time for r in approach_results) / len(approach_results)
        avg_generation_time = sum(r.generation_time for r in approach_results) / len(approach_results)
        avg_token_usage = sum(r.token_usage['total'] for r in approach_results) / len(approach_results)
        avg_answer_quality = sum(
            (r.answer_quality['factual_correctness'] + 
             r.answer_quality['completeness'] + 
             r.answer_quality['clarity']) / 3 
            for r in approach_results
        ) / len(approach_results)
        
        # Store approach metrics
        comparison['approaches'][approach_name] = {
            'text_accuracy': avg_text_accuracy,
            'image_accuracy': avg_image_accuracy,
            'retrieval_time': avg_retrieval_time,
            'generation_time': avg_generation_time,
            'token_usage': avg_token_usage,
            'answer_quality': avg_answer_quality,
            'efficiency': avg_token_usage / (avg_text_accuracy + avg_image_accuracy)  # tokens per correct answer
        }
        
        # Update summary
        if avg_text_accuracy > comparison['summary']['best_text_accuracy']['value']:
            comparison['summary']['best_text_accuracy'] = {'approach': approach_name, 'value': avg_text_accuracy}
        
        if avg_image_accuracy > comparison['summary']['best_image_accuracy']['value']:
            comparison['summary']['best_image_accuracy'] = {'approach': approach_name, 'value': avg_image_accuracy}
        
        if avg_retrieval_time < comparison['summary']['fastest_retrieval']['value']:
            comparison['summary']['fastest_retrieval'] = {'approach': approach_name, 'value': avg_retrieval_time}
        
        if avg_generation_time < comparison['summary']['fastest_generation']['value']:
            comparison['summary']['fastest_generation'] = {'approach': approach_name, 'value': avg_generation_time}
        
        if avg_answer_quality > comparison['summary']['most_comprehensive']['value']:
            comparison['summary']['most_comprehensive'] = {'approach': approach_name, 'value': avg_answer_quality}
        
        efficiency = avg_token_usage / (avg_text_accuracy + avg_image_accuracy)
        if efficiency < comparison['summary']['most_efficient']['value']:
            comparison['summary']['most_efficient'] = {'approach': approach_name, 'value': efficiency}
    
    return comparison

def main():
    """Main function to run the comparison."""
    # Create output directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Initialize test suite
    test_suite = RAGTestSuite("data/synthetic_questions.json")
    
    # Initialize approaches
    approaches = {
        "open_source": OpenSourceRAG(),
        "openai_clip": OpenAIClipRAG(),
        "openai_vision": OpenAIVisionRAG(),
        "hybrid": HybridRAG()
    }
    
    # Run tests for each approach
    results = {}
    for name, approach in approaches.items():
        print(f"\nRunning tests for {name} approach...")
        results[name] = test_suite.run_all_tests(approach)
        test_suite.save_report(f"evaluation_results/{name}_results.json")
    
    # Generate and save comparison report
    comparison = generate_comparison_report(results)
    with open("evaluation_results/comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\nComparison Summary:")
    print("------------------")
    for metric, data in comparison['summary'].items():
        print(f"{metric.replace('_', ' ').title()}: {data['approach']} ({data['value']:.2f})")

if __name__ == "__main__":
    main() 