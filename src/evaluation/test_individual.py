"""
Test individual RAG approaches with detailed evaluation metrics.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import psutil
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.approaches.open_source_rag import OpenSourceRAG
from src.approaches.openai_clip_rag import OpenAIClipRAG
from src.approaches.openai_vision_rag import OpenAIVisionRAG
from src.approaches.hybrid_rag import HybridRAG

@dataclass
class TestResult:
    """Container for test results with metadata."""
    question_id: str
    question: str
    generated_answer: str
    ground_truth: str
    response_time: float
    text_accuracy: float
    image_accuracy: float
    retrieved_sections: List[str]
    retrieved_figures: List[str]
    ground_truth_sections: List[str]
    ground_truth_figures: List[str]
    token_usage: int
    error: bool
    error_type: str
    error_details: str
    language: str

class RAGTestSuite:
    """Test suite for evaluating RAG system performance."""
    
    def __init__(self, questions_path: str):
        """Initialize the test suite with evaluation questions."""
        self.questions = self._load_questions(questions_path)
        self.results: List[TestResult] = []
    
    def _load_questions(self, path: str) -> List[Dict[str, Any]]:
        """Load evaluation questions from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['questions']
    
    def _calculate_text_accuracy(self, generated: str, ground_truth: str) -> float:
        """Calculate text accuracy using semantic similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        generated_embedding = model.encode(generated)
        ground_truth_embedding = model.encode(ground_truth)
        return float(np.dot(generated_embedding, ground_truth_embedding))
    
    def _calculate_image_accuracy(self, retrieved: List[str], ground_truth: List[str]) -> float:
        """Calculate image accuracy based on overlap."""
        if not retrieved or not ground_truth:
            return 0.0
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        intersection = retrieved_set.intersection(ground_truth_set)
        return len(intersection) / len(ground_truth_set)
    
    def run_test(self, question: Dict[str, Any], approach: Any) -> TestResult:
        """
        Run a single test with the given approach.
        
        Args:
            question: The question to test
            approach: The RAG approach to test
            
        Returns:
            TestResult containing the results
        """
        try:
            # Get language from question, default to english
            language = question.get("language", "english")
            
            # Run the test
            start_time = time.time()
            answer, metrics = approach.answer_question(question["question"], language=language)
            end_time = time.time()
            
            # Calculate metrics
            response_time = end_time - start_time
            text_accuracy = self._calculate_text_accuracy(answer, question.get("answer", ""))
            image_accuracy = self._calculate_image_accuracy(metrics.get("retrieved_figures", []), question.get("ground_truth_figures", []))
            
            # Create result
            result = TestResult(
                question_id=question["id"],
                question=question["question"],
                generated_answer=answer,
                ground_truth=question.get("answer", ""),
                response_time=response_time,
                text_accuracy=text_accuracy,
                image_accuracy=image_accuracy,
                retrieved_sections=metrics.get("retrieved_sections", []),
                retrieved_figures=metrics.get("retrieved_figures", []),
                ground_truth_sections=question.get("ground_truth_sections", []),
                ground_truth_figures=question.get("ground_truth_figures", []),
                token_usage=metrics.get("total_tokens", 0),
                error=False,
                error_type=None,
                error_details=None,
                language=language
            )
            
            return result
            
        except Exception as e:
            # Handle errors
            error_type = "retrieval_failure" if isinstance(e, RetrievalError) else "generation_failure"
            return TestResult(
                question_id=question["id"],
                question=question["question"],
                generated_answer="",
                ground_truth=question.get("answer", ""),
                response_time=0,
                text_accuracy=0,
                image_accuracy=0,
                retrieved_sections=[],
                retrieved_figures=[],
                ground_truth_sections=question.get("ground_truth_sections", []),
                ground_truth_figures=question.get("ground_truth_figures", []),
                token_usage=0,
                error=True,
                error_type=error_type,
                error_details=str(e),
                language=question.get("language", "english")
            )
    
    def run_all_tests(self, approach: Any) -> List[TestResult]:
        """Run all tests in the suite."""
        return [self.run_test(q, approach) for q in self.questions]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate aggregate metrics
        metrics = {
            'average_response_time': np.mean([r.response_time for r in self.results]),
            'average_text_accuracy': np.mean([r.text_accuracy for r in self.results]),
            'average_image_accuracy': np.mean([r.image_accuracy for r in self.results]),
            'average_token_usage': np.mean([r.token_usage for r in self.results])
        }
        
        # Group results by language
        language_results = {}
        for result in self.results:
            if result.language not in language_results:
                language_results[result.language] = []
            language_results[result.language].append(result)
        
        # Calculate metrics by language
        language_metrics = {
            lang: {
                'count': len(results),
                'average_response_time': np.mean([r.response_time for r in results]),
                'average_text_accuracy': np.mean([r.text_accuracy for r in results]),
                'average_image_accuracy': np.mean([r.image_accuracy for r in results]),
                'average_token_usage': np.mean([r.token_usage for r in results])
            }
            for lang, results in language_results.items()
        }
        
        return {
            'overall_metrics': metrics,
            'language_breakdown': language_metrics,
            'detailed_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'generated_answer': r.generated_answer,
                    'ground_truth': r.ground_truth,
                    'response_time': r.response_time,
                    'text_accuracy': r.text_accuracy,
                    'image_accuracy': r.image_accuracy,
                    'retrieved_sections': r.retrieved_sections,
                    'retrieved_figures': r.retrieved_figures,
                    'ground_truth_sections': r.ground_truth_sections,
                    'ground_truth_figures': r.ground_truth_figures,
                    'token_usage': r.token_usage,
                    'error': r.error,
                    'error_type': r.error_type,
                    'error_details': r.error_details,
                    'language': r.language
                }
                for r in self.results
            ]
        }
    
    def save_report(self, output_path: str):
        """Save the evaluation report to a file."""
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

def main():
    """Run individual RAG approach tests."""
    parser = argparse.ArgumentParser(description="Test individual RAG approaches")
    parser.add_argument("--approach", type=str, choices=["open_source_rag", "openai_clip_rag", "openai_vision_rag", "hybrid_rag"], 
                        help="The specific RAG approach to test")
    parser.add_argument("--query", type=str, help="A specific query to test")
    parser.add_argument("--language", type=str, default="english", help="The language for the query")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--run_all", action="store_true", help="Run all approaches on predefined questions")
    
    args = parser.parse_args()
    
    # Map approach names to their classes
    approach_map = {
        "open_source_rag": OpenSourceRAG,
        "openai_clip_rag": OpenAIClipRAG,
        "openai_vision_rag": OpenAIVisionRAG,
        "hybrid_rag": HybridRAG
    }
    
    # If a specific approach and query are provided, run that test
    if args.approach and args.query:
        approach_class = approach_map[args.approach]
        approach = approach_class()
        
        print(f"\n=== Testing {args.approach} with query: '{args.query}' ===")
        
        start_time = time.time()
        answer, metrics = approach.answer_question(args.query, language=args.language)
        end_time = time.time()
        
        print(f"\nAnswer: {answer}")
        print(f"\nResponse time: {end_time - start_time:.2f} seconds")
        
        if args.verbose:
            print("\nRetrieved sections:")
            for i, section in enumerate(metrics.get("retrieved_sections", [])):
                print(f"{i+1}. {section[:100]}...")
                
            print("\nRetrieved figures:")
            for i, figure in enumerate(metrics.get("retrieved_figures", [])):
                print(f"{i+1}. {figure}")
                
            print(f"\nTotal tokens: {metrics.get('total_tokens', 0)}")
            print(f"Retrieval tokens: {metrics.get('retrieval_tokens', 0)}")
            print(f"Generation tokens: {metrics.get('generation_tokens', 0)}")
        
        return
    
    # If run_all is specified, run the full test suite
    if args.run_all:
        # Create output directory
        output_dir = "./evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load questions
        questions = [
            # English questions
            {
                "id": "en_wifi_1",
                "question": "How do I configure the WiFi network?",
                "language": "english",
                "answer": "To configure the WiFi network, access the control panel and select network settings.",
                "ground_truth_sections": ["3.2.1"],
                "ground_truth_figures": ["fig3.2"]
            },
            {
                "id": "en_vm_1",
                "question": "What are the first steps to create a new VM?",
                "language": "english",
                "answer": "The first steps to create a new VM include accessing the cloud manager and selecting the operating system.",
                "ground_truth_sections": ["5.5.1"],
                "ground_truth_figures": ["fig5.5"]
            },
            # Italian questions
            {
                "id": "it_wifi_1",
                "question": "Come posso configurare la rete WiFi?",
                "language": "italian",
                "answer": "Per configurare la rete WiFi, accedere al pannello di controllo e selezionare le impostazioni di rete.",
                "ground_truth_sections": ["3.2.1"],
                "ground_truth_figures": ["fig3.2"]
            },
            {
                "id": "it_vm_1",
                "question": "Quali sono i primi passi per creare una nuova VM?",
                "language": "italian",
                "answer": "I primi passi per creare una nuova VM includono l'accesso al cloud manager e la selezione del sistema operativo.",
                "ground_truth_sections": ["5.5.1"],
                "ground_truth_figures": ["fig5.5"]
            }
        ]
        
        # Save questions to a temporary file
        temp_questions_file = os.path.join(output_dir, "temp_questions.json")
        with open(temp_questions_file, 'w', encoding='utf-8') as f:
            json.dump({"questions": questions}, f)
        
        # Initialize test suite
        test_suite = RAGTestSuite(temp_questions_file)
        
        # Test each approach
        approaches = {
            "Open Source RAG": OpenSourceRAG,
            "OpenAI + CLIP RAG": OpenAIClipRAG,
            "OpenAI Vision RAG": OpenAIVisionRAG,
            "Hybrid RAG": HybridRAG
        }
        
        for name, approach_class in approaches.items():
            print(f"\n=== Testing {name} ===")
            approach = approach_class()
            results = test_suite.run_all_tests(approach)
            test_suite.results = results
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.json")
            test_suite.save_report(output_file)
            print(f"Report saved to {output_file}")
        
        # Clean up temporary file
        os.remove(temp_questions_file)
    else:
        # If no specific options are provided, show help
        parser.print_help()

if __name__ == "__main__":
    main() 