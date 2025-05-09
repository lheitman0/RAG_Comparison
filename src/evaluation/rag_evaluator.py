import os
import json
import numpy as np
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.retrieval.rag_retriever import TechnicalManualRetriever
from src.embeddings.vector_store import TechnicalManualVectorStore

class RAGEvaluator:
    def __init__(self, persist_directory: str, manual_type: str):
        self.persist_directory = persist_directory
        self.manual_type = manual_type
        
        vector_store = TechnicalManualVectorStore(persist_directory=persist_directory)
        self.retriever = TechnicalManualRetriever(vector_store=vector_store)
    
    def generate_synthetic_queries(self, num_queries: int = 20) -> List[Dict[str, Any]]:
        base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        chunks_path = base_path / "data" / self.manual_type / f"cleaned_{self.manual_type.split('_')[0]}_chunks.json"
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        
        np.random.seed(42)
        sampled_chunk_indices = np.random.choice(
            len(all_chunks),
            min(num_queries, len(all_chunks)),
            replace=False
        )
        sampled_chunks = [all_chunks[i] for i in sampled_chunk_indices]
        
        synthetic_queries = []
        
        client = openai.OpenAI()
        
        for chunk in tqdm(sampled_chunks, desc="Generating synthetic queries"):
            content = chunk.get("content", "")
            figures = []
            if "figures" in chunk and chunk["figures"]:
                figures = [f["caption"] for f in chunk["figures"] if "caption" in f]
            
            context = f"Document section content: {content[:1000]}"
            if figures:
                context += f"\n\nRelated figures: {'; '.join(figures)}"
            
            system_prompt = """
            You are an evaluation assistant. Your task is to generate realistic questions that a technical user
            might ask about the provided documentation. Generate questions that:
            
            1. Are specific to the information in the context
            2. Range from simple factual questions to complex procedural questions
            3. Sometimes refer to visual elements if figures are mentioned
            4. Use realistic language a user would use when asking for help
            5. Have clear ground truth answers in the content
            
            Generate a single specific question that can be answered using the provided content.
            """
            
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_completion_tokens=150
            )
            
            query = response.choices[0].message.content.strip()
            
            answer_prompt = f"""
            Given the following technical documentation and the question, provide the correct answer:
            
            Documentation:
            {context}
            
            Question: {query}
            
            Provide a concise but complete answer based solely on the provided documentation.
            """
            
            answer_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            expected_answer = answer_response.choices[0].message.content.strip()
            
            synthetic_queries.append({
                "query": query,
                "source_chunk_id": f"{chunk.get('document', 'unknown')}_{chunk.get('section_id', 'unknown')}",
                "expected_answer": expected_answer,
                "figures": figures,
                "is_visual": len(figures) > 0
            })
        
        return synthetic_queries
    
    def evaluate_retrieval(self, synthetic_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {
            "queries": [],
            "metrics": {
                "text_recall": 0.0,
                "image_recall": 0.0,
                "mean_reciprocal_rank": 0.0,
                "relevance_score": 0.0
            }
        }
        
        client = openai.OpenAI()
        
        for query_obj in tqdm(synthetic_queries, desc="Evaluating queries"):
            query = query_obj["query"]
            source_chunk_id = query_obj["source_chunk_id"]
            expected_answer = query_obj["expected_answer"]
            has_visual = query_obj["is_visual"]
            
            retrieval_result = self.retriever.retrieve(
                query=query,
                n_text_results=5,
                n_image_results=3 if has_visual else 2
            )
            
            source_retrieved = False
            retrieval_position = -1
            
            for i, result in enumerate(retrieval_result["text_results"]):
                metadata = result.get("metadata", {})
                result_id = f"{metadata.get('document', 'unknown')}_{metadata.get('section_id', 'unknown')}"
                if result_id == source_chunk_id:
                    source_retrieved = True
                    retrieval_position = i
                    break
            
            image_recall = 0.0
            if has_visual and retrieval_result["image_results"]:
                image_recall = 0.5
                
                expected_figures = query_obj["figures"]
                retrieved_captions = [
                    r.get("metadata", {}).get("figure_caption", "")
                    for r in retrieval_result["image_results"]
                ]
                
                for expected in expected_figures:
                    for retrieved in retrieved_captions:
                        if expected.lower() in retrieved.lower() or retrieved.lower() in expected.lower():
                            image_recall = 1.0
                            break
            
            mrr = 0.0
            if retrieval_position >= 0:
                mrr = 1.0 / (retrieval_position + 1)
            
            relevance_score = 0.0
            if "answer" in retrieval_result and retrieval_result["answer"]:
                system_prompt = """
                You are an evaluation assistant. Your task is to score the relevance of a generated answer
                compared to the expected ground truth answer.
                
                Score on a scale of 0-10 where:
                0: Completely irrelevant or incorrect
                5: Partially relevant but missing key information
                10: Fully relevant and accurate
                
                Return only the numeric score.
                """
                
                user_prompt = f"""
                Question: {query}
                
                Expected ground truth answer:
                {expected_answer}
                
                Generated answer:
                {retrieval_result["answer"]}
                
                Score (0-10):
                """
                
                response = client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                
                try:
                    relevance_score = float(response.choices[0].message.content.strip())
                    relevance_score = relevance_score / 10.0
                except:
                    relevance_score = 0.5
            
            query_result = {
                "query": query,
                "source_retrieved": source_retrieved,
                "retrieval_position": retrieval_position,
                "mrr": mrr,
                "image_recall": image_recall,
                "answer_relevance": relevance_score,
                "expected_answer": expected_answer,
                "generated_answer": retrieval_result.get("answer", "")
            }
            
            results["queries"].append(query_result)
        
        if results["queries"]:
            results["metrics"]["text_recall"] = sum(1 for q in results["queries"] if q["source_retrieved"]) / len(results["queries"])
            results["metrics"]["mean_reciprocal_rank"] = sum(q["mrr"] for q in results["queries"]) / len(results["queries"])
            results["metrics"]["image_recall"] = sum(q["image_recall"] for q in results["queries"]) / len(results["queries"])
            results["metrics"]["relevance_score"] = sum(q["answer_relevance"] for q in results["queries"]) / len(results["queries"])
        
        return results
    
    def run_full_evaluation(self, num_queries: int = 20, output_path: Optional[str] = None) -> Dict[str, Any]:
        print(f"Generating {num_queries} synthetic queries...")
        synthetic_queries = self.generate_synthetic_queries(num_queries)
        
        print("Evaluating retrieval performance...")
        results = self.evaluate_retrieval(synthetic_queries)
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Text Recall: {results['metrics']['text_recall']:.2f}")
        print(f"Image Recall: {results['metrics']['image_recall']:.2f}")
        print(f"Mean Reciprocal Rank: {results['metrics']['mean_reciprocal_rank']:.2f}")
        print(f"Answer Relevance: {results['metrics']['relevance_score']:.2f}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to {output_path}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--manual-type", 
        type=str, 
        choices=["VM_manual", "wifi_manual"],
        default="VM_manual",
        help="Type of manual to evaluate"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./vector_db",
        help="Base directory where the vector store is persisted"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of synthetic queries to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file to save evaluation results"
    )
    
    args = parser.parse_args()
    
    persist_dir = os.path.join(args.persist_dir, args.manual_type)
    
    evaluator = RAGEvaluator(persist_dir, args.manual_type)
    evaluator.run_full_evaluation(args.num_queries, args.output)