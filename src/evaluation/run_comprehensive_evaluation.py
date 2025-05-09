import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.retrieval.base import retrieve, RetrievalRecipe, RetrievalApproach

class RAGEvaluator:
    def __init__(self, 
                 output_dir: str = "./evaluation_results",
                 questions_per_manual: int = 15,
                 languages: List[str] = ["english", "italian"],
                 seed: int = 42):
        self.output_dir = output_dir
        self.questions_per_manual = questions_per_manual
        self.languages = languages
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.openai_client = OpenAI()
        
        self.approaches = {
            "OpenSource RAG": {
                "recipe": RetrievalRecipe.open_source(),
                "vector_store": "vector_store/opensource"
            },
            "OpenAI CLIP RAG": {
                "recipe": RetrievalRecipe.openai_clip(),
                "vector_store": "vector_store/openai"
            },
            "OpenAI Vision RAG": {
                "recipe": RetrievalRecipe.openai_vision(),
                "vector_store": "vector_store/openai"
            },
            "Hybrid RAG": {
                "recipe": RetrievalRecipe.hybrid(),
                "vector_store": "vector_store/openai"
            }
        }
    
    def load_questions(self, manual_type: str) -> List[Dict[str, Any]]:
        if manual_type == "VM_manual":
            file_path = "evaluation_data/vm_synthetic_questions.json"
        else:
            file_path = "evaluation_data/wifi_synthetic_questions.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_questions = data["questions"]
                
                english_questions = [q for q in all_questions if "question_en" in q]
                italian_questions = [q for q in all_questions if "question_it" in q]
                
                random.seed(self.seed)
                
                sampled_questions = []
                
                if "english" in self.languages:
                    random.shuffle(english_questions)
                    english_sample = english_questions[:self.questions_per_manual]
                    for q in english_sample:
                        q["language"] = "english"
                        q["question"] = q["question_en"]
                        q["ground_truth"] = q["ground_truth_en"]
                    sampled_questions.extend(english_sample)
                
                if "italian" in self.languages:
                    random.shuffle(italian_questions)
                    italian_sample = italian_questions[:self.questions_per_manual]
                    for q in italian_sample:
                        q["language"] = "italian"
                        q["question"] = q["question_it"]
                        q["ground_truth"] = q["ground_truth_it"]
                    sampled_questions.extend(italian_sample)
                
                for q in sampled_questions:
                    if "manual_type" not in q:
                        q["manual_type"] = manual_type
                
                return sampled_questions
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading questions for {manual_type}: {e}")
            return []
    
    def generate_answer(self, 
                      query: str, 
                      context_hits: List[Any],
                      figure_paths: List[str] = None,
                      language: str = "english") -> Tuple[str, Dict[str, Any]]:
        context_text = ""
        for i, hit in enumerate(context_hits):
            if hasattr(hit, 'content'):
                content = hit.content
                source = hit.metadata.get('document', 'unknown')
                section = hit.metadata.get('section_id', '')
                context_text += f"\n--- Document {i+1} [Source: {source}, Section: {section}] ---\n{content}\n"
            elif isinstance(hit, dict) and 'content' in hit:
                content = hit['content']
                source = hit.get('metadata', {}).get('document', 'unknown')
                section = hit.get('metadata', {}).get('section_id', '')
                context_text += f"\n--- Document {i+1} [Source: {source}, Section: {section}] ---\n{content}\n"
        
        language_instruction = f"Please respond in {language}." if language != "english" else ""
        
        has_figures = figure_paths and len(figure_paths) > 0
        
        metrics = {
            "generation_tokens": 0,
            "has_figures": has_figures,
            "num_figures": len(figure_paths) if has_figures else 0
        }
        
        try:
            if has_figures:
                messages = [
                    {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context and images to answer questions accurately and comprehensively. {language_instruction}"},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"""Based on the following technical documentation and images, please answer the question:
                        
Question: {query}

Relevant Documentation:
{context_text}

Please analyze both the text and the provided images to give a complete answer."""}
                    ]}
                ]
                
                for figure_path in figure_paths[:5]:
                    if os.path.exists(figure_path):
                        with open(figure_path, "rb") as image_file:
                            import base64
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            messages[1]["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content
                metrics["generation_tokens"] = response.usage.total_tokens
                metrics["model"] = "gpt-4o"
                
            else:
                messages = [
                    {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context to answer questions accurately and comprehensively. {language_instruction}"},
                    {"role": "user", "content": f"""Based on the following technical documentation, please answer the question:
                    
Question: {query}

Relevant Documentation:
{context_text}"""}
                ]
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                
                answer = response.choices[0].message.content
                metrics["generation_tokens"] = response.usage.total_tokens
                metrics["model"] = "gpt-4o"
            
            return answer, metrics
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}", {"error": str(e), "generation_tokens": 0}
    
    def evaluate_answer(self, 
                       question: str, 
                       ground_truth: str, 
                       generated_answer: str) -> Dict[str, float]:
        prompt = f"""You are an expert evaluator of RAG (Retrieval Augmented Generation) systems. 
Please evaluate this answer based on the question and ground truth.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

Score the answer on these metrics (0-10 scale):
1. RELEVANCE: How relevant is the answer to the question?
2. CORRECTNESS: How factually accurate is the answer compared to ground truth?
3. COMPLETENESS: How completely does the answer address the question?
4. CONCISENESS: How concise and to-the-point is the answer?
5. COHERENCE: How well-structured and logical is the answer?
6. HARMFULNESS: Is the answer free from harmful content? (10=completely safe)

Also provide:
- OVERALL_SCORE: A weighted average of all scores (0-10)
- EXPLANATION: Brief explanation of your evaluation (1-2 sentences)

Format your response as a JSON object with these exact fields.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG systems."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            
            for key, value in evaluation.items():
                if key != "EXPLANATION":
                    evaluation[key] = float(value)
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "RELEVANCE": 0.0,
                "CORRECTNESS": 0.0,
                "COMPLETENESS": 0.0,
                "CONCISENESS": 0.0,
                "COHERENCE": 0.0,
                "HARMFULNESS": 10.0,
                "OVERALL_SCORE": 0.0,
                "EXPLANATION": f"Evaluation failed with error: {str(e)}"
            }
    
    def run_evaluation(self, approach_name: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        approach_info = self.approaches[approach_name]
        retrieval_recipe = approach_info["recipe"]
        
        print(f"\n=== Initialized {approach_name} ===")
        
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Evaluating question: {question['id']}")
            print(f"Query: {question['question'][:100]}...")
            print(f"Language: {question.get('language', 'english')}")
            print(f"Manual: {question.get('manual_type', 'unknown')}")
            
            try:
                start_time = time.time()
                
                language = question.get("language", "english")
                
                context_hits, retrieval_metrics = retrieve(
                    query=question["question"],
                    recipe=retrieval_recipe,
                    k=8
                )
                
                figure_paths = []
                for hit in context_hits:
                    if hasattr(hit, 'metadata') and 'figure_path' in hit.metadata:
                        figure_path = hit.metadata['figure_path']
                        if os.path.exists(figure_path):
                            figure_paths.append(figure_path)
                    elif isinstance(hit, dict) and 'metadata' in hit and 'figure_path' in hit['metadata']:
                        figure_path = hit['metadata']['figure_path']
                        if os.path.exists(figure_path):
                            figure_paths.append(figure_path)
                
                answer, generation_metrics = self.generate_answer(
                    query=question["question"],
                    context_hits=context_hits,
                    figure_paths=figure_paths,
                    language=language
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"Generated answer: {answer[:100]}...")
                print(f"Response time: {response_time:.2f}s")
                
                evaluation = self.evaluate_answer(
                    question=question["question"],
                    ground_truth=question["ground_truth"],
                    generated_answer=answer
                )
                
                retrieved_sections = [hit.content if hasattr(hit, 'content') else hit.get('content', '') 
                                     for hit in context_hits]
                
                total_tokens = retrieval_metrics.get("retrieval_tokens", 0) + generation_metrics.get("generation_tokens", 0)
                
                result = {
                    "question_id": question["id"],
                    "question": question["question"],
                    "language": language,
                    "manual_type": question.get("manual_type", "unknown"),
                    "category": question.get("category", "unknown"),
                    "complexity": question.get("complexity", "unknown"),
                    "generated_answer": answer,
                    "ground_truth": question["ground_truth"],
                    "relevant_sections": question.get("relevant_sections", []),
                    "relevant_figures": question.get("relevant_figures", []),
                    "retrieved_sections": retrieved_sections,
                    "retrieved_figures": figure_paths,
                    "response_time": response_time,
                    "total_tokens": total_tokens,
                    "retrieval_tokens": retrieval_metrics.get("retrieval_tokens", 0),
                    "generation_tokens": generation_metrics.get("generation_tokens", 0),
                    "evaluation": evaluation,
                    "approach": approach_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                models_used = {
                    "embedding": {
                        "name": retrieval_recipe.approach.value,
                        "type": "openai" if retrieval_recipe.approach.value in ["openai_clip", "openai_vision", "hybrid"] else "open_source",
                        "billable": retrieval_recipe.approach.value != "open_source",
                        "tokens": retrieval_metrics.get("retrieval_tokens", 0) // 2
                    },
                    "generation": {
                        "name": generation_metrics.get("model", "gpt-4o"),
                        "type": "openai",
                        "billable": True,
                        "tokens": generation_metrics.get("generation_tokens", 0)
                    }
                }
                
                if generation_metrics.get("has_figures", False):
                    models_used["image_embedding"] = {
                        "name": "CLIP",
                        "type": "open_source",
                        "billable": False,
                        "tokens": len(figure_paths) * 50
                    }
                
                result["models_used"] = models_used
                
                result["billable_tokens"] = {
                    "total": (retrieval_metrics.get("retrieval_tokens", 0) // 2 if models_used["embedding"]["billable"] else 0) + 
                             generation_metrics.get("generation_tokens", 0),
                    "embedding": retrieval_metrics.get("retrieval_tokens", 0) // 2 if models_used["embedding"]["billable"] else 0,
                    "generation": generation_metrics.get("generation_tokens", 0)
                }
                
                results.append(result)
                
                print(f"Evaluation: Overall score = {evaluation.get('OVERALL_SCORE', 0):.1f}/10")
                
            except Exception as e:
                print(f"Error processing question {question['id']}: {e}")
                traceback.print_exc()
                
                results.append({
                    "question_id": question["id"],
                    "question": question["question"],
                    "language": question.get("language", "english"),
                    "manual_type": question.get("manual_type", "unknown"),
                    "error": True,
                    "error_message": str(e),
                    "approach": approach_name,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_results = [r for r in results if "error" not in r or not r["error"]]
        
        if not valid_results:
            return {
                "overall": {
                    "total_questions": len(results),
                    "successful_questions": 0,
                    "success_rate": 0.0,
                    "avg_response_time": 0.0,
                    "avg_overall_score": 0.0,
                    "avg_relevance": 0.0,
                    "avg_correctness": 0.0,
                    "avg_completeness": 0.0,
                    "avg_clarity": 0.0
                },
                "token_usage": {
                    "total_tokens": 0,
                    "embedding_tokens": 0,
                    "generation_tokens": 0,
                    "query_processing_tokens": 0
                },
                "models_used": {},
                "model_token_usage": {}
            }
        
        total_questions = len(results)
        successful_questions = len(valid_results)
        success_rate = successful_questions / total_questions if total_questions > 0 else 0
        
        avg_response_time = np.mean([r.get("response_time", 0) for r in valid_results])
        
        avg_overall_score = np.mean([r.get("evaluation", {}).get("OVERALL_SCORE", 0) for r in valid_results])
        avg_relevance = np.mean([r.get("evaluation", {}).get("RELEVANCE", 0) for r in valid_results])
        avg_correctness = np.mean([r.get("evaluation", {}).get("CORRECTNESS", 0) for r in valid_results]) 
        avg_completeness = np.mean([r.get("evaluation", {}).get("COMPLETENESS", 0) for r in valid_results])
        avg_clarity = np.mean([r.get("evaluation", {}).get("CONCISENESS", 0) for r in valid_results])
        
        total_tokens = sum([r.get("total_tokens", 0) for r in valid_results])
        retrieval_tokens = sum([r.get("retrieval_tokens", 0) for r in valid_results])
        generation_tokens = sum([r.get("generation_tokens", 0) for r in valid_results])
        
        query_processing_tokens = 0
        models_used = {}
        model_token_usage = {}
        
        for r in valid_results:
            if "models_used" in r and isinstance(r["models_used"], dict):
                for model_key, model_info in r["models_used"].items():
                    model_name = model_info.get("name", "unknown")
                    model_type = model_info.get("type", "unknown")
                    is_billable = model_info.get("billable", False)
                    tokens = model_info.get("tokens", 0)
                    
                    if model_key not in models_used:
                        models_used[model_key] = {
                            "name": model_name,
                            "type": model_type,
                            "billable": is_billable,
                            "tokens": 0
                        }
                    
                    models_used[model_key]["tokens"] += tokens
                    
                    if model_name not in model_token_usage:
                        model_token_usage[model_name] = {
                            "tokens": 0,
                            "type": model_type,
                            "billable": is_billable,
                            "purpose": model_key
                        }
                    
                    model_token_usage[model_name]["tokens"] += tokens
                    
                    if model_key == "query_processor":
                        query_processing_tokens += tokens
            
            elif "model_token_usage" in r and isinstance(r["model_token_usage"], dict):
                for model_name, usage in r["model_token_usage"].items():
                    tokens = usage.get("tokens", 0)
                    model_type = usage.get("type", "unknown")
                    is_billable = usage.get("billable", False)
                    purpose = usage.get("purpose", "unknown")
                    
                    if model_name not in model_token_usage:
                        model_token_usage[model_name] = {
                            "tokens": 0,
                            "type": model_type,
                            "billable": is_billable,
                            "purpose": purpose
                        }
                    
                    model_token_usage[model_name]["tokens"] += tokens
                    
                    if purpose == "query_processing":
                        query_processing_tokens += tokens
            
            elif "query_processing_tokens" in r:
                query_processing_tokens += r["query_processing_tokens"]
                
                if "query_processor" not in model_token_usage:
                    model_token_usage["query_processor"] = {
                        "tokens": 0,
                        "type": "unknown",
                        "billable": True,
                        "purpose": "query_processing"
                    }
                
                model_token_usage["query_processor"]["tokens"] += r["query_processing_tokens"]
                    
        embedding_tokens = retrieval_tokens
        for model_key, model_info in models_used.items():
            if "embedding" in model_key.lower():
                embedding_tokens = model_info.get("tokens", retrieval_tokens)
                break
        
        return {
            "overall": {
                "total_questions": total_questions,
                "successful_questions": successful_questions,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "avg_overall_score": avg_overall_score,
                "avg_relevance": avg_relevance,
                "avg_correctness": avg_correctness,
                "avg_completeness": avg_completeness,
                "avg_clarity": avg_clarity
            },
            "token_usage": {
                "total_tokens": total_tokens,
                "embedding_tokens": embedding_tokens,
                "generation_tokens": generation_tokens,
                "query_processing_tokens": query_processing_tokens
            },
            "models_used": models_used,
            "model_token_usage": model_token_usage
        }
    
    def generate_visualizations(self, approach_results: Dict[str, List[Dict[str, Any]]]) -> None:
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        
        approaches = list(approach_results.keys())
        summaries = {
            name: self.generate_summary(results) for name, results in approach_results.items()
        }
        
        avg_scores = [summaries[name]["overall"]["avg_overall_score"] for name in approaches]
        response_times = [summaries[name]["overall"]["avg_response_time"] for name in approaches]
        
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(approaches))
        
        plt.bar(x, avg_scores, bar_width, label='Overall Score (out of 10)')
        
        plt.xlabel('Approach')
        plt.ylabel('Average Score')
        plt.title('Overall Performance Comparison')
        plt.xticks(x, approaches, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "visualizations", "overall_comparison.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(x, response_times, bar_width, label='Response Time (seconds)')
        
        plt.xlabel('Approach')
        plt.ylabel('Average Response Time (s)')
        plt.title('Response Time Comparison')
        plt.xticks(x, approaches, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "visualizations", "response_time_comparison.png"))
        plt.close()
        
        plt.figure(figsize=(14, 8))
        
        model_token_data = {}
        for name in approaches:
            if "model_token_usage" in summaries[name]:
                model_token_data[name] = summaries[name]["model_token_usage"]
        
        if model_token_data:
            all_models = set()
            for approach_data in model_token_data.values():
                all_models.update(approach_data.keys())
            
            sorted_models = sorted(all_models, key=lambda m: (0 if "gpt" in m.lower() else (1 if "text-embedding" in m.lower() else 2), m))
            
            model_usage_data = []
            for approach in approaches:
                if approach in model_token_data:
                    for model in sorted_models:
                        if model in model_token_data[approach]:
                            tokens = model_token_data[approach][model].get("tokens", 0)
                            purpose = model_token_data[approach][model].get("purpose", "unknown")
                            is_billable = model_token_data[approach][model].get("billable", False)
                            
                            model_usage_data.append({
                                "Approach": approach,
                                "Model": model,
                                "Tokens": tokens,
                                "Purpose": purpose,
                                "Billable": "Yes" if is_billable else "No"
                            })
            
            if model_usage_data:
                model_df = pd.DataFrame(model_usage_data)
                
                pivot_df = model_df.pivot_table(
                    index='Approach', 
                    columns='Model', 
                    values='Tokens',
                    aggfunc='sum',
                    fill_value=0
                )
                
                pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
                plt.title('Token Usage by Model', fontsize=16)
                plt.xlabel('Approach', fontsize=14)
                plt.ylabel('Number of Tokens', fontsize=14)
                plt.xticks(rotation=30, ha='right')
                plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                for i, approach in enumerate(pivot_df.index):
                    total = pivot_df.loc[approach].sum()
                    plt.text(i, total + total*0.02, f"{int(total):,}", ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "visualizations", "token_usage_by_model.png"), dpi=300)
                plt.close()
                
                bill_data = []
                for approach in approaches:
                    if approach in model_token_data:
                        billable_tokens = sum(
                            info.get("tokens", 0) 
                            for model, info in model_token_data[approach].items() 
                            if info.get("billable", False)
                        )
                        non_billable_tokens = sum(
                            info.get("tokens", 0) 
                            for model, info in model_token_data[approach].items() 
                            if not info.get("billable", False)
                        )
                        
                        bill_data.append({
                            "Approach": approach,
                            "Billable": billable_tokens,
                            "Non-Billable": non_billable_tokens
                        })
                
                if bill_data:
                    bill_df = pd.DataFrame(bill_data)
                    bill_df.set_index("Approach", inplace=True)
                    
                    bill_df.plot(kind='bar', stacked=True, figsize=(12, 7), color=['#F44336', '#4CAF50'])
                    plt.title('Billable vs Non-Billable Token Usage', fontsize=16)
                    plt.xlabel('Approach', fontsize=14)
                    plt.ylabel('Number of Tokens', fontsize=14)
                    plt.xticks(rotation=30, ha='right')
                    plt.legend(title='Token Type')
                    plt.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    for i, approach in enumerate(bill_df.index):
                        total = bill_df.loc[approach].sum()
                        plt.text(i, total + total*0.02, f"{int(total):,}", ha='center', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "visualizations", "billable_token_usage.png"), dpi=300)
                    plt.close()
        
        print(f"Visualizations saved to {os.path.join(self.output_dir, 'visualizations')}")
    
    def run_comprehensive_evaluation(self) -> None:
        print("Loading questions...")
        wifi_questions = self.load_questions("wifi_manual")
        vm_questions = self.load_questions("VM_manual")
        
        all_questions = wifi_questions + vm_questions
        
        if not all_questions:
            print("Error: No questions loaded. Make sure the evaluation data exists.")
            return
        
        print(f"Loaded {len(all_questions)} questions total")
        print(f"- WiFi Manual: {len(wifi_questions)} questions")
        print(f"- VM Manual: {len(vm_questions)} questions")
        
        approach_results = {}
        
        for approach_name in self.approaches.keys():
            print(f"\n======== Evaluating {approach_name} ========")
            results = self.run_evaluation(approach_name, all_questions)
            approach_results[approach_name] = results
            
            if results:
                summary = self.generate_summary(results)
                
                results_file = os.path.join(
                    self.output_dir, 
                    f"{approach_name.lower().replace(' ', '_')}_{self.timestamp}.json"
                )
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "approach": approach_name,
                        "summary": summary,
                        "results": results
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"Results saved to {results_file}")
                
                self.print_approach_summary(approach_name, summary)
            else:
                print(f"No results generated for {approach_name}")
        
        print("\n======== Generating Visualizations ========")
        self.generate_visualizations(approach_results)
        
        combined_file = os.path.join(self.output_dir, f"combined_results_{self.timestamp}.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": self.timestamp,
                "approaches": {name: self.generate_summary(results) for name, results in approach_results.items() if results}
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Combined results saved to {combined_file}")
        print("\nComprehensive evaluation complete!")

    def print_approach_summary(self, approach_name: str, summary: Dict[str, Any]) -> None:
        print("\nSummary Statistics:")
        print(f"Total Questions: {summary['overall']['total_questions']}")
        print(f"Success Rate: {summary['overall']['success_rate']:.2f}")
        print(f"Average Overall Score: {summary['overall']['avg_overall_score']:.2f}/10")
        print(f"Average Response Time: {summary['overall']['avg_response_time']:.2f}s")
        
        if "token_usage" in summary:
            print("\nToken Usage:")
            print(f"Total Tokens: {summary['token_usage']['total_tokens']:,}")
            
            if "model_token_usage" in summary and summary["model_token_usage"]:
                print("\nToken Usage by Model:")
                for model_name, usage in summary["model_token_usage"].items():
                    billable = "billable" if usage.get("billable", False) else "non-billable"
                    purpose = usage.get("purpose", "unknown")
                    tokens = usage.get("tokens", 0)
                    print(f"  - {model_name} ({purpose}, {billable}): {tokens:,}")
            else:
                print(f"  - Embedding: {summary['token_usage']['embedding_tokens']:,}")
                print(f"  - Generation: {summary['token_usage']['generation_tokens']:,}")
                if summary['token_usage'].get('query_processing_tokens', 0) > 0:
                    print(f"  - Query Processing: {summary['token_usage']['query_processing_tokens']:,}")
        
        print("")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a comprehensive evaluation of RAG approaches")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--questions", type=int, default=15, help="Number of questions per manual")
    parser.add_argument("--languages", type=str, nargs="+", default=["english", "italian"], help="Languages to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(
        output_dir=args.output_dir,
        questions_per_manual=args.questions,
        languages=args.languages,
        seed=args.seed
    )
    
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main() 