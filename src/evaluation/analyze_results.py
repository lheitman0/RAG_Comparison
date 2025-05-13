"""
Script to analyze existing evaluation results and generate final report
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def load_results(output_dir: str) -> List[Dict[str, Any]]:
    results = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json") and not filename.startswith("comparison_"):
            with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                results.append(json.load(f))
    return results

def perform_statistical_tests(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    approach_values = {}
    for result in results:
        approach = result["approach"]
        values = []
        for manual in ["wifi_manual", "vm_manual"]:
            if metric in result[manual]["overall_metrics"]:
                values.append(float(result[manual]["overall_metrics"][metric]))
        approach_values[approach] = values
    
    if len(approach_values) > 2:
        f_stat, p_value = stats.f_oneway(*[values for values in approach_values.values()])
        anova_result = {
            "f_statistic": float(f_stat) if not np.isnan(f_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "significant": bool(p_value < 0.05) if not np.isnan(p_value) else None
        }
    else:
        anova_result = None
    
    pairwise_comparisons = {}
    approaches = list(approach_values.keys())
    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            approach1, approach2 = approaches[i], approaches[j]
            t_stat, p_value = stats.ttest_ind(
                approach_values[approach1],
                approach_values[approach2]
            )
            pairwise_comparisons[f"{approach1}_vs_{approach2}"] = {
                "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                "p_value": float(p_value) if not np.isnan(p_value) else None,
                "significant": bool(p_value < 0.05) if not np.isnan(p_value) else None
            }
    
    return {
        "anova": anova_result,
        "pairwise_comparisons": pairwise_comparisons
    }

def analyze_errors(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    error_categories = {
        "retrieval_failure": 0,
        "generation_failure": 0,
        "hallucination": 0,
        "incomplete_answer": 0,
        "irrelevant_content": 0
    }
    
    for result in results:
        for manual in ["wifi_manual", "vm_manual"]:
            if "error_analysis" in result[manual]:
                for category, count in result[manual]["error_analysis"].items():
                    if category in error_categories:
                        error_categories[category] += count
    
    return error_categories

def calculate_cost_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    cost_analysis = {}
    
    for result in results:
        approach = result["approach"]
        total_tokens = 0
        total_cost = 0
        
        for manual in ["wifi_manual", "vm_manual"]:
            if "token_usage" in result[manual]:
                tokens = result[manual]["token_usage"]
                total_tokens += tokens.get("total", 0)
                cost_per_1k_tokens = 0.002  
                total_cost += (tokens.get("total", 0) / 1000) * cost_per_1k_tokens
        
        questions_per_manual = 15  # We know this from our evaluation setup
        
        cost_analysis[approach] = {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "cost_per_query": total_cost / (questions_per_manual * 2)  
        }
    
    return cost_analysis

def generate_visualizations(results: List[Dict[str, Any]], output_dir: str) -> None:
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    metrics = [
        "average_retrieval_time", 
        "average_text_accuracy", 
        "average_image_accuracy",
        "semantic_accuracy",
        "technical_correctness",
        "completeness",
        "clarity",
        "relevance"
    ]
    data = []
    
    for result in results:
        approach = result["approach"]
        for manual in ["wifi_manual", "vm_manual"]:
            for metric in ["average_retrieval_time", "average_text_accuracy", "average_image_accuracy"]:
                if metric in result[manual]["overall_metrics"]:
                    data.append({
                        "approach": approach,
                        "manual": manual.replace("_manual", ""),
                        "metric": metric,
                        "value": result[manual]["overall_metrics"][metric]
                    })
            
            if "average_answer_quality" in result[manual]["overall_metrics"]:
                for quality_metric in ["semantic_accuracy", "technical_correctness", 
                                     "completeness", "clarity", "relevance"]:
                    data.append({
                        "approach": approach,
                        "manual": manual.replace("_manual", ""),
                        "metric": quality_metric,
                        "value": result[manual]["overall_metrics"]["average_answer_quality"][quality_metric]
                    })
    
    df = pd.DataFrame(data)
    
    performance_metrics = ["average_retrieval_time", "average_text_accuracy", "average_image_accuracy"]
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(performance_metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(data=df[df["metric"] == metric], x="approach", y="value", hue="manual")
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "performance_metrics.png"))
    plt.close()
    
    # Generate plots for answer quality metrics
    quality_metrics = ["semantic_accuracy", "technical_correctness", "completeness", "clarity", "relevance"]
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(quality_metrics, 1):
        plt.subplot(2, 3, i)
        sns.barplot(data=df[df["metric"] == metric], x="approach", y="value", hue="manual")
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "answer_quality_metrics.png"))
    plt.close()

def main():
    output_dir = "./evaluation_results"
    
    results = load_results(output_dir)
    
    metrics_to_test = ["average_retrieval_time", "average_text_accuracy", "average_image_accuracy"]
    statistical_results = {}
    for metric in metrics_to_test:
        statistical_results[metric] = perform_statistical_tests(results, metric)
    
    error_analysis = analyze_errors(results)
    
    cost_analysis = calculate_cost_analysis(results)
    
    generate_visualizations(results, output_dir)
    
    comparison = {
        "approaches": [],
        "statistical_analysis": statistical_results,
        "error_analysis": error_analysis,
        "cost_analysis": cost_analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    for result in results:
        approach_data = {
            "name": result["approach"],
            "wifi_manual": {
                "average_retrieval_time": result["wifi_manual"]["overall_metrics"]["average_retrieval_time"],
                "average_generation_time": result["wifi_manual"]["overall_metrics"]["average_generation_time"],
                "average_text_accuracy": result["wifi_manual"]["overall_metrics"]["average_text_accuracy"],
                "average_image_accuracy": result["wifi_manual"]["overall_metrics"]["average_image_accuracy"],
                "average_answer_quality": result["wifi_manual"]["overall_metrics"]["average_answer_quality"],
                "resource_usage": result["wifi_manual"].get("resource_usage", {})
            },
            "vm_manual": {
                "average_retrieval_time": result["vm_manual"]["overall_metrics"]["average_retrieval_time"],
                "average_generation_time": result["vm_manual"]["overall_metrics"]["average_generation_time"],
                "average_text_accuracy": result["vm_manual"]["overall_metrics"]["average_text_accuracy"],
                "average_image_accuracy": result["vm_manual"]["overall_metrics"]["average_image_accuracy"],
                "average_answer_quality": result["vm_manual"]["overall_metrics"]["average_answer_quality"],
                "resource_usage": result["vm_manual"].get("resource_usage", {})
            }
        }
        comparison["approaches"].append(approach_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"{output_dir}/comparison_{timestamp}.json"
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Comparison saved to {comparison_file}")
    print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")

if __name__ == "__main__":
    main() 