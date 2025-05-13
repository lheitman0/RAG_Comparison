"""
Visualization Generator for RAG Evaluation Results

This script generates detailed visualizations showing:
1. Question distribution by category, complexity, and language
2. Performance comparisons across different RAG approaches
3. Cross-language performance analysis
4. Cost-efficiency metrics

"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from pathlib import Path
from scipy import stats

# Set style for all visualizations
plt.style.use('ggplot')
# Use a more professional and distinct color palette
sns.set_palette("deep")
sns.set_context("talk")

def load_evaluation_results(results_dir: str = "./evaluation_results") -> Dict[str, Any]:
    """
    Load the combined evaluation results file.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Find the most recent combined results file
    combined_files = [f for f in os.listdir(results_dir) if f.startswith("combined_results_")]
    if not combined_files:
        raise FileNotFoundError("No combined results file found")
    
    latest_file = sorted(combined_files)[-1]
    file_path = os.path.join(results_dir, latest_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_test_questions(data_dir: str = "./evaluation_data") -> List[Dict[str, Any]]:
    """
    Load the test questions from both VM and WiFi manuals.
    
    Args:
        data_dir: Directory containing evaluation data
        
    Returns:
        List of test questions
    """
    all_questions = []
    
    # Load VM questions
    vm_path = os.path.join(data_dir, "vm_synthetic_questions.json")
    if os.path.exists(vm_path):
        with open(vm_path, 'r', encoding='utf-8') as f:
            vm_data = json.load(f)
            if "questions" in vm_data:
                all_questions.extend(vm_data["questions"])
    
    # Load WiFi questions
    wifi_path = os.path.join(data_dir, "wifi_synthetic_questions.json")
    if os.path.exists(wifi_path):
        with open(wifi_path, 'r', encoding='utf-8') as f:
            wifi_data = json.load(f)
            if "questions" in wifi_data:
                all_questions.extend(wifi_data["questions"])
    
    return all_questions

def visualize_question_distribution(questions: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations showing the distribution of questions.
    
    Args:
        questions: List of test questions
        output_dir: Directory to save visualizations
    """
    # This function is simplified to remove unnecessary visualizations
    pass

def analyze_language_performance(results: Dict[str, Any], questions: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Analyze and visualize performance differences between languages.
    
    Args:
        results: Dictionary with evaluation results
        questions: List of test questions
        output_dir: Directory to save visualizations
    """
    # This function is simplified to remove unnecessary visualizations
    pass

def visualize_performance_metrics(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create visualizations for performance metrics across approaches.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save visualizations
    """
    approaches = results.get("approaches", {})
    approach_names = list(approaches.keys())
    
    # Extract metrics
    metrics = {
        "Overall Score": [approaches[a]["overall"]["avg_overall_score"] for a in approach_names],
        "Response Time (s)": [approaches[a]["overall"]["avg_response_time"] for a in approach_names],
        "Token Usage": [approaches[a]["token_usage"]["total_tokens"] for a in approach_names],
        "Relevance": [approaches[a]["overall"]["avg_relevance"] for a in approach_names],
        "Correctness": [approaches[a]["overall"]["avg_correctness"] for a in approach_names],
        "Completeness": [approaches[a]["overall"]["avg_completeness"] for a in approach_names]
    }
    
    # Extract model-specific token usage if available
    model_token_data = {}
    for approach in approach_names:
        if "model_token_usage" in approaches[approach]:
            model_token_data[approach] = approaches[approach]["model_token_usage"]
    
    # 1. Response Time (separate chart)
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=approach_names, y=metrics["Response Time (s)"], palette="deep")
    
    plt.title("Average Response Time", fontsize=16, pad=20)
    plt.ylabel("Seconds", fontsize=14)
    plt.xlabel("RAG Approach", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "response_time_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Token Usage Breakdown by Model (stacked bar chart - per query average)
    if model_token_data:
        # Get all unique model names across all approaches
        all_models = set()
        for approach_data in model_token_data.values():
            all_models.update(approach_data.keys())
        
        # Create sorted list of model names - put the most commonly used models first
        model_count = {model: 0 for model in all_models}
        for approach_data in model_token_data.values():
            for model in approach_data:
                model_count[model] += 1
        sorted_models = sorted(all_models, key=lambda m: (model_count[m], m), reverse=True)
        
        # Prepare data for stacked bar chart - calculate PER QUERY AVERAGES
        model_usage_data = []
        for approach in approach_names:
            if approach in model_token_data:
                # Get number of questions for this approach
                num_questions = approaches[approach]["overall"]["total_questions"]
                if num_questions < 1:
                    num_questions = 1  # Avoid division by zero
                
                for model in sorted_models:
                    if model in model_token_data[approach]:
                        # Calculate per query average
                        total_tokens = model_token_data[approach][model].get("tokens", 0)
                        avg_tokens = total_tokens / num_questions
                        
                        purpose = model_token_data[approach][model].get("purpose", "unknown")
                        is_billable = model_token_data[approach][model].get("billable", False)
                        
                        model_usage_data.append({
                            "Approach": approach,
                            "Model": model,
                            "Tokens": avg_tokens,  # Average per query
                            "Purpose": purpose,
                            "Billable": "Yes" if is_billable else "No"
                        })
        
        # Convert to DataFrame
        model_df = pd.DataFrame(model_usage_data)
        
        if not model_df.empty:
            # Create figure for token usage by model
            plt.figure(figsize=(14, 9))
            
            # Use model_df to create stacked bar plot
            ax = sns.barplot(
                x="Approach", 
                y="Tokens", 
                hue="Model", 
                data=model_df,
                palette="deep"
            )
            
            plt.title("Average Token Usage by Model (Per Query)", fontsize=16, pad=20)
            plt.ylabel("Number of Tokens", fontsize=14)
            plt.xlabel("RAG Approach", fontsize=14)
            plt.xticks(rotation=30, ha="right")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "token_usage_by_model.png"), dpi=300)
            plt.close()
            
            # Create a breakdown by purpose (embedding, generation, query processing) - per query
            plt.figure(figsize=(14, 9))
            
            # Group by purpose
            purpose_df = model_df.copy()
            # Filter out unknown purpose
            purpose_df = purpose_df[purpose_df["Purpose"] != "unknown"]
            purpose_df = purpose_df.groupby(["Approach", "Purpose"]).sum().reset_index()
            
            # Create stacked bar chart
            ax = sns.barplot(
                x="Approach", 
                y="Tokens", 
                hue="Purpose", 
                data=purpose_df,
                palette="viridis"
            )
            
            plt.title("Average Token Usage by Purpose (Per Query)", fontsize=16, pad=20)
            plt.ylabel("Number of Tokens", fontsize=14)
            plt.xlabel("RAG Approach", fontsize=14)
            plt.xticks(rotation=30, ha="right")
            plt.legend(title="Purpose", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "token_usage_by_purpose.png"), dpi=300)
            plt.close()
        else:
            print("Warning: Model token usage data found but couldn't create visualizations")
    else:
        # Just show total token usage if breakdown not available
        plt.figure(figsize=(12, 7))
        
        # Calculate per query averages
        avg_token_usage = []
        for approach in approach_names:
            total_tokens = metrics["Token Usage"][approach_names.index(approach)]
            num_questions = approaches[approach]["overall"]["total_questions"]
            if num_questions < 1:
                num_questions = 1  # Avoid division by zero
            avg_token_usage.append(total_tokens / num_questions)
        
        ax = sns.barplot(x=approach_names, y=avg_token_usage, palette="deep")
        
        plt.title("Average Token Usage Per Query", fontsize=16, pad=20)
        plt.ylabel("Number of Tokens", fontsize=14)
        plt.xlabel("RAG Approach", fontsize=14)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_usage_total.png"), dpi=300)
        plt.close()
    
    # 3. Quality metrics breakdown (with improved legend placement)
    quality_data = []
    for i, approach in enumerate(approach_names):
        for metric in ["Relevance", "Correctness", "Completeness"]:
            quality_data.append({
                "Approach": approach,
                "Metric": metric,
                "Score": metrics[metric][i]
            })
    
    quality_df = pd.DataFrame(quality_data)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Approach", y="Score", hue="Metric", data=quality_df, palette="deep")
    
    plt.title("Quality Metrics Breakdown", fontsize=16, pad=20)
    plt.ylim(0, 10)
    plt.ylabel("Score (0-10)", fontsize=14)
    plt.xlabel("RAG Approach", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    
    # Move legend outside of plot area
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_metrics_breakdown.png"), dpi=300)
    plt.close()
    
    # 4. Cost Analysis and Efficiency Visualization
    # Define cost per 1K tokens for each model
    model_costs = {
        "gpt-4o": 0.01,  # $10 per million tokens combined
        "text-embedding-3-small": 0.0002,  # $0.20 per million tokens
        "text-embedding-3-large": 0.0013,  # $1.30 per million tokens
        "gpt-4": 0.03,  # $30 per million tokens
        "gpt-3.5-turbo": 0.001,  # $1 per million tokens
        # Add other models as needed
    }
    
    # Default cost for unknown models
    default_cost = 0.01  # Use a reasonably high default
    
    # Calculate cost per approach
    approach_costs = {}
    for approach in approach_names:
        total_cost = 0
        if approach in model_token_data:
            for model, data in model_token_data[approach].items():
                tokens = data.get("tokens", 0)
                is_billable = data.get("billable", False)
                
                if is_billable:
                    # Find closest matching model name in cost dictionary
                    model_cost = next((cost for model_name, cost in model_costs.items() 
                                     if model_name.lower() in model.lower()), default_cost)
                    model_cost_per_query = (tokens / 1000) * model_cost
                    total_cost += model_cost_per_query
        
        approach_costs[approach] = total_cost
    
    # Create cost per query visualization
    if approach_costs:
        cost_data = []
        for approach, cost in approach_costs.items():
            avg_cost = cost / (approaches[approach]["overall"]["total_questions"] or 1)
            score = approaches[approach]["overall"]["avg_overall_score"]
            
            cost_data.append({
                "Approach": approach,
                "Cost per Query": avg_cost,
                "Score": score
            })
        
        cost_df = pd.DataFrame(cost_data)
        
        # Cost per query
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x="Approach", y="Cost per Query", data=cost_df, palette="deep")
        
        plt.title("Estimated Cost per Query", fontsize=16, pad=20)
        plt.ylabel("Cost ($)", fontsize=14)
        plt.xlabel("RAG Approach", fontsize=14)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add explanation of cost calculation
        plt.figtext(0.5, 0.01, 
                    "Cost calculation based on standard OpenAI pricing.\n" +
                    "Actual costs may vary based on specific API rates and usage patterns.",
                    ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cost_per_query.png"), dpi=300)
        plt.close()

    # New plots: Quality vs Cost scatter and Latency vs Quality bubble with cost size

    # Compute cost per query list for later plots using approach_costs (already computed above) if available
    cost_per_query_list = []
    for approach in approach_names:
        if approach in approach_costs and approaches[approach]["overall"]["total_questions"] > 0:
            cost_per_query_list.append(approach_costs[approach] / approaches[approach]["overall"]["total_questions"])
        else:
            # Fallback to 0 if cost not available
            cost_per_query_list.append(0)

    # Scatter: Quality vs Cost
    plt.figure(figsize=(12, 8))
    for i, approach in enumerate(approach_names):
        plt.scatter(cost_per_query_list[i], metrics["Overall Score"][i], s=200, label=approach)
        plt.text(cost_per_query_list[i], metrics["Overall Score"][i]+0.02, approach, fontsize=12, ha='center')

    plt.title("Quality vs Cost per Query", fontsize=16, pad=20)
    plt.xlabel("Estimated Cost per Query ($)", fontsize=14)
    plt.ylabel("Average Overall Score (0–10)", fontsize=14)
    plt.xscale('log')
    plt.ylim(0, 10)
    plt.grid(True, which="both", ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_vs_cost.png"), dpi=300)
    plt.close()

    # Bubble chart: Latency vs Quality with bubble area proportional to cost
    plt.figure(figsize=(12, 8))
    max_cost = max(cost_per_query_list) if cost_per_query_list else 1
    for i, approach in enumerate(approach_names):
        bubble_size = 1500 * (cost_per_query_list[i] / max_cost + 0.1)  # scale for visibility
        plt.scatter(metrics["Response Time (s)"][i], metrics["Overall Score"][i], s=bubble_size, alpha=0.7, label=approach)
        plt.text(metrics["Response Time (s)"][i], metrics["Overall Score"][i]+0.02, approach, fontsize=12, ha='center')

    plt.title("Latency vs Quality (bubble size = cost)", fontsize=16, pad=20)
    plt.xlabel("Average Response Time (s)", fontsize=14)
    plt.ylabel("Average Overall Score (0–10)", fontsize=14)
    plt.ylim(0, 10)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_quality.png"), dpi=300)
    plt.close()

    # Token usage by purpose – percent
    if model_token_data:
        purpose_percent_rows = []
        for approach in approach_names:
            # gather per approach billable/non-billable tokens by purpose
            purpose_totals = {}
            total_tokens = 0
            if approach in model_token_data:
                for m, d in model_token_data[approach].items():
                    purpose = d.get("purpose", "unknown")
                    tok = d.get("tokens", 0)
                    purpose_totals[purpose] = purpose_totals.get(purpose, 0) + tok
                    total_tokens += tok
            if total_tokens == 0:
                continue
            for purpose, tok in purpose_totals.items():
                purpose_percent_rows.append({
                    "Approach": approach,
                    "Purpose": purpose,
                    "Percent": (tok/total_tokens)*100
                })
        if purpose_percent_rows:
            purpose_percent_df = pd.DataFrame(purpose_percent_rows)
            plt.figure(figsize=(14, 9))
            ax = sns.barplot(x="Approach", y="Percent", hue="Purpose", data=purpose_percent_df, palette="deep")
            plt.title("Token Usage Share by Purpose", fontsize=16, pad=20)
            plt.ylabel("Percent (%)", fontsize=14)
            plt.xlabel("RAG Approach", fontsize=14)
            plt.xticks(rotation=30, ha="right")
            plt.legend(title="Purpose", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "token_usage_percent_purpose.png"), dpi=300)
            plt.close()

def create_cross_comparison_analysis(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create a cross-comparison analysis showing where each approach excels or fails.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save visualizations
    """
    # Extract approaches and their results
    approaches = results.get("approaches", {})
    approach_names = list(approaches.keys())
    
    if len(approach_names) < 2:
        print("Need at least two approaches for cross-comparison analysis")
        return
    
    # Create directory for cross-comparison results
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    # Load detailed results for each approach to get per-question scores
    result_files = {}
    for approach in approach_names:
        # Find the results file for this approach
        approach_file = os.path.join(
            os.path.dirname(output_dir),
            f"{approach.lower().replace(' ', '_')}_{results.get('timestamp', '')}.json"
        )
        
        if os.path.exists(approach_file):
            with open(approach_file, 'r', encoding='utf-8') as f:
                result_files[approach] = json.load(f)
    
    if not result_files:
        print("Could not find detailed results files for cross-comparison analysis")
        return
    
    # Gather all unique question IDs and their properties across approaches
    question_metadata = {}
    for approach, data in result_files.items():
        if "results" in data:
            for result in data["results"]:
                if "question_id" in result:
                    qid = result["question_id"]
                    
                    if qid not in question_metadata:
                        question_metadata[qid] = {
                            "id": qid,
                            "question": result.get("question", ""),
                            "category": result.get("category", "unknown"),
                            "complexity": result.get("complexity", "unknown"),
                            "manual_type": result.get("manual_type", "unknown"),
                            "language": result.get("language", "unknown"),
                            "scores": {app: {} for app in approach_names}
                        }
                    
                    # Add scores for this approach
                    if "evaluation" in result:
                        question_metadata[qid]["scores"][approach] = {
                            metric: score for metric, score in result["evaluation"].items()
                            if metric not in ["EXPLANATION", "HARMFULNESS"]
                        }
    
    # Create a DataFrame for analysis
    rows = []
    for qid, metadata in question_metadata.items():
        # Only include questions that have been evaluated by all approaches
        if all(bool(metadata["scores"][app]) for app in approach_names):
            base_row = {
                "question_id": qid,
                "question": metadata["question"],
                "category": metadata["category"],
                "complexity": metadata["complexity"],
                "manual_type": metadata["manual_type"],
                "language": metadata["language"]
            }
            
            # Find the best and worst approach for each question
            for metric in ["OVERALL_SCORE", "RELEVANCE", "CORRECTNESS", "COMPLETENESS", "CONCISENESS"]:
                scores = {app: metadata["scores"][app].get(metric, 0) for app in approach_names}
                max_score = max(scores.values())
                min_score = min(scores.values())
                
                # Add all scores to the row
                for app in approach_names:
                    base_row[f"{app}_{metric}"] = scores.get(app, 0)
                
                # Add best and worst approach
                best_apps = [app for app, score in scores.items() if score == max_score]
                worst_apps = [app for app, score in scores.items() if score == min_score]
                
                base_row[f"best_{metric}"] = best_apps[0] if best_apps else "none"
                base_row[f"worst_{metric}"] = worst_apps[0] if worst_apps else "none"
                
                # Add score differences between best and worst
                base_row[f"diff_{metric}"] = max_score - min_score
            
            rows.append(base_row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No common questions evaluated by all approaches")
        return
    
    # Save the DataFrame for reference
    df.to_csv(os.path.join(output_dir, "comparisons", "question_level_comparison.csv"), index=False)
    
    # 1. Create "win count" visualization - how many times each approach had the best score
    win_counts = {app: {} for app in approach_names}
    for metric in ["OVERALL_SCORE", "RELEVANCE", "CORRECTNESS", "COMPLETENESS", "CONCISENESS"]:
        for app in approach_names:
            win_counts[app][metric] = sum(df[f"best_{metric}"] == app)
    
    # Convert to DataFrame for plotting
    win_df = pd.DataFrame.from_dict(
        {(app, metric): win_counts[app][metric] for app in approach_names for metric in 
         ["OVERALL_SCORE", "RELEVANCE", "CORRECTNESS", "COMPLETENESS", "CONCISENESS"]},
        orient='index'
    ).reset_index()
    win_df[['Approach', 'Metric']] = pd.DataFrame(win_df['index'].tolist(), index=win_df.index)
    win_df = win_df.rename(columns={0: 'Wins'}).drop('index', axis=1)
    
    # Plot wins by metric and approach
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='Metric', y='Wins', hue='Approach', data=win_df, palette='deep')
    
    # Remove data labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%d')
    
    plt.title("Number of Questions Where Each Approach Performed Best", fontsize=16, pad=20)
    plt.ylabel("Number of Wins", fontsize=14)
    plt.xlabel("Evaluation Metric", fontsize=14)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "win_count_by_metric.png"), dpi=300)
    plt.close()
    
    # 2. Create category-specific performance comparison
    # Group data by category and calculate mean scores
    category_performance = df.groupby("category")[
        [f"{app}_OVERALL_SCORE" for app in approach_names]
    ].mean().reset_index()
    
    # Reshape for better plotting
    category_perf_melted = pd.melt(
        category_performance,
        id_vars=["category"],
        value_vars=[f"{app}_OVERALL_SCORE" for app in approach_names],
        var_name="approach",
        value_name="score"
    )
    
    # Clean up approach names
    category_perf_melted["approach"] = category_perf_melted["approach"].apply(
        lambda x: x.replace("_OVERALL_SCORE", "")
    )
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        x="category", 
        y="score", 
        hue="approach", 
        data=category_perf_melted,
        palette="deep"
    )
    
    # Remove data labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.title("Performance Comparison by Question Category", fontsize=16, pad=20)
    plt.ylabel("Average Overall Score", fontsize=14)
    plt.xlabel("Question Category", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "category_performance.png"), dpi=300)
    plt.close()
    
    # 3. Create complexity-specific performance comparison
    complexity_performance = df.groupby("complexity")[
        [f"{app}_OVERALL_SCORE" for app in approach_names]
    ].mean().reset_index()
    
    # Reshape for better plotting
    complexity_perf_melted = pd.melt(
        complexity_performance,
        id_vars=["complexity"],
        value_vars=[f"{app}_OVERALL_SCORE" for app in approach_names],
        var_name="approach",
        value_name="score"
    )
    
    # Clean up approach names
    complexity_perf_melted["approach"] = complexity_perf_melted["approach"].apply(
        lambda x: x.replace("_OVERALL_SCORE", "")
    )
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="complexity", 
        y="score", 
        hue="approach", 
        data=complexity_perf_melted,
        palette="deep"
    )
    
    # Remove data labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.title("Performance Comparison by Question Complexity", fontsize=16, pad=20)
    plt.ylabel("Average Overall Score", fontsize=14)
    plt.xlabel("Question Complexity", fontsize=14)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "complexity_performance.png"), dpi=300)
    plt.close()
    
    # 4. Create language-specific performance comparison
    language_performance = df.groupby("language")[
        [f"{app}_OVERALL_SCORE" for app in approach_names]
    ].mean().reset_index()
    
    # Reshape for better plotting
    language_perf_melted = pd.melt(
        language_performance,
        id_vars=["language"],
        value_vars=[f"{app}_OVERALL_SCORE" for app in approach_names],
        var_name="approach",
        value_name="score"
    )
    
    # Clean up approach names
    language_perf_melted["approach"] = language_perf_melted["approach"].apply(
        lambda x: x.replace("_OVERALL_SCORE", "")
    )
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="language", 
        y="score", 
        hue="approach", 
        data=language_perf_melted,
        palette="deep"
    )
    
    # Remove data labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.title("Performance Comparison by Question Language", fontsize=16, pad=20)
    plt.ylabel("Average Overall Score", fontsize=14)
    plt.xlabel("Language", fontsize=14)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "language_performance.png"), dpi=300)
    plt.close()
    
    # 5. Create a "strength and weakness" summary for each approach
    strengths_weaknesses = {}
    
    for approach in approach_names:
        # Calculate advantage/disadvantage for each category
        category_advantage = {}
        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            scores = {app: category_df[f"{app}_OVERALL_SCORE"].mean() for app in approach_names}
            approach_score = scores[approach]
            other_scores = [s for a, s in scores.items() if a != approach]
            advantage = approach_score - np.mean(other_scores)
            category_advantage[category] = advantage
        
        # Find top 3 strengths and weaknesses
        sorted_advantages = sorted(category_advantage.items(), key=lambda x: x[1], reverse=True)
        strengths = sorted_advantages[:3]
        weaknesses = sorted_advantages[-3:]
        
        strengths_weaknesses[approach] = {
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    
    # Save strengths and weaknesses analysis
    with open(os.path.join(output_dir, "comparisons", "strengths_weaknesses.json"), 'w', encoding='utf-8') as f:
        json.dump(strengths_weaknesses, f, indent=2)
    
    print(f"Cross-comparison analysis complete. Results saved to {os.path.join(output_dir, 'comparisons')}")
    
    # Create a summary visualization of the strengths and weaknesses
    plt.figure(figsize=(16, 12))
    
    for i, approach in enumerate(approach_names):
        sw = strengths_weaknesses[approach]
        strengths = sw["strengths"]
        weaknesses = sw["weaknesses"]
        
        # Draw the approach name
        plt.text(0.5, 1 - (i * 0.25 + 0.05), approach, 
                ha='center', va='center', fontsize=16, fontweight='bold', 
                bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle="round,pad=0.5"))
        
        # Draw strengths
        plt.text(0.25, 1 - (i * 0.25 + 0.1), "Strengths:", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='darkgreen')
        
        for j, (category, advantage) in enumerate(strengths):
            plt.text(0.25, 1 - (i * 0.25 + 0.15 + j * 0.03), 
                    f"{category}: +{advantage:.2f}", 
                    ha='center', va='center', fontsize=12, color='green')
        
        # Draw weaknesses
        plt.text(0.75, 1 - (i * 0.25 + 0.1), "Weaknesses:", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='darkred')
        
        for j, (category, advantage) in enumerate(weaknesses):
            plt.text(0.75, 1 - (i * 0.25 + 0.15 + j * 0.03), 
                    f"{category}: {advantage:.2f}", 
                    ha='center', va='center', fontsize=12, color='red')
    
    plt.axis('off')
    plt.title("Approach Strengths and Weaknesses by Question Category", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparisons", "strengths_weaknesses_summary.png"), dpi=300)
    plt.close()
    
    print(f"Strengths and weaknesses summary visualization created.")

def main():
    """Main function to generate all visualizations."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(base_dir, "evaluation_results")
    data_dir = os.path.join(base_dir, "evaluation_data")
    output_dir = os.path.join(results_dir, "visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        results = load_evaluation_results(results_dir)
        questions = load_test_questions(data_dir)
        
        # Generate only the needed visualizations
        visualize_performance_metrics(results, output_dir)
        
        # Add cross-comparison analysis
        create_cross_comparison_analysis(results, output_dir)
        
        print(f"Visualizations generated successfully in {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 