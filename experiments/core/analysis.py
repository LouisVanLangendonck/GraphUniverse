"""
Analysis and visualization of experiment results.

This module provides functions for analyzing and visualizing the results of MMSB graph learning experiments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime
from utils.parameter_analysis import analyze_graph_parameters

def analyze_results(results: List[Dict[str, Any]], args) -> Optional[pd.DataFrame]:
    """Analyze experiment results with a simplified approach."""
    if not results:  # Handle case where all experiments failed
        print("No results to analyze - all experiments failed.")
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create analysis directory
    analysis_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save full results first
    df.to_csv(os.path.join(analysis_dir, "full_results.csv"), index=False)
    
    # Separate analysis by task
    tasks = ["community", "k_hop_community_counts"]
    models = ["gcn", "sage", "mlp", "rf"]
    
    for task in tasks:
        # Create task-specific dataframe
        task_columns = [col for col in df.columns if task in col or not any(m in col for m in ["community", "k_hop"])]
        task_df = df[task_columns]
        
        # Save task-specific dataframe
        task_df.to_csv(os.path.join(analysis_dir, f"{task}_results.csv"), index=False)
        
        # For each varied parameter, create an analysis
        for param in args.vary:
            if param not in df.columns:
                print(f"Warning: Parameter {param} not found in results")
                continue
                
            # Calculate aggregations by the parameter
            model_metrics = {}
            for model in models:
                # Classification metrics
                if task == "community":
                    for metric in ["test_acc", "test_f1", "test_precision", "test_recall"]:
                        col = f"{task}-{model}-{metric}"
                        if col in df.columns:
                            model_metrics[f"{model}_{metric}"] = ['mean', 'std']
                # Regression metrics
                else:
                    for metric in ["test_mse", "test_rmse", "test_mae", "test_r2"]:
                        col = f"{task}-{model}-{metric}"
                        if col in df.columns:
                            model_metrics[f"{model}_{metric}"] = ['mean', 'std']
                            
            # Add training time
            for model in models:
                col = f"{task}-{model}-train_time"
                if col in df.columns:
                    model_metrics[f"{model}_train_time"] = ['mean', 'std']
            
            # Graph metrics to include
            graph_metrics = {}
            for metric in ["density", "avg_degree", "clustering_coefficient", "homophily"]:
                col = f"graph-{metric}"
                if col in df.columns:
                    graph_metrics[col] = ['mean', 'std']
            
            # Create aggregation dictionary
            agg_dict = {**model_metrics, **graph_metrics}
            
            if not agg_dict:  # Skip if no metrics available
                print(f"Warning: No metrics available for parameter {param} in task {task}")
                continue
                
            # Calculate mean performance for each parameter value
            try:
                param_analysis = df.groupby(param).agg(agg_dict).round(4)
                param_analysis.to_csv(os.path.join(analysis_dir, f"{task}_{param}_analysis.csv"))
            except Exception as e:
                print(f"Error analyzing parameter {param} for task {task}: {str(e)}")
    
    # Create summary of successful vs failed experiments
    total_experiments = len(results)
    successful = sum(1 for r in results if any(f"{task}-{model}-test_acc" in r for task in tasks for model in models))
    print(f"\nExperiment Summary:")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_experiments - successful}")
    
    return df

def plot_model_comparison(df: pd.DataFrame, args, analysis_dir: str) -> None:
    """Create comparison plots of model performance across varied parameters."""
    # Get the varied parameters
    varied_params = args.vary
    
    # Get available model metrics
    model_metrics = []
    for model in ['GAT', 'GCN', 'SAGE', 'MLP', 'RandomForest']:
        # Check if model metrics exist in the dataframe
        if f"{model}_metrics_macro_f1" in df.columns:
            model_metrics.append(model)
    
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    # Create a figure for each varied parameter
    for param in varied_params:
        if param not in df.columns:
            continue
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy and F1 for each model
        for model in model_metrics:
            # Plot accuracy
            if f"{model}_accuracy" in df.columns:
                plt.plot(df[param], df[f"{model}_accuracy"], 
                        marker='o', label=f"{model} Accuracy")
            # Plot F1 score
            if f"{model}_metrics_macro_f1" in df.columns:
                plt.plot(df[param], df[f"{model}_metrics_macro_f1"], 
                        marker='s', linestyle='--', label=f"{model} F1")
        
        plt.xlabel(param)
        plt.ylabel('Performance')
        plt.title(f'Model Performance vs {param}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(analysis_dir, f"model_comparison_{param}_{timestamp}.png"))
        plt.close()
        
        # Create a second plot for training time comparison
        plt.figure(figsize=(12, 8))
        for model in model_metrics:
            if f"{model}_train_time" in df.columns:
                plt.plot(df[param], df[f"{model}_train_time"], 
                        marker='o', label=f"{model} Training Time")
        
        plt.xlabel(param)
        plt.ylabel('Training Time (s)')
        plt.title(f'Training Time vs {param}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(analysis_dir, f"training_time_{param}_{timestamp}.png"))
        plt.close()
        
        # Create plots showing the relationship between real graph properties and model performance
        real_graph_properties = [
            'avg_degree',
            'clustering_coefficient',
            'homophily',
            'triangle_density',
            'power_law_exponent',
            'density'
        ]
        
        for prop in real_graph_properties:
            if prop in df.columns:
                plt.figure(figsize=(12, 8))
                for model in model_metrics:
                    if f"{model}_accuracy" in df.columns:
                        plt.scatter(df[prop], df[f"{model}_accuracy"],
                                  alpha=0.5, label=f"{model} Accuracy")
                
                plt.xlabel(prop.replace('_', ' ').title())
                plt.ylabel('Model Accuracy')
                plt.title(f'Model Accuracy vs {prop.replace("_", " ").title()}')
                plt.legend()
                plt.grid(True)
                
                plt.savefig(os.path.join(analysis_dir, f"{prop}_accuracy_{timestamp}.png"))
                plt.close() 