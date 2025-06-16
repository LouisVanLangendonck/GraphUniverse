"""
Analysis and visualization of transductive experiment results.
Updated to match the inductive experiment analysis strategies.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

def analyze_transductive_results(results: List[Dict[str, Any]], output_dir: str) -> Optional[pd.DataFrame]:
    """
    Analyze transductive experiment results.
    
    Args:
        results: List of experiment result dictionaries
        output_dir: Directory to save analysis results
        
    Returns:
        DataFrame with analysis results
    """
    if not results:
        print("No results to analyze - all experiments failed.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save full results first
    df.to_csv(os.path.join(analysis_dir, "full_results.csv"), index=False)
    
    # Extract model results for each task
    expanded_results = []
    
    for _, row in df.iterrows():
        base_info = {k: v for k, v in row.items() if not k.startswith('model_results')}
        
        if 'model_results' in row and isinstance(row['model_results'], dict):
            for task, task_results in row['model_results'].items():
                if isinstance(task_results, dict):
                    for model, model_results in task_results.items():
                        result_row = base_info.copy()
                        result_row['task'] = task
                        result_row['model'] = model
                        
                        # Extract test metrics
                        if 'test_metrics' in model_results and isinstance(model_results['test_metrics'], dict):
                            for metric, value in model_results['test_metrics'].items():
                                result_row[f'test_{metric}'] = value
                        
                        # Extract other info
                        result_row['train_time'] = model_results.get('train_time', 0.0)
                        result_row['success'] = bool(model_results.get('test_metrics', {}))
                        result_row['error'] = model_results.get('error', None)
                        
                        expanded_results.append(result_row)
    
    if not expanded_results:
        print("No valid model results found.")
        return df
    
    expanded_df = pd.DataFrame(expanded_results)
    
    # Save expanded results
    expanded_df.to_csv(os.path.join(analysis_dir, "expanded_results.csv"), index=False)
    
    # Create task-specific analyses
    for task in expanded_df['task'].unique():
        task_df = expanded_df[expanded_df['task'] == task]
        task_df.to_csv(os.path.join(analysis_dir, f"{task}_results.csv"), index=False)
        
        # Create model performance summary for this task
        create_task_performance_summary(task_df, task, analysis_dir)
    
    # Create overall performance summary
    create_overall_performance_summary(expanded_df, analysis_dir)
    
    # Create parameter analysis if applicable
    create_parameter_analysis(expanded_df, analysis_dir)
    
    print(f"\nAnalysis complete. Results saved to {analysis_dir}")
    return expanded_df

def create_task_performance_summary(
    task_df: pd.DataFrame, 
    task: str, 
    analysis_dir: str
) -> None:
    """Create performance summary for a specific task."""
    
    summary_lines = []
    summary_lines.append(f"TASK: {task.upper()}")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    # Determine if this is regression or classification
    is_regression = 'r2' in task_df.columns or 'mse' in [col.replace('test_', '') for col in task_df.columns if col.startswith('test_')]
    
    if is_regression:
        primary_metrics = ['test_r2', 'test_mse', 'test_mae', 'test_rmse']
        primary_metric = 'test_r2'
    else:
        primary_metrics = ['test_accuracy', 'test_f1_macro', 'test_precision_macro', 'test_recall_macro']
        primary_metric = 'test_f1_macro'
    
    # Model performance comparison
    summary_lines.append("MODEL PERFORMANCE:")
    summary_lines.append("-" * 30)
    
    best_score = float('-inf') if is_regression else 0.0
    best_model = None
    
    for model in task_df['model'].unique():
        model_df = task_df[task_df['model'] == model]
        successful = model_df['success'].sum()
        total = len(model_df)
        
        summary_lines.append(f"\n{model.upper()}:")
        summary_lines.append(f"  Success rate: {successful}/{total} ({successful/total:.1%})")
        
        if successful > 0:
            successful_df = model_df[model_df['success']]
            
            for metric in primary_metrics:
                if metric in successful_df.columns:
                    mean_val = successful_df[metric].mean()
                    std_val = successful_df[metric].std()
                    summary_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    
                    if metric == primary_metric:
                        if (is_regression and mean_val > best_score) or (not is_regression and mean_val > best_score):
                            best_score = mean_val
                            best_model = model.upper()
            
            # Training time
            mean_time = successful_df['train_time'].mean()
            summary_lines.append(f"  Avg training time: {mean_time:.2f}s")
        else:
            summary_lines.append("  No successful runs")
    
    if best_model:
        summary_lines.append(f"\nBEST MODEL: {best_model} ({primary_metric}: {best_score:.4f})")
    
    summary_lines.append("")
    
    # Save task summary
    task_summary_file = os.path.join(analysis_dir, f"{task}_summary.txt")
    with open(task_summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

def create_overall_performance_summary(
    expanded_df: pd.DataFrame, 
    analysis_dir: str
) -> None:
    """Create overall performance summary across all tasks."""
    
    summary_lines = []
    summary_lines.append("OVERALL PERFORMANCE SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    # Overall statistics
    total_runs = len(expanded_df)
    successful_runs = expanded_df['success'].sum()
    success_rate = successful_runs / total_runs if total_runs > 0 else 0
    
    summary_lines.append(f"Total model runs: {total_runs}")
    summary_lines.append(f"Successful runs: {successful_runs}")
    summary_lines.append(f"Overall success rate: {success_rate:.1%}")
    summary_lines.append("")
    
    # Performance by model type
    summary_lines.append("PERFORMANCE BY MODEL:")
    summary_lines.append("-" * 30)
    
    for model in expanded_df['model'].unique():
        model_df = expanded_df[expanded_df['model'] == model]
        model_success = model_df['success'].sum()
        model_total = len(model_df)
        model_success_rate = model_success / model_total if model_total > 0 else 0
        
        summary_lines.append(f"\n{model.upper()}:")
        summary_lines.append(f"  Runs: {model_total}")
        summary_lines.append(f"  Success rate: {model_success_rate:.1%}")
        
        if model_success > 0:
            successful_model_df = model_df[model_df['success']]
            avg_time = successful_model_df['train_time'].mean()
            summary_lines.append(f"  Avg training time: {avg_time:.2f}s")
    
    summary_lines.append("")
    
    # Performance by task
    summary_lines.append("PERFORMANCE BY TASK:")
    summary_lines.append("-" * 30)
    
    for task in expanded_df['task'].unique():
        task_df = expanded_df[expanded_df['task'] == task]
        task_success = task_df['success'].sum()
        task_total = len(task_df)
        task_success_rate = task_success / task_total if task_total > 0 else 0
        
        summary_lines.append(f"\n{task.upper()}:")
        summary_lines.append(f"  Runs: {task_total}")
        summary_lines.append(f"  Success rate: {task_success_rate:.1%}")
    
    # Save overall summary
    overall_summary_file = os.path.join(analysis_dir, "overall_summary.txt")
    with open(overall_summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

def create_parameter_analysis(
    expanded_df: pd.DataFrame, 
    analysis_dir: str
) -> None:
    """Create parameter effect analysis."""
    
    # Find parameter columns (exclude standard columns)
    standard_cols = {'task', 'model', 'success', 'error', 'train_time'}
    test_metric_cols = {col for col in expanded_df.columns if col.startswith('test_')}
    
    param_cols = []
    for col in expanded_df.columns:
        if col not in standard_cols and col not in test_metric_cols:
            if expanded_df[col].dtype in ['float64', 'int64'] and len(expanded_df[col].unique()) > 1:
                param_cols.append(col)
    
    if not param_cols:
        print("No parameter columns found for analysis.")
        return
    
    # Find performance metric columns
    performance_cols = [col for col in test_metric_cols if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2'])]
    
    if not performance_cols:
        print("No performance metric columns found for parameter analysis.")
        return
    
    # Calculate correlations
    correlations = {}
    for perf_col in performance_cols:
        correlations[perf_col] = {}
        for param_col in param_cols:
            # Filter out rows where the performance metric is NaN
            valid_data = expanded_df[expanded_df[perf_col].notna()]
            if len(valid_data) > 3:  # Need at least 4 points for correlation
                try:
                    corr = valid_data[param_col].corr(valid_data[perf_col])
                    if not np.isnan(corr):
                        correlations[perf_col][param_col] = corr
                except Exception:
                    continue
    
    # Save correlation analysis
    correlation_analysis = {
        'parameter_performance_correlations': correlations,
        'parameter_columns_analyzed': param_cols,
        'performance_columns_analyzed': performance_cols
    }
    
    correlation_file = os.path.join(analysis_dir, "parameter_correlations.json")
    with open(correlation_file, 'w') as f:
        json.dump(correlation_analysis, f, indent=2)
    
    print(f"Parameter correlation analysis saved to {correlation_file}")

def create_analysis_plots(results_dir: str, output_dir: Optional[str] = None) -> None:
    """Create analysis plots from transductive experiment results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib and seaborn required for plotting. Skipping plots.")
        return
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    analysis_dir = os.path.join(results_dir, "analysis")
    expanded_csv_path = os.path.join(analysis_dir, "expanded_results.csv")
    
    if not os.path.exists(expanded_csv_path):
        print(f"No expanded results CSV found at {expanded_csv_path}")
        return
    
    df = pd.read_csv(expanded_csv_path)
    
    # Filter successful runs only
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        print("No successful runs found for plotting.")
        return
    
    plt.style.use('default')
    
    # 1. Model performance comparison by task
    if 'task' in successful_df.columns and 'model' in successful_df.columns:
        tasks = successful_df['task'].unique()
        
        for task in tasks:
            task_df = successful_df[successful_df['task'] == task]
            
            # Determine primary metric
            if 'test_r2' in task_df.columns:
                metric_col = 'test_r2'
                metric_name = 'R² Score'
            elif 'test_f1_macro' in task_df.columns:
                metric_col = 'test_f1_macro'
                metric_name = 'F1 Macro Score'
            elif 'test_accuracy' in task_df.columns:
                metric_col = 'test_accuracy'
                metric_name = 'Accuracy'
            else:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Box plot of performance by model
            models = task_df['model'].unique()
            model_data = []
            model_labels = []
            
            for model in models:
                model_scores = task_df[task_df['model'] == model][metric_col].dropna()
                if len(model_scores) > 0:
                    model_data.append(model_scores)
                    model_labels.append(model.upper())
            
            if model_data:
                plt.boxplot(model_data, labels=model_labels)
                plt.title(f'{metric_name} by Model - Task: {task.upper()}')
                plt.xlabel('Model')
                plt.ylabel(metric_name)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f'model_performance_{task}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # 2. Training time comparison
    if 'train_time' in successful_df.columns and 'model' in successful_df.columns:
        plt.figure(figsize=(10, 6))
        
        models = successful_df['model'].unique()
        time_data = []
        time_labels = []
        
        for model in models:
            model_times = successful_df[successful_df['model'] == model]['train_time'].dropna()
            if len(model_times) > 0:
                time_data.append(model_times)
                time_labels.append(model.upper())
        
        if time_data:
            plt.boxplot(time_data, labels=time_labels)
            plt.title('Training Time by Model')
            plt.xlabel('Model')
            plt.ylabel('Training Time (seconds)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Parameter effect plots (if parameter columns exist)
    param_cols = []
    for col in successful_df.columns:
        if (col not in {'task', 'model', 'success', 'error', 'train_time'} and 
            not col.startswith('test_') and
            successful_df[col].dtype in ['float64', 'int64'] and 
            len(successful_df[col].unique()) > 1):
            param_cols.append(col)
    
    performance_cols = [col for col in successful_df.columns 
                       if col.startswith('test_') and 
                       any(metric in col for metric in ['accuracy', 'f1_macro', 'r2'])]
    
    # Create scatter plots for parameter effects (limit to first few to avoid too many plots)
    for param in param_cols[:3]:  # Limit to 3 parameters
        for perf_col in performance_cols[:2]:  # Limit to 2 performance metrics
            plt.figure(figsize=(10, 6))
            
            # Color by model if multiple models
            if len(successful_df['model'].unique()) > 1:
                for model in successful_df['model'].unique():
                    model_data = successful_df[successful_df['model'] == model]
                    if len(model_data) > 0:
                        plt.scatter(model_data[param], model_data[perf_col], 
                                  alpha=0.7, label=model.upper())
                plt.legend()
            else:
                plt.scatter(successful_df[param], successful_df[perf_col], alpha=0.7)
            
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel(perf_col.replace('test_', '').replace('_', ' ').title())
            plt.title(f'{perf_col.replace("test_", "").replace("_", " ").title()} vs {param.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{param}_vs_{perf_col}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Analysis plots saved to: {output_dir}")

def generate_experiment_report(
    results_dir: str,
    report_path: Optional[str] = None
) -> str:
    """Generate comprehensive experiment report."""
    
    if report_path is None:
        report_path = os.path.join(results_dir, "experiment_report.txt")
    
    # Load analysis files
    analysis_dir = os.path.join(results_dir, "analysis")
    
    report_lines = []
    report_lines.append("TRANSDUCTIVE GRAPH LEARNING EXPERIMENT REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Load overall summary if available
    overall_summary_path = os.path.join(analysis_dir, "overall_summary.txt")
    if os.path.exists(overall_summary_path):
        with open(overall_summary_path, 'r') as f:
            report_lines.append(f.read())
        report_lines.append("")
    
    # Load task summaries
    for task_file in os.listdir(analysis_dir):
        if task_file.endswith("_summary.txt") and task_file != "overall_summary.txt":
            task_summary_path = os.path.join(analysis_dir, task_file)
            with open(task_summary_path, 'r') as f:
                report_lines.append(f.read())
            report_lines.append("")
    
    # Add parameter correlation analysis if available
    correlation_path = os.path.join(analysis_dir, "parameter_correlations.json")
    if os.path.exists(correlation_path):
        with open(correlation_path, 'r') as f:
            correlations = json.load(f)
        
        report_lines.append("PARAMETER CORRELATIONS")
        report_lines.append("-" * 40)
        
        for perf_metric, param_corrs in correlations.get('parameter_performance_correlations', {}).items():
            if param_corrs:
                report_lines.append(f"\n{perf_metric.upper()}:")
                
                # Sort by absolute correlation value
                sorted_corrs = sorted(param_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for param, corr in sorted_corrs[:5]:  # Top 5 correlations
                    report_lines.append(f"  {param}: {corr:.3f}")
        
        report_lines.append("")
    
    # Save report
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Experiment report saved to: {report_path}")
    return "\n".join(report_lines)

def compare_model_performance(
    results_df: pd.DataFrame,
    task: str,
    metric: str = 'auto'
) -> Dict[str, Any]:
    """
    Compare model performance for a specific task.
    
    Args:
        results_df: DataFrame with experiment results
        task: Task name to analyze
        metric: Metric to compare ('auto' for automatic selection)
        
    Returns:
        Dictionary with comparison results
    """
    task_df = results_df[results_df['task'] == task]
    successful_df = task_df[task_df['success'] == True]
    
    if len(successful_df) == 0:
        return {'error': 'No successful runs found for this task'}
    
    # Auto-select metric if needed
    if metric == 'auto':
        if 'test_r2' in successful_df.columns:
            metric = 'test_r2'
        elif 'test_f1_macro' in successful_df.columns:
            metric = 'test_f1_macro'
        elif 'test_accuracy' in successful_df.columns:
            metric = 'test_accuracy'
        else:
            return {'error': 'No suitable metric found'}
    
    if metric not in successful_df.columns:
        return {'error': f'Metric {metric} not found in results'}
    
    # Calculate statistics for each model
    model_stats = {}
    for model in successful_df['model'].unique():
        model_data = successful_df[successful_df['model'] == model][metric].dropna()
        
        if len(model_data) > 0:
            model_stats[model] = {
                'mean': float(model_data.mean()),
                'std': float(model_data.std()),
                'min': float(model_data.min()),
                'max': float(model_data.max()),
                'count': len(model_data)
            }
    
    # Find best model
    if model_stats:
        best_model = max(model_stats.keys(), key=lambda m: model_stats[m]['mean'])
        
        return {
            'task': task,
            'metric': metric,
            'model_statistics': model_stats,
            'best_model': best_model,
            'best_score': model_stats[best_model]['mean']
        }
    
    return {'error': 'No valid model statistics calculated'}