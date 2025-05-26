"""
Multi-experiment runner for inductive learning with parameter sweeps and random sampling.

This module orchestrates multiple inductive experiments with systematic parameter
variations and random sampling from specified ranges.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd

from experiments.inductive.multi_config import MultiInductiveExperimentConfig
from experiments.inductive.experiment import run_inductive_experiment
from experiments.inductive.config import InductiveExperimentConfig
from experiments.inductive.training import get_total_classes_from_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiInductiveExperimentRunner:
    """
    Runner for multiple inductive experiments with parameter sweeps and random sampling.
    """
    
    def __init__(self, config: MultiInductiveExperimentConfig):
        """
        Initialize multi-experiment runner.
        
        Args:
            config: Multi-experiment configuration
        """
        self.config = config
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"{config.experiment_name}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "multi_config.json")
        config.save(config_path)
        
        # Initialize result storage
        self.all_results = []
        self.failed_runs = []
        self.run_metadata = []
        
        print(f"Multi-experiment runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Total runs planned: {config.get_total_runs()}")
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured experiments."""
        print(f"\n{'='*80}")
        print(f"STARTING MULTI-INDUCTIVE EXPERIMENT SUITE")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Get parameter combinations
        sweep_combinations = self.config.get_parameter_combinations()
        total_combinations = len(sweep_combinations)
        total_runs = total_combinations * self.config.n_repetitions
        
        print(f"Parameter sweep combinations: {total_combinations}")
        print(f"Repetitions per combination: {self.config.n_repetitions}")
        print(f"Total runs: {total_runs}")
        
        run_id = 0
        
        # Set up fresh random state for each run
        base_seed = 42  # or get from config
        
        # Progress tracking
        with tqdm(total=total_runs, desc="Running experiments") as pbar:
            
            for combo_idx, sweep_params in enumerate(sweep_combinations):
                print(f"\n{'-'*60}")
                print(f"COMBINATION {combo_idx + 1}/{total_combinations}")
                print(f"Sweep parameters: {sweep_params}")
                print(f"{'-'*60}")
                
                for rep in range(self.config.n_repetitions):
                    print(f"\nRepetition {rep + 1}/{self.config.n_repetitions}")
                    
                    try:
                        # Set unique seed for this specific run
                        run_seed = base_seed + run_id
                        np.random.seed(run_seed)
                        
                        # Sample random parameters for THIS specific run
                        random_params = self.config.sample_random_parameters()
                        print(f"Random parameters: {random_params}")
                        
                        # Process tuple parameters here directly
                        processed_random = self._process_tuple_parameters(random_params)
                        
                        # Create run configuration
                        run_config = self.config.create_run_config(
                            sweep_params=sweep_params,
                            random_params=processed_random,  # Use processed params
                            run_id=run_id
                        )
                        
                        # Set the run's seed in the config
                        run_config.seed = run_seed
                        
                        # Use subfolder for each run
                        run_subdir = os.path.join(self.output_dir, f"run_{run_id:04d}")
                        os.makedirs(run_subdir, exist_ok=True)
                        run_config.output_dir = run_subdir
                        
                        # Run single experiment
                        result = self._run_single_experiment(
                            config=run_config,
                            run_id=run_id,
                            sweep_params=sweep_params,
                            random_params=random_params,  # Store original params for tracking
                            run_subdir=run_subdir
                        )
                        
                        if result is not None:
                            self.all_results.append(result)
                            print(f"✓ Run {run_id} completed successfully")
                        else:
                            print(f"✗ Run {run_id} failed")
                        
                    except Exception as e:
                        error_info = {
                            'run_id': run_id,
                            'sweep_params': sweep_params,
                            'random_params': random_params,
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                        }
                        self.failed_runs.append(error_info)
                        
                        print(f"✗ Run {run_id} failed with error: {str(e)}")
                        
                        if not self.config.continue_on_failure:
                            print("Stopping due to failure (continue_on_failure=False)")
                            break
                    
                    run_id += 1
                    pbar.update(1)
                    
                    # Save intermediate results
                    if run_id % 5 == 0:
                        self._save_intermediate_results()
                
                if not self.config.continue_on_failure and self.failed_runs:
                    break
        
        total_time = time.time() - start_time
        
        # Final save
        final_results = self._save_final_results(total_time)
        
        # Generate summary
        summary = self._generate_summary(total_time)
        print(f"\n{summary}")
        
        return final_results
    
    def _run_single_experiment(
        self,
        config: InductiveExperimentConfig,
        run_id: int,
        sweep_params: Dict[str, float],
        random_params: Dict[str, float],
        run_subdir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single inductive experiment.
        
        Args:
            config: Configuration for this run
            run_id: Unique run identifier
            sweep_params: Systematic parameter values
            random_params: Random parameter values
            run_subdir: Directory to save run files
            
        Returns:
            Experiment results or None if failed
        """
        try:
            # Run the experiment
            experiment_results = run_inductive_experiment(config)
            
            # Extract key metrics from results
            result_summary = self._extract_result_summary(experiment_results)
            
            # Compile complete result record
            complete_result = {
                # Run metadata
                'run_id': run_id,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'config_path': config.output_dir,
                
                # Input parameters
                'sweep_parameters': sweep_params,
                'random_parameters': random_params,
                'all_parameters': {**sweep_params, **random_params},
                
                # Graph family metrics (if available)
                'family_stats': self._extract_family_stats(experiment_results),
                
                # Model results
                'model_results': result_summary,
                
                # Experiment metadata
                'total_time': experiment_results.get('total_time', 0),
                'n_graphs': config.n_graphs,
                'universe_K': config.universe_K,
                'tasks': config.tasks,
                'models_run': self._get_models_run(config)
            }
            
            # Save individual result if configured
            if self.config.save_individual_results:
                if run_subdir is None:
                    run_subdir = self.output_dir
                result_path = os.path.join(run_subdir, f"result.json")
                with open(result_path, 'w') as f:
                    json.dump(self._make_json_serializable(complete_result), f, indent=2)
            
            # Save individual config if configured
            if self.config.save_individual_configs:
                if run_subdir is None:
                    run_subdir = self.output_dir
                config_path = os.path.join(run_subdir, f"config.json")
                config.save(config_path)
            
            return complete_result
            
        except Exception as e:
            logger.error(f"Error in run {run_id}: {str(e)}", exc_info=True)
            return None
    
    def _extract_result_summary(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from experiment results."""
        if 'results' not in experiment_results:
            return {}
        
        summary = {}
        
        for task, task_results in experiment_results['results'].items():
            task_summary = {}
            
            for model, model_results in task_results.items():
                if 'test_metrics' in model_results:
                    test_metrics = model_results['test_metrics']
                    
                    # Extract key metrics based on task type
                    if 'accuracy' in test_metrics:
                        # Classification metrics
                        task_summary[f"{model}_accuracy"] = test_metrics['accuracy']
                        task_summary[f"{model}_f1_macro"] = test_metrics.get('f1_macro', 0.0)
                    else:
                        # Regression metrics
                        task_summary[f"{model}_mse"] = test_metrics.get('mse', float('inf'))
                        task_summary[f"{model}_r2"] = test_metrics.get('r2', 0.0)
                    
                    # Training time
                    task_summary[f"{model}_train_time"] = model_results.get('train_time', 0.0)
            
            summary[task] = task_summary
        
        return summary
    
    def _extract_family_stats(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract graph family statistics."""
        if 'family_graphs' not in experiment_results:
            return {}
        
        family_graphs = experiment_results['family_graphs']
        
        if not family_graphs:
            return {}
        
        # Calculate family statistics
        stats = {
            'n_graphs': len(family_graphs),
            'node_counts': [],
            'edge_counts': [],
            'densities': [],
            'avg_degrees': [],
            'community_counts': []
        }
        
        for graph in family_graphs:
            stats['node_counts'].append(graph.n_nodes)
            stats['edge_counts'].append(graph.graph.number_of_edges())
            
            if graph.n_nodes > 1:
                density = graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2)
                stats['densities'].append(density)
            else:
                stats['densities'].append(0.0)
            
            if graph.n_nodes > 0:
                avg_degree = sum(dict(graph.graph.degree()).values()) / graph.n_nodes
                stats['avg_degrees'].append(avg_degree)
            else:
                stats['avg_degrees'].append(0.0)
            
            stats['community_counts'].append(len(np.unique(graph.community_labels)))
        
        # Calculate summary statistics
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'community_counts']:
            values = stats[key]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        return stats
    
    def _get_models_run(self, config: InductiveExperimentConfig) -> List[str]:
        """Get list of models that were run."""
        models = []
        if config.run_gnn:
            models.extend(config.gnn_types)
        if config.run_mlp:
            models.append('mlp')
        if config.run_rf:
            models.append('rf')
        return models
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results."""
        intermediate_path = os.path.join(self.output_dir, "intermediate_results.json")
        
        data = {
            'all_results': self.all_results,
            'failed_runs': self.failed_runs,
            'n_completed': len(self.all_results),
            'n_failed': len(self.failed_runs),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(intermediate_path, 'w') as f:
            json.dump(self._make_json_serializable(data), f, indent=2)
    
    def _save_final_results(self, total_time: float) -> Dict[str, Any]:
        """Save final results and create summary files."""
        
        # Save complete results
        final_results = {
            'config': self.config.__dict__,
            'all_results': self.all_results,
            'failed_runs': self.failed_runs,
            'summary_stats': {
                'total_runs_attempted': len(self.all_results) + len(self.failed_runs),
                'successful_runs': len(self.all_results),
                'failed_runs': len(self.failed_runs),
                'success_rate': len(self.all_results) / (len(self.all_results) + len(self.failed_runs)) if (len(self.all_results) + len(self.failed_runs)) > 0 else 0,
                'total_time': total_time,
                'avg_time_per_run': total_time / len(self.all_results) if self.all_results else 0
            },
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save JSON results
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(self._make_json_serializable(final_results), f, indent=2)
        
        # Create and save DataFrame for analysis
        if self.all_results:
            df = self._create_results_dataframe()
            csv_path = os.path.join(self.output_dir, "results_summary.csv")
            df.to_csv(csv_path, index=False)
            
            # Save parameter summary
            param_summary = self._create_parameter_summary(df)
            param_path = os.path.join(self.output_dir, "parameter_summary.json")
            with open(param_path, 'w') as f:
                json.dump(param_summary, f, indent=2)
        
        return final_results
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results for easy analysis."""
        rows = []
        
        for result in self.all_results:
            row = {
                'run_id': result['run_id'],
                'timestamp': result['timestamp'],
                'total_time': result['total_time'],
                'n_graphs': result['n_graphs']
            }
            
            # Add all parameters
            row.update(result['all_parameters'])
            
            # Add family stats
            if 'family_stats' in result:
                family_stats = result['family_stats']
                for key, value in family_stats.items():
                    if isinstance(value, (int, float)):
                        row[f'family_{key}'] = value
            
            # Add model results
            if 'model_results' in result:
                for task, task_results in result['model_results'].items():
                    for metric, value in task_results.items():
                        row[f'{task}_{metric}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_parameter_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary of parameter effects on model performance."""
        summary = {
            'sweep_parameters': list(self.config.sweep_parameters.keys()),
            'random_parameters': list(self.config.random_parameters.keys()),
            'parameter_correlations': {},
            'best_configurations': {},
            'worst_configurations': {}
        }
        
        # Find numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        param_cols = [col for col in numeric_cols if col in self.config.sweep_parameters or col in self.config.random_parameters]
        
        # Calculate correlations between parameters and performance metrics
        performance_cols = [col for col in numeric_cols if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2', 'mse'])]
        
        if param_cols and performance_cols:
            correlation_matrix = df[param_cols + performance_cols].corr()
            
            for perf_col in performance_cols:
                param_correlations = {}
                for param_col in param_cols:
                    if param_col in correlation_matrix.index and perf_col in correlation_matrix.columns:
                        corr_val = correlation_matrix.loc[param_col, perf_col]
                        if not np.isnan(corr_val):
                            param_correlations[param_col] = float(corr_val)
                
                if param_correlations:
                    summary['parameter_correlations'][perf_col] = param_correlations
        
        # Find best and worst configurations for each performance metric
        for perf_col in performance_cols:
            if perf_col in df.columns:
                # Best configuration (highest for accuracy/f1/r2, lowest for mse)
                if 'mse' in perf_col.lower():
                    best_idx = df[perf_col].idxmin()
                    worst_idx = df[perf_col].idxmax()
                else:
                    best_idx = df[perf_col].idxmax()
                    worst_idx = df[perf_col].idxmin()
                
                if not pd.isna(best_idx):
                    best_config = {}
                    worst_config = {}
                    
                    for param in param_cols:
                        if param in df.columns:
                            best_config[param] = df.loc[best_idx, param]
                            worst_config[param] = df.loc[worst_idx, param]
                    
                    summary['best_configurations'][perf_col] = {
                        'parameters': best_config,
                        'performance': float(df.loc[best_idx, perf_col]),
                        'run_id': int(df.loc[best_idx, 'run_id'])
                    }
                    
                    summary['worst_configurations'][perf_col] = {
                        'parameters': worst_config,
                        'performance': float(df.loc[worst_idx, perf_col]),
                        'run_id': int(df.loc[worst_idx, 'run_id'])
                    }
        
        return summary
    
    def _generate_summary(self, total_time: float) -> str:
        """Generate a text summary of the multi-experiment run."""
        lines = []
        lines.append("MULTI-INDUCTIVE EXPERIMENT SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic statistics
        total_attempted = len(self.all_results) + len(self.failed_runs)
        success_rate = len(self.all_results) / total_attempted if total_attempted > 0 else 0
        
        lines.append(f"Total runs attempted: {total_attempted}")
        lines.append(f"Successful runs: {len(self.all_results)}")
        lines.append(f"Failed runs: {len(self.failed_runs)}")
        lines.append(f"Success rate: {success_rate:.1%}")
        lines.append(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        lines.append(f"Average time per run: {total_time/total_attempted:.1f}s" if total_attempted > 0 else "Average time per run: N/A")
        lines.append("")
        
        # Parameter sweep summary
        sweep_combinations = len(self.config.get_parameter_combinations())
        lines.append(f"Parameter sweep combinations: {sweep_combinations}")
        lines.append(f"Repetitions per combination: {self.config.n_repetitions}")
        lines.append("")
        
        lines.append("Sweep parameters:")
        for param, param_range in self.config.sweep_parameters.items():
            values = param_range.get_sweep_values()
            lines.append(f"  {param}: {values}")
        lines.append("")
        
        lines.append("Random parameters:")
        for param, param_range in self.config.random_parameters.items():
            lines.append(f"  {param}: [{param_range.min_val}, {param_range.max_val}] ({param_range.distribution})")
        lines.append("")
        
        # Performance summary (if we have results)
        if self.all_results:
            lines.append("Performance Summary:")
            df = self._create_results_dataframe()
            
            # Find performance columns
            performance_cols = [col for col in df.columns if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2', 'mse'])]
            
            for col in performance_cols:
                if col in df.columns and not df[col].isna().all():
                    lines.append(f"  {col}:")
                    lines.append(f"    Mean: {df[col].mean():.4f}")
                    lines.append(f"    Std:  {df[col].std():.4f}")
                    lines.append(f"    Min:  {df[col].min():.4f}")
                    lines.append(f"    Max:  {df[col].max():.4f}")
        
        lines.append("")
        lines.append(f"Results saved to: {self.output_dir}")
        
        return "\n".join(lines)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            try:
                return self._make_json_serializable(obj.__dict__)
            except:
                return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj

    def _process_tuple_parameters(self, random_params: Dict[str, float]) -> Dict[str, Any]:
        """Process parameters that need to be converted to tuples (like ranges)."""
        processed = random_params.copy()
        
        # Convert range parameters from separate values to tuples
        if 'homophily_range_0' in processed:
            hr_val = processed.pop('homophily_range_0')
            dr_val = processed.pop('density_range_0', hr_val)
            processed['homophily_range'] = (0.0, hr_val)
            processed['density_range'] = (0.0, dr_val)
        
        # Convert DCCC-SBM range parameters
        if 'community_imbalance_range_0' in processed:
            ci_val = processed.pop('community_imbalance_range_0')
            processed['community_imbalance_range'] = (0.0, ci_val)
        
        if 'degree_separation_range_0' in processed:
            ds_val = processed.pop('degree_separation_range_0')
            processed['degree_separation_range'] = (0.1, ds_val)
        
        # Convert distribution-specific range parameters
        if 'power_law_exponent_range_0' in processed:
            ple_val = processed.pop('power_law_exponent_range_0')
            processed['power_law_exponent_range'] = (2.0, ple_val)
        
        if 'exponential_rate_range_0' in processed:
            er_val = processed.pop('exponential_rate_range_0')
            processed['exponential_rate_range'] = (0.1, er_val)
        
        if 'uniform_min_factor_range_0' in processed:
            umf_val = processed.pop('uniform_min_factor_range_0')
            processed['uniform_min_factor_range'] = (0.1, umf_val)
        
        if 'uniform_max_factor_range_0' in processed:
            umxf_val = processed.pop('uniform_max_factor_range_0')
            processed['uniform_max_factor_range'] = (1.0, umxf_val)
        
        return processed


def run_multi_inductive_experiments(config: MultiInductiveExperimentConfig) -> Dict[str, Any]:
    """
    Convenience function to run multi-inductive experiments.
    
    Args:
        config: Multi-experiment configuration
        
    Returns:
        Complete results dictionary
    """
    runner = MultiInductiveExperimentRunner(config)
    return runner.run_all_experiments()


def create_analysis_plots(results_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Create analysis plots from multi-experiment results.
    
    Args:
        results_dir: Directory containing multi-experiment results
        output_dir: Directory to save plots (defaults to results_dir/plots)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    csv_path = os.path.join(results_dir, "results_summary.csv")
    if not os.path.exists(csv_path):
        print(f"No results CSV found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Load config to get parameter information
    config_path = os.path.join(results_dir, "multi_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        sweep_params = list(config_data['sweep_parameters'].keys())
        random_params = list(config_data['random_parameters'].keys())
    else:
        # Try to infer from data
        sweep_params = []
        random_params = []
    
    # Find performance columns
    performance_cols = [col for col in df.columns if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2'])]
    
    # Create plots
    plt.style.use('default')
    
    # 1. Sweep parameter effects
    for sweep_param in sweep_params:
        if sweep_param in df.columns:
            fig, axes = plt.subplots(1, len(performance_cols), figsize=(5*len(performance_cols), 4))
            if len(performance_cols) == 1:
                axes = [axes]
            
            for i, perf_col in enumerate(performance_cols):
                if perf_col in df.columns:
                    # Group by sweep parameter and calculate mean/std
                    grouped = df.groupby(sweep_param)[perf_col].agg(['mean', 'std']).reset_index()
                    
                    axes[i].errorbar(grouped[sweep_param], grouped['mean'], yerr=grouped['std'], 
                                   marker='o', capsize=5)
                    axes[i].set_xlabel(sweep_param)
                    axes[i].set_ylabel(perf_col)
                    axes[i].set_title(f'{perf_col} vs {sweep_param}')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sweep_{sweep_param}_effects.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Parameter correlation heatmap
    param_cols = [col for col in df.columns if col in sweep_params + random_params]
    if param_cols and performance_cols:
        correlation_data = df[param_cols + performance_cols].corr()
        
        # Extract parameter-performance correlations
        param_perf_corr = correlation_data.loc[param_cols, performance_cols]
        
        plt.figure(figsize=(max(8, len(performance_cols) * 1.5), max(6, len(param_cols) * 0.8)))
        sns.heatmap(param_perf_corr, annot=True, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Correlation'})
        plt.title('Parameter-Performance Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance distribution plots
    if performance_cols:
        fig, axes = plt.subplots(1, len(performance_cols), figsize=(5*len(performance_cols), 4))
        if len(performance_cols) == 1:
            axes = [axes]
        
        for i, perf_col in enumerate(performance_cols):
            if perf_col in df.columns:
                df[perf_col].hist(bins=20, alpha=0.7, ax=axes[i])
                axes[i].axvline(df[perf_col].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df[perf_col].mean():.3f}')
                axes[i].set_xlabel(perf_col)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {perf_col}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Analysis plots saved to: {output_dir}")