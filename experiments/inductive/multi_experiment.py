"""
Clean multi-experiment runner for parameter sweeps and random sampling.
Removes old parameters and focuses on DC-SBM and DCCC-SBM methods only.
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
import tempfile

from experiments.inductive.multi_config import CleanMultiExperimentConfig
from experiments.run_inductive_experiments import run_inductive_experiment
from experiments.inductive.config import InductiveExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleanMultiExperimentRunner:
    """Runner for multiple clean inductive experiments with parameter sweeps."""
    
    def __init__(self, config: CleanMultiExperimentConfig):
        """Initialize multi-experiment runner."""
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
        
        print(f"Multi-experiment runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Total runs planned: {config.get_total_runs()}")
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured experiments."""
        print(f"\n{'='*80}")
        print(f"STARTING CLEAN MULTI-EXPERIMENT SUITE")
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
        base_seed = 42
        
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
                        
                        # Process special parameters
                        processed_sweep = self._process_special_parameters(sweep_params)
                        processed_random = self._process_special_parameters(random_params)
                        
                        # Create run configuration
                        run_config = self.config.create_run_config(
                            sweep_params=processed_sweep,
                            random_params=processed_random,
                            run_id=run_id
                        )
                        
                        # Set the run's seed in the config
                        run_config.seed = run_seed
                        
                        # Run single experiment
                        result = self._run_single_experiment(
                            config=run_config,
                            run_id=run_id,
                            sweep_params=processed_sweep,
                            random_params=processed_random,
                            run_subdir=None
                        )
                        
                        if result is not None:
                            self.all_results.append(result)
                            print(f"Run {run_id} completed successfully")
                        else:
                            print(f"Run {run_id} failed")
                        
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
                        
                        print(f"Run {run_id} failed with error: {str(e)}")
                        
                        if not self.config.continue_on_failure:
                            print("Stopping due to failure (continue_on_failure=False)")
                            break
                    
                    run_id += 1
                    pbar.update(1)
                    
                    # Save intermediate results after every run
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
    
    def _process_special_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Process special parameters that need conversion."""
        processed = params.copy()
        
        # Convert degree_distribution index to string
        if 'degree_distribution' in processed:
            dist_map = {0: 'standard', 1: 'power_law', 2: 'exponential', 3: 'uniform'}
            idx = int(processed['degree_distribution'])
            processed['degree_distribution'] = dist_map.get(idx, 'power_law')
        
        # Convert use_dccc_sbm index to boolean
        if 'use_dccc_sbm' in processed:
            processed['use_dccc_sbm'] = bool(int(processed['use_dccc_sbm']))
        
        return processed
    
    def _run_single_experiment(
        self,
        config: InductiveExperimentConfig,
        run_id: int,
        sweep_params: Dict[str, Any],
        random_params: Dict[str, Any],
        run_subdir: str
    ) -> Optional[Dict[str, Any]]:
        """Run a single inductive experiment."""
        try:
            # Create a temporary directory for this run
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set the output directory to the temporary directory
                config.output_dir = temp_dir
                
                # Run the experiment
                experiment_results = run_inductive_experiment(config)
                
                # Compile complete result record
                complete_result = {
                    # Run metadata
                    'run_id': run_id,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    
                    # Input parameters
                    'sweep_parameters': sweep_params,
                    'random_parameters': random_params,
                    'all_parameters': {**sweep_params, **random_params},
                    
                    # Method info
                    'method': 'dccc_sbm' if config.use_dccc_sbm else 'dc_sbm',
                    'degree_distribution': config.degree_distribution if config.use_dccc_sbm else 'standard',
                    
                    # Graph family metrics
                    'family_properties': self._extract_family_properties(experiment_results),
                    
                    # Family consistency metrics
                    'family_consistency': experiment_results.get('family_consistency', {}),
                    
                    # Community signal metrics
                    'community_signals': experiment_results.get('graph_signals', {}),
                    
                    # Model results
                    'model_results': self._extract_model_results(experiment_results),
                    
                    # Experiment metadata
                    'total_time': experiment_results.get('total_time', 0),
                    'n_graphs': config.n_graphs,
                    'universe_K': config.universe_K,
                    'tasks': config.tasks
                }
                
                return complete_result
            
        except Exception as e:
            logger.error(f"Error in run {run_id}: {str(e)}", exc_info=True)
            return None
    
    def _extract_family_properties(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract family properties from experiment results."""
        if 'family_graphs' not in experiment_results:
            return {}
        
        family_graphs = experiment_results['family_graphs']
        if not family_graphs:
            return {}
        
        # Calculate basic statistics
        properties = {
            'n_graphs': len(family_graphs),
            'node_counts': [g.n_nodes for g in family_graphs],
            'edge_counts': [g.graph.number_of_edges() for g in family_graphs],
            'community_counts': [len(np.unique(g.community_labels)) for g in family_graphs]
        }
        
        # Calculate summary statistics
        for key in ['node_counts', 'edge_counts', 'community_counts']:
            values = properties[key]
            if values:
                properties[f'{key}_mean'] = float(np.mean(values))
                properties[f'{key}_std'] = float(np.std(values))
                properties[f'{key}_min'] = float(np.min(values))
                properties[f'{key}_max'] = float(np.max(values))
        
        return properties
    
    def _extract_model_results(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model results from experiment results."""
        if 'results' not in experiment_results:
            return {}
        
        clean_results = {}
        for task, task_results in experiment_results['results'].items():
            clean_results[task] = {}
            for model_name, model_results in task_results.items():
                # Extract key metrics
                test_metrics = model_results.get('test_metrics', {})
                optimal_hyperparams = model_results.get('optimal_hyperparams', {})
                
                # # Extract hyperopt results if available
                # optimal_hyperparams = None
                # if 'hyperopt_results' in model_results and model_results['hyperopt_results'] is not None:
                #     hyperopt = model_results['hyperopt_results']
                #     if isinstance(hyperopt, dict) and 'optimal_hyperparams' in hyperopt:
                #         optimal_hyperparams = hyperopt['optimal_hyperparams']
                
                clean_results[task][model_name] = {
                    'test_metrics': test_metrics,
                    'train_time': model_results.get('train_time', 0.0),
                    'success': bool(test_metrics),  # True if we have test metrics
                    'optimal_hyperparams': optimal_hyperparams
                }
        
        return clean_results
    
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
            'config': self._make_json_serializable(self.config.__dict__),
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
            
            # Save parameter analysis
            param_analysis = self._analyze_parameter_effects(df)
            param_path = os.path.join(self.output_dir, "parameter_analysis.json")
            with open(param_path, 'w') as f:
                json.dump(param_analysis, f, indent=2)
        
        return final_results
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results for easy analysis."""
        rows = []
        
        for result in self.all_results:
            row = {
                'run_id': result['run_id'],
                'timestamp': result['timestamp'],
                'total_time': result['total_time'],
                'method': result['method'],
                'degree_distribution': result['degree_distribution'],
                'n_graphs': result['n_graphs']
            }
            
            # Add all parameters
            row.update(result['all_parameters'])
            
            # Add family properties
            family_props = result.get('family_properties', {})
            for key, value in family_props.items():
                if isinstance(value, (int, float)):
                    row[f'family_{key}'] = value
            
            # Add family consistency metrics
            family_consistency = result.get('family_consistency', {})
            if family_consistency:
                row['consistency_overall'] = family_consistency.get('overall', {}).get('score', 0.0)
                row['consistency_pattern'] = family_consistency.get('pattern_preservation', {}).get('score', 0.0)
                row['consistency_fidelity'] = family_consistency.get('generation_fidelity', {}).get('score', 0.0)
                row['consistency_degree'] = family_consistency.get('degree_consistency', {}).get('score', 0.0)
            
            # Add community signals
            community_signals = result.get('community_signals', {})
            for signal_type in ['degree_signals', 'structure_signals', 'feature_signals']:
                signal_data = community_signals.get(signal_type, {})
                if signal_data and 'mean' in signal_data:
                    row[f'signal_{signal_type}_mean'] = signal_data['mean']
                    row[f'signal_{signal_type}_std'] = signal_data['std']
            
            # Add model results
            model_results = result.get('model_results', {})
            for task, task_results in model_results.items():
                for model, model_data in task_results.items():
                    test_metrics = model_data.get('test_metrics', {})
                    
                    # Add key metrics
                    for metric in ['accuracy', 'f1_macro', 'r2', 'mse']:
                        if metric in test_metrics:
                            row[f'{task}_{model}_{metric}'] = test_metrics[metric]
                    
                    # Add training time and success
                    row[f'{task}_{model}_train_time'] = model_data.get('train_time', 0.0)
                    row[f'{task}_{model}_success'] = model_data.get('success', False)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _analyze_parameter_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze parameter effects on model performance."""
        analysis = {
            'sweep_parameters': list(self.config.sweep_parameters.keys()),
            'random_parameters': list(self.config.random_parameters.keys()),
            'parameter_correlations': {},
            'best_configurations': {},
            'method_comparison': {}
        }
        
        # Find numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        param_cols = [col for col in numeric_cols if col in analysis['sweep_parameters'] + analysis['random_parameters']]
        
        # Find performance metrics
        performance_cols = [col for col in numeric_cols if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2']) and 'mse' not in col]
        
        # Calculate correlations
        if param_cols and performance_cols:
            try:
                correlation_matrix = df[param_cols + performance_cols].corr()
                
                for perf_col in performance_cols:
                    param_correlations = {}
                    for param_col in param_cols:
                        if param_col in correlation_matrix.index and perf_col in correlation_matrix.columns:
                            corr_val = correlation_matrix.loc[param_col, perf_col]
                            if not pd.isna(corr_val):
                                param_correlations[param_col] = float(corr_val)
                    
                    if param_correlations:
                        analysis['parameter_correlations'][perf_col] = param_correlations
            except Exception as e:
                print(f"Warning: Could not calculate correlations: {e}")
        
        # Find best configurations
        for perf_col in performance_cols:
            if perf_col in df.columns and not df[perf_col].isna().all():
                best_idx = df[perf_col].idxmax()
                
                if not pd.isna(best_idx):
                    best_config = {}
                    for param in param_cols:
                        if param in df.columns:
                            best_config[param] = df.loc[best_idx, param]
                    
                    analysis['best_configurations'][perf_col] = {
                        'parameters': best_config,
                        'performance': float(df.loc[best_idx, perf_col]),
                        'run_id': int(df.loc[best_idx, 'run_id']),
                        'method': df.loc[best_idx, 'method']
                    }
        
        # Method comparison
        if 'method' in df.columns:
            method_stats = {}
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                method_stats[method] = {
                    'n_runs': len(method_df),
                    'avg_performance': {}
                }
                
                for perf_col in performance_cols:
                    if perf_col in method_df.columns and not method_df[perf_col].isna().all():
                        method_stats[method]['avg_performance'][perf_col] = {
                            'mean': float(method_df[perf_col].mean()),
                            'std': float(method_df[perf_col].std()),
                            'min': float(method_df[perf_col].min()),
                            'max': float(method_df[perf_col].max())
                        }
            
            analysis['method_comparison'] = method_stats
        
        return analysis
    
    def _generate_summary(self, total_time: float) -> str:
        """Generate a text summary of the multi-experiment run."""
        lines = []
        lines.append("CLEAN MULTI-EXPERIMENT SUMMARY")
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
        
        # Performance summary
        if self.all_results:
            lines.append("Performance Summary:")
            df = self._create_results_dataframe()
            
            # Method comparison
            if 'method' in df.columns:
                lines.append("Method comparison:")
                for method in df['method'].unique():
                    method_df = df[df['method'] == method]
                    lines.append(f"  {method.upper()}: {len(method_df)} runs")
                    
                    # Find best performance metric for this method
                    perf_cols = [col for col in df.columns if 'f1_macro' in col or 'accuracy' in col]
                    for col in perf_cols[:2]:  # Show top 2 metrics
                        if col in method_df.columns and not method_df[col].isna().all():
                            mean_perf = method_df[col].mean()
                            std_perf = method_df[col].std()
                            lines.append(f"    {col}: {mean_perf:.3f} ± {std_perf:.3f}")
                lines.append("")
            
            # Signal summary
            signal_cols = [col for col in df.columns if 'signal_' in col and '_mean' in col]
            if signal_cols:
                lines.append("Average Community Signals:")
                for col in signal_cols:
                    signal_name = col.replace('signal_', '').replace('_mean', '').replace('_', ' ').title()
                    avg_signal = df[col].mean()
                    lines.append(f"  {signal_name}: {avg_signal:.3f}")
                lines.append("")
            
            # Consistency summary
            if 'consistency_overall' in df.columns:
                avg_consistency = df['consistency_overall'].mean()
                lines.append(f"Average Family Consistency: {avg_consistency:.3f}")
                lines.append("")
        
        lines.append(f"Results saved to: {self.output_dir}")
        
        return "\n".join(lines)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
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

    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """Get all model configurations to run."""
        model_configs = []
        
        # Add GNN models if enabled
        if self.config.base_config.run_gnn:
            for gnn_type in self.config.gnn_models:
                model_configs.append({
                    'gnn_types': [gnn_type],
                    'run_gnn': True,
                    'run_mlp': False,
                    'run_rf': False,
                    'run_transformers': False
                })
        
        # Add transformer models if enabled
        if self.config.run_transformers:
            for transformer_type in self.config.transformer_models:
                model_configs.append({
                    'transformer_types': [transformer_type],
                    'run_gnn': False,
                    'run_mlp': False,
                    'run_rf': False,
                    'run_transformers': True,
                    **self.config.transformer_params
                })
        
        # Add MLP model if enabled
        if self.config.base_config.run_mlp:
            model_configs.append({
                'run_gnn': False,
                'run_mlp': True,
                'run_rf': False,
                'run_transformers': False
            })
        
        # Add Random Forest model if enabled
        if self.config.base_config.run_rf:
            model_configs.append({
                'run_gnn': False,
                'run_mlp': False,
                'run_rf': True,
                'run_transformers': False
            })
        
        return model_configs


def run_clean_multi_experiments(config: CleanMultiExperimentConfig) -> Dict[str, Any]:
    """Convenience function to run multi-experiments."""
    runner = CleanMultiExperimentRunner(config)
    return runner.run_all_experiments()


def create_analysis_plots(results_dir: str, output_dir: Optional[str] = None) -> None:
    """Create analysis plots from multi-experiment results."""
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
    csv_path = os.path.join(results_dir, "results_summary.csv")
    if not os.path.exists(csv_path):
        print(f"No results CSV found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Find performance and parameter columns
    performance_cols = [col for col in df.columns if any(metric in col for metric in ['accuracy', 'f1_macro', 'r2']) and 'mse' not in col]
    param_cols = [col for col in df.columns if any(param in col for param in ['universe_', 'homophily', 'density', 'degree_'])]
    
    plt.style.use('default')
    
    # 1. Method comparison
    if 'method' in df.columns and performance_cols:
        fig, axes = plt.subplots(1, len(performance_cols), figsize=(5*len(performance_cols), 4))
        if len(performance_cols) == 1:
            axes = [axes]
        
        for i, perf_col in enumerate(performance_cols):
            if perf_col in df.columns:
                df.boxplot(column=perf_col, by='method', ax=axes[i])
                axes[i].set_title(f'{perf_col} by Method')
                axes[i].set_xlabel('Method')
                axes[i].set_ylabel(perf_col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Parameter effects (first few parameters)
    for param in param_cols[:4]:  # Limit to avoid too many plots
        if param in df.columns and performance_cols:
            n_plots = min(2, len(performance_cols))
            fig, axes = plt.subplots(1, n_plots, figsize=(10, 4))
            # Ensure axes is always a list
            if n_plots == 1:
                axes = [axes]
            elif not isinstance(axes, list):
                axes = list(axes)
            
            for i, perf_col in enumerate(performance_cols[:2]):
                if perf_col in df.columns:
                    axes[i].scatter(df[param], df[perf_col], alpha=0.6)
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel(perf_col)
                    axes[i].set_title(f'{perf_col} vs {param}')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'param_effect_{param}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Signal analysis
    signal_cols = [col for col in df.columns if 'signal_' in col and '_mean' in col]
    if signal_cols:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        signal_data = []
        signal_names = []
        for col in signal_cols:
            signal_data.append(df[col].dropna())
            signal_names.append(col.replace('signal_', '').replace('_mean', '').replace('_', ' ').title())
        
        ax.boxplot(signal_data, labels=signal_names)
        ax.set_title('Community Signal Distributions')
        ax.set_ylabel('Signal Strength')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'signal_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Analysis plots saved to: {output_dir}")