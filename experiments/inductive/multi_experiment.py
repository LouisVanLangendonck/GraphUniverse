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
import sys
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.inductive.multi_config import CleanMultiExperimentConfig, SSLMultiExperimentConfig
from experiments.run_inductive_experiments import run_inductive_experiment
from experiments.inductive.config import InductiveExperimentConfig, PreTrainingConfig
from experiments.inductive.multi_config import ParameterRange
from experiments.inductive.experiment import PreTrainingRunner
from experiments.inductive.data import GraphFamilyManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CleanMultiExperimentRunner:
    """Runner for multiple clean inductive experiments with parameter sweeps."""
    
    def __init__(self, config: CleanMultiExperimentConfig, continue_from_results: dict = None):
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
        
        # If continuing from previous results, load them
        if continue_from_results:
            self.all_results = continue_from_results.get('all_results', [])
            self.failed_runs = continue_from_results.get('failed_runs', [])
            print(f"Loaded {len(self.all_results)} completed runs and {len(self.failed_runs)} failed runs")
        
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
        
        # Get completed sweep parameter combinations
        completed_combos = set()
        for result in self.all_results:
            sweep_params = result.get('sweep_parameters', {})
            # Convert to tuple of sorted items for hashability
            param_tuple = tuple(sorted(sweep_params.items()))
            completed_combos.add(param_tuple)
        
        # Filter out completed combinations
        remaining_combinations = []
        for combo in sweep_combinations:
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in completed_combos:
                remaining_combinations.append(combo)
        
        print(f"Remaining combinations to run: {len(remaining_combinations)}")
        
        run_id = len(self.all_results)  # Start from the next run ID
        base_seed = 42 + run_id  # Adjust base seed to avoid overlap
        
        # Progress tracking
        with tqdm(total=len(remaining_combinations) * self.config.n_repetitions, desc="Running experiments") as pbar:
            
            for combo_idx, sweep_params in enumerate(remaining_combinations):
                print(f"\n{'-'*60}")
                print(f"COMBINATION {combo_idx + 1}/{len(remaining_combinations)}")
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
        
        # Use the same analysis function as individual experiments
        try:
            from experiments.inductive.data import analyze_graph_family_properties
            properties = analyze_graph_family_properties(family_graphs)
            return self._make_json_serializable(properties)
        except Exception as e:
            print(f"Warning: Failed to analyze family properties: {e}")
            # Fallback to basic properties
            properties = {
                'n_graphs': len(family_graphs),
                'node_counts': [g.n_nodes for g in family_graphs],
                'edge_counts': [g.graph.number_of_edges() for g in family_graphs],
                'community_counts': [len(np.unique(g.community_labels)) for g in family_graphs]
            }
            
            # Calculate summary statistics only for properties that exist
            for key in ['node_counts', 'edge_counts', 'community_counts']:
                if key in properties and properties[key]:
                    values = properties[key]
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
                # Extract fold metrics
                fold_test_metrics = model_results.get('fold_test_metrics', {})
                fold_best_val_metrics = model_results.get('fold_best_val_metrics', {})
                fold_train_time = model_results.get('fold_train_time', {})
                
                # Calculate statistics for each metric
                test_metrics = {}
                best_val_metrics = []
                train_times = []
                
                # Process test metrics
                for fold_name, fold_data in fold_test_metrics.items():
                    for metric, value in fold_data.items():
                        if metric not in test_metrics:
                            test_metrics[metric] = []
                        test_metrics[metric].append(value)
                
                # Process best val metrics
                for fold_name, value in fold_best_val_metrics.items():
                    best_val_metrics.append(value)
                
                # Process train times
                for fold_name, value in fold_train_time.items():
                    train_times.append(value)
                
                # Calculate final statistics
                final_test_metrics = {}
                for metric, values in test_metrics.items():
                    final_test_metrics[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'list': values
                    }
                
                final_best_val_metrics = {
                    'mean': float(np.mean(best_val_metrics)) if best_val_metrics else 0.0,
                    'std': float(np.std(best_val_metrics)) if best_val_metrics else 0.0,
                    'list': best_val_metrics
                }
                
                final_train_time = {
                    'mean': float(np.mean(train_times)) if train_times else 0.0,
                    'std': float(np.std(train_times)) if train_times else 0.0,
                    'list': train_times
                }
                
                clean_results[task][model_name] = {
                    'test_metrics': final_test_metrics,
                    'best_val_metrics': final_best_val_metrics,
                    'train_time': final_train_time,
                    'optimal_hyperparams': model_results.get('optimal_hyperparams', {}),
                    'success': bool(fold_test_metrics)  # True if we have fold metrics
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
                    # Add test metrics
                    test_metrics = model_data.get('test_metrics', {})
                    for metric, metric_data in test_metrics.items():
                        if isinstance(metric_data, dict):
                            row[f'{task}_{model}_{metric}_mean'] = metric_data.get('mean', 0.0)
                            row[f'{task}_{model}_{metric}_std'] = metric_data.get('std', 0.0)
                    
                    # Add best val metrics
                    best_val_metrics = model_data.get('best_val_metrics', {})
                    if isinstance(best_val_metrics, dict):
                        row[f'{task}_{model}_best_val_mean'] = best_val_metrics.get('mean', 0.0)
                        row[f'{task}_{model}_best_val_std'] = best_val_metrics.get('std', 0.0)
                    
                    # Add training time
                    train_time = model_data.get('train_time', {})
                    if isinstance(train_time, dict):
                        row[f'{task}_{model}_train_time_mean'] = train_time.get('mean', 0.0)
                        row[f'{task}_{model}_train_time_std'] = train_time.get('std', 0.0)
                    
                    # Add success flag
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
                    
                    # Find performance metrics for this method
                    perf_cols = [col for col in df.columns if '_mean' in col and any(metric in col for metric in ['f1_macro', 'accuracy', 'r2'])]
                    for col in perf_cols[:2]:  # Show top 2 metrics
                        if col in method_df.columns and not method_df[col].isna().all():
                            mean_perf = method_df[col].mean()
                            std_perf = method_df[col].std()
                            metric_name = col.replace('_mean', '').replace('_', ' ').title()
                            lines.append(f"    {metric_name}: {mean_perf:.3f} ± {std_perf:.3f}")
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
            
            # Family properties summary
            family_prop_cols = [col for col in df.columns if col.startswith('family_') and '_mean' in col]
            if family_prop_cols:
                lines.append("Average Family Properties:")
                for col in family_prop_cols:
                    prop_name = col.replace('family_', '').replace('_mean', '').replace('_', ' ').title()
                    avg_prop = df[col].mean()
                    lines.append(f"  {prop_name}: {avg_prop:.3f}")
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
                    'run_transformers': False,
                    'run_neural_sheaf': False
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
                    'run_neural_sheaf': False,
                    **self.config.transformer_params
                })
        
        # Add neural sheaf models if enabled
        if self.config.run_neural_sheaf:
            model_configs.append({
                'run_gnn': False,
                'run_mlp': False,
                'run_rf': False,
                'run_transformers': False,
                'run_neural_sheaf': True
            })
        
        # Add MLP model if enabled
        if self.config.base_config.run_mlp:
            model_configs.append({
                'run_gnn': False,
                'run_mlp': True,
                'run_rf': False,
                'run_transformers': False,
                'run_neural_sheaf': False
            })
        
        # Add Random Forest model if enabled
        if self.config.base_config.run_rf:
            model_configs.append({
                'run_gnn': False,
                'run_mlp': False,
                'run_rf': True,
                'run_transformers': False,
                'run_neural_sheaf': False
            })
        
        return model_configs


class SSLMultiExperimentRunner:
    """Runner for multiple SSL experiments."""
    
    def __init__(self, config: SSLMultiExperimentConfig):
        self.config = config
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            config.output_dir, 
            f"{config.experiment_name}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
        # Initialize tracking
        self.all_results = []
        self.failed_experiments = []
        self.generated_families = {}  # family_config_hash -> family_id
        
        print(f"Multi-SSL experiment runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Total experiments planned: {config.get_total_experiments()}")
    
    def save_config(self):
        """Save experiment configuration."""
        config_dict = {
            'experiment_name': self.config.experiment_name,
            'total_experiments': self.config.get_total_experiments(),
            'family_configurations': len(self.config.get_family_configurations()),
            'model_configurations': len(self.config.get_model_configurations()),
            'parameters': {
                'n_repetitions': self.config.n_repetitions,
                'continue_on_failure': self.config.continue_on_failure,
                'save_individual_results': self.config.save_individual_results,
                'max_concurrent_families': self.config.max_concurrent_families,
                'reuse_families': self.config.reuse_families,
                'base_seed': self.config.base_seed
            },
            'sweep_parameters': {
                param: {
                    'min_val': range_obj.min_val,
                    'max_val': range_obj.max_val,
                    'step': range_obj.step,
                    'is_sweep': range_obj.is_sweep,
                    'discrete_values': range_obj.discrete_values,
                    'distribution': getattr(range_obj, 'distribution', None)
                } for param, range_obj in self.config.sweep_parameters.items()
            },
            'random_parameters': {
                param: {
                    'min_val': range_obj.min_val,
                    'max_val': range_obj.max_val,
                    'distribution': range_obj.distribution,
                    'is_sweep': range_obj.is_sweep
                } for param, range_obj in self.config.random_parameters.items()
            },
            'base_config': self.config.base_config.to_dict() if self.config.base_config else None,
            'gnn_models': self.config.gnn_models,
            'transformer_models': self.config.transformer_models,
            'run_transformers': self.config.run_transformers,
            'transformer_params': self.config.transformer_params,
            'skip_gnn': self.config.skip_gnn,
            'patience': self.config.patience
        }
        
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_family_hash(self, family_config: Dict[str, Any]) -> str:
        """Get a unique hash for a family configuration."""
        # Create a hash from the important family parameters
        hash_parts = [
            str(family_config['n_graphs']),
            str(family_config['n_extra_graphs']),
            str(family_config['universe_K']),
            str(family_config['universe_homophily']),
            str(family_config['universe_edge_density']),
            str(family_config['use_dccc_sbm']),
            str(family_config['degree_distribution'])
        ]
        return "_".join(hash_parts)
    
    def generate_or_reuse_family(self, family_config: Dict[str, Any]) -> str:
        """Generate a new graph family or reuse existing one."""
        family_hash = self.get_family_hash(family_config)
        
        # Check if we can reuse this family configuration
        if self.config.reuse_families and family_hash in self.generated_families:
            family_id = self.generated_families[family_hash]
            print(f"  ♻️  Reusing graph family: {family_id}")
            return family_id
        
        # Generate new family
        print(f"  🔄 Generating new graph family...")
        
        # Create PreTrainingConfig for family generation
        pretraining_config = PreTrainingConfig(
            n_graphs=family_config['n_graphs'],
            n_extra_graphs_for_finetuning=family_config['n_extra_graphs'],
            universe_K=family_config['universe_K'],
            universe_homophily=family_config['universe_homophily'],
            universe_edge_density=family_config['universe_edge_density'],
            use_dccc_sbm=family_config['use_dccc_sbm'],
            degree_distribution=family_config['degree_distribution'],
            
            # Use default values for other parameters
            universe_feature_dim=32,
            min_n_nodes=family_config['min_n_nodes'],
            max_n_nodes=family_config['max_n_nodes'],
            min_communities=family_config['min_communities'],
            max_communities=family_config['max_communities'],
            
            # Dummy values for model config (not used for family generation)
            gnn_type='gcn',
            pretraining_task='link_prediction',
            
            graph_family_dir=os.path.join(self.output_dir, "graph_families"),
            seed=self.config.base_seed + len(self.generated_families)
        )
        
        # Generate family
        family_manager = GraphFamilyManager(pretraining_config)
        family_graphs, family_id = family_manager.generate_and_save_family(
            family_id=family_config['family_id']
        )
        
        # Store for reuse
        self.generated_families[family_hash] = family_id
        
        print(f"  ✅ Generated family: {family_id} ({len(family_graphs)} graphs)")
        
        return family_id
    
    def run_single_experiment(
        self,
        exp_id: int,
        rep: int,
        family_config: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single experiment with given configurations."""
        
        print(f"\nRunning experiment {exp_id} (repetition {rep})")
        print(f"Family config: {family_config}")
        print(f"Model config: {model_config}")
        
        try:
            # Generate or reuse graph family
            family_id = self.generate_or_reuse_family(family_config)
            
            # Create complete PreTrainingConfig
            pretraining_config = PreTrainingConfig(
                # Experiment setup
                output_dir=os.path.join(self.output_dir, f"exp_{exp_id:03d}"),
                experiment_name=f"{self.config.experiment_name}_exp_{exp_id:03d}",
                seed=self.config.base_seed + exp_id,
                
                # Family parameters
                n_graphs=family_config['n_graphs'],
                n_extra_graphs_for_finetuning=family_config['n_extra_graphs'],
                universe_K=family_config['universe_K'],
                universe_feature_dim=32,
                universe_edge_density=family_config['universe_edge_density'],
                universe_homophily=family_config['universe_homophily'],
                use_dccc_sbm=family_config['use_dccc_sbm'],
                degree_distribution=family_config['degree_distribution'],
                
                # Graph size parameters
                min_n_nodes=family_config['min_n_nodes'],
                max_n_nodes=family_config['max_n_nodes'],
                min_communities=family_config['min_communities'],
                max_communities=family_config['max_communities'],
                
                # Model parameters
                gnn_type=model_config.get('gnn_type', 'gcn'),  # Default to gcn if not specified
                pretraining_task=model_config['pretraining_task'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                
                # Transformer configuration
                transformer_type=model_config.get('transformer_type', 'graphormer'),
                run_transformers=model_config.get('run_transformers', False),
                transformer_num_heads=model_config.get('transformer_num_heads', 8),
                transformer_max_nodes=model_config.get('transformer_max_nodes', 200),
                transformer_max_path_length=model_config.get('transformer_max_path_length', 10),
                transformer_precompute_encodings=model_config.get('transformer_precompute_encodings', True),
                transformer_cache_encodings=model_config.get('transformer_cache_encodings', True),
                local_gnn_type=model_config.get('local_gnn_type', 'gcn'),
                global_model_type=model_config.get('global_model_type', 'transformer'),
                transformer_prenorm=model_config.get('transformer_prenorm', True),
                
                # Training parameters
                epochs=self.config.base_config.epochs,
                patience=self.config.patience,
                optimize_hyperparams=self.config.base_config.optimize_hyperparams,
                n_trials=self.config.base_config.n_trials,
                optimization_timeout=self.config.base_config.optimization_timeout,

                # Task-specific parameters
                negative_sampling_ratio=model_config.get('negative_sampling_ratio', self.config.base_config.negative_sampling_ratio),
                link_pred_loss=model_config.get('link_pred_loss', self.config.base_config.link_pred_loss),
                dgi_corruption_type=model_config.get('dgi_corruption_type', self.config.base_config.dgi_corruption_type),
                dgi_noise_std=model_config.get('dgi_noise_std', self.config.base_config.dgi_noise_std),
                dgi_perturb_rate=model_config.get('dgi_perturb_rate', self.config.base_config.dgi_perturb_rate),
                dgi_corruption_rate=model_config.get('dgi_corruption_rate', self.config.base_config.dgi_corruption_rate),
                graphmae_mask_rate=model_config.get('graphmae_mask_rate', self.config.base_config.graphmae_mask_rate),
                graphmae_replace_rate=model_config.get('graphmae_replace_rate', self.config.base_config.graphmae_replace_rate),
                graphmae_gamma=model_config.get('graphmae_gamma', self.config.base_config.graphmae_gamma),
                graphmae_decoder_type=model_config.get('graphmae_decoder_type', self.config.base_config.graphmae_decoder_type),
                graphmae_decoder_gnn_type=model_config.get('graphmae_decoder_gnn_type', self.config.base_config.graphmae_decoder_gnn_type),

                # Graph family management
                graph_family_dir=os.path.join(self.output_dir, "graph_families"),
                save_graph_family=True,
                
                # Model type for hyperparameter optimization
                model_type="transformer" if model_config.get('run_transformers', False) else "gnn"
            )
            
            # Run pre-training
            runner = PreTrainingRunner(pretraining_config)
            results = runner.run_pretraining_only(
                family_id=family_id,
                use_existing_family=True
            )
            
            # Compile complete result record
            experiment_result = {
                'exp_id': exp_id,
                'repetition': rep,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'family_config': family_config,
                'model_config': model_config,
                'family_id': family_id,
                'model_id': results['model_id'],
                'pretraining_results': results,
                'success': True
            }
            
            return experiment_result
            
        except Exception as e:
            print(f"❌ Experiment {exp_id} failed: {str(e)}")
            if not self.config.continue_on_failure:
                raise
            
            return {
                'exp_id': exp_id,
                'repetition': rep,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'family_config': family_config,
                'model_config': model_config,
                'error': str(e),
                'success': False
            }
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured experiments."""
        print(f"\n{'='*80}")
        print(f"STARTING MULTI-SSL EXPERIMENT SUITE")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Get all configurations
        family_configs = self.config.get_family_configurations()
        model_configs = self.config.get_model_configurations()
        
        total_experiments = len(family_configs) * len(model_configs) * self.config.n_repetitions
        
        print(f"Graph family configurations: {len(family_configs)}")
        print(f"Model configurations: {len(model_configs)}")
        print(f"Repetitions: {self.config.n_repetitions}")
        print(f"Total experiments: {total_experiments}")
        
        # Run experiments
        exp_id = 0
        
        with tqdm(total=total_experiments, desc="Running SSL experiments") as pbar:
            for family_config in family_configs:
                for model_config in model_configs:
                    for rep in range(self.config.n_repetitions):
                        
                        result = self.run_single_experiment(
                            exp_id=exp_id,
                            rep=rep,
                            family_config=family_config,
                            model_config=model_config
                        )
                        
                        if result:
                            if result.get('success', False):
                                self.all_results.append(result)
                            else:
                                self.failed_experiments.append(result)
                        
                        exp_id += 1
                        pbar.update(1)
                        
                        # Save intermediate results
                        self.save_intermediate_results()
        
        total_time = time.time() - start_time
        
        # Save final results
        final_results = self.save_final_results(total_time)
        
        # Generate summary
        self.print_summary(total_time)
        
        return final_results
    
    def save_intermediate_results(self):
        """Save intermediate results."""
        intermediate_data = {
            'successful_experiments': len(self.all_results),
            'failed_experiments': len(self.failed_experiments),
            'total_attempted': len(self.all_results) + len(self.failed_experiments),
            'generated_families': self.generated_families,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        intermediate_path = os.path.join(self.output_dir, "intermediate_status.json")
        with open(intermediate_path, 'w') as f:
            json.dump(intermediate_data, f, indent=2)
    
    def save_final_results(self, total_time: float) -> Dict[str, Any]:
        """Save final results and generate summary files."""
        
        # Compile final results
        final_results = {
            'experiment_config': {
                'experiment_name': self.config.experiment_name,
                'total_planned': self.config.get_total_experiments(),
                'family_configurations': len(self.config.get_family_configurations()),
                'model_configurations': len(self.config.get_model_configurations()),
                'repetitions': self.config.n_repetitions
            },
            'execution_summary': {
                'successful_experiments': len(self.all_results),
                'failed_experiments': len(self.failed_experiments),
                'total_attempted': len(self.all_results) + len(self.failed_experiments),
                'success_rate': len(self.all_results) / (len(self.all_results) + len(self.failed_experiments)) if (len(self.all_results) + len(self.failed_experiments)) > 0 else 0,
                'total_time_seconds': total_time,
                'avg_time_per_experiment': total_time / len(self.all_results) if self.all_results else 0
            },
            'generated_families': self.generated_families,
            'successful_results': self.all_results,
            'failed_experiments': self.failed_experiments,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save complete results
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Create summary table
        if self.all_results:
            self.create_summary_table()
        
        return final_results
    
    def create_summary_table(self):
        """Create a summary table of all successful experiments."""
        import pandas as pd
        
        # Extract data for table
        table_data = []
        
        for result in self.all_results:
            family_config = result['family_config']
            model_config = result['model_config']
            pretraining_results = result['pretraining_results']
            
            row = {
                'exp_id': result['exp_id'],
                'model_id': result['model_id'],
                'family_id': result['family_id'],
                'gnn_type': model_config['gnn_type'],
                'pretraining_task': model_config['pretraining_task'],
                'hidden_dim': model_config['hidden_dim'],
                'num_layers': model_config['num_layers'],
                'n_graphs': family_config['n_graphs'],
                'n_extra_graphs': family_config['n_extra_graphs'],
                'universe_K': family_config['universe_K'],
                'universe_homophily': family_config['universe_homophily'],
                'universe_edge_density': family_config['universe_edge_density'],
                'use_dccc_sbm': family_config['use_dccc_sbm'],
                'degree_distribution': family_config['degree_distribution'],
                'total_time': pretraining_results.get('total_time', 0)
            }
            
            # Add final metrics if available
            final_metrics = pretraining_results.get('final_metrics', {})
            if final_metrics:
                for metric, value in final_metrics.items():
                    row[f'final_{metric}'] = value
            
            table_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        csv_path = os.path.join(self.output_dir, "experiment_summary.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Summary table saved: {csv_path}")
    
    def print_summary(self, total_time: float):
        """Print experiment summary."""
        total_attempted = len(self.all_results) + len(self.failed_experiments)
        success_rate = len(self.all_results) / total_attempted if total_attempted > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"MULTI-SSL EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments attempted: {total_attempted}")
        print(f"Successful experiments: {len(self.all_results)}")
        print(f"Failed experiments: {len(self.failed_experiments)}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Average time per experiment: {total_time/total_attempted:.1f}s" if total_attempted > 0 else "N/A")
        print(f"Graph families generated: {len(self.generated_families)}")
        print(f"Results saved to: {self.output_dir}")
        
        # Method breakdown
        if self.all_results:
            print(f"\nSuccessful experiments by method:")
            method_counts = {}
            for result in self.all_results:
                model_config = result['model_config']
                method_key = f"{model_config['gnn_type']}+{model_config['pretraining_task']}"
                method_counts[method_key] = method_counts.get(method_key, 0) + 1
            
            for method, count in sorted(method_counts.items()):
                print(f"  {method}: {count}")


def run_clean_multi_experiments(config: CleanMultiExperimentConfig, continue_from_results: dict = None) -> Dict[str, Any]:
    """Convenience function to run multi-experiments."""
    runner = CleanMultiExperimentRunner(config, continue_from_results)
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