#!/usr/bin/env python3
"""
Multi-experiment runner for self-supervised learning.
Supports running multiple pre-training configurations across different graph families,
GNN types, and pre-training tasks.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import itertools
from tqdm import tqdm
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.config import PreTrainingConfig
from experiments.run_multi_inductive_experiments import ParameterRange
from experiments.inductive.experiment import PreTrainingRunner
from experiments.inductive.data import GraphFamilyManager, PreTrainedModelSaver


class SSLMultiExperimentConfig:
    """Configuration for multi-experiment SSL runs."""
    
    def __init__(
        self,
        output_dir: str = "multi_ssl_experiments",
        experiment_name: str = "ssl_sweep",
        
        # Base configuration
        base_config: PreTrainingConfig = None,
        
        # Parameter sweeps
        sweep_parameters: Dict[str, ParameterRange] = None,
        
        # Random parameters
        random_parameters: Dict[str, ParameterRange] = None,
        
        # Execution parameters
        n_repetitions: int = 1,
        continue_on_failure: bool = True,
        save_individual_results: bool = True,
        
        # Resource management
        max_concurrent_families: int = 1,  # Number of graph families to generate concurrently
        reuse_families: bool = True,  # Reuse graph families across experiments
        
        # Random seed management
        base_seed: int = 42,
        
        # Model management
        gnn_models: List[str] = None
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.base_config = base_config
        self.sweep_parameters = sweep_parameters or {}
        self.random_parameters = random_parameters or {}
        self.n_repetitions = n_repetitions
        self.continue_on_failure = continue_on_failure
        self.save_individual_results = save_individual_results
        self.max_concurrent_families = max_concurrent_families
        self.reuse_families = reuse_families
        self.base_seed = base_seed
        self.gnn_models = gnn_models or []
    
    def get_family_configurations(self) -> List[Dict[str, Any]]:
        """Get all unique graph family configurations."""
        # Get all combinations of sweep parameters
        sweep_values = {}
        for param, param_range in self.sweep_parameters.items():
            if param_range.is_sweep:
                sweep_values[param] = param_range.get_sweep_values()
            else:
                sweep_values[param] = [param_range.min_val]
        
        # Generate all combinations
        param_names = list(sweep_values.keys())
        param_values = list(sweep_values.values())
        combinations = list(itertools.product(*param_values))
        
        family_configs = []
        for i, combo in enumerate(combinations):
            # Start with base configuration values
            family_config = {
                'family_id': f"family_{self.experiment_name}_{i:03d}",
                'n_graphs': self.base_config.n_graphs,
                'n_extra_graphs': self.base_config.n_extra_graphs_for_finetuning,
                'universe_K': self.base_config.universe_K,
                'universe_feature_dim': self.base_config.universe_feature_dim,
                'use_dccc_sbm': self.base_config.use_dccc_sbm,
                'degree_distribution': self.base_config.degree_distribution,
                'min_n_nodes': self.base_config.min_n_nodes,
                'max_n_nodes': self.base_config.max_n_nodes,
                'min_communities': self.base_config.min_communities,
                'max_communities': self.base_config.max_communities
            }
            
            # Add sweep parameter values
            family_config.update(dict(zip(param_names, combo)))
            
            family_configs.append(family_config)
        
        return family_configs
    
    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """Get all model configurations."""
        # For SSL, we'll use the base config's model parameters
        if not self.base_config:
            return []
            
        model_configs = []
        for gnn_type in self.gnn_models:
            model_config = {
                'gnn_type': gnn_type,
                'pretraining_task': self.base_config.pretraining_task,
                'hidden_dim': self.base_config.hidden_dim,
                'num_layers': self.base_config.num_layers
            }
            model_configs.append(model_config)
        
        return model_configs
    
    def get_total_experiments(self) -> int:
        """Get total number of experiments."""
        n_families = len(self.get_family_configurations())
        n_models = len(self.get_model_configurations())
        return n_families * n_models * self.n_repetitions


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
            'gnn_models': self.config.gnn_models
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
            min_n_nodes=80,
            max_n_nodes=120,
            min_communities=3,
            max_communities=7,
            
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
        family_config: Dict[str, Any],
        model_config: Dict[str, Any],
        rep: int,
        exp_id: int
    ) -> Optional[Dict[str, Any]]:
        """Run a single SSL experiment."""
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_id:03d} (Rep {rep+1}/{self.config.n_repetitions})")
        print(f"{'='*60}")
        print(f"Family: {family_config['family_id']}")
        print(f"Model: {model_config['gnn_type']} + {model_config['pretraining_task']}")
        print(f"{'='*60}")
        
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
                min_n_nodes=80,
                max_n_nodes=120,
                min_communities=3,
                max_communities=7,
                
                # Model parameters
                gnn_type=model_config['gnn_type'],
                pretraining_task=model_config['pretraining_task'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                
                # Training parameters
                epochs=self.config.base_config.epochs,
                optimize_hyperparams=self.config.base_config.optimize_hyperparams,
                n_trials=self.config.base_config.n_trials,
                optimization_timeout=self.config.base_config.optimization_timeout,
                
                # Graph family management
                graph_family_dir=os.path.join(self.output_dir, "graph_families"),
                save_graph_family=True
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
            
            print(f"✅ Experiment {exp_id:03d} completed successfully")
            print(f"   Model ID: {results['model_id']}")
            print(f"   Family ID: {family_id}")
            
            return experiment_result
            
        except Exception as e:
            error_info = {
                'exp_id': exp_id,
                'repetition': rep,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'family_config': family_config,
                'model_config': model_config,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }
            
            print(f"❌ Experiment {exp_id:03d} failed: {str(e)}")
            
            if not self.config.continue_on_failure:
                raise
            
            return error_info
    
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
                            family_config=family_config,
                            model_config=model_config,
                            rep=rep,
                            exp_id=exp_id
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple SSL experiments')
    
    # === EXPERIMENT SETUP ===
    parser.add_argument('--output_dir', type=str, default='multi_ssl_experiments',
                        help='Base output directory')
    parser.add_argument('--experiment_name', type=str, default='ssl_sweep',
                        help='Name for this experiment sweep')
    
    # === EXECUTION PARAMETERS ===
    parser.add_argument('--n_repetitions', type=int, default=1,
                        help='Number of repetitions per configuration')
    parser.add_argument('--continue_on_failure', action='store_true', default=True,
                        help='Continue experiments even if some fail')
    parser.add_argument('--reuse_families', action='store_true', default=True,
                        help='Reuse graph families across experiments')
    
    # === GRAPH FAMILY PARAMETERS ===
    parser.add_argument('--n_graphs', type=int, default=50,
                        help='Number of graphs per family')
    parser.add_argument('--n_extra_graphs', type=int, default=30,
                        help='Number of extra graphs for finetuning')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=120,
                        help='Maximum nodes per graph')
    parser.add_argument('--min_communities', type=int, default=3,
                        help='Minimum number of communities')
    parser.add_argument('--max_communities', type=int, default=7,
                        help='Maximum number of communities')
    parser.add_argument('--universe_K', type=int, default=10,
                        help='Number of communities in universe')
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM instead of DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution for DCCC-SBM')
    
    # === SWEEP PARAMETERS ===
    # Homophily sweep
    parser.add_argument('--homophily_min', type=float, default=0.2,
                        help='Minimum homophily value for sweep')
    parser.add_argument('--homophily_max', type=float, default=0.8,
                        help='Maximum homophily value for sweep')
    parser.add_argument('--homophily_step', type=float, default=0.4,
                        help='Step size for homophily sweep')
    
    # Density sweep
    parser.add_argument('--density_min', type=float, default=0.05,
                        help='Minimum edge density value for sweep')
    parser.add_argument('--density_max', type=float, default=0.15,
                        help='Maximum edge density value for sweep')
    parser.add_argument('--density_step', type=float, default=0.10,
                        help='Step size for density sweep')
    
    # === MODEL PARAMETERS ===
    parser.add_argument('--gnn_types', type=str, nargs='+', 
                        default=['gcn'],
                        choices=['gcn', 'sage', 'gat'],
                        help='GNN types to test')
    parser.add_argument('--pretraining_task', type=str,
                        default='link_prediction',
                        choices=['link_prediction', 'contrastive'],
                        help='Pre-training task to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    
    # === TASK-SPECIFIC HYPERPARAMETERS ===
    # Link prediction parameters
    parser.add_argument('--neg_sampling_ratio_min', type=float, default=1.0,
                        help='Minimum negative sampling ratio for link prediction')
    parser.add_argument('--neg_sampling_ratio_max', type=float, default=5.0,
                        help='Maximum negative sampling ratio for link prediction')
    parser.add_argument('--neg_sampling_ratio_step', type=float, default=5.0,
                        help='Step size for negative sampling ratio sweep')
    parser.add_argument('--link_pred_threshold_min', type=float, default=0.3,
                        help='Minimum link prediction threshold')
    parser.add_argument('--link_pred_threshold_max', type=float, default=0.7,
                        help='Maximum link prediction threshold')
    parser.add_argument('--link_pred_threshold_step', type=float, default=0.4,
                        help='Step size for link prediction threshold sweep')
    
    # Contrastive learning parameters
    parser.add_argument('--temperature_min', type=float, default=0.1,
                        help='Minimum temperature for contrastive learning')
    parser.add_argument('--temperature_max', type=float, default=1.0,
                        help='Maximum temperature for contrastive learning')
    parser.add_argument('--temperature_step', type=float, default=0.5,
                        help='Step size for temperature sweep')
    parser.add_argument('--aug_strength_min', type=float, default=0.1,
                        help='Minimum augmentation strength')
    parser.add_argument('--aug_strength_max', type=float, default=0.5,
                        help='Maximum augmentation strength')
    parser.add_argument('--aug_strength_step', type=float, default=0.25,
                        help='Step size for augmentation strength sweep')
    
    # === TRAINING PARAMETERS ===
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs for pre-training')
    parser.add_argument('--optimize_hyperparams', action='store_true', default=True,
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--optimization_timeout', type=int, default=1200,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # === RANDOM SEED ===
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_preset_config(preset: str, args) -> SSLMultiExperimentConfig:
    """Create configuration based on preset."""
    
    if preset == 'quick':
        # Quick test configuration
        return SSLMultiExperimentConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            
            # Minimal parameter sweep
            n_graphs=[20],
            n_extra_graphs=[10],
            universe_K_values=[5],
            universe_homophily_values=[0.7],
            universe_edge_density_values=[0.1],
            use_dccc_sbm_values=[False],
            degree_distribution_values=['power_law'],
            
            gnn_types=['gcn'],
            pretraining_tasks=['link_prediction'],
            hidden_dims=[64],
            num_layers_values=[2],
            
            epochs=100,
            optimize_hyperparams=False,
            n_repetitions=1
        )
    
    elif preset == 'standard':
        # Standard configuration for thorough testing
        return SSLMultiExperimentConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            
            # Moderate parameter sweep
            n_graphs=[40, 60],
            n_extra_graphs=[20, 30],
            universe_K_values=[8, 10],
            universe_homophily_values=[0.6, 0.8],
            universe_edge_density_values=[0.08, 0.12],
            use_dccc_sbm_values=[False, True],
            degree_distribution_values=['power_law', 'exponential'],
            
            gnn_types=['sage'],
            pretraining_tasks=['link_prediction', 'contrastive'],
            hidden_dims=[128, 256],
            num_layers_values=[2, 3],
            
            epochs=args.epochs,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
            n_repetitions=args.n_repetitions
        )
    
    elif preset == 'comprehensive':
        # Comprehensive configuration for full evaluation
        return SSLMultiExperimentConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            
            # Full parameter sweep
            n_graphs=[30, 50, 70],
            n_extra_graphs=[20, 30, 40],
            universe_K_values=[6, 8, 10, 12],
            universe_homophily_values=[0.5, 0.7, 0.9],
            universe_edge_density_values=[0.06, 0.1, 0.14],
            use_dccc_sbm_values=[False, True],
            degree_distribution_values=['standard', 'power_law', 'exponential', 'uniform'],
            
            gnn_types=['gcn', 'sage', 'gat'],
            pretraining_tasks=['link_prediction', 'contrastive'],
            hidden_dims=[64, 128, 256],
            num_layers_values=[2, 3, 4],
            
            epochs=args.epochs,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
            n_repetitions=args.n_repetitions
        )
    
    elif preset == 'custom':
        # Custom configuration from command line arguments
        use_dccc_sbm_values = [s.lower() == 'true' for s in args.use_dccc_sbm_values]
        
        return SSLMultiExperimentConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            
            # Use command line parameters
            n_graphs=[50],
            n_extra_graphs=[30],
            universe_K_values=args.universe_K_values,
            universe_homophily_values=[0.6, 0.8],
            universe_edge_density_values=[0.08, 0.12],
            use_dccc_sbm_values=use_dccc_sbm_values,
            degree_distribution_values=['power_law'],
            
            gnn_types=args.gnn_types,
            pretraining_tasks=args.pretraining_tasks,
            hidden_dims=[128],
            num_layers_values=[2, 3],
            
            epochs=args.epochs,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
            n_repetitions=args.n_repetitions
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}")


def create_custom_experiment(args) -> SSLMultiExperimentConfig:
    """Create a custom experiment configuration from arguments."""
    from experiments.inductive.config import PreTrainingConfig
    
    # Base configuration
    base_config = PreTrainingConfig(
        n_graphs=args.n_graphs,
        n_extra_graphs_for_finetuning=args.n_extra_graphs,
        universe_K=args.universe_K,
        universe_feature_dim=32,
        universe_edge_density=args.density_min,
        universe_homophily=args.homophily_min,
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,
        
        # Graph size parameters
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=args.min_communities,
        max_communities=args.max_communities,
        
        # Model parameters - use first values as base
        gnn_type=args.gnn_types[0],  # Will be overridden for each model
        pretraining_task=args.pretraining_task,  # Single task
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        
        # Training parameters
        epochs=args.epochs,
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        
        # Graph family management
        graph_family_dir=os.path.join(args.output_dir, "graph_families"),
        save_graph_family=True,
        
        # Seed
        seed=args.seed
    )
    
    # Define sweep parameters
    sweep_parameters = {
        'universe_homophily': ParameterRange(
            min_val=args.homophily_min,
            max_val=args.homophily_max,
            step=args.homophily_step,
            is_sweep=True
        ),
        'universe_edge_density': ParameterRange(
            min_val=args.density_min,
            max_val=args.density_max,
            step=args.density_step,
            is_sweep=True
        )
    }
    
    # Add pre-training task specific hyperparameters only for the selected task
    if args.pretraining_task == 'link_prediction':
        sweep_parameters.update({
            'negative_sampling_ratio': ParameterRange(
                min_val=args.neg_sampling_ratio_min,
                max_val=args.neg_sampling_ratio_max,
                step=args.neg_sampling_ratio_step,
                is_sweep=True
            ),
            'link_prediction_threshold': ParameterRange(
                min_val=args.link_pred_threshold_min,
                max_val=args.link_pred_threshold_max,
                step=args.link_pred_threshold_step,
                is_sweep=True
            )
        })
    elif args.pretraining_task == 'contrastive':
        sweep_parameters.update({
            'temperature': ParameterRange(
                min_val=args.temperature_min,
                max_val=args.temperature_max,
                step=args.temperature_step,
                is_sweep=True
            ),
            'augmentation_strength': ParameterRange(
                min_val=args.aug_strength_min,
                max_val=args.aug_strength_max,
                step=args.aug_strength_step,
                is_sweep=True
            )
        })
    
    # Define random parameters
    random_parameters = {
        'homophily_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.2,
            distribution="uniform",
            is_sweep=False
        ),
        'density_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.05,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.1,
            distribution="uniform",
            is_sweep=False
        ),
        'cluster_count_factor': ParameterRange(
            min_val=0.5,
            max_val=1.5,
            distribution="uniform",
            is_sweep=False
        ),
        'cluster_variance': ParameterRange(
            min_val=0.01,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'center_variance': ParameterRange(
            min_val=0.01,
            max_val=1.5,
            distribution="uniform",
            is_sweep=False
        ),
        'assignment_skewness': ParameterRange(
            min_val=0.0,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'community_exclusivity': ParameterRange(
            min_val=0.6,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'universe_randomness_factor': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    # Add DCCC-specific random parameters if needed
    if args.use_dccc_sbm:
        random_parameters.update({
            'community_imbalance_range_width': ParameterRange(
                min_val=0.0,
                max_val=0.5,
                distribution="uniform",
                is_sweep=False
            ),
            'degree_separation_range_width': ParameterRange(
                min_val=0.3,
                max_val=1.0,
                distribution="uniform",
                is_sweep=False
            )
        })
    
    return SSLMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=args.n_repetitions,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        continue_on_failure=True,
        save_individual_results=True,
        reuse_families=args.reuse_families,
        gnn_models=args.gnn_types  # List of GNN models to test
    )


def main():
    """Main function to run multi-SSL experiments."""
    args = parse_args()
    
    print("MULTI-EXPERIMENT SSL BENCHMARK SUITE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment name: {args.experiment_name}")
    
    try:
        # Create configuration
        config = create_custom_experiment(args)
        
        print(f"\nExperiment Configuration:")
        print(f"  Total experiments: {config.get_total_experiments()}")
        print(f"  Family configurations: {len(config.get_family_configurations())}")
        print(f"  Model configurations: {len(config.get_model_configurations())}")
        print(f"  Repetitions: {config.n_repetitions}")
        print(f"  Reuse families: {config.reuse_families}")
        print(f"  Hyperparameter optimization: {config.base_config.optimize_hyperparams}")
        
        print(f"\nSweep parameters ({len(config.sweep_parameters)}):")
        for param, param_range in config.sweep_parameters.items():
            if hasattr(param_range, 'get_sweep_values'):
                try:
                    values = param_range.get_sweep_values()
                    print(f"  {param}: {len(values)} values from {min(values)} to {max(values)}")
                except:
                    print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
            else:
                print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
        
        print(f"\nRandom parameters ({len(config.random_parameters)}):")
        for param, param_range in config.random_parameters.items():
            print(f"  {param}: [{param_range.min_val}, {param_range.max_val}] ({param_range.distribution})")
        
        # Confirm before starting large experiments
        if config.get_total_experiments() > 10:
            response = input(f"\nThis will run {config.get_total_experiments()} experiments. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 0
        
        # Run experiments
        runner = SSLMultiExperimentRunner(config)
        results = runner.run_all_experiments()
        
        print(f"\n✅ Multi-SSL experiment suite completed!")
        print(f"Results directory: {runner.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nMulti-SSL experiment suite interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nMulti-SSL experiment suite failed with error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)