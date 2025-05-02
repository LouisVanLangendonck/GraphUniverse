"""
Run multiple MMSB Graph Learning Experiments with different parameters.

This script runs a series of experiments using different parameter combinations
to analyze the impact of various graph generation parameters on model performance.
Includes hyperparameter optimization for each model.
"""

import os
import sys
import numpy as np
import pandas as pd
import itertools
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime
from copy import deepcopy

# Import core experiment functionality
from experiments.core import (
    Experiment,
    ExperimentConfig,
    analyze_results,
    plot_model_comparison
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple MMSB Graph Learning Experiments')
    
    # Parameters to vary
    parser.add_argument('--vary', type=str, nargs='+', default=['homophily'],
                        choices=[
                            'num_communities',
                            'num_nodes',
                            'feature_dim',
                            'edge_density',
                            'homophily', 
                            'randomness_factor',
                            'overlap_density',
                            'min_connection_strength',
                            'min_component_size',
                            'degree_heterogeneity',
                            'indirect_influence',
                            'mixed_membership',
                            # Feature regime parameters
                            'regimes_per_community',
                            'intra_community_regime_similarity',
                            'inter_community_regime_similarity',
                            'feature_regime_balance'
                        ],
                        help='Parameters to vary across experiments')
    
    # Add mixed membership flag
    parser.add_argument('--mixed_membership', action='store_true',
                        help='Enable mixed membership model')
    parser.add_argument('--no_mixed_membership', action='store_false', dest='mixed_membership',
                        help='Disable mixed membership model (use standard SBM)')
    parser.set_defaults(mixed_membership=False)
    
    # Task selection
    parser.add_argument('--tasks', type=str, nargs='+', default=['regime'], #, 'regime', 'role'],
                        choices=['community', 'regime', 'role'],
                        help='Learning tasks to run')
    
    # Feature regime parameters
    parser.add_argument('--regimes_per_community', type=int, default=4,
                        help='Number of feature regimes per community')
    parser.add_argument('--intra_community_regime_similarity', type=float, default=0.2,
                        help='How similar regimes within same community should be (0-1)')
    parser.add_argument('--inter_community_regime_similarity', type=float, default=0.9,
                        help='How similar regimes between communities should be (0-1)')
    parser.add_argument('--feature_regime_balance', type=float, default=0.3,
                        help='How evenly regimes are distributed within communities (0-1)')
    
    # Feature regime task parameters
    parser.add_argument('--regime_task_min_hop', type=int, default=2,
                        help='Minimum hop distance for regime task neighborhood analysis')
    parser.add_argument('--regime_task_max_hop', type=int, default=3,
                        help='Maximum hop distance for regime task neighborhood analysis')
    parser.add_argument('--regime_task_n_labels', type=int, default=5,
                        help='Number of labels to generate for regime task')
    parser.add_argument('--regime_task_min_support', type=float, default=0.1,
                        help='Minimum support for regime task rules')
    parser.add_argument('--regime_task_max_rules_per_label', type=int, default=3,
                        help='Maximum number of rules per label for regime task')
    
    # Role task parameters
    parser.add_argument('--role_task_max_motif_size', type=int, default=3,
                        help='Maximum motif size for role analysis')
    parser.add_argument('--role_task_n_roles', type=int, default=5,
                        help='Number of structural roles to discover')
    
    # Model selection and training parameters
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn'], #, 'sage'],
                        choices=['gcn', 'gat', 'sage'],
                        help='Types of GNN models to run')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip GNN models')
    parser.add_argument('--skip_mlp', action='store_true',
                        help='Skip MLP model')
    parser.add_argument('--skip_rf', action='store_true',
                        help='Skip Random Forest model')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping in neural models')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of epochs for neural models')
    
    # Hyperparameter optimization parameters
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization for each model')
    parser.add_argument('--n_trials', type=int, default=15,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--opt_timeout', type=int, default=300,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Parameter ranges
    parser.add_argument('--num_communities_range', type=int, nargs=3, default=[5, 15, 5],
                        help='Range for num_communities (start, end, step)')
    parser.add_argument('--num_nodes_range', type=int, nargs=3, default=[60, 160, 50],
                        help='Range for num_nodes (start, end, step)')
    parser.add_argument('--feature_dim_range', type=int, nargs=3, default=[32, 32, 0],
                        help='Range for feature_dim (start, end, step)')
    parser.add_argument('--edge_density_range', type=float, nargs=3, default=[0.02, 0.15, 0.015],
                        help='Range for overall edge density (start, end, step)')
    parser.add_argument('--homophily_range', type=float, nargs=3, default=[0.0, 1.0, 0.1],
                        help='Range for homophily - controls intra/inter community ratio (0=equal, 1=max homophily)')
    parser.add_argument('--randomness_factor_range', type=float, nargs=3, default=[0.0, 1.0, 0.25],
                        help='Range for randomness_factor (start, end, step)')
    parser.add_argument('--overlap_density_range', type=float, nargs=3, default=[0.0, 0.5, 0.25],
                        help='Range for overlap_density (start, end, step)')
    parser.add_argument('--min_connection_strength_range', type=float, nargs=3, default=[0.02, 0.1, 0.02],
                        help='Range for min_connection_strength (start, end, step)')
    parser.add_argument('--min_component_size_range', type=int, nargs=3, default=[2, 10, 4],
                        help='Range for min_component_size (start, end, step)')
    parser.add_argument('--degree_heterogeneity_range', type=float, nargs=3, default=[0, 1, 0.2],
                        help='Range for degree_heterogeneity (start, end, step)')
    parser.add_argument('--indirect_influence_range', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help='Range for indirect_influence (start, end, step)')
    
    # Feature regime parameter ranges
    parser.add_argument('--regimes_per_community_range', type=int, nargs=3, default=[1, 3, 1],
                        help='Range for regimes_per_community (start, end, step)')
    parser.add_argument('--intra_community_regime_similarity_range', type=float, nargs=3, default=[0.3, 1.0, 0.05],
                        help='Range for intra_community_regime_similarity (start, end, step)')
    parser.add_argument('--inter_community_regime_similarity_range', type=float, nargs=3, default=[0.85, 1.0, 0.05],
                        help='Range for inter_community_regime_similarity (start, end, step)')
    parser.add_argument('--feature_regime_balance_range', type=float, nargs=3, default=[0.1, 0.9, 0.2],
                        help='Range for feature_regime_balance (start, end, step)')
    
    # Experiment control
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Number of times to repeat each parameter combination')
    parser.add_argument('--results_dir', type=str, default='single_graph_multiple_results',
                        help='Directory to save all experiment results')
    
    args = parser.parse_args()
    
    # Set run flags based on skip flags
    args.run_gnn = not args.skip_gnn
    args.run_mlp = not args.skip_mlp
    args.run_rf = not args.skip_rf
    
    # Ensure GNN types are properly set
    if args.run_gnn and not args.gnn_types:
        args.gnn_types = ['gcn']  # Default to GCN if no types specified
    
    return args

def generate_parameter_combinations(args) -> List[Dict[str, Any]]:
    """Generate all parameter combinations to test."""
    param_ranges = {}
    
    for param in args.vary:
        range_attr = f"{param}_range"
        if hasattr(args, range_attr):
            start, end, step = getattr(args, range_attr)
            if isinstance(start, int):
                values = list(range(start, end + 1, step))
            else:
                values = np.arange(start, end + step/2, step).tolist()  # Add step/2 to include end value
            param_ranges[param] = values
        elif param == 'mixed_membership':  # Special case for boolean parameter
            # For now, only test non-mixed membership case!!
            param_ranges[param] = [False] # [True, False]
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    # Convert to list of dictionaries
    param_dicts = []
    for combo in combinations:
        param_dict = {name: value for name, value in zip(param_names, combo)}
        param_dicts.append(param_dict)

    # Print how many combinations are being tested
    print(f"Running {len(param_dicts)} parameter combinations")
    
    return param_dicts

def run_experiments(args):
    """Run all experiments with different parameter combinations."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate parameter combinations for varied parameters
    param_combinations = generate_parameter_combinations(args)
    print(f"Running {len(param_combinations)} parameter combinations, {args.n_repeats} times each")
    
    # Store all results
    all_results = []
    
    # Get all possible parameters and their ranges
    all_param_ranges = {
        'num_communities': args.num_communities_range,
        'num_nodes': args.num_nodes_range,
        'feature_dim': args.feature_dim_range,
        'edge_density': args.edge_density_range,  # Now controls overall density
        'homophily': args.homophily_range,       # Now controls intra/inter ratio
        'randomness_factor': args.randomness_factor_range,
        'overlap_density': args.overlap_density_range,
        'min_connection_strength': args.min_connection_strength_range,
        'min_component_size': args.min_component_size_range,
        'degree_heterogeneity': args.degree_heterogeneity_range,
        'indirect_influence': args.indirect_influence_range,
        # Feature regime parameters
        'regimes_per_community': args.regimes_per_community_range,
        'intra_community_regime_similarity': args.intra_community_regime_similarity_range,
        'inter_community_regime_similarity': args.inter_community_regime_similarity_range,
        'feature_regime_balance': args.feature_regime_balance_range
    }
    
    # Run experiments
    for i, params in enumerate(param_combinations):
        print(f"\nParameter combination {i+1}/{len(param_combinations)}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        for repeat in range(args.n_repeats):
            print(f"\nRepeat {repeat+1}/{args.n_repeats}")
            
            # Create experiment config
            config = ExperimentConfig()
            
            # Set the varied parameters
            for param, value in params.items():
                setattr(config, param, value)
            
            # Randomize non-varied parameters
            for param, range_vals in all_param_ranges.items():
                if param not in args.vary:
                    start, end, step = range_vals
                    if isinstance(start, int):
                        random_value = np.random.randint(start, end + 1)
                    else:
                        random_value = np.random.uniform(start, end)
                    setattr(config, param, random_value)
            
            # Set task parameters
            config.tasks = args.tasks
            config.regime_task_min_hop = args.regime_task_min_hop
            config.regime_task_max_hop = args.regime_task_max_hop
            config.regime_task_n_labels = args.regime_task_n_labels
            config.regime_task_min_support = args.regime_task_min_support
            config.regime_task_max_rules_per_label = args.regime_task_max_rules_per_label
            config.role_task_max_motif_size = args.role_task_max_motif_size
            config.role_task_n_roles = args.role_task_n_roles
            
            # Set model parameters
            config.gnn_types = args.gnn_types
            config.run_gnn = args.run_gnn
            config.run_mlp = args.run_mlp
            config.run_rf = args.run_rf
            config.patience = args.patience
            config.epochs = args.epochs
            
            # Set hyperparameter optimization parameters
            config.optimize_hyperparams = args.optimize_hyperparams
            config.n_trials = args.n_trials
            config.opt_timeout = args.opt_timeout
            
            # Set output directory
            config.output_dir = os.path.join(results_dir, f"combo_{i}_repeat_{repeat}")
            
            print(f"Starting experiment with parameters:")
            for param in all_param_ranges.keys():
                print(f"  {param}: {getattr(config, param)}")
            
            # Run experiment
            print("Generating graph...")
            experiment = Experiment(config)
            results = experiment.run()
            
            # Extract results
            result = {
                "experiment_id": f"{i}_{repeat}",
                "combination": i,
                "repeat": repeat,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            
            # Store input parameters
            input_params = [
                'num_communities',
                'num_nodes',
                'feature_dim',
                'edge_density',
                'homophily',
                'randomness_factor',
                'min_connection_strength',
                'min_component_size',
                'indirect_influence',
                # Feature regime parameters
                'regimes_per_community',
                'intra_community_regime_similarity',
                'inter_community_regime_similarity',
                'feature_regime_balance'
            ]
            
            for param in input_params:
                if hasattr(config, param):
                    result[param] = getattr(config, param)
            
            # Store real graph properties
            if hasattr(results["graph_sample"], "real_graph_properties"):
                real_props = results["graph_sample"].real_graph_properties
                for key, value in real_props.items():
                    result[key] = value
            else:
                # If real graph properties are not in graph_sample, try to get them from results
                real_props = results.get("real_graph_properties", {})
                for key, value in real_props.items():
                    result[key] = value
            
            # Add model metrics for each task
            for task_name, task_results in results["model_results"].items():
                for model_name, model_result in task_results.items():
                    # Basic metrics
                    result[f"{task_name}_{model_name}_accuracy"] = model_result.get('test_acc', 0)
                    result[f"{task_name}_{model_name}_train_time"] = model_result.get('train_time', 0)
                    
                    # Detailed metrics
                    metrics = model_result.get('metrics', {})
                    for metric_type in ['metrics_macro', 'metrics_micro', 'metrics_weighted']:
                        if metric_type in metrics:
                            for metric_name, metric_value in metrics[metric_type].items():
                                result[f"{task_name}_{model_name}_{metric_type}_{metric_name}"] = metric_value
                    
                    # Training history
                    if 'history' in model_result:
                        history = model_result['history']
                        result[f"{task_name}_{model_name}_final_train_loss"] = history['train_loss'][-1]
                        result[f"{task_name}_{model_name}_final_val_loss"] = history['val_loss'][-1]
                        result[f"{task_name}_{model_name}_best_val_acc"] = max(history['val_acc'])
                        result[f"{task_name}_{model_name}_epochs_trained"] = len(history['train_loss'])
                    
                    # Hyperparameter optimization results
                    if 'hyperopt_results' in model_result:
                        hyperopt = model_result['hyperopt_results']
                        result[f"{task_name}_{model_name}_best_val_score"] = hyperopt.get('best_value', 0)
                        result[f"{task_name}_{model_name}_n_trials"] = hyperopt.get('n_trials', 0)
                        
                        # Store best hyperparameters
                        best_params = hyperopt.get('best_params', {})
                        for param_name, param_value in best_params.items():
                            result[f"{task_name}_{model_name}_param_{param_name}"] = param_value
            
            all_results.append(result)
            
            # Save all results to CSV after each experiment
            df = pd.DataFrame(all_results)
            
            # Save both CSV and JSON for flexibility
            csv_path = os.path.join(results_dir, "all_results.csv")
            json_path = os.path.join(results_dir, "all_results.json")
            
            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient='records', indent=2)
            
            # Also save a metadata file with experiment configuration
            metadata = {
                "timestamp": timestamp,
                "varied_parameters": args.vary,
                "parameter_ranges": {k: v for k, v in all_param_ranges.items()},
                "tasks": args.tasks,
                "model_types": {
                    "gnn": args.gnn_types if args.run_gnn else [],
                    "mlp": not args.skip_mlp,
                    "rf": not args.skip_rf
                },
                "training_params": {
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "n_repeats": args.n_repeats
                },
                "hyperparameter_optimization": {
                    "enabled": args.optimize_hyperparams,
                    "n_trials": args.n_trials,
                    "timeout": args.opt_timeout
                },
                "task_params": {
                    "regime": {
                        "min_hop": args.regime_task_min_hop,
                        "max_hop": args.regime_task_max_hop,
                        "n_labels": args.regime_task_n_labels,
                        "min_support": args.regime_task_min_support,
                        "max_rules_per_label": args.regime_task_max_rules_per_label
                    },
                    "role": {
                        "max_motif_size": args.role_task_max_motif_size,
                        "n_roles": args.role_task_n_roles
                    }
                }
            }
            
            with open(os.path.join(results_dir, "experiment_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"\nAll experiments complete. Results saved to {results_dir}")
    return all_results

def main():
    """Main function to run multiple experiments."""
    args = parse_args()
    results = run_experiments(args)
    
    if results:
        # Create analysis directory
        analysis_dir = os.path.join(args.results_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Analyze results
        analysis_df = analyze_results(results, args)
        
        if analysis_df is not None:
            # Generate comparison plots
            plot_model_comparison(analysis_df, args, analysis_dir)
        
        print("\nExperiment series complete!")
        print(f"Results and analysis saved to {args.results_dir}")
    else:
        print("\nNo successful experiments to analyze.")
    
    # Properly exit
    sys.exit(0)

if __name__ == "__main__":
    main()