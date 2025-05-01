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
    parser.add_argument('--vary', type=str, nargs='+', default=['feature_signal', 'homophily'],
                        choices=[
                            'num_communities',
                            'num_nodes',
                            'feature_dim',
                            'edge_density',
                            'homophily', 
                            'feature_signal',
                            'randomness_factor',
                            'overlap_density',
                            'min_connection_strength',
                            'min_component_size',
                            'degree_heterogeneity',
                            'indirect_influence'
                        ],
                        help='Parameters to vary across experiments')
    
    # Model selection and training parameters
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn', 'sage'],#['gat', 'gcn', 'sage'],
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
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--opt_timeout', type=int, default=300,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Parameter ranges
    parser.add_argument('--num_communities_range', type=int, nargs=3, default=[10, 15, 5],
                        help='Range for num_communities (start, end, step)')
    parser.add_argument('--num_nodes_range', type=int, nargs=3, default=[80, 180, 50],
                        help='Range for num_nodes (start, end, step)')
    parser.add_argument('--feature_dim_range', type=int, nargs=3, default=[32, 32, 0],
                        help='Range for feature_dim (start, end, step)')
    parser.add_argument('--edge_density_range', type=float, nargs=3, default=[0.01, 0.7, 0.015],
                        help='Range for overall edge density (start, end, step)')
    parser.add_argument('--homophily_range', type=float, nargs=3, default=[0.0, 1.0, 0.2],
                        help='Range for homophily - controls intra/inter community ratio (0=equal, 1=max homophily)')
    parser.add_argument('--feature_signal_range', type=float, nargs=3, default=[0.00, 0.08, 0.02],
                        help='Range for feature_signal (start, end, step)')
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
    parser.add_argument('--indirect_influence_range', type=float, nargs=3, default=[0.0, 0.5, 0.15],
                        help='Range for indirect_influence (start, end, step)')
    
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
        args.gnn_types = ['gat']  # Default to GAT if no types specified
    
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
        'feature_signal': args.feature_signal_range,
        'randomness_factor': args.randomness_factor_range,
        'overlap_density': args.overlap_density_range,
        'min_connection_strength': args.min_connection_strength_range,
        'min_component_size': args.min_component_size_range,
        'degree_heterogeneity': args.degree_heterogeneity_range,
        'indirect_influence': args.indirect_influence_range
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
                'feature_signal',
                'randomness_factor',
                'min_connection_strength',
                'min_component_size',
                'indirect_influence'
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
            
            # Add model metrics
            for model_name, model_result in results["model_results"].items():
                # Basic metrics
                result[f"{model_name}_accuracy"] = model_result.get('test_acc', 0)
                result[f"{model_name}_train_time"] = model_result.get('train_time', 0)
                
                # Detailed metrics
                metrics = model_result.get('metrics', {})
                for metric_type in ['metrics_macro', 'metrics_micro', 'metrics_weighted']:
                    if metric_type in metrics:
                        for metric_name, metric_value in metrics[metric_type].items():
                            result[f"{model_name}_{metric_type}_{metric_name}"] = metric_value
                
                # Training history
                if 'history' in model_result:
                    history = model_result['history']
                    result[f"{model_name}_final_train_loss"] = history['train_loss'][-1]
                    result[f"{model_name}_final_val_loss"] = history['val_loss'][-1]
                    result[f"{model_name}_best_val_acc"] = max(history['val_acc'])
                    result[f"{model_name}_epochs_trained"] = len(history['train_loss'])
                
                # Hyperparameter optimization results
                if 'hyperopt_results' in model_result:
                    hyperopt = model_result['hyperopt_results']
                    result[f"{model_name}_best_val_score"] = hyperopt.get('best_value', 0)
                    result[f"{model_name}_n_trials"] = hyperopt.get('n_trials', 0)
                    
                    # Store best hyperparameters
                    best_params = hyperopt.get('best_params', {})
                    for param_name, param_value in best_params.items():
                        result[f"{model_name}_param_{param_name}"] = param_value
            
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