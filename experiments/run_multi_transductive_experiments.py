"""
Script to run multiple transductive experiments with parameter sweeps.
Based on the inductive multi-experiment runner but adapted for transductive learning.
"""

import os
import sys
import argparse
from datetime import datetime
import itertools
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.core.config import TransductiveExperimentConfig
from experiments.core.experiment import run_transductive_experiment
from experiments.core.analysis import (
    analyze_transductive_results, 
    create_analysis_plots,
    generate_experiment_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple transductive graph learning experiments')
    
    # Parameters to vary
    parser.add_argument('--vary', type=str, nargs='+', default=['homophily'],
                        choices=[
                            'num_nodes',
                            'num_communities',
                            'universe_edge_density',
                            'universe_homophily', 
                            'universe_randomness_factor',
                            'degree_heterogeneity',
                            'edge_noise',
                            'cluster_count_factor',
                            'center_variance',
                            'cluster_variance',
                            'assignment_skewness',
                            'community_exclusivity'
                        ],
                        help='Parameters to vary across experiments')
    
    # Method selection
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM (for experiments)')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution for DCCC-SBM')
    
    # Task selection
    parser.add_argument('--tasks', type=str, nargs='+', default=['community'],
                        choices=['community', 'k_hop_community_counts'],
                        help='Learning tasks to run')
    
    # K-hop community counts task parameters
    parser.add_argument('--khop_community_counts_k', type=int, default=2,
                        help='Number of hops to consider for community counts task')
    
    # Regression-specific parameters
    parser.add_argument('--regression_loss', type=str, default='mse',
                        choices=['mse', 'mae'],
                        help='Loss function for regression tasks')
    parser.add_argument('--regression_metrics', type=str, nargs='+', 
                        default=['mse', 'rmse', 'mae', 'r2'],
                        choices=['mse', 'rmse', 'mae', 'r2'],
                        help='Metrics to compute for regression tasks')
    
    # Model selection and training parameters
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn', 'sage', 'gat', 'fagcn'],
                        choices=['gcn', 'gat', 'sage', 'fagcn'],
                        help='Types of GNN models to run')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip GNN models')
    parser.add_argument('--skip_mlp', action='store_true',
                        help='Skip MLP model')
    parser.add_argument('--skip_rf', action='store_true',
                        help='Skip Random Forest model')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping in neural models')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs for neural models')
    
    # Hyperparameter optimization parameters
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization for each model')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--opt_timeout', type=int, default=300,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Transformer configuration
    parser.add_argument('--transformer_types', type=str, nargs='+', 
                        default=['graphormer', 'graphgps'], 
                        choices=['graphormer', 'graphgps'],
                        help='Types of Graph Transformer models to run')
    parser.add_argument('--run_transformers', action='store_true',
                        help='Run Graph Transformer models')
    parser.add_argument('--transformer_num_heads', type=int, default=8,
                        help='Number of attention heads for transformers')
    parser.add_argument('--transformer_max_nodes', type=int, default=200,
                        help='Maximum nodes for encoding precomputation')
    parser.add_argument('--transformer_max_path_length', type=int, default=10,
                        help='Maximum path length for shortest path encoding')
    parser.add_argument('--transformer_precompute_encodings', action='store_true', default=True,
                        help='Precompute structural encodings for transformers')
    parser.add_argument('--no_precompute_encodings', action='store_false', 
                        dest='transformer_precompute_encodings',
                        help='Disable encoding precomputation')
    parser.add_argument('--transformer_cache_encodings', action='store_true', default=True,
                        help='Cache structural encodings between graphs')
    parser.add_argument('--local_gnn_type', type=str, default='gcn',
                        choices=['gcn', 'sage'],
                        help='Local GNN type for GraphGPS')
    parser.add_argument('--global_model_type', type=str, default='transformer',
                        help='Global model type for GraphGPS')
    parser.add_argument('--transformer_prenorm', action='store_true', default=True,
                        help='Use pre-normalization in transformers')
    
    # Parameter ranges for sweep
    parser.add_argument('--num_nodes_range', type=int, nargs=2, default=[80, 120],
                        help='Range for num_nodes (min, max)')
    parser.add_argument('--num_communities_range', type=int, nargs=2, default=[5, 6],
                        help='Range for num_communities (min, max)')
    parser.add_argument('--universe_edge_density_range', type=float, nargs=3, default=[0.05, 0.25, 0.20],
                        help='Range for universe edge density (start, end, step)')
    parser.add_argument('--universe_homophily_range', type=float, nargs=3, default=[0.0, 1.0, 0.2],
                        help='Range for universe homophily (start, end, step)')
    parser.add_argument('--universe_randomness_factor_range', type=float, nargs=2, default=[0.0, 1.0],
                        help='Range for universe randomness factor (min, max)')
    parser.add_argument('--degree_heterogeneity_range', type=float, nargs=2, default=[0.0, 1.0],
                        help='Range for degree_heterogeneity (min, max)')
    parser.add_argument('--edge_noise_range', type=float, nargs=2, default=[0.0, 0.3],
                        help='Range for edge_noise (min, max)')
    
    # Universe parameter ranges
    parser.add_argument('--cluster_count_factor_range', type=float, nargs=2, default=[0.5, 1.5],
                        help='Range for cluster_count_factor (min, max)')
    parser.add_argument('--center_variance_range', type=float, nargs=2, default=[0.01, 1.0],
                        help='Range for center_variance (min, max)')
    parser.add_argument('--cluster_variance_range', type=float, nargs=2, default=[0.05, 1.5],
                        help='Range for cluster_variance (min, max)')
    parser.add_argument('--assignment_skewness_range', type=float, nargs=2, default=[0.0, 0.5],
                        help='Range for assignment_skewness (min, max)')
    parser.add_argument('--community_exclusivity_range', type=float, nargs=2, default=[0.6, 1.0],
                        help='Range for community_exclusivity (min, max)')
    
    # Experiment control
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Number of times to repeat each parameter combination')
    parser.add_argument('--results_dir', type=str, default='multi_transductive_results',
                        help='Directory to save all experiment results')
    parser.add_argument('--max_training_nodes', type=int, default=20,
                        help='Maximum number of nodes to use for training')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    
    # Analysis options
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Create analysis plots after experiments')
    parser.add_argument('--analyze_existing', type=str, default=None,
                        help='Analyze existing results at given path')
    
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
    
    # Only vary homophily and edge density with steps
    for param in ['universe_homophily', 'universe_edge_density']:
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
    failed_runs = []
    
    # Get all possible parameters and their ranges for random sampling
    all_param_ranges = {
        'num_nodes': (args.num_nodes_range[0], args.num_nodes_range[1]),  # Only use min/max
        'num_communities': (args.num_communities_range[0], args.num_communities_range[1]),
        'universe_randomness_factor': (args.universe_randomness_factor_range[0], args.universe_randomness_factor_range[1]),
        'degree_heterogeneity': (args.degree_heterogeneity_range[0], args.degree_heterogeneity_range[1]),
        'edge_noise': (args.edge_noise_range[0], args.edge_noise_range[1]),
        'cluster_count_factor': (args.cluster_count_factor_range[0], args.cluster_count_factor_range[1]),
        'center_variance': (args.center_variance_range[0], args.center_variance_range[1]),
        'cluster_variance': (args.cluster_variance_range[0], args.cluster_variance_range[1]),
        'assignment_skewness': (args.assignment_skewness_range[0], args.assignment_skewness_range[1]),
        'community_exclusivity': (args.community_exclusivity_range[0], args.community_exclusivity_range[1])
    }
    
    total_runs = len(param_combinations) * args.n_repeats
    successful_runs = 0
    failed_runs = []
    
    # Create progress bar
    pbar = tqdm(total=total_runs, desc="Running experiments")
    
    # Run each parameter combination
    for param_dict in param_combinations:
        for repeat in range(args.n_repeats):
            try:
                # Generate random values for other parameters
                random_params = {}
                for param, (min_val, max_val) in all_param_ranges.items():
                    if isinstance(min_val, int):
                        random_params[param] = np.random.randint(min_val, max_val + 1)
                    else:
                        random_params[param] = np.random.uniform(min_val, max_val)

                print(f"\n🔄 Run {successful_runs + 1}/{total_runs}")
                print(f"📊 Sweep parameters:")
                for key, value in param_dict.items():
                    print(f"   {key}: {value}")
                print(f"🎲 Random parameters:")
                for key, value in random_params.items():
                    print(f"   {key}: {value}")
                
                # Create experiment config
                config = TransductiveExperimentConfig(
                    # Base parameters (varied)
                    universe_edge_density=param_dict['universe_edge_density'],
                    universe_homophily=param_dict['universe_homophily'],
                    
                    # Random parameters
                    num_nodes=random_params['num_nodes'],
                    num_communities=random_params['num_communities'],
                    universe_randomness_factor=random_params['universe_randomness_factor'],
                    degree_heterogeneity=random_params['degree_heterogeneity'],
                    edge_noise=random_params['edge_noise'],
                    cluster_count_factor=random_params['cluster_count_factor'],
                    center_variance=random_params['center_variance'],
                    cluster_variance=random_params['cluster_variance'],
                    assignment_skewness=random_params['assignment_skewness'],
                    community_exclusivity=random_params['community_exclusivity'],
                    
                    # Method configuration
                    use_dccc_sbm=args.use_dccc_sbm,
                    degree_distribution=args.degree_distribution,
                    
                    # Task configuration
                    tasks=args.tasks,
                    khop_community_counts_k=args.khop_community_counts_k,
                    regression_loss=args.regression_loss,
                    regression_metrics=args.regression_metrics,
                    
                    # Model configuration
                    gnn_types=args.gnn_types,
                    run_gnn=args.run_gnn,
                    run_mlp=args.run_mlp,
                    run_rf=args.run_rf,
                    
                    # Transformer configuration
                    transformer_types=args.transformer_types,
                    run_transformers=args.run_transformers,
                    transformer_num_heads=args.transformer_num_heads,
                    transformer_max_nodes=args.transformer_max_nodes,
                    transformer_max_path_length=args.transformer_max_path_length,
                    transformer_precompute_encodings=args.transformer_precompute_encodings,
                    transformer_cache_encodings=getattr(args, 'transformer_cache_encodings', True),
                    local_gnn_type=args.local_gnn_type,
                    global_model_type=args.global_model_type,
                    transformer_prenorm=getattr(args, 'transformer_prenorm', True),
                    
                    # Training configuration
                    patience=args.patience,
                    epochs=args.epochs,
                    
                    # Hyperparameter optimization
                    optimize_hyperparams=args.optimize_hyperparams,
                    n_trials=args.n_trials,
                    optimization_timeout=args.opt_timeout,
                    
                    # Node limit
                    max_training_nodes=args.max_training_nodes,
                    
                    # Random seed
                    seed=args.seed + repeat
                )
                
                # Create run directory
                run_dir = os.path.join(results_dir, f"run_{len(all_results)}")
                os.makedirs(run_dir, exist_ok=True)
                
                # Save config
                config_path = os.path.join(run_dir, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)
                
                # Run experiment
                start_time = time.time()
                result = run_transductive_experiment(config)
                end_time = time.time()
                
                # Add metadata
                result['config_path'] = config_path
                result['run_time'] = end_time - start_time
                result['parameters'] = param_dict
                result['repeat'] = repeat
                
                # Save result
                result_path = os.path.join(run_dir, "result.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                all_results.append(result)
                successful_runs += 1
                
            except Exception as e:
                logger.error(f"Failed run with parameters {param_dict}, repeat {repeat}: {str(e)}")
                failed_runs.append({
                    'parameters': param_dict,
                    'repeat': repeat,
                    'error': str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Save summary
    summary = {
        'total_runs': total_runs,
        'successful_runs': successful_runs,
        'failed_runs': len(failed_runs),
        'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
        'failed_runs_details': failed_runs,
        'timestamp': timestamp
    }
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create analysis plots if requested
    if args.create_plots and successful_runs > 0:
        print("\nCreating analysis plots...")
        create_analysis_plots(results_dir)
        print(f"Analysis plots created in: {os.path.join(results_dir, 'plots')}")
    
    return {
        'results_dir': results_dir,
        'all_results': all_results,
        'summary': summary
    }


def main():
    """Main function to run multi-experiments."""
    print("TRANSDUCTIVE MULTI-EXPERIMENT PARAMETER SWEEPS")
    print("=" * 60)
    
    try:
        args = parse_args()
        
        # Handle analysis of existing results
        if args.analyze_existing:
            print(f"Analyzing existing results at: {args.analyze_existing}")
            create_analysis_plots(args.analyze_existing)
            print("Analysis complete!")
            return 0
        
        # Print configuration summary
        print(f"\nMulti-Experiment Configuration:")
        print(f"  Parameters to vary: {', '.join(args.vary)}")
        print(f"  Number of repeats: {args.n_repeats}")
        print(f"  Results directory: {args.results_dir}")
        print(f"  Base method: {'DCCC-SBM' if args.use_dccc_sbm else 'DC-SBM'}")
        
        # Print model configuration
        print("\nModel Configuration:")
        if args.run_gnn:
            print(f"  GNN types: {', '.join(args.gnn_types)}")
        if args.run_transformers:
            print(f"  Transformer types: {', '.join(args.transformer_types)}")
            print(f"  Transformer heads: {args.transformer_num_heads}")
            print(f"  Max nodes: {args.transformer_max_nodes}")
            print(f"  Max path length: {args.transformer_max_path_length}")
            print(f"  Precompute encodings: {args.transformer_precompute_encodings}")
            print(f"  Cache encodings: {getattr(args, 'transformer_cache_encodings', True)}")
            print(f"  Local GNN type: {args.local_gnn_type}")
            print(f"  Global model type: {args.global_model_type}")
            print(f"  Pre-norm: {getattr(args, 'transformer_prenorm', True)}")
        if args.run_mlp:
            print("  MLP: enabled")
        if args.run_rf:
            print("  Random Forest: enabled")
        
        # Print parameter ranges
        print("\nParameter ranges:")
        for param in args.vary:
            range_attr = f"{param}_range"
            if hasattr(args, range_attr):
                start, end, step = getattr(args, range_attr)
                print(f"  {param}: {start} to {end} (step {step})")
        
        # Run experiments
        print(f"\nStarting multi-experiment suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results = run_experiments(args)
        
        print(f"\nMulti-experiment suite completed successfully!")
        
        # Print summary
        if 'summary' in results:
            stats = results['summary']
            print(f"\nFinal Summary:")
            print(f"  Total runs attempted: {stats['total_runs']}")
            print(f"  Successful runs: {stats['successful_runs']}")
            print(f"  Failed runs: {stats['failed_runs']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nMulti-experiment suite interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nMulti-experiment suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)