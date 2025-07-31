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

from experiments.transductive.config import TransductiveExperimentConfig
from experiments.transductive.experiment import run_transductive_experiment
from experiments.transductive.data import resplit_transductive_indices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple transductive graph learning experiments')
    
    # Method selection
    parser.add_argument('--use_dccc_sbm', action='store_true', default=True,
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
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn', 'sage', 'gat', 'gin'],
                        choices=['gcn', 'gat', 'sage', 'fagcn'],
                        help='Types of GNN models to run')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip GNN models')
    parser.add_argument('--skip_mlp', action='store_true',
                        help='Skip MLP model')
    parser.add_argument('--skip_rf', action='store_true', default=True,
                        help='Skip Random Forest model')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping in neural models')
    parser.add_argument('--epochs', type=int, default=500,
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
                        default=['graphgps'], 
                        choices=['graphgps'],
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
    
    # Neural Sheaf Diffusion Model options
    parser.add_argument('--run_neural_sheaf', action='store_true', help='Include neural sheaf diffusion model in experiments')
    parser.add_argument('--sheaf_type', type=str, default='diagonal', choices=['diagonal', 'bundle', 'general'], help='Type of sheaf restriction map')
    parser.add_argument('--sheaf_d', type=int, default=2, help='Stalk dimension for neural sheaf diffusion')
    parser.add_argument('--pe_type', type=str, default='laplacian', choices=['laplacian', 'degree', 'rwse', 'none'], help='Type of positional encoding')
    parser.add_argument('--max_pe_dim', type=int, default=8, help='Maximum positional encoding dimension')
    
    # Parameter ranges for sweep (only these three are ranges)
    parser.add_argument('--homophily_range', type=float, nargs='+', default=[0.2, 0.8, 0.3], help='Range of homophily values to sweep (start, end, [step])')
    parser.add_argument('--density_range', type=float, nargs='+', default=[0.1, 0.3, 0.2], help='Range of density values to sweep (start, end, [step])')
    parser.add_argument('--num_nodes_range', type=int, nargs='+', default=[150, 150, 50], help='Range of num_nodes to sweep (start, end, [step])')
    parser.add_argument('--degree_separation_values', type=float, nargs='+', default=[0.5, 0.5, 0.5], help='List of degree_separation to sweep (start, end, [step])')

    # All other parameters are fixed (single value)
    parser.add_argument('--num_communities', type=int, default=8, help='Number of communities (fixed)')
    parser.add_argument('--universe_feature_dim', type=int, default=16, help='Universe feature dimension (fixed)')
    parser.add_argument('--universe_randomness_factor', type=float, default=1.0, help='Universe randomness factor (fixed)')
    parser.add_argument('--degree_heterogeneity', type=float, default=1.0, help='Degree heterogeneity (fixed)')
    parser.add_argument('--edge_noise', type=float, default=0.0, help='Edge noise (fixed)')
    parser.add_argument('--cluster_count_factor', type=float, default=1.0, help='Cluster count factor (fixed)')
    parser.add_argument('--center_variance', type=float, default=0.1, help='Center variance (fixed)')
    parser.add_argument('--cluster_variance', type=float, default=1.0, help='Cluster variance (fixed)')
    parser.add_argument('--assignment_skewness', type=float, default=0.0, help='Assignment skewness (fixed)')
    parser.add_argument('--community_exclusivity', type=float, default=1.0, help='Community exclusivity (fixed)')
    
    
    # Data split sizes
    parser.add_argument('--n_val', type=int, default=20, help='Number of validation nodes (overrides val_ratio)')
    parser.add_argument('--n_test', type=int, default=20, help='Number of test nodes (overrides test_ratio)')
    
    # Experiment control
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Number of times to repeat each parameter combination')
    parser.add_argument('--k_fold', type=int, default=3,
                        help='Number of folds for cross-validation')
    parser.add_argument('--results_dir', type=str, default='multi_transductive_results',
                        help='Directory to save all experiment results')
    
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

def generate_parameter_combinations(args) -> list:
    # Only sweep over homophily, density, num_nodes
    def get_range(val, is_int=False):
        if len(val) == 3:
            start, end, step = val
            if is_int:
                return list(range(int(start), int(end)+1, int(step)))
            else:
                return list(np.arange(start, end+step/2, step))
        elif len(val) == 2:
            start, end = val
            if is_int:
                return list(range(int(start), int(end)+1))
            else:
                return list(np.linspace(start, end, num=3))
        else:
            return [val[0]]
        
    homophily_values = get_range(args.homophily_range, is_int=False)
    density_values = get_range(args.density_range, is_int=False)
    num_nodes_values = get_range(args.num_nodes_range, is_int=True)
    degree_separation_values = get_range(args.degree_separation_values, is_int=False)

    combos = list(itertools.product(homophily_values, density_values, num_nodes_values, degree_separation_values))
    param_combinations = []
    for h, d, n, ds in combos:
        param_combinations.append({
            'homophily': h,
            'density': d,
            'num_nodes': n,
            'degree_separation': ds
        })
    return param_combinations

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
    
    total_runs = len(param_combinations) * args.n_repeats
    successful_runs = 0
    failed_runs = []
    
    # Create progress bar
    pbar = tqdm(total=total_runs, desc="Running experiments")
    
    # Run each parameter combination
    for param_dict in param_combinations:
        for repeat in range(args.n_repeats):
            try:

                print(f"\n🔄 Run {successful_runs + 1}/{total_runs}")
                print(f"📊 Sweep parameters:")
                for key, value in param_dict.items():
                    print(f"   {key}: {value}")
                
                # Create experiment config
                config = TransductiveExperimentConfig(
                    # Base parameters (varied)
                    universe_edge_density=param_dict['density'],
                    universe_homophily=param_dict['homophily'],

                    universe_K=args.num_communities,
                    universe_feature_dim=args.universe_feature_dim,
                    
                    # Random parameters
                    num_nodes=param_dict['num_nodes'],
                    num_communities=args.num_communities,
                    universe_randomness_factor=args.universe_randomness_factor,
                    degree_heterogeneity=args.degree_heterogeneity,
                    edge_noise=args.edge_noise,
                    cluster_count_factor=args.cluster_count_factor,
                    center_variance=args.center_variance,
                    cluster_variance=args.cluster_variance,
                    assignment_skewness=args.assignment_skewness,
                    community_exclusivity=args.community_exclusivity,
                    
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
                    
                    # Neural Sheaf Diffusion Model options
                    run_neural_sheaf=args.run_neural_sheaf,
                    sheaf_type=args.sheaf_type,
                    sheaf_d=args.sheaf_d,
                    pe_type=args.pe_type,
                    
                    # Training configuration
                    patience=args.patience,
                    epochs=args.epochs,
                    k_fold=args.k_fold,
                    
                    # Hyperparameter optimization
                    optimize_hyperparams=args.optimize_hyperparams,
                    n_trials=args.n_trials,
                    optimization_timeout=args.opt_timeout,
                    
                    # Data split sizes
                    n_val=args.n_val,
                    n_test=args.n_test,
                    
                    # Random seed
                    seed=args.seed + repeat,
                    degree_separation_range=(param_dict['degree_separation'], param_dict['degree_separation'])
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

        
        # Print configuration summary
        print(f"\nMulti-Experiment Configuration:")
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
        if args.run_neural_sheaf:
            print("  Neural Sheaf Diffusion Model: enabled")
            print(f"    Sheaf type: {args.sheaf_type}")
            print(f"    Stalk dimension: {args.sheaf_d}")
            print(f"    PE type: {args.pe_type}")
            print(f"    Max PE dim: {args.max_pe_dim}")
        
        # Print parameter ranges
        print("\nParameter ranges:")
        for param in ['homophily', 'density', 'num_nodes']:
            range_attr = f"{param}_range"
            if hasattr(args, range_attr):
                range_val = getattr(args, range_attr)[:2]
                if len(range_val) == 3:
                    start, end, step = range_val
                    print(f"  {param}: {start} to {end} (step {step})")
                elif len(range_val) == 2:
                    start, end = range_val
                    print(f"  {param}: {start} to {end} (no step specified)")
                else:
                    print(f"  {param}: {range_val} (unexpected length)")
        
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