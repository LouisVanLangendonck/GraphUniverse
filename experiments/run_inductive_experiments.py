"""
Script to run inductive learning experiments on graph families.

This script provides a command-line interface for running inductive experiments
where models are trained on some graphs and tested on others.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.config import InductiveExperimentConfig
from experiments.inductive.experiment import run_inductive_experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inductive graph learning experiments')
    
    # Experiment setup
    parser.add_argument('--output_dir', type=str, default='inductive_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID to use')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    # Graph family parameters
    parser.add_argument('--n_graphs', type=int, default=10,
                        help='Number of graphs to generate in family')
    parser.add_argument('--min_n_nodes', type=int, default=90,
                        help='Minimum number of nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=100,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--min_communities', type=int, default=5,
                        help='Minimum number of communities per graph')
    parser.add_argument('--max_communities', type=int, default=5,
                        help='Maximum number of communities per graph')
    
    # Universe parameters
    parser.add_argument('--universe_K', type=int, default=5,
                        help='Number of communities in universe')
    parser.add_argument('--universe_feature_dim', type=int, default=32,
                        help='Feature dimension for universe')
    parser.add_argument('--universe_edge_density', type=float, default=0.1,
                        help='Base edge density for universe')
    parser.add_argument('--universe_homophily', type=float, default=0.2,
                        help='Homophily parameter for universe')
    
    # Graph family variation
    parser.add_argument('--homophily_range', type=float, nargs=2, default=[0.0, 0.1],
                        help='Range around universe homophily')
    parser.add_argument('--density_range', type=float, nargs=2, default=[0.0, 0.02],
                        help='Range around universe density')
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM model for graph generation')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution type')
    
    # Data split configuration
    parser.add_argument('--train_graph_ratio', type=float, default=0.6,
                        help='Fraction of graphs for training')
    parser.add_argument('--val_graph_ratio', type=float, default=0.2,
                        help='Fraction of graphs for validation')
    parser.add_argument('--test_graph_ratio', type=float, default=0.2,
                        help='Fraction of graphs for testing')
    parser.add_argument('--inductive_mode', type=str, default='graph_level',
                        choices=['graph_level', 'mixed'],
                        help='Inductive learning mode')
    
    # Task configuration
    parser.add_argument('--tasks', type=str, nargs='+', 
                        default=['community'], #, 'k_hop_community_counts'],
                        choices=['community', 'k_hop_community_counts'],
                        help='Learning tasks to run')
    parser.add_argument('--khop_community_counts_k', type=int, default=2,
                        help='Number of hops for community counts task')
    
    # Model configuration
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn', 'sage'],
                        choices=['gcn', 'gat', 'sage'],
                        help='Types of GNN models to run')
    parser.add_argument('--run_gnn', action='store_true', default=True,
                        help='Run GNN models')
    parser.add_argument('--run_mlp', action='store_true', default=True,
                        help='Run MLP model')
    parser.add_argument('--run_rf', action='store_true', default=True,
                        help='Run Random Forest model')
    parser.add_argument('--no_gnn', action='store_false', dest='run_gnn',
                        help='Skip GNN models')
    parser.add_argument('--no_mlp', action='store_false', dest='run_mlp',
                        help='Skip MLP model')
    parser.add_argument('--no_rf', action='store_false', dest='run_rf',
                        help='Skip Random Forest model')
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for neural models')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for neural models')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for neural models')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for neural models')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for neural models')
    
    # Hyperparameter optimization
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--optimization_timeout', type=int, default=600,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Family generation constraints
    parser.add_argument('--max_mean_community_deviation', type=float, default=0.07,
                        help='Maximum allowed mean community deviation')
    parser.add_argument('--max_max_community_deviation', type=float, default=0.17,
                        help='Maximum allowed maximum community deviation')
    parser.add_argument('--min_family_consistency', type=float, default=0.2,
                        help='Minimum required family consistency score')
    parser.add_argument('--require_consistency_check', action='store_true', default=False,
                        help='Require family consistency check')
    parser.add_argument('--no_consistency_check', action='store_false', 
                        dest='require_consistency_check',
                        help='Skip family consistency check')
    
    # Output options
    parser.add_argument('--save_individual_graphs', action='store_true',
                        help='Save individual graph objects (large files)')
    parser.add_argument('--collect_family_stats', action='store_true', default=True,
                        help='Collect detailed family generation statistics')
    
    return parser.parse_args()


def create_config_from_args(args) -> InductiveExperimentConfig:
    """Create experiment configuration from command line arguments."""
    
    config = InductiveExperimentConfig(
        # Task configuration
        tasks=args.tasks,
        khop_community_counts_k=args.khop_community_counts_k,
        
        # Model configuration
        gnn_types=args.gnn_types,
        run_gnn=args.run_gnn,
        run_mlp=args.run_mlp,
        run_rf=args.run_rf,
        
        # Training configuration
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        
        # Model architecture
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        
        # Hyperparameter optimization
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        
        # Graph family parameters
        n_graphs=args.n_graphs,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=args.min_communities,
        max_communities=args.max_communities,
        
        # Universe parameters
        universe_K=args.universe_K,
        universe_feature_dim=args.universe_feature_dim,
        universe_edge_density=args.universe_edge_density,
        universe_homophily=args.universe_homophily,
        
        # Graph family variation
        homophily_range=tuple(args.homophily_range),
        density_range=tuple(args.density_range),
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,
        
        # Data split configuration
        train_graph_ratio=args.train_graph_ratio,
        val_graph_ratio=args.val_graph_ratio,
        test_graph_ratio=args.test_graph_ratio,
        inductive_mode=args.inductive_mode,
        
        # Family generation constraints
        max_mean_community_deviation=args.max_mean_community_deviation,
        max_max_community_deviation=args.max_max_community_deviation,
        min_family_consistency=args.min_family_consistency,
        require_consistency_check=args.require_consistency_check,
        
        # Output configuration
        output_dir=args.output_dir,
        device_id=args.device_id,
        force_cpu=args.force_cpu,
        seed=args.seed,
        save_individual_graphs=args.save_individual_graphs,
        collect_family_stats=args.collect_family_stats
    )
    
    return config


def main():
    """Main function to run inductive experiments."""
    print("INDUCTIVE GRAPH LEARNING EXPERIMENTS")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration summary
    print("\nExperiment Configuration:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Graph family size: {config.n_graphs}")
    print(f"  Node range: [{config.min_n_nodes}, {config.max_n_nodes}]")
    print(f"  Community range: [{config.min_communities}, {config.max_communities}]")
    print(f"  Inductive mode: {config.inductive_mode}")
    print(f"  Tasks: {', '.join(config.tasks)}")
    print(f"  Models: ", end="")
    models = []
    if config.run_gnn:
        models.extend(config.gnn_types)
    if config.run_mlp:
        models.append("mlp")
    if config.run_rf:
        models.append("rf")
    print(", ".join(models))
    print(f"  Hyperparameter optimization: {config.optimize_hyperparams}")
    print(f"  Random seed: {config.seed}")
    
    try:
        # Run experiment
        print(f"\nStarting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = run_inductive_experiment(config)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
        # Print quick summary
        if 'results' in results:
            total_models = sum(len(task_results) for task_results in results['results'].values())
            successful_models = sum(
                sum(1 for model_results in task_results.values() if 'test_metrics' in model_results)
                for task_results in results['results'].values()
            )
            print(f"Models trained: {total_models}")
            print(f"Successful: {successful_models}")
            print(f"Success rate: {successful_models/total_models:.1%}" if total_models > 0 else "Success rate: 0%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)