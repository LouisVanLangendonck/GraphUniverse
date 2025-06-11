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
from experiments.inductive.multi_config import SSLMultiExperimentConfig, ParameterRange
from experiments.inductive.multi_experiment import SSLMultiExperimentRunner
from experiments.inductive.experiment import PreTrainingRunner
from experiments.inductive.data import GraphFamilyManager, PreTrainedModelSaver


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-experiment SSL sweeps')
    
    # === EXPERIMENT SETUP ===
    parser.add_argument('--output_dir', type=str, default='multi_ssl_experiments',
                        help='Base output directory')
    parser.add_argument('--experiment_name', type=str, default='ssl_sweep',
                        help='Name for this experiment')
    parser.add_argument('--n_repetitions', type=int, default=1,
                        help='Number of repetitions per configuration')
    
    # === PRETRAINING TASK ===
    parser.add_argument('--pretraining_task', type=str, default='dgi',
                        choices=['link_prediction', 'dgi', 'graphmae'],
                        help='Self-supervised pre-training task')
    
    # === TASK-SPECIFIC PARAMETERS ===
    # Link prediction
    parser.add_argument('--negative_sampling_ratio', type=float, default=1.0,
                        help='Negative sampling ratio for link prediction')
    parser.add_argument('--link_pred_loss', type=str, default='bce',
                        choices=['bce', 'margin'],
                        help='Loss function for link prediction')
    
    # DGI
    parser.add_argument('--dgi_corruption_types', type=str, nargs='+', default=['edge_dropout', 'feature_dropout'],
                        choices=['feature_shuffle', 'edge_dropout', 'feature_dropout', 'feature_noise', 'edge_perturbation'],
                        help='Type of corruption for DGI when using feature_noise')
    parser.add_argument('--dgi_noise_std', type=float, default=0.1,
                        help='Noise standard deviation for DGI')
    parser.add_argument('--dgi_perturb_rate', type=float, default=0.1,
                        help='Perturbation rate for DGI')
    parser.add_argument('--dgi_corruption_rate', type=float, default=0.2,
                        help='Corruption rate for DGI')
    
    # GraphMAE
    parser.add_argument('--graphmae_mask_rate', type=float, default=0.5,
                        help='Mask rate for GraphMAE')
    parser.add_argument('--graphmae_replace_rate', type=float, default=0.1,
                        help='Replace rate for GraphMAE')
    parser.add_argument('--graphmae_gamma', type=float, default=2.0,
                        help='Gamma for GraphMAE')
    parser.add_argument('--graphmae_decoder_type', type=str, default='gnn',
                        choices=['gnn', 'mlp'],
                        help='Decoder type for GraphMAE')
    parser.add_argument('--graphmae_decoder_gnn_type', type=str, default='gcn',
                        choices=['gcn', 'sage'],
                        help='GNN type for GraphMAE decoder')
    
    # Contrastive learning
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning')
    parser.add_argument('--corruption_type', type=str, default='edge_dropout',
                        choices=['feature_shuffle', 'edge_dropout'],
                        help='Type of corruption for contrastive learning')
    parser.add_argument('--corruption_rate', type=float, default=0.5,
                        help='Corruption rate for contrastive learning')
    
    
    # === MODEL SELECTION ===
    parser.add_argument('--gnn_models', type=str, nargs='+', 
                        default=['gcn', 'sage', 'fagcn', 'gin'],
                        choices=['gcn', 'sage', 'gat', 'fagcn', 'gin'],
                        help='GNN models to run')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip running GNN models (only run transformers if specified)')
    parser.add_argument('--transformer_models', type=str, nargs='+', 
                        default=['graphormer', 'graphgps'],
                        choices=['graphormer', 'graphgps'],
                        help='Transformer models to run')
    parser.add_argument('--run_transformers', action='store_true',
                        help='Run transformer models')
    
    # === TRANSFORMER CONFIGURATION ===
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
    
    # === GRAPH FAMILY PARAMETERS ===
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
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM instead of DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution for DCCC-SBM')
    
    # === SWEEP PARAMETERS ===
    # Homophily sweep
    parser.add_argument('--homophily_min', type=float, default=0.0,
                        help='Minimum homophily value for sweep')
    parser.add_argument('--homophily_max', type=float, default=1.0,
                        help='Maximum homophily value for sweep')
    parser.add_argument('--homophily_step', type=float, default=0.25,
                        help='Step size for homophily sweep')
    
    # Density sweep
    parser.add_argument('--density_min', type=float, default=0.05,
                        help='Minimum edge density value for sweep')
    parser.add_argument('--density_max', type=float, default=0.15,
                        help='Maximum edge density value for sweep')
    parser.add_argument('--density_step', type=float, default=0.10,
                        help='Step size for density sweep')
    
    # Number of graphs sweep
    parser.add_argument('--n_graphs_min', type=int, default=100,
                        help='Minimum number of graphs per family')
    parser.add_argument('--n_graphs_max', type=int, default=250,
                        help='Maximum number of graphs per family')
    parser.add_argument('--n_graphs_step', type=int, default=150,
                        help='Step size for number of graphs sweep')
    
    # Universe K sweep
    parser.add_argument('--universe_K_min', type=int, default=10,
                        help='Minimum number of communities')
    parser.add_argument('--universe_K_max', type=int, default=10,
                        help='Maximum number of communities')
    parser.add_argument('--universe_K_step', type=int, default=1,
                        help='Step size for number of communities sweep')

    # === MODEL PARAMETERS ===
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    
    # === TRAINING PARAMETERS ===
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs for pre-training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--optimize_hyperparams', action='store_true', default=True,
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--optimization_timeout', type=int, default=1200,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # === RANDOM SEED ===
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_preset_config(preset: str, args) -> SSLMultiExperimentConfig:
    """Create configuration based on preset."""
    
    if preset == 'custom':
        
        return SSLMultiExperimentConfig(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            
            # Use command line parameters
            n_graphs=args.n_graphs_min,
            n_extra_graphs=args.n_extra_graphs,
            universe_K=args.universe_K_min,
            universe_homophily=args.universe_homophily,
            universe_edge_density=args.universe_edge_density,
            use_dccc_sbm=args.use_dccc_sbm,
            degree_distribution=args.degree_distribution,
            
            gnn_types=args.gnn_types,
            pretraining_tasks=args.pretraining_tasks,
            hidden_dims=args.hidden_dim,
            num_layers=args.num_layers,
            
            epochs=args.epochs,
            patience=args.patience,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
            n_repetitions=args.n_repetitions,
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}")


def create_custom_experiment(args) -> SSLMultiExperimentConfig:
    """Create a custom experiment configuration."""
    # from experiments.inductive.config import PreTrainingConfig
    
    # Base configuration
    base_config = PreTrainingConfig(
        n_graphs=args.n_graphs_min,
        n_extra_graphs_for_finetuning=args.n_extra_graphs,
        universe_K=args.universe_K_min,
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
        gnn_type=args.gnn_models[0],  # Will be overridden for each model
        pretraining_task=args.pretraining_task,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        
        # Training parameters
        epochs=args.epochs,
        patience=args.patience,
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
        ),
        'n_graphs': ParameterRange(
            min_val=args.n_graphs_min,
            max_val=args.n_graphs_max,
            step=args.n_graphs_step,
            is_sweep=True
        ),
        'universe_K': ParameterRange(
            min_val=args.universe_K_min,
            max_val=args.universe_K_max,
            step=args.universe_K_step,
            is_sweep=True
        )
    }

    model_sweep_parameters = {}
    
    # Add pre-training task specific hyperparameters only for the selected task
    if args.pretraining_task == 'dgi':
        model_sweep_parameters.update({
            'dgi_corruption_type': ParameterRange(
                min_val=0,
                max_val=1,
                discrete_values=args.dgi_corruption_types,
                is_sweep=True
            )
        })
    # if args.pretraining_task == 'link_prediction':
    #     sweep_parameters.update({
    #         'negative_sampling_ratio': ParameterRange(
    #             min_val=args.neg_sampling_ratio_min,
    #             max_val=args.neg_sampling_ratio_max,
    #             step=args.neg_sampling_ratio_step,
    #             is_sweep=True
    #         ),
    #         'link_prediction_threshold': ParameterRange(
    #             min_val=args.link_pred_threshold_min,
    #             max_val=args.link_pred_threshold_max,
    #             step=args.link_pred_threshold_step,
    #             is_sweep=True
    #         )
    #     })
    # elif args.pretraining_task == 'dgi':
    #     sweep_parameters.update({
    #         'corruption_type': ParameterRange(
    #             min_val=0,
    #             max_val=1,
    #             discrete_values=args.corruption_types,
    #             is_sweep=True
    #         )
    #     })
    # elif args.pretraining_task == 'graphcl':
    #     sweep_parameters.update({
    #         'temperature': ParameterRange(
    #             min_val=args.temperature_min,
    #             max_val=args.temperature_max,
    #             step=args.temperature_step,
    #             is_sweep=True
    #         ),
    #         'num_augmentations': ParameterRange(
    #             min_val=1,
    #             max_val=3,
    #             step=1,
    #             is_sweep=True
    #         )
    #     })
    
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
    
    # Create multi-experiment configuration
    config = SSLMultiExperimentConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        model_sweep_parameters=model_sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=args.n_repetitions,
        gnn_models=[] if args.skip_gnn else args.gnn_models,
        transformer_models=args.transformer_models,
        run_transformers=args.run_transformers,
        transformer_params={
            'transformer_num_heads': args.transformer_num_heads,
            'transformer_max_nodes': args.transformer_max_nodes,
            'transformer_max_path_length': args.transformer_max_path_length,
            'transformer_precompute_encodings': args.transformer_precompute_encodings,
            'transformer_cache_encodings': getattr(args, 'transformer_cache_encodings', True),
            'local_gnn_type': args.local_gnn_type,
            'global_model_type': args.global_model_type,
            'transformer_prenorm': getattr(args, 'transformer_prenorm', True)
        },
        skip_gnn=args.skip_gnn,
        patience=args.patience
    )
    
    return config


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