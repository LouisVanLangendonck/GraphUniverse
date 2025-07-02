"""
Clean script to run inductive learning experiments.
Removes old parameters and focuses on DC-SBM and DCCC-SBM methods only.
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import clean implementations
from experiments.inductive.config import InductiveExperimentConfig
from experiments.inductive.experiment import run_inductive_experiment


def parse_args():
    """Parse command line arguments for clean experiment."""
    parser = argparse.ArgumentParser(description='Run clean inductive graph learning experiments')
    
    # === EXPERIMENT SETUP ===
    parser.add_argument('--output_dir', type=str, default='clean_inductive_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID to use')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--use_parallel_training', action='store_true', default=False,
                        help='Use parallel training')
    parser.add_argument('--max_parallel_gpu_jobs', type=int, default=None,
                        help='Maximum number of parallel GPU jobs to use')
    parser.add_argument('--cross_task_parallelization', action='store_true', default=False,
                        help='Enable parallelization across tasks (not just within tasks)')

    # === PRETRAINED MODELS ===
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pre-trained models instead of random initialization')
    parser.add_argument('--pretrained_model_dir', type=str, default='ssl_experiments',
                        help='Directory containing pre-trained models')
    parser.add_argument('--pretrained_model_id', type=str, default=None,
                        help='Specific pre-trained model ID to use for fine-tuning')
    parser.add_argument('--graph_family_id', type=str, default=None,
                        help='Specific graph family ID to use (should match pre-trained model)')
    parser.add_argument('--graph_family_dir', type=str, default='graph_families',
                        help='Directory containing graph families')
    parser.add_argument('--auto_load_family', action='store_true', default=True,
                        help='Automatically load graph family associated with pre-trained model')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights during fine-tuning')
    parser.add_argument('--calculate_silhouette_score', action='store_true', default=True,
                        help='Calculate silhouette score of communities of pre-trained models')
    parser.add_argument('--max_train_graphs_for_finetuning', type=int, default=3,
                        help='Maximum number of training graphs for fine-tuning')
    parser.add_argument('--only_pretrained_experiments', action='store_true',
                        help='Dont do any other experiments. Only fine-tune pre-trained models and from scratch version of it and hyperparameter optimization of that model type.')    
        
    # === TASKS ===
    parser.add_argument('--tasks', type=str, nargs='+', default=['k_hop_community_counts_k2'],
                        choices=['community', 'k_hop_community_counts_k1', 'k_hop_community_counts_k2', 'k_hop_community_counts_k3', 'triangle_count'],
                        help='Learning tasks to run')
    parser.add_argument('--khop_k', type=int, default=2,
                        help='k value for k-hop community counting task')
    
    # === DATA SPLITS ===
    parser.add_argument('--no_unseen_community_combinations_for_eval', action='store_true', default=False,
                        help='Do not allow unseen community combinations for evaluation')

    # === GRAPH FAMILY GENERATION ===
    parser.add_argument('--n_graphs', type=int, default=20,
                        help='Number of graphs to generate in family')
    parser.add_argument('--min_n_nodes', type=int, default=70,
                        help='Minimum number of nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=130,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--min_communities', type=int, default=4,
                        help='Minimum number of communities per graph')
    parser.add_argument('--max_communities', type=int, default=7,
                        help='Maximum number of communities per graph')
    
    # === UNIVERSE PARAMETERS ===
    parser.add_argument('--universe_K', type=int, default=15,
                        help='Number of communities in universe')
    parser.add_argument('--universe_feature_dim', type=int, default=32,
                        help='Feature dimension for universe')
    parser.add_argument('--universe_edge_density', type=float, default=0.07,
                        help='Base edge density for universe')
    parser.add_argument('--universe_homophily', type=float, default=0.25,
                        help='Homophily parameter for universe')
    parser.add_argument('--universe_randomness_factor', type=float, default=1.0,
                        help='Randomness factor for universe')
    
    # === METHOD SELECTION ===
    parser.add_argument('--use_dccc_sbm', action='store_true', default=True,
                        help='Use DCCC-SBM instead of standard DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution type for DCCC-SBM')
    
    # === GRAPH FAMILY VARIATION ===
    parser.add_argument('--homophily_range', type=float, nargs=2, default=[0.0, 0.1],
                        help='Range around universe homophily')
    parser.add_argument('--density_range', type=float, nargs=2, default=[0.0, 0.1],
                        help='Range around universe density')
    parser.add_argument('--degree_heterogeneity', type=float, default=0.5,
                        help='Degree heterogeneity parameter')
    parser.add_argument('--edge_noise', type=float, default=0.1,
                        help='Edge noise level')
    
    # === DCCC-SBM PARAMETERS ===
    parser.add_argument('--community_imbalance_range', type=float, nargs=2, default=[0.0, 0.3],
                        help='Range for community size imbalance (DCCC-SBM)')
    parser.add_argument('--degree_separation_range', type=float, nargs=2, default=[0.3, 1.0],
                        help='Range for degree distribution separation (DCCC-SBM)')
    
    # === METAPATH TASK ARGUMENTS === 
    parser.add_argument('--enable_metapath_tasks', action='store_true',
                        help='Enable metapath-based classification tasks')
    parser.add_argument('--metapath_k_values', type=int, nargs='+', default=[4, 5],
                        help='K-values for metapath lengths (4+ for proper loops)')
    parser.add_argument('--metapath_require_loop', action='store_true', default=False,
                        help='Require metapaths to form loops')
    parser.add_argument('--metapath_degree_weight', type=float, default=0.3,
                        help='Weight for degree center influence in metapath selection')
    parser.add_argument('--max_community_participation', type=float, default=0.95,
                        help='Maximum allowed participation rate per community')
    
    # === MODELS ===
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn'],
                        choices=['gcn', 'fagcn', 'sage', 'gat', 'gin'],
                        help='Types of GNN models to run')
    parser.add_argument('--transformer_types', type=str, nargs='+', 
                        default=['graphgps'], 
                        choices=['graphormer', 'graphgps'],
                        help='Types of Graph Transformer models to run')
    parser.add_argument('--run_transformers', action='store_true',
                        help='Run Graph Transformer models')
    parser.add_argument('--run_neural_sheaf', action='store_true',
                        help='Run Neural Sheaf Diffusion models')
    parser.add_argument('--no_transformers', action='store_false', dest='run_transformers',
                        help='Skip Graph Transformer models')
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
    
    # === TRAINING ===
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=70,
                        help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for neural models')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for neural models')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='Number of trials for hyperparameter optimization')


    # === ANALYSIS ===
    parser.add_argument('--require_consistency_check', action='store_true', default=False,
                        help='Require family consistency check')
    parser.add_argument('--collect_signal_metrics', action='store_true', default=True,
                        help='Collect community signal metrics')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization')
    
    # === FEATURE GENERATION ===
    parser.add_argument('--cluster_count_factor', type=float, default=1.0,
                        help='Factor for cluster count')
    parser.add_argument('--center_variance', type=float, default=0.4,
                        help='Variance for center of clusters')
    parser.add_argument('--cluster_variance', type=float, default=0.2,
                        help='Variance for cluster sizes')
    parser.add_argument('--assignment_skewness', type=float, default=0.0,
                        help='Skewness for feature assignment')
    parser.add_argument('--community_exclusivity', type=float, default=1.0,
                        help='Exclusivity for community assignment')
    parser.add_argument('--degree_center_method', type=str, default='linear',
                        choices=["linear", "random", "shuffled"],
                        help='Degree center method')

    # Transformer-specific parameters
    parser.add_argument('--transformer_num_heads', type=int, default=3,
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
    
    # GraphGPS specific
    parser.add_argument('--local_gnn_type', type=str, default='gcn',
                        choices=['gcn', 'sage'],
                        help='Local GNN type for GraphGPS')
    parser.add_argument('--global_model_type', type=str, default='performer',
                        help='Global model type for GraphGPS')
    parser.add_argument('--transformer_prenorm', action='store_true', default=True,
                        help='Use pre-normalization in transformers')

    return parser.parse_args()


def create_config_from_args(args) -> InductiveExperimentConfig:
    """Create clean configuration from command line arguments."""    
    config = InductiveExperimentConfig(
        # === EXPERIMENT SETUP ===
        output_dir=args.output_dir,
        seed=args.seed,
        device_id=args.device_id,
        force_cpu=args.force_cpu,
        use_parallel_training=args.use_parallel_training,
        max_parallel_gpu_jobs=args.max_parallel_gpu_jobs,
        cross_task_parallelization=args.cross_task_parallelization,
        
        # === SSL FINE-TUNING SETUP ===
        use_pretrained=args.use_pretrained,
        pretrained_model_dir=args.pretrained_model_dir,
        pretrained_model_id=getattr(args, 'pretrained_model_id', None),
        graph_family_id=getattr(args, 'graph_family_id', None),
        graph_family_dir=getattr(args, 'graph_family_dir', 'graph_families'),
        auto_load_family=getattr(args, 'auto_load_family', True),
        freeze_encoder=args.freeze_encoder,
        calculate_silhouette_score=args.calculate_silhouette_score,
        only_pretrained_experiments=args.only_pretrained_experiments,
        max_train_graphs_for_finetuning=getattr(args, 'max_train_graphs_for_finetuning', 5),
        minimum_train_graphs_to_cover_k=False,

        # === TASKS ===
        tasks=args.tasks,
        khop_community_counts_k=args.khop_k,

        # === DATA SPLITS ===
        allow_unseen_community_combinations_for_eval=False if args.no_unseen_community_combinations_for_eval else True,

        # === METAPATH CONFIGURATION ===
        enable_metapath_tasks=args.enable_metapath_tasks,
        metapath_k_values=args.metapath_k_values,
        metapath_require_loop=args.metapath_require_loop,
        metapath_degree_weight=args.metapath_degree_weight,
        max_community_participation=args.max_community_participation,
        
        # === GRAPH FAMILY GENERATION ===
        n_graphs=args.n_graphs,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=args.min_communities,
        max_communities=args.max_communities,
        
        # === UNIVERSE PARAMETERS ===
        universe_K=args.universe_K,
        universe_feature_dim=args.universe_feature_dim,
        universe_edge_density=args.universe_edge_density,
        universe_homophily=args.universe_homophily,
        universe_randomness_factor=args.universe_randomness_factor,
        
        # === METHOD SELECTION ===
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,
        
        # === GRAPH FAMILY VARIATION ===
        homophily_range=tuple(args.homophily_range),
        density_range=tuple(args.density_range),
        degree_heterogeneity=args.degree_heterogeneity,
        edge_noise=args.edge_noise,
        
        # === DCCC-SBM PARAMETERS ===
        community_imbalance_range=tuple(args.community_imbalance_range),
        degree_separation_range=tuple(args.degree_separation_range),

        # === FEATURE GENERATION ===
        cluster_count_factor=args.cluster_count_factor,
        center_variance=args.center_variance,
        cluster_variance=args.cluster_variance,
        assignment_skewness=args.assignment_skewness,
        community_exclusivity=args.community_exclusivity,
        degree_center_method=args.degree_center_method,
        
        # === MODELS ===
        gnn_types=args.gnn_types,
        run_gnn=args.run_gnn,
        run_mlp=args.run_mlp,
        run_rf=args.run_rf,

        # === GRAPH TRANSFORMER CONFIGURATION ===
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

        # === NEURAL SHEAF DIFFUSION ===
        run_neural_sheaf=args.run_neural_sheaf,
        
        # === TRAINING ===
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        
        # === ANALYSIS ===
        require_consistency_check=args.require_consistency_check,
        collect_signal_metrics=args.collect_signal_metrics,
        optimize_hyperparams=args.optimize_hyperparams,
    )
    
    return config


def main():
    """Main function to run clean inductive experiments."""
    print("CLEAN INDUCTIVE GRAPH LEARNING EXPERIMENTS")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration summary
    print("\nExperiment Configuration:")
    print(f"  Method: {'DCCC-SBM' if config.use_dccc_sbm else 'DC-SBM'}")
    if config.use_dccc_sbm:
        print(f"  Degree distribution: {config.degree_distribution}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Graph family size: {config.n_graphs}")
    print(f"  Node range: [{config.min_n_nodes}, {config.max_n_nodes}]")
    print(f"  Community range: [{config.min_communities}, {config.max_communities}]")
    print(f"  Tasks: {', '.join(config.tasks)}")
    
    models = []
    if config.run_gnn:
        models.extend(config.gnn_types)
    if config.run_mlp:
        models.append("mlp")
    if config.run_rf:
        models.append("rf")
    print(f"  Models: {', '.join(models)}")
    print(f"  Collect signals: {config.collect_signal_metrics}")
    print(f"  Random seed: {config.seed}")
    
    if config.use_parallel_training:
        parallelization_mode = "Cross-task" if config.cross_task_parallelization else "Per-task"
        print(f"  Parallel training: {parallelization_mode}")
        if config.max_parallel_gpu_jobs:
            print(f"  Max parallel jobs: {config.max_parallel_gpu_jobs}")
    
    try:
        # Run experiment
        print(f"\nStarting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = run_inductive_experiment(config)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
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