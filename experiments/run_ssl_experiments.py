#!/usr/bin/env python3
"""
Self-supervised learning experiments for graph neural networks.
Supports link prediction and contrastive learning pre-training tasks.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.config import PreTrainingConfig
from experiments.inductive.experiment import PreTrainingRunner
from experiments.inductive.data import GraphFamilyManager, PreTrainedModelSaver


def parse_args():
    """Parse command line arguments for SSL experiments."""
    parser = argparse.ArgumentParser(description='Run self-supervised learning experiments')
    
    # === MODE SELECTION ===
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'list_families', 'list_models'],
                        help='Experiment mode')
    
    # === EXPERIMENT SETUP ===
    parser.add_argument('--output_dir', type=str, default='ssl_experiments',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default='ssl_experiment',
                        help='Name for this experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID to use')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    # === GRAPH FAMILY GENERATION ===
    parser.add_argument('--n_graphs', type=int, default=200,
                        help='Number of graphs for pre-training')
    parser.add_argument('--n_extra_graphs', type=int, default=30,
                        help='Extra graphs to generate for fine-tuning')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum number of nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=150,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--min_communities', type=int, default=4,
                        help='Minimum number of communities per graph')
    parser.add_argument('--max_communities', type=int, default=10,
                        help='Maximum number of communities per graph')
    
    # === UNIVERSE PARAMETERS ===
    parser.add_argument('--universe_K', type=int, default=20,
                        help='Number of communities in universe')
    parser.add_argument('--universe_feature_dim', type=int, default=32,
                        help='Feature dimension for universe')
    parser.add_argument('--universe_edge_density', type=float, default=0.05,
                        help='Base edge density for universe')
    parser.add_argument('--universe_homophily', type=float, default=0.4,
                        help='Homophily parameter for universe')
    
    # === METHOD SELECTION ===
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM instead of standard DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution type for DCCC-SBM')
    
    # === PRE-TRAINING TASK ===
    parser.add_argument('--pretraining_task', type=str, default='graphmae',
                        choices=['link_prediction', 'dgi', 'graphmae'],
                        help='Self-supervised pre-training task')
    
    # === MODEL CONFIGURATION ===
    parser.add_argument('--gnn_type', type=str, default='gin',
                        choices=['gcn', 'sage', 'gat', 'fagcn', 'gin'],
                        help='Type of GNN to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # === GRAPH TRANSFORMER CONFIGURATION ===
    parser.add_argument('--transformer_type', type=str, default='graphgps',
                        choices=['graphormer', 'graphgps'],
                        help='Type of Graph Transformer to use')
    parser.add_argument('--run_transformers', action='store_true',
                        help='Run Graph Transformer models')
    parser.add_argument('--transformer_num_heads', type=int, default=4,
                        help='Number of attention heads for transformers')
    parser.add_argument('--transformer_max_nodes', type=int, default=50,
                        help='Maximum nodes for encoding precomputation')
    parser.add_argument('--transformer_max_path_length', type=int, default=5,
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
    
    # === TRAINING PARAMETERS ===
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping')
    
    # === HYPERPARAMETER OPTIMIZATION ===
    parser.add_argument('--optimize_hyperparams', action='store_true', default=False,
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--optimization_timeout', type=int, default=1200,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # === TASK-SPECIFIC PARAMETERS ===
    # Link prediction
    parser.add_argument('--negative_sampling_ratio', type=float, default=1.0,
                        help='Negative sampling ratio for link prediction')
    parser.add_argument('--link_pred_loss', type=str, default='bce',
                        choices=['bce', 'margin'],
                        help='Loss function for link prediction')
    
    # DGI
    parser.add_argument('--dgi_corruption_type', type=str, default='edge_dropout',
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
    
    # === GRAPH FAMILY MANAGEMENT ===
    parser.add_argument('--family_id', type=str, default=None,
                        help='ID of existing graph family to use')
    parser.add_argument('--use_existing_family', action='store_true',
                        help='Use existing graph family instead of generating new one')
    parser.add_argument('--graph_family_dir', type=str, default='graph_families',
                        help='Directory for graph families')
    
    return parser.parse_args()


def create_config_from_args(args) -> PreTrainingConfig:
    """Create pre-training configuration from command line arguments."""
    
    config = PreTrainingConfig(
        # === EXPERIMENT SETUP ===
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        device_id=args.device_id,
        force_cpu=args.force_cpu,
        
        # === PRE-TRAINING TASK ===
        pretraining_task=args.pretraining_task,
        
        # === GRAPH FAMILY PERSISTENCE ===
        n_extra_graphs_for_finetuning=args.n_extra_graphs,
        save_graph_family=True,
        graph_family_dir=args.graph_family_dir,
        
        # === FAMILY GENERATION ===
        n_graphs=args.n_graphs,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=args.min_communities,
        max_communities=args.max_communities,
        universe_K=args.universe_K,
        universe_feature_dim=args.universe_feature_dim,
        universe_edge_density=args.universe_edge_density,
        universe_homophily=args.universe_homophily,
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,
        
        # === MODEL CONFIGURATION ===
        gnn_type=args.gnn_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        
        # === GRAPH TRANSFORMER CONFIGURATION ===
        transformer_type=args.transformer_type,
        run_transformers=args.run_transformers,
        transformer_num_heads=args.transformer_num_heads,
        transformer_max_nodes=args.transformer_max_nodes,
        transformer_max_path_length=args.transformer_max_path_length,
        transformer_precompute_encodings=args.transformer_precompute_encodings,
        transformer_cache_encodings=getattr(args, 'transformer_cache_encodings', True),
        local_gnn_type=args.local_gnn_type,
        global_model_type=args.global_model_type,
        transformer_prenorm=getattr(args, 'transformer_prenorm', True),
        
        # === TRAINING PARAMETERS ===
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        
        # === HYPERPARAMETER OPTIMIZATION ===
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        
        # === TASK-SPECIFIC PARAMETERS ===
        # Link prediction
        negative_sampling_ratio=args.negative_sampling_ratio,
        link_pred_loss=args.link_pred_loss,
        contrastive_temperature=args.contrastive_temperature,
        
        # DGI
        dgi_corruption_type=args.dgi_corruption_type,
        dgi_noise_std=args.dgi_noise_std,
        dgi_perturb_rate=args.dgi_perturb_rate,
        dgi_corruption_rate=args.dgi_corruption_rate,

        # GraphMAE
        graphmae_mask_rate=args.graphmae_mask_rate,
        graphmae_replace_rate=args.graphmae_replace_rate,
        graphmae_gamma=args.graphmae_gamma,
        graphmae_decoder_type=args.graphmae_decoder_type,
        graphmae_decoder_gnn_type=args.graphmae_decoder_gnn_type,

    )
    
    return config


def run_pretraining_mode(args):
    """Run pre-training mode."""
    print("="*80)
    print("SELF-SUPERVISED PRE-TRAINING MODE")
    print("="*80)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print configuration summary
    print(f"\nPre-training Configuration:")
    print(f"  Task: {config.pretraining_task}")
    print(f"  GNN Type: {config.gnn_type}")
    print(f"  Method: {'DCCC-SBM' if config.use_dccc_sbm else 'DC-SBM'}")
    if config.use_dccc_sbm:
        print(f"  Degree distribution: {config.degree_distribution}")
    print(f"  Pre-training graphs: {config.n_graphs}")
    print(f"  Extra graphs for fine-tuning: {config.n_extra_graphs_for_finetuning}")
    print(f"  Total graphs: {config.get_total_graphs()}")
    print(f"  Hyperparameter optimization: {config.optimize_hyperparams}")
    if config.optimize_hyperparams:
        print(f"    Trials: {config.n_trials}")
        print(f"    Timeout: {config.optimization_timeout}s")
    
    # Run pre-training
    runner = PreTrainingRunner(config)
    results = runner.run_pretraining_only(
        family_id=args.family_id,
        use_existing_family=args.use_existing_family
    )
    
    print(f"\n✓ Pre-training completed successfully!")
    print(f"Model ID: {results['model_id']}")
    print(f"Graph Family ID: {results['family_id']}")
    
    return results

def list_families_mode(args):
    """List available graph families."""
    print("="*60)
    print("AVAILABLE GRAPH FAMILIES")
    print("="*60)
    
    family_manager = GraphFamilyManager(PreTrainingConfig(graph_family_dir=args.graph_family_dir))
    families = family_manager.list_families()
    
    if not families:
        print("No graph families found.")
        return
    
    for i, family in enumerate(families, 1):
        print(f"\n{i}. Family ID: {family['family_id']}")
        print(f"   Created: {family['creation_timestamp']}")
        print(f"   Total graphs: {family['total_graphs']}")
        print(f"   Fine-tuning graphs: {family.get('n_finetuning', 0)}")
        
        config = family.get('config', {})
        if config:
            print(f"   Method: {'DCCC-SBM' if config.get('use_dccc_sbm') else 'DC-SBM'}")
            print(f"   Universe K: {config.get('universe_K', 'N/A')}")


def list_models_mode(args):
    """List available pre-trained models."""
    print("="*60)
    print("AVAILABLE PRE-TRAINED MODELS")
    print("="*60)
    
    model_saver = PreTrainedModelSaver(args.output_dir)
    models = model_saver.list_models()
    
    if not models:
        print("No pre-trained models found.")
        return
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. Model ID: {model['model_id']}")
        print(f"   Created: {model['creation_timestamp']}")
        print(f"   Task: {model['task']}")
        print(f"   GNN Type: {model['gnn_type']}")
        print(f"   Graph Family: {model.get('family_id', 'N/A')}")
        print(f"   Fine-tuning ready: {model.get('finetuning_ready', False)}")
        
        final_metrics = model.get('final_metrics', {})
        if final_metrics:
            if model['task'] == 'link_prediction':
                auc = final_metrics.get('auc', 'N/A')
                print(f"   AUC: {auc}")
            elif model['task'] == 'contrastive':
                acc = final_metrics.get('accuracy', 'N/A')
                print(f"   Accuracy: {acc}")


def main():
    """Main function to run SSL experiments."""
    args = parse_args()
    
    print("GRAPH NEURAL NETWORK SELF-SUPERVISED LEARNING")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Random seed: {args.seed}")
    
    try:
        if args.mode == 'pretrain':
            results = run_pretraining_mode(args)
            
        elif args.mode == 'list_families':
            list_families_mode(args)
            results = None
            
        elif args.mode == 'list_models':
            list_models_mode(args)
            results = None
            
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
        
        if results:
            print(f"\n✓ {args.mode} completed successfully!")
            
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{args.mode} interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n{args.mode} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)