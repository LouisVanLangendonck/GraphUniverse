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
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pretrain', 'finetune', 'list_families', 'list_models'],
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
    parser.add_argument('--n_graphs', type=int, default=50,
                        help='Number of graphs for pre-training')
    parser.add_argument('--n_extra_graphs', type=int, default=30,
                        help='Extra graphs to generate for fine-tuning')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum number of nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=120,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--min_communities', type=int, default=3,
                        help='Minimum number of communities per graph')
    parser.add_argument('--max_communities', type=int, default=7,
                        help='Maximum number of communities per graph')
    
    # === UNIVERSE PARAMETERS ===
    parser.add_argument('--universe_K', type=int, default=10,
                        help='Number of communities in universe')
    parser.add_argument('--universe_feature_dim', type=int, default=32,
                        help='Feature dimension for universe')
    parser.add_argument('--universe_edge_density', type=float, default=0.1,
                        help='Base edge density for universe')
    parser.add_argument('--universe_homophily', type=float, default=0.8,
                        help='Homophily parameter for universe')
    
    # === METHOD SELECTION ===
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM instead of standard DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution type for DCCC-SBM')
    
    # === PRE-TRAINING TASK ===
    parser.add_argument('--pretraining_task', type=str, default='link_prediction',
                        choices=['link_prediction', 'contrastive'],
                        help='Self-supervised pre-training task')
    
    # === MODEL CONFIGURATION ===
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gat'],
                        help='Type of GNN to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # === TRAINING PARAMETERS ===
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    
    # === HYPERPARAMETER OPTIMIZATION ===
    parser.add_argument('--optimize_hyperparams', action='store_true', default=True,
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
    
    # Contrastive learning
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning')
    parser.add_argument('--corruption_type', type=str, default='feature_shuffle',
                        choices=['feature_shuffle', 'edge_dropout'],
                        help='Type of corruption for contrastive learning')
    parser.add_argument('--corruption_rate', type=float, default=0.2,
                        help='Corruption rate for contrastive learning')
    
    # === GRAPH FAMILY MANAGEMENT ===
    parser.add_argument('--family_id', type=str, default=None,
                        help='ID of existing graph family to use')
    parser.add_argument('--use_existing_family', action='store_true',
                        help='Use existing graph family instead of generating new one')
    parser.add_argument('--graph_family_dir', type=str, default='graph_families',
                        help='Directory for graph families')
    
    # === FINE-TUNING SPECIFIC ===
    parser.add_argument('--model_id', type=str, default=None,
                        help='ID of pre-trained model to fine-tune (for finetune mode)')
    parser.add_argument('--tasks', type=str, nargs='+', default=['community'],
                        choices=['community', 'k_hop_community_counts', 'metapath'],
                        help='Downstream tasks for fine-tuning')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights during fine-tuning')
    parser.add_argument('--fine_tune_lr_multiplier', type=float, default=0.1,
                        help='Learning rate multiplier for fine-tuning')
    
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
        negative_sampling_ratio=args.negative_sampling_ratio,
        link_pred_loss=args.link_pred_loss,
        contrastive_temperature=args.contrastive_temperature,
        corruption_type=args.corruption_type,
        corruption_rate=args.corruption_rate
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


def run_finetuning_mode(args):
    """Run fine-tuning mode using pre-trained models."""
    print("="*80)
    print("FINE-TUNING MODE")
    print("="*80)
    
    if not args.model_id:
        print("Error: --model_id required for fine-tuning mode")
        return None
    
    # Load pre-trained model and associated graph family
    print(f"Loading pre-trained model: {args.model_id}")
    
    # Import necessary modules for fine-tuning
    from experiments.inductive.config import InductiveExperimentConfig
    from experiments.inductive.experiment import InductiveExperiment
    from experiments.inductive.data import PreTrainedModelSaver
    
    # Load pre-trained model
    model_saver = PreTrainedModelSaver(args.output_dir)
    try:
        model, metadata = model_saver.load_model(args.model_id)
        print(f"✓ Successfully loaded pre-trained model")
        print(f"  Architecture: {metadata['architecture']}")
        print(f"  Pre-training task: {metadata['config']['pretraining_task']}")
        print(f"  Final metrics: {metadata['final_metrics']}")
        
        # Get associated graph family info
        family_id = metadata.get('family_id')
        if not family_id:
            print("Error: Pre-trained model has no associated graph family")
            return None
        
        print(f"  Graph family: {family_id}")
        
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None
    
    # Load fine-tuning graphs
    print(f"Loading fine-tuning graphs from family: {family_id}")
    
    family_manager = GraphFamilyManager(PreTrainingConfig(graph_family_dir=args.graph_family_dir))
    try:
        family_graphs, family_metadata = family_manager.load_family(family_id)
        graph_splits = family_manager.get_graph_splits(family_graphs, family_metadata)
        finetuning_graphs = graph_splits['finetuning']
        
        print(f"✓ Loaded {len(finetuning_graphs)} graphs for fine-tuning")
        
    except Exception as e:
        print(f"Error loading graph family: {e}")
        return None
    
    # Create inductive experiment configuration for fine-tuning
    inductive_config = InductiveExperimentConfig(
        # Use same basic parameters as pre-training
        output_dir=os.path.join(args.output_dir, f"finetune_{args.model_id}"),
        seed=args.seed,
        device_id=args.device_id,
        force_cpu=args.force_cpu,
        
        # Use the fine-tuning graphs
        n_graphs=len(finetuning_graphs),
        min_n_nodes=min(g.n_nodes for g in finetuning_graphs),
        max_n_nodes=max(g.n_nodes for g in finetuning_graphs),
        
        # Tasks
        tasks=args.tasks,
        
        # Model configuration (match pre-trained model)
        gnn_types=[metadata['config']['gnn_type']],
        run_gnn=True,
        run_mlp=False,  # Only use GNN for fine-tuning
        run_rf=False,
        
        # Training configuration
        learning_rate=metadata['config']['learning_rate'] * args.fine_tune_lr_multiplier,
        weight_decay=metadata['config']['weight_decay'],
        epochs=100,  # Fewer epochs for fine-tuning
        patience=20,
        batch_size=2,
        hidden_dim=metadata['config']['hidden_dim'],
        num_layers=metadata['config']['num_layers'],
        dropout=metadata['config']['dropout'],
        
        # Analysis
        collect_signal_metrics=True,
        require_consistency_check=False
    )
    
    print(f"\nFine-tuning Configuration:")
    print(f"  Tasks: {inductive_config.tasks}")
    print(f"  Fine-tuning graphs: {len(finetuning_graphs)}")
    print(f"  Learning rate: {inductive_config.learning_rate}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    
    # Run fine-tuning experiment
    # Note: This would require extending the InductiveExperiment class to support pre-trained models
    # For now, we'll create a placeholder
    
    print("\n⚠️  Fine-tuning implementation requires extending InductiveExperiment class")
    print("    to support loading pre-trained models. This is a TODO item.")
    
    return {
        'model_id': args.model_id,
        'family_id': family_id,
        'n_finetuning_graphs': len(finetuning_graphs),
        'tasks': args.tasks,
        'status': 'pending_implementation'
    }


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
            
        elif args.mode == 'finetune':
            results = run_finetuning_mode(args)
            
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