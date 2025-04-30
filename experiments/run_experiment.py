"""
MMSB Graph Learning Experiment

This script runs a comparative experiment on a graph generated using MMSB,
training GNN, MLP, and Random Forest models for node classification.

Usage:
    python run_experiment.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MMSB modules
from mmsb.model import GraphUniverse, GraphSample

# Import experiment modules
from models import GNNModel, MLPModel, SklearnModel
from training import train_gnn_model, train_mlp_model, train_sklearn_model, split_node_indices
from metrics import model_performance_summary, compare_models
from graph_data import graph_sample_to_pyg, create_sklearn_compatible_data, generate_graph_statistics
from utils.parameter_analysis import analyze_graph_parameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MMSB Graph Learning Experiment')
    
    # Graph generation parameters
    parser.add_argument('--num_communities', type=int, default=10,
                        help='Number of communities in the universe')
    parser.add_argument('--num_nodes', type=int, default=70,
                        help='Number of nodes in the graph')
    parser.add_argument('--feature_dim', type=int, default=32,
                        help='Dimension of node features')
    parser.add_argument('--edge_density', type=float, default=0.1,
                        help='Edge density within communities')
    parser.add_argument('--inter_community_density', type=float, default=0.04,
                        help='Edge density between communities')
    parser.add_argument('--homophily', type=float, default=0.8,
                        help='Target homophily level (0-1)')
    parser.add_argument('--feature_signal', type=float, default=0.25,
                        help='How strongly features correlate with community membership (0=random, 1=perfect)')
    parser.add_argument('--randomness_factor', type=float, default=0.6,
                        help='Amount of random variation in edge probabilities (0=deterministic, 1=highly random)')
    parser.add_argument('--overlap_density', type=float, default=0.2,
                        help='Density of community overlaps')
    parser.add_argument('--min_connection_strength', type=float, default=0.04,
                        help='Minimum edge probability between communities')
    parser.add_argument('--min_component_size', type=int, default=4,
                        help='Minimum size for a component to be kept (0 keeps all)')
    parser.add_argument('--degree_heterogeneity', type=float, default=0.5,
                        help='Controls variability in node degrees (0=homogeneous, 1=highly skewed)')
    parser.add_argument('--indirect_influence', type=float, default=0.1,
                        help='How strongly co-memberships influence edge formation (0=no effect, 0.5=strong effect)')
    parser.add_argument('--overlap_structure', type=str, default='modular',
                        choices=['modular', 'hierarchical', 'hub-spoke'],
                        help='Structure of community overlaps')
    
    # Training parameters
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Ratio of nodes for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of nodes for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of nodes for testing')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for neural networks')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in neural networks')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    
    # Model selection - now defaults to running all models
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gat', 'gcn', 'sage'],
                        choices=['gcn', 'gat', 'sage'],
                        help='Types of GNN models to run')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip GNN models')
    parser.add_argument('--skip_mlp', action='store_true',
                        help='Skip MLP model')
    parser.add_argument('--skip_rf', action='store_true',
                        help='Skip Random Forest model')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Set run flags based on skip flags
    args.run_gnn = not args.skip_gnn
    args.run_mlp = not args.skip_mlp
    args.run_rf = not args.skip_rf
    
    return args


def generate_graph(args):
    """Generate a graph using MMSB."""
    print(f"Generating graph with {args.num_nodes} nodes and {args.num_communities} communities...")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Initialize device
    device = None
    try:
        if torch.cuda.is_available():
            # Get number of available CUDA devices
            num_devices = torch.cuda.device_count()
            if num_devices > 0:
                # Use first available device
                device = torch.device(f'cuda:0')
                torch.cuda.set_device(device)
                # Clear CUDA cache
                torch.cuda.empty_cache()
                print(f"Successfully initialized CUDA device {device}")
            else:
                print("No CUDA devices available")
                device = torch.device('cpu')
        else:
            print("CUDA is not available")
            device = torch.device('cpu')
    except Exception as e:
        print(f"Warning: CUDA initialization failed: {str(e)}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
    
    # Set seeds after device initialization
    try:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"Successfully set random seeds for {device.type}")
    except Exception as e:
        print(f"Warning: Setting random seeds failed: {str(e)}")
        print("Continuing with default seeds...")
    
    # Validate parameters
    try:
        # Create the universe with all communities
        universe = GraphUniverse(
            K=args.num_communities,
            feature_dim=args.feature_dim,
            feature_signal=args.feature_signal,
            block_structure="assortative",
            edge_density=args.edge_density,
            inter_community_density=args.inter_community_density,
            randomness_factor=args.randomness_factor
        )
        
        # Generate community co-membership matrix
        co_membership = universe.generate_community_co_membership_matrix(
            overlap_density=args.overlap_density,
            structure=args.overlap_structure
        )
        universe.community_co_membership = co_membership
        
        # Generate a graph with all communities
        communities = list(range(args.num_communities))
        
        # Adjust edge densities to target homophily if specified
        if args.homophily > 0:
            # Simple adjustment: increase intra-community density relative to inter-community
            if args.homophily > 0.5:
                factor = (args.homophily - 0.5) * 2  # Scale from 0 to 1
                universe.P = universe.P.copy()
                
                # Increase diagonal (intra-community) values
                np.fill_diagonal(universe.P, min(1.0, args.edge_density * (1 + factor)))
                
                # Decrease off-diagonal (inter-community) values
                mask = ~np.eye(args.num_communities, dtype=bool)
                universe.P[mask] = args.inter_community_density * (1 - factor/2)
        
        # Create the graph sample
        graph_sample = GraphSample(
            universe=universe,
            communities=communities,
            n_nodes=args.num_nodes,
            min_component_size=args.min_component_size,
            degree_heterogeneity=args.degree_heterogeneity,
            edge_noise=0.0,
            indirect_influence=args.indirect_influence
        )
        
        print(f"Generated graph with {graph_sample.n_nodes} nodes and {graph_sample.graph.number_of_edges()} edges")
        
        # Generate real graph properties using analyze_graph_parameters
        real_graph_properties = analyze_graph_parameters(
            graph=graph_sample.graph,
            membership_vectors=graph_sample.membership_vectors,
            communities=communities
        )
        
        # Print real graph properties
        print("\nReal Graph Properties:")
        for key, value in real_graph_properties.items():
            print(f"  {key}: {value:.4f}")
        
        # Store real graph properties in the graph sample for later use
        graph_sample.real_graph_properties = real_graph_properties
        
        return graph_sample
        
    except Exception as e:
        print(f"Error generating graph: {str(e)}")
        print("Parameters used:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        raise


def prepare_data(graph_sample, args):
    """Prepare data for model training."""
    # Initialize device
    device = None
    try:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            if num_devices > 0:
                device = torch.device(f'cuda:0')
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
                print(f"Using CUDA device {device} for data preparation")
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    except Exception as e:
        print(f"Warning: CUDA initialization failed: {str(e)}")
        print("Falling back to CPU for data preparation...")
        device = torch.device('cpu')
    
    # Convert graph to PyTorch tensors
    try:
        features, edge_index, labels = graph_sample_to_pyg(graph_sample, feature_type="generated")
        
        # Move tensors to device
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)
        
        # Create scikit-learn compatible data
        X, y = create_sklearn_compatible_data(features, edge_index, labels)
        
        # Split indices
        indices = split_node_indices(
            graph_sample.n_nodes,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=y,
            random_state=args.seed
        )
        
        train_idx, val_idx, test_idx = [torch.tensor(idx, device=device) for idx in indices]
        
        # Print data statistics
        num_classes = len(torch.unique(labels))
        print(f"\nData Statistics:")
        print(f"  Number of nodes: {len(features)}")
        print(f"  Number of edges: {edge_index.shape[1] // 2}")  # Divide by 2 for undirected
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Class distribution: {torch.bincount(labels)}")
        print(f"  Training nodes: {len(train_idx)}")
        print(f"  Validation nodes: {len(val_idx)}")
        print(f"  Test nodes: {len(test_idx)}")
        print(f"  Device: {device}")
        
        return features, edge_index, labels, train_idx, val_idx, test_idx, num_classes
        
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        raise


def run_models(features, edge_index, labels, train_idx, val_idx, test_idx, num_classes, args):
    """Train and evaluate models."""
    model_results = {}
    
    # Run GNN models
    if args.run_gnn:
        for gnn_type in args.gnn_types:
            try:
                print(f"\nTraining {gnn_type.upper()} model...")
                print(f"Input dimensions: features={features.shape}, edge_index={edge_index.shape}")
                print(f"Label distribution: {torch.bincount(labels)}")
                
                gnn_model = GNNModel(
                    input_dim=features.shape[1],
                    hidden_dim=args.hidden_dim,
                    output_dim=num_classes,
                    num_layers=args.num_layers,
                    dropout=0.5,
                    gnn_type=gnn_type
                )
                
                print(f"Model architecture: {gnn_model}")
                
                gnn_results = train_gnn_model(
                    model=gnn_model,
                    features=features,
                    edge_index=edge_index,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    epochs=args.epochs,
                    lr=args.learning_rate,
                    patience=args.patience
                )
                
                print(f"{gnn_type.upper()} Test Accuracy: {gnn_results['test_acc']:.4f}")
                print(f"{gnn_type.upper()} F1 Score (Macro): {gnn_results['metrics']['metrics_macro']['f1']:.4f}")
                
                model_results[gnn_type.upper()] = gnn_results
            except Exception as e:
                print(f"Error training {gnn_type.upper()} model: {str(e)}")
                print("Traceback:")
                import traceback
                traceback.print_exc()
    
    # Run MLP model
    if args.run_mlp:
        try:
            print("\nTraining MLP model...")
            print(f"Input dimensions: features={features.shape}")
            print(f"Label distribution: {torch.bincount(labels)}")
            
            mlp_model = MLPModel(
                input_dim=features.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                num_layers=args.num_layers,
                dropout=0.5
            )
            
            print(f"Model architecture: {mlp_model}")
            
            mlp_results = train_mlp_model(
                model=mlp_model,
                features=features,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                epochs=args.epochs,
                lr=args.learning_rate,
                patience=args.patience
            )
            
            print(f"MLP Test Accuracy: {mlp_results['test_acc']:.4f}")
            print(f"MLP F1 Score (Macro): {mlp_results['metrics']['metrics_macro']['f1']:.4f}")
            
            model_results['MLP'] = mlp_results
        except Exception as e:
            print(f"Error training MLP model: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
    
    # Run Random Forest model
    if args.run_rf:
        try:
            print("\nTraining Random Forest model...")
            print(f"Input dimensions: features={features.shape}")
            print(f"Label distribution: {torch.bincount(labels)}")
            
            rf_model = SklearnModel(
                input_dim=features.shape[1],
                output_dim=num_classes,
                model_type="random_forest",
                n_estimators=100,
                max_depth=None,
                random_state=args.seed
            )
            
            # Convert data to numpy
            X_np = features.numpy()
            y_np = labels.numpy()
            
            rf_results = train_sklearn_model(
                model=rf_model.model,
                features=X_np,
                labels=y_np,
                train_idx=train_idx.numpy(),
                val_idx=val_idx.numpy(),
                test_idx=test_idx.numpy()
            )
            
            print(f"RF Test Accuracy: {rf_results['test_acc']:.4f}")
            print(f"RF F1 Score (Macro): {rf_results['metrics']['metrics_macro']['f1']:.4f}")
            
            model_results['RandomForest'] = rf_results
        except Exception as e:
            print(f"Error training Random Forest model: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
    
    # Compare models
    if len(model_results) > 1:
        try:
            comparison = compare_models(
                {name: results['metrics'] for name, results in model_results.items()},
                metric='f1'
            )
            
            print("\nModel Comparison:")
            print(f"  Best model: {comparison['best_model']} (F1: {comparison['best_score']:.4f})")
            
            print("  Relative improvement:")
            for model, improvement in comparison['relative_improvement'].items():
                print(f"    {model}: {improvement:.2f}%")
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
    
    return model_results


def visualize_results(graph_sample, model_results, args):
    """Create visualizations of results."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Visualize graph
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            # Get primary community for each node
            primary_communities = np.argmax(graph_sample.membership_vectors, axis=1)
            
            # Plot graph colored by community
            plt.figure(figsize=(10, 8))
            
            # For large graphs, use a smaller subset for visualization
            if graph_sample.n_nodes > 500:
                # Sample nodes for visualization
                vis_size = 500
                sampled_nodes = np.random.choice(graph_sample.n_nodes, size=vis_size, replace=False)
                subgraph = graph_sample.graph.subgraph(sampled_nodes)
                node_colors = [primary_communities[i] for i in sampled_nodes]
            else:
                subgraph = graph_sample.graph
                node_colors = primary_communities
            
            # Position nodes using layout algorithm
            pos = nx.spring_layout(subgraph, seed=args.seed)
            
            # Draw nodes colored by community
            nx.draw_networkx_nodes(
                subgraph, pos,
                node_color=node_colors,
                cmap=plt.cm.tab10,
                node_size=50,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph, pos,
                alpha=0.1,
                width=0.5
            )
            
            plt.title(f"MMSB Graph (colored by community)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "graph_visualization.png"), dpi=300)
            plt.close()
            
            # 2. Visualize training history for neural models
            plt.figure(figsize=(12, 5))
            
            for i, (name, results) in enumerate(['GCN', 'GAT', 'SAGE', 'MLP']):
                if name in model_results:
                    history = model_results[name]['history']
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(history['train_loss'], label=f"{name} Train")
                    plt.plot(history['val_loss'], linestyle='--', label=f"{name} Val")
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(history['train_acc'], label=f"{name} Train")
                    plt.plot(history['val_acc'], linestyle='--', label=f"{name} Val")
            
            plt.subplot(1, 2, 1)
            plt.title("Loss vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.title("Accuracy vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "training_history.png"), dpi=300)
            plt.close()
            
            # 3. Visualize model comparison
            if len(model_results) > 1:
                # Extract metrics
                accuracies = []
                f1_scores = []
                train_times = []
                model_names = []
                
                for name, results in model_results.items():
                    model_names.append(name)
                    accuracies.append(results.get('test_acc', 0))
                    f1_scores.append(results.get('metrics', {}).get('metrics_macro', {}).get('f1', 0))
                    train_times.append(results.get('train_time', 0))
                
                # Create bar chart
                plt.figure(figsize=(12, 6))
                
                x = np.arange(len(model_names))
                width = 0.3
                
                plt.subplot(1, 2, 1)
                plt.bar(x - width/2, accuracies, width, label='Accuracy')
                plt.bar(x + width/2, f1_scores, width, label='F1 Score')
                
                plt.xlabel('Model')
                plt.ylabel('Score')
                plt.title('Model Performance Comparison')
                plt.xticks(x, model_names)
                plt.ylim(0, 1)
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.bar(x, train_times)
                plt.xlabel('Model')
                plt.ylabel('Training Time (s)')
                plt.title('Training Time Comparison')
                plt.xticks(x, model_names)
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "model_comparison.png"), dpi=300)
                plt.close()
        
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    # Save results to JSON
    results_summary = {
        "graph_stats": generate_graph_statistics(graph_sample),
        "model_results": {}
    }
    
    for name, results in model_results.items():
        # Extract serializable results
        serializable_results = {
            "test_acc": results.get('test_acc', 0),
            "train_time": results.get('train_time', 0),
            "metrics": {
                "accuracy": results.get('metrics', {}).get('metrics_macro', {}).get('accuracy', 0),
                "precision": results.get('metrics', {}).get('metrics_macro', {}).get('precision', 0),
                "recall": results.get('metrics', {}).get('metrics_macro', {}).get('recall', 0),
                "f1": results.get('metrics', {}).get('metrics_macro', {}).get('f1', 0)
            }
        }
        
        # Add training history for neural models
        if 'history' in results:
            serializable_results["history"] = {
                "train_loss": results['history']['train_loss'],
                "val_loss": results['history']['val_loss'],
                "train_acc": results['history']['train_acc'],
                "val_acc": results['history']['val_acc']
            }
        
        results_summary["model_results"][name] = serializable_results
    
    with open(os.path.join(args.output_dir, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)


def plot_results(results_df, output_dir):
    """Plot the results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert NA values to 0 for plotting
    results_df = results_df.fillna(0)
    
    # Plot accuracy vs. homophily
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='homophily', y='accuracy', hue='model', style='model', markers=True)
    plt.title('Accuracy vs. Homophily')
    plt.xlabel('Homophily')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_homophily.png'))
    plt.close()
    
    # Plot modularity vs. homophily
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='homophily', y='modularity', hue='model', style='model', markers=True)
    plt.title('Modularity vs. Homophily')
    plt.xlabel('Homophily')
    plt.ylabel('Modularity')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'modularity_vs_homophily.png'))
    plt.close()
    
    # Plot conductance vs. homophily
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='homophily', y='conductance', hue='model', style='model', markers=True)
    plt.title('Conductance vs. Homophily')
    plt.xlabel('Homophily')
    plt.ylabel('Conductance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'conductance_vs_homophily.png'))
    plt.close()
    
    # Plot coverage vs. homophily
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='homophily', y='coverage', hue='model', style='model', markers=True)
    plt.title('Coverage vs. Homophily')
    plt.xlabel('Homophily')
    plt.ylabel('Coverage')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'coverage_vs_homophily.png'))
    plt.close()
    
    # Plot performance metrics together
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'modularity', 'conductance', 'coverage']
    for metric in metrics:
        sns.lineplot(data=results_df, x='homophily', y=metric, hue='model', style='model', 
                    markers=True, label=metric.capitalize())
    plt.title('Performance Metrics vs. Homophily')
    plt.xlabel('Homophily')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.legend(title='Metric')
    plt.savefig(os.path.join(output_dir, 'all_metrics_vs_homophily.png'))
    plt.close()


def main(args=None):
    """Main function to run the experiment."""
    # Parse command line arguments if not provided
    if args is None:
        args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate graph
    graph_sample = generate_graph(args)
    
    # Prepare data for models
    features, edge_index, labels, train_idx, val_idx, test_idx, num_classes = prepare_data(graph_sample, args)
    
    # Train and evaluate models
    model_results = run_models(features, edge_index, labels, train_idx, val_idx, test_idx, num_classes, args)
    
    # Visualize results
    visualize_results(graph_sample, model_results, args)
    
    print(f"\nExperiment complete. Results saved to {args.output_dir}")
    
    return graph_sample, model_results


if __name__ == "__main__":
    main()