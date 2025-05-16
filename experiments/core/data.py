"""
Data preparation utilities for MMSB graph learning experiments.

This module provides functions for preparing data for model training, including
graph format conversions and feature generation.
"""

import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, to_undirected
from typing import Dict, List, Optional, Tuple, Union, Any

from mmsb.model import GraphSample
from experiments.core.config import ExperimentConfig
from mmsb.feature_regimes import FeatureRegimeGenerator, NeighborhoodFeatureAnalyzer, GenerativeRuleBasedLabeler
from utils.motif_and_role_analysis import MotifRoleAnalyzer


def prepare_data(
    graph_sample: GraphSample,
    config: ExperimentConfig,
    feature_type: str = "membership"
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare data for model training from a GraphSample.
    
    Args:
        graph_sample: The graph sample to prepare data from
        config: Experiment configuration
        feature_type: Type of features to use ("membership" or "random")
        
    Returns:
        Dictionary containing data for each task
    """
    # Debug information
    print("\nDebugging graph structure:")
    print(f"Number of nodes: {len(graph_sample.graph.nodes())}")
    print(f"Number of edges: {len(graph_sample.graph.edges())}")
    print(f"Graph is connected: {nx.is_connected(graph_sample.graph)}")
    print(f"Graph has self loops: {len(list(nx.nodes_with_selfloops(graph_sample.graph)))}")
    print(f"Graph is directed: {graph_sample.graph.is_directed()}")
    print(f"Graph density: {nx.density(graph_sample.graph):.4f}")
    print(f"Average degree: {sum(dict(graph_sample.graph.degree()).values()) / len(graph_sample.graph.nodes()):.2f}")
    
    # Convert graph to PyTorch Geometric format
    try:
        edges = list(graph_sample.graph.edges())
        print(f"First few edges: {edges[:5]}")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    except Exception as e:
        print(f"Error converting edges to tensor: {e}")
        print(f"Edge type: {type(graph_sample.graph.edges())}")
        print(f"Edge content: {list(graph_sample.graph.edges())[:5]}")
        raise
    
    # Get features based on type
    if feature_type == "membership":
        features = torch.tensor(graph_sample.community_labels, dtype=torch.long).unsqueeze(1)
    else:  # random features
        features = torch.randn((len(graph_sample.graph.nodes()), config.feature_dim))
    
    # Split data
    n_nodes = len(graph_sample.graph.nodes())
    indices = torch.randperm(n_nodes)
    
    train_size = int(n_nodes * config.train_ratio)
    val_size = int(n_nodes * config.val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Initialize results dictionary
    task_data = {}
    
    # Prepare data for each task
    for task in config.tasks:
        if task == "community":
            # Community prediction task (original task)
            labels = torch.tensor(graph_sample.community_labels, dtype=torch.long)
            num_classes = len(torch.unique(labels))
            
            task_data["community"] = {
                "features": features,
                "edge_index": edge_index,
                "labels": labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "num_classes": num_classes
            }
            
        elif task == "regime":
            # Feature regime prediction task using rule-based generation
            print("\nPreparing regime task data...")
            
            # Compute neighborhood features if not already done
            if graph_sample.neighborhood_analyzer is None:
                print("Computing neighborhood features...")
                graph_sample.compute_neighborhood_features(max_hops=config.regime_task_max_hop)
            
            # Get frequency vectors for all hop distances in the specified range
            freq_vectors_by_hop = {}
            for k in range(config.regime_task_min_hop, config.regime_task_max_hop + 1):
                freq_vectors_by_hop[k] = graph_sample.neighborhood_analyzer.get_all_frequency_vectors(k)
            
            # Generate rule-based labels
            rule_generator = GenerativeRuleBasedLabeler(
                n_labels=config.regime_task_n_labels,
                min_support=config.regime_task_min_support,
                max_rules_per_label=config.regime_task_max_rules_per_label,
                min_hop=config.regime_task_min_hop,
                max_hop=config.regime_task_max_hop,
                seed=config.seed
            )
            
            print("Generating and applying rules...")
            rules = rule_generator.generate_rules(freq_vectors_by_hop)
            regime_labels, applied_rules = rule_generator.apply_rules(freq_vectors_by_hop)
            
            # Convert to tensor
            regime_labels = torch.tensor(regime_labels, dtype=torch.long)
            
            task_data["regime"] = {
                "features": features,
                "edge_index": edge_index,
                "labels": regime_labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "num_classes": config.regime_task_n_labels,
                "rules": rules,
                "applied_rules": applied_rules,
                "freq_vectors_by_hop": freq_vectors_by_hop
            }
            print("Regime task data preparation complete.")
            
        elif task == "role":
            # Role prediction task using motif analysis
            print("\nPreparing role task data...")
            motif_analyzer = MotifRoleAnalyzer(
                graph=graph_sample.graph,
                max_motif_size=config.role_task_max_motif_size,
                n_roles=config.role_task_n_roles
            )
            
            role_labels = motif_analyzer.get_node_roles()
            role_labels = torch.tensor(role_labels, dtype=torch.long)
            
            task_data["role"] = {
                "features": features,
                "edge_index": edge_index,
                "labels": role_labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "num_classes": config.role_task_n_roles,
                "motif_counts": motif_analyzer.motif_counts,
                "role_assignments": motif_analyzer.role_assignments
            }
            print("Role task data preparation complete.")
    
    return task_data


def networkx_to_pyg(
    graph: nx.Graph,
    label_key: str = "label"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a NetworkX graph to PyTorch tensors for model training.
    
    Args:
        graph: NetworkX graph
        label_key: Node attribute key for labels
        
    Returns:
        Tuple of (features, edge_index, labels)
    """
    # Convert to PyG format using from_networkx
    # Features are automatically extracted from the "features" node attribute
    data = from_networkx(graph, group_node_attrs=["features"])
    
    # Get edge index and ensure undirected
    edge_index = to_undirected(data.edge_index)
    
    # Get features from the converted data
    features = data.x if data.x is not None else torch.eye(graph.number_of_nodes(), dtype=torch.float)
    
    # Get labels
    labels = torch.tensor([data[1].get(label_key, 0) for data in sorted(graph.nodes(data=True))], dtype=torch.long)
    
    return features, edge_index, labels


def create_sklearn_compatible_data(
    features: Union[np.ndarray, torch.Tensor],
    edge_index: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    include_graph_features: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create scikit-learn compatible features by optionally adding graph structure information.
    
    Args:
        features: Node features [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        labels: Node labels [num_nodes]
        include_graph_features: Whether to include graph structure features
        
    Returns:
        Tuple of (X, y) for scikit-learn models
    """
    # Convert tensors to numpy, ensuring they're on CPU first
    if isinstance(features, torch.Tensor):
        X = features.cpu().numpy()
    else:
        X = features
        
    if isinstance(labels, torch.Tensor):
        y = labels.cpu().numpy()
    else:
        y = labels
    
    if include_graph_features:
        # Create adjacency-based features
        num_nodes = X.shape[0]
        
        if isinstance(edge_index, torch.Tensor):
            edge_list = edge_index.t().cpu().numpy()
        else:
            edge_list = np.array(edge_index).T
            
        # Create mapping to ensure edge indices are within bounds
        unique_nodes = np.unique(edge_list.flatten())
        node_to_idx = {node: idx for idx, node in enumerate(range(num_nodes))}
        
        # Create adjacency matrix
        adjacency = np.zeros((num_nodes, num_nodes))
        for edge in edge_list:
            u, v = edge[0], edge[1]
            if u in node_to_idx and v in node_to_idx:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                adjacency[u_idx, v_idx] = 1
        
        # Feature 1: Node degree
        degree = np.sum(adjacency, axis=1, keepdims=True)
        
        # Feature 2: Clustering coefficient (approximated by local triangle count)
        triangle_count = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) >= 2:
                # Count triangles (connections between neighbors)
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        if adjacency[neighbors[j], neighbors[k]] > 0:
                            triangle_count[i, 0] += 1
        
        # Feature 3: Average neighbor degree
        avg_neighbor_degree = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) > 0:
                avg_neighbor_degree[i, 0] = np.mean(degree[neighbors])
        
        # Combine all features
        graph_features = np.hstack([
            degree,
            triangle_count,
            avg_neighbor_degree
        ])
        
        # Normalize graph features
        graph_features = (graph_features - np.mean(graph_features, axis=0)) / (np.std(graph_features, axis=0) + 1e-8)
        
        # Combine with original features
        X = np.hstack([X, graph_features])
    
    return X, y


def generate_graph_statistics(graph_sample: GraphSample) -> Dict[str, float]:
    """
    Generate statistics for a graph sample.
    
    Args:
        graph_sample: MMSB GraphSample
        
    Returns:
        Dictionary of graph statistics
    """
    graph = graph_sample.graph
    
    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "density": nx.density(graph),
        "clustering_coefficient": nx.average_clustering(graph),
        "num_components": nx.number_connected_components(graph),
        "largest_component_size": len(max(nx.connected_components(graph), key=len)),
        "avg_shortest_path": nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf'),
        "diameter": nx.diameter(graph) if nx.is_connected(graph) else float('inf'),
        "homophily": graph_sample.homophily if hasattr(graph_sample, 'homophily') else None,
        "feature_signal": graph_sample.feature_signal if hasattr(graph_sample, 'feature_signal') else None
    }
    
    return stats 