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


def prepare_data(
    graph_sample: GraphSample,
    config: Any,
    feature_type: str = "generated"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepare data for model training from a GraphSample.
    
    Args:
        graph_sample: MMSB GraphSample
        config: Configuration object with train/val/test ratios
        feature_type: Type of features to use
        
    Returns:
        Tuple of (features, edge_index, labels, train_idx, val_idx, test_idx, num_classes)
    """
    # Convert graph to PyG format using networkx_to_pyg
    if feature_type == "generated":
        feature_key = "features"
    elif feature_type == "membership":
        feature_key = "memberships"
    elif feature_type == "onehot":
        feature_key = None  # Will use one-hot node IDs
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    features, edge_index, labels = networkx_to_pyg(
        graph_sample.graph,
        label_key="primary_community"
    )
    
    # Print initial label information
    print("\nInitial Label Information:")
    print(f"Original labels: {labels.tolist()}")
    print(f"Unique labels: {torch.unique(labels).tolist()}")
    print(f"Label range: [{labels.min().item()}, {labels.max().item()}]")
    
    # Get unique labels and remap to 0-based indices
    unique_labels = torch.unique(labels)
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    
    # Print label mapping
    print("\nLabel Mapping:")
    for old, new in label_map.items():
        print(f"  {old} -> {new}")
    
    # Remap labels using the mapping
    remapped_labels = torch.tensor([label_map[label.item()] for label in labels])
    
    # Get number of classes after remapping
    num_classes = len(unique_labels)
    
    # Split indices
    num_nodes = len(remapped_labels)
    indices = torch.randperm(num_nodes)
    
    train_size = int(config.train_ratio * num_nodes)
    val_size = int(config.val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {remapped_labels.shape}")
    print(f"Number of unique labels: {num_classes}")
    print(f"Label range: [{remapped_labels.min().item()}, {remapped_labels.max().item()}]")
    print(f"Train indices range: [{train_idx.min().item()}, {train_idx.max().item()}] (length: {len(train_idx)})")
    print(f"Val indices range: [{val_idx.min().item()}, {val_idx.max().item()}] (length: {len(val_idx)})")
    print(f"Test indices range: [{test_idx.min().item()}, {test_idx.max().item()}] (length: {len(test_idx)})")
    
    print("\nLabel distribution (after remapping):")
    for i in range(num_classes):
        train_count = (remapped_labels[train_idx] == i).sum().item()
        val_count = (remapped_labels[val_idx] == i).sum().item()
        test_count = (remapped_labels[test_idx] == i).sum().item()
        print(f"Class {i}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    return features, edge_index, remapped_labels, train_idx, val_idx, test_idx, num_classes


def networkx_to_pyg(
    graph: nx.Graph,
    label_key: str = "primary_community"
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
    labels = torch.tensor([data[1][label_key] for data in sorted(graph.nodes(data=True))], dtype=torch.long)
    
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