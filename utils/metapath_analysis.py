"""
Metapath analysis for stochastic block model graphs.

This module provides functions to extract likely metapaths from community-level
edge probability matrices and verify their presence in actual graph instances.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from mmsb.model import GraphSample
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import optuna
from sklearn.model_selection import cross_val_score
from torch.nn import BatchNorm1d, LayerNorm
import copy
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def extract_metapaths(
    P_matrix: np.ndarray,
    theta: float = 0.1,
    max_length: int = 3,
    min_length: int = 2,
    allow_loops: bool = False,
    allow_backtracking: bool = False,
) -> List[List[int]]:
    """
    Extract statistically likely meta-paths from a community-level edge probability matrix.
    
    Args:
        P_matrix: Community-to-community edge probability matrix
        theta: Threshold for considering an edge likely
        max_length: Maximum length of meta-paths to extract
        min_length: Minimum length of meta-paths to extract
        allow_loops: Whether to allow repeated communities in paths
        allow_backtracking: Whether to allow returning to previous communities
        
    Returns:
        List of meta-paths, where each meta-path is a list of community indices
    """
    # Create thresholded adjacency matrix for the community graph
    n_communities = P_matrix.shape[0]
    A_thresh = (P_matrix > theta).astype(int)
    
    # Create community-level graph
    G_comm = nx.from_numpy_array(A_thresh, create_using=nx.DiGraph())
    
    # Store all extracted metapaths
    all_metapaths = []
    
    # Helper function for DFS path extraction
    def dfs_paths(current_path, visited):
        if len(current_path) >= min_length:  # Only store paths that meet minimum length
            all_metapaths.append(current_path.copy())
            
        if len(current_path) >= max_length:
            return
            
        current = current_path[-1]
        for neighbor in G_comm.neighbors(current):
            if not allow_loops and neighbor in current_path:
                continue
                
            if not allow_backtracking and len(current_path) >= 2 and neighbor == current_path[-2]:
                continue
                
            if allow_loops or neighbor not in visited:
                current_path.append(neighbor)
                new_visited = visited.copy()
                new_visited.add(neighbor)
                dfs_paths(current_path, new_visited)
                current_path.pop()
    
    # Extract paths starting from each community
    for start in range(n_communities):
        dfs_paths([start], {start})
    
    # Calculate path probabilities
    metapath_probs = []
    for path in all_metapaths:
        # Calculate the product of probabilities along the path
        prob = 1.0
        for i in range(len(path) - 1):
            prob *= P_matrix[path[i], path[i+1]]
        metapath_probs.append((path, prob))
    
    # Sort by probability (highest first)
    metapath_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Return paths sorted by probability
    return [path for path, _ in metapath_probs]


def find_metapath_instances(
    graph: nx.Graph,
    community_labels: np.ndarray,
    metapath: List[int],
    max_instances: int = 1000,
    sample_seed: Optional[int] = None
) -> List[List[int]]:
    """
    Find instances of a meta-path in an actual graph.
    
    Args:
        graph: NetworkX graph
        community_labels: Community assignment for each node
        metapath: Sequence of community indices defining the metapath
        max_instances: Maximum number of instances to find
        sample_seed: Random seed for sampling starting nodes
        
    Returns:
        List of node sequences that follow the meta-path
    """
    if len(metapath) < 2:
        return []
        
    # Set random seed if provided
    if sample_seed is not None:
        np.random.seed(sample_seed)
    
    # Find all nodes in the first community of the metapath
    start_nodes = [node for node in graph.nodes() if community_labels[node] == metapath[0]]
    
    # If there are too many start nodes, sample a subset
    if len(start_nodes) > 100:
        start_nodes = np.random.choice(start_nodes, size=100, replace=False).tolist()
    
    # Store valid instances
    instances = []
    
    # For each starting node, try to find a path that follows the metapath
    for start_node in start_nodes:
        if len(instances) >= max_instances:
            break
            
        # BFS to find paths following the metapath
        queue = [(start_node, [start_node], 0)]
        while queue and len(instances) < max_instances:
            current, path, index = queue.pop(0)
            
            # If we've reached the end of the metapath, add this path
            if index == len(metapath) - 1:
                instances.append(path)
                continue
            
            # Try to extend the path
            next_community = metapath[index + 1]
            for neighbor in graph.neighbors(current):
                if neighbor not in path and community_labels[neighbor] == next_community:
                    queue.append((neighbor, path + [neighbor], index + 1))
    
    return instances


def calculate_metapath_statistics(
    graph: nx.Graph,
    community_labels: np.ndarray,
    metapaths: List[List[int]],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Calculate statistics about metapath occurrences in a graph.
    
    Args:
        graph: NetworkX graph
        community_labels: Community assignment for each node
        metapaths: List of metapaths to analyze
        top_k: Number of top metapaths to analyze in detail
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'metapaths': [],
        'instances_count': [],
        'avg_path_length': [],
        'participation': []
    }
    
    # Limit to top_k metapaths
    metapaths = metapaths[:top_k]
    
    # Find instances for each metapath
    for metapath in metapaths:
        instances = find_metapath_instances(graph, community_labels, metapath)
        
        # Store path as string for display
        stats['metapaths'].append(' → '.join([str(c) for c in metapath]))
        stats['instances_count'].append(len(instances))
        
        # Calculate average path length if instances exist
        if instances:
            path_lengths = [len(path) - 1 for path in instances]  # Number of edges
            stats['avg_path_length'].append(np.mean(path_lengths))
        else:
            stats['avg_path_length'].append(0)
        
        # Calculate node participation
        participating_nodes = set()
        for instance in instances:
            participating_nodes.update(instance)
        
        stats['participation'].append(len(participating_nodes) / graph.number_of_nodes())
    
    return stats


def visualize_metapaths(
    graph: nx.Graph,
    community_labels: np.ndarray,
    metapath: List[int],
    instances: List[List[int]],
    title: str = "Metapath Instances",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize instances of a metapath in a graph.
    
    Args:
        graph: NetworkX graph
        community_labels: Community assignment for each node
        metapath: The metapath being visualized
        instances: Instances of the metapath in the graph
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get a layout for the graph
    pos = nx.spring_layout(graph, seed=42)
    
    # Get edges involved in metapath instances
    metapath_edges = set()
    for instance in instances:
        metapath_edges.update(zip(instance[:-1], instance[1:]))
    
    # Draw all nodes by community
    communities = sorted(set(community_labels))
    cmap = plt.cm.tab20
    
    # Draw all nodes by community
    for i, comm in enumerate(communities):
        nodes = [node for node in graph.nodes() if community_labels[node] == comm]
        if nodes:  # Only draw if there are nodes in this community
            nx.draw_networkx_nodes(
                graph, 
                pos, 
                nodelist=nodes,
                node_color=[cmap(i % 20)],
                node_size=50,
                alpha=0.6,
                ax=ax
            )
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(
        graph,
        pos,
        width=1,
        alpha=0.2,
        edge_color='gray',
        ax=ax
    )
    
    # Highlight metapath instances with bold edges
    for i, instance in enumerate(instances[:10]):  # Limit to 10 instances
        # Draw edges in the path
        edges = list(zip(instance[:-1], instance[1:]))
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            width=3,
            alpha=1.0,
            edge_color=f'C{i}',
            ax=ax
        )
    
    # Add legend for communities
    handles = []
    labels = []
    for i, comm in enumerate(communities):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=cmap(i % 20), markersize=10))
        labels.append(f'Community {comm}')
    
    # Add legend for metapath
    handles.append(plt.Line2D([0], [0], color='k', lw=2))
    labels.append(f'Metapath: {" → ".join([str(c) for c in metapath])}')
    
    ax.legend(handles, labels, loc='best')
    
    # Add title
    ax.set_title(title)
    
    # Turn off axis
    ax.axis('off')
    
    return fig


def visualize_community_metapath_graph(
    P_matrix: np.ndarray,
    theta: float = 0.1,
    community_mapping: Optional[Dict[int, int]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize the community-level graph with edges representing likely metapaths.
    
    Args:
        P_matrix: Community-to-community edge probability matrix
        theta: Threshold for considering an edge likely
        community_mapping: Mapping from indices to community IDs
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create thresholded adjacency matrix
    A_thresh = (P_matrix > theta).astype(int)
    
    # Create community-level graph
    G_comm = nx.from_numpy_array(A_thresh, create_using=nx.DiGraph())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get layout for the graph
    pos = nx.spring_layout(G_comm, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_comm,
        pos,
        node_size=500,
        node_color='lightblue',
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G_comm,
        pos,
        width=2,
        arrowsize=20,
        ax=ax
    )
    
    # Add node labels
    if community_mapping:
        labels = {i: f"C{community_mapping[i]}" for i in G_comm.nodes()}
    else:
        labels = {i: f"C{i}" for i in G_comm.nodes()}
        
    nx.draw_networkx_labels(G_comm, pos, labels=labels, font_size=12, ax=ax)
    
    # Add edge weights as labels
    edge_labels = {}
    for u, v in G_comm.edges():
        edge_labels[(u, v)] = f"{P_matrix[u, v]:.2f}"
        
    nx.draw_networkx_edge_labels(G_comm, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # Add title
    ax.set_title(f"Community Metapath Graph (θ = {theta})")
    
    # Turn off axis
    ax.axis('off')
    
    return fig


def prepare_node_features(graph, community_labels=None, use_degree=True, use_clustering=True, use_node_features=True):
    """Prepare node features for baselines (RF/MLP) with flexible selection.
    Args:
        graph: NetworkX graph OR GraphSample object
        community_labels: (unused, kept for compatibility)
        use_degree: Whether to include normalized degree
        use_clustering: Whether to include normalized clustering coefficient
        use_node_features: Whether to include node features from graph.features
    Returns:
        features: np.ndarray of shape (n_nodes, n_features)
    """
    # If a GraphSample is passed, use its .graph for nx operations
    if hasattr(graph, 'graph') and isinstance(graph.graph, nx.Graph):
        nx_graph = graph.graph
    else:
        nx_graph = graph

    feature_list = []
    n_nodes = nx_graph.number_of_nodes()

    if use_degree:
        degrees = np.array([d for n, d in nx_graph.degree()])
        degrees_std = degrees.std()
        if degrees_std > 0:
            degrees = (degrees - degrees.mean()) / degrees_std
        else:
            degrees = degrees - degrees.mean()
        feature_list.append(degrees.reshape(-1, 1))

    if use_clustering:
        clustering = np.array(list(nx.clustering(nx_graph).values()))
        clustering_std = clustering.std()
        if clustering_std > 0:
            clustering = (clustering - clustering.mean()) / clustering_std
        else:
            clustering = clustering - clustering.mean()
        feature_list.append(clustering.reshape(-1, 1))

    if use_node_features:
        # Try to get node features from graph object
        if hasattr(graph, 'features') and graph.features is not None:
            node_features = graph.features
        elif hasattr(graph, 'graph') and hasattr(graph.graph, 'features') and graph.graph.features is not None:
            node_features = graph.graph.features
        else:
            raise ValueError("Node features not found in graph object. Set use_node_features=False or provide features.")
        feature_list.append(node_features)

    if not feature_list:
        raise ValueError("No features selected: all use_degree, use_clustering, use_node_features are False.")

    features = np.column_stack(feature_list) if len(feature_list) > 1 else feature_list[0]
    return features


def prepare_graph_data(graph, features, labels=None):
    """
    Convert NetworkX graph to PyTorch Geometric format with improved mapping.
    Args:
        graph: NetworkX graph
        features: Node feature matrix
        labels: Optional node labels
    Returns:
        PyTorch Geometric Data object
    """
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edge_list = list(graph.edges())
    if edge_list:
        edge_index = [[node_mapping[u], node_mapping[v]] for u, v in edge_list]
        edge_index += [[node_mapping[v], node_mapping[u]] for u, v in edge_list]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.long if labels.ndim == 1 else torch.float)
        data.y = y
    return data


def train_gnn_model(model, data, optimizer=None, criterion=None, epochs=200, lr=0.01, early_stopping=True, patience=20, verbose=True):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Detect multi-label vs single-label
    is_multilabel = (data.y.ndim == 2 and data.y.dtype in [torch.float, torch.double])
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss() if is_multilabel else torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = float('-inf')
    best_model_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            if is_multilabel:
                # Multi-label: use sigmoid > 0.5 for prediction
                train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float()
                train_true = data.y[data.train_mask]
                train_f1 = f1_score(train_true.cpu().numpy(), train_pred.cpu().numpy(), average='macro')
                val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float()
                val_true = data.y[data.val_mask]
                val_f1 = f1_score(val_true.cpu().numpy(), val_pred.cpu().numpy(), average='macro')
                train_acc = (train_pred == train_true).float().mean().item()
                val_acc = (val_pred == val_true).float().mean().item()
            else:
                train_pred = out[data.train_mask].argmax(dim=1)
                train_true = data.y[data.train_mask]
                train_f1 = f1_score(train_true.cpu().numpy(), train_pred.cpu().numpy(), average='macro')
                train_acc = train_pred.eq(train_true).float().mean().item()
                val_pred = out[data.val_mask].argmax(dim=1)
                val_true = data.y[data.val_mask]
                val_f1 = f1_score(val_true.cpu().numpy(), val_pred.cpu().numpy(), average='macro')
                val_acc = val_pred.eq(val_true).float().mean().item()
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
        if early_stopping:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history


def evaluate_model(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[mask].argmax(dim=1)
        labels = data.y[mask]
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        return {
            'accuracy': acc,
            'f1_score': f1,
            'predictions': preds.cpu().numpy(),
            'true_labels': labels.cpu().numpy()
        }


def metapath_node_classification(graph, community_labels, metapath, model_type='rf', baseline_feature_opts=None):
    """
    Perform node classification for metapath participation using various models.
    Args:
        graph: NetworkX graph
        community_labels: List of community labels for each node
        metapath: List of community IDs representing the metapath
        model_type: Type of model to use ('rf', 'mlp', 'gcn', 'sage')
        baseline_feature_opts: dict with keys 'use_degree', 'use_clustering', 'use_node_features' (for RF/MLP)
    Returns:
        Dictionary containing classification results
    """
    # Find metapath instances
    instances = find_metapath_instances(graph, community_labels, metapath)
    participating_nodes = set()
    for instance in instances:
        participating_nodes.update(instance)
    labels = np.array([1 if i in participating_nodes else 0 for i in range(graph.number_of_nodes())])

    # Feature selection logic
    if model_type in ['rf', 'mlp']:
        opts = baseline_feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
        features = prepare_node_features(
            graph,
            community_labels,
            use_degree=opts.get('use_degree', True),
            use_clustering=opts.get('use_clustering', True),
            use_node_features=opts.get('use_node_features', True)
        )
    elif model_type in ['gcn', 'sage']:
        # Always use only node features for GNNs
        features = prepare_node_features(
            graph,
            community_labels,
            use_degree=False,
            use_clustering=False,
            use_node_features=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        predictions = model.predict(features)
        feature_importance = model.feature_importances_
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        model.fit(features, labels)
        predictions = model.predict(features)
        feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
    elif model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph, features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if model_type == 'gcn':
            model = GCN(in_channels=features.shape[1], hidden_channels=64, out_channels=2)
        else:
            model = GraphSAGE(in_channels=features.shape[1], hidden_channels=64, out_channels=2)
        model = train_gnn_model(model, data, epochs=200)[0]
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1).numpy()
        if model_type == "gcn":
            feature_importance = np.abs(model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        elif model_type == "sage":
            feature_importance = np.abs(model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            feature_importance = np.zeros(features.shape[1])
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    participation_rate = labels.mean()
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'participation_rate': participation_rate,
        'feature_importance': feature_importance,
        'predictions': predictions,
        'true_labels': labels
    }


def run_all_classifications(graph, community_labels, metapath, baseline_feature_opts=None):
    """Run all classification models and return combined results."""
    results = {}
    for model_type in ['rf', 'mlp', 'gcn', 'sage']:
        if model_type in ['rf', 'mlp']:
            results[model_type] = metapath_node_classification(
                graph, community_labels, metapath, model_type=model_type, baseline_feature_opts=baseline_feature_opts
            )
        else:
            results[model_type] = metapath_node_classification(
                graph, community_labels, metapath, model_type=model_type
            )
    return results


def visualize_P_sub_matrix(
    P_sub: np.ndarray,
    title: str = "Community Edge Probability Matrix (P_sub)",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Visualize the P_sub matrix as a heatmap.
    
    Args:
        P_sub: Community-to-community edge probability matrix
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(P_sub, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Probability')
    
    # Add labels
    n_communities = P_sub.shape[0]
    ax.set_xticks(np.arange(n_communities))
    ax.set_yticks(np.arange(n_communities))
    ax.set_xticklabels([f'C{i}' for i in range(n_communities)])
    ax.set_yticklabels([f'C{i}' for i in range(n_communities)])
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax.set_title(title)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, n_communities, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_communities, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    return fig


def analyze_metapaths(
    graph_sample: GraphSample,
    theta: float = 0.1,
    max_length: int = 3,
    min_length: int = 2,
    top_k: int = 5,
    allow_loops: bool = False,
    allow_backtracking: bool = False
) -> Dict[str, Any]:
    """
    Analyze metapaths in a graph sample.
    
    Args:
        graph_sample: GraphSample object
        theta: Threshold for considering an edge likely
        max_length: Maximum length of metapaths to extract
        min_length: Minimum length of metapaths to extract
        top_k: Number of top metapaths to analyze in detail
        allow_loops: Whether to allow repeated communities in paths
        allow_backtracking: Whether to allow returning to previous communities
        
    Returns:
        Dictionary containing analysis results
    """
    # Extract metapaths
    metapaths = extract_metapaths(
        graph_sample.P_sub,
        theta=theta,
        max_length=max_length,
        min_length=min_length,
        allow_loops=allow_loops,
        allow_backtracking=allow_backtracking
    )
    
    # Calculate statistics
    stats = calculate_metapath_statistics(
        graph_sample.graph,
        graph_sample.community_labels,
        metapaths,
        top_k=top_k
    )
    
    # Find instances for each metapath
    instances = []
    for metapath in metapaths[:top_k]:
        path_instances = find_metapath_instances(
            graph_sample.graph,
            graph_sample.community_labels,
            metapath
        )
        instances.append(path_instances)
    
    # Create P_sub visualization
    P_sub_fig = visualize_P_sub_matrix(graph_sample.P_sub)
    
    return {
        'P_matrix': graph_sample.P_sub,
        'P_sub_figure': P_sub_fig,
        'community_mapping': {i: i for i in range(graph_sample.P_sub.shape[0])},
        'metapaths': metapaths[:top_k],
        'instances': instances,
        'statistics': stats
    }


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, normalization='none', skip_connection=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.skip_connection:
            x = x + identity[:, :x.shape[1]] if identity.shape[1] >= x.shape[1] else x
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        if self.skip_connection:
            x = x + identity2[:, :x.shape[1]] if identity2.shape[1] >= x.shape[1] else x
        return F.log_softmax(x, dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr='mean', normalization='none', skip_connection=False):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.skip_connection:
            x = x + identity[:, :x.shape[1]] if identity.shape[1] >= x.shape[1] else x
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.skip_connection:
            x = x + identity2[:, :x.shape[1]] if identity2.shape[1] >= x.shape[1] else x
        x = self.lin(x)
        return x  # No activation - handled by loss function


def optimize_hyperparameters(
    graph,
    community_labels,
    metapath,
    model_type='rf',
    n_trials=50,
    timeout=600
):
    """
    Optimize hyperparameters for a model using Optuna.
    """
    instances = find_metapath_instances(graph, community_labels, metapath)
    participating_nodes = set()
    for instance in instances:
        participating_nodes.update(instance)
    labels = np.array([1 if i in participating_nodes else 0 for i in range(graph.number_of_nodes())])
    features = prepare_node_features(graph, community_labels)

    if model_type == 'rf':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            cv_scores = cross_val_score(model, features, labels, cv=5, scoring='f1')
            return np.mean(cv_scores)
    elif model_type == 'mlp':
        def objective(trial):
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 16, 256))
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
            model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=learning_rate,
                alpha=alpha,
                max_iter=1000,
                random_state=42
            )
            cv_scores = cross_val_score(model, features, labels, cv=5, scoring='f1')
            return np.mean(cv_scores)
    elif model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph, features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        def objective(trial):
            hidden_channels = trial.suggest_int('hidden_channels', 16, 256)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.8)
            normalization = trial.suggest_categorical('normalization', ['none', 'batch', 'layer'])
            skip_connection = trial.suggest_categorical('skip_connection', [False, True])
            n_layers = trial.suggest_int('n_layers', 2, 5)
            if model_type == 'gcn':
                model = build_gcn_dynamic(
                    in_channels=features.shape[1],
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    n_layers=n_layers,
                    dropout=dropout,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            else:
                aggr = trial.suggest_categorical('aggr', ['mean', 'max', 'min', 'sum'])
                model = build_sage_dynamic(
                    in_channels=features.shape[1],
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    n_layers=n_layers,
                    dropout=dropout,
                    aggr=aggr,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out, labels_tensor)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                f1 = f1_score(labels_tensor.numpy(), pred.numpy())
            return f1
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params
    if model_type == 'rf':
        best_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
        best_model.fit(features, labels)
    elif model_type == 'mlp':
        hidden_layer_sizes = []
        for i in range(best_params['n_layers']):
            hidden_layer_sizes.append(best_params[f'n_units_l{i}'])
        best_model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=best_params['learning_rate'],
            alpha=best_params['alpha'],
            max_iter=1000,
            random_state=42
        )
        best_model.fit(features, labels)
    elif model_type in ['gcn', 'sage']:
        if model_type == 'gcn':
            best_model = build_gcn_dynamic(
                in_channels=features.shape[1],
                hidden_channels=best_params['hidden_channels'],
                out_channels=2,
                n_layers=best_params.get('n_layers', 2),
                dropout=best_params['dropout'],
                normalization=best_params.get('normalization', 'none'),
                skip_connection=best_params.get('skip_connection', False)
            )
        else:
            best_model = build_sage_dynamic(
                in_channels=features.shape[1],
                hidden_channels=best_params['hidden_channels'],
                out_channels=2,
                n_layers=best_params.get('n_layers', 2),
                dropout=best_params['dropout'],
                aggr=best_params.get('aggr', 'mean'),
                normalization=best_params.get('normalization', 'none'),
                skip_connection=best_params.get('skip_connection', False)
            )
        optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        best_model = train_gnn_model(best_model, data, optimizer=optimizer, epochs=500)[0]
    return best_params, best_model


def create_train_val_test_split(num_nodes, labels, train_size=0.7, val_size=0.15, test_size=0.15, stratify=True, seed=42):
    """
    Create stratified train/validation/test splits.
    Args:
        num_nodes: Number of nodes in the graph
        labels: Node labels for stratification
        train_size: Fraction of nodes in training set
        val_size: Fraction of nodes in validation set
        test_size: Fraction of nodes in test set
        stratify: Whether to use stratified sampling
        seed: Random seed
    Returns:
        Dictionary with train, val, test masks as torch tensors
    """
    assert train_size + val_size + test_size <= 1.0, "Split fractions must sum to at most 1"
    indices = np.arange(num_nodes)
    stratify_data = labels if stratify else None
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_size + test_size) / (train_size + val_size + test_size),
        random_state=seed,
        stratify=stratify_data
    )
    if stratify and len(temp_indices) > 0:
        stratify_temp = labels[temp_indices]
    else:
        stratify_temp = None
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=stratify_temp
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }


def visualize_model_performance(results, model_type, multi_label=False, figsize=(12, 5)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    metrics = ['train_f1', 'val_f1', 'test_f1']
    values = [results[m] for m in metrics]
    axs[0].bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0].set_title(f'{model_type.upper()} Model F1 Scores')
    axs[0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axs[0].text(i, v + 0.05, f'{v:.3f}', ha='center')
    if 'feature_importance' in results and results['feature_importance'] is not None:
        importance = results['feature_importance']
        if len(importance) > 10:
            indices = np.argsort(importance)[-10:]
            importance = importance[indices]
            feature_names = [f"Feature {i}" for i in indices]
        else:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        axs[1].barh(feature_names, importance, color='#9b59b6')
        axs[1].set_title('Feature Importance')
        axs[1].set_xlabel('Importance')
    elif model_type in ['gcn', 'sage'] and 'history' in results:
        history = results['history']
        epochs = range(1, len(history['train_loss']) + 1)
        axs[1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axs[1].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axs[1].set_title('Training History')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
    plt.tight_layout()
    return fig


def multi_metapath_node_classification(
    graph,
    community_labels,
    metapaths,
    model_type='rf',
    feature_opts=None,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    feature_opts = feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
    X = prepare_node_features(
        graph,
        community_labels,
        use_degree=feature_opts.get('use_degree', True),
        use_clustering=feature_opts.get('use_clustering', True),
        use_node_features=feature_opts.get('use_node_features', True)
    )
    n_nodes = X.shape[0]
    Y = np.zeros((n_nodes, len(metapaths)), dtype=int)
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    for j, metapath in enumerate(metapaths):
        instances = find_metapath_instances(graph_nx, community_labels, metapath)
        participating_nodes = set()
        for instance in instances:
            participating_nodes.update(instance)
        for i in range(n_nodes):
            if i in participating_nodes:
                Y[i, j] = 1
    label_counts = Y.sum(axis=0)
    participation_rates = label_counts / n_nodes
    indices = np.arange(n_nodes)
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_size + test_size), random_state=seed)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_size/(val_size + test_size), random_state=seed)
    splits = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    if model_type == 'rf':
        from sklearn.multioutput import MultiOutputClassifier
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=seed))
        model.fit(X[train_indices], Y[train_indices])
        train_pred = model.predict(X[train_indices])
        val_pred = model.predict(X[val_indices])
        test_pred = model.predict(X[test_indices])
        train_acc = accuracy_score(Y[train_indices].flatten(), train_pred.flatten())
        val_acc = accuracy_score(Y[val_indices].flatten(), val_pred.flatten())
        test_acc = accuracy_score(Y[test_indices].flatten(), test_pred.flatten())
        train_f1 = f1_score(Y[train_indices], train_pred, average='macro')
        val_f1 = f1_score(Y[val_indices], val_pred, average='macro')
        test_f1 = f1_score(Y[test_indices], test_pred, average='macro')
        feature_importance = model.estimators_[0].feature_importances_
        history = None
    elif model_type == 'mlp':
        from sklearn.multioutput import MultiOutputClassifier
        model = MultiOutputClassifier(MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=seed
        ))
        model.fit(X[train_indices], Y[train_indices])
        train_pred = model.predict(X[train_indices])
        val_pred = model.predict(X[val_indices])
        test_pred = model.predict(X[test_indices])
        train_acc = accuracy_score(Y[train_indices].flatten(), train_pred.flatten())
        val_acc = accuracy_score(Y[val_indices].flatten(), val_pred.flatten())
        test_acc = accuracy_score(Y[test_indices].flatten(), test_pred.flatten())
        train_f1 = f1_score(Y[train_indices], train_pred, average='macro')
        val_f1 = f1_score(Y[val_indices], val_pred, average='macro')
        test_f1 = f1_score(Y[test_indices], test_pred, average='macro')
        feature_importance = np.abs(model.estimators_[0].coefs_[0]).mean(axis=1)
        history = None
    elif model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph_nx, X)
        
        # Add splits to data
        data.train_mask = splits['train_mask']
        data.val_mask = splits['val_mask']
        data.test_mask = splits['test_mask']
        data.y = torch.tensor(Y, dtype=torch.float)
        
        # Initialize appropriate model
        if model_type == 'gcn':
            model = GCNMultiLabel(
                in_channels=X.shape[1],
                hidden_channels=64,
                out_channels=Y.shape[1],
                dropout=0.5
            )
        else:
            model = GraphSAGEMultiLabel(
                in_channels=X.shape[1],
                hidden_channels=64,
                out_channels=Y.shape[1],
                dropout=0.5
            )
        
        # Train model with improved training function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Use weighted BCE loss for handling class imbalance
        pos_weight = torch.tensor([
            len(Y[train_indices]) / (2 * Y[train_indices, j].sum())
            for j in range(Y.shape[1])
        ])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model, history = train_gnn_model_improved(
            model,
            data,
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,
            early_stopping=True,
            patience=50,
            min_epochs=100,
            verbose=True
        )
        
        # Evaluate with improved evaluation function
        train_results = evaluate_model_improved(model, data, data.train_mask, multilabel=True)
        val_results = evaluate_model_improved(model, data, data.val_mask, multilabel=True)
        test_results = evaluate_model_improved(model, data, data.test_mask, multilabel=True)
        
        train_acc = train_results['accuracy']
        val_acc = val_results['accuracy']
        test_acc = test_results['accuracy']
        
        train_f1 = train_results['f1_score']
        val_f1 = val_results['f1_score']
        test_f1 = test_results['f1_score']
        
        train_pred = train_results['predictions']
        val_pred = val_results['predictions']
        test_pred = test_results['predictions']
        
        # Extract feature importance
        if model_type == 'gcn':
            feature_importance = np.abs(model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            feature_importance = np.abs(model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
    
    return {
        'model': model,
        'accuracy': {
            'train': train_acc,
            'val': val_acc,
            'test': test_acc
        },
        'f1_score': {
            'train': train_f1,
            'val': val_f1,
            'test': test_f1
        },
        'participation_rates': participation_rates,
        'feature_importance': feature_importance,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'true_labels': {
            'train': Y[train_indices],
            'val': Y[val_indices],
            'test': Y[test_indices]
        },
        'metapaths': metapaths,
        'history': history,
        'splits': splits
    }


def feature_regime_metapath_classification(
    graph,
    community_labels,
    metapath,
    node_regimes,
    target_regime,
    hop_distance=1,
    model_type='rf'
):
    """
    Classify nodes based on whether they connect (via a metapath) to nodes with specific feature regimes.
    """
    instances = find_metapath_instances(graph, community_labels, metapath)
    positive_nodes = set()
    for instance in instances:
        if hop_distance < len(instance):
            target_node = instance[hop_distance]
            if node_regimes[target_node] == target_regime:
                source_node = instance[0]
                positive_nodes.add(source_node)
    labels = np.array([1 if i in positive_nodes else 0 for i in range(graph.number_of_nodes())])
    features = prepare_node_features(graph, community_labels)
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        predictions = model.predict(features)
        feature_importance = model.feature_importances_
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        model.fit(features, labels)
        predictions = model.predict(features)
        feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
    elif model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph, features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if model_type == 'gcn':
            model = GCN(in_channels=features.shape[1], hidden_channels=64, out_channels=2)
        else:
            model = GraphSAGE(in_channels=features.shape[1], hidden_channels=64, out_channels=2)
        model = train_gnn_model(model, data, epochs=200)[0]
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1).numpy()
        if model_type == "gcn":
            feature_importance = np.abs(model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        elif model_type == "sage":
            feature_importance = np.abs(model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            feature_importance = np.zeros(features.shape[1])
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    participation_rate = labels.mean()
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'participation_rate': participation_rate,
        'feature_importance': feature_importance,
        'predictions': predictions,
        'true_labels': labels,
        'metapath': metapath,
        'target_regime': target_regime,
        'hop_distance': hop_distance
    }


def optimize_hyperparameters_for_metapath(
    graph,
    community_labels,
    metapaths,
    model_type='rf',
    multi_label=False,
    feature_opts=None,
    n_trials=20,
    timeout=300,
    seed=42
):
    """
    Run hyperparameter optimization for metapath classification.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        metapaths: List of metapaths (for single-path) or list of lists (for multi-path)
        model_type: Model type ('rf', 'mlp', 'gcn', 'sage')
        multi_label: Whether to do multi-label classification
        feature_opts: Dict with feature options
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        seed: Random seed
        
    Returns:
        Dictionary with optimization results
    """
    # Prepare features
    try:
        feature_opts = feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
        X = prepare_node_features(
            graph, 
            community_labels,
            use_degree=feature_opts.get('use_degree', True),
            use_clustering=feature_opts.get('use_clustering', True),
            use_node_features=feature_opts.get('use_node_features', True)
        )
    except ValueError as e:
        return {'error': str(e)}
    
    # Prepare labels - either single metapath or multi-metapath
    if not multi_label:
        # Single metapath classification
        metapath = metapaths[0] if isinstance(metapaths[0], list) else metapaths
        instances = find_metapath_instances(graph.graph if hasattr(graph, 'graph') else graph, 
                                          community_labels, metapath)
        participating_nodes = set()
        for instance in instances:
            participating_nodes.update(instance)
        Y = np.array([1 if i in participating_nodes else 0 for i in range(X.shape[0])])
    else:
        # Multi-label classification
        Y = np.zeros((X.shape[0], len(metapaths)), dtype=int)
        for j, metapath in enumerate(metapaths):
            instances = find_metapath_instances(graph.graph if hasattr(graph, 'graph') else graph, 
                                              community_labels, metapath)
            participating_nodes = set()
            for instance in instances:
                participating_nodes.update(instance)
            for i in range(X.shape[0]):
                if i in participating_nodes:
                    Y[i, j] = 1
    
    # Create splits
    n_classes = 2 if not multi_label else Y.shape[1]
    n_nodes = X.shape[0]
    
    if not multi_label:
        splits = create_train_val_test_split(n_nodes, Y, seed=seed)
    else:
        # For multi-label, use a regular split (stratification is more complex)
        indices = np.arange(n_nodes)
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=seed)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=seed)
        
        splits = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
    
    # Create the objective function for the appropriate model type
    if model_type == 'rf':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=seed
            )
            
            if not multi_label:
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return f1_score(Y[splits['val_indices']], y_pred, average='macro')
            else:
                from sklearn.multioutput import MultiOutputClassifier
                model = MultiOutputClassifier(model)
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return f1_score(Y[splits['val_indices']], y_pred, average='macro')
    
    elif model_type == 'mlp':
        def objective(trial):
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 16, 256))
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
            
            model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=learning_rate,
                alpha=alpha,
                max_iter=500,
                random_state=seed
            )
            
            if not multi_label:
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return f1_score(Y[splits['val_indices']], y_pred, average='macro')
            else:
                from sklearn.multioutput import MultiOutputClassifier
                model = MultiOutputClassifier(model)
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return f1_score(Y[splits['val_indices']], y_pred, average='macro')
    
    elif model_type in ['gcn', 'sage']:
        # Create PyG data
        graph_nx = graph.graph if hasattr(graph, 'graph') else graph
        n_nodes = X.shape[0]
        data = prepare_graph_data(graph_nx, X)
        # Add train/val/test masks and labels
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[splits['train_indices']] = True
        val_mask[splits['val_indices']] = True
        test_mask[splits['test_indices']] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        if not multi_label:
            data.y = torch.tensor(Y, dtype=torch.long)
            labels_tensor = torch.tensor(Y, dtype=torch.long)
        else:
            data.y = torch.tensor(Y, dtype=torch.float)  # Use float for multi-label
            labels_tensor = torch.tensor(Y, dtype=torch.float)
        
        def objective(trial):
            hidden_channels = trial.suggest_int('hidden_channels', 16, 256)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.8)
            normalization = trial.suggest_categorical('normalization', ['none', 'batch', 'layer'])
            skip_connection = trial.suggest_categorical('skip_connection', [False, True])
            n_layers = trial.suggest_int('n_layers', 2, 5)
            if model_type == 'gcn':
                model = build_gcn_dynamic(
                    in_channels=X.shape[1],
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    n_layers=n_layers,
                    dropout=dropout,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            else:
                aggr = trial.suggest_categorical('aggr', ['mean', 'max', 'min', 'sum'])
                model = build_sage_dynamic(
                    in_channels=X.shape[1],
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    n_layers=n_layers,
                    dropout=dropout,
                    aggr=aggr,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out, labels_tensor)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                if multi_label:
                    pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float().cpu().numpy()
                    true = labels_tensor[data.val_mask].cpu().numpy()
                    f1 = f1_score(true, pred, average='macro')
                else:
                    pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                    true = labels_tensor[data.val_mask].cpu().numpy()
                    f1 = f1_score(true, pred, average='macro')
            return f1
    
    else:
        return {'error': f"Unknown model type: {model_type}"}
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Train final model with best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Train final model
    if model_type == 'rf':
        if not multi_label:
            best_model = RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                random_state=seed
            )
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            # Evaluate
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            train_f1 = f1_score(Y[splits['train_indices']], train_pred, average='macro')
            val_f1 = f1_score(Y[splits['val_indices']], val_pred, average='macro')
            test_f1 = f1_score(Y[splits['test_indices']], test_pred, average='macro')
            feature_importance = best_model.feature_importances_
        else:
            from sklearn.multioutput import MultiOutputClassifier
            best_model = MultiOutputClassifier(RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                random_state=seed
            ))
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            # Evaluate
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            train_f1 = f1_score(Y[splits['train_indices']], train_pred, average='macro')
            val_f1 = f1_score(Y[splits['val_indices']], val_pred, average='macro')
            test_f1 = f1_score(Y[splits['test_indices']], test_pred, average='macro')
            feature_importance = best_model.estimators_[0].feature_importances_
    elif model_type == 'mlp':
        if not multi_label:
            hidden_layer_sizes = [best_params[f'n_units_l{i}'] 
                                for i in range(best_params['n_layers'])]
            best_model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=best_params.get('learning_rate', 0.001),
                alpha=best_params.get('alpha', 0.0001),
                max_iter=1000,
                random_state=seed
            )
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            # Evaluate
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            train_f1 = f1_score(Y[splits['train_indices']], train_pred, average='macro')
            val_f1 = f1_score(Y[splits['val_indices']], val_pred, average='macro')
            test_f1 = f1_score(Y[splits['test_indices']], test_pred, average='macro')
            feature_importance = np.abs(best_model.coefs_[0]).mean(axis=1)
        else:
            from sklearn.multioutput import MultiOutputClassifier
            hidden_layer_sizes = [best_params[f'n_units_l{i}'] 
                                for i in range(best_params['n_layers'])]
            best_model = MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=best_params.get('learning_rate', 0.001),
                alpha=best_params.get('alpha', 0.0001),
                max_iter=1000,
                random_state=seed
            ))
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            # Evaluate
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            train_f1 = f1_score(Y[splits['train_indices']], train_pred, average='macro')
            val_f1 = f1_score(Y[splits['val_indices']], val_pred, average='macro')
            test_f1 = f1_score(Y[splits['test_indices']], test_pred, average='macro')
            feature_importance = np.abs(best_model.estimators_[0].coefs_[0]).mean(axis=1)
    elif model_type in ['gcn', 'sage']:
        # Create the best model
        if model_type == 'gcn':
            if not multi_label:
                best_model = build_gcn_dynamic(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=2 if not multi_label else Y.shape[1],
                    n_layers=best_params.get('n_layers', 2),
                    dropout=best_params['dropout'],
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
            else:
                best_model = GCNMultiLabel(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=Y.shape[1],
                    dropout=best_params['dropout'],
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
        else:  # sage
            if not multi_label:
                best_model = build_sage_dynamic(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=2 if not multi_label else Y.shape[1],
                    n_layers=best_params.get('n_layers', 2),
                    dropout=best_params['dropout'],
                    aggr=best_params.get('aggr', 'mean'),
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
            else:
                best_model = GraphSAGEMultiLabel(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=Y.shape[1],
                    dropout=best_params['dropout'],
                    aggr=best_params.get('aggr', 'mean'),
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
        
        optimizer = torch.optim.Adam(
            best_model.parameters(), 
            lr=best_params.get('learning_rate', 0.01),
            weight_decay=best_params.get('weight_decay', 5e-4)
        )
        
        # Use appropriate loss function
        if not multi_label:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Train model with best parameters
        best_model, history = train_gnn_model(
            best_model, 
            data, 
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,  # More epochs for final model
            early_stopping=True,
            patience=50,
            min_epochs=100,
            verbose=True
        )
        
        # Evaluate with best model
        best_model.eval()
        with torch.no_grad():
            out = best_model(data.x, data.edge_index)
            
            if not multi_label:
                train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
                val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
                
                train_true = data.y[data.train_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                test_true = data.y[data.test_mask].cpu().numpy()
                
                train_f1 = f1_score(train_true, train_pred, average='macro')
                val_f1 = f1_score(val_true, val_pred, average='macro')
                test_f1 = f1_score(test_true, test_pred, average='macro')
            else:
                train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float().cpu().numpy()
                val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float().cpu().numpy()
                test_pred = (torch.sigmoid(out[data.test_mask]) > 0.5).float().cpu().numpy()
                
                train_true = data.y[data.train_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                test_true = data.y[data.test_mask].cpu().numpy()
                
                train_f1 = f1_score(train_true, train_pred, average='macro')
                val_f1 = f1_score(val_true, val_pred, average='macro')
                test_f1 = f1_score(test_true, test_pred, average='macro')
        
        # Get feature importance from first layer weights (approximate)
        if model_type == 'gcn':
            feature_importance = np.abs(best_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:  # sage
            if hasattr(best_model.conv1, 'lin_l'):
                feature_importance = np.abs(best_model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
            else:
                feature_importance = np.abs(best_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_model': best_model,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'feature_importance': feature_importance,
        'splits': splits
    }


def run_all_metapath_tasks(
    graph,
    community_labels,
    node_regimes,
    metapaths,
    target_regime=0,
    hop_distance=1,
    combination_type='any',
    model_types=['rf', 'mlp', 'gcn', 'sage']
):
    """
    Run all metapath-based tasks and return combined results.
    """
    results = {
        'single_metapath': {},
        'multi_metapath': {},
        'feature_regime': {}
    }
    for metapath in metapaths:
        metapath_key = '_'.join(map(str, metapath))
        results['single_metapath'][metapath_key] = {}
        for model_type in model_types:
            results['single_metapath'][metapath_key][model_type] = metapath_node_classification(
                graph, community_labels, metapath, model_type=model_type
            )
    results['multi_metapath'][combination_type] = {}
    for model_type in model_types:
        results['multi_metapath'][combination_type][model_type] = multi_metapath_node_classification(
            graph, community_labels, metapaths,
            combination_type=combination_type, model_type=model_type
        )
    for metapath in metapaths:
        metapath_key = '_'.join(map(str, metapath))
        results['feature_regime'][metapath_key] = {}
        for model_type in model_types:
            results['feature_regime'][metapath_key][model_type] = feature_regime_metapath_classification(
                graph, community_labels, metapath, node_regimes,
                target_regime=target_regime, hop_distance=hop_distance,
                model_type=model_type
            )
    return results 

class DynamicGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, normalization='none', skip_connection=False):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.normalization = normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        # Normalization layers
        for i in range(n_layers):
            if normalization == 'batch':
                if i < n_layers - 1:
                    self.norms.append(BatchNorm1d(hidden_channels))
                else:
                    self.norms.append(BatchNorm1d(out_channels))
            elif normalization == 'layer':
                if i < n_layers - 1:
                    self.norms.append(LayerNorm(hidden_channels))
                else:
                    self.norms.append(LayerNorm(out_channels))
            else:
                self.norms.append(None)
    def forward(self, x, edge_index):
        identity = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norms[i] is not None:
                x = self.norms[i](x)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection and i > 0:
                x = x + identity[:, :x.shape[1]] if identity.shape[1] >= x.shape[1] else x
            identity = x
        return F.log_softmax(x, dim=1)

def build_gcn_dynamic(in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, normalization='none', skip_connection=False):
    return DynamicGCN(in_channels, hidden_channels, out_channels, n_layers, dropout, normalization, skip_connection)

class DynamicGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, aggr='mean', normalization='none', skip_connection=False):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.normalization = normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        # Normalization layers
        for i in range(n_layers):
            if normalization == 'batch':
                if i < n_layers - 1:
                    self.norms.append(BatchNorm1d(hidden_channels))
                else:
                    self.norms.append(BatchNorm1d(out_channels))
            elif normalization == 'layer':
                if i < n_layers - 1:
                    self.norms.append(LayerNorm(hidden_channels))
                else:
                    self.norms.append(LayerNorm(out_channels))
            else:
                self.norms.append(None)
    def forward(self, x, edge_index):
        identity = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norms[i] is not None:
                x = self.norms[i](x)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection and i > 0:
                x = x + identity[:, :x.shape[1]] if identity.shape[1] >= x.shape[1] else x
            identity = x
        return F.log_softmax(x, dim=1)

def build_sage_dynamic(in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, aggr='mean', normalization='none', skip_connection=False):
    return DynamicGraphSAGE(in_channels, hidden_channels, out_channels, n_layers, dropout, aggr, normalization, skip_connection)

# Add these improved functions to utils/metapath_analysis.py

def create_consistent_train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, stratify=True, seed=42):
    """
    Create a consistent train/val/test split for all models to use.
    Works for both single-label and multi-label classification.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) for single-label or (n_samples, n_labels) for multi-label
        train_size: Fraction of nodes in training set
        val_size: Fraction of nodes in validation set
        test_size: Fraction of nodes in test set
        stratify: Whether to use stratified sampling (only for single-label)
        seed: Random seed
        
    Returns:
        Dictionary with train, val, test indices and masks
    """
    assert train_size + val_size + test_size <= 1.0, "Split fractions must sum to at most 1"
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # For multi-label data, we can't use stratification directly
    # We'll use a simple random split if y is multi-label
    is_multilabel = len(y.shape) > 1 and y.shape[1] > 1
    stratify_data = None if is_multilabel else y if stratify else None
    
    # First split: train vs (val+test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_size + test_size) / (train_size + val_size + test_size),
        random_state=seed,
        stratify=stratify_data
    )
    
    # Second split: val vs test
    if stratify and not is_multilabel and len(temp_indices) > 0:
        stratify_temp = y[temp_indices]
    else:
        stratify_temp = None
        
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=stratify_temp
    )
    
    # Create boolean masks for PyTorch models
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }

def calculate_f1_score(y_true, y_pred, multilabel=False, average='macro'):
    """
    Standardized F1 score calculation for both single-label and multi-label.
    Works with both numpy arrays and PyTorch tensors.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or probabilities for multi-label that need thresholding)
        multilabel: Whether this is a multi-label classification task
        average: Averaging method for F1 score calculation
        
    Returns:
        F1 score
    """
    # Convert PyTorch tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # For multilabel classification
    if multilabel:
        # Ensure predictions are binary
        if np.any((y_pred > 0) & (y_pred < 1)):
            y_pred = (y_pred > 0.5).astype(float)
        return f1_score(y_true, y_pred, average=average)
    
    # For single-label classification
    else:
        # If predictions are probabilities, get class predictions
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return f1_score(y_true, y_pred, average=average)

class GCNMultiLabel(nn.Module):
    """GCN model for multi-label classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 normalization='none', skip_connection=False):
        super(GCNMultiLabel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(hidden_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(hidden_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity.shape[1] >= x.shape[1]:
            x = x + identity[:, :x.shape[1]]
            
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        
        if self.skip_connection and identity2.shape[1] >= x.shape[1]:
            x = x + identity2[:, :x.shape[1]]
            
        x = self.lin(x)
        return x  # No activation - handled by loss function

class GraphSAGEMultiLabel(nn.Module):
    """GraphSAGE model for multi-label classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 aggr='mean', normalization='none', skip_connection=False):
        super(GraphSAGEMultiLabel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(hidden_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(hidden_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity.shape[1] >= x.shape[1]:
            x = x + identity[:, :x.shape[1]]
            
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity2.shape[1] >= x.shape[1]:
            x = x + identity2[:, :x.shape[1]]
            
        x = self.lin(x)
        return x  # No activation - handled by loss function

def train_gnn_model_improved(model, data, optimizer=None, criterion=None, epochs=200, 
                            lr=0.01, early_stopping=True, patience=20, min_epochs=100, verbose=True):
    """Improved training function that tracks best validation performance throughout training."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    best_val_loss = float('inf')  # Start with infinity for loss minimization
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # Check if this is a multilabel task or regression task based on loss function
    is_multilabel = isinstance(criterion, nn.BCEWithLogitsLoss)
    is_regression = isinstance(criterion, nn.MSELoss)
    
    # Add appropriate metric tracking based on task type
    if not is_regression:
        history['train_f1'] = []
        history['val_f1'] = []
    else:
        history['train_mse'] = []
        history['val_mse'] = []
        history['train_r2'] = []
        history['val_r2'] = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask]).item()
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
            
            # Calculate appropriate metrics based on task
            if is_regression:
                from sklearn.metrics import mean_squared_error, r2_score
                
                # For regression, use MSE and R^2
                train_pred = out[data.train_mask].cpu().numpy()
                val_pred = out[data.val_mask].cpu().numpy()
                train_true = data.y[data.train_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                
                train_mse = mean_squared_error(train_true, train_pred)
                val_mse = mean_squared_error(val_true, val_pred)
                
                # Handle case where R^2 calculation might fail
                try:
                    train_r2 = r2_score(train_true, train_pred)
                    val_r2 = r2_score(val_true, val_pred)
                except:
                    train_r2 = 0.0
                    val_r2 = 0.0
                    
                # Store metrics in history
                history['train_mse'].append(train_mse)
                history['val_mse'].append(val_mse)
                history['train_r2'].append(train_r2)
                history['val_r2'].append(val_r2)
                
                # Use MSE for early stopping
                monitoring_metric = val_mse
                is_better = monitoring_metric < best_val_loss
                
            else:
                # For classification tasks (single or multi-label)
                if is_multilabel:
                    # For multilabel, use sigmoid and threshold
                    train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float().cpu().numpy()
                    val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float().cpu().numpy()
                    train_true = data.y[data.train_mask].cpu().numpy()
                    val_true = data.y[data.val_mask].cpu().numpy()
                else:
                    # For single label, use argmax
                    train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
                    val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                    train_true = data.y[data.train_mask].cpu().numpy()
                    val_true = data.y[data.val_mask].cpu().numpy()
                
                # Calculate F1 scores - use sklearn metrics directly to avoid compatibility issues
                from sklearn.metrics import f1_score
                
                # Make sure shapes match for calculation
                if is_multilabel:
                    train_f1 = f1_score(train_true, train_pred, average='macro')
                    val_f1 = f1_score(val_true, val_pred, average='macro')
                else:
                    train_f1 = f1_score(train_true, train_pred, average='macro')
                    val_f1 = f1_score(val_true, val_pred, average='macro')
                
                # Store F1 scores in history
                history['train_f1'].append(train_f1)
                history['val_f1'].append(val_f1)
                
                # Use validation loss for early stopping
                monitoring_metric = val_loss
                is_better = monitoring_metric < best_val_loss
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Track best model - only save if we have a new best
        if is_better:
            best_val_loss = monitoring_metric
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            if verbose:
                if is_regression:
                    print(f"New best model at epoch {epoch} with val_mse: {val_mse:.4f}")
                else:
                    print(f"New best model at epoch {epoch} with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping and epoch >= min_epochs and patience_counter >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            break
        
        if verbose and epoch % 10 == 0:
            if is_regression:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, '
                      f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}')
            else:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation to ensure history is consistent with best model
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        if is_regression:
            # For regression, update final metrics
            train_pred = out[data.train_mask].cpu().numpy()
            val_pred = out[data.val_mask].cpu().numpy()
            train_true = data.y[data.train_mask].cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            
            from sklearn.metrics import mean_squared_error, r2_score
            train_mse = mean_squared_error(train_true, train_pred)
            val_mse = mean_squared_error(val_true, val_pred)
            
            try:
                train_r2 = r2_score(train_true, train_pred)
                val_r2 = r2_score(val_true, val_pred)
            except:
                train_r2 = 0.0
                val_r2 = 0.0
                
            if verbose:
                print(f'\n=== Final Results ===')
                print(f'Best model from epoch {best_epoch}')
                print(f'Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}')
                print(f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}')
        else:
            # For classification, update final F1 scores
            if is_multilabel:
                train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float().cpu().numpy()
                val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float().cpu().numpy()
            else:
                train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
                val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                
            train_true = data.y[data.train_mask].cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            
            from sklearn.metrics import f1_score
            train_f1 = f1_score(train_true, train_pred, average='macro')
            val_f1 = f1_score(val_true, val_pred, average='macro')
            
            if verbose:
                print(f'\n=== Final Results ===')
                print(f'Best model from epoch {best_epoch}')
                print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
    
    return model, history

def evaluate_model_improved(model, data, mask, multilabel=False):
    """
    Evaluate a trained model with consistent metrics.
    
    Args:
        model: Trained model
        data: PyTorch Geometric data object
        mask: Boolean mask for test data
        multilabel: Whether this is a multi-label task
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        if multilabel:
            preds = (torch.sigmoid(out[mask]) > 0.5).float()
        else:
            preds = out[mask].argmax(dim=1)
            
        labels = data.y[mask]
        
        # Use the standardized F1 calculation
        f1 = calculate_f1_score(labels, preds, multilabel=multilabel, average='macro')
        
        # Calculate accuracy
        if multilabel:
            acc = (preds == labels).float().mean().item()
        else:
            acc = preds.eq(labels).float().mean().item()
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'predictions': preds.cpu().numpy(),
            'true_labels': labels.cpu().numpy()
        }

def multi_metapath_node_classification_improved(
    graph,
    community_labels,
    metapaths,
    model_type='rf',
    feature_opts=None,
    splits=None,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    """
    Improved multi-label classification for metapath prediction.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        metapaths: List of metapaths
        model_type: Model type ('rf', 'mlp', 'gcn', 'sage')
        feature_opts: Feature options dictionary
        splits: Predefined train/val/test split to use (if None, creates new splits)
        train_size, val_size, test_size: Split proportions (used only if splits is None)
        seed: Random seed
        
    Returns:
        Dictionary with classification results
    """
    # Prepare features with consistent options
    feature_opts = feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
    X = prepare_node_features(
        graph,
        community_labels,
        use_degree=feature_opts.get('use_degree', True),
        use_clustering=feature_opts.get('use_clustering', True),
        use_node_features=feature_opts.get('use_node_features', True)
    )
    
    # Prepare multi-label targets
    n_nodes = X.shape[0]
    Y = np.zeros((n_nodes, len(metapaths)), dtype=int)
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    
    for j, metapath in enumerate(metapaths):
        instances = find_metapath_instances(graph_nx, community_labels, metapath)
        participating_nodes = set()
        for instance in instances:
            participating_nodes.update(instance)
        for i in range(n_nodes):
            if i in participating_nodes:
                Y[i, j] = 1
    
    # Calculate participation rates
    label_counts = Y.sum(axis=0)
    participation_rates = label_counts / n_nodes
    
    # Create or use provided train/val/test splits
    if splits is None:
        splits = create_consistent_train_val_test_split(
            X, Y, 
            train_size=train_size, 
            val_size=val_size, 
            test_size=test_size,
            stratify=False,  # Can't easily stratify multi-label
            seed=seed
        )
    
    train_indices = splits['train_indices']
    val_indices = splits['val_indices']
    test_indices = splits['test_indices']
    
    # Train the appropriate model type
    if model_type == 'rf':
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        # Calculate class weights for each label
        class_weights = []
        for j in range(Y.shape[1]):
            pos_weight = len(Y[train_indices]) / (2 * Y[train_indices, j].sum())
            neg_weight = len(Y[train_indices]) / (2 * (len(Y[train_indices]) - Y[train_indices, j].sum()))
            class_weights.append({0: neg_weight, 1: pos_weight})
        
        # Create base classifier with class weights
        base_clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=seed
        )
        
        model = MultiOutputClassifier(base_clf)
        model.fit(X[train_indices], Y[train_indices])
        
        # Make predictions with adjusted threshold
        train_pred_proba = model.predict_proba(X[train_indices])
        val_pred_proba = model.predict_proba(X[val_indices])
        test_pred_proba = model.predict_proba(X[test_indices])
        
        # Convert probabilities to predictions using class-specific thresholds
        train_pred = np.zeros_like(Y[train_indices])
        val_pred = np.zeros_like(Y[val_indices])
        test_pred = np.zeros_like(Y[test_indices])
        
        for j in range(Y.shape[1]):
            # Use class-specific thresholds based on participation rates
            threshold = 0.5 * (1 - participation_rates[j])  # Adjust threshold based on class imbalance
            train_pred[:, j] = (train_pred_proba[j][:, 1] > threshold).astype(int)
            val_pred[:, j] = (val_pred_proba[j][:, 1] > threshold).astype(int)
            test_pred[:, j] = (test_pred_proba[j][:, 1] > threshold).astype(int)
        
        # Calculate metrics with standardized function
        train_f1 = calculate_f1_score(Y[train_indices], train_pred, multilabel=True)
        val_f1 = calculate_f1_score(Y[val_indices], val_pred, multilabel=True)
        test_f1 = calculate_f1_score(Y[test_indices], test_pred, multilabel=True)
        
        # Calculate accuracy
        train_acc = (train_pred == Y[train_indices]).mean()
        val_acc = (val_pred == Y[val_indices]).mean()
        test_acc = (test_pred == Y[test_indices]).mean()
        
        # Extract feature importance from first estimator
        feature_importance = model.estimators_[0].feature_importances_
        history = None
        
    elif model_type == 'mlp':
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.neural_network import MLPClassifier
        
        # Calculate class weights for each label
        class_weights = []
        for j in range(Y.shape[1]):
            pos_weight = len(Y[train_indices]) / (2 * Y[train_indices, j].sum())
            neg_weight = len(Y[train_indices]) / (2 * (len(Y[train_indices]) - Y[train_indices, j].sum()))
            class_weights.append({0: neg_weight, 1: pos_weight})
        
        # Create base classifier with class weights
        base_clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=seed
        )
        
        model = MultiOutputClassifier(base_clf)
        model.fit(X[train_indices], Y[train_indices])
        
        # Make predictions with adjusted threshold
        train_pred_proba = model.predict_proba(X[train_indices])
        val_pred_proba = model.predict_proba(X[val_indices])
        test_pred_proba = model.predict_proba(X[test_indices])
        
        # Convert probabilities to predictions using class-specific thresholds
        train_pred = np.zeros_like(Y[train_indices])
        val_pred = np.zeros_like(Y[val_indices])
        test_pred = np.zeros_like(Y[test_indices])
        
        for j in range(Y.shape[1]):
            # Use class-specific thresholds based on participation rates
            threshold = 0.5 * (1 - participation_rates[j])  # Adjust threshold based on class imbalance
            train_pred[:, j] = (train_pred_proba[j][:, 1] > threshold).astype(int)
            val_pred[:, j] = (val_pred_proba[j][:, 1] > threshold).astype(int)
            test_pred[:, j] = (test_pred_proba[j][:, 1] > threshold).astype(int)
        
        # Calculate metrics with standardized function
        train_f1 = calculate_f1_score(Y[train_indices], train_pred, multilabel=True)
        val_f1 = calculate_f1_score(Y[val_indices], val_pred, multilabel=True)
        test_f1 = calculate_f1_score(Y[test_indices], test_pred, multilabel=True)
        
        # Calculate accuracy
        train_acc = (train_pred == Y[train_indices]).mean()
        val_acc = (val_pred == Y[val_indices]).mean()
        test_acc = (test_pred == Y[test_indices]).mean()
        
        # Extract feature importance
        feature_importance = np.abs(model.estimators_[0].coefs_[0]).mean(axis=1)
        history = None
        
    elif model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph_nx, X)
        
        # Add splits to data
        data.train_mask = splits['train_mask']
        data.val_mask = splits['val_mask']
        data.test_mask = splits['test_mask']
        data.y = torch.tensor(Y, dtype=torch.float)
        
        # Initialize appropriate model
        if model_type == 'gcn':
            model = GCNMultiLabel(
                in_channels=X.shape[1],
                hidden_channels=64,
                out_channels=Y.shape[1],
                dropout=0.5
            )
        else:
            model = GraphSAGEMultiLabel(
                in_channels=X.shape[1],
                hidden_channels=64,
                out_channels=Y.shape[1],
                dropout=0.5
            )
        
        # Train model with improved training function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Use weighted BCE loss for handling class imbalance
        pos_weight = torch.tensor([
            len(Y[train_indices]) / (2 * Y[train_indices, j].sum())
            for j in range(Y.shape[1])
        ])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model, history = train_gnn_model_improved(
            model,
            data,
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,
            early_stopping=True,
            patience=20,
            verbose=True
        )
        
        # Evaluate with improved evaluation function
        train_results = evaluate_model_improved(model, data, data.train_mask, multilabel=True)
        val_results = evaluate_model_improved(model, data, data.val_mask, multilabel=True)
        test_results = evaluate_model_improved(model, data, data.test_mask, multilabel=True)
        
        train_acc = train_results['accuracy']
        val_acc = val_results['accuracy']
        test_acc = test_results['accuracy']
        
        train_f1 = train_results['f1_score']
        val_f1 = val_results['f1_score']
        test_f1 = test_results['f1_score']
        
        train_pred = train_results['predictions']
        val_pred = val_results['predictions']
        test_pred = test_results['predictions']
        
        # Extract feature importance
        if model_type == 'gcn':
            feature_importance = np.abs(model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            feature_importance = np.abs(model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
    
    return {
        'model': model,
        'accuracy': {
            'train': train_acc,
            'val': val_acc,
            'test': test_acc
        },
        'f1_score': {
            'train': train_f1,
            'val': val_f1,
            'test': test_f1
        },
        'participation_rates': participation_rates,
        'feature_importance': feature_importance,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'true_labels': {
            'train': Y[train_indices],
            'val': Y[val_indices],
            'test': Y[test_indices]
        },
        'metapaths': metapaths,
        'history': history,
        'splits': splits
    }

def optimize_hyperparameters_for_metapath_improved(
    graph,
    community_labels,
    metapaths,
    model_type='rf',
    multi_label=False,
    feature_opts=None,
    splits=None,
    n_trials=20,
    timeout=300,
    seed=42
):
    """
    Improved hyperparameter optimization for metapath classification.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        metapaths: List of metapaths
        model_type: Model type ('rf', 'mlp', 'gcn', 'sage')
        multi_label: Whether to do multi-label classification
        feature_opts: Feature options dictionary
        splits: Predefined train/val/test split to use (if None, creates new splits)
        n_trials: Number of Optuna trials
        timeout: Optuna timeout in seconds
        seed: Random seed
        
    Returns:
        Dictionary with optimization results
    """
    # Prepare features with consistent options
    feature_opts = feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
    X = prepare_node_features(
        graph, 
        community_labels,
        use_degree=feature_opts.get('use_degree', True),
        use_clustering=feature_opts.get('use_clustering', True),
        use_node_features=feature_opts.get('use_node_features', True)
    )
    
    # Prepare labels based on metapaths
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    
    if not multi_label:
        # Single metapath classification
        metapath = metapaths[0] if isinstance(metapaths[0], list) else metapaths
        instances = find_metapath_instances(graph_nx, community_labels, metapath)
        participating_nodes = set()
        for instance in instances:
            participating_nodes.update(instance)
        Y = np.array([1 if i in participating_nodes else 0 for i in range(X.shape[0])])
    else:
        # Multi-label classification
        Y = np.zeros((X.shape[0], len(metapaths)), dtype=int)
        for j, metapath in enumerate(metapaths):
            instances = find_metapath_instances(graph_nx, community_labels, metapath)
            participating_nodes = set()
            for instance in instances:
                participating_nodes.update(instance)
            for i in range(X.shape[0]):
                if i in participating_nodes:
                    Y[i, j] = 1
    
    # Create or use provided train/val/test splits
    if splits is None:
        splits = create_consistent_train_val_test_split(
            X, Y, 
            stratify=not multi_label,  # Only stratify for single-label
            seed=seed
        )
    
    # Create PyTorch Geometric data object for GNN models
    if model_type in ['gcn', 'sage']:
        data = prepare_graph_data(graph_nx, X)
        data.train_mask = splits['train_mask']
        data.val_mask = splits['val_mask']
        data.test_mask = splits['test_mask']
        
        if not multi_label:
            data.y = torch.tensor(Y, dtype=torch.long)
        else:
            data.y = torch.tensor(Y, dtype=torch.float)
    
    # Define Optuna objective function for the selected model type
    if model_type == 'rf':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            
            if not multi_label:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed
                )
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return calculate_f1_score(Y[splits['val_indices']], y_pred, multilabel=False)
            else:
                from sklearn.multioutput import MultiOutputClassifier
                model = MultiOutputClassifier(RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed
                ))
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return calculate_f1_score(Y[splits['val_indices']], y_pred, multilabel=True)
    
    elif model_type == 'mlp':
        def objective(trial):
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 16, 256))
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
            
            if not multi_label:
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    learning_rate_init=learning_rate,
                    alpha=alpha,
                    max_iter=500,
                    random_state=seed
                )
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return calculate_f1_score(Y[splits['val_indices']], y_pred, multilabel=False)
            else:
                from sklearn.multioutput import MultiOutputClassifier
                model = MultiOutputClassifier(MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    learning_rate_init=learning_rate,
                    alpha=alpha,
                    max_iter=500,
                    random_state=seed
                ))
                model.fit(X[splits['train_indices']], Y[splits['train_indices']])
                y_pred = model.predict(X[splits['val_indices']])
                return calculate_f1_score(Y[splits['val_indices']], y_pred, multilabel=True)
    
    elif model_type in ['gcn', 'sage']:
        def objective(trial):
            hidden_channels = trial.suggest_int('hidden_channels', 16, 256)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.8)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            normalization = trial.suggest_categorical('normalization', ['none', 'batch', 'layer'])
            skip_connection = trial.suggest_categorical('skip_connection', [False, True])
            
            if model_type == 'gcn':
                if not multi_label:
                    model = GCN(
                        in_channels=X.shape[1],
                        hidden_channels=hidden_channels,
                        out_channels=2 if not multi_label else Y.shape[1],
                        dropout=dropout,
                        normalization=normalization,
                        skip_connection=skip_connection
                    )
                else:
                    model = GCNMultiLabel(
                        in_channels=X.shape[1],
                        hidden_channels=hidden_channels,
                        out_channels=Y.shape[1],
                        dropout=dropout,
                        normalization=normalization,
                        skip_connection=skip_connection
                    )
            else:  # sage
                aggr = trial.suggest_categorical('aggr', ['mean', 'max', 'sum'])
                if not multi_label:
                    model = GraphSAGE(
                        in_channels=X.shape[1],
                        hidden_channels=hidden_channels,
                        out_channels=2 if not multi_label else Y.shape[1],
                        dropout=dropout,
                        aggr=aggr,
                        normalization=normalization,
                        skip_connection=skip_connection
                    )
                else:
                    model = GraphSAGEMultiLabel(
                        in_channels=X.shape[1],
                        hidden_channels=hidden_channels,
                        out_channels=Y.shape[1],
                        dropout=dropout,
                        aggr=aggr,
                        normalization=normalization,
                        skip_connection=skip_connection
                    )
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
            
            # Use shorter training for optimization
            model, history = train_gnn_model_improved(
                model,
                data,
                optimizer=optimizer,
                criterion=criterion,
                epochs=100,  # Shorter for optimization
                early_stopping=True,
                patience=10,  # Shorter for optimization
                verbose=False
            )
            
            # Use the standardized evaluation function
            val_results = evaluate_model_improved(model, data, data.val_mask, multilabel=multi_label)
            return val_results['f1_score']
    
    else:
        return {'error': f"Unknown model type: {model_type}"}
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters and value
    best_params = study.best_params
    best_value = study.best_value
    
    # Train final model with best parameters
    if model_type == 'rf':
        if not multi_label:
            best_model = RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                random_state=seed
            )
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            
            # Make predictions
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            
            # Calculate metrics
            train_f1 = calculate_f1_score(Y[splits['train_indices']], train_pred, multilabel=False)
            val_f1 = calculate_f1_score(Y[splits['val_indices']], val_pred, multilabel=False)
            test_f1 = calculate_f1_score(Y[splits['test_indices']], test_pred, multilabel=False)
            
            # Feature importance
            feature_importance = best_model.feature_importances_
        else:
            from sklearn.multioutput import MultiOutputClassifier
            best_model = MultiOutputClassifier(RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                random_state=seed
            ))
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            
            # Make predictions
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            
            # Calculate metrics
            train_f1 = calculate_f1_score(Y[splits['train_indices']], train_pred, multilabel=True)
            val_f1 = calculate_f1_score(Y[splits['val_indices']], val_pred, multilabel=True)
            test_f1 = calculate_f1_score(Y[splits['test_indices']], test_pred, multilabel=True)
            
            # Feature importance
            feature_importance = best_model.estimators_[0].feature_importances_
    
    elif model_type == 'mlp':
        if not multi_label:
            hidden_layer_sizes = [best_params[f'n_units_l{i}'] 
                               for i in range(best_params['n_layers'])]
            best_model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=best_params.get('learning_rate', 0.001),
                alpha=best_params.get('alpha', 0.0001),
                max_iter=1000,
                random_state=seed
            )
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            
            # Make predictions
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            
            # Calculate metrics
            train_f1 = calculate_f1_score(Y[splits['train_indices']], train_pred, multilabel=False)
            val_f1 = calculate_f1_score(Y[splits['val_indices']], val_pred, multilabel=False)
            test_f1 = calculate_f1_score(Y[splits['test_indices']], test_pred, multilabel=False)
            
            # Feature importance
            feature_importance = np.abs(best_model.coefs_[0]).mean(axis=1)
        else:
            from sklearn.multioutput import MultiOutputClassifier
            hidden_layer_sizes = [best_params[f'n_units_l{i}'] 
                               for i in range(best_params['n_layers'])]
            best_model = MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=best_params.get('learning_rate', 0.001),
                alpha=best_params.get('alpha', 0.0001),
                max_iter=1000,
                random_state=seed
            ))
            best_model.fit(X[splits['train_indices']], Y[splits['train_indices']])
            
            # Make predictions
            train_pred = best_model.predict(X[splits['train_indices']])
            val_pred = best_model.predict(X[splits['val_indices']])
            test_pred = best_model.predict(X[splits['test_indices']])
            
            # Calculate metrics
            train_f1 = calculate_f1_score(Y[splits['train_indices']], train_pred, multilabel=True)
            val_f1 = calculate_f1_score(Y[splits['val_indices']], val_pred, multilabel=True)
            test_f1 = calculate_f1_score(Y[splits['test_indices']], test_pred, multilabel=True)
            
            # Feature importance
            feature_importance = np.abs(best_model.estimators_[0].coefs_[0]).mean(axis=1)
    
    elif model_type in ['gcn', 'sage']:
        # Create the best model
        if model_type == 'gcn':
            if not multi_label:
                best_model = GCN(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=2 if not multi_label else Y.shape[1],
                    dropout=best_params['dropout'],
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
            else:
                best_model = GCNMultiLabel(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=Y.shape[1],
                    dropout=best_params['dropout'],
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
        else:  # sage
            if not multi_label:
                best_model = GraphSAGE(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=2 if not multi_label else Y.shape[1],
                    dropout=best_params['dropout'],
                    aggr=best_params.get('aggr', 'mean'),
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
            else:
                best_model = GraphSAGEMultiLabel(
                    in_channels=X.shape[1],
                    hidden_channels=best_params['hidden_channels'],
                    out_channels=Y.shape[1],
                    dropout=best_params['dropout'],
                    aggr=best_params.get('aggr', 'mean'),
                    normalization=best_params.get('normalization', 'none'),
                    skip_connection=best_params.get('skip_connection', False)
                )
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            best_model.parameters(), 
            lr=best_params.get('learning_rate', 0.01),
            weight_decay=best_params.get('weight_decay', 5e-4)
        )
        
        # Define loss function
        criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        
        # Train model with best parameters (longer training)
        best_model, history = train_gnn_model_improved(
            best_model, 
            data, 
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,
            early_stopping=True,
            patience=20,
            min_epochs=100,
            verbose=True
        )
        
        # Evaluate with best model
        best_model.eval()
        with torch.no_grad():
            out = best_model(data.x, data.edge_index)
            
            if not multi_label:
                train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
                val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
                
                train_true = data.y[data.train_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                test_true = data.y[data.test_mask].cpu().numpy()
                
                train_f1 = f1_score(train_true, train_pred, average='macro')
                val_f1 = f1_score(val_true, val_pred, average='macro')
                test_f1 = f1_score(test_true, test_pred, average='macro')
            else:
                train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float().cpu().numpy()
                val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float().cpu().numpy()
                test_pred = (torch.sigmoid(out[data.test_mask]) > 0.5).float().cpu().numpy()
                
                train_true = data.y[data.train_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                test_true = data.y[data.test_mask].cpu().numpy()
                
                train_f1 = f1_score(train_true, train_pred, average='macro')
                val_f1 = f1_score(val_true, val_pred, average='macro')
                test_f1 = f1_score(test_true, test_pred, average='macro')
        
        # Get feature importance from first layer weights (approximate)
        if model_type == 'gcn':
            feature_importance = np.abs(best_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:  # sage
            if hasattr(best_model.conv1, 'lin_l'):
                feature_importance = np.abs(best_model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
            else:
                feature_importance = np.abs(best_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_model': best_model,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'feature_importance': feature_importance,
        'splits': splits
    }

def khop_metapath_detection(
    graph: nx.Graph,
    community_labels: np.ndarray,
    node_regimes: np.ndarray,
    metapath: List[int],
    k: int
) -> Dict[str, Any]:
    """
    Detect k-hop metapath relationships and label starting nodes based on feature regimes.
    
    Args:
        graph: NetworkX graph
        community_labels: Community labels for each node
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath as a list of community indices
        k: Hop distance along the metapath
        
    Returns:
        Dictionary with results including:
        - labeled_nodes: IDs of nodes labeled by this process
        - labels: New labels for the starting nodes
        - starting_community: The starting community
        - target_community: The k-hop target community
        - path_community_sequence: Sequence of communities along the path
    """
    # Check if k is valid for the given metapath
    if k >= len(metapath):
        raise ValueError(f"k ({k}) must be less than the length of the metapath ({len(metapath)})")
    
    # Identify starting and target communities
    starting_community = metapath[0]
    target_community = metapath[k]
    
    # Find nodes in starting and target communities
    starting_nodes = [node for node in graph.nodes() if community_labels[node] == starting_community]
    target_nodes = [node for node in graph.nodes() if community_labels[node] == target_community]
    
    # Find metapath instances
    instances = find_metapath_instances(graph, community_labels, metapath)
    
    # Collect k-hop relationships
    khop_relationships = []
    for instance in instances:
        if len(instance) > k:
            start_node = instance[0]
            khop_node = instance[k]
            if community_labels[start_node] == starting_community and community_labels[khop_node] == target_community:
                khop_relationships.append((start_node, khop_node))
    
    # Create labels based on feature regimes
    labeled_nodes = []
    labels = {}
    regime_counts = {}
    
    for start_node, khop_node in khop_relationships:
        regime = node_regimes[khop_node]
        labeled_nodes.append(start_node)
        labels[start_node] = regime
        
        # Keep track of regime distribution for statistics
        if regime not in regime_counts:
            regime_counts[regime] = 0
        regime_counts[regime] += 1
    
    return {
        "labeled_nodes": labeled_nodes,
        "labels": labels,
        "starting_community": starting_community,
        "target_community": target_community,
        "path_community_sequence": metapath[:k+1],
        "regime_counts": regime_counts,
        "total_relationships": len(khop_relationships)
    }

def visualize_khop_metapath_detection(
    graph: nx.Graph,
    community_labels: np.ndarray,
    node_regimes: np.ndarray,
    metapath: List[int],
    k: int,
    khop_result: Dict[str, Any],
    title: str = "K-hop Metapath Detection",
    figsize: Tuple[int, int] = (10, 8),
    show_all_nodes: bool = True,
    highlight_starting: bool = True,
    node_size: int = 80,
    edge_width: int = 2
) -> plt.Figure:
    """
    Visualize k-hop metapath detection results.
    
    Args:
        graph: NetworkX graph
        community_labels: Community labels for each node
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath
        k: Hop distance along the metapath
        khop_result: Results from khop_metapath_detection
        title: Plot title
        figsize: Figure size
        show_all_nodes: Whether to show all nodes in the graph
        highlight_starting: Whether to highlight starting nodes
        node_size: Base size for nodes
        edge_width: Width of highlighted edges
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get a layout for the graph
    pos = nx.spring_layout(graph, seed=42)
    
    # Get all communities involved in the metapath up to k
    metapath_communities = metapath[:k+1]
    
    # Draw nodes by community with different colors
    communities = sorted(set(community_labels))
    cmap = plt.cm.tab20
    
    if show_all_nodes:
        # Draw background nodes with light gray color
        nx.draw_networkx_nodes(
            graph, 
            pos, 
            node_color='lightgray',
            node_size=node_size * 0.5,
            alpha=0.3,
            ax=ax
        )
        
        # Draw edges in light gray
        nx.draw_networkx_edges(
            graph,
            pos,
            width=0.5,
            alpha=0.2,
            edge_color='gray',
            ax=ax
        )
    
    # Draw nodes in the metapath communities
    for i, comm in enumerate(metapath_communities):
        nodes = [node for node in graph.nodes() if community_labels[node] == comm]
        if nodes:
            nx.draw_networkx_nodes(
                graph, 
                pos, 
                nodelist=nodes,
                node_color=[cmap(i % 20)],
                node_size=node_size * 1.2 if i in [0, k] else node_size,  # Highlight start and k-hop communities
                alpha=0.8,
                ax=ax
            )
    
    # Highlight labeled nodes with special marker
    labeled_nodes = khop_result["labeled_nodes"]
    if labeled_nodes and highlight_starting:
        # Create a colormap for regimes
        regime_cmap = plt.cm.Paired
        
        for node in labeled_nodes:
            regime = khop_result["labels"][node]
            nx.draw_networkx_nodes(
                graph, 
                pos, 
                nodelist=[node],
                node_color=[regime_cmap(regime % 12)],
                node_shape='*',  # Star shape for labeled nodes
                node_size=node_size * 1.5,
                linewidths=1.5,
                edgecolors='black',
                ax=ax
            )
    
    # Highlight metapath instances
    instances = find_metapath_instances(graph, community_labels, metapath)
    for i, instance in enumerate(instances[:5]):  # Limit to 5 instances
        # Draw only the part of the path up to k hops
        path = instance[:k+1]
        if len(path) > 1:  # Make sure path has at least 2 nodes to create edges
            edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edges,
                width=edge_width,
                alpha=0.8,
                edge_color=f'C{i}',
                ax=ax
            )
    
    # Add legend for communities
    handles = []
    labels = []
    for i, comm in enumerate(metapath_communities):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=cmap(i % 20), markersize=10))
        labels.append(f'Community {comm}')
    
    # Add legend for regime labels
    regime_counts = khop_result["regime_counts"]
    if regime_counts:  # Only add if we have regime counts
        regime_cmap = plt.cm.Paired
        for regime, count in regime_counts.items():
            handles.append(plt.Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor=regime_cmap(regime % 12), markersize=12))
            labels.append(f'Regime {regime} ({count})')
    
    ax.legend(handles, labels, loc='best')
    
    # Add title
    ax.set_title(title)
    
    # Turn off axis
    ax.axis('off')
    
    return fig

def get_or_create_regime_prediction_splits(
    graph,
    community_labels,
    node_regimes,
    metapath,
    k,
    normalize_counts=False,
    feature_opts=None,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    force_recreate=False
):
    """
    Get existing or create new consistent train/val/test splits for regime prediction.
    This function will store splits in session state to reuse them across model runs.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath
        k: Hop distance
        normalize_counts: Whether to normalize counts to distributions
        feature_opts: Feature options dictionary
        train_size, val_size, test_size: Split proportions
        seed: Random seed
        force_recreate: Whether to force creation of new splits
        
    Returns:
        Dictionary with splits
    """
    import streamlit as st
    
    # Generate a key for storing these specific splits
    metapath_str = '_'.join(map(str, metapath))
    feature_opts_str = '_'.join([f"{k}_{v}" for k, v in (feature_opts or {}).items()])
    split_key = f"regime_splits_{metapath_str}_{k}_{normalize_counts}_{feature_opts_str}_{seed}"
    
    # Initialize regime_prediction_splits in session state if it doesn't exist
    if 'regime_prediction_splits' not in st.session_state:
        st.session_state.regime_prediction_splits = {}
        
    # Check if we already have these splits
    if not force_recreate and split_key in st.session_state.regime_prediction_splits:
        return st.session_state.regime_prediction_splits[split_key]
    
    # We need to create new splits
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    
    # First, prepare features
    X = prepare_node_features(
        graph,
        community_labels,
        use_degree=feature_opts.get('use_degree', True) if feature_opts else True,
        use_clustering=feature_opts.get('use_clustering', True) if feature_opts else True,
        use_node_features=feature_opts.get('use_node_features', True) if feature_opts else True
    )
    
    # Prepare k-hop regime data
    khop_data = prepare_khop_regime_data(
        graph_nx,
        community_labels,
        node_regimes,
        metapath,
        k,
        normalize=normalize_counts
    )
    
    # Extract data
    starting_nodes = khop_data["starting_nodes"]
    Y = khop_data["regime_counts"]
    
    # Filter to nodes with at least one k-hop neighbor
    valid_nodes = khop_data["nodes_with_neighbors"]
    valid_indices = [i for i, node in enumerate(starting_nodes) if node in valid_nodes]
    
    if not valid_indices:
        raise ValueError(f"No valid nodes found with k-hop neighbors for the given metapath")
    
    starting_nodes = [starting_nodes[i] for i in valid_indices]
    Y = Y[valid_indices]
    
    # Ensure we only use starting nodes that are valid for the feature array
    max_feature_idx = X.shape[0] - 1
    valid_starting_nodes = []
    valid_Y = []
    
    for i, node in enumerate(starting_nodes):
        if node <= max_feature_idx:
            valid_starting_nodes.append(node)
            valid_Y.append(Y[i])
    
    if not valid_starting_nodes:
        raise ValueError(f"No valid nodes found with k-hop neighbors that match the feature array size")
    
    starting_nodes = valid_starting_nodes
    Y = np.array(valid_Y)
    
    # Filter features to only include starting nodes
    node_indices = np.array(starting_nodes)
    node_features = X[node_indices]
    
    # Create the splits
    splits = create_consistent_train_val_test_split(
        node_features, Y, 
        train_size=train_size, 
        val_size=val_size, 
        test_size=test_size,
        stratify=False,
        seed=seed
    )
    
    # Save additional context with the splits for reuse
    splits_with_context = {
        'splits': splits,
        'node_indices': node_indices,
        'X': node_features,
        'Y': Y,
        'starting_nodes': starting_nodes
    }
    
    # Store in session state for future use
    st.session_state.regime_prediction_splits[split_key] = splits_with_context
    
    return splits_with_context

def prepare_khop_regime_data(
    graph,
    community_labels,
    node_regimes,
    metapath,
    k,
    total_regimes=None,
    normalize=False  # New parameter to control normalization
):
    """
    For each node in the starting community, calculate the counts or distribution of
    feature regimes among its k-hop neighbors along the given metapath.
    
    Args:
        graph: NetworkX graph
        community_labels: Node community labels
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath as a list of community indices
        k: Hop distance along the metapath
        total_regimes: Total number of possible regimes (if None, calculated from data)
        normalize: Whether to normalize counts to distributions (default: False)
        
    Returns:
        Dictionary containing:
        - starting_nodes: List of nodes in the starting community
        - regime_counts: Matrix of regime counts for each starting node
        - starting_community: The starting community index
        - target_community: The k-hop target community index
        - nodes_with_neighbors: Nodes that have at least one k-hop neighbor
    """
    # Check if k is valid for the given metapath
    if k >= len(metapath):
        raise ValueError(f"k ({k}) must be less than the length of the metapath ({len(metapath)})")
    
    # Identify starting and target communities
    starting_community = metapath[0]
    target_community = metapath[k]
    
    # Find nodes in starting community
    starting_nodes = [node for node in graph.nodes() if community_labels[node] == starting_community]
    
    # Find metapath instances
    instances = find_metapath_instances(graph, community_labels, metapath)
    
    # Collect all regimes to determine total number if not provided
    all_regimes = set()
    for r in node_regimes:
        if r is not None:
            all_regimes.add(r)
    
    # Set total regimes if not provided
    if total_regimes is None:
        total_regimes = max(all_regimes) + 1
    
    # Initialize counts matrix with zeros
    regime_counts = np.zeros((len(starting_nodes), total_regimes), dtype=int)
    
    # Maps from node ID to index in the starting_nodes list
    node_to_idx = {node: i for i, node in enumerate(starting_nodes)}
    
    # Track nodes with valid k-hop neighbors
    nodes_with_neighbors = set()
    
    # For each metapath instance, find k-hop neighbors
    for instance in instances:
        if len(instance) > k:
            start_node = instance[0]
            khop_node = instance[k]
            
            # Check if start node is in starting community
            if start_node in starting_nodes:
                # Get regime of k-hop neighbor
                regime = node_regimes[khop_node]
                if regime is not None:
                    # Update count
                    start_idx = node_to_idx[start_node]
                    regime_counts[start_idx, regime] += 1
                    nodes_with_neighbors.add(start_node)
    
    # Normalize counts to get probabilities if requested
    if normalize:
        normalized_counts = np.zeros_like(regime_counts, dtype=float)
        for i, node in enumerate(starting_nodes):
            if node in nodes_with_neighbors:
                row_sum = np.sum(regime_counts[i])
                if row_sum > 0:
                    normalized_counts[i] = regime_counts[i] / row_sum
        regime_counts = normalized_counts
    
    return {
        "starting_nodes": starting_nodes,
        "regime_counts": regime_counts,  # Now contains raw counts, not normalized
        "starting_community": starting_community,
        "target_community": target_community,
        "nodes_with_neighbors": list(nodes_with_neighbors)
    }

def khop_regime_prediction(
    graph,
    community_labels,
    node_regimes,
    metapath,
    k,
    model_type='rf',
    task_type='regression',
    normalize_counts=False,
    feature_opts=None,
    splits=None,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    use_cached_splits=True
):
    """
    Perform prediction of k-hop feature regime counts or distributions.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath
        k: Hop distance along the metapath
        model_type: Model type ('rf', 'mlp', 'gcn', 'sage')
        task_type: 'regression' for regime counts/distribution or 'classification' for binary participation
        normalize_counts: Whether to normalize counts to distributions (default: False)
        feature_opts: Feature options dictionary
        splits: Predefined train/val/test split to use (if None, creates new splits)
        train_size, val_size, test_size: Split proportions (used only if splits is None)
        seed: Random seed
        use_cached_splits: Whether to use cached splits from session state
        
    Returns:
        Dictionary with prediction results
    """
    # Prepare features with consistent options (same as multi-metapath classification)
    feature_opts = feature_opts or {'use_degree': True, 'use_clustering': True, 'use_node_features': True}
    
    # Convert to NetworkX graph if it's a GraphSample
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    
    # First, prepare features so we know the feature array shape
    X = prepare_node_features(
        graph,
        community_labels,
        use_degree=feature_opts.get('use_degree', True),
        use_clustering=feature_opts.get('use_clustering', True),
        use_node_features=feature_opts.get('use_node_features', True)
    )
    
    # Prepare k-hop regime data - use raw counts, not normalized
    khop_data = prepare_khop_regime_data(
        graph_nx,
        community_labels,
        node_regimes,
        metapath,
        k,
        normalize=normalize_counts
    )
    
    # Get or create splits - reuse from cache if available
    if splits is None and use_cached_splits:
        try:
            splits_with_context = get_or_create_regime_prediction_splits(
                graph,
                community_labels,
                node_regimes,
                metapath,
                k,
                normalize_counts=normalize_counts,
                feature_opts=feature_opts,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=seed
            )
            splits = splits_with_context['splits']
            node_indices = splits_with_context['node_indices']
            node_features = splits_with_context['X']
            Y = splits_with_context['Y']
            starting_nodes = splits_with_context['starting_nodes']
            
            # We can skip the data preparation steps below since we have everything
            data_preparation_done = True
        except Exception as e:
            print(f"Could not use cached splits: {e}")
            data_preparation_done = False
    else:
        data_preparation_done = False
    
    # If we couldn't use cached splits, prepare the data as before
    if not data_preparation_done:
        # Extract data
        starting_nodes = khop_data["starting_nodes"]
        Y = khop_data["regime_counts"]  # Now contains raw counts, not normalized
        
        # Filter to nodes with at least one k-hop neighbor
        valid_nodes = khop_data["nodes_with_neighbors"]
        valid_indices = [i for i, node in enumerate(starting_nodes) if node in valid_nodes]
        
        if not valid_indices:
            raise ValueError(f"No valid nodes found with k-hop neighbors for the given metapath")
        
        starting_nodes = [starting_nodes[i] for i in valid_indices]
        Y = Y[valid_indices]
        
        # Ensure we only use starting nodes that are valid for the feature array
        max_feature_idx = X.shape[0] - 1
        valid_starting_nodes = []
        valid_Y = []
        
        for i, node in enumerate(starting_nodes):
            if node <= max_feature_idx:
                valid_starting_nodes.append(node)
                valid_Y.append(Y[i])
        
        if not valid_starting_nodes:
            raise ValueError(f"No valid nodes found with k-hop neighbors that match the feature array size")
        
        starting_nodes = valid_starting_nodes
        Y = np.array(valid_Y)
        
        # Filter features to only include starting nodes
        node_indices = np.array(starting_nodes)
        node_features = X[node_indices]
        
        # Create splits if not provided
        if splits is None:
            splits = create_consistent_train_val_test_split(
                node_features, Y, 
                train_size=train_size, 
                val_size=val_size, 
                test_size=test_size,
                stratify=False,  # Can't easily stratify regression targets
                seed=seed
            )
    
    # The rest of the function remains the same...
    train_indices = splits['train_indices']
    val_indices = splits['val_indices']
    test_indices = splits['test_indices']
    
    # Train the appropriate model type
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        # Create base regressor
        base_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=seed
        )
        
        # Wrap in MultiOutputRegressor for multi-target regression
        model = MultiOutputRegressor(base_regressor)
        
        # Train model
        model.fit(node_features[train_indices], Y[train_indices])
        
        # Make predictions
        train_pred = model.predict(node_features[train_indices])
        val_pred = model.predict(node_features[val_indices])
        test_pred = model.predict(node_features[test_indices])
        
        # Round predictions to nearest integer since we're predicting counts
        if not normalize_counts:
            train_pred = np.round(train_pred).astype(int)
            val_pred = np.round(val_pred).astype(int)
            test_pred = np.round(test_pred).astype(int)
            
            # Ensure non-negative counts
            train_pred = np.maximum(train_pred, 0)
            val_pred = np.maximum(val_pred, 0)
            test_pred = np.maximum(test_pred, 0)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(Y[train_indices], train_pred)
        val_mse = mean_squared_error(Y[val_indices], val_pred)
        test_mse = mean_squared_error(Y[test_indices], test_pred)
        
        train_r2 = r2_score(Y[train_indices], train_pred)
        val_r2 = r2_score(Y[val_indices], val_pred)
        test_r2 = r2_score(Y[test_indices], test_pred)
        
        # Extract feature importance
        feature_importance = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
        history = None
        
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        # Create base regressor
        base_regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=seed
        )
        
        # Wrap in MultiOutputRegressor for multi-target regression
        model = MultiOutputRegressor(base_regressor)
        
        # Train model
        model.fit(node_features[train_indices], Y[train_indices])
        
        # Make predictions
        train_pred = model.predict(node_features[train_indices])
        val_pred = model.predict(node_features[val_indices])
        test_pred = model.predict(node_features[test_indices])
        
        # Round predictions to nearest integer since we're predicting counts
        if not normalize_counts:
            train_pred = np.round(train_pred).astype(int)
            val_pred = np.round(val_pred).astype(int)
            test_pred = np.round(test_pred).astype(int)
            
            # Ensure non-negative counts
            train_pred = np.maximum(train_pred, 0)
            val_pred = np.maximum(val_pred, 0)
            test_pred = np.maximum(test_pred, 0)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(Y[train_indices], train_pred)
        val_mse = mean_squared_error(Y[val_indices], val_pred)
        test_mse = mean_squared_error(Y[test_indices], test_pred)
        
        train_r2 = r2_score(Y[train_indices], train_pred)
        val_r2 = r2_score(Y[val_indices], val_pred)
        test_r2 = r2_score(Y[test_indices], test_pred)
        
        # Extract feature importance - may not be directly available for MLP
        try:
            # Try to get coefficients from first layer of first estimator
            feature_importance = np.abs(model.estimators_[0].coefs_[0]).mean(axis=1)
        except:
            # If not available, create placeholder
            feature_importance = np.ones(node_features.shape[1]) / node_features.shape[1]
            
        history = None
        
    elif model_type in ['gcn', 'sage']:
        # Create a mapping from original node IDs to continuous indices for PyG
        node_mapping = {node: i for i, node in enumerate(starting_nodes)}
        
        # Prepare PyG data
        data = prepare_graph_data_for_subset(
            graph_nx,
            X,
            node_indices,
            node_mapping
        )
        
        # Add labels
        data.y = torch.tensor(Y, dtype=torch.float)
        
        # Add splits
        num_nodes = data.x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Ensure indices are within bounds
        valid_train = [i for i in train_indices if i < num_nodes]
        valid_val = [i for i in val_indices if i < num_nodes]
        valid_test = [i for i in test_indices if i < num_nodes]
        
        train_mask[valid_train] = True
        val_mask[valid_val] = True
        test_mask[valid_test] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # Initialize appropriate model
        if model_type == 'gcn':
            if task_type == 'regression':
                model = GCNRegression(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=Y.shape[1],
                    dropout=0.5
                )
            else:
                model = GCNMultiLabel(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=Y.shape[1],
                    dropout=0.5
                )
        else:
            if task_type == 'regression':
                model = GraphSAGERegression(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=Y.shape[1],
                    dropout=0.5
                )
            else:
                model = GraphSAGEMultiLabel(
                    in_channels=data.x.size(1),
                    hidden_channels=64,
                    out_channels=Y.shape[1],
                    dropout=0.5
                )
        
        # Train model with improved training function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Use appropriate loss function based on task
        if task_type == 'regression':
            criterion = nn.MSELoss()  # Use MSE loss for regression
        else:
            # Use weighted BCE loss for handling class imbalance in classification
            pos_weight = torch.ones(Y.shape[1]) * 2.0  # Default weight if calculation fails
            try:
                pos_weight = torch.tensor([
                    len(Y[train_indices]) / (2 * max(Y[train_indices, j].sum(), 1))
                    for j in range(Y.shape[1])
                ])
            except Exception as e:
                print(f"Warning: Could not calculate pos_weight: {e}")
                    
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Handle the case where there are no nodes in one of the masks
        if not train_mask.any() or not val_mask.any():
            raise ValueError("Not enough nodes for train/val/test split. Try with more nodes or different metapath.")
            
        model, history = train_gnn_model_improved(
            model,
            data,
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,
            early_stopping=True,
            patience=50,
            verbose=False
        )
        
        # Calculate metrics
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # No activation needed for regression, use sigmoid for classification
            if task_type == 'classification':
                train_pred = torch.sigmoid(out[data.train_mask]).cpu().numpy()
                val_pred = torch.sigmoid(out[data.val_mask]).cpu().numpy()
                test_pred = torch.sigmoid(out[data.test_mask]).cpu().numpy()
            else:
                train_pred = out[data.train_mask].cpu().numpy()
                val_pred = out[data.val_mask].cpu().numpy()
                test_pred = out[data.test_mask].cpu().numpy()
                
                # Round predictions to nearest integer since we're predicting counts
                if not normalize_counts:
                    train_pred = np.round(train_pred).astype(int)
                    val_pred = np.round(val_pred).astype(int)
                    test_pred = np.round(test_pred).astype(int)
                    
                    # Ensure non-negative counts
                    train_pred = np.maximum(train_pred, 0)
                    val_pred = np.maximum(val_pred, 0)
                    test_pred = np.maximum(test_pred, 0)
            
            train_true = Y[train_indices]
            val_true = Y[val_indices]
            test_true = Y[test_indices]
            
            # Calculate appropriate metrics
            if task_type == 'regression':
                from sklearn.metrics import mean_squared_error, r2_score
                
                train_mse = mean_squared_error(train_true, train_pred)
                val_mse = mean_squared_error(val_true, val_pred)
                test_mse = mean_squared_error(test_true, test_pred)
                
                train_r2 = r2_score(train_true, train_pred)
                val_r2 = r2_score(val_true, val_pred)
                test_r2 = r2_score(test_true, test_pred)
            else:
                # For classification, keep original metrics
                from sklearn.metrics import accuracy_score, f1_score
                
                train_pred_binary = (train_pred > 0.5).astype(float)
                val_pred_binary = (val_pred > 0.5).astype(float)
                test_pred_binary = (test_pred > 0.5).astype(float)
                
                train_acc = accuracy_score(train_true.flatten(), train_pred_binary.flatten())
                val_acc = accuracy_score(val_true.flatten(), val_pred_binary.flatten())
                test_acc = accuracy_score(test_true.flatten(), test_pred_binary.flatten())
                
                train_f1 = f1_score(train_true, train_pred_binary, average='macro')
                val_f1 = f1_score(val_true, val_pred_binary, average='macro')
                test_f1 = f1_score(test_true, test_pred_binary, average='macro')
        
        # Extract feature importance from first layer weights
        if model_type == 'gcn':
            feature_importance = np.abs(model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            feature_importance = np.abs(model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
    
    # Return results with metrics appropriate to the task
    results = {
        'model': model,
        'node_indices': node_indices,
        'feature_importance': feature_importance,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'true_values': {
            'train': Y[train_indices],
            'val': Y[val_indices],
            'test': Y[test_indices]
        },
        'metapath': metapath,
        'k': k,
        'history': history,
        'splits': splits,
        'khop_data': khop_data,
        'normalize_counts': normalize_counts  # Store whether counts were normalized
    }
    
    # Add task-specific metrics
    if task_type == 'regression':
        results['metrics'] = {
            'mse': {
                'train': train_mse,
                'val': val_mse,
                'test': test_mse
            },
            'r2': {
                'train': train_r2,
                'val': val_r2,
                'test': test_r2
            }
        }
    else:
        results['metrics'] = {
            'accuracy': {
                'train': train_acc,
                'val': val_acc,
                'test': test_acc
            },
            'f1_score': {
                'train': train_f1,
                'val': val_f1,
                'test': test_f1
            }
        }
    
    return results

def prepare_graph_data_for_subset(graph, features, node_indices, node_mapping=None):
    """
    Convert NetworkX graph to PyTorch Geometric format for a subset of nodes.
    
    Args:
        graph: NetworkX graph
        features: Node feature matrix
        node_indices: Indices of nodes to include
        node_mapping: Optional mapping from original node IDs to new indices
        
    Returns:
        PyTorch Geometric Data object
    """
    if node_mapping is None:
        node_mapping = {node: i for i, node in enumerate(node_indices)}
    
    # Get subgraph containing only the nodes of interest
    # Note: We need to ensure all nodes are valid for the graph
    valid_nodes = [n for n in node_indices if n in graph.nodes()]
    subgraph = graph.subgraph(valid_nodes)
    
    # Extract edges, mapping to new node indices
    edge_list = list(subgraph.edges())
    if edge_list:
        edge_index = []
        for u, v in edge_list:
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                # Make undirected (add reverse edges)
                edge_index.append([node_mapping[v], node_mapping[u]])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Extract features for the subset
    # Make sure we're only accessing valid indices
    max_idx = features.shape[0] - 1
    valid_indices = [idx for idx in node_indices if idx <= max_idx]
    if len(valid_indices) < len(node_indices):
        print(f"Warning: {len(node_indices) - len(valid_indices)} nodes had indices out of bounds for features array")
    
    x = torch.tensor(features[valid_indices], dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data

class GCNRegression(nn.Module):
    """GCN model for regression tasks."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 normalization='none', skip_connection=False):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(hidden_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(hidden_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity.shape[1] >= x.shape[1]:
            x = x + identity[:, :x.shape[1]]
            
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        
        if self.skip_connection and identity2.shape[1] >= x.shape[1]:
            x = x + identity2[:, :x.shape[1]]
            
        x = self.lin(x)
        return x  # No activation for regression

class GraphSAGERegression(nn.Module):
    """GraphSAGE model for regression tasks."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 aggr='mean', normalization='none', skip_connection=False):
        super(GraphSAGERegression, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.normalization = normalization
        self.skip_connection = skip_connection
        
        if normalization == 'batch':
            self.norm1 = BatchNorm1d(hidden_channels)
            self.norm2 = BatchNorm1d(hidden_channels)
        elif normalization == 'layer':
            self.norm1 = LayerNorm(hidden_channels)
            self.norm2 = LayerNorm(hidden_channels)
        else:
            self.norm1 = self.norm2 = None

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity.shape[1] >= x.shape[1]:
            x = x + identity[:, :x.shape[1]]
            
        identity2 = x
        x = self.conv2(x, edge_index)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.skip_connection and identity2.shape[1] >= x.shape[1]:
            x = x + identity2[:, :x.shape[1]]
            
        x = self.lin(x)
        return x  # No activation for regression
    
def visualize_regime_prediction_performance(results, model_type):
    """
    Visualize the performance of regime count/distribution prediction.
    
    Args:
        results: Results dictionary from khop_regime_prediction
        model_type: Type of model used for prediction
        
    Returns:
        Matplotlib figure with visualizations
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot metrics
    metrics = ['train', 'val', 'test']
    mse_values = [results['metrics']['mse'][m] for m in metrics]
    r2_values = [results['metrics']['r2'][m] for m in metrics]
    
    # MSE plot
    axs[0, 0].bar(metrics, mse_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 0].set_title(f'{model_type.upper()} Model - MSE')
    axs[0, 0].set_ylabel('Mean Squared Error')
    for i, v in enumerate(mse_values):
        axs[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # R² plot
    axs[0, 1].bar(metrics, r2_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 1].set_title(f'{model_type.upper()} Model - R² Score')
    axs[0, 1].set_ylabel('R² Score')
    axs[0, 1].set_ylim(-0.1, 1.0)  # R² can be negative but most useful range is 0-1
    for i, v in enumerate(r2_values):
        axs[0, 1].text(i, max(0, v) + 0.05, f'{v:.3f}', ha='center')
    
    # Plot predictions vs. true values for all sets with different colors
    # Get the data for all sets
    train_true = results['true_values']['train'].flatten()
    train_pred = results['predictions']['train'].flatten()
    val_true = results['true_values']['val'].flatten()
    val_pred = results['predictions']['val'].flatten()
    test_true = results['true_values']['test'].flatten()
    test_pred = results['predictions']['test'].flatten()
    
    # Determine if we're working with counts or normalized values
    is_normalized = results.get('normalize_counts', False)
    
    # Adjust scatter plot based on counts vs. normalized values
    if is_normalized:
        max_value = 1.0
        axs[1, 0].set_xlim(0, max_value)
        axs[1, 0].set_ylim(0, max_value)
        ideal_line = [0, max_value]
    else:
        # For counts, determine appropriate range
        all_values = np.concatenate([train_true, train_pred, val_true, val_pred, test_true, test_pred])
        max_value = np.max(all_values) * 1.1
        axs[1, 0].set_xlim(0, max_value)
        axs[1, 0].set_ylim(0, max_value)
        ideal_line = [0, max_value]
    
    # Plot each set with different colors and markers
    axs[1, 0].scatter(train_true, train_pred, alpha=0.5, color='#3498db', marker='o', label='Train')
    axs[1, 0].scatter(val_true, val_pred, alpha=0.5, color='#2ecc71', marker='s', label='Validation')
    axs[1, 0].scatter(test_true, test_pred, alpha=0.5, color='#e74c3c', marker='^', label='Test')
    
    # Plot the ideal line
    axs[1, 0].plot(ideal_line, ideal_line, 'k--')  # Diagonal line for perfect predictions
    
    # Add labels and legend
    axs[1, 0].set_xlabel('True Values')
    axs[1, 0].set_ylabel('Predicted Values')
    axs[1, 0].set_title('Predictions vs. True Values')
    axs[1, 0].legend(loc='best')
    
    # Feature importance
    if 'feature_importance' in results and results['feature_importance'] is not None:
        importance = results['feature_importance']
        if len(importance) > 10:
            indices = np.argsort(importance)[-10:]
            importance = importance[indices]
            feature_names = [f"Feature {i}" for i in indices]
        else:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        
        # Sort by importance for better visualization
        sorted_indices = np.argsort(importance)
        importance = importance[sorted_indices]
        feature_names = [feature_names[i] for i in sorted_indices]
        
        axs[1, 1].barh(feature_names, importance, color='#9b59b6')
        axs[1, 1].set_title('Feature Importance')
        axs[1, 1].set_xlabel('Importance')
    elif model_type in ['gcn', 'sage'] and 'history' in results:
        history = results['history']
        if history and 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axs[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
            if 'val_loss' in history:
                axs[1, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
            axs[1, 1].set_title('Training History')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].legend()
    
    plt.tight_layout()
    return fig

def optimize_hyperparameters_for_regime_prediction(
    graph,
    community_labels,
    node_regimes,
    metapath,
    k,
    model_type='rf',
    feature_opts=None,
    normalize_counts=False,
    splits=None,
    n_trials=20,
    timeout=300,
    seed=42
):
    """
    Run hyperparameter optimization for regime count/distribution prediction.
    
    Args:
        graph: NetworkX graph or GraphSample
        community_labels: Community labels for each node
        node_regimes: Feature regime assignments for each node
        metapath: Selected metapath
        k: Hop distance
        model_type: Model type ('rf', 'mlp', 'gcn', 'sage')
        feature_opts: Feature options dictionary
        normalize_counts: Whether to normalize counts to distributions
        splits: Predefined train/val/test split to use (if None, creates new splits)
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        seed: Random seed
        
    Returns:
        Dictionary with optimization results
    """
    # Prepare data
    graph_nx = graph.graph if hasattr(graph, 'graph') else graph
    
    # First, prepare features so we know the feature array shape
    X = prepare_node_features(
        graph,
        community_labels,
        use_degree=feature_opts.get('use_degree', True),
        use_clustering=feature_opts.get('use_clustering', True),
        use_node_features=feature_opts.get('use_node_features', True)
    )
    
    # Prepare k-hop regime data
    khop_data = prepare_khop_regime_data(
        graph_nx,
        community_labels,
        node_regimes,
        metapath,
        k,
        normalize=normalize_counts
    )
    
    # Extract data
    starting_nodes = khop_data["starting_nodes"]
    Y = khop_data["regime_counts"]
    
    # Filter to nodes with at least one k-hop neighbor
    valid_nodes = khop_data["nodes_with_neighbors"]
    valid_indices = [i for i, node in enumerate(starting_nodes) if node in valid_nodes]
    
    if not valid_indices:
        raise ValueError(f"No valid nodes found with k-hop neighbors for the given metapath")
    
    starting_nodes = [starting_nodes[i] for i in valid_indices]
    Y = Y[valid_indices]
    
    # Ensure we only use starting nodes that are valid for the feature array
    max_feature_idx = X.shape[0] - 1
    valid_starting_nodes = []
    valid_Y = []
    
    for i, node in enumerate(starting_nodes):
        if node <= max_feature_idx:
            valid_starting_nodes.append(node)
            valid_Y.append(Y[i])
    
    if not valid_starting_nodes:
        raise ValueError(f"No valid nodes found with k-hop neighbors that match the feature array size")
    
    starting_nodes = valid_starting_nodes
    Y = np.array(valid_Y)
    
    # Filter features to only include starting nodes
    node_indices = np.array(starting_nodes)
    node_features = X[node_indices]
    
    # Create splits if not provided
    if splits is None:
        splits = create_consistent_train_val_test_split(
            node_features, Y, 
            train_size=0.7, 
            val_size=0.15, 
            test_size=0.15,
            stratify=False,
            seed=seed
        )
    
    # Import libraries needed for all model types
    import optuna
    from sklearn.metrics import mean_squared_error, r2_score
    import copy
    
    # Define objective function based on model type
    objective = None  # Initialize to avoid UnboundLocalError
    best_model = None
    best_score = float('-inf')  # Changed to negative infinity since we're maximizing
    
    if model_type == 'rf':
        def rf_objective(trial):
            nonlocal best_model, best_score
            
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            base_regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=seed
            )
            
            model = MultiOutputRegressor(base_regressor)
            
            model.fit(node_features[splits['train_indices']], Y[splits['train_indices']])
            y_pred = model.predict(node_features[splits['val_indices']])
            
            # Round predictions if we're predicting raw counts
            if not normalize_counts:
                y_pred = np.round(y_pred).astype(int)
                # Ensure non-negative counts
                y_pred = np.maximum(y_pred, 0)
            
            # Calculate both metrics
            mse = mean_squared_error(Y[splits['val_indices']], y_pred)
            r2 = r2_score(Y[splits['val_indices']], y_pred)
            
            # Combine metrics (R² - MSE)
            # We negate MSE since we want to minimize it
            combined_score = r2 - mse
            
            # Update best model if this is better
            if combined_score > best_score:
                best_score = combined_score
                best_model = copy.deepcopy(model)
            
            return combined_score
        
        objective = rf_objective
            
    elif model_type == 'mlp':
        def mlp_objective(trial):
            nonlocal best_model, best_score
            
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 16, 256))
            
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
            
            from sklearn.neural_network import MLPRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            base_regressor = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                learning_rate_init=learning_rate,
                alpha=alpha,
                max_iter=500,
                random_state=seed
            )
            
            model = MultiOutputRegressor(base_regressor)
            
            model.fit(node_features[splits['train_indices']], Y[splits['train_indices']])
            y_pred = model.predict(node_features[splits['val_indices']])
            
            # Round predictions if we're predicting raw counts
            if not normalize_counts:
                y_pred = np.round(y_pred).astype(int)
                # Ensure non-negative counts
                y_pred = np.maximum(y_pred, 0)
            
            # Calculate both metrics
            mse = mean_squared_error(Y[splits['val_indices']], y_pred)
            r2 = r2_score(Y[splits['val_indices']], y_pred)
            
            # Combine metrics (R² - MSE)
            combined_score = r2 - mse
            
            # Update best model if this is better
            if combined_score > best_score:
                best_score = combined_score
                best_model = copy.deepcopy(model)
            
            return combined_score
        
        objective = mlp_objective
            
    elif model_type in ['gcn', 'sage']:
        # Create PyG data for GNN optimization
        node_mapping = {node: i for i, node in enumerate(starting_nodes)}
        data = prepare_graph_data_for_subset(graph_nx, X, node_indices, node_mapping)
        
        # Add labels and masks
        data.y = torch.tensor(Y, dtype=torch.float)
        
        train_mask = torch.zeros(len(starting_nodes), dtype=torch.bool)
        val_mask = torch.zeros(len(starting_nodes), dtype=torch.bool)
        test_mask = torch.zeros(len(starting_nodes), dtype=torch.bool)
        
        # Ensure indices are within bounds
        valid_train = [i for i in splits['train_indices'] if i < len(starting_nodes)]
        valid_val = [i for i in splits['val_indices'] if i < len(starting_nodes)]
        valid_test = [i for i in splits['test_indices'] if i < len(starting_nodes)]
        
        train_mask[valid_train] = True
        val_mask[valid_val] = True
        test_mask[valid_test] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        def gnn_objective(trial):
            nonlocal best_model, best_score
            
            hidden_channels = trial.suggest_int('hidden_channels', 16, 256)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.8)
            normalization = trial.suggest_categorical('normalization', ['none', 'batch', 'layer'])
            skip_connection = trial.suggest_categorical('skip_connection', [False, True])
            
            if model_type == 'gcn':
                model = GCNRegression(
                    in_channels=data.x.size(1),
                    hidden_channels=hidden_channels,
                    out_channels=Y.shape[1],
                    dropout=dropout,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            else:  # sage
                aggr = trial.suggest_categorical('aggr', ['mean', 'max', 'sum'])
                model = GraphSAGERegression(
                    in_channels=data.x.size(1),
                    hidden_channels=hidden_channels,
                    out_channels=Y.shape[1],
                    dropout=dropout,
                    aggr=aggr,
                    normalization=normalization,
                    skip_connection=skip_connection
                )
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            criterion = nn.MSELoss()
            
            # Train model with early stopping
            model, history = train_gnn_model_improved(
                model,
                data,
                optimizer=optimizer,
                criterion=criterion,
                epochs=100,  # Shorter for optimization
                early_stopping=True,
                patience=10,  # Shorter for optimization
                verbose=False
            )
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                y_pred = out[data.val_mask].cpu().numpy()
                
                # Round predictions if we're predicting raw counts
                if not normalize_counts:
                    y_pred = np.round(y_pred).astype(int)
                    # Ensure non-negative counts
                    y_pred = np.maximum(y_pred, 0)
                
                # Calculate both metrics
                mse = mean_squared_error(Y[splits['val_indices']], y_pred)
                r2 = r2_score(Y[splits['val_indices']], y_pred)
                
                # Combine metrics (R² - MSE)
                combined_score = r2 - mse
                
                # Update best model if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = copy.deepcopy(model)
                
                return combined_score
        
        objective = gnn_objective
    
    # Run optimization
    study = optuna.create_study(direction='maximize')  # Changed to maximize since we're using R² - MSE
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters and value
    best_params = study.best_params
    best_value = study.best_value
    
    # Train final model with best parameters on full training set
    if model_type in ['rf', 'mlp']:
        # Use the best model we found during optimization
        final_model = best_model
        
        # Make predictions
        train_pred = final_model.predict(node_features[splits['train_indices']])
        val_pred = final_model.predict(node_features[splits['val_indices']])
        test_pred = final_model.predict(node_features[splits['test_indices']])
        
        # Round predictions for counts
        if not normalize_counts:
            train_pred = np.round(train_pred).astype(int)
            val_pred = np.round(val_pred).astype(int)
            test_pred = np.round(test_pred).astype(int)
            
            # Ensure non-negative counts
            train_pred = np.maximum(train_pred, 0)
            val_pred = np.maximum(val_pred, 0)
            test_pred = np.maximum(test_pred, 0)
        
        # Calculate metrics
        train_mse = mean_squared_error(Y[splits['train_indices']], train_pred)
        val_mse = mean_squared_error(Y[splits['val_indices']], val_pred)
        test_mse = mean_squared_error(Y[splits['test_indices']], test_pred)
        
        train_r2 = r2_score(Y[splits['train_indices']], train_pred)
        val_r2 = r2_score(Y[splits['val_indices']], val_pred)
        test_r2 = r2_score(Y[splits['test_indices']], test_pred)
        
        # Feature importance
        if model_type == 'rf':
            feature_importance = np.mean([estimator.feature_importances_ for estimator in final_model.estimators_], axis=0)
        else:  # mlp
            try:
                feature_importance = np.abs(final_model.estimators_[0].coefs_[0]).mean(axis=1)
            except:
                feature_importance = np.ones(node_features.shape[1]) / node_features.shape[1]
        
        history = None
        
    else:  # GNN models
        # Create and train final model with best parameters
        if model_type == 'gcn':
            final_model = GCNRegression(
                in_channels=data.x.size(1),
                hidden_channels=best_params['hidden_channels'],
                out_channels=Y.shape[1],
                dropout=best_params['dropout'],
                normalization=best_params.get('normalization', 'none'),
                skip_connection=best_params.get('skip_connection', False)
            )
        else:  # sage
            final_model = GraphSAGERegression(
                in_channels=data.x.size(1),
                hidden_channels=best_params['hidden_channels'],
                out_channels=Y.shape[1],
                dropout=best_params['dropout'],
                aggr=best_params.get('aggr', 'mean'),
                normalization=best_params.get('normalization', 'none'),
                skip_connection=best_params.get('skip_connection', False)
            )
        
        optimizer = torch.optim.Adam(
            final_model.parameters(), 
            lr=best_params.get('learning_rate', 0.01),
            weight_decay=best_params.get('weight_decay', 5e-4)
        )
        
        criterion = nn.MSELoss()
        
        # Train final model with longer training
        final_model, history = train_gnn_model_improved(
            final_model,
            data,
            optimizer=optimizer,
            criterion=criterion,
            epochs=300,
            early_stopping=True,
            patience=20,
            min_epochs=100,
            verbose=True
        )
        
        # Evaluate with final model
        final_model.eval()
        with torch.no_grad():
            out = final_model(data.x, data.edge_index)
            
            train_pred = out[data.train_mask].cpu().numpy()
            val_pred = out[data.val_mask].cpu().numpy()
            test_pred = out[data.test_mask].cpu().numpy()
            
            # Round predictions for counts
            if not normalize_counts:
                train_pred = np.round(train_pred).astype(int)
                val_pred = np.round(val_pred).astype(int)
                test_pred = np.round(test_pred).astype(int)
                
                # Ensure non-negative counts
                train_pred = np.maximum(train_pred, 0)
                val_pred = np.maximum(val_pred, 0)
                test_pred = np.maximum(test_pred, 0)
            
            # Calculate metrics
            train_mse = mean_squared_error(Y[splits['train_indices']], train_pred)
            val_mse = mean_squared_error(Y[splits['val_indices']], val_pred)
            test_mse = mean_squared_error(Y[splits['test_indices']], test_pred)
            
            train_r2 = r2_score(Y[splits['train_indices']], train_pred)
            val_r2 = r2_score(Y[splits['val_indices']], val_pred)
            test_r2 = r2_score(Y[splits['test_indices']], test_pred)
        
        # Get feature importance from first layer weights
        if model_type == 'gcn':
            feature_importance = np.abs(final_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
        else:
            if hasattr(final_model.conv1, 'lin_l'):
                feature_importance = np.abs(final_model.conv1.lin_l.weight.detach().cpu().numpy()).mean(axis=0)
            else:
                feature_importance = np.abs(final_model.conv1.lin.weight.detach().cpu().numpy()).mean(axis=0)
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_model': final_model,
        'study': study,
        'splits': splits,
        'khop_data': khop_data,
        'metrics': {
            'mse': {
                'train': train_mse,
                'val': val_mse,
                'test': test_mse
            },
            'r2': {
                'train': train_r2,
                'val': val_r2,
                'test': test_r2
            }
        },
        'feature_importance': feature_importance,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'true_values': {
            'train': Y[splits['train_indices']],
            'val': Y[splits['val_indices']],
            'test': Y[splits['test_indices']]
        },
        'normalize_counts': normalize_counts
    }