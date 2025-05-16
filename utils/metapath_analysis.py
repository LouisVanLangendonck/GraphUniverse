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
    best_val_f1 = 0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'val_raw_outputs': [],  # Store raw model outputs for best epochs
        'val_predictions': [],   # Store predictions for best epochs
        'val_true_labels': []    # Store true labels for best epochs
    }
    
    # Check if this is a multilabel task
    is_multilabel = data.y.dim() > 1 and data.y.size(1) > 1
    
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
            
            # Calculate F1 scores
            if is_multilabel:
                # For multilabel, use sigmoid and threshold
                train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float()
                val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float()
                train_true = data.y[data.train_mask]
                val_true = data.y[data.val_mask]
            else:
                # For single label, use argmax
                train_pred = out[data.train_mask].argmax(dim=1)
                val_pred = out[data.val_mask].argmax(dim=1)
                train_true = data.y[data.train_mask]
                val_true = data.y[data.val_mask]
            
            # Calculate F1 scores using the standardized function
            train_f1 = calculate_f1_score(train_true, train_pred, multilabel=is_multilabel)
            val_f1 = calculate_f1_score(val_true, val_pred, multilabel=is_multilabel)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Track best model - only save if we have a new best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            if verbose:
                print(f"New best model at epoch {epoch} with val_f1: {val_f1:.4f}")
                # Store the raw outputs and predictions for the best model
                history['val_raw_outputs'] = out[data.val_mask].clone()
                history['val_predictions'] = val_pred.clone()
                history['val_true_labels'] = val_true.clone()
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping and epoch >= min_epochs and patience_counter >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            break
        
        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Verification: Check if the restored model gives the same validation F1
    model.eval()
    with torch.no_grad():
        if verbose:
            print("\n=== Model State Verification ===")
            print(f"Best model was from epoch {best_epoch}")
            print(f"Best validation F1 during training: {best_val_f1:.4f}")
            
            # Get current model state before comparison
            current_state = model.state_dict()
            best_model_state = copy.deepcopy(model.state_dict())
            state_changed = False
            for key in best_model_state:
                if not torch.allclose(current_state[key], best_model_state[key]):
                    state_changed = True
                    print(f"WARNING: Model state changed for parameter {key}")
            if not state_changed:
                print("Model state loaded correctly")
        
        # Now evaluate
        out = model(data.x, data.edge_index)
        if is_multilabel:
            val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float()
        else:
            val_pred = out[data.val_mask].argmax(dim=1)
        val_true = data.y[data.val_mask]
        restored_val_f1 = calculate_f1_score(val_true, val_pred, multilabel=is_multilabel)
        
        # Also check training F1 for completeness
        if is_multilabel:
            train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float()
        else:
            train_pred = out[data.train_mask].argmax(dim=1)
        train_true = data.y[data.train_mask]
        restored_train_f1 = calculate_f1_score(train_true, train_pred, multilabel=is_multilabel)
        
        if verbose:
            print("\n=== Prediction Verification ===")
            print(f"Validation predictions shape: {val_pred.shape}")
            print(f"Validation true labels shape: {val_true.shape}")
            print(f"Unique values in val_pred: {torch.unique(val_pred)}")
            print(f"Unique values in val_true: {torch.unique(val_true)}")
            
            # Print distributions based on task type
            if is_multilabel:
                print("\nValidation predictions per class:")
                for i in range(val_pred.shape[1]):
                    pred_counts = torch.bincount(val_pred[:, i].long())
                    true_counts = torch.bincount(val_true[:, i].long())
                    print(f"Class {i}:")
                    print(f"  Predictions: {pred_counts.tolist()}")
                    print(f"  True labels: {true_counts.tolist()}")
            else:
                print(f"Validation predictions distribution: {torch.bincount(val_pred.long())}")
                print(f"Validation true labels distribution: {torch.bincount(val_true.long())}")
            
            # Compare raw outputs and predictions between best epoch and restored model
            print("\n=== Detailed Comparison ===")
            print("Best epoch raw outputs shape:", history['val_raw_outputs'].shape)
            print("Restored model raw outputs shape:", out[data.val_mask].shape)
            
            # Compare raw outputs
            raw_output_diff = torch.abs(history['val_raw_outputs'] - out[data.val_mask])
            print(f"Max difference in raw outputs: {raw_output_diff.max().item():.6f}")
            print(f"Mean difference in raw outputs: {raw_output_diff.mean().item():.6f}")
            
            # Compare predictions
            pred_diff = torch.abs(history['val_predictions'] - val_pred)
            print(f"Number of different predictions: {pred_diff.sum().item()}")
            print(f"Percentage of different predictions: {(pred_diff.sum() / pred_diff.numel() * 100):.2f}%")
            
            # Print some example predictions
            print("\nExample predictions (first 5 samples):")
            print("Best epoch predictions:")
            print(history['val_predictions'][:5])
            print("Restored model predictions:")
            print(val_pred[:5])
            print("True labels:")
            print(val_true[:5])
    
    if verbose:
        print(f'\n=== Final Results ===')
        print(f'Best validation F1 during training: {best_val_f1:.4f}')
        print(f'Restored model validation F1: {restored_val_f1:.4f}')
        print(f'Restored model training F1: {restored_train_f1:.4f}')
        
        # Check if the restored F1 matches the best F1 from history
        max_val_f1_from_history = max(history['val_f1'])
        print(f'Maximum validation F1 from history: {max_val_f1_from_history:.4f}')
        
        if abs(restored_val_f1 - best_val_f1) > 1e-6:
            print("WARNING: Restored model validation F1 does not match best validation F1!")
        if abs(restored_val_f1 - max_val_f1_from_history) > 1e-6:
            print("WARNING: Restored model validation F1 does not match maximum F1 from history!")
    
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