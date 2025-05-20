"""
Evaluation metrics for graph community detection and embedding.

This module provides functions to evaluate community detection algorithms,
measure embedding quality, and assess transfer learning performance.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def modularity(graph: nx.Graph, communities: List[List[int]]) -> float:
    """
    Calculate Newman's modularity for a graph partition.
    
    Args:
        graph: NetworkX graph
        communities: List of communities, where each community is a list of node indices
        
    Returns:
        Modularity score in [-0.5, 1.0]
    """
    # Convert communities to a map of node -> community_id
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
            
    # Check if all nodes are assigned
    for node in graph.nodes():
        if node not in community_map:
            community_map[node] = len(communities)  # Assign to a new community
            
    return nx.algorithms.community.modularity(graph, communities)


def conductance(graph: nx.Graph, community: List[int]) -> float:
    """
    Calculate conductance of a single community.
    Conductance = (number of edges leaving the community) / 
                  min(total degree of community, total degree of rest of graph)
    
    Args:
        graph: NetworkX graph
        community: List of node indices
        
    Returns:
        Conductance in [0, 1] (lower is better)
    """
    community_set = set(community)
    rest_set = set(graph.nodes()) - community_set
    
    # Count edges crossing the boundary
    cut_size = 0
    for u in community_set:
        for v in graph.neighbors(u):
            if v not in community_set:
                cut_size += 1
                
    # Compute volumes
    vol_community = sum(graph.degree(u) for u in community_set)
    vol_rest = sum(graph.degree(u) for u in rest_set)
    
    denominator = min(vol_community, vol_rest)
    if denominator == 0:
        return 0.0  # Isolated component
        
    return cut_size / denominator


def avg_conductance(graph: nx.Graph, communities: List[List[int]]) -> float:
    """
    Calculate average conductance across all communities.
    
    Args:
        graph: NetworkX graph
        communities: List of communities
        
    Returns:
        Average conductance (lower is better)
    """
    return np.mean([conductance(graph, comm) for comm in communities])


def clustering_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate Normalized Mutual Information between two clusterings.
    
    Args:
        true_labels: Ground truth community labels
        pred_labels: Predicted community labels
        
    Returns:
        NMI score in [0, 1] (higher is better)
    """
    return normalized_mutual_info_score(true_labels, pred_labels)


def clustering_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate Adjusted Rand Index between two clusterings.
    
    Args:
        true_labels: Ground truth community labels
        pred_labels: Predicted community labels
        
    Returns:
        ARI score in [-1, 1] (higher is better)
    """
    return adjusted_rand_score(true_labels, pred_labels)


def link_prediction_auc(
    graph: nx.Graph,
    test_edges: List[Tuple[int, int]],
    test_non_edges: List[Tuple[int, int]],
    edge_score_fn: Callable[[int, int], float]
) -> float:
    """
    Evaluate link prediction using AUC score.
    
    Args:
        graph: Original graph
        test_edges: Positive test edges (should not be in the graph)
        test_non_edges: Negative test edges (non-edges)
        edge_score_fn: Function that gives a score to each node pair
        
    Returns:
        AUC score in [0, 1] (higher is better)
    """
    # Get scores for all test edges and non-edges
    pos_scores = [edge_score_fn(u, v) for u, v in test_edges]
    neg_scores = [edge_score_fn(u, v) for u, v in test_non_edges]
    
    # Create labels and scores arrays
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    return roc_auc_score(y_true, y_score)


def node_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for node classification tasks.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    # Single-label metrics
    results["accuracy"] = accuracy_score(true_labels, pred_labels)
    results["f1_micro"] = f1_score(true_labels, pred_labels, average="micro")
    results["f1_macro"] = f1_score(true_labels, pred_labels, average="macro")
    
    # Per-class metrics
    n_classes = len(np.unique(true_labels))
    class_f1 = []
    class_precision = []
    class_recall = []
    
    for i in range(n_classes):
        class_true = (true_labels == i)
        class_pred = (pred_labels == i)
        
        precision = (class_pred & class_true).sum() / max(class_pred.sum(), 1)
        recall = (class_pred & class_true).sum() / max(class_true.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        class_f1.append(f1)
        class_precision.append(precision)
        class_recall.append(recall)
        
    results["avg_precision"] = np.mean(class_precision)
    results["avg_recall"] = np.mean(class_recall)
    results["avg_f1"] = np.mean(class_f1)
    
    return results


def split_edges_for_link_prediction(
    graph: nx.Graph,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1
) -> Dict[str, Union[List[Tuple[int, int]], nx.Graph]]:
    """
    Split edges into train/val/test sets for link prediction.
    
    Args:
        graph: Original graph
        test_ratio: Fraction of edges to use for testing
        val_ratio: Fraction of edges to use for validation
        
    Returns:
        Dictionary containing:
        - train_graph: Graph with only training edges
        - val_edges: List of validation edges
        - test_edges: List of test edges
        - val_non_edges: List of validation non-edges
        - test_non_edges: List of test non-edges
    """
    # Get all edges
    edges = list(graph.edges())
    n_edges = len(edges)
    
    # Calculate split sizes
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    n_train = n_edges - n_test - n_val
    
    # Shuffle edges
    np.random.shuffle(edges)
    
    # Split edges
    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train + n_val]
    test_edges = edges[n_train + n_val:]
    
    # Create training graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(graph.nodes())
    train_graph.add_edges_from(train_edges)
    
    # Sample non-edges for validation and testing
    n_nodes = graph.number_of_nodes()
    max_possible_edges = n_nodes * (n_nodes - 1) // 2
    
    # Get all possible non-edges
    all_edges = set(graph.edges())
    all_possible_edges = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            all_possible_edges.add((i, j))
    
    non_edges = list(all_possible_edges - all_edges)
    np.random.shuffle(non_edges)
    
    # Split non-edges
    n_val_non = min(len(non_edges), n_val)
    n_test_non = min(len(non_edges) - n_val_non, n_test)
    
    val_non_edges = non_edges[:n_val_non]
    test_non_edges = non_edges[n_val_non:n_val_non + n_test_non]
    
    return {
        "train_graph": train_graph,
        "val_edges": val_edges,
        "test_edges": test_edges,
        "val_non_edges": val_non_edges,
        "test_non_edges": test_non_edges
    }


def community_detection_metrics(
    graph: nx.Graph,
    true_communities: List[List[int]],
    pred_communities: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate metrics for community detection evaluation.
    
    Args:
        graph: NetworkX graph
        true_communities: Ground truth communities
        pred_communities: Predicted communities
        
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    # Convert communities to label arrays
    n_nodes = graph.number_of_nodes()
    true_labels = np.zeros(n_nodes, dtype=int)
    pred_labels = np.zeros(n_nodes, dtype=int)
    
    for i, comm in enumerate(true_communities):
        true_labels[comm] = i
    for i, comm in enumerate(pred_communities):
        pred_labels[comm] = i
    
    # Calculate metrics
    results["nmi"] = clustering_nmi(true_labels, pred_labels)
    results["ari"] = clustering_ari(true_labels, pred_labels)
    results["modularity"] = modularity(graph, pred_communities)
    results["avg_conductance"] = avg_conductance(graph, pred_communities)
    
    return results


def embedding_link_prediction(
    embeddings: np.ndarray,
    train_graph: nx.Graph,
    test_data: Dict[str, List[Tuple[int, int]]],
    score_fn: str = "dot"
) -> Dict[str, float]:
    """
    Evaluate embeddings for link prediction.
    
    Args:
        embeddings: Node embeddings
        train_graph: Training graph
        test_data: Dictionary containing test edges and non-edges
        score_fn: Function to compute edge scores ("dot", "cosine", or "l2")
        
    Returns:
        Dictionary of metrics
    """
    # Define edge scoring function
    if score_fn == "dot":
        def edge_score(u, v):
            return np.dot(embeddings[u], embeddings[v])
    elif score_fn == "cosine":
        def edge_score(u, v):
            return np.dot(embeddings[u], embeddings[v]) / (
                np.linalg.norm(embeddings[u]) * np.linalg.norm(embeddings[v])
            )
    else:  # l2
        def edge_score(u, v):
            return -np.linalg.norm(embeddings[u] - embeddings[v])
    
    # Calculate AUC scores
    val_auc = link_prediction_auc(
        train_graph,
        test_data["val_edges"],
        test_data["val_non_edges"],
        edge_score
    )
    
    test_auc = link_prediction_auc(
        train_graph,
        test_data["test_edges"],
        test_data["test_non_edges"],
        edge_score
    )
    
    return {
        "val_auc": val_auc,
        "test_auc": test_auc
    }


def embedding_node_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    n_repeats: int = 5,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate embeddings for node classification.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels
        train_ratio: Fraction of nodes to use for training
        val_ratio: Fraction of nodes to use for validation
        test_ratio: Fraction of nodes to use for testing
        n_repeats: Number of random splits to average over
        seed: Random seed
        
    Returns:
        Dictionary of metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize results
    all_results = []
    
    for _ in range(n_repeats):
        # Split data
        n_nodes = len(labels)
        indices = np.random.permutation(n_nodes)
        
        n_train = int(n_nodes * train_ratio)
        n_val = int(n_nodes * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Train classifier
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(embeddings[train_idx], labels[train_idx])
        
        # Get predictions
        val_pred = clf.predict(embeddings[val_idx])
        test_pred = clf.predict(embeddings[test_idx])
        
        # Calculate metrics
        val_metrics = node_classification_metrics(labels[val_idx], val_pred)
        test_metrics = node_classification_metrics(labels[test_idx], test_pred)
        
        all_results.append({
            "val": val_metrics,
            "test": test_metrics
        })
    
    # Average results
    avg_results = {}
    for split in ["val", "test"]:
        for metric in all_results[0][split].keys():
            values = [r[split][metric] for r in all_results]
            avg_results[f"{split}_{metric}"] = np.mean(values)
            avg_results[f"{split}_{metric}_std"] = np.std(values)
    
    return avg_results


def transfer_learning_metrics(
    pretrain_embeddings: np.ndarray,
    transfer_embeddings: np.ndarray,
    pretrain_labels: np.ndarray,
    transfer_labels: np.ndarray,
    train_ratio: float = 0.1  # Small training set to simulate few-shot learning
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate transfer learning performance.
    
    Args:
        pretrain_embeddings: Embeddings from pretraining
        transfer_embeddings: Embeddings from transfer learning
        pretrain_labels: Labels for pretraining task
        transfer_labels: Labels for transfer task
        train_ratio: Fraction of nodes to use for training
        
    Returns:
        Dictionary of metrics for both tasks
    """
    results = {}
    
    # Evaluate pretraining task
    results["pretrain"] = embedding_node_classification(
        pretrain_embeddings,
        pretrain_labels,
        train_ratio=train_ratio
    )
    
    # Evaluate transfer task
    results["transfer"] = embedding_node_classification(
        transfer_embeddings,
        transfer_labels,
        train_ratio=train_ratio
    )
    
    return results


def topology_similarity_metrics(
    graph1: nx.Graph,
    graph2: nx.Graph
) -> Dict[str, float]:
    """
    Calculate similarity metrics between two graphs.
    
    Args:
        graph1: First graph
        graph2: Second graph
        
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    # Basic statistics
    results["nodes_diff"] = abs(graph1.number_of_nodes() - graph2.number_of_nodes())
    results["edges_diff"] = abs(graph1.number_of_edges() - graph2.number_of_edges())
    
    # Degree distribution similarity
    deg1 = sorted([d for _, d in graph1.degree()])
    deg2 = sorted([d for _, d in graph2.degree()])
    
    # Pad with zeros if needed
    max_deg = max(len(deg1), len(deg2))
    deg1 = np.pad(deg1, (0, max_deg - len(deg1)))
    deg2 = np.pad(deg2, (0, max_deg - len(deg2)))
    
    results["degree_correlation"] = np.corrcoef(deg1, deg2)[0, 1]
    
    # Clustering coefficient similarity
    cc1 = nx.average_clustering(graph1)
    cc2 = nx.average_clustering(graph2)
    results["clustering_diff"] = abs(cc1 - cc2)
    
    # Average shortest path similarity
    try:
        asp1 = nx.average_shortest_path_length(graph1)
        asp2 = nx.average_shortest_path_length(graph2)
        results["path_length_diff"] = abs(asp1 - asp2)
    except nx.NetworkXError:
        results["path_length_diff"] = float("inf")
    
    return results


def community_structure_similarity(
    communities1: List[List[int]],
    communities2: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate similarity metrics between two community structures.
    
    Args:
        communities1: First set of communities
        communities2: Second set of communities
        
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    # Convert to label arrays
    n_nodes = max(max(max(comm) for comm in communities1),
                 max(max(comm) for comm in communities2)) + 1
    
    labels1 = np.zeros(n_nodes, dtype=int)
    labels2 = np.zeros(n_nodes, dtype=int)
    
    for i, comm in enumerate(communities1):
        labels1[comm] = i
    for i, comm in enumerate(communities2):
        labels2[comm] = i
    
    # Calculate metrics
    results["nmi"] = clustering_nmi(labels1, labels2)
    results["ari"] = clustering_ari(labels1, labels2)
    
    # Community size distribution similarity
    size1 = sorted([len(comm) for comm in communities1])
    size2 = sorted([len(comm) for comm in communities2])
    
    # Pad with zeros if needed
    max_size = max(len(size1), len(size2))
    size1 = np.pad(size1, (0, max_size - len(size1)))
    size2 = np.pad(size2, (0, max_size - len(size2)))
    
    results["size_correlation"] = np.corrcoef(size1, size2)[0, 1]
    
    return results


def calculate_task_metrics(
    model,
    data,
    task_type: str,
    split: str = 'test',
    best_model: bool = False
) -> Dict[str, float]:
    """Calculate metrics for a specific task and split."""
    if not best_model and split == 'test':
        return {}  # Don't calculate test metrics during optimization
        
    if task_type == 'community':
        # For community classification
        pred = model(data.x, data.edge_index)
        pred_labels = pred.argmax(dim=1)
        
        # Convert tensors to numpy for metric calculation
        true_labels = data.y.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        
        return {
            'accuracy': float(accuracy_score(true_labels, pred_labels)),
            'f1': float(f1_score(true_labels, pred_labels, average='weighted'))
        }
    else:
        # For counting task
        pred = model(data.x, data.edge_index)
        
        # Convert tensors to numpy for metric calculation
        true_values = data.y.cpu().numpy()
        pred_values = pred.cpu().numpy()
        
        return {
            'mse': float(mean_squared_error(true_values, pred_values)),
            'r2': float(r2_score(true_values, pred_values))
        }


def evaluate_model(
    model,
    train_data,
    val_data,
    test_data,
    task_type: str,
    best_model: bool = False
) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance across all splits."""
    metrics = {}
    
    # Calculate metrics for each split
    for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        metrics[split] = calculate_task_metrics(
            model,
            data,
            task_type,
            split,
            best_model
        )
    
    return metrics