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
    average_precision_score
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
    pred_labels: np.ndarray,
    multilabel: bool = False
) -> Dict[str, float]:
    """
    Calculate metrics for node classification tasks.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        multilabel: Whether this is a multi-label classification
        
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    if multilabel:
        # Multi-label metrics
        results["accuracy"] = (true_labels == pred_labels).mean()
        results["f1_micro"] = f1_score(true_labels, pred_labels, average="micro")
        results["f1_macro"] = f1_score(true_labels, pred_labels, average="macro")
        
        # Per-class metrics
        n_classes = true_labels.shape[1]
        class_f1 = []
        class_precision = []
        class_recall = []
        
        for i in range(n_classes):
            class_true = true_labels[:, i]
            class_pred = pred_labels[:, i]
            
            precision = (class_pred & class_true).sum() / max(class_pred.sum(), 1)
            recall = (class_pred & class_true).sum() / max(class_true.sum(), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            class_f1.append(f1)
            class_precision.append(precision)
            class_recall.append(recall)
            
        results["avg_precision"] = np.mean(class_precision)
        results["avg_recall"] = np.mean(class_recall)
        results["avg_f1"] = np.mean(class_f1)
        
    else:
        # Single-label metrics
        results["accuracy"] = accuracy_score(true_labels, pred_labels)
        results["f1_micro"] = f1_score(true_labels, pred_labels, average="micro")
        results["f1_macro"] = f1_score(true_labels, pred_labels, average="macro")
        
    return results


def split_edges_for_link_prediction(
    graph: nx.Graph,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1
) -> Dict[str, Union[List[Tuple[int, int]], nx.Graph]]:
    """
    Split graph edges for link prediction task.
    
    Args:
        graph: Input graph
        test_ratio: Fraction of edges for testing
        val_ratio: Fraction of edges for validation
        
    Returns:
        Dictionary with train_graph, test_edges, test_non_edges, val_edges, val_non_edges
    """
    # Convert to directed to simplify edge handling
    if not graph.is_directed():
        directed_graph = nx.DiGraph(graph)
    else:
        directed_graph = graph.copy()
        
    # Get all edges
    edges = list(directed_graph.edges())
    n_edges = len(edges)
    
    # Get all non-edges
    non_edges = []
    n_nodes = len(directed_graph)
    n_non_edges = min(n_edges * 2, n_nodes * (n_nodes - 1) // 4)  # Limit number of non-edges
    
    all_nodes = list(directed_graph.nodes())
    np.random.shuffle(all_nodes)
    
    # Sample non-edges more efficiently for large graphs
    existing_edges = set(edges)
    count = 0
    
    while count < n_non_edges and len(all_nodes) > 1:
        u = all_nodes[0]
        candidates = [v for v in all_nodes[1:] if v != u and (u, v) not in existing_edges]
        
        if candidates:
            v = np.random.choice(candidates)
            non_edges.append((u, v))
            count += 1
            
        # Remove processed node
        all_nodes.pop(0)
    
    # If we couldn't sample enough non-edges, continue with random sampling
    while count < n_non_edges:
        u, v = np.random.choice(n_nodes, 2, replace=False)
        if not directed_graph.has_edge(u, v) and (u, v) not in non_edges:
            non_edges.append((u, v))
            count += 1
    
    # Split edges
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    
    np.random.shuffle(edges)
    test_edges = edges[:n_test]
    val_edges = edges[n_test:n_test+n_val]
    train_edges = edges[n_test+n_val:]
    
    # Split non-edges
    n_test_non = int(n_non_edges * test_ratio)
    n_val_non = int(n_non_edges * val_ratio)
    
    np.random.shuffle(non_edges)
    test_non_edges = non_edges[:n_test_non]
    val_non_edges = non_edges[n_test_non:n_test_non+n_val_non]
    
    # Create training graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(directed_graph.nodes(data=True))
    train_graph.add_edges_from(train_edges)
    
    return {
        "train_graph": train_graph,
        "test_edges": test_edges,
        "test_non_edges": test_non_edges,
        "val_edges": val_edges,
        "val_non_edges": val_non_edges
    }


def community_detection_metrics(
    graph: nx.Graph,
    true_communities: List[List[int]],
    pred_communities: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate metrics for community detection.
    
    Args:
        graph: Input graph
        true_communities: Ground truth communities
        pred_communities: Predicted communities
        
    Returns:
        Dictionary of metrics
    """
    # Convert communities to node labels
    true_labels = np.zeros(len(graph), dtype=int)
    for i, comm in enumerate(true_communities):
        for node in comm:
            true_labels[node] = i
            
    pred_labels = np.zeros(len(graph), dtype=int)
    for i, comm in enumerate(pred_communities):
        for node in comm:
            pred_labels[node] = i
    
    # Calculate metrics
    metrics = {
        "nmi": clustering_nmi(true_labels, pred_labels),
        "ari": clustering_ari(true_labels, pred_labels),
        "modularity": modularity(graph, pred_communities),
        "conductance": avg_conductance(graph, pred_communities)
    }
    
    return metrics


def community_overlap_metrics(
    graph: nx.Graph,
    true_memberships: np.ndarray,
    pred_memberships: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Calculate metrics for overlapping community detection.
    
    Args:
        graph: Input graph
        true_memberships: Ground truth membership matrix (n_nodes × n_communities)
        pred_memberships: Predicted membership matrix
        threshold: Threshold for community membership
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary membership indicators
    true_binary = true_memberships > threshold
    pred_binary = pred_memberships > threshold
    
    # Calculate metrics
    metrics = {
        "f1_micro": f1_score(true_binary.flatten(), pred_binary.flatten(), average="micro"),
        "f1_macro": f1_score(true_binary.flatten(), pred_binary.flatten(), average="macro"),
        "avg_precision": average_precision_score(true_binary.flatten(), pred_memberships.flatten())
    }
    
    # Calculate omega index - an overlapping extension of ARI
    omega = overlapping_omega_index(true_binary, pred_binary)
    metrics["omega"] = omega
    
    # Membership prediction metrics per community
    n_communities = true_memberships.shape[1]
    avg_community_auc = 0.0
    
    for c in range(n_communities):
        true_c = true_binary[:, c]
        pred_c = pred_memberships[:, c]
        
        # Skip if all true values are the same (AUC undefined)
        if np.any(true_c) and not np.all(true_c):
            avg_community_auc += roc_auc_score(true_c, pred_c)
    
    metrics["avg_community_auc"] = avg_community_auc / n_communities
    
    return metrics


def overlapping_omega_index(
    true_memberships: np.ndarray,
    pred_memberships: np.ndarray
) -> float:
    """
    Calculate the Omega Index for overlapping community detection.
    
    The Omega Index is a chance-adjusted measure of similarity between
    two overlapping clusterings, extending the Adjusted Rand Index.
    
    Args:
        true_memberships: Binary matrix of true memberships (n_nodes × n_communities)
        pred_memberships: Binary matrix of predicted memberships
        
    Returns:
        Omega Index in [-1, 1] (higher is better)
    """
    n_nodes = true_memberships.shape[0]
    
    # Calculate observed agreement
    same_cluster_count = np.zeros(n_nodes + 1, dtype=int)
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Count communities where both nodes are present
            true_shared = np.sum(np.logical_and(true_memberships[i], true_memberships[j]))
            pred_shared = np.sum(np.logical_and(pred_memberships[i], pred_memberships[j]))
            
            # If they share the same number of communities, count agreement
            if true_shared == pred_shared:
                same_cluster_count[true_shared] += 1
    
    # Total number of pairs
    total_pairs = n_nodes * (n_nodes - 1) // 2
    
    # Calculate observed agreement
    observed = sum(same_cluster_count) / total_pairs
    
    # Calculate expected agreement (random model)
    expected = 0.0
    
    # For each possible number of shared communities
    for k in range(len(same_cluster_count)):
        # Count pairs that share k communities in each clustering
        true_k_count = 0
        pred_k_count = 0
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                true_shared = np.sum(np.logical_and(true_memberships[i], true_memberships[j]))
                if true_shared == k:
                    true_k_count += 1
                    
                pred_shared = np.sum(np.logical_and(pred_memberships[i], pred_memberships[j]))
                if pred_shared == k:
                    pred_k_count += 1
        
        # Probability of randomly having k shared communities
        p_true_k = true_k_count / total_pairs
        p_pred_k = pred_k_count / total_pairs
        
        # Add to expected agreement
        expected += p_true_k * p_pred_k
    
    # Calculate Omega Index
    if abs(observed - expected) < 1e-10:
        return 0.0
    elif expected > 0.999:  # Avoid division by zero
        return 0.0
    else:
        return (observed - expected) / (1 - expected)


def embedding_link_prediction(
    embeddings: np.ndarray,
    train_graph: nx.Graph,
    test_data: Dict[str, List[Tuple[int, int]]],
    score_fn: str = "dot"
) -> Dict[str, float]:
    """
    Evaluate link prediction using node embeddings.
    
    Args:
        embeddings: Node embedding matrix (n_nodes × dim)
        train_graph: Training graph
        test_data: Dictionary with test_edges and test_non_edges
        score_fn: Scoring function ("dot", "cosine", "l2")
        
    Returns:
        Dictionary of metrics
    """
    test_edges = test_data["test_edges"]
    test_non_edges = test_data["test_non_edges"]
    
    # Define scoring function
    if score_fn == "dot":
        def edge_score(u, v):
            return np.dot(embeddings[u], embeddings[v])
    elif score_fn == "cosine":
        def edge_score(u, v):
            norm_u = np.linalg.norm(embeddings[u])
            norm_v = np.linalg.norm(embeddings[v])
            return np.dot(embeddings[u], embeddings[v]) / (norm_u * norm_v + 1e-8)
    elif score_fn == "l2":
        def edge_score(u, v):
            return -np.linalg.norm(embeddings[u] - embeddings[v])
    else:
        raise ValueError(f"Unknown scoring function: {score_fn}")
    
    # Calculate AUC score
    auc_score = link_prediction_auc(train_graph, test_edges, test_non_edges, edge_score)
    
    # Calculate PR-AUC
    pos_scores = [edge_score(u, v) for u, v in test_edges]
    neg_scores = [edge_score(u, v) for u, v in test_non_edges]
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    return {
        "auc": auc_score,
        "pr_auc": pr_auc
    }


def embedding_node_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    multilabel: bool = False,
    n_repeats: int = 5,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate node classification using embeddings.
    
    Args:
        embeddings: Node embedding matrix (n_nodes × dim)
        labels: Node labels
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        test_ratio: Fraction of nodes for testing
        multilabel: Whether labels are multi-label
        n_repeats: Number of random train/val/test splits to average over
        seed: Random seed
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n_nodes = embeddings.shape[0]
    
    # Initialize random generator
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Initialize metrics
    metrics = {
        "accuracy": 0.0,
        "f1_micro": 0.0,
        "f1_macro": 0.0
    }
    
    # Run multiple times with different splits
    for _ in range(n_repeats):
        # Create indices for train/val/test split
        indices = np.arange(n_nodes)
        rng.shuffle(indices)
        
        train_size = int(train_ratio * n_nodes)
        val_size = int(val_ratio * n_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        # Split data
        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_val, y_val = embeddings[val_idx], labels[val_idx]
        X_test, y_test = embeddings[test_idx], labels[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Train classifier
        if multilabel:
            clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear'))
        else:
            clf = LogisticRegression(max_iter=1000, solver='liblinear')
            
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        if multilabel:
            split_metrics = node_classification_metrics(y_test, y_pred, multilabel=True)
        else:
            split_metrics = node_classification_metrics(y_test, y_pred, multilabel=False)
        
        # Accumulate metrics
        for k, v in split_metrics.items():
            if k in metrics:
                metrics[k] += v / n_repeats
            else:
                metrics[k] = v / n_repeats
    
    return metrics


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
        pretrain_embeddings: Embeddings from pretraining graphs
        transfer_embeddings: Embeddings from transfer graphs
        pretrain_labels: Labels from pretraining graphs
        transfer_labels: Labels from transfer graphs
        train_ratio: Fraction of transfer nodes for training
        
    Returns:
        Dictionary of metrics for different scenarios
    """
    results = {}
    
    # Baseline: Train and test on transfer data only
    transfer_metrics = embedding_node_classification(
        transfer_embeddings,
        transfer_labels,
        train_ratio=train_ratio,
        val_ratio=(1-train_ratio)/2,
        test_ratio=(1-train_ratio)/2
    )
    results["transfer_only"] = transfer_metrics
    
    # Transfer from pretrain: Train on pretrain, test on transfer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    scaler = StandardScaler()
    pretrain_scaled = scaler.fit_transform(pretrain_embeddings)
    transfer_scaled = scaler.transform(transfer_embeddings)
    
    # Train on pretrain data
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(pretrain_scaled, pretrain_labels)
    
    # Test on transfer data
    y_pred = clf.predict(transfer_scaled)
    
    # Calculate metrics
    direct_transfer_metrics = node_classification_metrics(transfer_labels, y_pred)
    results["direct_transfer"] = direct_transfer_metrics
    
    # Fine-tuning: Train on pretrain + small subset of transfer
    n_transfer = transfer_embeddings.shape[0]
    n_train = int(train_ratio * n_transfer)
    
    # Random split
    indices = np.arange(n_transfer)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # Combine pretrain and transfer train data
    X_combined = np.vstack([pretrain_scaled, transfer_scaled[train_idx]])
    y_combined = np.concatenate([pretrain_labels, transfer_labels[train_idx]])
    
    # Train on combined data
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_combined, y_combined)
    
    # Test on transfer test data
    y_pred = clf.predict(transfer_scaled[test_idx])
    
    # Calculate metrics
    fine_tuning_metrics = node_classification_metrics(
        transfer_labels[test_idx],
        y_pred
    )
    results["fine_tuning"] = fine_tuning_metrics
    
    return results


def topology_similarity_metrics(
    graph1: nx.Graph,
    graph2: nx.Graph
) -> Dict[str, float]:
    """
    Calculate similarity metrics between graph topologies.
    
    Args:
        graph1: First graph
        graph2: Second graph
        
    Returns:
        Dictionary of similarity metrics
    """
    metrics = {}
    
    # Basic properties comparison
    metrics["n_nodes_ratio"] = graph2.number_of_nodes() / max(1, graph1.number_of_nodes())
    metrics["n_edges_ratio"] = graph2.number_of_edges() / max(1, graph1.number_of_edges())
    metrics["density_ratio"] = nx.density(graph2) / max(0.001, nx.density(graph1))
    
    # Degree distribution similarity
    deg1 = np.array([d for _, d in graph1.degree()])
    deg2 = np.array([d for _, d in graph2.degree()])
    
    # Convert to probability distributions
    bins = np.linspace(0, max(deg1.max(), deg2.max()), 20)
    hist1, _ = np.histogram(deg1, bins=bins, density=True)
    hist2, _ = np.histogram(deg2, bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Calculate KL divergence
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    metrics["degree_kl_div"] = kl_div
    
    # Calculate Jensen-Shannon divergence (symmetric)
    m = (hist1 + hist2) / 2
    js_div = (np.sum(hist1 * np.log(hist1 / m)) + np.sum(hist2 * np.log(hist2 / m))) / 2
    metrics["degree_js_div"] = js_div
    
    # Calculate clustering coefficient similarity
    try:
        cc1 = nx.average_clustering(graph1)
        cc2 = nx.average_clustering(graph2)
        metrics["clustering_ratio"] = cc2 / max(0.001, cc1)
    except:
        metrics["clustering_ratio"] = 1.0  # Default if calculation fails
    
    # Calculate assortativity similarity
    try:
        assort1 = nx.degree_assortativity_coefficient(graph1)
        assort2 = nx.degree_assortativity_coefficient(graph2)
        # Handle the case where assortativity can be negative
        if assort1 * assort2 >= 0:  # Same sign
            metrics["assortativity_ratio"] = abs(assort2) / max(0.001, abs(assort1))
        else:
            metrics["assortativity_ratio"] = 0.0  # Different signs
    except:
        metrics["assortativity_ratio"] = 1.0  # Default if calculation fails
    
    return metrics


def community_structure_similarity(
    communities1: List[List[int]],
    communities2: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate similarity between two community structures.
    
    Args:
        communities1: First community structure
        communities2: Second community structure
        
    Returns:
        Dictionary of similarity metrics
    """
    metrics = {}
    
    # Number of communities
    metrics["n_communities_ratio"] = len(communities2) / max(1, len(communities1))
    
    # Community size distributions
    sizes1 = np.array([len(c) for c in communities1])
    sizes2 = np.array([len(c) for c in communities2])
    
    # Mean community size ratio
    metrics["mean_size_ratio"] = sizes2.mean() / max(1, sizes1.mean())
    
    # Community size distribution similarity
    bins = np.linspace(0, max(sizes1.max(), sizes2.max()), 10)
    hist1, _ = np.histogram(sizes1, bins=bins, density=True)
    hist2, _ = np.histogram(sizes2, bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Calculate Jensen-Shannon divergence (symmetric)
    m = (hist1 + hist2) / 2
    js_div = (np.sum(hist1 * np.log(hist1 / m)) + np.sum(hist2 * np.log(hist2 / m))) / 2
    metrics["size_js_div"] = js_div
    
    # Overlap distribution
    overlap_counts1 = []
    all_nodes1 = set()
    for comm in communities1:
        all_nodes1.update(comm)
    
    node_counts1 = {}
    for node in all_nodes1:
        node_counts1[node] = 0
        for comm in communities1:
            if node in comm:
                node_counts1[node] += 1
    
    overlap_counts1 = list(node_counts1.values())
    
    overlap_counts2 = []
    all_nodes2 = set()
    for comm in communities2:
        all_nodes2.update(comm)
    
    node_counts2 = {}
    for node in all_nodes2:
        node_counts2[node] = 0
        for comm in communities2:
            if node in comm:
                node_counts2[node] += 1
    
    overlap_counts2 = list(node_counts2.values())
    
    # Average number of communities per node
    metrics["mean_overlap_ratio"] = np.mean(overlap_counts2) / max(1, np.mean(overlap_counts1))
    
    # Overlap distribution similarity
    max_overlap = max(max(overlap_counts1), max(overlap_counts2))
    bins = np.arange(0.5, max_overlap + 1.5)  # Bins centered on integers
    hist1, _ = np.histogram(overlap_counts1, bins=bins, density=True)
    hist2, _ = np.histogram(overlap_counts2, bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Calculate Jensen-Shannon divergence for overlap
    m = (hist1 + hist2) / 2
    js_div = (np.sum(hist1 * np.log(hist1 / m)) + np.sum(hist2 * np.log(hist2 / m))) / 2
    metrics["overlap_js_div"] = js_div
    
    return metrics