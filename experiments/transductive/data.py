"""
Clean data preparation utilities for transductive graph learning.
Based on inductive data preparation but adapted for single-graph transductive learning.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
from collections import defaultdict

from mmsb.model import GraphSample, GraphUniverse
from mmsb.feature_regimes import graphsample_to_pyg
from utils.metapath_analysis import MetapathAnalyzer, UniverseMetapathSelector


def prepare_transductive_data(
    graph_sample: GraphSample,
    config
) -> Dict[str, Dict[str, Any]]:
    """
    Central function to prepare single graph data for transductive learning.
    
    Args:
        graph_sample: Single GraphSample object  
        config: Experiment configuration
        
    Returns:
        Dictionary containing data for each task with train/val/test node splits
    """
    # Get universe from graph sample
    universe = graph_sample.universe
    if not universe:
        raise ValueError("No universe found in graph sample")
    
    # Get universe K
    universe_K = universe.K
    print(f"\nUsing universe K: {universe_K}")
    
    # Calculate split sizes for nodes
    n_nodes = graph_sample.n_nodes
    n_train = int(n_nodes * config.train_ratio)
    n_val = int(n_nodes * config.val_ratio)
    n_test = n_nodes - n_train - n_val
    
    print(f"\nSplitting {n_nodes} nodes: {n_train} train, {n_val} val, {n_test} test")
    
    # Split nodes randomly
    np.random.seed(config.seed)
    indices = np.random.permutation(n_nodes)
    
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    # Convert to PyG format
    pyg_data = graphsample_to_pyg(graph_sample)
    
    # Add universe K to graph
    pyg_data.universe_K = universe_K
    
    # Initialize results dictionary
    transductive_data = {}
    
    # Generate metapath tasks if enabled
    metapath_data = None
    if hasattr(config, 'enable_metapath_tasks') and config.enable_metapath_tasks:
        print("\nGenerating metapath tasks...")
        metapath_data = generate_transductive_metapath_tasks(
            graph_sample=graph_sample,
            universe=universe,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            k_values=getattr(config, 'metapath_k_values', [3, 4, 5]),
            require_loop=getattr(config, 'metapath_require_loop', False),
            degree_weight=getattr(config, 'metapath_degree_weight', 0.3),
            max_community_participation=getattr(config, 'max_community_participation', 1.0)
        )
    
    # Process each task
    for task in config.tasks:
        print(f"\nPreparing transductive data for task: {task}")
        
        # Create base data structure
        task_data = {
            'features': pyg_data.x,
            'edge_index': pyg_data.edge_index,
            'train_idx': torch.tensor(train_indices, dtype=torch.long),
            'val_idx': torch.tensor(val_indices, dtype=torch.long),
            'test_idx': torch.tensor(test_indices, dtype=torch.long),
            'num_nodes': n_nodes
        }
        
        # Generate task-specific labels
        if task == "community":
            # Standard community prediction - use universe-indexed labels
            task_data['labels'] = torch.tensor(graph_sample.community_labels_universe_level, dtype=torch.long)
            
        elif task == "k_hop_community_counts":
            # K-hop community counting - already universe-indexed
            community_counts = compute_khop_community_counts_universe_indexed(
                graph_sample.graph,
                graph_sample.community_labels,
                graph_sample.community_id_mapping,
                universe_K,
                getattr(config, 'khop_community_counts_k', 2)
            )
            task_data['labels'] = community_counts
            
        elif task == "metapath" and metapath_data:
            # Metapath task
            task_info = list(metapath_data['tasks'].values())[0]
            metapath_labels = task_info['labels']
            if metapath_labels is not None:
                binary_labels = (metapath_labels > 0).astype(int)
                task_data['labels'] = torch.tensor(binary_labels, dtype=torch.long)
            else:
                continue
        
        # Add task-specific metadata
        is_regression = config.is_regression.get(task, False)
        
        # Calculate output dimension based on task type
        if task == "community":
            # For community prediction, use universe K
            output_dim = universe_K
            
        elif task == "k_hop_community_counts":
            # For k-hop counting, use universe K
            output_dim = universe_K
            
        elif task == "metapath" and metapath_data:
            # For metapath tasks, use binary classification
            output_dim = 2
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': pyg_data.x.shape[1],
            'task_type': task,
            'universe_K': universe_K,
            'num_classes': output_dim  # For compatibility
        }
        
        # Add task-specific metadata
        if task == "k_hop_community_counts":
            task_data['metadata'].update({
                'k_value': getattr(config, 'khop_community_counts_k', 2),
                'universe_K': universe_K
            })
            
        elif task == "metapath" and metapath_data:
            task_info = list(metapath_data['tasks'].values())[0]
            task_data['metadata'].update({
                'metapath': task_info['metapath'],
                'universe_probability': task_info['universe_probability'],
                'coverage': task_info['coverage'],
                'avg_positive_rate': task_info['avg_positive_rate'],
                'is_loop_task': task_info['is_loop_task']
            })
        
        transductive_data[task] = task_data
    
    # Add metapath analysis if available
    if metapath_data:
        transductive_data['metapath_analysis'] = {
            'evaluation_results': metapath_data['evaluation_results'],
            'candidate_analysis': metapath_data['candidate_analysis'],
            'universe_info': metapath_data['universe_info']
        }
    
    return transductive_data


def compute_khop_community_counts_universe_indexed(
    graph: nx.Graph,
    community_labels: np.ndarray,
    universe_communities: Dict[int, int],
    universe_K: int,
    k: int
) -> torch.Tensor:
    """
    Compute k-hop community counts with universe indexing.
    
    Args:
        graph: NetworkX graph
        community_labels: Node community labels
        universe_communities: Mapping from local to universe community indices
        universe_K: Number of communities in universe
        k: Number of hops
        
    Returns:
        Tensor of shape (n_nodes, universe_K) containing k-hop community counts
    """
    n_nodes = graph.number_of_nodes()
    counts = np.zeros((n_nodes, universe_K))
    
    # For each node
    for node in range(n_nodes):
        # Get k-hop neighborhood
        neighbors = set([node])
        for _ in range(k):
            new_neighbors = set()
            for n in neighbors:
                new_neighbors.update(graph.neighbors(n))
            neighbors.update(new_neighbors)
        
        # Count communities in neighborhood
        for neighbor in neighbors:
            local_comm = community_labels[neighbor]
            if local_comm in universe_communities:
                universe_comm = universe_communities[local_comm]
                counts[node, universe_comm] += 1
            else:
                raise ValueError(f"Community {local_comm} not found in universe communities")
    
    return torch.tensor(counts, dtype=torch.float)


def generate_transductive_metapath_tasks(
    graph_sample: GraphSample,
    universe: 'GraphUniverse',
    train_indices: List[int],
    val_indices: List[int], 
    test_indices: List[int],
    k_values: List[int] = [3, 4, 5],
    require_loop: bool = False,
    degree_weight: float = 0.3,
    max_community_participation: float = 1.0
) -> Dict[str, Any]:
    """
    Generate metapath tasks for transductive learning on a single graph.
    
    Args:
        graph_sample: Single graph sample
        universe: Graph universe
        train_indices: Training node indices
        val_indices: Validation node indices
        test_indices: Test node indices
        k_values: List of k values for metapath length
        require_loop: Whether to require loop in metapath
        degree_weight: Weight for degree-based scoring
        max_community_participation: Maximum community participation ratio
        
    Returns:
        Dictionary containing metapath task data
    """
    # This would implement the metapath generation logic for single graph
    # For now, return empty dict as placeholder
    return {}


def analyze_graph_properties(graph_sample: GraphSample) -> Dict[str, Any]:
    """
    Analyze properties of a single graph for transductive learning insights.
    
    Args:
        graph_sample: GraphSample object
        
    Returns:
        Dictionary containing graph analysis
    """
    properties = {
        'n_nodes': graph_sample.n_nodes,
        'n_edges': graph_sample.graph.number_of_edges(),
        'n_communities': len(np.unique(graph_sample.community_labels)),
        'density': 0.0,
        'avg_degree': 0.0,
        'clustering_coefficient': 0.0
    }
    
    if graph_sample.n_nodes > 1:
        properties['density'] = graph_sample.graph.number_of_edges() / (graph_sample.n_nodes * (graph_sample.n_nodes - 1) / 2)
    
    if graph_sample.n_nodes > 0:
        properties['avg_degree'] = sum(dict(graph_sample.graph.degree()).values()) / graph_sample.n_nodes
    
    try:
        properties['clustering_coefficient'] = nx.average_clustering(graph_sample.graph)
    except:
        properties['clustering_coefficient'] = 0.0
    
    # Add signal metrics if available
    if hasattr(graph_sample, 'degree_signal'):
        properties['degree_signal'] = graph_sample.degree_signal
    if hasattr(graph_sample, 'structure_signal'):
        properties['structure_signal'] = graph_sample.structure_signal
    if hasattr(graph_sample, 'feature_signal'):
        properties['feature_signal'] = graph_sample.feature_signal
    
    return properties


def validate_transductive_data(
    transductive_data: Dict[str, Dict[str, Any]], 
    config
) -> Dict[str, Any]:
    """
    Validate that the transductive data is properly prepared.
    
    Args:
        transductive_data: Prepared transductive data
        config: Experiment configuration
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'task_info': {}
    }
    
    for task, task_data in transductive_data.items():
        if task in ['metapath_analysis']:
            continue
            
        task_info = {
            'has_features': 'features' in task_data,
            'has_edges': 'edge_index' in task_data,
            'has_labels': 'labels' in task_data,
            'has_splits': all(key in task_data for key in ['train_idx', 'val_idx', 'test_idx'])
        }
        
        # Check data consistency
        if task_info['has_features'] and task_info['has_labels']:
            n_nodes_features = task_data['features'].shape[0]
            if task_data['labels'].dim() == 1:
                n_nodes_labels = task_data['labels'].shape[0]
            else:
                n_nodes_labels = task_data['labels'].shape[0]
                
            if n_nodes_features != n_nodes_labels:
                validation_results['valid'] = False
                validation_results['issues'].append(
                    f"Task {task}: Feature and label node counts don't match "
                    f"({n_nodes_features} vs {n_nodes_labels})"
                )
        
        # Check split sizes
        if task_info['has_splits']:
            train_size = len(task_data['train_idx'])
            val_size = len(task_data['val_idx'])
            test_size = len(task_data['test_idx'])
            total_size = train_size + val_size + test_size
            
            expected_total = task_data.get('num_nodes', config.num_nodes)
            
            if total_size != expected_total:
                validation_results['valid'] = False
                validation_results['issues'].append(
                    f"Task {task}: Split sizes don't sum to total nodes "
                    f"({total_size} vs {expected_total})"
                )
        
        validation_results['task_info'][task] = task_info
    
    return validation_results