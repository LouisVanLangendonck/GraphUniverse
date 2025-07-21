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

from graph_universe.model import GraphSample, GraphUniverse
from graph_universe.feature_regimes import graphsample_to_pyg
from utils.metapath_analysis import MetapathAnalyzer, UniverseMetapathSelector
from experiments.inductive.data import PositionalEncodingComputer  # Import PE computer from inductive


def prepare_transductive_data(
    graph_sample: GraphSample,
    config
) -> Dict[str, Any]:
    """
    Central function to prepare multiple splits for transductive learning.
    Returns a dict with a list of splits, each with train/val/test indices, but test split is constant.
    """
    universe = graph_sample.universe
    if not universe:
        raise ValueError("No universe found in graph sample")
    universe_K = universe.K
    n_nodes = graph_sample.n_nodes
    n_train = int(n_nodes * config.train_ratio)
    n_val = int(n_nodes * config.val_ratio)
    n_test = n_nodes - n_train - n_val
    np.random.seed(config.seed)
    indices = np.random.permutation(n_nodes)
    # Constant test split
    test_indices = indices[n_train + n_val:].tolist()
    # Convert to PyG format
    pyg_data = graphsample_to_pyg(graph_sample)
    # Add PE computation
    pe_types = getattr(config, 'pe_types', ['laplacian', 'degree', 'rwse'])
    max_pe_dim = getattr(config, 'max_pe_dim', 16)
    from experiments.inductive.data import PositionalEncodingComputer
    pe_computer = PositionalEncodingComputer(max_pe_dim=max_pe_dim, pe_types=pe_types)
    pe_dict = pe_computer.compute_all_pe(pyg_data.edge_index, pyg_data.x.size(0))
    for pe_name, pe_tensor in pe_dict.items():
        setattr(pyg_data, pe_name, pe_tensor)
    pyg_data.universe_K = universe_K
    # Prepare splits
    splits = []
    for rep in range(config.n_repetitions):
        # Reseed for reproducibility
        split_seed = config.seed + rep
        np.random.seed(split_seed)
        indices = np.random.permutation(n_nodes)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        # test_indices is constant
        split = {
            'features': pyg_data.x,
            'edge_index': pyg_data.edge_index,
            'pyg_graph': pyg_data,
            'labels': None,  # Set per task below
            'train_idx': torch.tensor(train_indices, dtype=torch.long),
            'val_idx': torch.tensor(val_indices, dtype=torch.long),
            'test_idx': torch.tensor(test_indices, dtype=torch.long),
            'num_nodes': n_nodes,
            'metadata': {},
        }
        splits.append(split)
    # Assign labels and metadata per task (for each split)
    for split in splits:
        for task in config.tasks:
            if task == "community":
                split['labels'] = torch.tensor(graph_sample.community_labels_universe_level, dtype=torch.long)
                output_dim = universe_K
                is_regression = config.is_regression.get(task, False)
            elif task == "k_hop_community_counts":
                community_counts = compute_khop_community_counts_universe_indexed(
                    graph_sample.graph,
                    graph_sample.community_labels,
                    graph_sample.community_id_mapping,
                    universe_K,
                    getattr(config, 'khop_community_counts_k', 2)
                )
                split['labels'] = community_counts
                output_dim = universe_K
                is_regression = config.is_regression.get(task, False)
            else:
                continue
            split['metadata'] = {
                'is_regression': is_regression,
                'output_dim': output_dim,
                'input_dim': pyg_data.x.shape[1],
                'task_type': task,
                'universe_K': universe_K,
                'num_classes': output_dim
            }
            if task == "k_hop_community_counts":
                split['metadata'].update({
                    'k_value': getattr(config, 'khop_community_counts_k', 2),
                    'universe_K': universe_K
                })
    return {'splits': splits}

def compute_khop_community_counts_universe_indexed(
    graph: nx.Graph,
    community_labels: np.ndarray,
    universe_communities: Dict[int, int],
    universe_K: int,
    k: int
) -> torch.Tensor:
    """
    Compute k-hop community counts (only nodes at exactly k-hops) with universe indexing.
    """
    n_nodes = graph.number_of_nodes()
    counts = np.zeros((n_nodes, universe_K))
    
    for node in range(n_nodes):
        # Get nodes at exact distance k using single-source shortest path
        sp_lengths = nx.single_source_shortest_path_length(graph, node, cutoff=k)
        khop_nodes = [n for n, dist in sp_lengths.items() if dist == k]
        
        for neighbor in khop_nodes:
            local_comm = community_labels[neighbor]
            if local_comm in universe_communities:
                universe_comm = universe_communities[local_comm]
                counts[node, universe_comm] += 1
            else:
                raise ValueError(f"Community {local_comm} not in universe_communities")
    
    return torch.tensor(counts, dtype=torch.float)

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

# Add a function to re-split train/val/test indices for a given seed
def resplit_transductive_indices(num_nodes, train_ratio, val_ratio, test_ratio, seed):
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    n_test = num_nodes - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx