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
    n_train, n_val, n_test = config.get_splits()
    np.random.seed(config.seed)
    indices = np.random.permutation(n_nodes)
    # Constant test split
    test_indices = indices[n_train + n_val:].tolist()
    # Convert to PyG format
    pyg_data = graphsample_to_pyg(graph_sample)
    # Add PE computation
    pe_types = getattr(config, 'pe_types', ['laplacian', 'degree', 'rwse'])
    max_pe_dim = getattr(config, 'max_pe_dim', 8)
    from experiments.inductive.data import PositionalEncodingComputer
    pe_computer = PositionalEncodingComputer(max_pe_dim=max_pe_dim, pe_types=pe_types)
    pe_dict = pe_computer.compute_all_pe(pyg_data.edge_index, pyg_data.x.size(0))
    for pe_name, pe_tensor in pe_dict.items():
        setattr(pyg_data, pe_name, pe_tensor)
    pyg_data.universe_K = universe_K
    input_dim = pyg_data.x.shape[1]
    # Prepare splits
    splits = []
    community_labels = np.array(graph_sample.community_labels_universe_level)
    for rep in range(config.k_fold):
        split_seed = config.seed + rep
        np.random.seed(split_seed)
        val_indices = []
        test_indices = []
        per_comm_val = n_val // universe_K
        per_comm_test = n_test // universe_K
        all_indices = set(range(n_nodes))
        used_indices = set()
        for k in range(universe_K):
            comm_indices = np.where(community_labels == k)[0]
            comm_indices = np.random.permutation(comm_indices)
            # Select for val and test
            val_k = comm_indices[:per_comm_val]
            test_k = comm_indices[per_comm_val:per_comm_val+per_comm_test]
            val_indices.extend(val_k.tolist())
            test_indices.extend(test_k.tolist())
            used_indices.update(val_k.tolist())
            used_indices.update(test_k.tolist())
        # The rest go to train
        train_indices = list(all_indices - used_indices)
        # Shuffle train indices for randomness
        train_indices = np.random.permutation(train_indices).tolist()
        split = {
            'pyg_graph': pyg_data,
            'labels': None,  # Set per task below
            'train_idx': torch.tensor(train_indices, dtype=torch.long),
            'val_idx': torch.tensor(val_indices, dtype=torch.long),
            'test_idx': torch.tensor(test_indices, dtype=torch.long),
            'input_dim': input_dim,
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

def prepare_transductive_data_gpu_resident(
    graph_sample: GraphSample,
    config,
    device: torch.device
) -> Dict[str, Any]:
    """
    Prepare transductive data with all tensors pre-loaded to GPU to avoid repeated CPU-GPU transfers.
    
    Args:
        graph_sample: GraphSample object
        config: Experiment configuration
        device: Device to load data onto
        
    Returns:
        Dictionary with GPU-resident data for transductive learning
    """
    print(f"🚀 Pre-loading transductive data to {device}...")
    
    universe = graph_sample.universe
    if not universe:
        raise ValueError("No universe found in graph sample")
    universe_K = universe.K
    n_nodes = graph_sample.n_nodes
    n_train, n_val, n_test = config.get_splits()
    
    # Convert to PyG format
    pyg_data = graphsample_to_pyg(graph_sample)
    
    # Add PE computation
    pe_types = getattr(config, 'pe_types', ['laplacian', 'degree', 'rwse'])
    max_pe_dim = getattr(config, 'max_pe_dim', 8)
    from experiments.inductive.data import PositionalEncodingComputer
    pe_computer = PositionalEncodingComputer(max_pe_dim=max_pe_dim, pe_types=pe_types)
    pe_dict = pe_computer.compute_all_pe(pyg_data.edge_index, pyg_data.x.size(0))
    for pe_name, pe_tensor in pe_dict.items():
        setattr(pyg_data, pe_name, pe_tensor)
    
    pyg_data.universe_K = universe_K
    
    # Move the ENTIRE PyG graph object to GPU at once
    pyg_data = pyg_data.to(device)
    
    # Extract tensors from the GPU-resident PyG graph
    input_dim = pyg_data.x.shape[1]
    
    # Prepare splits
    splits = []
    community_labels = np.array(graph_sample.community_labels_universe_level)
    
    for rep in range(config.k_fold):
        split_seed = config.seed + rep
        np.random.seed(split_seed)
        val_indices = []
        test_indices = []
        per_comm_val = n_val // universe_K
        per_comm_test = n_test // universe_K
        all_indices = set(range(n_nodes))
        used_indices = set()
        
        for k in range(universe_K):
            comm_indices = np.where(community_labels == k)[0]
            comm_indices = np.random.permutation(comm_indices)
            # Select for val and test
            val_k = comm_indices[:per_comm_val]
            test_k = comm_indices[per_comm_val:per_comm_val+per_comm_test]
            val_indices.extend(val_k.tolist())
            test_indices.extend(test_k.tolist())
            used_indices.update(val_k.tolist())
            used_indices.update(test_k.tolist())
        
        # The rest go to train
        train_indices = list(all_indices - used_indices)
        # Shuffle train indices for randomness
        train_indices = np.random.permutation(train_indices).tolist()
        
        # Move indices to GPU
        train_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
        val_idx = torch.tensor(val_indices, dtype=torch.long, device=device)
        test_idx = torch.tensor(test_indices, dtype=torch.long, device=device)
        
        split = {
            'pyg_graph': pyg_data,  # Already on GPU
            'labels': None,  # Set per task below
            'train_idx': train_idx,  # Already on GPU
            'val_idx': val_idx,  # Already on GPU
            'test_idx': test_idx,  # Already on GPU
            'input_dim': input_dim,
            'num_nodes': n_nodes,
            'metadata': {},
        }
        splits.append(split)
    
    # Assign labels and metadata per task (for each split)
    for split in splits:
        for task in config.tasks:
            if task == "community":
                # Move labels to GPU once
                labels = torch.tensor(graph_sample.community_labels_universe_level, dtype=torch.long, device=device)
                split['labels'] = labels
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
                # Move to GPU
                split['labels'] = community_counts.to(device)
                output_dim = universe_K
                is_regression = config.is_regression.get(task, False)
            else:
                continue
            
            split['metadata'] = {
                'is_regression': is_regression,
                'output_dim': output_dim,
                'input_dim': input_dim,
                'task_type': task,
                'universe_K': universe_K,
                'num_classes': output_dim
            }
            
            if task == "k_hop_community_counts":
                split['metadata'].update({
                    'k_value': getattr(config, 'khop_community_counts_k', 2),
                    'universe_K': universe_K
                })
    
    print(f"✅ All transductive data loaded to {device}")
    return {'splits': splits}

def verify_transductive_gpu_data(task_data: Dict[str, Any], device: torch.device) -> bool:
    """
    Verify that all data in transductive task_data is actually on the specified device.
    
    Args:
        task_data: GPU-resident task data to verify
        device: Expected device
        
    Returns:
        True if all data is on the correct device, False otherwise
    """
    print(f"🔍 Verifying transductive data is on {device}...")
    
    # Check main tensors
    if 'features' in task_data and task_data['features'].device != device:
        print(f"❌ features tensor on {task_data['features'].device}, expected {device}")
        return False
    
    if 'edge_index' in task_data and task_data['edge_index'].device != device:
        print(f"❌ edge_index tensor on {task_data['edge_index'].device}, expected {device}")
        return False
    
    if 'labels' in task_data and task_data['labels'].device != device:
        print(f"❌ labels tensor on {task_data['labels'].device}, expected {device}")
        return False
    
    if 'train_idx' in task_data and task_data['train_idx'].device != device:
        print(f"❌ train_idx tensor on {task_data['train_idx'].device}, expected {device}")
        return False
    
    if 'val_idx' in task_data and task_data['val_idx'].device != device:
        print(f"❌ val_idx tensor on {task_data['val_idx'].device}, expected {device}")
        return False
    
    if 'test_idx' in task_data and task_data['test_idx'].device != device:
        print(f"❌ test_idx tensor on {task_data['test_idx'].device}, expected {device}")
        return False
    
    # Check PyG graph tensors - this is the key fix
    if 'pyg_graph' in task_data:
        pyg_graph = task_data['pyg_graph']
        
        # Check that the PyG graph object itself is on the correct device
        if hasattr(pyg_graph, 'x') and pyg_graph.x.device != device:
            print(f"❌ pyg_graph.x tensor on {pyg_graph.x.device}, expected {device}")
            return False
        
        if hasattr(pyg_graph, 'edge_index') and pyg_graph.edge_index.device != device:
            print(f"❌ pyg_graph.edge_index tensor on {pyg_graph.edge_index.device}, expected {device}")
            return False
        
        # Check PE tensors
        for attr_name in dir(pyg_graph):
            if attr_name.endswith('_pe'):
                pe_tensor = getattr(pyg_graph, attr_name)
                if hasattr(pe_tensor, 'device') and pe_tensor.device != device:
                    print(f"❌ pyg_graph.{attr_name} tensor on {pe_tensor.device}, expected {device}")
                    return False
    
    print(f"✅ All transductive data verified to be on {device}")
    return True

def cleanup_transductive_gpu_data(task_data: Dict[str, Any], device: torch.device):
    """
    Clean up GPU memory by removing all data from GPU.
    
    Args:
        task_data: GPU-resident task data to clean up
        device: Device to clean up
    """
    print(f"🧹 Cleaning up transductive GPU memory on {device}...")
    
    # Move main tensors to CPU
    if 'features' in task_data:
        task_data['features'] = task_data['features'].cpu()
    
    if 'edge_index' in task_data:
        task_data['edge_index'] = task_data['edge_index'].cpu()
    
    if 'labels' in task_data:
        task_data['labels'] = task_data['labels'].cpu()
    
    if 'train_idx' in task_data:
        task_data['train_idx'] = task_data['train_idx'].cpu()
    
    if 'val_idx' in task_data:
        task_data['val_idx'] = task_data['val_idx'].cpu()
    
    if 'test_idx' in task_data:
        task_data['test_idx'] = task_data['test_idx'].cpu()
    
    # Clean up PyG graph - move the entire object to CPU
    if 'pyg_graph' in task_data:
        pyg_graph = task_data['pyg_graph']
        # Move the entire PyG graph object to CPU
        task_data['pyg_graph'] = pyg_graph.cpu()
    
    # Force GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print(f"✅ Transductive GPU memory cleaned up")

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