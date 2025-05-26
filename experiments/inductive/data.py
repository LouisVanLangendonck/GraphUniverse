"""
Data preparation utilities for inductive graph learning experiments.

This module provides functions for preparing graph families for inductive learning,
where models are trained on some graphs and tested on others.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx

from mmsb.model import GraphSample
from mmsb.graph_family import GraphFamilyGenerator
from experiments.inductive.config import InductiveExperimentConfig
from mmsb.feature_regimes import graphsample_to_pyg


def split_graphs(
    family_graphs: List[GraphSample],
    config: InductiveExperimentConfig
) -> Dict[str, List[GraphSample]]:
    """
    Split graph family into train/val/test sets.
    
    Args:
        family_graphs: List of GraphSample objects
        config: Inductive experiment configuration
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing graph lists
    """
    n_graphs = len(family_graphs)
    n_train, n_val, n_test = config.get_graph_splits()
    
    print(f"Splitting {n_graphs} graphs: {n_train} train, {n_val} val, {n_test} test")
    
    # Shuffle graphs
    np.random.seed(config.seed)
    indices = np.random.permutation(n_graphs)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create splits
    splits = {
        'train': [family_graphs[i] for i in train_indices],
        'val': [family_graphs[i] for i in val_indices],
        'test': [family_graphs[i] for i in test_indices]
    }
    
    return splits


def prepare_inductive_data(
    family_graphs: List[GraphSample],
    config: InductiveExperimentConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare graph family data for inductive learning.
    
    Args:
        family_graphs: List of GraphSample objects from the same family
        config: Inductive experiment configuration
        
    Returns:
        Dictionary containing data for each task, organized by split
    """
    # Calculate actual split sizes based on generated graphs
    n_graphs = len(family_graphs)
    n_train = int(n_graphs * config.train_graph_ratio)
    n_val = int(n_graphs * config.val_graph_ratio)
    n_test = n_graphs - n_train - n_val  # Remainder goes to test
    
    print(f"\nSplitting {n_graphs} graphs: {n_train} train, {n_val} val, {n_test} test")
    
    # Split graphs
    np.random.seed(config.seed)
    indices = np.random.permutation(n_graphs)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    graph_split_dict = {
        'train': [family_graphs[i] for i in train_indices],
        'val': [family_graphs[i] for i in val_indices],
        'test': [family_graphs[i] for i in test_indices]
    }
    
    # Initialize results dictionary
    inductive_data = {}
    
    # Process each task
    for task in config.tasks:
        print(f"\nPreparing inductive data for task: {task}")
        
        task_data = {}
        
        # Process each split
        for split_name, graphs_in_split in graph_split_dict.items():
            n_graphs_in_split = len(graphs_in_split)
            print(f"  Processing {split_name} split with {n_graphs_in_split} graphs")
            
            if n_graphs_in_split == 0:
                print(f"  Warning: No graphs in {split_name} split, skipping")
                task_data[split_name] = {
                    'graphs': [],
                    'n_graphs': 0,
                    'batch_size': 1  # Minimum batch size
                }
                continue
            
            # Convert graphs to PyTorch Geometric format
            pyg_graphs = []
            for graph_sample in graphs_in_split:
                # Convert to PyG format
                pyg_data = graphsample_to_pyg(graph_sample)
                
                # Add task-specific labels
                if task == "community":
                    # Use community labels as-is
                    pyg_data.y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
                    
                elif task == "k_hop_community_counts":
                    # Compute k-hop community counts
                    community_counts = compute_khop_community_counts_batch(
                        graph_sample.graph,
                        graph_sample.community_labels,
                        config.khop_community_counts_k
                    )
                    pyg_data.y = community_counts
                
                # Add graph-level metadata
                pyg_data.graph_id = len(pyg_graphs)  # Assign unique ID within split
                pyg_data.n_communities = len(np.unique(graph_sample.community_labels))
                
                pyg_graphs.append(pyg_data)
            
            # Calculate batch size for this split
            if split_name == 'train':
                batch_size = min(n_graphs_in_split, config.batch_size)
            else:
                batch_size = n_graphs_in_split  # Use all graphs for val/test
            
            # Store data for this split
            task_data[split_name] = {
                'graphs': pyg_graphs,
                'n_graphs': n_graphs_in_split,
                'batch_size': max(1, batch_size)  # Ensure batch size is at least 1
            }
        
        # Add task-specific metadata
        is_regression = config.is_regression.get(task, False)
        
        # Calculate output dimension
        if is_regression:
            # For regression, use number of communities (for counting tasks)
            sample_graph = family_graphs[0]
            output_dim = len(np.unique(sample_graph.community_labels))
        else:
            # For classification, use number of unique labels across all graphs
            all_labels = []
            for graph in family_graphs:
                all_labels.extend(graph.community_labels.tolist())
            output_dim = len(np.unique(all_labels))
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': family_graphs[0].features.shape[1] if family_graphs[0].features is not None else 0
        }
        
        inductive_data[task] = task_data
    
    return inductive_data


def compute_khop_community_counts_batch(
    graph: nx.Graph,
    community_labels: np.ndarray,
    k: int
) -> torch.Tensor:
    """
    Compute k-hop community counts for a single graph.
    
    Args:
        graph: NetworkX graph
        community_labels: Array of community labels for each node
        k: Number of hops to consider
        
    Returns:
        Tensor of shape [num_nodes, num_communities] containing community counts
    """
    num_nodes = len(graph)
    num_communities = len(np.unique(community_labels))
    
    # Initialize count matrix
    community_counts = torch.zeros((num_nodes, num_communities), dtype=torch.float)
    
    # For each node, compute k-hop neighborhood and count communities
    for node in range(num_nodes):
        # Get nodes at exactly k hops away
        khop_nodes = set(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        # Remove nodes that are closer than k hops
        if k > 1:
            closer_nodes = set(nx.single_source_shortest_path_length(graph, node, cutoff=k-1).keys())
            khop_nodes = khop_nodes - closer_nodes
        
        # Count communities in k-hop neighborhood
        for neighbor in khop_nodes:
            if neighbor < len(community_labels):  # Safety check
                community = community_labels[neighbor]
                if 0 <= community < num_communities:  # Safety check
                    community_counts[node, community] += 1
    
    return community_counts


def create_inductive_dataloaders(
    inductive_data: Dict[str, Dict[str, Any]],
    config: InductiveExperimentConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Create dataloaders for inductive learning.
    
    Args:
        inductive_data: Prepared inductive data
        config: Experiment configuration
        
    Returns:
        Dictionary containing dataloaders for each task and split
    """
    from torch_geometric.loader import DataLoader
    
    dataloaders = {}
    
    for task, task_data in inductive_data.items():
        if task == 'metadata':
            continue
            
        task_loaders = {}
        
        for split_name, split_data in task_data.items():
            if split_name == 'metadata':
                continue
                
            # Create dataloader
            batch_size = split_data['batch_size']
            shuffle = (split_name == 'train')
            
            loader = DataLoader(
                split_data['graphs'],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            task_loaders[split_name] = loader
        
        # Add metadata
        task_loaders['metadata'] = task_data['metadata']
        
        dataloaders[task] = task_loaders
    
    return dataloaders


def prepare_mixed_inductive_data(
    family_graphs: List[GraphSample],
    config: InductiveExperimentConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare data for mixed inductive learning where some graphs may overlap
    between splits but focus is on generalization.
    
    Args:
        family_graphs: List of GraphSample objects
        config: Inductive experiment configuration
        
    Returns:
        Dictionary containing prepared data
    """
    # For mixed mode, we can allow some overlap but ensure test graphs
    # have different characteristics
    
    # First, analyze graph characteristics
    graph_characteristics = []
    for i, graph in enumerate(family_graphs):
        char = {
            'graph_id': i,
            'n_nodes': graph.n_nodes,
            'n_edges': graph.graph.number_of_edges(),
            'density': graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2) if graph.n_nodes > 1 else 0,
            'avg_degree': sum(dict(graph.graph.degree()).values()) / graph.n_nodes if graph.n_nodes > 0 else 0,
            'n_communities': len(np.unique(graph.community_labels))
        }
        graph_characteristics.append(char)
    
    # Sort by characteristics to ensure diverse test set
    sorted_chars = sorted(graph_characteristics, key=lambda x: (x['density'], x['avg_degree']))
    
    # Split ensuring diversity
    n_train, n_val, n_test = config.get_graph_splits()
    
    # Interleaved sampling for diversity
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, char in enumerate(sorted_chars):
        if i % 3 == 0 and len(test_indices) < n_test:
            test_indices.append(char['graph_id'])
        elif i % 3 == 1 and len(val_indices) < n_val:
            val_indices.append(char['graph_id'])
        else:
            train_indices.append(char['graph_id'])
    
    # Adjust if we don't have enough in each split
    while len(train_indices) < n_train and (len(val_indices) > n_val or len(test_indices) > n_test):
        if len(val_indices) > n_val:
            train_indices.append(val_indices.pop())
        elif len(test_indices) > n_test:
            train_indices.append(test_indices.pop())
    
    # Create splits
    splits = {
        'train': [family_graphs[i] for i in train_indices],
        'val': [family_graphs[i] for i in val_indices],
        'test': [family_graphs[i] for i in test_indices]
    }
    
    print(f"Mixed inductive split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    
    # Use the same data preparation as graph_level mode
    return prepare_inductive_data_from_splits(splits, config)


def prepare_inductive_data_from_splits(
    graph_splits: Dict[str, List[GraphSample]],
    config: InductiveExperimentConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare inductive data from pre-computed graph splits.
    
    Args:
        graph_splits: Dictionary with train/val/test graph lists
        config: Experiment configuration
        
    Returns:
        Prepared inductive data
    """
    # Get all graphs for metadata calculation
    all_graphs = []
    for split_graphs in graph_splits.values():
        all_graphs.extend(split_graphs)
    
    # Initialize results
    inductive_data = {}
    
    # Process each task
    for task in config.tasks:
        task_data = {}
        
        # Process each split
        for split_name, split_graphs in graph_splits.items():
            # Convert to PyG format with task-specific labels
            pyg_graphs = []
            for graph_sample in split_graphs:
                pyg_data = graphsample_to_pyg(graph_sample)
                
                if task == "community":
                    pyg_data.y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
                elif task == "k_hop_community_counts":
                    community_counts = compute_khop_community_counts_batch(
                        graph_sample.graph,
                        graph_sample.community_labels,
                        config.khop_community_counts_k
                    )
                    pyg_data.y = community_counts
                
                pyg_data.graph_id = len(pyg_graphs)
                pyg_data.n_communities = len(np.unique(graph_sample.community_labels))
                pyg_graphs.append(pyg_data)
            
            task_data[split_name] = {
                'graphs': pyg_graphs,
                'n_graphs': len(pyg_graphs),
                'batch_size': min(len(pyg_graphs), config.batch_size) if split_name == 'train' else len(pyg_graphs)
            }
        
        # Add metadata
        is_regression = config.is_regression.get(task, False)
        if is_regression:
            output_dim = len(np.unique(all_graphs[0].community_labels))
        else:
            all_labels = []
            for graph in all_graphs:
                all_labels.extend(graph.community_labels.tolist())
            output_dim = len(np.unique(all_labels))
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': all_graphs[0].features.shape[1] if all_graphs[0].features is not None else 0
        }
        
        inductive_data[task] = task_data
    
    return inductive_data


def analyze_graph_family_properties(
    family_graphs: List[GraphSample]
) -> Dict[str, Any]:
    """
    Analyze properties of a graph family for inductive learning insights.
    
    Args:
        family_graphs: List of GraphSample objects
        
    Returns:
        Dictionary containing family analysis
    """
    properties = {
        'n_graphs': len(family_graphs),
        'node_counts': [],
        'edge_counts': [],
        'densities': [],
        'avg_degrees': [],
        'clustering_coefficients': [],
        'community_counts': []
    }
    
    for graph in family_graphs:
        properties['node_counts'].append(graph.n_nodes)
        properties['edge_counts'].append(graph.graph.number_of_edges())
        
        if graph.n_nodes > 1:
            density = graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2)
            properties['densities'].append(density)
        else:
            properties['densities'].append(0.0)
        
        if graph.n_nodes > 0:
            avg_degree = sum(dict(graph.graph.degree()).values()) / graph.n_nodes
            properties['avg_degrees'].append(avg_degree)
        else:
            properties['avg_degrees'].append(0.0)
        
        try:
            clustering = nx.average_clustering(graph.graph)
            properties['clustering_coefficients'].append(clustering)
        except:
            properties['clustering_coefficients'].append(0.0)
        
        properties['community_counts'].append(len(np.unique(graph.community_labels)))
    
    # Calculate statistics and convert to native Python types
    for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts']:
        values = properties[key]
        properties[f'{key}_mean'] = float(np.mean(values))
        properties[f'{key}_std'] = float(np.std(values))
        properties[f'{key}_min'] = float(np.min(values))
        properties[f'{key}_max'] = float(np.max(values))
    
    return properties