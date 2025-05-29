"""
Clean data preparation utilities for inductive graph learning.
Removes old parameters and focuses on DC-SBM and DCCC-SBM methods only.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
from collections import defaultdict
from datetime import datetime
import json
import pickle
import os
import time
from dataclasses import asdict
import pandas as pd

from mmsb.model import GraphSample, GraphUniverse
from mmsb.feature_regimes import graphsample_to_pyg
from utils.metapath_analysis import MetapathAnalyzer, UniverseMetapathSelector, FamilyMetapathEvaluator
from experiments.inductive.config import PreTrainingConfig

class GraphFamilyManager:
    """Manages graph family generation, saving, and loading for SSL experiments."""
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
        self.graph_family_dir = config.graph_family_dir
        os.makedirs(self.graph_family_dir, exist_ok=True)
    
    def generate_and_save_family(self, family_id: Optional[str] = None) -> Tuple[List, str]:
        """Generate graph family and save it for later use."""
        
        if family_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            family_id = f"family_{self.config.gnn_type}_{self.config.pretraining_task}_{timestamp}"
        
        print(f"\nGenerating graph family: {family_id}")
        print(f"Total graphs: {self.config.get_total_graphs()}")
        print(f"Pretraining graphs: {self.config.n_graphs}")
        print(f"Extra fine-tuning graphs: {self.config.n_extra_graphs_for_finetuning}")
        
        # Generate graph family using existing infrastructure
        family_graphs = self._generate_graph_family()
        
        # Save the family
        if self.config.save_graph_family:
            self._save_family(family_graphs, family_id)
        
        return family_graphs, family_id
    
    def _generate_graph_family(self) -> List:
        """Generate the graph family using existing infrastructure."""
        from mmsb.model import GraphUniverse
        from mmsb.graph_family import GraphFamilyGenerator
        
        # Create universe
        universe = GraphUniverse(
            K=self.config.universe_K,
            feature_dim=self.config.universe_feature_dim,
            edge_density=self.config.universe_edge_density,
            homophily=self.config.universe_homophily,
            seed=self.config.seed
        )
        
        # Create family generator
        family_generator = GraphFamilyGenerator(
            universe=universe,
            n_graphs=self.config.get_total_graphs(),  # Generate ALL graphs at once
            min_n_nodes=self.config.min_n_nodes,
            max_n_nodes=self.config.max_n_nodes,
            min_communities=self.config.min_communities,
            max_communities=self.config.max_communities,
            use_dccc_sbm=self.config.use_dccc_sbm,
            degree_distribution=self.config.degree_distribution,
            seed=self.config.seed
        )
        
        # Generate family
        print("Generating complete graph family...")
        start_time = time.time()
        family_graphs = family_generator.generate_family(show_progress=True)
        generation_time = time.time() - start_time
        
        print(f"Family generation completed in {generation_time:.2f} seconds")
        print(f"Generated {len(family_graphs)} graphs total")
        
        return family_graphs
    
    def _save_family(self, family_graphs: List, family_id: str) -> None:
        """Save graph family to disk."""
        family_dir = os.path.join(self.graph_family_dir, family_id)
        os.makedirs(family_dir, exist_ok=True)
        
        print(f"Saving graph family to: {family_dir}")
        
        # Save graphs
        graphs_file = os.path.join(family_dir, "graphs.pkl")
        with open(graphs_file, 'wb') as f:
            pickle.dump(family_graphs, f)
        
        # Save metadata
        n_actual_pretraining, n_warmup, n_finetuning = self.config.get_graph_splits()
        
        metadata = {
            'family_id': family_id,
            'creation_timestamp': datetime.now().isoformat(),
            'total_graphs': len(family_graphs),
            'n_actual_pretraining': n_actual_pretraining,
            'n_warmup': n_warmup,
            'n_finetuning': n_finetuning,
            'config': asdict(self.config),
            'graphs_file': graphs_file,
            'splits': {
                'pretraining': list(range(n_actual_pretraining)),
                'warmup': list(range(n_actual_pretraining, n_actual_pretraining + n_warmup)),
                'finetuning': list(range(n_actual_pretraining + n_warmup, len(family_graphs)))
            }
        }
        
        metadata_file = os.path.join(family_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Saved {len(family_graphs)} graphs")
        print(f"✓ Metadata saved to: {metadata_file}")
    
    def load_family(self, family_id: str) -> Tuple[List, Dict]:
        """Load saved graph family."""
        family_dir = os.path.join(self.graph_family_dir, family_id)
        
        if not os.path.exists(family_dir):
            raise FileNotFoundError(f"Graph family not found: {family_id}")
        
        # Load metadata
        metadata_file = os.path.join(family_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load graphs
        graphs_file = os.path.join(family_dir, "graphs.pkl")
        with open(graphs_file, 'rb') as f:
            family_graphs = pickle.load(f)
        
        print(f"Loaded graph family: {family_id}")
        print(f"Total graphs: {len(family_graphs)}")
        print(f"Splits - Pretraining: {len(metadata['splits']['pretraining'])}, "
              f"Warmup: {len(metadata['splits']['warmup'])}, "
              f"Fine-tuning: {len(metadata['splits']['finetuning'])}")
        
        return family_graphs, metadata
    
    def get_graph_splits(self, family_graphs: List, metadata: Dict) -> Dict[str, List]:
        """Get graph splits based on metadata."""
        splits = metadata['splits']
        
        return {
            'pretraining': [family_graphs[i] for i in splits['pretraining']],
            'warmup': [family_graphs[i] for i in splits['warmup']],
            'finetuning': [family_graphs[i] for i in splits['finetuning']]
        }
    
    def list_families(self) -> List[Dict]:
        """List all available graph families."""
        families = []
        
        if not os.path.exists(self.graph_family_dir):
            return families
        
        for family_id in os.listdir(self.graph_family_dir):
            family_dir = os.path.join(self.graph_family_dir, family_id)
            metadata_file = os.path.join(family_dir, "metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                families.append({
                    'family_id': family_id,
                    'creation_timestamp': metadata.get('creation_timestamp'),
                    'total_graphs': metadata.get('total_graphs', 0),
                    'config': metadata.get('config', {}),
                    'n_finetuning': metadata.get('n_finetuning', 0)
                })
        
        return sorted(families, key=lambda x: x.get('creation_timestamp', ''), reverse=True)

class PreTrainedModelSaver:
    """Handles saving and loading of pre-trained models."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_model(
        self,
        model: torch.nn.Module,
        config: PreTrainingConfig,
        training_history: Dict[str, List[float]],
        metrics: Dict[str, float],
        hyperopt_results: Optional[Dict] = None,
        enhanced_metadata: Optional[Dict] = None,
        model_id: Optional[str] = None
    ) -> str:
        """Save pre-trained model with enhanced metadata including family references."""
        
        if model_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{config.gnn_type}_{config.pretraining_task}_{timestamp}"
        
        model_dir = os.path.join(self.output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(model_dir, "model.pth")
        model_copy = copy.deepcopy(model)
        torch.save(model_copy.state_dict(), model_path)
        
        # Save model architecture info
        arch_info = {
            'model_class': model.__class__.__name__,
            'encoder_class': model.encoder.__class__.__name__,
            'input_dim': getattr(model.encoder, 'input_dim', None),
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'gnn_type': config.gnn_type,
            'dropout': config.dropout
        }
        
        # Create complete metadata with enhanced info
        metadata = {
            'model_id': model_id,
            'creation_timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'architecture': arch_info,
            'training_history': training_history,
            'final_metrics': metrics,
            'hyperopt_results': hyperopt_results,
            'model_path': model_path
        }
        
        # Add enhanced metadata (family info, etc.)
        if enhanced_metadata:
            metadata['enhanced_info'] = enhanced_metadata
            
            # Special handling for family info
            if 'family_id' in enhanced_metadata:
                metadata['family_id'] = enhanced_metadata['family_id']
                metadata['finetuning_ready'] = enhanced_metadata.get('finetuning_graphs_available', 0) > 0
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save config separately for easy loading
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save family reference file for easy lookup
        if enhanced_metadata and 'family_id' in enhanced_metadata:
            family_ref = {
                'family_id': enhanced_metadata['family_id'],
                'model_id': model_id,
                'finetuning_graphs_available': enhanced_metadata.get('finetuning_graphs_available', 0),
                'pretraining_graphs_used': enhanced_metadata.get('pretraining_graphs_used', 0),
                'creation_timestamp': datetime.now().isoformat()
            }
            
            family_ref_path = os.path.join(model_dir, "family_reference.json")
            with open(family_ref_path, 'w') as f:
                json.dump(family_ref, f, indent=2)
        
        print(f"Saved pre-trained model to: {model_dir}")
        if enhanced_metadata and 'family_id' in enhanced_metadata:
            print(f"✓ Linked to graph family: {enhanced_metadata['family_id']}")
            print(f"✓ Fine-tuning graphs available: {enhanced_metadata.get('finetuning_graphs_available', 0)}")
        
        return model_id
    
    def load_model(self, model_id: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict]:
        """Load pre-trained model and metadata."""
        model_dir = os.path.join(self.output_dir, model_id)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = PreTrainingConfig(**config_dict)
        
        # Recreate model architecture
        from experiments.inductive.self_supervised_task import LinkPredictionTask, ContrastiveTask
        
        if metadata['config']['pretraining_task'] == 'link_prediction':
            task = LinkPredictionTask(config)
        elif metadata['config']['pretraining_task'] == 'contrastive':
            task = ContrastiveTask(config)
        else:
            raise ValueError(f"Unknown task: {metadata['config']['pretraining_task']}")
        
        # Need input_dim to create model
        input_dim = metadata['architecture']['input_dim']
        if input_dim is None:
            raise ValueError("Cannot recreate model: input_dim not saved in metadata")
        
        model = task.create_model(input_dim)
        
        # Load weights
        model_path = os.path.join(model_dir, "model.pth")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available pre-trained models with enhanced info."""
        models = []
        
        if not os.path.exists(self.output_dir):
            return models
        
        for model_id in os.listdir(self.output_dir):
            model_dir = os.path.join(self.output_dir, model_id)
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_info = {
                    'model_id': model_id,
                    'task': metadata['config']['pretraining_task'],
                    'gnn_type': metadata['config']['gnn_type'],
                    'creation_timestamp': metadata.get('creation_timestamp'),
                    'final_metrics': metadata.get('final_metrics', {}),
                    'model_dir': model_dir,
                    'family_id': metadata.get('family_id', None),
                    'finetuning_ready': metadata.get('finetuning_ready', False)
                }
                
                # Add enhanced info if available
                if 'enhanced_info' in metadata:
                    enhanced = metadata['enhanced_info']
                    model_info.update({
                        'finetuning_graphs_available': enhanced.get('finetuning_graphs_available', 0),
                        'pretraining_graphs_used': enhanced.get('pretraining_graphs_used', 0),
                        'family_total_graphs': enhanced.get('family_total_graphs', 0)
                    })
                
                models.append(model_info)
        
        return sorted(models, key=lambda x: x.get('creation_timestamp', ''), reverse=True)
    
    def get_models_by_family(self, family_id: str) -> List[Dict[str, Any]]:
        """Get all models trained on a specific graph family."""
        all_models = self.list_models()
        return [model for model in all_models if model.get('family_id') == family_id]

def prepare_inductive_data(
    family_graphs: List[GraphSample],
    config
) -> Dict[str, Dict[str, Any]]:
    """Updated to use universe-based metapath generation only."""
    
    # Get universe from first graph for metapath generation
    universe = family_graphs[0].universe if family_graphs else None
    
    if universe and hasattr(config, 'enable_metapath_tasks') and config.enable_metapath_tasks:
        # Use universe-based approach
        return prepare_inductive_data_with_universe_metapaths(
            family_graphs=family_graphs,
            universe=universe,
            config=config
        )
    else:
        # Standard preparation without metapaths
        return prepare_inductive_data_standard(family_graphs, config)

def prepare_inductive_data_standard(
    family_graphs: List[GraphSample],
    config
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare graph family data for inductive learning.
    Now includes metapath-based classification tasks.
    
    Args:
        family_graphs: List of GraphSample objects from the same family
        config: Clean inductive experiment configuration
        
    Returns:
        Dictionary containing data for each task, organized by split
    """
    # Calculate split sizes
    n_graphs = len(family_graphs)
    n_train = int(n_graphs * config.train_graph_ratio)
    n_val = int(n_graphs * config.val_graph_ratio)
    n_test = n_graphs - n_train - n_val
    
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
    
    # Generate metapath tasks if enabled
    metapath_data = None
    if hasattr(config, 'enable_metapath_tasks') and config.enable_metapath_tasks:
        print("\nGenerating metapath-based classification tasks...")
        metapath_data = generate_metapath_tasks(
            family_graphs,
            k_values=getattr(config, 'metapath_k_values', [3, 4, 5]),
            require_loop=getattr(config, 'metapath_require_loop', True),
            min_prob_threshold=getattr(config, 'metapath_min_prob_threshold', 0.01),
            degree_weight=getattr(config, 'metapath_degree_weight', 0.3),
            max_tasks_per_graph=getattr(config, 'metapath_max_tasks', 2)
        )
        
        # Add metapath tasks to config.tasks
        if metapath_data and metapath_data['tasks']:
            config.tasks.extend(list(metapath_data['tasks'].keys()))
            print(f"Added {len(metapath_data['tasks'])} metapath tasks: {list(metapath_data['tasks'].keys())}")
    
    # Process each task
    for task in config.tasks:
        print(f"\nPreparing inductive data for task: {task}")
        
        task_data = {}
        
        # Check if this is a metapath task
        is_metapath_task = (metapath_data and task in metapath_data['tasks'])
        
        # Process each split
        for split_name, graphs_in_split in graph_split_dict.items():
            n_graphs_in_split = len(graphs_in_split)
            print(f"  Processing {split_name} split with {n_graphs_in_split} graphs")
            
            if n_graphs_in_split == 0:
                print(f"  Warning: No graphs in {split_name} split, skipping")
                task_data[split_name] = {
                    'graphs': [],
                    'n_graphs': 0,
                    'batch_size': 1
                }
                continue
            
            # Convert graphs to PyTorch Geometric format
            pyg_graphs = []
            for i, graph_sample in enumerate(graphs_in_split):
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
                
                elif is_metapath_task:
                    # Use metapath labels
                    # Find corresponding labels for this graph
                    graph_idx = train_indices[i] if split_name == 'train' else (
                        val_indices[i] if split_name == 'val' else test_indices[i]
                    )
                    
                    # Find metapath labels for this specific graph
                    metapath_labels = None
                    for task_entry in metapath_data['tasks'][task]['data']:
                        if task_entry['graph_idx'] == graph_idx:
                            metapath_labels = task_entry['labels']
                            break
                    
                    if metapath_labels is not None:
                        pyg_data.y = torch.tensor(metapath_labels, dtype=torch.long)
                    else:
                        # Generate labels for this specific graph
                        metapath_labels = generate_metapath_labels_for_graph(
                            graph_sample, 
                            metapath_data['tasks'][task]['metapath']
                        )
                        pyg_data.y = torch.tensor(metapath_labels, dtype=torch.long)
                
                # Add graph-level metadata
                pyg_data.graph_id = len(pyg_graphs)
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
                'batch_size': max(1, batch_size)
            }
        
        # Add task-specific metadata
        is_regression = config.is_regression.get(task, False)
        
        # Calculate output dimension
        if is_regression:
            # For regression, use number of communities
            sample_graph = family_graphs[0]
            output_dim = len(np.unique(sample_graph.community_labels))
        elif is_metapath_task:
            # For metapath tasks, check if it's a loop task (3 classes) or binary (2 classes)
            task_info = metapath_data['tasks'][task]
            # Check if any graph has loop instances (label 2)
            has_loops = False
            for task_entry in task_info['data']:
                if np.any(np.array(task_entry['labels']) == 2):
                    has_loops = True
                    break
            output_dim = 3 if has_loops else 2
        else:
            # For classification, use number of unique labels across all graphs
            all_labels = []
            for graph in family_graphs:
                all_labels.extend(graph.community_labels.tolist())
            output_dim = len(np.unique(all_labels))
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': family_graphs[0].features.shape[1] if family_graphs[0].features is not None else 0,
            'is_metapath_task': is_metapath_task
        }
        
        # Add metapath-specific metadata
        if is_metapath_task:
            task_data['metadata'].update({
                'metapath': metapath_data['tasks'][task]['metapath'],
                'avg_positive_rate': metapath_data['tasks'][task]['avg_positive_rate'],
                'task_coverage': metapath_data['tasks'][task]['coverage']
            })
        
        inductive_data[task] = task_data
    
    # Add metapath analysis to results
    if metapath_data:
        inductive_data['metapath_analysis'] = metapath_data['analysis']
    
    return inductive_data

def generate_metapath_labels_for_graph(
    graph_sample: GraphSample,
    metapath: List[int]
) -> np.ndarray:
    """
    Generate metapath labels for a specific graph given a metapath.
    
    Args:
        graph_sample: GraphSample object
        metapath: List of communities forming the metapath
        
    Returns:
        Binary array indicating metapath participation
    """
    analyzer = MetapathAnalyzer(
        P_sub=graph_sample.P_sub,
        community_labels=graph_sample.community_labels,
        degree_factors=graph_sample.degree_factors,
        verbose=False
    )
    
    labels = analyzer.create_node_labels_from_metapath(
        graph_sample.graph, metapath, "metapath_temp"
    )
    
    return labels

def analyze_metapath_properties(
    analyzer: MetapathAnalyzer,
    metapath: List[int]
) -> Dict[str, Any]:
    """
    Analyze properties of a discovered metapath.
    
    Args:
        analyzer: MetapathAnalyzer instance
        metapath: List of communities forming the metapath
        
    Returns:
        Dictionary with metapath analysis
    """
    analysis = {
        'metapath': metapath,
        'length': len(metapath),
        'is_loop': metapath[0] == metapath[-1] if len(metapath) > 1 else False,
        'unique_communities': len(set(metapath)),
        'community_transitions': []
    }
    
    # Analyze each transition
    total_prob = 1.0
    for i in range(len(metapath) - 1):
        from_comm = metapath[i]
        to_comm = metapath[i + 1]
        
        base_prob = analyzer.P_sub[from_comm, to_comm]
        
        # Apply degree weighting
        degree_multiplier = 1.0
        if analyzer.degree_weight > 0:
            attraction = analyzer.community_degree_stats[to_comm]['attraction_weight']
            degree_multiplier = (1 - analyzer.degree_weight) + analyzer.degree_weight * attraction
        
        final_prob = base_prob * degree_multiplier
        total_prob *= final_prob
        
        transition_info = {
            'from_community': from_comm,
            'to_community': to_comm,
            'base_probability': base_prob,
            'degree_multiplier': degree_multiplier,
            'final_probability': final_prob,
            'from_community_size': analyzer.community_degree_stats[from_comm]['node_count'],
            'to_community_size': analyzer.community_degree_stats[to_comm]['node_count']
        }
        analysis['community_transitions'].append(transition_info)
    
    analysis['total_probability'] = total_prob
    
    return analysis

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
    config
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
        if task == 'metadata' or task == 'metapath_analysis':
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
        'community_counts': [],
        'generation_methods': []
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
        
        # Track generation method
        if hasattr(graph, 'generation_method'):
            properties['generation_methods'].append(graph.generation_method)
    
    # Calculate statistics and convert to native Python types
    for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts']:
        values = properties[key]
        if values:
            properties[f'{key}_mean'] = float(np.mean(values))
            properties[f'{key}_std'] = float(np.std(values))
            properties[f'{key}_min'] = float(np.min(values))
            properties[f'{key}_max'] = float(np.max(values))
        else:
            properties[f'{key}_mean'] = 0.0
            properties[f'{key}_std'] = 0.0
            properties[f'{key}_min'] = 0.0
            properties[f'{key}_max'] = 0.0
    
    # Add generation method summary
    if properties['generation_methods']:
        from collections import Counter
        method_counts = Counter(properties['generation_methods'])
        properties['generation_method_distribution'] = dict(method_counts)
    
    return properties

def create_metapath_benchmark_config():
    """
    Create a sample configuration for metapath-based benchmarking.
    
    Returns:
        Configuration object with metapath settings
    """
    class MetapathConfig:
        def __init__(self):
            # Basic experiment settings
            self.seed = 42
            self.batch_size = 32
            self.train_graph_ratio = 0.6
            self.val_graph_ratio = 0.2
            
            # Task configuration
            self.tasks = ["community"]  # Will be extended with metapath tasks
            self.is_regression = {"community": False}
            
            # Metapath task settings
            self.enable_metapath_tasks = True
            self.metapath_k_values = [3, 4, 5]
            self.metapath_require_loop = True
            self.metapath_min_prob_threshold = 0.01
            self.metapath_degree_weight = 0.3
            self.metapath_max_tasks = 2
            
            # K-hop community count settings (if needed)
            self.khop_community_counts_k = 2
    
    return MetapathConfig()

def generate_universe_based_metapath_tasks(
    family_graphs: List[GraphSample],
    universe: 'GraphUniverse',
    train_indices: List[int],
    val_indices: List[int], 
    test_indices: List[int],
    k_values: List[int] = [3, 4, 5],
    require_loop: bool = True,
    degree_weight: float = 0.3,
    max_community_participation: float = 0.95,
    n_candidates_per_k: int = 30
) -> Dict[str, Any]:
    """
    Generate metapath tasks using the new universe-based approach.
    
    Args:
        family_graphs: List of GraphSample objects
        universe: GraphUniverse object used to generate the family
        train_indices: Indices for training graphs
        val_indices: Indices for validation graphs
        test_indices: Indices for test graphs
        k_values: List of metapath lengths to try
        require_loop: Whether metapaths must form loops
        degree_weight: Weight for degree center influence
        max_community_participation: Max allowed participation rate per community
        n_candidates_per_k: Number of candidates to generate per k-value
        
    Returns:
        Dictionary containing valid metapath tasks and detailed analysis
    """
    
    print("\n[DEBUG] Starting generate_universe_based_metapath_tasks")
    print(f"[DEBUG] Input validation:")
    print(f"  - family_graphs length: {len(family_graphs)}")
    print(f"  - universe K: {universe.K}")
    print(f"  - train/val/test indices: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"  - k_values: {k_values}")
    print(f"  - require_loop: {require_loop}")
    print(f"  - degree_weight: {degree_weight}")
    print(f"  - max_community_participation: {max_community_participation}")
    print(f"  - n_candidates_per_k: {n_candidates_per_k}")
    
    # Step 1: Generate candidates based purely on universe parameters
    print("\n[DEBUG] Step 1: Creating UniverseMetapathSelector")
    selector = UniverseMetapathSelector(
        universe=universe,
        degree_weight=degree_weight,
        verbose=True
    )
    
    print("\n[DEBUG] Generating diverse metapath candidates")
    candidates = selector.generate_diverse_metapath_candidates(
        k_values=k_values,
        require_loop=require_loop,
        n_candidates_per_k=n_candidates_per_k,
        diversity_factor=0.4,
        min_prob_threshold=0.001
    )
    
    print(f"[DEBUG] Generated candidates per k: {[len(cands) for k, cands in candidates.items()]}")
    
    # Analyze candidate properties
    candidate_analysis = selector.analyze_candidate_properties(candidates)
    
    print(f"\n[DEBUG] Candidate analysis:")
    print(f"  - Total candidates: {candidate_analysis['total_candidates']}")
    print(f"  - Candidates per k: {candidate_analysis['candidates_per_k']}")
    
    # Step 2: Evaluate candidates on family splits
    print("\n[DEBUG] Step 2: Creating FamilyMetapathEvaluator")
    evaluator = FamilyMetapathEvaluator(verbose=True)
    
    print("\n[DEBUG] Evaluating candidates on family")
    evaluation_results = evaluator.evaluate_candidates_on_family(
        candidates=candidates,
        family_graphs=family_graphs,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        max_community_participation=max_community_participation
    )
    
    print(f"\n[DEBUG] Evaluation results:")
    print(f"  - Valid metapaths: {len(evaluation_results['valid_metapaths'])}")
    print(f"  - Rejected metapaths: {len(evaluation_results['rejected_metapaths'])}")
    
    # Step 3: Convert valid metapaths to task format
    print("\n[DEBUG] Step 3: Converting to task format")
    valid_tasks = {}
    
    for task_name, metapath_data in evaluation_results['valid_metapaths'].items():
        print(f"\n[DEBUG] Processing task: {task_name}")
        metapath = metapath_data['metapath']
        k = metapath_data['k']
        
        # Create labels for each graph in the family
        task_labels_by_graph = {}
        
        for graph_idx, graph_sample in enumerate(family_graphs):
            try:
                # Map metapath to this graph's local communities
                local_metapath = []
                for universe_comm_id in metapath:
                    try:
                        local_idx = graph_sample.communities.index(universe_comm_id)
                        local_metapath.append(local_idx)
                    except ValueError:
                        local_metapath = None
                        break
                
                if local_metapath is not None:
                    # Generate labels using MetapathAnalyzer
                    analyzer = MetapathAnalyzer(
                        P_sub=graph_sample.P_sub,
                        community_labels=graph_sample.community_labels,
                        degree_factors=graph_sample.degree_factors,
                        verbose=False
                    )
                    
                    labels = analyzer.create_node_labels_from_metapath(
                        graph_sample.graph, local_metapath, f"{task_name}_graph_{graph_idx}"
                    )
                    
                    task_labels_by_graph[graph_idx] = labels
                else:
                    # Graph doesn't contain all required communities
                    task_labels_by_graph[graph_idx] = None
                    
            except Exception as e:
                print(f"[DEBUG] Warning: Failed to create labels for graph {graph_idx}: {e}")
                task_labels_by_graph[graph_idx] = None
        
        # Calculate coverage (how many graphs can use this task)
        valid_graphs = [idx for idx, labels in task_labels_by_graph.items() if labels is not None]
        coverage = len(valid_graphs) / len(family_graphs)
        
        print(f"[DEBUG] Task {task_name} coverage: {coverage:.3f} ({len(valid_graphs)}/{len(family_graphs)} graphs)")
        
        # Only keep tasks with reasonable coverage
        if coverage >= 0.5:  # At least 50% of graphs must support the task
            
            # Calculate average positive rate across valid graphs
            positive_rates = []
            for graph_idx in valid_graphs:
                labels = task_labels_by_graph[graph_idx]
                positive_rate = np.mean(labels > 0)
                positive_rates.append(positive_rate)
            
            avg_positive_rate = np.mean(positive_rates) if positive_rates else 0.0
            
            valid_tasks[task_name] = {
                'metapath': metapath,
                'universe_probability': metapath_data['universe_probability'],
                'k': k,
                'is_loop_task': metapath[0] == metapath[-1] and len(metapath) > 3,
                'coverage': coverage,
                'avg_positive_rate': avg_positive_rate,
                'labels_by_graph': task_labels_by_graph,
                'valid_graphs': valid_graphs,
                'evaluation_data': metapath_data['evaluation']
            }
            
            print(f"[DEBUG] Added valid task: {task_name}")
            print(f"  - Metapath: {' -> '.join(map(str, metapath))}")
            print(f"  - Coverage: {coverage:.3f}")
            print(f"  - Avg positive rate: {avg_positive_rate:.3f}")
    
    # Generate detailed report
    print("\n[DEBUG] Generating detailed report")
    detailed_report = evaluator.generate_detailed_report(evaluation_results)
    
    print("\n[DEBUG] Final results:")
    print(f"  - Valid tasks created: {len(valid_tasks)}")
    
    return {
        'tasks': valid_tasks,
        'evaluation_results': evaluation_results,
        'candidate_analysis': candidate_analysis,
        'detailed_report': detailed_report,
        'universe_info': {
            'K': universe.K,
            'degree_centers': universe.degree_centers.tolist(),
            'P_matrix': universe.P.tolist()
        }
    }

def prepare_inductive_data_with_universe_metapaths(
    family_graphs: List[GraphSample],
    universe: 'GraphUniverse',
    config
) -> Dict[str, Dict[str, Any]]:
    """
    Updated prepare_inductive_data that uses universe-based metapath generation.
    """
    # Calculate split sizes
    n_graphs = len(family_graphs)
    n_train = int(n_graphs * config.train_graph_ratio)
    n_val = int(n_graphs * config.val_graph_ratio)
    n_test = n_graphs - n_train - n_val
    
    print(f"\nSplitting {n_graphs} graphs: {n_train} train, {n_val} val, {n_test} test")
    
    # Split graphs
    np.random.seed(config.seed)
    indices = np.random.permutation(n_graphs)
    
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    graph_split_dict = {
        'train': [family_graphs[i] for i in train_indices],
        'val': [family_graphs[i] for i in val_indices],
        'test': [family_graphs[i] for i in test_indices]
    }
    
    # Initialize results dictionary
    inductive_data = {}
    
    # Generate universe-based metapath tasks if enabled
    metapath_data = None
    if hasattr(config, 'enable_metapath_tasks') and config.enable_metapath_tasks:
        print("\nGenerating universe-based metapath tasks...")

        metapath_data = generate_universe_based_metapath_tasks(
            family_graphs=family_graphs,
            universe=universe,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            k_values=getattr(config, 'metapath_k_values', [3, 4, 5]),
            require_loop=getattr(config, 'metapath_require_loop', False),
            degree_weight=getattr(config, 'metapath_degree_weight', 0.3),
            max_community_participation=getattr(config, 'max_community_participation', 1.0),
            n_candidates_per_k=getattr(config, 'n_candidates_per_k', 40)
        )
        
        # Add metapath tasks to config.tasks
        if metapath_data and metapath_data['tasks']:
            # Use a consistent task name for metapath tasks
            config.tasks = ["metapath"]  # Replace all tasks with just "metapath"
            print(f"Added metapath task")
    
    # Process each task
    for task in config.tasks:
        print(f"\nPreparing inductive data for task: {task}")
        
        task_data = {}
        
        # Check if this is a metapath task
        is_metapath_task = (metapath_data and metapath_data['tasks'])
        
        # Process each split
        for split_name, graphs_in_split in graph_split_dict.items():
            n_graphs_in_split = len(graphs_in_split)
            print(f"  Processing {split_name} split with {n_graphs_in_split} graphs")
            
            if n_graphs_in_split == 0:
                task_data[split_name] = {
                    'graphs': [],
                    'n_graphs': 0,
                    'batch_size': 1
                }
                continue
            
            # Convert graphs to PyTorch Geometric format
            pyg_graphs = []
            
            if split_name == 'train':
                graph_indices = train_indices
            elif split_name == 'val':
                graph_indices = val_indices
            else:
                graph_indices = test_indices
            
            for i, graph_sample in enumerate(graphs_in_split):
                # Convert to PyG format
                pyg_data = graphsample_to_pyg(graph_sample)
                
                # Add task-specific labels
                if task == "community":
                    pyg_data.y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
                    
                elif is_metapath_task:
                    # Use universe-based metapath labels
                    original_graph_idx = graph_indices[i]
                    task_info = list(metapath_data['tasks'].values())[0]  # Get the first (and only) task
                    
                    if original_graph_idx in task_info['labels_by_graph']:
                        metapath_labels = task_info['labels_by_graph'][original_graph_idx]
                        if metapath_labels is not None:
                            # Convert to binary labels (0 = not participating, 1 = participating)
                            binary_labels = (metapath_labels > 0).astype(int)
                            pyg_data.y = torch.tensor(binary_labels, dtype=torch.long)
                        else:
                            # Skip this graph - doesn't support the metapath
                            continue
                    else:
                        continue
                
                # Add graph-level metadata
                pyg_data.graph_id = len(pyg_graphs)
                pyg_data.n_communities = len(np.unique(graph_sample.community_labels))
                
                pyg_graphs.append(pyg_data)
            
            # Calculate batch size for this split
            if split_name == 'train':
                batch_size = min(len(pyg_graphs), config.batch_size) if pyg_graphs else 1
            else:
                batch_size = len(pyg_graphs) if pyg_graphs else 1
            
            # Store data for this split
            task_data[split_name] = {
                'graphs': pyg_graphs,
                'n_graphs': len(pyg_graphs),
                'batch_size': max(1, batch_size)
            }
        
        # Add task-specific metadata
        is_regression = config.is_regression.get(task, False)
        
        # Calculate output dimension
        if is_regression:
            output_dim = len(np.unique(family_graphs[0].community_labels))
        elif is_metapath_task:
            output_dim = 2  # Binary classification: participating or not
        else:
            all_labels = []
            for graph in family_graphs:
                all_labels.extend(graph.community_labels.tolist())
            output_dim = len(np.unique(all_labels))
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': family_graphs[0].features.shape[1] if family_graphs[0].features is not None else 0,
            'is_metapath_task': is_metapath_task
        }
        
        # Add metapath-specific metadata
        if is_metapath_task:
            task_info = list(metapath_data['tasks'].values())[0]  # Get the first (and only) task
            task_data['metadata'].update({
                'metapath': task_info['metapath'],
                'universe_probability': task_info['universe_probability'],
                'coverage': task_info['coverage'],
                'avg_positive_rate': task_info['avg_positive_rate'],
                'is_loop_task': task_info['is_loop_task']
            })
        
        inductive_data[task] = task_data
    
    # Add metapath analysis to results
    if metapath_data:
        inductive_data['metapath_analysis'] = {
            'evaluation_results': metapath_data['evaluation_results'],
            'candidate_analysis': metapath_data['candidate_analysis'],
            'detailed_report': metapath_data['detailed_report'],
            'universe_info': metapath_data['universe_info']
        }
    
    return inductive_data

def load_graphs_for_finetuning(family_id: str, graph_family_dir: str = "graph_families") -> Tuple[List, Dict]:
    """Load fine-tuning graphs from a saved family."""
    from experiments.inductive.config import PreTrainingConfig
    
    config = PreTrainingConfig(graph_family_dir=graph_family_dir)
    manager = GraphFamilyManager(config)
    
    family_graphs, metadata = manager.load_family(family_id)
    graph_splits = manager.get_graph_splits(family_graphs, metadata)
    
    return graph_splits['finetuning'], metadata

def list_graph_families(graph_family_dir: str = "graph_families") -> List[Dict]:
    """List available graph families."""
    from experiments.inductive.config import PreTrainingConfig
    
    config = PreTrainingConfig(graph_family_dir=graph_family_dir)
    manager = GraphFamilyManager(config)
    return manager.list_families()
