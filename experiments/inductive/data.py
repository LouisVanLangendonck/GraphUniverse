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
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
import copy

from mmsb.model import GraphSample, GraphUniverse
from mmsb.feature_regimes import graphsample_to_pyg
from utils.metapath_analysis import MetapathAnalyzer, UniverseMetapathSelector, FamilyMetapathEvaluator
from experiments.inductive.config import PreTrainingConfig, InductiveExperimentConfig

class PositionalEncodingComputer:
    """Compute various types of positional encodings for graphs."""
    
    def __init__(self, max_pe_dim: int = 16, pe_types: List[str] = None):
        """
        Initialize PE computer.
        
        Args:
            max_pe_dim: Maximum PE dimension
            pe_types: List of PE types to compute ['laplacian', 'degree', 'rwse']
        """
        self.max_pe_dim = max_pe_dim
        self.pe_types = pe_types or ['laplacian']
    
    def compute_degree_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Degree-based PE - most transferable across graphs."""
        from torch_geometric.utils import degree
        
        degrees = degree(edge_index[0], num_nodes=num_nodes).float()
        pe = torch.zeros(num_nodes, self.max_pe_dim)
        
        for i in range(min(self.max_pe_dim, 8)):
            pe[:, i] = (degrees ** (i / 4.0)) / (1 + degrees ** (i / 4.0))
        
        return pe
    
    def compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Random Walk Structural Encoding - landing probabilities after k steps."""
        try:
            from torch_geometric.utils import degree
            
            # Get node degrees
            degrees = degree(edge_index[0], num_nodes=num_nodes).float()
            
            # Handle isolated nodes
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
            
            # Create adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            # Transition matrix: P[i,j] = A[i,j] / degree[i]
            P = adj / degrees.unsqueeze(1)
            
            # Compute powers of transition matrix for different walk lengths
            rwse = torch.zeros(num_nodes, self.max_pe_dim)
            P_power = torch.eye(num_nodes)  # P^0 = I
            
            for k in range(self.max_pe_dim):
                if k > 0:
                    P_power = P_power @ P  # P^k
                
                # Use diagonal entries (return probabilities) as features
                rwse[:, k] = P_power.diag()
            
            return rwse
            
        except Exception as e:
            print(f"Warning: RWSE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)
    
    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Laplacian Positional Encoding using eigenvectors."""
        try:
            # Handle empty/trivial graphs
            if edge_index.shape[1] == 0 or num_nodes <= 1:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            # Get normalized Laplacian
            edge_index_lap, edge_weight = get_laplacian(
                edge_index, 
                edge_weight=None,
                normalization='sym', 
                num_nodes=num_nodes
            )
            
            # Convert to scipy sparse matrix
            L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
            
            # Compute eigenvalues/eigenvectors
            k = min(self.max_pe_dim, num_nodes - 2)
            if k <= 0:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            try:
                eigenvals, eigenvecs = eigsh(
                    L, 
                    k=k, 
                    which='SM',  # Smallest eigenvalues
                    return_eigenvectors=True,
                    tol=1e-6
                )
            except:
                # Fallback for small graphs
                L_dense = L.toarray()
                eigenvals, eigenvecs = np.linalg.eigh(L_dense)
                idx = np.argsort(eigenvals)
                eigenvecs = eigenvecs[:, idx[1:k+1]]  # Skip first (constant) eigenvector
            
            # Handle sign ambiguity
            for i in range(eigenvecs.shape[1]):
                if eigenvecs[0, i] < 0:
                    eigenvecs[:, i] *= -1
            
            # Pad or truncate to max_pe_dim
            if eigenvecs.shape[1] < self.max_pe_dim:
                pad_width = self.max_pe_dim - eigenvecs.shape[1]
                eigenvecs = np.pad(eigenvecs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                eigenvecs = eigenvecs[:, :self.max_pe_dim]
            
            return torch.tensor(eigenvecs, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: Laplacian PE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)
    
    def compute_all_pe(self, edge_index: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
        """Compute all requested PE types."""
        pe_dict = {}
        
        for pe_type in self.pe_types:
            if pe_type == 'laplacian':
                pe = self.compute_laplacian_pe(edge_index, num_nodes)
                pe_dict['laplacian_pe'] = pe
                    
            elif pe_type == 'degree':
                pe = self.compute_degree_pe(edge_index, num_nodes)
                pe_dict['degree_pe'] = pe
                    
            elif pe_type == 'rwse':
                pe = self.compute_rwse(edge_index, num_nodes)
                pe_dict['rwse_pe'] = pe
                    
            elif pe_type == 'random_walk':  # Legacy support
                pe = self.compute_degree_pe(edge_index, num_nodes)  # Use degree PE as fallback
                pe_dict['random_walk_pe'] = pe
        
        return pe_dict
 
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
        n_actual_pretraining, n_warmup, n_finetuning = self.config.get_graph_splits(len(family_graphs))
        print("SAVING GRAPH FAMILY METADATA")
        print(f"n_actual_pretraining: {n_actual_pretraining}, n_warmup: {n_warmup}, n_finetuning: {n_finetuning}")
        
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
            # Use transformer_type for transformer models, otherwise use gnn_type
            model_type = config.transformer_type if config.run_transformers else config.gnn_type
            model_id = f"{model_type}_{config.pretraining_task}_{timestamp}"
        
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
        from experiments.inductive.self_supervised_task import LinkPredictionTask, DeepGraphInfoMaxTask, GraphMAETask
        
        if metadata['config']['pretraining_task'] == 'link_prediction':
            task = LinkPredictionTask(config)
        elif metadata['config']['pretraining_task'] == 'dgi':
            task = DeepGraphInfoMaxTask(config)
        elif metadata['config']['pretraining_task'] == 'graphmae':
            task = GraphMAETask(config)
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
   
def select_graphs_for_maximum_coverage(
    family_graphs: List[GraphSample],
    universe_K: int,
    max_graphs: int,
    seed: Optional[int] = None
) -> List[int]:
    """
    Select graphs that maximize coverage of universe communities.
    
    Args:
        family_graphs: List of graph samples
        universe_K: Number of communities in universe
        max_graphs: Maximum number of graphs to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected graph indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_graphs = len(family_graphs)
    selected_indices = []
    covered_communities = set()
    
    # Create a list of available indices
    available_indices = list(range(n_graphs))
    
    while len(selected_indices) < max_graphs and available_indices:
        best_idx = None
        best_new_communities = 0
        
        # Find graph with most new communities
        for idx in available_indices:
            graph = family_graphs[idx]
            graph_communities = set(graph.communities)
            new_communities = len(graph_communities - covered_communities)
            
            if new_communities > best_new_communities:
                best_new_communities = new_communities
                best_idx = idx
        
        if best_idx is None:
            break
            
        # Add the best graph to selection
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
        
        # Update covered communities
        graph = family_graphs[best_idx]
        covered_communities.update(graph.communities)
        
        # If we've covered all communities, we can stop
        if len(covered_communities) == universe_K:
            break
    
    return selected_indices

def prepare_inductive_data(
    family_graphs: List[GraphSample],
    config,
    pe_types: List[str] = ['laplacian', 'degree', 'rwse'],
    family_graphs_training_indices: Optional[List[int]] = None,
    family_graphs_val_test_indices: Optional[List[int]] = None,
    family_graph_community_labels_list: Optional[List[List[int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Central function to prepare graph family data for inductive learning.
    Handles multiple task types (community prediction, k-hop counting, metapath tasks)
    in a unified way.
    
    Args:
        family_graphs: List of GraphSample objects from the same family
        config: Experiment configuration
        
    Returns:
        Dictionary containing data for each task, organized by split
    """
    # Get universe from first graph if available
    universe = family_graphs[0].universe if family_graphs else None
    if not universe:
        raise ValueError("No universe found in graph family")
    
    # Get universe K
    universe_K = universe.K
    print(f"\nUsing universe K: {universe_K}")
    
    # Calculate split sizes
    if config.allow_unseen_community_combinations_for_eval:
        n_graphs = len(family_graphs)
    else:
        n_graphs = len(family_graphs_training_indices)

    n_train = int(n_graphs * config.train_graph_ratio)
    n_val = int(n_graphs * config.val_graph_ratio)
    n_test = n_graphs - n_train - n_val

    print(f"\nGraph split for each run: {n_graphs} graphs, {n_train} train, {n_val} val, {n_test} test")

    # Initialize results dictionary
    inductive_data = {}
    
    # First convert all graphs to PyG format and compute labels for each task
    print("\nConverting graphs to PyG format and computing labels...")
    pyg_graphs_by_task = {}
    
    for task in config.tasks:
        print(f"  Processing task: {task}")
        pyg_graphs = []
        
        for i, graph_sample in enumerate(family_graphs):
            # Convert to PyG format
            pyg_data = graphsample_to_pyg(graph_sample)
            
            # Add universe K to each graph
            pyg_data.universe_K = universe_K
            
            # Generate task-specific labels
            if task == "community":
                # Standard community prediction - use universe-indexed labels
                pyg_data.y = torch.tensor(graph_sample.community_labels_universe_level, dtype=torch.long)
            
            elif task == "triangle_count":
                # Triangle counting (via networkx then to tensor)
                pyg_data.y = torch.tensor(count_triangles_graph(graph_sample.graph), dtype=torch.float)
                
            elif task.startswith("k_hop_community_counts_k"):
                # Extract k value from task name
                k = int(task.split("k")[-1])
                # K-hop community counting - already universe-indexed
                community_counts = compute_khop_community_counts_universe_indexed(
                    graph_sample.graph,
                    graph_sample.community_labels,
                    graph_sample.community_id_mapping,
                    universe_K,
                    k
                )
                pyg_data.y = community_counts
            
            # Add graph-level metadata
            pyg_data.graph_id = i
            pyg_data.n_communities = len(np.unique(graph_sample.community_labels))
            
            pyg_graphs.append(pyg_data)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(family_graphs)} graphs")
        
        pyg_graphs_by_task[task] = pyg_graphs

    # Add PE precomputation
    if config.precompute_pe:
        print(f"\nPrecomputing positional encodings...")
        print(f"PE types: {pe_types}")
        print(f"PE dimension: {config.max_pe_dim}")

        # Create temporary data structure for PE computation
        temp_data = {task: {'graphs': graphs} for task, graphs in pyg_graphs_by_task.items()}
        temp_data = add_positional_encodings_to_data(
            temp_data, 
            pe_types=pe_types,
            max_pe_dim=config.max_pe_dim
        )
        # Update graphs with PE
        for task in pyg_graphs_by_task:
            pyg_graphs_by_task[task] = temp_data[task]['graphs']

    # Now create k-fold splits
    if config.k_fold <= 0:
        raise ValueError(f"K-fold must be greater than 0")
    
    # For all folds, fix the test set to be the same for all folds
    if config.allow_unseen_community_combinations_for_eval:
        test_indices = np.random.permutation(n_graphs)[:n_test].tolist()
        remaining_indices = [i for i in range(n_graphs) if i not in test_indices]
    else:
        print(family_graphs_val_test_indices)
        test_indices = np.random.permutation(family_graphs_val_test_indices)[:n_test].tolist()
        test_community_combinations = []
        for idx in test_indices:
            test_community_combinations.append(set(family_graph_community_labels_list[idx].tolist()))
        print(f"Test community combinations: {test_community_combinations}")

        mandatory_train_indices = []
        seen_combinations_for_mandatory_train = []
        for training_index in family_graphs_training_indices:
            graph_community_labels = set(family_graph_community_labels_list[training_index].tolist())
            if any(graph_community_labels == test_comb for test_comb in test_community_combinations) and not any(graph_community_labels == seen_comb for seen_comb in seen_combinations_for_mandatory_train):
                mandatory_train_indices.append(training_index)
                seen_combinations_for_mandatory_train.append(graph_community_labels)
        
        # Remove mandatory train indices from remaining train indices
        remaining_train_indices = [i for i in family_graphs_training_indices if i not in mandatory_train_indices]
        remaining_val_indices = [i for i in family_graphs_val_test_indices if i not in test_indices]

    # Process each task
    for task in config.tasks:
        print(f"\nPreparing k-fold splits for task: {task}")
        task_data = {}
        
        # Get the pre-processed graphs for this task
        pyg_graphs = pyg_graphs_by_task[task]
        test_graphs = [pyg_graphs[i] for i in test_indices]

        # Create k-fold splits
        fold_indices = {}
        for i in range(config.k_fold):
            fold_indices[i] = {'train': [], 'val': [], 'test': []}
            # Randomly shuffle the indices
            np.random.seed(config.seed + i)  # Different seed for each fold

            if config.allow_unseen_community_combinations_for_eval:
                indices = np.random.permutation(remaining_indices)
                train_indices = indices[:n_train].tolist()
                val_indices = indices[n_train:n_train + n_val].tolist()
                fold_indices[i]['train'] = train_indices
                fold_indices[i]['val'] = val_indices
                fold_indices[i]['test'] = test_indices
            else:
                # First add mandatory train indices
                train_indices = mandatory_train_indices.copy()
                
                # Get remaining number of train indices needed
                remaining_n_train_indices_to_sample = n_train - len(mandatory_train_indices)
                
                # Randomly sample remaining train indices
                permuted_train_indices = np.random.permutation(remaining_train_indices)
                train_indices.extend(permuted_train_indices[:remaining_n_train_indices_to_sample].tolist())
                
                # Get all community combinations in current train fold
                train_community_combinations = set()
                for train_idx in train_indices:
                    train_community_combinations.add(tuple(sorted(family_graph_community_labels_list[train_idx].tolist())))
                
                # Select validation indices that have community combinations seen in train
                val_indices = []
                permuted_val_indices = np.random.permutation(remaining_val_indices)
                
                for val_idx in permuted_val_indices:
                    val_community_combination = tuple(sorted(family_graph_community_labels_list[val_idx].tolist()))
                    if val_community_combination in train_community_combinations and len(val_indices) < n_val:
                        val_indices.append(val_idx)
                
                print(f"Fold {i}:")
                print(f"  Train size: {len(train_indices)} (including {len(mandatory_train_indices)} mandatory)")
                print(f"  Val size: {len(val_indices)}")
                print(f"  Test size: {len(test_indices)}")
                
                fold_indices[i]['train'] = train_indices
                fold_indices[i]['val'] = val_indices
                fold_indices[i]['test'] = test_indices

            fold_name = f'fold_{i}'
            task_data[fold_name] = {
                'train': {
                    'graphs': [pyg_graphs[i] for i in train_indices],
                    'n_graphs': len(train_indices),
                    'batch_size': min(len(train_indices), config.batch_size),
                    'indices': train_indices
                },
                'val': {
                    'graphs': [pyg_graphs[i] for i in val_indices],
                    'n_graphs': len(val_indices),
                    'batch_size': len(val_indices),
                    'indices': val_indices
                },
                'test': {
                    'graphs': test_graphs,
                    'n_graphs': len(test_graphs),
                    'batch_size': len(test_graphs),
                    'indices': test_indices
                }
            }
        
        # Add task-specific metadata
        is_regression = config.is_regression.get(task, False)
        
        # Calculate output dimension based on task type
        if task == "community":
            output_dim = universe_K
        elif task == "triangle_count":
            output_dim = 1
        elif task.startswith("k_hop_community_counts"):
            output_dim = universe_K
        else:
            raise ValueError(f"Unknown task: {task}")
        
        task_data['metadata'] = {
            'is_regression': is_regression,
            'output_dim': output_dim,
            'input_dim': family_graphs[0].features.shape[1] if family_graphs[0].features is not None else 0,
            'task_type': task,
            'universe_K': universe_K
        }
        
        # Add task-specific metadata
        if task.startswith("k_hop_community_counts_k"):
            k = int(task.split("k")[-1])
            task_data['metadata'].update({
                'k_value': k,
                'universe_K': universe_K
            })
        
        inductive_data[task] = task_data
    
    # After all splits are created for each task, create sheaf-specific splits with batch size 1
    sheaf_inductive_data = {}
    for task, task_data in inductive_data.items():
        sheaf_task_data = {}
        for fold_name, fold_data in task_data.items():
            if fold_name == 'metadata' or fold_name == 'metapath_analysis':
                sheaf_task_data[fold_name] = copy.deepcopy(fold_data)
                continue
                
            sheaf_fold_data = {}
            for split_name, split_data in fold_data.items():
                if split_name == 'metadata' or split_name == 'metapath_analysis':
                    sheaf_fold_data[split_name] = copy.deepcopy(split_data)
                    continue
                    
                # Deep copy graphs and attach cache
                sheaf_graphs = [precompute_sheaf_laplacian(copy.deepcopy(g)) for g in split_data['graphs']]
                sheaf_fold_data[split_name] = {
                    'graphs': sheaf_graphs,
                    'n_graphs': len(sheaf_graphs),
                    'batch_size': 1,  # Always batch size 1 for sheaf
                    'indices': split_data['indices']
                }
            sheaf_task_data[fold_name] = sheaf_fold_data
        sheaf_inductive_data[task] = sheaf_task_data
    
    # Return normal splits, sheaf-specific splits, and fold indices
    return inductive_data, sheaf_inductive_data, fold_indices

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

def generate_universe_based_metapath_tasks(
    family_graphs: List[GraphSample],
    universe: 'GraphUniverse',
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    k_values: List[int] = [3, 4, 5],
    require_loop: bool = False,
    degree_weight: float = 0.3,
    max_community_participation: float = 1.0,
    n_candidates_per_k: int = 40
) -> Dict[str, Any]:
    """
    Generate metapath tasks using universe-based approach.
    
    Args:
        family_graphs: List of graph samples
        universe: Graph universe
        train_indices: Indices of training graphs
        val_indices: Indices of validation graphs
        test_indices: Indices of test graphs
        k_values: List of k values for metapath length
        require_loop: Whether to require loop in metapath
        degree_weight: Weight for degree-based scoring
        max_community_participation: Maximum community participation ratio
        n_candidates_per_k: Number of candidates per k value
        
    Returns:
        Dictionary containing metapath task data
    """
    # Implementation details omitted for brevity
    # This function should implement the universe-based metapath generation logic
    pass

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

def validate_khop_consistency(family_graphs: List, universe_K: int) -> Dict[str, Any]:
    """
    Validate that community indexing is consistent across graphs in the family.
    
    Args:
        family_graphs: List of GraphSample objects
        universe_K: Total number of communities in universe
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'all_communities_used': set(),
        'graph_community_ranges': [],
        'max_universe_community': 0,
        'indexing_consistent': True,
        'issues': []
    }
    
    for i, graph in enumerate(family_graphs):
        # Check community range
        universe_communities = graph.communities
        local_communities = np.unique(graph.community_labels)
        
        validation_results['all_communities_used'].update(universe_communities)
        validation_results['graph_community_ranges'].append({
            'graph_id': i,
            'universe_communities': list(universe_communities),
            'local_communities': list(local_communities),
            'max_universe_comm': max(universe_communities) if universe_communities else 0
        })
        
        # Check for issues
        if max(universe_communities) >= universe_K:
            validation_results['issues'].append(
                f"Graph {i}: Universe community {max(universe_communities)} >= universe_K {universe_K}"
            )
            validation_results['indexing_consistent'] = False
        
        if len(local_communities) != len(universe_communities):
            validation_results['issues'].append(
                f"Graph {i}: Mismatch between local ({len(local_communities)}) and universe ({len(universe_communities)}) community counts"
            )
            validation_results['indexing_consistent'] = False
    
    validation_results['max_universe_community'] = max(validation_results['all_communities_used']) if validation_results['all_communities_used'] else 0
    validation_results['total_unique_communities'] = len(validation_results['all_communities_used'])
    
    return validation_results

# # --- Custom Collate Functions ---
# def triangle_count_collate_fn(batch):
#     """
#     Custom collate function for triangle count tasks.
#     Sums the y values (triangle counts) in each batch.
#     """
#     from torch_geometric.data import Batch
    
#     # Create the batch using PyG's default collate
#     batched_data = Batch.from_data_list(batch)
    
#     # Sum the triangle counts (y values) for the batch
#     if hasattr(batched_data, 'y') and batched_data.y is not None:
#         # Sum all triangle counts in the batch
#         batched_data.y = batched_data.y.sum().unsqueeze(0)  # Keep as tensor with shape [1]
    
#     return batched_data

def ensure_batch_attribute_collate_fn(batch):
    """
    Custom collate function that ensures the batch attribute is properly set.
    This function explicitly creates the batch attribute for each node.
    """
    from torch_geometric.data import Batch
    
    # Create the batch using PyG's default collate
    batched_data = Batch.from_data_list(batch)
    
    # The batch attribute should already be set by Batch.from_data_list()
    # But we can verify it exists and has the correct shape
    if not hasattr(batched_data, 'batch') or batched_data.batch is None:
        # Manually create batch attribute if it doesn't exist
        batch_attr = []
        for i, data in enumerate(batch):
            batch_attr.extend([i] * data.x.size(0))
        batched_data.batch = torch.tensor(batch_attr, dtype=torch.long)
    
    return batched_data

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
        task_loaders = {}
        for fold_name, fold_data in task_data.items():
            fold_loaders = {}
            if fold_name == 'metadata' or fold_name == 'metapath_analysis':
                continue
            
            # Determine batch size for this task
            # No need to enforce consistent batch size for triangle_count; use split batch size
            
            for split_name, split_data in fold_data.items():
                if split_name == 'metadata' or split_name == 'metapath_analysis':
                    continue
                
                shuffle = (split_name == 'train')
                
                batch_size = split_data['batch_size']
                
                # Use custom collate function for triangle count tasks
                
                if task == "triangle_count":
                    collate_fn = ensure_batch_attribute_collate_fn
                
                    loader = DataLoader(
                        split_data['graphs'],
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                        collate_fn=collate_fn
                    )
                else:
                    loader = DataLoader(
                        split_data['graphs'],
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                    )
                
                fold_loaders[split_name] = loader
                
            
            task_loaders[fold_name] = fold_loaders
        
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
        'homophily_levels': [],  # New property for homophily
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
        
        # Calculate homophily level
        if graph.n_nodes > 0 and graph.graph.number_of_edges() > 0:
            # Count edges between nodes of same community
            same_community_edges = 0
            for u, v in graph.graph.edges():
                if graph.community_labels[u] == graph.community_labels[v]:
                    same_community_edges += 1
            homophily = same_community_edges / graph.graph.number_of_edges()
            properties['homophily_levels'].append(homophily)
        else:
            properties['homophily_levels'].append(0.0)
        
        # Track generation method
        if hasattr(graph, 'generation_method'):
            properties['generation_methods'].append(graph.generation_method)
    
    # Calculate statistics and convert to native Python types
    for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels']:
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

def add_positional_encodings_to_data(
    inductive_data: Dict[str, Dict[str, Any]], 
    pe_types: List[str] = ['laplacian', 'degree', 'rwse'],
    max_pe_dim: int = 16
) -> Dict[str, Dict[str, Any]]:
    """
    Add precomputed positional encodings to all graphs in inductive data.
    
    Args:
        inductive_data: The prepared inductive data dictionary
        pe_types: Types of PE to compute ['laplacian', 'degree', 'rwse']
        max_pe_dim: Maximum PE dimension
    
    Returns:
        Updated inductive data with PE added to each graph
    """
    pe_computer = PositionalEncodingComputer(
        max_pe_dim=max_pe_dim, 
        pe_types=pe_types
    )
    
    print(f"Precomputing positional encodings: {pe_types}")
    print("Note: No graph normalization (good for inductive learning)")
    total_graphs = 0
    
    # Process each task
    for task_name, task_data in inductive_data.items():
        if task_name in ['metadata', 'metapath_analysis']:
            continue
            
        print(f"  Processing task: {task_name}")
        
        # Get graphs for this task
        graphs = task_data['graphs']
        print(f"    Processing {len(graphs)} graphs")
        
        for i, graph in enumerate(graphs):
            # Compute PE for this graph
            pe_dict = pe_computer.compute_all_pe(graph.edge_index, graph.x.size(0))
            
            # Add PE to graph data
            for pe_name, pe_tensor in pe_dict.items():
                setattr(graph, pe_name, pe_tensor)
            
            total_graphs += 1
            
            if (i + 1) % 10 == 0:
                print(f"      Processed {i + 1}/{len(graphs)} graphs")
    
    print(f"✓ Added PE to {total_graphs} graphs total")
    return inductive_data

def count_triangles_graph(graph: nx.Graph) -> int:
    """Count the number of triangles in a graph."""
    # Use networkx's triangles function which is more reliable
    triangle_counts = nx.triangles(graph)
    # Sum all triangle counts and divide by 3 (each triangle is counted 3 times)
    total_triangles = sum(triangle_counts.values()) // 3
    return total_triangles

def create_gpu_resident_dataloaders(
    inductive_data: Dict[str, Dict[str, Dict[str, Any]]],
    config,
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """
    Create dataloaders with all data pre-loaded to GPU to avoid repeated CPU-GPU transfers.
    
    Args:
        inductive_data: Prepared inductive data
        config: Experiment configuration
        device: Device to load data onto
        
    Returns:
        Dictionary containing GPU-resident dataloaders for each task and split
    """
    from torch_geometric.loader import DataLoader
    
    print(f"🚀 Pre-loading all graph data to {device}...")
    
    dataloaders = {}
    
    for task, task_data in inductive_data.items():
        task_loaders = {}
        for fold_name, fold_data in task_data.items():
            fold_loaders = {}
            if fold_name == 'metadata' or fold_name == 'metapath_analysis':
                continue
            
            # Determine batch size for this task
            # No need to enforce consistent batch size for triangle_count; use split batch size
            
            for split_name, split_data in fold_data.items():
                if split_name == 'metadata' or split_name == 'metapath_analysis':
                    continue
                
                print(f"  Loading {split_name} data for {task}/{fold_name}...")
                
                # Move all graphs to GPU at once
                gpu_graphs = []
                for graph in split_data['graphs']:
                    # Create a copy of the graph on GPU
                    gpu_graph = graph.to(device)
                    gpu_graphs.append(gpu_graph)
                
                shuffle = (split_name == 'train')
                
                # Use consistent batch size for triangle count tasks
                batch_size = split_data['batch_size']
                
                # Use custom collate function for triangle count tasks
                # Use the batch attribute ensuring collate function for all tasks
                
                if task == "triangle_count":
                    collate_fn = ensure_batch_attribute_collate_fn
                
                    # Create dataloader with GPU-resident data
                    loader = DataLoader(
                        gpu_graphs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,  # Keep 0 since data is already on GPU
                        collate_fn=collate_fn
                    )
                else:
                    # Create dataloader with GPU-resident data
                    loader = DataLoader(
                        gpu_graphs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,  # Keep 0 since data is already on GPU
                    )
                
                fold_loaders[split_name] = loader
                
                print(f"    ✓ Loaded {len(gpu_graphs)} graphs to {device}")
            
            task_loaders[fold_name] = fold_loaders
        
        dataloaders[task] = task_loaders
    
    print(f"✅ All data loaded to {device}")
    return dataloaders

def verify_gpu_resident_data(dataloaders: Dict[str, Dict[str, Any]], device: torch.device) -> bool:
    """
    Verify that all data in dataloaders is actually on the specified device.
    
    Args:
        dataloaders: GPU-resident dataloaders to verify
        device: Expected device
        
    Returns:
        True if all data is on the correct device, False otherwise
    """
    print(f"🔍 Verifying data is on {device}...")
    
    for task, task_data in dataloaders.items():
        for fold_name, fold_data in task_data.items():
            for split_name, dataloader in fold_data.items():
                if split_name == 'metadata':
                    continue
                
                # Check first few batches
                batch_count = 0
                for batch in dataloader:
                    if batch_count >= 3:  # Only check first 3 batches
                        break
                    
                    # Check main tensors
                    if hasattr(batch, 'x') and batch.x.device != device:
                        print(f"❌ {task}/{fold_name}/{split_name}: x tensor on {batch.x.device}, expected {device}")
                        return False
                    
                    if hasattr(batch, 'edge_index') and batch.edge_index.device != device:
                        print(f"❌ {task}/{fold_name}/{split_name}: edge_index tensor on {batch.edge_index.device}, expected {device}")
                        return False
                    
                    if hasattr(batch, 'y') and batch.y.device != device:
                        print(f"❌ {task}/{fold_name}/{split_name}: y tensor on {batch.y.device}, expected {device}")
                        return False
                    
                    batch_count += 1
    
    print(f"✅ All data verified to be on {device}")
    return True

# --- Sheaf Laplacian Precomputation ---
def precompute_sheaf_laplacian(graph):
    """
    Precompute and cache hyperparameter-independent indices for sheaf diffusion models.
    Only computes the expensive index operations that are independent of hyperparameters.
    """
    from experiments.neural_sheaf_diffusion.lib import laplace as lap
    from torch_geometric.utils import degree
    
    size = graph.x.size(0)
    edge_index = graph.edge_index
    
    # Precompute the expensive index operations that are independent of hyperparameters
    start_time = time.time()
    full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)
    left_right_idx, vertex_tril_idx = lap.compute_left_right_map_index(edge_index)
    end_time = time.time()
    # print(f"TIME DEBUG: Index precomputation for {size} nodes: {end_time - start_time:.4f}s")
    
    # Compute degree (also independent of hyperparameters)
    deg = degree(edge_index[0], num_nodes=size)
    
    # Store only the hyperparameter-independent components
    graph.sheaf_indices_cache = {
        'size': size,
        'edge_index': edge_index,
        'full_left_right_idx': full_left_right_idx,
        'left_right_idx': left_right_idx,
        'vertex_tril_idx': vertex_tril_idx,
        'deg': deg,
        'edges': edge_index.size(1) // 2
    }
    
    return graph

# --- Sheaf-specific dataloader creation ---
def create_sheaf_dataloaders(sheaf_inductive_data, config):
    """
    Create dataloaders for sheaf splits (batch size 1 everywhere).
    Usage: sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    """
    from torch_geometric.loader import DataLoader
    dataloaders = {}
    for task, task_data in sheaf_inductive_data.items():
        task_loaders = {}
        for fold_name, fold_data in task_data.items():
            fold_loaders = {}
            if fold_name == 'metadata' or fold_name == 'metapath_analysis':
                continue
            for split_name, split_data in fold_data.items():
                if split_name == 'metadata' or split_name == 'metapath_analysis':
                    continue
                
                # Use custom collate function for triangle count tasks
                collate_fn = triangle_count_collate_fn if task == "triangle_count" else None
                
                loader = DataLoader(
                    split_data['graphs'],
                    batch_size=1,  # Always batch size 1
                    shuffle=(split_name == 'train'),
                    num_workers=0,
                    collate_fn=collate_fn
                )
                fold_loaders[split_name] = loader
            task_loaders[fold_name] = fold_loaders
        dataloaders[task] = task_loaders
    return dataloaders

# --- Usage in experiment runner ---
# When running the sheaf model, use the sheaf_inductive_data and create_sheaf_dataloaders.
# For other models, use the normal splits and dataloaders.
# This ensures identical splits but with sheaf-specific precomputation and batch size 1.