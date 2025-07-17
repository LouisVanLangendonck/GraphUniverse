"""
Clean inductive experiment orchestration.
Focuses on DC-SBM and DCCC-SBM with improved metrics collection.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import numpy as np
import torch
import optuna
from dataclasses import asdict
from torch.cuda.amp import autocast, GradScaler
from torch.nn import init
from torch.nn.init import xavier_uniform_, kaiming_uniform_, normal_
import torch.nn as nn
import copy
import pickle

from graph_universe.model import GraphUniverse
from graph_universe.graph_family import GraphFamilyGenerator, FamilyConsistencyAnalyzer
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders,
    analyze_graph_family_properties,
    create_gpu_resident_dataloaders,
    verify_gpu_resident_data
)
from experiments.inductive.training import (
    train_and_evaluate_inductive, 
    get_total_classes_from_dataloaders,
    cleanup_gpu_dataloaders
)
from experiments.models import GNNModel, MLPModel, SklearnModel, SheafDiffusionModel
from experiments.inductive.self_supervised_task import (
    PreTrainingConfig, 
    SelfSupervisedTask, 
    create_ssl_task,
    PreTrainedModelSaver,
    create_pretraining_dataloader
)
from experiments.inductive.data import GraphFamilyManager, create_gpu_resident_dataloaders
from experiments.inductive.config import PreTrainingConfig, InductiveExperimentConfig
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InductiveExperiment:
    """Clean inductive learning experiment runner."""
    
    def __init__(self, config):
        """Initialize experiment with clean config."""
        self.config = config
        
        # Set up device
        if config.force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = self._setup_device(config.device_id)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"inductive_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.results = {}
        self.family_graphs = None
        self.family_consistency = None
        self.graph_signals = None
        
        print(f"Experiment initialized. Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Method: {'DCCC-SBM' if config.use_dccc_sbm else 'DC-SBM'}")
    
    def _setup_device(self, device_id: int = 0) -> torch.device:
        """Set up compute device."""
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
                return device
        return torch.device("cpu")
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def generate_graph_family(self) -> List:
        """Generate graph family using clean parameters or load from specified family."""
        
        # Check if we should load existing graphs for fine-tuning
        if self.config.use_pretrained:
            finetuning_graphs = load_finetuning_graphs_from_model(self.config)
            if finetuning_graphs:
                print(f"Using {len(finetuning_graphs)} graphs for fine-tuning")
                self.family_graphs = finetuning_graphs
                return finetuning_graphs
        
        # Otherwise, generate new graph family
        print("\n" + "="*60)
        print("GENERATING GRAPH FAMILY")
        print("="*60)
        
        # Create universe and family generator
        universe, family_generator = self._create_universe_and_generator()
        
        # Generate graphs based on distributional shift configuration
        if self.config.distributional_shift_in_eval:
            self.family_graphs, self.communities_per_graph = self._generate_with_distributional_shift(
                family_generator, universe
            )
        else:
            self.family_graphs, self.communities_per_graph = self._generate_standard_family(
                family_generator
            )
        
        return self.family_graphs
    
    def _create_universe_and_generator(self):
        """Create universe and family generator with clean parameters."""
        print("Creating graph universe...")
        universe = GraphUniverse(
            K=self.config.universe_K,
            feature_dim=self.config.universe_feature_dim,
            edge_density=self.config.universe_edge_density,
            homophily=self.config.universe_homophily,
            randomness_factor=self.config.universe_randomness_factor,
            cluster_count_factor=self.config.cluster_count_factor,
            center_variance=self.config.center_variance,
            cluster_variance=self.config.cluster_variance,
            assignment_skewness=self.config.assignment_skewness,
            community_exclusivity=self.config.community_exclusivity,
            degree_center_method=self.config.degree_center_method,
            community_density_variation=self.config.community_density_variation,
            community_cooccurrence_homogeneity=self.config.community_cooccurrence_homogeneity,
            triangle_density=self.config.triangle_density,
            triangle_community_relation_homogeneity=self.config.triangle_community_relation_homogeneity,
            seed=self.config.seed
        )

        print("Setting up graph family generator...")
        family_generator = GraphFamilyGenerator(
            universe=universe,
            min_n_nodes=self.config.min_n_nodes,
            max_n_nodes=self.config.max_n_nodes,
            min_communities=self.config.min_communities,
            max_communities=self.config.max_communities,
            min_component_size=self.config.min_component_size,
            homophily_range=self.config.homophily_range,
            density_range=self.config.density_range,
            
            # Method selection
            use_dccc_sbm=self.config.use_dccc_sbm,
            degree_distribution=self.config.degree_distribution,
            
            # DCCC-SBM parameters
            community_imbalance_range=self.config.community_imbalance_range,
            degree_separation_range=self.config.degree_separation_range,
            power_law_exponent_range=self.config.power_law_exponent_range,
            exponential_rate_range=self.config.exponential_rate_range,
            uniform_min_factor_range=self.config.uniform_min_factor_range,
            uniform_max_factor_range=self.config.uniform_max_factor_range,
            
            # Fixed parameters
            degree_heterogeneity=self.config.degree_heterogeneity,
            edge_noise=self.config.edge_noise,
            
            # Generation constraints
            max_parameter_search_attempts=self.config.max_parameter_search_attempts,
            parameter_search_range=self.config.parameter_search_range,
            max_retries=self.config.max_retries,
            min_edge_density=self.config.min_edge_density,
            disable_deviation_limiting=self.config.disable_deviation_limiting,
            max_mean_community_deviation=self.config.max_mean_community_deviation,
            max_max_community_deviation=self.config.max_max_community_deviation,
            
            seed=self.config.seed
        )
        
        return universe, family_generator
    
    def _generate_standard_family(self, family_generator):
        """Generate standard graph family without distributional shifts."""
        print(f"Generating family of {self.config.n_graphs} graphs...")
        start_time = time.time()
        
        family_graphs = family_generator.generate_family(
            show_progress=False,
            collect_stats=self.config.collect_family_stats,
            n_graphs=self.config.n_graphs
        )
        communities_per_graph = family_generator.community_labels_per_graph
        
        generation_time = time.time() - start_time
        print(f"Family generation completed in {generation_time:.2f} seconds")
        print(f"Successfully generated {len(family_graphs)} graphs")
        
        # Set indices for standard case (all graphs available for all splits)
        self.family_graphs_training_indices = list(range(len(family_graphs)))
        self.family_graphs_val_test_indices = list(range(len(family_graphs)))
        self.family_graphs_training_val_indices = None
        self.family_graphs_test_indices = None
        
        return family_graphs, communities_per_graph
    
    def _generate_with_distributional_shift(self, family_generator, universe):
        """Generate graph family with distributional shifts."""
        shift_type = self.config.distributional_shift_in_eval_type
        
        if shift_type in ['homophily', 'density', 'n_nodes']:
            return self._generate_property_shift(family_generator, universe, shift_type)
        else:
            raise ValueError(f"Unknown distributional shift type: {shift_type}")
        
    def _generate_property_shift(self, family_generator, universe, shift_type):
        """Generate graphs with property shift (homophily, density, or n_nodes)."""
        print(f"Generating family with {shift_type} shift...")
        
        # Determine which splits should have the shift
        if self.config.distributional_shift_test_only:
            # Generate training+val graphs normally
            print(f"Generating training+validation graphs...")
            training_val_graphs = family_generator.generate_family(
                show_progress=False,
                collect_stats=self.config.collect_family_stats,
                n_graphs=self.config.n_graphs
            )
            training_val_communities = family_generator.community_labels_per_graph
            
            # Generate test graphs with shift
            print(f"Generating test graphs with {shift_type} shift...")
            shifted_graphs = self._generate_shifted_graphs(
                family_generator, universe, shift_type, 
                n_graphs=int(self.config.n_graphs * self.config.test_graph_ratio)
            )
            shifted_communities = family_generator.community_labels_per_graph
            
            # Merge and set indices
            family_graphs = training_val_graphs + shifted_graphs
            communities_per_graph = training_val_communities + shifted_communities
            
            self.family_graphs_training_val_indices = list(range(len(training_val_graphs)))
            self.family_graphs_test_indices = list(range(len(training_val_graphs), len(family_graphs)))
            self.family_graphs_training_indices = None
            self.family_graphs_val_test_indices = None
            
        else:
            # Generate training graphs normally
            print(f"Generating training graphs...")
            training_graphs = family_generator.generate_family(
                show_progress=False,
                collect_stats=self.config.collect_family_stats,
                n_graphs=self.config.n_graphs
            )
            training_communities = family_generator.community_labels_per_graph
            
            # Generate val+test graphs with shift
            print(f"Generating validation+test graphs with {shift_type} shift...")
            shifted_graphs = self._generate_shifted_graphs(
                family_generator, universe, shift_type, 
                n_graphs=self.config.n_graphs
            )
            shifted_communities = family_generator.community_labels_per_graph
            
            # Merge and set indices
            family_graphs = training_graphs + shifted_graphs
            communities_per_graph = training_communities + shifted_communities
            
            self.family_graphs_training_indices = list(range(len(training_graphs)))
            self.family_graphs_val_test_indices = list(range(len(training_graphs), len(family_graphs)))
            self.family_graphs_training_val_indices = None
            self.family_graphs_test_indices = None
        
        print(f"Total graphs: {len(family_graphs)}")
        print(f"Training graphs: {len(training_graphs) if 'training_graphs' in locals() else len(training_val_graphs)}")
        print(f"Shifted graphs: {len(shifted_graphs)}")
        
        return family_graphs, communities_per_graph
    
    def _generate_shifted_graphs(self, family_generator, universe, shift_type, n_graphs):
        """Generate graphs with a specific property shift."""
        # Create shifted universe
        shifted_universe = copy.deepcopy(universe)
        
        # Apply the shift
        if shift_type == 'homophily':
            shift_amount = self.config.distributional_shift_in_eval_homophily_shift
            direction = -1 # Fix to have a homophily shift in the negative direction
            shifted_universe.homophily = np.clip(
                universe.homophily + direction * shift_amount, 0, 1
            )
            print(f"Homophily shift: {universe.homophily:.3f} -> {shifted_universe.homophily:.3f}")
            
        elif shift_type == 'density':
            shift_amount = self.config.distributional_shift_in_eval_density_shift
            direction = 1 # Fix to have a density shift in the positive direction
            shifted_universe.edge_density = np.clip(
                universe.edge_density + direction * shift_amount, 0, 0.5
            )
            print(f"Density shift: {universe.edge_density:.3f} -> {shifted_universe.edge_density:.3f}")
            
        elif shift_type == 'n_nodes':
            shift_amount = self.config.distributional_shift_in_eval_n_nodes_shift
            min_nodes = self.config.min_n_nodes + shift_amount
            max_nodes = self.config.max_n_nodes + shift_amount
            print(f"Node count shift: [{self.config.min_n_nodes}, {self.config.max_n_nodes}] -> [{min_nodes}, {max_nodes}]")
            
            # Create new family generator with shifted node range
            shifted_family_generator = GraphFamilyGenerator(
                universe=shifted_universe,
                min_n_nodes=min_nodes,
                max_n_nodes=max_nodes,
                min_communities=self.config.min_communities,
                max_communities=self.config.max_communities,
                min_component_size=self.config.min_component_size,
                homophily_range=self.config.homophily_range,
                density_range=self.config.density_range,
                
                # Method selection
                use_dccc_sbm=self.config.use_dccc_sbm,
                degree_distribution=self.config.degree_distribution,
                
                # DCCC-SBM parameters
                community_imbalance_range=self.config.community_imbalance_range,
                degree_separation_range=self.config.degree_separation_range,
                power_law_exponent_range=self.config.power_law_exponent_range,
                exponential_rate_range=self.config.exponential_rate_range,
                uniform_min_factor_range=self.config.uniform_min_factor_range,
                uniform_max_factor_range=self.config.uniform_max_factor_range,
                
                # Fixed parameters
                degree_heterogeneity=self.config.degree_heterogeneity,
                edge_noise=self.config.edge_noise,
                
                # Generation constraints
                max_parameter_search_attempts=self.config.max_parameter_search_attempts,
                parameter_search_range=self.config.parameter_search_range,
                max_retries=self.config.max_retries,
                min_edge_density=self.config.min_edge_density,
                disable_deviation_limiting=self.config.disable_deviation_limiting,
                max_mean_community_deviation=self.config.max_mean_community_deviation,
                max_max_community_deviation=self.config.max_max_community_deviation,
                
                seed=self.config.seed
            )
            
            return shifted_family_generator.generate_family(
                show_progress=False,
                collect_stats=self.config.collect_family_stats,
                n_graphs=n_graphs
            )
        
        shifted_family_generator = GraphFamilyGenerator(
            universe=shifted_universe,
            min_n_nodes=self.config.min_n_nodes,
            max_n_nodes=self.config.max_n_nodes,
            min_communities=self.config.min_communities,
            max_communities=self.config.max_communities,
            min_component_size=self.config.min_component_size,
            homophily_range=self.config.homophily_range,
            density_range=self.config.density_range,
            
            # Method selection
            use_dccc_sbm=self.config.use_dccc_sbm,
            degree_distribution=self.config.degree_distribution,
            
            # DCCC-SBM parameters
            community_imbalance_range=self.config.community_imbalance_range,
            degree_separation_range=self.config.degree_separation_range,
            power_law_exponent_range=self.config.power_law_exponent_range,
            exponential_rate_range=self.config.exponential_rate_range,
            uniform_min_factor_range=self.config.uniform_min_factor_range,
            uniform_max_factor_range=self.config.uniform_max_factor_range,
            
            # Fixed parameters
            degree_heterogeneity=self.config.degree_heterogeneity,
            edge_noise=self.config.edge_noise,
            
            # Generation constraints
            max_parameter_search_attempts=self.config.max_parameter_search_attempts,
            parameter_search_range=self.config.parameter_search_range,
            max_retries=self.config.max_retries,
            min_edge_density=self.config.min_edge_density,
            disable_deviation_limiting=self.config.disable_deviation_limiting,
            max_mean_community_deviation=self.config.max_mean_community_deviation,
            max_max_community_deviation=self.config.max_max_community_deviation,
            
            seed=self.config.seed
        )
        
        return shifted_family_generator.generate_family(
            show_progress=False,
            collect_stats=self.config.collect_family_stats,
            n_graphs=n_graphs
        )
    
    def analyze_family_consistency(self) -> Dict[str, Any]:
        """Analyze family consistency using existing methods."""
        if self.family_graphs is None:
            raise ValueError("Must generate graph family first")
        
        print("\nAnalyzing family consistency...")
        
        # Get universe from first graph
        universe = self.family_graphs[0].universe
        
        # Create consistency analyzer
        consistency_analyzer = FamilyConsistencyAnalyzer(self.family_graphs, universe)
        consistency_results = consistency_analyzer.analyze_consistency()
        
        overall_consistency = consistency_results.get('overall', {}).get('score', 0.0)
        print(f"Family consistency score: {overall_consistency:.3f}")
        
        if self.config.require_consistency_check:
            if overall_consistency < self.config.min_family_consistency:
                raise ValueError(
                    f"Family consistency {overall_consistency:.3f} below required minimum "
                    f"{self.config.min_family_consistency:.3f}"
                )
        
        self.family_consistency = self._make_json_serializable(consistency_results)
        return self.family_consistency
    
    def calculate_graph_signals(self) -> Dict[str, Any]:
        """Calculate community signals for each graph in the family."""
        if self.family_graphs is None:
            raise ValueError("Must generate graph family first")
        
        if not self.config.collect_signal_metrics:
            return {}
        
        print("\nCalculating community signals for each graph...")
        
        all_signals = {
            'degree_signals': [],
            'structure_signals': [],
            'feature_signals': []
        }
        
        for i, graph in enumerate(self.family_graphs):
            try:
                # Calculate community signals using existing methods
                signals = graph.calculate_community_signals(
                    structure_metric='kl',
                    degree_method='naive_bayes',
                    degree_metric='accuracy',
                    random_state=self.config.seed + i
                )
                
                # Store individual signals
                all_signals['degree_signals'].append(signals.get('degree_signal', 0.0))
                all_signals['structure_signals'].append(signals.get('mean_structure_signal', 0.0))
                
                # Feature signal (may be None)
                feature_signal = signals.get('feature_signal')
                if feature_signal is not None:
                    all_signals['feature_signals'].append(feature_signal)
                
            except Exception as e:
                print(f"Warning: Failed to calculate signals for graph {i}: {e}")
                all_signals['degree_signals'].append(0.0)
                all_signals['structure_signals'].append(0.0)
        
        # Calculate aggregated statistics
        aggregated_signals = {}
        
        for signal_type, values in all_signals.items():
            if values:  # Only if we have values
                aggregated_signals[signal_type] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'individual_values': [float(v) for v in values]
                }
            else:
                aggregated_signals[signal_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'individual_values': []
                }
        
        self.graph_signals = aggregated_signals
        return aggregated_signals
    
    def prepare_data(self) -> Dict[str, Dict[str, Any]]:
        """Prepare data for inductive learning."""
        print("\n" + "="*60)
        print("PREPARING INDUCTIVE DATA")
        print("="*60)
        
        if self.family_graphs is None:
            raise ValueError("Must generate graph family before preparing data")
        
        # Prepare inductive data
        inductive_data, _, split_index_dict = prepare_inductive_data(self.family_graphs, 
                                                self.config,
                                                family_graphs_training_indices=self.family_graphs_training_indices,
                                                family_graphs_val_test_indices=self.family_graphs_val_test_indices,
                                                family_graphs_training_val_indices=self.family_graphs_training_val_indices,
                                                family_graphs_test_indices=self.family_graphs_test_indices,
                                                family_graph_community_labels_list=self.communities_per_graph,
                                                )
        
        # Store data dictionary to save later
        self.inductive_data = inductive_data
        
        # Create GPU-resident dataloaders for efficiency
        print("Creating GPU-resident dataloaders...")
        dataloaders = create_gpu_resident_dataloaders(inductive_data, 
                                                   self.config,
                                                   self.device)
        
        # Verify data is properly loaded to GPU
        if not verify_gpu_resident_data(dataloaders, self.device):
            raise RuntimeError("Failed to load data to GPU properly")
        
        # Print split information
        for task in self.config.tasks:
            print(f"\nTask: {task}")
            if 'split' in inductive_data[task]:
                # New single split structure
                split_data = inductive_data[task]['split']
                for split_name, split_info in split_data.items():
                    n_graphs = split_info['n_graphs']
                    batch_size = split_info['batch_size']
                    print(f"  {split_name}: {n_graphs} graphs, batch size {batch_size}")
            else:
                # Old fold-based structure (for backward compatibility)
                for fold_name, fold_data in inductive_data[task].items():
                    if fold_name == 'metadata':  # Skip metadata entry
                        continue
                    print(f"  Fold: {fold_name}")
                    for split_name, split_data in fold_data.items():
                        if split_name == 'metadata':  # Skip metadata entry
                            continue
                        n_graphs = split_data['n_graphs']
                        batch_size = split_data['batch_size']
                        print(f"    {split_name}: {n_graphs} graphs, batch size {batch_size}")
        
        self.dataloaders = dataloaders
        return dataloaders, split_index_dict
    
    def run_experiments(self) -> Dict[str, Any]:
        """Run experiments for all configured tasks and models."""
        print("\n" + "="*60)
        print("RUNNING INDUCTIVE EXPERIMENTS")
        print("="*60)
        
        if not hasattr(self, 'dataloaders'):
            raise ValueError("Must prepare data before running experiments")
        
        # Use parallel training if configured
        if self.config.use_parallel_training:
            from experiments.parallel_training import run_experiments_parallel
            
            if self.config.cross_task_parallelization:
                print("🔄 Using CROSS-TASK parallelization (all tasks and models in parallel)")
            else:
                print("🔄 Using PER-TASK parallelization (models within each task in parallel)")
            
            return run_experiments_parallel(self)
        
        # Original sequential training implementation
        all_results = {}
        
        # Process each task
        for task in self.config.tasks:
            print(f"\n{'='*40}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*40}")
            
            task_dataloaders = self.dataloaders[task]
            task_results = {}
            
            # Get dimensions from sample batch
            # Handle flattened structure: task_dataloaders[split_name] = DataLoader
            sample_batch = next(iter(task_dataloaders['train']))
            
            input_dim = sample_batch.x.shape[1]
            
            is_regression = task.startswith('k_hop_community_counts') or task == 'triangle_count'
            is_graph_level_task = task == 'triangle_count'
            
            # Determine output dimension
            if is_regression:
                output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
            else:
                from experiments.inductive.training import get_total_classes_from_dataloaders
                output_dim = get_total_classes_from_dataloaders(task_dataloaders)

            if self.config.differentiate_with_and_without_PE:
                PE_options = [True, False]
            else:
                PE_options = [True]
            
            # Train GNN models
            if self.config.run_gnn:
                for gnn_type in self.config.gnn_types:
                    for PE_option in PE_options:

                        from experiments.models import GNNModel
                        
                        model = GNNModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            gnn_type=gnn_type,
                            is_regression=is_regression,
                            is_graph_level_task=is_graph_level_task
                        )
                        
                        from experiments.inductive.training import train_and_evaluate_inductive
                        results = train_and_evaluate_inductive(
                            model=model,
                            model_name=gnn_type,
                            dataloaders=task_dataloaders,
                            config=self.config,
                            task=task,
                            device=self.device,
                            optimize_hyperparams=self.config.optimize_hyperparams,
                            experiment_name=getattr(self.config, 'experiment_name', None),
                            run_id=getattr(self.config, 'run_id', None),
                            PE_option=PE_option
                        )
                        
                        pe_model_name = gnn_type + "_PE" if PE_option else gnn_type
                        task_results[pe_model_name] = results
            
            # Train transformer models
            if self.config.run_transformers:
                for transformer_type in self.config.transformer_types:
                    from experiments.models import GraphTransformerModel
                    
                    
                    model = GraphTransformerModel(
                        input_dim=input_dim,
                        hidden_dim=self.config.hidden_dim,
                        output_dim=output_dim,
                        transformer_type=transformer_type,
                        num_layers=self.config.num_layers,
                        dropout=self.config.dropout,
                        is_regression=is_regression,
                        is_graph_level_task=is_graph_level_task,
                    )
                    
                    from experiments.inductive.training import train_and_evaluate_inductive
                    results = train_and_evaluate_inductive(
                        model=model,
                        model_name=transformer_type,
                        dataloaders=task_dataloaders,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams,
                        experiment_name=getattr(self.config, 'experiment_name', None),
                        run_id=getattr(self.config, 'run_id', None),
                        PE_option=True # Always add PE to transformer models
                    )
                    
                    task_results[transformer_type] = results
            
            # Train sheaf diffusion models
            if self.config.run_neural_sheaf:
                for PE_option in PE_options:
                    # from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import SheafDiffusionModel
                    from experiments.models import SheafDiffusionModel

                    model = SheafDiffusionModel(
                        input_dim=input_dim,
                        hidden_dim=self.config.hidden_dim,
                        output_dim=output_dim,
                        sheaf_type=self.config.sheaf_type,
                        d=self.config.sheaf_d,
                        num_layers=self.config.num_layers,
                        dropout=self.config.dropout,
                        is_regression=is_regression,
                        is_graph_level_task=is_graph_level_task)
                
                    from experiments.inductive.training import train_and_evaluate_inductive
                    results = train_and_evaluate_inductive(
                        model=model,
                        model_name='sheaf_diffusion',
                        dataloaders=task_dataloaders,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams,
                        experiment_name=getattr(self.config, 'experiment_name', None),
                        run_id=getattr(self.config, 'run_id', None),
                        PE_option=PE_option
                    )
                    
                    pe_model_name = "sheaf_diffusion_PE" if PE_option else "sheaf_diffusion"
                    task_results[pe_model_name] = results

            # Train MLP model
            if self.config.run_mlp:
                for PE_option in PE_options:
                    from experiments.models import MLPModel
                    
                    model = MLPModel(
                        input_dim=input_dim,
                        hidden_dim=self.config.hidden_dim,
                        output_dim=output_dim,
                        num_layers=self.config.num_layers,
                        dropout=self.config.dropout,
                        is_regression=is_regression,
                        is_graph_level_task=is_graph_level_task,
                    )
                    
                    from experiments.inductive.training import train_and_evaluate_inductive
                    results = train_and_evaluate_inductive(
                        model=model,
                        model_name='mlp',
                        dataloaders=task_dataloaders,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams,
                        experiment_name=getattr(self.config, 'experiment_name', None),
                        run_id=getattr(self.config, 'run_id', None),
                        PE_option=PE_option
                    )
                    
                    pe_model_name = "mlp_PE" if PE_option else 'mlp'
                    task_results[pe_model_name] = results
            
            all_results[task] = task_results
        
        return all_results
    
    def save_results(self) -> None:
        """Save results in clean format with two JSON files."""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # 1. Save clean configuration
        config_file = os.path.join(self.output_dir, "config.json")
        self.config.save(config_file)
        print(f"Configuration saved: {config_file}")
        
        # 2. Create comprehensive results JSON
        comprehensive_results = {
            # Experiment metadata
            'experiment_info': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'method': 'dccc_sbm' if self.config.use_dccc_sbm else 'dc_sbm',
                'degree_distribution': self.config.degree_distribution if self.config.use_dccc_sbm else 'standard',
                'n_graphs': self.config.n_graphs,
                'universe_K': self.config.universe_K
            },
            
            # Graph family properties
            'family_properties': self._get_family_properties(),
            
            # Family consistency metrics (NEW)
            'family_consistency': self.family_consistency or {},
            
            # Community signal metrics (NEW)
            'community_signals': self.graph_signals or {},
            
            # Model results
            'model_results': self._clean_model_results(),
            
            # Generation statistics
            'generation_stats': self._get_generation_stats()
        }

        # Save clean model results as self.clean_model_results
        self.clean_model_results = self._clean_model_results()
        
        # Save comprehensive results
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self._make_json_serializable(comprehensive_results), f, indent=2)
        
        print(f"Comprehensive results saved: {results_file}")
    
    def _get_family_properties(self) -> Dict[str, Any]:
        """Get family properties using existing analysis."""
        if self.family_graphs is None:
            return {}
        
        try:
            properties = analyze_graph_family_properties(self.family_graphs)
            return self._make_json_serializable(properties)
        except Exception as e:
            print(f"Warning: Failed to analyze family properties: {e}")
            return {}
    
    def _clean_model_results(self) -> Dict[str, Any]:
        """Clean and organize model results."""
        if not self.results:
            return {}
        
        clean_results = {}

        for task, task_results in self.results.items():
            clean_results[task] = {}

            for model_name, model_results in task_results.items():
                # Handle new repetition-based structure
                if 'repetition_test_metrics' in model_results:
                    # New structure with repetitions
                    repetition_test_metrics = model_results.get('repetition_test_metrics', {})
                    repetition_best_val_metrics = model_results.get('repetition_best_val_metrics', {})
                    repetition_train_time = model_results.get('repetition_train_time', {})
                    
                    # Aggregate test metrics across repetitions
                    test_metrics = {}
                    best_val_metrics = []
                    train_times = []
                    
                    for repetition_name, repetition_data in repetition_test_metrics.items():
                        for metric in repetition_data:
                            if metric not in test_metrics:
                                test_metrics[metric] = [repetition_data[metric]]
                            else:
                                test_metrics[metric].append(repetition_data[metric])
                    
                    # Collect best val metrics and train times
                    for repetition_name in repetition_best_val_metrics:
                        best_val_metrics.append(repetition_best_val_metrics[repetition_name])
                        train_times.append(repetition_train_time[repetition_name])
                    
                    success = True
                    
                elif 'fold_test_metrics' in model_results:
                    # Old fold-based structure (for backward compatibility)
                    model_fold_test_metrics = model_results.get('fold_test_metrics', {})
                    model_fold_best_val_metrics = model_results.get('fold_best_val_metrics', [])
                    model_fold_train_time = model_results.get('fold_train_time', 0.0)

                    test_metrics = {}
                    best_val_metrics = []
                    train_times = []

                    for fold_name, fold_data in model_fold_test_metrics.items():
                        for metric in fold_data:
                            if metric not in test_metrics:
                                test_metrics[metric] = [fold_data[metric]]
                            else:
                                test_metrics[metric].append(fold_data[metric])

                        best_val_metrics.append(model_fold_best_val_metrics[fold_name])
                        train_times.append(model_fold_train_time[fold_name])
                    
                    success = True
                    
                else:
                    # Fallback for other result structures
                    test_metrics = {}
                    best_val_metrics = []
                    train_times = []
                    success = False

                # Calculate final aggregated metrics
                final_test_metrics = {}
                for metric in test_metrics:
                    final_test_metrics[metric] = {
                        'mean': np.mean(test_metrics[metric]),
                        'std': np.std(test_metrics[metric]),
                        'list': test_metrics[metric]
                    }
                    
                # Same for best val metrics
                final_best_val_metrics = {
                    'mean': np.mean(best_val_metrics),
                    'std': np.std(best_val_metrics),
                    'list': best_val_metrics
                }
                    
                # Same for train time
                final_train_time = {
                    'mean': np.mean(train_times),
                    'std': np.std(train_times),
                    'list': train_times
                }

                clean_results[task][model_name] = {
                    'test_metrics': final_test_metrics,
                    'best_val_metrics': final_best_val_metrics,
                    'train_time': final_train_time,
                    'optimal_hyperparams': model_results.get('optimal_hyperparams', {}),
                    'success': success
                }
        
        return clean_results
    
    def _get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        if not hasattr(self, 'family_graphs') or not self.family_graphs:
            return {}
        
        # Get stats from family generator if available
        first_graph = self.family_graphs[0]
        if hasattr(first_graph, 'timing_info'):
            # Aggregate timing info
            timing_stats = {}
            for graph in self.family_graphs:
                if hasattr(graph, 'timing_info'):
                    for key, value in graph.timing_info.items():
                        if key not in timing_stats:
                            timing_stats[key] = []
                        timing_stats[key].append(value)
            
            # Calculate statistics
            aggregated_timing = {}
            for key, values in timing_stats.items():
                aggregated_timing[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            return {'timing_statistics': aggregated_timing}
        
        return {}
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            try:
                return self._make_json_serializable(obj.__dict__)
            except:
                return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj
    


    def validate_distributional_shifts(self) -> Dict[str, Any]:
        """Comprehensive validation of distributional shifts."""
        print("\n" + "="*60)
        print("VALIDATING DISTRIBUTIONAL SHIFTS")
        print("="*60)
        
        validation_results = {
            'property_shifts': {},
            'overall_status': 'PASS'
        }
        
        # Validate property shifts if applicable
        if (self.config.distributional_shift_in_eval and 
            self.config.distributional_shift_in_eval_type in ['homophily', 'density', 'n_nodes']):
            validation_results['property_shifts'] = self._validate_property_shifts()
        
        # Print summary
        self._print_validation_summary(validation_results)
        
        return validation_results
    

    
    def _validate_property_shifts(self) -> Dict[str, Any]:
        """Validate that property shifts are correctly applied."""
        print(f"\nValidating {self.config.distributional_shift_in_eval_type} shifts...")
        
        shift_type = self.config.distributional_shift_in_eval_type
        
        # Determine which graphs should have the shift
        if self.config.distributional_shift_test_only:
            # Test graphs should have shift
            if self.family_graphs_test_indices is not None:
                shifted_indices = self.family_graphs_test_indices
                baseline_indices = self.family_graphs_training_val_indices
            else:
                # Fallback: assume last portion is test
                n_graphs = len(self.family_graphs)
                test_size = int(n_graphs * self.config.test_graph_ratio)
                shifted_indices = list(range(n_graphs - test_size, n_graphs))
                baseline_indices = list(range(n_graphs - test_size))
        else:
            # Val and test graphs should have shift
            if self.family_graphs_val_test_indices is not None:
                shifted_indices = self.family_graphs_val_test_indices
                baseline_indices = self.family_graphs_training_indices
            else:
                # Fallback: assume second half has shift
                n_graphs = len(self.family_graphs)
                shifted_indices = list(range(n_graphs // 2, n_graphs))
                baseline_indices = list(range(n_graphs // 2))
        
        # Calculate property statistics
        baseline_stats = self._calculate_property_statistics(baseline_indices, shift_type)
        shifted_stats = self._calculate_property_statistics(shifted_indices, shift_type)
        
        # Validate the shift
        validation_result = {
            'shift_type': shift_type,
            'baseline_stats': baseline_stats,
            'shifted_stats': shifted_stats,
            'shift_magnitude': self._calculate_shift_magnitude(baseline_stats, shifted_stats, shift_type),
            'status': 'PASS'  # Will be updated based on validation logic
        }
        
        # Print statistics
        print(f"  Baseline {shift_type}: {baseline_stats['mean']:.3f} ± {baseline_stats['std']:.3f}")
        print(f"  Shifted {shift_type}: {shifted_stats['mean']:.3f} ± {shifted_stats['std']:.3f}")
        print(f"  Shift magnitude: {validation_result['shift_magnitude']:.3f}")
        
        # Validate that shift is significant
        if shift_type in ['homophily', 'density']:
            expected_shift = self.config.distributional_shift_in_eval_homophily_shift if shift_type == 'homophily' else self.config.distributional_shift_in_eval_density_shift
            actual_shift = abs(shifted_stats['mean'] - baseline_stats['mean'])
            if actual_shift < expected_shift * 0.5:  # Allow some tolerance
                validation_result['status'] = 'FAIL'
                print(f"  WARNING: Shift magnitude ({actual_shift:.3f}) is smaller than expected ({expected_shift:.3f})")
            else:
                print(f"  ✓ Shift magnitude is appropriate")
        
        elif shift_type == 'n_nodes':
            expected_shift = self.config.distributional_shift_in_eval_n_nodes_shift
            actual_shift = shifted_stats['mean'] - baseline_stats['mean']
            if abs(actual_shift) < expected_shift * 0.5:  # Allow some tolerance
                validation_result['status'] = 'FAIL'
                print(f"  WARNING: Node count shift ({actual_shift:.1f}) is smaller than expected ({expected_shift:.1f})")
            else:
                print(f"  ✓ Node count shift is appropriate")
        
        return validation_result
    
    def _calculate_property_statistics(self, indices: List[int], property_type: str) -> Dict[str, float]:
        """Calculate statistics for a specific property across a set of graphs."""
        if not indices or not self.family_graphs:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        values = []
        for idx in indices:
            if idx < len(self.family_graphs):
                graph = self.family_graphs[idx]
                
                if property_type == 'homophily':
                    # Calculate homophily from graph structure
                    homophily = self._calculate_graph_homophily(graph)
                    values.append(homophily)
                    
                elif property_type == 'density':
                    # Calculate edge density
                    n_edges = graph.graph.number_of_edges()
                    n_nodes = graph.graph.number_of_nodes()
                    density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
                    values.append(density)
                    
                elif property_type == 'n_nodes':
                    # Node count
                    values.append(graph.graph.number_of_nodes())
        
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    def _calculate_graph_homophily(self, graph_sample) -> float:
        """Calculate homophily for a single graph."""
        try:
            # Use community labels to calculate homophily
            if hasattr(graph_sample, 'community_labels') and graph_sample.community_labels is not None:
                labels = graph_sample.community_labels
                edges = list(graph_sample.graph.edges())
                
                if not edges:
                    return 0.0
                
                same_community_edges = 0
                for u, v in edges:
                    if labels[u] == labels[v]:
                        same_community_edges += 1
                
                return same_community_edges / len(edges)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_shift_magnitude(self, baseline_stats: Dict, shifted_stats: Dict, shift_type: str) -> float:
        """Calculate the magnitude of the property shift."""
        baseline_mean = baseline_stats['mean']
        shifted_mean = shifted_stats['mean']
        
        if shift_type in ['homophily', 'density']:
            return abs(shifted_mean - baseline_mean)
        elif shift_type == 'n_nodes':
            return shifted_mean - baseline_mean
        else:
            return 0.0
    
    def _print_validation_summary(self, validation_results: Dict[str, Any]):
        """Print a summary of validation results."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        overall_status = 'PASS'
        
        # Property shift validation
        if validation_results['property_shifts']:
            ps_result = validation_results['property_shifts']
            print(f"Property Shifts ({ps_result['shift_type']}): {ps_result['status']}")
            if ps_result['status'] == 'FAIL':
                overall_status = 'FAIL'
        
        print(f"\nOverall Validation Status: {overall_status}")
        
        if overall_status == 'PASS':
            print("✓ All distributional shifts are correctly implemented")
        else:
            print("✗ Some distributional shifts have issues")
        
        validation_results['overall_status'] = overall_status
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment pipeline."""
        try:
            print("Starting inductive learning experiment...")
            experiment_start = time.time()
            
            # Generate graph family
            self.family_graphs = self.generate_graph_family()
            
            # Validate distributional shifts
            validation_results = self.validate_distributional_shifts()
            
            # Analyze family consistency
            family_consistency = self.analyze_family_consistency()
            
            # Calculate community signals
            graph_signals = self.calculate_graph_signals()
            
            # Prepare data
            dataloaders, split_index_dict = self.prepare_data()
            
            # Run experiments
            results = self.run_experiments()
            
            # Store results in instance variable for saving and reporting
            self.results = results

            # Save self.inductive_data to file as pickle
            with open(os.path.join(self.output_dir, "inductive_data.pkl"), 'wb') as f:
                pickle.dump(self.inductive_data, f)
            
            # Save results
            self.save_results()
            
            # Generate summary report
            summary_report = self.generate_summary_report()
            
            with open(os.path.join(self.output_dir, "summary.txt"), 'w') as f:
                f.write(summary_report)
            
            print("\n" + summary_report)
            
            total_time = time.time() - experiment_start
            print(f"\nExperiment completed in {total_time:.2f} seconds")
            
            return {
                'output_dir': self.output_dir,
                'family_graphs': self.family_graphs,
                'validation_results': validation_results,
                'family_consistency': family_consistency,
                'graph_signals': graph_signals,
                'results': results,
                'config': self.config,
                'summary_report': summary_report,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Error in inductive experiment: {str(e)}", exc_info=True)
            
            # Save error information
            error_info = {
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'config': self.config.to_dict()
            }
            
            error_file = os.path.join(self.output_dir, "error.json")
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            raise
        finally:
            # Clean up GPU memory at the end
            if hasattr(self, 'dataloaders'):
                cleanup_gpu_dataloaders(self.dataloaders, self.device)
    
    def precompute_transformer_encodings(self):
        """Precompute encodings for transformer models if enabled."""
        if not self.config.run_transformers or not self.config.transformer_precompute_encodings:
            return
        
        if self.family_graphs is None:
            raise ValueError("Must generate graph family before precomputing encodings")
        
        print("\n" + "="*60)
        print("PRECOMPUTING TRANSFORMER ENCODINGS")
        print("="*60)
        
        from experiments.models import GraphTransformerModel
        
        # Create a temporary transformer model for precomputation
        sample_graph = self.family_graphs[0]
        input_dim = sample_graph.features.shape[1] if sample_graph.features is not None else 32
        
        for transformer_type in self.config.transformer_types:
            print(f"\nPrecomputing encodings for {transformer_type}...")
            
            temp_model = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.universe_K,
                transformer_type=transformer_type,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                precompute_encodings=True,
                cache_encodings=True
            )
            
            # Precompute for all graphs
            from experiments.models import precompute_family_encodings
            precompute_family_encodings(self.family_graphs, temp_model)
            
            # Store the cache globally for use during training
            if not hasattr(self, 'transformer_caches'):
                self.transformer_caches = {}
            self.transformer_caches[transformer_type] = temp_model._encoding_cache

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        if not self.results:
            return "No results available."
        
        lines = []
        lines.append("INDUCTIVE LEARNING EXPERIMENT SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Configuration summary
        lines.append("CONFIGURATION:")
        lines.append(f"  Method: {'DCCC-SBM' if self.config.use_dccc_sbm else 'DC-SBM'}")
        if self.config.use_dccc_sbm:
            lines.append(f"  Degree distribution: {self.config.degree_distribution}")
        lines.append(f"  Graph family size: {self.config.n_graphs}")
        lines.append(f"  Node range: [{self.config.min_n_nodes}, {self.config.max_n_nodes}]")
        lines.append(f"  Community range: [{self.config.min_communities}, {self.config.max_communities}]")
        lines.append(f"  Tasks: {', '.join(self.config.tasks)}")
        lines.append("")
        
        # Family consistency summary 
        if self.family_consistency:
            overall_score = self.family_consistency.get('overall', {}).get('score', 0.0)
            lines.append("FAMILY CONSISTENCY:")
            lines.append(f"  Overall score: {overall_score:.3f}")
            
            pattern_score = self.family_consistency.get('pattern_preservation', {}).get('score', 0.0)
            fidelity_score = self.family_consistency.get('generation_fidelity', {}).get('score', 0.0)
            degree_score = self.family_consistency.get('degree_consistency', {}).get('score', 0.0)
            triangle_score = self.family_consistency.get('triangle_consistency', {}).get('score', 0.0)
            cooccurrence_score = self.family_consistency.get('cooccurrence_consistency', {}).get('score', 0.0)
            
            lines.append(f"  Pattern preservation: {pattern_score:.3f}")
            lines.append(f"  Generation fidelity: {fidelity_score:.3f}")
            lines.append(f"  Degree consistency: {degree_score:.3f}")
            lines.append(f"  Triangle consistency: {triangle_score:.3f}")
            lines.append(f"  Co-occurrence consistency: {cooccurrence_score:.3f}")
            lines.append("")
        
        # Community signals summary 
        if self.graph_signals:
            lines.append("COMMUNITY SIGNALS (averaged over graphs):")
            
            degree_mean = self.graph_signals.get('degree_signals', {}).get('mean', 0.0)
            degree_std = self.graph_signals.get('degree_signals', {}).get('std', 0.0)
            lines.append(f"  Degree signal: {degree_mean:.3f} ± {degree_std:.3f}")
            
            structure_mean = self.graph_signals.get('structure_signals', {}).get('mean', 0.0)
            structure_std = self.graph_signals.get('structure_signals', {}).get('std', 0.0)
            lines.append(f"  Structure signal: {structure_mean:.3f} ± {structure_std:.3f}")
            
            if 'feature_signals' in self.graph_signals and self.graph_signals['feature_signals']['individual_values']:
                feature_mean = self.graph_signals.get('feature_signals', {}).get('mean', 0.0)
                feature_std = self.graph_signals.get('feature_signals', {}).get('std', 0.0)
                lines.append(f"  Feature signal: {feature_mean:.3f} ± {feature_std:.3f}")
            lines.append("")
        
        # Results summary using clean_model_results
        for task, task_results in self.clean_model_results.items():
            lines.append(f"TASK: {task.upper()}")
            lines.append("-" * 40)
            
            is_regression = task.startswith('k_hop_community_counts') or task == 'triangle_count'
            primary_metric = 'mae' if is_regression else 'f1_macro'
            
            best_score = 0.0
            best_model = None
            
            for model_name, model_results in task_results.items():
                test_metrics = model_results.get('test_metrics', {})
                train_time = model_results.get('train_time', {}).get('mean', 0.0)
                
                lines.append(f"  {model_name.upper()}:")
                
                # Display metrics using pre-calculated mean and std
                for metric_name, metric_values in test_metrics.items():
                    mean_val = metric_values.get('mean', 0.0)
                    std_val = metric_values.get('std', 0.0)
                    lines.append(f"    {metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f}")
                
                lines.append(f"    Training time: {train_time:.2f}s")
                
                # Update best score using mean value of primary metric
                if primary_metric in test_metrics:
                    score = test_metrics[primary_metric]['mean']
                    if is_regression and primary_metric in ['mae', 'mse']:
                        if score < best_score or best_score == 0.0:
                            best_score = score
                            best_model = model_name.upper()
                    else:
                        if score > best_score:
                            best_score = score
                            best_model = model_name.upper()
            
            if best_model:
                lines.append(f"  BEST: {best_model} ({primary_metric.upper()}: {best_score:.4f})")
            lines.append("")
        
        # Overall summary
        total_models = sum(len(task_results) for task_results in self.clean_model_results.values())
        
        successful_models = sum(
            sum(1 for model_results in task_results.values() 
                if model_results['success'])
            for task_results in self.clean_model_results.values()
        )
        
        lines.append("OVERALL:")
        lines.append(f"  Models trained: {total_models}")
        lines.append(f"  Successful: {successful_models}")
        lines.append(f"  Success rate: {successful_models/total_models:.1%}" if total_models > 0 else "  Success rate: 0%")
        lines.append(f"  Repetitions per model: {self.config.n_repetitions}")
        
        return "\n".join(lines)

    def load_pretrained_model(self, model_dir: str, model_id: str):
        """Load a pre-trained model for fine-tuning."""
        from experiments.inductive.data import PreTrainedModelSaver
        
        model_saver = PreTrainedModelSaver(model_dir)
        model, metadata = model_saver.load_model(model_id, self.device)
        
        print(f"Loaded pre-trained model: {model_id}")
        print(f"  Pre-training task: {metadata['config']['pretraining_task']}")
        print(f"  Architecture: {metadata['architecture']}")
        
        return model, metadata

    def create_model_from_pretrained(self, pretrained_model, metadata, output_dim: int, is_regression: bool, is_graph_level_task: bool):
        """Create a model for fine-tuning from a pre-trained model."""
        # Get the encoder from the pre-trained model
        if hasattr(pretrained_model, 'encoder'):
            encoder = pretrained_model.encoder
        else:
            raise ValueError("Pre-trained model does not have an encoder")
        
        # Create fine-tuning model
        from experiments.models import FineTuningModel
        
        return FineTuningModel(
            encoder=encoder,
            hidden_dim=metadata['architecture']['hidden_dim'],
            output_dim=output_dim,
            dropout=metadata['config']['dropout'],
            is_regression=is_regression,
            is_graph_level_task=is_graph_level_task,
            freeze_encoder=self.config.freeze_encoder
        )

class PreTrainingRunner:
    """Main runner for self-supervised pre-training with hyperparameter optimization."""
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model_saver = PreTrainedModelSaver(config.output_dir)
        self.family_manager = GraphFamilyManager(config)
        
        # Set random seeds
        self._set_seeds(config.seed)
    
    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        if self.config.force_cpu:
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            if self.config.device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{self.config.device_id}")
                torch.cuda.set_device(self.config.device_id)
                return device
        return torch.device("cpu")
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _generate_controlled_family(self, family_id: Optional[str] = None) -> Tuple[List, str]:
        """Generate graph family with full parameter control."""
        
        if family_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            family_id = f"enhanced_{self.config.gnn_type}_{self.config.pretraining_task}_{timestamp}"
        
        print(f"Generating graph family: {family_id}")
        print(f"Universe parameters:")
        print(f"  K: {self.config.universe_K}")
        print(f"  Feature dim: {self.config.universe_feature_dim}")
        print(f"  Edge density: {self.config.universe_edge_density}")
        print(f"  Homophily: {self.config.universe_homophily}")
        print(f"  DCCC-SBM: {self.config.use_dccc_sbm}")
        if self.config.use_dccc_sbm:
            print(f"  Degree distribution: {self.config.degree_distribution}")
        
        # Create universe with controlled parameters
        universe = GraphUniverse(**self.config.to_universe_params())
        
        # Create family generator with controlled parameters
        family_generator = GraphFamilyGenerator(
            universe=universe,
            **self.config.to_graph_family_params()
        )
        
        # Generate family
        print(f"Generating {self.config.get_total_graphs()} graphs total...")
        start_time = time.time()
        family_graphs = family_generator.generate_family(show_progress=False)
        generation_time = time.time() - start_time
        
        print(f"Family generation completed in {generation_time:.2f} seconds")
        print(f"Generated {len(family_graphs)} graphs")
        
        # Save the family if requested
        if self.config.save_graph_family:
            self._save_enhanced_family(family_graphs, family_id, family_generator)
        
        return family_graphs, family_id
    
    def _save_enhanced_family(self, family_graphs: List, family_id: str, family_generator) -> None:
        """Save graph family with enhanced metadata."""
        family_dir = os.path.join(self.config.graph_family_dir, family_id)
        os.makedirs(family_dir, exist_ok=True)
        
        print(f"Saving enhanced graph family to: {family_dir}")
        
        # Save graphs
        import pickle
        graphs_file = os.path.join(family_dir, "graphs.pkl")
        with open(graphs_file, 'wb') as f:
            pickle.dump(family_graphs, f)
        
        # Save enhanced metadata
        n_actual_pretraining, n_warmup, n_finetuning = self.config.get_graph_splits(len(family_graphs))
        
        metadata = {
            'family_id': family_id,
            'creation_timestamp': datetime.now().isoformat(),
            'total_graphs': len(family_graphs),
            'n_actual_pretraining': n_actual_pretraining,
            'n_warmup': n_warmup,
            'n_finetuning': n_finetuning,
            'config': self.config.to_dict(),
            'graphs_file': graphs_file,
            'splits': {
                'pretraining': list(range(n_actual_pretraining)),
                'warmup': list(range(n_actual_pretraining, n_actual_pretraining + n_warmup)),
                'finetuning': list(range(n_actual_pretraining + n_warmup, len(family_graphs)))
            },
            'generation_stats': family_generator.generation_stats,
            'family_summary': self._get_family_generation_summary(family_graphs)
        }
        
        metadata_file = os.path.join(family_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Saved {len(family_graphs)} graphs with enhanced metadata")
    
    def _get_family_generation_summary(self, family_graphs: List) -> Dict[str, Any]:
        """Get summary of graph family generation parameters."""
        if not family_graphs:
            return {}
        
        # Extract generation parameters from graphs
        generation_methods = [g.generation_method for g in family_graphs]
        node_counts = [g.n_nodes for g in family_graphs]
        edge_counts = [g.graph.number_of_edges() for g in family_graphs]
        community_counts = [len(g.communities) for g in family_graphs]
        
        summary = {
            'generation_methods': list(set(generation_methods)),
            'node_count_stats': {
                'mean': float(np.mean(node_counts)),
                'std': float(np.std(node_counts)),
                'min': int(np.min(node_counts)),
                'max': int(np.max(node_counts))
            },
            'edge_count_stats': {
                'mean': float(np.mean(edge_counts)),
                'std': float(np.std(edge_counts)),
                'min': int(np.min(edge_counts)),
                'max': int(np.max(edge_counts))
            },
            'community_count_stats': {
                'mean': float(np.mean(community_counts)),
                'std': float(np.std(community_counts)),
                'min': int(np.min(community_counts)),
                'max': int(np.max(community_counts))
            }
        }
        
        return summary

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for optimization INCLUDING task parameters."""
        params = {}
        
        # Common hyperparameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        params['hidden_dim'] = trial.suggest_int('hidden_dim', 32, 96, step=32)
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Model architecture parameters
        if self.config.model_type == "transformer":
            # Sample num_heads first, then ensure hidden_dim is divisible
            num_heads = trial.suggest_categorical('transformer_num_heads', [2, 4, 6, 8])
            while num_heads * params['num_layers'] > 14:
                num_heads = trial.suggest_categorical('transformer_num_heads', [2, 4, 6, 8])
            
            # Sample a base dimension and multiply by num_heads to ensure divisibility
            base_dim = trial.suggest_int('base_dim_multiplier', 8, 16, step=4)
            hidden_dim = base_dim * num_heads

            params['hidden_dim'] = hidden_dim
            params['transformer_num_heads'] = num_heads
            params['transformer_max_nodes'] = trial.suggest_int('transformer_max_nodes', 20, 80)
            params['transformer_max_path_length'] = trial.suggest_int('transformer_max_path_length', 3, 8)
            params['transformer_prenorm'] = trial.suggest_categorical('transformer_prenorm', [True, False])
            params['local_gnn_type'] = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
            params['global_model_type'] = trial.suggest_categorical('global_model_type', ['transformer', 'performer'])
        else:
            # GNN-specific parameters
            params['residual'] = trial.suggest_categorical('residual', [True, False])
            params['norm_type'] = trial.suggest_categorical('norm_type', ['none', 'layer'])
            params['agg_type'] = trial.suggest_categorical('agg_type', ['mean', 'sum'])

            if self.config.gnn_type == "gat":
                params['heads'] = trial.suggest_int('heads', 1, 8)
                # Make sure heads * num_layers is smaller than 16
                while params['heads'] * params['num_layers'] > 12:
                    params['heads'] = trial.suggest_int('heads', 1, 8)

                params['concat_heads'] = trial.suggest_categorical('concat_heads', [True, False])

            elif self.config.gnn_type == "fagcn":
                params['eps'] = trial.suggest_float('eps', 0.0, 1.0)
        
            elif self.config.gnn_type == "gin":
                params['eps'] = trial.suggest_float('eps', 0.0, 1.0)
        
        return params

    def _optimize_hyperparameters(self, warmup_graphs: List, task) -> Dict[str, Any]:
        """Optimize hyperparameters with proper evaluation."""
        
        print(f"Optimizing hyperparameters using {len(warmup_graphs)} warmup graphs")
        
        # Split warmup graphs into train/val for proper optimization
        val_size = max(1, len(warmup_graphs) // 4)  # 25% for validation
        train_graphs = warmup_graphs[:-val_size]
        val_graphs = warmup_graphs[-val_size:]
        
        print(f"  Train graphs: {len(train_graphs)}")
        print(f"  Val graphs: {len(val_graphs)}")
        
        # Create dataloaders
        train_loader = create_pretraining_dataloader(train_graphs, batch_size=32, shuffle=True)
        val_loader = create_pretraining_dataloader(val_graphs, batch_size=32, shuffle=False)
        
        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.x.shape[1]
        
        def objective(trial: optuna.Trial) -> float:
            """FIXED: Optuna objective function with proper SSL evaluation."""
            
            # Sample ALL hyperparameters including task-specific ones
            suggested_params = self._sample_hyperparameters(trial)
            
            # Create temporary config with suggested hyperparameters
            temp_config_dict = self.config.to_dict()
            temp_config_dict.update(suggested_params)
            
            temp_config = PreTrainingConfig.from_dict(temp_config_dict)
            
            # Use the GraphCL implementation if needed
            temp_task = create_ssl_task(temp_config)
            
            model = temp_task.create_model(input_dim).to(self.device)
            
            # Training with reduced epochs for speed
            max_epochs = min(50, self.config.epochs // 4)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=suggested_params['learning_rate'],
                weight_decay=suggested_params.get('weight_decay', 1e-5)
            )
            
            # Setup mixed precision training if enabled
            scaler = GradScaler() if temp_config.use_mixed_precision else None
            
            best_metric = float('-inf')
            patience_counter = 0
            patience = 10
            
            for epoch in range(max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    # Use mixed precision if enabled
                    if temp_config.use_mixed_precision:
                        with autocast():
                            loss = temp_task.compute_loss(model, batch)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = temp_task.compute_loss(model, batch)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
                print(f"\nTrial {trial.number} - Epoch {epoch}")
                print(f"Average training loss: {avg_train_loss}")
                
                # Validation every 5 epochs to save time
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        print(f"Computing validation metrics...")
                        val_metrics = temp_task.evaluate(model, val_loader)
                        print(f"Raw validation metrics: {val_metrics}")
                    
                    # Get primary metric based on task
                    if self.config.pretraining_task == "link_prediction":
                        metric = val_metrics.get('auc', 0.0)
                        print(f"Using AUC metric: {metric}")
                        if metric > best_metric:  # Higher is better for AUC
                            best_metric = metric
                            patience_counter = 0
                            print(f"New best metric: {best_metric}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                    elif self.config.pretraining_task in ["dgi"]:
                        metric = val_metrics.get('discrimination_score', 0.0)  # Use discrimination score instead of negative loss
                        print(f"Using discrimination score: {metric}")
                        if metric > best_metric:  # Higher is better for discrimination score
                            best_metric = metric
                            patience_counter = 0
                            print(f"New best metric: {best_metric}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                    elif self.config.pretraining_task in ["graphcl"]:
                        # Use alignment for contrastive tasks if accuracy is unreliable
                        metric = val_metrics.get('alignment', val_metrics.get('accuracy', 0.0))
                        print(f"Using alignment/accuracy metric: {metric}")
                        if metric > best_metric:  # Higher is better for alignment/accuracy
                            best_metric = metric
                            patience_counter = 0
                            print(f"New best metric: {best_metric}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                    elif self.config.pretraining_task in ["graphmae"]:
                        metric = val_metrics.get('cosine_similarity', 0.0)
                        print(f"Using cosine similarity metric: {metric}")
                        if metric > best_metric:  # Higher is better for cosine similarity
                            best_metric = metric
                            patience_counter = 0
                            print(f"New best metric: {best_metric}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                    else:
                        # Fallback to any available metric
                        metric = list(val_metrics.values())[0] if val_metrics else 0.0
                        print(f"Using fallback metric: {metric}")
                        if metric > best_metric:  # Higher is better for most metrics
                            best_metric = metric
                            patience_counter = 0
                            print(f"New best metric: {best_metric}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                        
                        if patience_counter >= patience:
                            print(f"Early stopping triggered after {epoch} epochs")
                            break
                
                print(f"\nTrial {trial.number} completed with final best metric: {best_metric}")
                return best_metric
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=5, n_ei_candidates=24),  # Faster convergence
            study_name=f"ssl_hyperopt_{self.config.pretraining_task}",
            pruner=HyperbandPruner(
                min_resource=3,      # Minimum epochs before pruning
                max_resource=15,     # Maximum epochs for any trial
                reduction_factor=3   # Aggressive pruning
            )
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.optimization_timeout,
            callbacks=[lambda study, trial: print(f"Trial {trial.number} finished with value: {trial.value}")]
        )
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial_number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in study.trials
            ]
        }

    def _train_model(self, pretraining_graphs: List, optimized_params: Optional[Dict] = None) -> Tuple[torch.nn.Module, Dict]:
        """Train a model on the pre-training task."""
        # Create task
        task = create_ssl_task(self.config)
        
        # Create model
        input_dim = pretraining_graphs[0].features.shape[1]
        model = task.create_model(input_dim)
        
        # Move to device
        model = model.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create dataloader
        dataloader = create_pretraining_dataloader(
            pretraining_graphs,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # Training loop
        best_loss = float('inf')
        best_model = None
        training_results = {}
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }
        
        for idx, epoch in enumerate(range(self.config.epochs)):
            # Training
            model.train()
            total_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                loss = task.compute_loss(model, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches if n_batches > 0 else 0.0
            training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            model.eval()
            val_metrics = task.evaluate(model, dataloader)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_metrics'].append(val_metrics)
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_model = copy.deepcopy(model)
                best_metrics = val_metrics
            
            if idx % 10 == 0:
                print(f"Epoch {idx} of {self.config.epochs} - Training Loss: {avg_train_loss} - Validation Loss: {val_metrics['loss']}")
                print(f"Best Validation Metrics: {best_metrics}")

        training_results['final_metrics'] = best_metrics
        training_results['training_history'] = training_history
        
        return best_model, training_results

    def run_pretraining_only(
        self, 
        family_id: Optional[str] = None,
        use_existing_family: bool = False
    ) -> Dict[str, Any]:
        """Run pre-training with proper optimization integration."""
        
        print("="*80)
        print("ENHANCED SSL PRE-TRAINING WITH OPTUNA OPTIMIZATION")
        print("="*80)
        
        pipeline_start = time.time()
        results = {'config': self.config.to_dict()}
        
        # Step 1: Generate or load graph family (unchanged)
        if use_existing_family and family_id:
            print(f"\nStep 1: Loading existing graph family")
            print("-" * 50)
            family_graphs, family_metadata = self.family_manager.load_family(family_id)
            graph_splits = self.family_manager.get_graph_splits(family_graphs, family_metadata)
            results['used_existing_family'] = True
            results['family_metadata'] = family_metadata
        else:
            print(f"\nStep 1: Generating new graph family")
            print("-" * 50)
            family_graphs, family_id = self._generate_controlled_family(family_id)
            
            # Create splits
            n_actual_pretraining, n_warmup, n_finetuning = self.config.get_graph_splits(len(family_graphs))
            graph_splits = {
                'pretraining': family_graphs[:n_actual_pretraining],
                'warmup': family_graphs[n_actual_pretraining:n_actual_pretraining + n_warmup],
                'finetuning': family_graphs[n_actual_pretraining + n_warmup:]
            }
            results['used_existing_family'] = False
        
        results['family_id'] = family_id
        results['graph_splits_sizes'] = {k: len(v) for k, v in graph_splits.items()}
        
        print(f"Graph splits: {results['graph_splits_sizes']}")
        
        # Step 2: Hyperparameter optimization
        hyperopt_results = None
        if self.config.optimize_hyperparams:
            print(f"\nStep 2: Hyperparameter Optimization")
            print("-" * 50)
            
            warmup_graphs = graph_splits['warmup']
            print(f"Using {len(warmup_graphs)} warmup graphs for hyperopt")
            
            # Create task for optimization (needed for interface compatibility)
            task = create_ssl_task(self.config)
            hyperopt_results = self._optimize_hyperparameters(warmup_graphs, task)
            results['hyperopt_results'] = hyperopt_results
            
            print(f"✅ Optimization completed")
            print(f"   Best value: {hyperopt_results['best_value']:.4f}")
            print(f"   Trials run: {hyperopt_results['n_trials']}")
        else:
            print("\nSkipping hyperparameter optimization")
        
        # Step 3: Pre-training with optimized parameters
        print(f"\nStep 3: Pre-training")
        print("-" * 50)
        
        pretraining_graphs = graph_splits['pretraining']
        print(f"Using {len(pretraining_graphs)} graphs for pre-training")
        
        optimized_params = None
        if hyperopt_results:
            optimized_params = hyperopt_results['best_params']
        
        model, training_results = self._train_model(pretraining_graphs, optimized_params)
        results.update(training_results)
        
        # Step 4: Save model (unchanged)
        print(f"\nStep 4: Saving pre-trained model")
        print("-" * 50)
        
        enhanced_metadata = {
            'family_id': family_id,
            'family_total_graphs': len(family_graphs),
            'finetuning_graphs_available': len(graph_splits['finetuning']),
            'pretraining_graphs_used': len(pretraining_graphs),
            'warmup_graphs_used': len(graph_splits['warmup']),
            'optimization_used': self.config.optimize_hyperparams,
            'graph_family_parameters': self._get_family_generation_summary(family_graphs)
        }
        
        if hyperopt_results:
            enhanced_metadata['optimization_summary'] = {
                'best_value': hyperopt_results['best_value'],
                'n_trials': hyperopt_results['n_trials'],
                'best_params': hyperopt_results['best_params']
            }
        
        model_id = self.model_saver.save_model(
            model=model,
            config=self.config,
            training_history=training_results['training_history'],
            metrics=training_results['final_metrics'],
            hyperopt_results=hyperopt_results,
            enhanced_metadata=enhanced_metadata
        )
        
        results['model_id'] = model_id
        
        # Final summary
        total_time = time.time() - pipeline_start
        results['total_time'] = total_time
        
        print(f"\n" + "="*80)
        print("ENHANCED PRE-TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Model ID: {model_id}")
        print(f"Graph family ID: {family_id}")
        print(f"Optimization used: {'Yes' if self.config.optimize_hyperparams else 'No'}")
        if hyperopt_results:
            print(f"Best optimization value: {hyperopt_results['best_value']:.4f}")
        print(f"Total time: {total_time:.2f} seconds")
        
        return results
    
def randomize_model_weights(model, initialization_scheme='xavier_uniform', verbose=True):
    """
    Properly randomize all model weights for fair from-scratch comparison.
    
    Args:
        model: PyTorch model to randomize
        initialization_scheme: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'
        verbose: Whether to print which layers are being reset
    """
    randomized_layers = []
    failed_layers = []
    
    for name, module in model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        try:
            # Handle different layer types explicitly
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Linear and convolutional layers
                if initialization_scheme == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif initialization_scheme == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif initialization_scheme == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif initialization_scheme == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                
                # Reset bias if it exists
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                randomized_layers.append(f"{name} ({type(module).__name__})")
                
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                # Normalization layers
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Reset running statistics for BatchNorm
                if hasattr(module, 'running_mean'):
                    module.running_mean.zero_()
                if hasattr(module, 'running_var'):
                    module.running_var.fill_(1)
                
                randomized_layers.append(f"{name} ({type(module).__name__})")
                
            elif isinstance(module, nn.Embedding):
                # Embedding layers
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
                randomized_layers.append(f"{name} ({type(module).__name__})")
                
            elif hasattr(module, 'reset_parameters'):
                # Fallback to reset_parameters if available
                module.reset_parameters()
                randomized_layers.append(f"{name} ({type(module).__name__}) - via reset_parameters")
                
            elif hasattr(module, 'weight'):
                # Generic weight tensor - use specified initialization
                if initialization_scheme == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif initialization_scheme == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                else:
                    # Fallback to normal initialization
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                randomized_layers.append(f"{name} ({type(module).__name__}) - generic weight")
                
        except Exception as e:
            failed_layers.append(f"{name} ({type(module).__name__}): {str(e)}")
    
    if verbose:
        print(f"Randomized {len(randomized_layers)} layers:")
        for layer in randomized_layers:
            print(f"  ✓ {layer}")
        
        if failed_layers:
            print(f"Failed to randomize {len(failed_layers)} layers:")
            for layer in failed_layers:
                print(f"  ✗ {layer}")
    
    return len(randomized_layers), len(failed_layers)

def calculate_silhouette_scores(model, task_dataloaders):
    """
    Calculate silhouette scores of communities of a model.
    
    Args:
        model: The pre-trained GNN model
        task_dataloaders: Dictionary of dataloaders for the task
        
    Returns:
        List of silhouette scores, one per graph in the training set
    """
    model.eval()
    silhouette_scores = []
    
    # Get training loader
    train_loader = task_dataloaders['train']
    
    with torch.no_grad():
        for batch in train_loader:
            # Move batch to same device as model
            device = next(model.parameters()).device
            batch = batch.to(device)
            
            # Get node embeddings
            if hasattr(model, 'encoder'):
                # For fine-tuning models
                embeddings = model.encoder(batch.x, batch.edge_index)
            else:
                # For regular models
                embeddings = model(batch.x, batch.edge_index)
            
            # Convert to numpy for sklearn
            embeddings_np = embeddings.cpu().numpy()
            
            # Get community labels
            community_labels = batch.y.cpu().numpy()
            
            # Calculate silhouette score
            try:
                score = silhouette_score(embeddings_np, community_labels)
                silhouette_scores.append(score)
            except Exception as e:
                print(f"Warning: Could not calculate silhouette score for a graph: {e}")
                silhouette_scores.append(0.0)
    
    return silhouette_scores

def calculate_tsne_coordinates(model, task_dataloaders, n_to_calculate=1):
    """
    Calculate TSNE coordinates of communities of a model.
    
    Args:
        model: The pre-trained GNN model
        task_dataloaders: Dictionary of dataloaders for the task
        n_to_calculate: Number of graphs to calculate TSNE for
        
    Returns:
        Dictionary containing:
        - 'coordinates': Dictionary mapping community IDs to their TSNE coordinates
        - 'silhouette_score': The silhouette score of the plotted graph
    """
    model.eval()
    results = {}
    
    # Get training loader
    train_loader = task_dataloaders['train']
    
    # Calculate silhouette scores first
    silhouette_scores = calculate_silhouette_scores(model, task_dataloaders)
    
    # Select graphs to plot (take first n_to_calculate)
    graphs_to_plot = min(n_to_calculate, len(silhouette_scores))
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= graphs_to_plot:
                break
                
            # Move batch to same device as model
            device = next(model.parameters()).device
            batch = batch.to(device)
                
            # Get node embeddings
            if hasattr(model, 'encoder'):
                # For fine-tuning models
                embeddings = model.encoder(batch.x, batch.edge_index)
            else:
                # For regular models
                embeddings = model(batch.x, batch.edge_index)
            
            # Convert to numpy for sklearn
            embeddings_np = embeddings.cpu().numpy()
            
            # Get community labels
            community_labels = batch.y.cpu().numpy()
            
            # Calculate TSNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_coords = tsne.fit_transform(embeddings_np)
            
            # Organize coordinates by community
            community_coords = {}
            for comm_id in np.unique(community_labels):
                mask = community_labels == comm_id
                community_coords[int(comm_id)] = tsne_coords[mask].tolist()
            
            # Store results
            results[f'graph_{i}'] = {
                'coordinates': community_coords,
                'silhouette_score': silhouette_scores[i]
            }
    
    return results
    
def list_graph_families(graph_family_dir: str = "graph_families") -> List[Dict]:
    """List available graph families."""
    config = PreTrainingConfig(graph_family_dir=graph_family_dir)
    manager = GraphFamilyManager(config)
    return manager.list_families()

def load_graphs_for_finetuning(family_id: str, graph_family_dir: str = "graph_families") -> Tuple[List, Dict]:
    """Load fine-tuning graphs from a saved family."""
    config = PreTrainingConfig(graph_family_dir=graph_family_dir)
    manager = GraphFamilyManager(config)
    
    family_graphs, metadata = manager.load_family(family_id)
    graph_splits = manager.get_graph_splits(family_graphs, metadata)
    
    return graph_splits['finetuning'], metadata

def run_inductive_experiment(config) -> Dict[str, Any]:
    """Convenience function to run an inductive experiment."""
    experiment = InductiveExperiment(config)
    return experiment.run()

def load_finetuning_graphs_from_model(
    config: InductiveExperimentConfig
) -> Optional[List]:
    """
    Load fine-tuning graphs based on config specification.
    Can use explicit family ID or auto-load from pre-trained model.
    """
    
    family_id = None
    
    # Method 1: Explicit family ID provided
    if config.graph_family_id:
        family_id = config.graph_family_id
        print(f"Using explicitly specified graph family: {family_id}")
    
    # Method 2: Auto-load family from pre-trained model
    elif config.auto_load_family and config.pretrained_model_id:
        try:
            from experiments.inductive.self_supervised_task import PreTrainedModelSaver
            
            model_saver = PreTrainedModelSaver(config.pretrained_model_dir)
            _, metadata, family_ref = model_saver.load_model(config.pretrained_model_id)
            
            family_id = family_ref.get('family_id')
            if family_id:
                print(f"Got graph family id from pre-trained model: {family_id}")
            else:
                print(f"Pre-trained model has no associated graph family")
                return None
                
        except Exception as e:
            print(f"Failed to auto-load graph family: {e}")
            return None
    
    # Method 3: No family specified - generate new graphs
    else:
        raise ValueError("No graph family specified, make sure to specify a graph family or use a pre-trained model with auto-load_family set to True")
    
    # Load the graph family
    if family_id:
        try:
            from experiments.inductive.data import GraphFamilyManager
            from experiments.inductive.config import PreTrainingConfig
            
            # Create temporary config for family loading
            temp_config = PreTrainingConfig(graph_family_dir=config.graph_family_dir)
            family_manager = GraphFamilyManager(temp_config)
            
            family_graphs, family_metadata = family_manager.load_family(family_id)
            graph_splits = family_manager.get_graph_splits(family_graphs, family_metadata)
            
            # Use fine-tuning graphs
            finetuning_graphs = graph_splits['finetuning']
            
            print(f"Loaded {len(finetuning_graphs)} fine-tuning graphs from family {family_id}")
            print(f"Total family size: {len(family_graphs)} graphs")
            print(f"Family metadata: {family_metadata.get('creation_timestamp', 'Unknown')}")
            
            return finetuning_graphs
            
        except Exception as e:
            print(f"Failed to load graph family {family_id}: {e}")
            return None
    
    return None

def test_distributional_shifts():
    """Legacy test function - use test_distributional_shifts_detailed() in test_distributional_shifts.py instead."""
    print("This function is deprecated. Use test_distributional_shifts_detailed() from test_distributional_shifts.py instead.")
    return {}



def test_property_shifts():
    """Specific test for property shifts."""
    print("="*80)
    print("TESTING PROPERTY SHIFTS")
    print("="*80)
    
    from experiments.inductive.config import InductiveExperimentConfig
    
    # Test different property shifts
    property_tests = [
        {
            'name': 'Homophily Shift',
            'shift_type': 'homophily',
            'shift_amount': 0.2
        },
        {
            'name': 'Density Shift',
            'shift_type': 'density',
            'shift_amount': 0.2
        },
        {
            'name': 'Node Count Shift',
            'shift_type': 'n_nodes',
            'shift_amount': 50
        }
    ]
    
    results = {}
    
    for test in property_tests:
        print(f"\nTesting {test['name']}...")
        
        # Create config
        config = InductiveExperimentConfig(
            # Basic settings
            n_graphs=100,
            train_graph_ratio=0.6,
            val_graph_ratio=0.2,
            test_graph_ratio=0.2,
            n_repetitions=3,
            
            # Property shift
            distributional_shift_in_eval=True,
            distributional_shift_in_eval_type=test['shift_type'],
            distributional_shift_test_only=True,
            
            # Shift amounts
            distributional_shift_in_eval_homophily_shift=test['shift_amount'] if test['shift_type'] == 'homophily' else 0.2,
            distributional_shift_in_eval_density_shift=test['shift_amount'] if test['shift_type'] == 'density' else 0.1,
            distributional_shift_in_eval_n_nodes_shift=test['shift_amount'] if test['shift_type'] == 'n_nodes' else 20,
            
            # Graph generation settings
            universe_K=10,
            universe_feature_dim=16,
            universe_edge_density=0.2,
            universe_homophily=0.3,
            min_n_nodes=70,
            max_n_nodes=120,
            min_communities=4,
            max_communities=7,
            
            # Tasks
            tasks=['community'],
            
            # Other settings
            seed=42,
            output_dir=f'test_property_{test["shift_type"]}_output',
            force_cpu=True
        )
        
        # Create experiment
        experiment = InductiveExperiment(config)
        
        # Generate graph family
        family_graphs = experiment.generate_graph_family()
        
        # Validate property shifts
        validation_results = experiment.validate_distributional_shifts()
        
        # Check if property shift is significant
        if validation_results['property_shifts']:
            property_result = validation_results['property_shifts']
            shift_magnitude = property_result['shift_magnitude']
            expected_shift = test['shift_amount']
            
            # Check if shift is significant (at least 50% of expected)
            if shift_magnitude >= expected_shift * 0.5:
                print(f"✓ {test['name']}: Shift magnitude {shift_magnitude:.3f} is appropriate")
                results[test['name']] = {'success': True, 'shift_magnitude': shift_magnitude}
            else:
                print(f"✗ {test['name']}: Shift magnitude {shift_magnitude:.3f} is too small")
                results[test['name']] = {'success': False, 'shift_magnitude': shift_magnitude}
        else:
            print(f"✗ {test['name']}: No property shift validation results")
            results[test['name']] = {'success': False, 'error': 'No validation results'}
    
    # Print summary
    print(f"\nProperty Shift Test Summary:")
    passed_tests = sum(1 for result in results.values() if result['success'])
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} property shift tests passed")
    
    return results

