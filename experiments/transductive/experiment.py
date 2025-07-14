"""
Clean transductive experiment orchestration.
Based on inductive experiment but adapted for single-graph transductive learning.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import numpy as np
import torch

from graph_universe.model import GraphUniverse
from graph_universe.graph_family import GraphFamilyGenerator
from experiments.transductive.data import (
    prepare_transductive_data,
    analyze_graph_properties,
    validate_transductive_data
)
from experiments.transductive.training import train_and_evaluate_transductive
from experiments.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel
from experiments.transductive.config import TransductiveExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransductiveExperiment:
    """Clean transductive learning experiment runner."""
    
    def __init__(self, config: TransductiveExperimentConfig):
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
        self.output_dir = os.path.join(config.output_dir, f"transductive_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.results = {}
        self.graph_sample = None
        self.graph_properties = None
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
    
    def generate_graph(self) -> Any:
        """Generate single graph for transductive learning."""
        print("\n" + "="*60)
        print("GENERATING SINGLE GRAPH FOR TRANSDUCTIVE LEARNING")
        print("="*60)
        
        # Create universe with clean parameters
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
            seed=self.config.seed
        )
        
        # Create family generator for single graph
        print("Setting up graph generator...")
        family_generator = GraphFamilyGenerator(
            universe=universe,
            n_graphs=1,  # Only generate one graph
            min_n_nodes=self.config.num_nodes,
            max_n_nodes=self.config.num_nodes,  # Fixed size
            min_communities=self.config.num_communities,
            max_communities=self.config.num_communities,  # Fixed communities
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
        
        # Generate single graph
        print(f"Generating single graph with {self.config.num_nodes} nodes and {self.config.num_communities} communities...")
        start_time = time.time()
        family_graphs = family_generator.generate_family(show_progress=False)
        generation_time = time.time() - start_time
        
        if not family_graphs:
            raise ValueError("Failed to generate graph")
        
        graph_sample = family_graphs[0]
        
        print(f"Graph generation completed in {generation_time:.2f} seconds")
        print(f"Successfully generated graph with {graph_sample.n_nodes} nodes and {len(np.unique(graph_sample.community_labels))} communities")
        
        self.graph_sample = graph_sample
        return graph_sample
    
    def calculate_graph_signals(self) -> Dict[str, Any]:
        """Calculate community signals for the graph."""
        if self.graph_sample is None:
            raise ValueError("Must generate graph first")
        
        if not self.config.collect_signal_metrics:
            return {}
        
        print("\nCalculating community signals...")
        
        try:
            # Calculate community signals using existing methods
            signals = self.graph_sample.calculate_community_signals(
                structure_metric='kl',
                degree_method='naive_bayes',
                degree_metric='accuracy',
                random_state=self.config.seed
            )
            
            # Store signals
            signal_results = {
                'degree_signal': signals.get('degree_signal', 0.0),
                'structure_signal': signals.get('mean_structure_signal', 0.0),
                'feature_signal': signals.get('feature_signal')
            }
            
            # Remove None values
            signal_results = {k: v for k, v in signal_results.items() if v is not None}
            
            self.graph_signals = signal_results
            return signal_results
            
        except Exception as e:
            print(f"Warning: Failed to calculate signals: {e}")
            return {}
    
    def prepare_data(self) -> Dict[str, Dict[str, Any]]:
        """Prepare data for transductive learning."""
        print("\n" + "="*60)
        print("PREPARING TRANSDUCTIVE DATA")
        print("="*60)
        
        if self.graph_sample is None:
            raise ValueError("Must generate graph before preparing data")
        
        # Prepare transductive data
        transductive_data = prepare_transductive_data(self.graph_sample, self.config)
        
        # Validate data
        validation_results = validate_transductive_data(transductive_data, self.config)
        if not validation_results['valid']:
            for issue in validation_results['issues']:
                print(f"Warning: {issue}")
        
        # Print split information
        for task in self.config.tasks:
            if task in transductive_data:
                task_data = transductive_data[task]
                n_train = len(task_data['train_idx'])
                n_val = len(task_data['val_idx'])
                n_test = len(task_data['test_idx'])
                print(f"\nTask: {task}")
                print(f"  Train nodes: {n_train}")
                print(f"  Val nodes: {n_val}")
                print(f"  Test nodes: {n_test}")
        
        self.transductive_data = transductive_data
        return transductive_data
    
    def run_experiments(self) -> Dict[str, Any]:
        """Run transductive experiments on single graph."""
        print("\n" + "="*60)
        print("RUNNING TRANSDUCTIVE EXPERIMENTS")
        print("="*60)
        
        if not hasattr(self, 'transductive_data'):
            raise ValueError("Must prepare data before running experiments")
        
        all_results = {}
        
        for task in self.config.tasks:
            print(f"\n{'='*40}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*40}")
            
            task_results = {}
            task_data = self.transductive_data[task]
            is_regression = task == 'k_hop_community_counts'
            
            # Get dimensions
            input_dim = task_data['features'].shape[1]
            output_dim = task_data['metadata']['output_dim']
            
            print(f"Model configuration:")
            print(f"  Input dim: {input_dim}")
            print(f"  Output dim: {output_dim}")
            print(f"  Is regression: {is_regression}")
            
            # Determine models to run
            models_to_run = []
            if self.config.run_gnn:
                models_to_run.extend(self.config.gnn_types)
            if self.config.run_transformers:
                models_to_run.extend(self.config.transformer_types)
            if self.config.run_mlp:
                models_to_run.append('mlp')
            if self.config.run_rf:
                models_to_run.append('rf')
            
            # Train each model
            for model_name in models_to_run:
                print(f"\n--- Training {model_name.upper()} ---")
                
                try:
                    # Create model
                    if model_name in self.config.gnn_types:
                        if model_name == 'fagcn':
                            model = GNNModel(
                                input_dim=input_dim,
                                hidden_dim=self.config.hidden_dim,
                                output_dim=output_dim,
                                num_layers=self.config.num_layers,
                                dropout=self.config.dropout,
                                gnn_type=model_name,
                                is_regression=is_regression,
                                eps=self.config.eps
                            )
                        else:
                            model = GNNModel(
                                input_dim=input_dim,
                                hidden_dim=self.config.hidden_dim,
                                output_dim=output_dim,
                                num_layers=self.config.num_layers,
                                dropout=self.config.dropout,
                                gnn_type=model_name,
                                is_regression=is_regression
                            )
                    
                    elif model_name in self.config.transformer_types:
                        model = GraphTransformerModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,
                            transformer_type=model_name,
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            is_regression=is_regression,
                            num_heads=self.config.transformer_num_heads,
                            max_nodes=self.config.transformer_max_nodes,
                            max_path_length=self.config.transformer_max_path_length,
                            precompute_encodings=self.config.transformer_precompute_encodings,
                            cache_encodings=self.config.transformer_cache_encodings,
                            local_gnn_type=self.config.local_gnn_type,
                            global_model_type=self.config.global_model_type,
                            prenorm=self.config.transformer_prenorm
                        )
                    
                    elif model_name == 'mlp':
                        model = MLPModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            is_regression=is_regression
                        )
                    
                    elif model_name == 'rf':
                        model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            is_regression=is_regression
                        )
                    
                    # Train model
                    results = train_and_evaluate_transductive(
                        model=model,
                        task_data=task_data,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams
                    )
                    
                    # Store results
                    task_results[model_name] = {
                        'test_metrics': results.get('test_metrics', {}),
                        'train_time': results.get('train_time', 0.0),
                        'training_history': results.get('training_history', {}),
                        'optimal_hyperparams': results.get('optimal_hyperparams', {})
                    }
                    
                    print(f"✓ {model_name.upper()} completed successfully")
                
                except Exception as e:
                    error_msg = f"Error in {model_name} model: {str(e)}"
                    print(f"✗ {error_msg}")
                    logger.error(error_msg, exc_info=True)
                    
                    task_results[model_name] = {
                        'error': error_msg,
                        'test_metrics': {}
                    }
                
            all_results[task] = task_results
        
        self.results = all_results
        return all_results
    
    def save_results(self) -> None:
        """Save results in clean format."""
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
                'num_nodes': self.config.num_nodes,
                'num_communities': self.config.num_communities,
                'universe_K': self.config.universe_K,
                'learning_type': 'transductive'
            },
            
            # Graph properties
            'graph_properties': self._get_graph_properties(),
            
            # Community signal metrics
            'community_signals': self.graph_signals or {},
            
            # Model results
            'model_results': self._clean_model_results(),
            
            # Split information
            'data_splits': self._get_split_info()
        }
        
        # Save comprehensive results
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self._make_json_serializable(comprehensive_results), f, indent=2)
        
        print(f"Comprehensive results saved: {results_file}")
    
    def _get_graph_properties(self) -> Dict[str, Any]:
        """Get graph properties."""
        if self.graph_sample is None:
            return {}
        
        try:
            properties = analyze_graph_properties(self.graph_sample)
            return self._make_json_serializable(properties)
        except Exception as e:
            print(f"Warning: Failed to analyze graph properties: {e}")
            return {}
    
    def _get_split_info(self) -> Dict[str, Any]:
        """Get information about data splits."""
        if not hasattr(self, 'transductive_data'):
            return {}
        
        split_info = {}
        for task in self.config.tasks:
            if task in self.transductive_data:
                task_data = self.transductive_data[task]
                split_info[task] = {
                    'train_nodes': len(task_data['train_idx']),
                    'val_nodes': len(task_data['val_idx']),
                    'test_nodes': len(task_data['test_idx']),
                    'total_nodes': task_data['num_nodes']
                }
        
        return split_info
    
    def _clean_model_results(self) -> Dict[str, Any]:
        """Clean and organize model results."""
        if not self.results:
            return {}
        
        clean_results = {}
        for task, task_results in self.results.items():
            clean_results[task] = {}
            for model_name, model_results in task_results.items():
                clean_results[task][model_name] = {
                    'test_metrics': model_results.get('test_metrics', {}),
                    'train_time': model_results.get('train_time', 0.0),
                    'error': model_results.get('error'),
                    'training_history': model_results.get('training_history', {}),
                    'optimal_hyperparams': model_results.get('optimal_hyperparams', {})
                }
        
        return clean_results
    
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
    
    def run(self) -> Dict[str, Any]:
        """Run complete transductive learning experiment pipeline."""
        try:
            print("Starting transductive learning experiment...")
            experiment_start = time.time()
            
            # Generate single graph
            graph_sample = self.generate_graph()
            
            # Calculate community signals
            graph_signals = self.calculate_graph_signals()
            
            # Prepare data
            transductive_data = self.prepare_data()
            
            # Run experiments
            results = self.run_experiments()
            
            # Save results
            self.save_results()
            
            # Generate summary report
            summary_report = self.generate_summary_report()
            
            with open(os.path.join(self.output_dir, "summary.txt"), 'w') as f:
                f.write(summary_report)
            
            print("\n" + summary_report)
            
            total_time = time.time() - experiment_start
            print(f"\nExperiment completed in {total_time:.2f} seconds")
            
            # Only return serializable data
            return {
                'graph_properties': self._get_graph_properties(),
                'graph_signals': graph_signals,
                'results': self._clean_model_results(),
                'split_info': self._get_split_info(),
                'config': self.config.to_dict(),
                'summary_report': summary_report,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Error in transductive experiment: {str(e)}", exc_info=True)
            
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
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        if not self.results:
            return "No results available."
        
        lines = []
        lines.append("TRANSDUCTIVE LEARNING EXPERIMENT SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Configuration summary
        lines.append("CONFIGURATION:")
        lines.append(f"  Method: {'DCCC-SBM' if self.config.use_dccc_sbm else 'DC-SBM'}")
        if self.config.use_dccc_sbm:
            lines.append(f"  Degree distribution: {self.config.degree_distribution}")
        lines.append(f"  Graph size: {self.config.num_nodes} nodes")
        lines.append(f"  Communities: {self.config.num_communities}")
        lines.append(f"  Tasks: {', '.join(self.config.tasks)}")
        lines.append("")
        
        # Graph properties summary
        if self.graph_sample:
            lines.append("GRAPH PROPERTIES:")
            properties = analyze_graph_properties(self.graph_sample)
            lines.append(f"  Nodes: {properties.get('n_nodes', 'N/A')}")
            lines.append(f"  Edges: {properties.get('n_edges', 'N/A')}")
            lines.append(f"  Density: {properties.get('density', 0.0):.4f}")
            lines.append(f"  Avg degree: {properties.get('avg_degree', 0.0):.2f}")
            lines.append(f"  Clustering: {properties.get('clustering_coefficient', 0.0):.4f}")
            lines.append("")
        
        # Community signals summary
        if self.graph_signals:
            lines.append("COMMUNITY SIGNALS:")
            degree_signal = self.graph_signals.get('degree_signal', 0.0)
            structure_signal = self.graph_signals.get('structure_signal', 0.0)
            lines.append(f"  Degree signal: {degree_signal:.3f}")
            lines.append(f"  Structure signal: {structure_signal:.3f}")
            
            if 'feature_signal' in self.graph_signals:
                feature_signal = self.graph_signals.get('feature_signal', 0.0)
                lines.append(f"  Feature signal: {feature_signal:.3f}")
            lines.append("")
        
        # Results summary
        for task, task_results in self.results.items():
            lines.append(f"TASK: {task.upper()}")
            lines.append("-" * 40)
            
            is_regression = self.config.is_regression.get(task, False)
            primary_metric = 'r2' if is_regression else 'f1_macro'
            
            best_score = float('-inf') if is_regression else 0.0
            best_model = None
            
            for model_name, model_results in task_results.items():
                if 'test_metrics' in model_results and model_results['test_metrics']:
                    test_metrics = model_results['test_metrics']
                    score = test_metrics.get(primary_metric, 0.0)
                    train_time = model_results.get('train_time', 0.0)
                    
                    lines.append(f"  {model_name.upper()}:")
                    lines.append(f"    {primary_metric.upper()}: {score:.4f}")
                    lines.append(f"    Training time: {train_time:.2f}s")
                    
                    if (is_regression and score > best_score) or (not is_regression and score > best_score):
                        best_score = score
                        best_model = model_name.upper()
                else:
                    lines.append(f"  {model_name.upper()}: Failed")
            
            if best_model:
                lines.append(f"  BEST: {best_model} ({primary_metric.upper()}: {best_score:.4f})")
            lines.append("")
        
        # Overall summary
        total_models = sum(len(task_results) for task_results in self.results.values())
        successful_models = sum(
            sum(1 for model_results in task_results.values() 
                if 'test_metrics' in model_results and model_results['test_metrics'])
            for task_results in self.results.values()
        )
        
        lines.append("OVERALL:")
        lines.append(f"  Models trained: {total_models}")
        lines.append(f"  Successful: {successful_models}")
        lines.append(f"  Success rate: {successful_models/total_models:.1%}" if total_models > 0 else "  Success rate: 0%")
        
        return "\n".join(lines)


def run_transductive_experiment(config: TransductiveExperimentConfig) -> Dict[str, Any]:
    """Convenience function to run a transductive experiment."""
    experiment = TransductiveExperiment(config)
    return experiment.run()