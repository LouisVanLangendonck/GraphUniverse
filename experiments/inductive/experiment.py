"""
Inductive experiment orchestration for graph families.

This module provides the main experiment class for running inductive learning
experiments on graph families.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import torch
import numpy as np

from mmsb.model import GraphUniverse
from mmsb.graph_family import GraphFamilyGenerator, FamilyConsistencyAnalyzer
from experiments.inductive.config import InductiveExperimentConfig
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders,
    analyze_graph_family_properties,
    prepare_mixed_inductive_data,
    split_graphs
)
from experiments.inductive.training import train_and_evaluate_inductive, get_total_classes_from_dataloaders
from experiments.core.models import GNNModel, MLPModel, SklearnModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InductiveExperiment:
    """
    Main class for running inductive learning experiments on graph families.
    """
    
    def __init__(self, config: InductiveExperimentConfig):
        """
        Initialize the inductive experiment.
        
        Args:
            config: Experiment configuration
        """
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
        
        # Save config
        config.save(os.path.join(self.output_dir, "config.json"))
        
        # Initialize storage for results
        self.results = {}
        self.family_graphs = None
        self.family_stats = None
        
        print(f"Inductive experiment initialized. Output directory: {self.output_dir}")
        print(f"Using device: {self.device}")
    
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
        """
        Generate a family of graphs for inductive learning.
        
        Returns:
            List of GraphSample objects
        """
        print("\n" + "="*60)
        print("GENERATING GRAPH FAMILY")
        print("="*60)
        
        # Create universe
        print("Creating graph universe...")
        universe = GraphUniverse(
            K=self.config.universe_K,
            feature_dim=self.config.universe_feature_dim,
            block_structure=self.config.universe_block_structure,
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
        
        # Create family generator
        print("Setting up graph family generator...")
        family_generator = GraphFamilyGenerator(
            universe=universe,
            n_graphs=self.config.n_graphs,
            min_n_nodes=self.config.min_n_nodes,
            max_n_nodes=self.config.max_n_nodes,
            min_communities=self.config.min_communities,
            max_communities=self.config.max_communities,
            min_component_size=self.config.min_component_size,
            feature_regime_balance=self.config.feature_regime_balance,
            homophily_range=self.config.homophily_range,
            density_range=self.config.density_range,
            use_dccc_sbm=self.config.use_dccc_sbm,
            community_imbalance_range=self.config.community_imbalance_range,
            degree_separation_range=self.config.degree_separation_range,
            degree_method="standard",
            disable_deviation_limiting=self.config.disable_deviation_limiting,
            max_mean_community_deviation=self.config.max_mean_community_deviation,
            max_max_community_deviation=self.config.max_max_community_deviation,
            min_edge_density=self.config.min_edge_density,
            degree_distribution=self.config.degree_distribution,
            power_law_exponent_range=self.config.power_law_exponent_range,
            exponential_rate_range=self.config.exponential_rate_range,
            uniform_min_factor_range=self.config.uniform_min_factor_range,
            uniform_max_factor_range=self.config.uniform_max_factor_range,
            degree_heterogeneity=self.config.degree_heterogeneity,
            edge_noise=self.config.edge_noise,
            target_avg_degree=self.config.target_avg_degree,
            triangle_enhancement=self.config.triangle_enhancement,
            max_parameter_search_attempts=self.config.max_parameter_search_attempts,
            parameter_search_range=self.config.parameter_search_range,
            max_retries=self.config.max_retries,
            seed=self.config.seed
        )
        
        # Generate family
        print(f"Generating family of {self.config.n_graphs} graphs...")
        start_time = time.time()
        family_graphs = family_generator.generate_family(
            show_progress=True,
            collect_stats=self.config.collect_family_stats
        )
        generation_time = time.time() - start_time
        
        print(f"Family generation completed in {generation_time:.2f} seconds")
        print(f"Successfully generated {len(family_graphs)} graphs")
        
        # Analyze family consistency if required
        if self.config.require_consistency_check:
            print("\nAnalyzing family consistency...")
            consistency_analyzer = FamilyConsistencyAnalyzer(family_graphs, universe)
            consistency_results = consistency_analyzer.analyze_consistency()
            
            overall_consistency = consistency_results.get('overall', {}).get('score', 0.0)
            print(f"Family consistency score: {overall_consistency:.3f}")
            
            if overall_consistency < self.config.min_family_consistency:
                raise ValueError(
                    f"Family consistency {overall_consistency:.3f} below required minimum "
                    f"{self.config.min_family_consistency:.3f}"
                )
            
            # Save consistency results
            with open(os.path.join(self.output_dir, "family_consistency.json"), 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = self._make_json_serializable(consistency_results)
                json.dump(serializable_results, f, indent=2)
        
        # Analyze family properties
        if self.config.collect_family_stats:
            print("\nAnalyzing family properties...")
            family_properties = analyze_graph_family_properties(family_graphs)
            
            with open(os.path.join(self.output_dir, "family_properties.json"), 'w') as f:
                json.dump(family_properties, f, indent=2)
            
            print(f"Family statistics:")
            print(f"  Node count range: [{family_properties['node_counts_min']}, {family_properties['node_counts_max']}]")
            print(f"  Average degree range: [{family_properties['avg_degrees_min']:.2f}, {family_properties['avg_degrees_max']:.2f}]")
            print(f"  Density range: [{family_properties['densities_min']:.4f}, {family_properties['densities_max']:.4f}]")
        
        self.family_graphs = family_graphs
        self.family_stats = family_generator.generation_stats
        
        return family_graphs
    
    def prepare_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Prepare data for inductive learning.
        
        Returns:
            Dictionary containing prepared data for all tasks
        """
        print("\n" + "="*60)
        print("PREPARING INDUCTIVE DATA")
        print("="*60)
        
        if self.family_graphs is None:
            raise ValueError("Must generate graph family before preparing data")
        
        # Choose data preparation method based on inductive mode
        if self.config.inductive_mode == "graph_level":
            print("Using graph-level inductive split...")
            inductive_data = prepare_inductive_data(self.family_graphs, self.config)
        elif self.config.inductive_mode == "mixed":
            print("Using mixed inductive split...")
            inductive_data = prepare_mixed_inductive_data(self.family_graphs, self.config)
        else:
            raise ValueError(f"Unknown inductive mode: {self.config.inductive_mode}")
        
        # Create dataloaders
        print("Creating dataloaders...")
        dataloaders = create_inductive_dataloaders(inductive_data, self.config)
        
        # Print data split information
        for task in self.config.tasks:
            print(f"\nTask: {task}")
            for split in ['train', 'val', 'test']:
                n_graphs = inductive_data[task][split]['n_graphs']
                batch_size = inductive_data[task][split]['batch_size']
                print(f"  {split}: {n_graphs} graphs, batch size {batch_size}")
        
        self.dataloaders = dataloaders
        return dataloaders
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run inductive learning experiments for all tasks and models.
        
        Returns:
            Dictionary containing all experiment results
        """
        print("\n" + "="*60)
        print("RUNNING INDUCTIVE EXPERIMENTS")
        print("="*60)
        
        if not hasattr(self, 'dataloaders'):
            raise ValueError("Must prepare data before running experiments")
        
        all_results = {}
        
        # Run experiments for each task
        for task in self.config.tasks:
            print(f"\n{'='*40}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*40}")
            
            task_results = {}
            task_dataloaders = self.dataloaders[task]
            is_regression = self.config.is_regression.get(task, False)
            
            # Get dimensions for model creation
            sample_batch = next(iter(task_dataloaders['train']))
            input_dim = sample_batch.x.shape[1]
            
            # CRITICAL FIX: Determine number of classes from ALL data
            if not is_regression:
                output_dim = get_total_classes_from_dataloaders(task_dataloaders)
            else:
                output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
            
            print(f"Model configuration:")
            print(f"  Input dim: {input_dim}")
            print(f"  Output dim: {output_dim}")
            print(f"  Is regression: {is_regression}")
            
            # Determine which models to run
            models_to_run = []
            if self.config.run_gnn:
                models_to_run.extend(self.config.gnn_types)
            if self.config.run_mlp:
                models_to_run.append('mlp')
            if self.config.run_rf:
                models_to_run.append('rf')
            
            # Train each model
            for model_name in models_to_run:
                print(f"\n--- Training {model_name.upper()} ---")
                
                try:
                    # Create model with CORRECT number of classes
                    if model_name in self.config.gnn_types:
                        model = GNNModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,  # Use correct output_dim here!
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            gnn_type=model_name,
                            is_regression=is_regression
                        )
                    elif model_name == 'mlp':
                        model = MLPModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,  # Use correct output_dim here!
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            is_regression=is_regression
                        )
                    elif model_name == 'rf':
                        model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,  # Use correct output_dim here!
                            is_regression=is_regression
                        )
                    
                    # Train model
                    results = train_and_evaluate_inductive(
                        model=model,
                        dataloaders=task_dataloaders,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams
                    )
                    
                    task_results[model_name] = results
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
        """Save all experiment results."""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save main results
        results_file = os.path.join(self.output_dir, "inductive_results.json")
        
        # Create a clean results dictionary with only serializable data
        clean_results = {}
        for task, task_results in self.results.items():
            clean_results[task] = {}
            for model_name, model_results in task_results.items():
                clean_results[task][model_name] = {
                    'test_metrics': model_results.get('test_metrics', {}),
                    'train_time': model_results.get('train_time', 0.0),
                    'best_epoch': model_results.get('best_epoch', 0),
                    'training_history': {
                        'train_loss': [float(x) for x in model_results.get('training_history', {}).get('train_loss', [])],
                        'val_loss': [float(x) for x in model_results.get('training_history', {}).get('val_loss', [])],
                        'train_metric': [float(x) for x in model_results.get('training_history', {}).get('train_metric', [])],
                        'val_metric': [float(x) for x in model_results.get('training_history', {}).get('val_metric', [])]
                    }
                }
        
        # Save clean results
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Save individual graphs if requested
        if self.config.save_individual_graphs and self.family_graphs:
            import pickle
            graphs_dir = os.path.join(self.output_dir, "graphs")
            os.makedirs(graphs_dir, exist_ok=True)
            
            for i, graph in enumerate(self.family_graphs):
                graph_file = os.path.join(graphs_dir, f"graph_{i:03d}.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            print(f"Individual graphs saved to: {graphs_dir}")
        
        # Save family statistics
        if self.family_stats:
            stats_file = os.path.join(self.output_dir, "family_generation_stats.json")
            # Convert numpy types to native Python types
            clean_stats = self._make_json_serializable(self.family_stats)
            with open(stats_file, 'w') as f:
                json.dump(clean_stats, f, indent=2)
            
            print(f"Family statistics saved to: {stats_file}")
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "config.json")
        config_dict = self.config.to_dict()
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {config_file}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            # Convert dictionary keys to strings if they're not already
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, try to serialize their attributes
            try:
                return self._make_json_serializable(obj.__dict__)
            except:
                return str(obj)
        elif hasattr(obj, 'items'):  # Handle mappingproxy and similar mapping objects
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, type) or hasattr(obj, '__origin__'):  # Handle type hints and generic types
            return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of the inductive experiment.
        
        Returns:
            Formatted string report
        """
        if not self.results:
            return "No results available. Run experiments first."
        
        report_lines = []
        report_lines.append("INDUCTIVE LEARNING EXPERIMENT SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Experiment configuration summary
        report_lines.append("CONFIGURATION:")
        report_lines.append(f"  Graph family size: {self.config.n_graphs}")
        report_lines.append(f"  Node range: [{self.config.min_n_nodes}, {self.config.max_n_nodes}]")
        report_lines.append(f"  Community range: [{self.config.min_communities}, {self.config.max_communities}]")
        report_lines.append(f"  Inductive mode: {self.config.inductive_mode}")
        n_train, n_val, n_test = self.config.get_graph_splits()
        report_lines.append(f"  Graph splits: {n_train} train, {n_val} val, {n_test} test")
        report_lines.append(f"  Tasks: {', '.join(self.config.tasks)}")
        report_lines.append("")
        
        # Family statistics
        if self.family_stats:
            report_lines.append("FAMILY GENERATION:")
            report_lines.append(f"  Success rate: {self.family_stats.get('success_rate', 0):.1%}")
            report_lines.append(f"  Total time: {self.family_stats.get('total_time', 0):.1f}s")
            report_lines.append(f"  Avg time per graph: {self.family_stats.get('avg_time_per_graph', 0):.2f}s")
            report_lines.append("")
        
        # Results summary for each task
        for task, task_results in self.results.items():
            report_lines.append(f"TASK: {task.upper()}")
            report_lines.append("-" * 40)
            
            is_regression = self.config.is_regression.get(task, False)
            primary_metric = 'r2' if is_regression else 'f1_macro'
            
            model_performances = []
            
            for model_name, model_results in task_results.items():
                if 'test_metrics' in model_results:
                    test_metrics = model_results['test_metrics']
                    primary_score = test_metrics.get(primary_metric, 0.0)
                    train_time = model_results.get('train_time', 0.0)
                    
                    model_performances.append({
                        'model': model_name.upper(),
                        'score': primary_score,
                        'train_time': train_time
                    })
                    
                    report_lines.append(f"  {model_name.upper()}:")
                    report_lines.append(f"    {primary_metric.upper()}: {primary_score:.4f}")
                    report_lines.append(f"    Training time: {train_time:.2f}s")
                    
                    if is_regression:
                        report_lines.append(f"    MSE: {test_metrics.get('mse', 0.0):.4f}")
                        report_lines.append(f"    RMSE: {test_metrics.get('rmse', 0.0):.4f}")
                    else:
                        report_lines.append(f"    Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
                    
                    report_lines.append("")
            
            # Best model summary
            if model_performances:
                best_model = max(model_performances, key=lambda x: x['score'])
                report_lines.append(f"  BEST MODEL: {best_model['model']} ({primary_metric.upper()}: {best_model['score']:.4f})")
                report_lines.append("")
        
        # Overall summary
        report_lines.append("OVERALL SUMMARY:")
        total_models = sum(len(task_results) for task_results in self.results.values())
        successful_models = sum(
            sum(1 for model_results in task_results.values() if 'test_metrics' in model_results)
            for task_results in self.results.values()
        )
        report_lines.append(f"  Total models trained: {total_models}")
        report_lines.append(f"  Successful models: {successful_models}")
        report_lines.append(f"  Success rate: {successful_models/total_models:.1%}" if total_models > 0 else "  Success rate: 0%")
        
        return "\n".join(report_lines)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete inductive experiment pipeline.
        
        Returns:
            Dictionary containing all results
        """
        try:
            print("Starting inductive learning experiment...")
            experiment_start = time.time()
            
            # Generate graph family
            family_graphs = self.generate_graph_family()
            
            # Prepare data
            dataloaders = self.prepare_data()
            
            # Run experiments
            results = self.run_experiments()
            
            # Save results
            self.save_results()
            
            # Generate and save summary report
            summary_report = self.generate_summary_report()
            
            with open(os.path.join(self.output_dir, "summary_report.txt"), 'w') as f:
                f.write(summary_report)
            
            print("\n" + summary_report)
            
            total_time = time.time() - experiment_start
            print(f"\nExperiment completed in {total_time:.2f} seconds")
            
            return {
                'family_graphs': family_graphs,
                'results': results,
                'dataloaders': dataloaders,
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
            
            error_file = os.path.join(self.output_dir, "error_info.json")
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            raise


def run_inductive_experiment(config: InductiveExperimentConfig) -> Dict[str, Any]:
    """
    Convenience function to run an inductive experiment with configuration.
    
    Args:
        config: Inductive experiment configuration
        
    Returns:
        Experiment results
    """
    experiment = InductiveExperiment(config)
    return experiment.run()