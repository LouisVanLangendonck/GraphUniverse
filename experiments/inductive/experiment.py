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

from mmsb.model import GraphUniverse
from mmsb.graph_family import GraphFamilyGenerator, FamilyConsistencyAnalyzer
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders,
    analyze_graph_family_properties
)
from experiments.inductive.training import train_and_evaluate_inductive, get_total_classes_from_dataloaders
from experiments.core.models import GNNModel, MLPModel, SklearnModel
from experiments.inductive.self_supervised_task import (
    PreTrainingConfig, 
    SelfSupervisedTask, 
    create_ssl_task,
    PreTrainedModelSaver,
    create_pretraining_dataloader
)
from experiments.inductive.data import GraphFamilyManager
from experiments.inductive.config import PreTrainingConfig, InductiveExperimentConfig

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
                print(f"🎯 Using {len(finetuning_graphs)} graphs for fine-tuning")
                self.family_graphs = finetuning_graphs
                return finetuning_graphs
        
        # Otherwise, generate new graph family (existing logic)
        print("\n" + "="*60)
        print("GENERATING GRAPH FAMILY")
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
        
        # Create family generator with clean parameters
        print("Setting up graph family generator...")
        family_generator = GraphFamilyGenerator(
            universe=universe,
            n_graphs=self.config.n_graphs,
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
        
        self.family_graphs = family_graphs
        
        return family_graphs
    
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
        inductive_data = prepare_inductive_data(self.family_graphs, self.config)
        
        # Create dataloaders
        print("Creating dataloaders...")
        dataloaders = create_inductive_dataloaders(inductive_data, self.config)
        
        # Print split information
        for task in self.config.tasks:
            print(f"\nTask: {task}")
            for split in ['train', 'val', 'test']:
                n_graphs = inductive_data[task][split]['n_graphs']
                batch_size = inductive_data[task][split]['batch_size']
                print(f"  {split}: {n_graphs} graphs, batch size {batch_size}")
        
        self.dataloaders = dataloaders
        return dataloaders
    
    def run_experiments(self) -> Dict[str, Any]:
        """Run inductive learning experiments."""
        print("\n" + "="*60)
        print("RUNNING INDUCTIVE EXPERIMENTS")
        print("="*60)
        
        if not hasattr(self, 'dataloaders'):
            raise ValueError("Must prepare data before running experiments")
        
        all_results = {}
        
        for task in self.config.tasks:
            print(f"\n{'='*40}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*40}")
            
            task_results = {}
            task_dataloaders = self.dataloaders[task]
            is_regression = self.config.is_regression.get(task, False)
            
            # Get dimensions
            sample_batch = next(iter(task_dataloaders['train']))
            input_dim = sample_batch.x.shape[1]
            
            if not is_regression:
                output_dim = get_total_classes_from_dataloaders(task_dataloaders)
            else:
                output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
            
            print(f"Model configuration:")
            print(f"  Input dim: {input_dim}")
            print(f"  Output dim: {output_dim}")
            print(f"  Is regression: {is_regression}")
            
            # Determine models to run
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
                    # Create model
                    if model_name in self.config.gnn_types:
                        model = GNNModel(
                            input_dim=input_dim,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            gnn_type=model_name,
                            is_regression=is_regression
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
                    results = train_and_evaluate_inductive(
                        model=model,
                        dataloaders=task_dataloaders,
                        config=self.config,
                        task=task,
                        device=self.device,
                        optimize_hyperparams=self.config.optimize_hyperparams,
                        experiment_name=self.config.experiment_name if hasattr(self.config, 'experiment_name') else None,
                        run_id=self.config.run_id if hasattr(self.config, 'run_id') else None
                    )
                    
                    # Store results including hyperopt results if available
                    task_results[model_name] = {
                        'test_metrics': results.get('test_metrics', {}),
                        'train_time': results.get('train_time', 0.0),
                        'training_history': results.get('training_history', {}),
                        'hyperopt_results': results.get('hyperopt_results', None)
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
                clean_results[task][model_name] = {
                    'test_metrics': model_results.get('test_metrics', {}),
                    'train_time': model_results.get('train_time', 0.0),
                    'best_epoch': model_results.get('best_epoch', 0),
                    'error': model_results.get('error'),
                    'training_history': {
                        'train_loss': [float(x) for x in model_results.get('training_history', {}).get('train_loss', [])],
                        'val_loss': [float(x) for x in model_results.get('training_history', {}).get('val_loss', [])],
                        'train_metric': [float(x) for x in model_results.get('training_history', {}).get('train_metric', [])],
                        'val_metric': [float(x) for x in model_results.get('training_history', {}).get('val_metric', [])]
                    }
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
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment pipeline."""
        try:
            print("Starting inductive learning experiment...")
            experiment_start = time.time()
            
            # Generate graph family
            family_graphs = self.generate_graph_family()
            
            # Analyze family consistency (NEW)
            family_consistency = self.analyze_family_consistency()
            
            # Calculate community signals (NEW)
            graph_signals = self.calculate_graph_signals()
            
            # Prepare data
            dataloaders = self.prepare_data()
            
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
            
            return {
                'family_graphs': family_graphs,
                'family_consistency': family_consistency,
                'graph_signals': graph_signals,
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
            
            error_file = os.path.join(self.output_dir, "error.json")
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            raise
    
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
        
        # Family consistency summary (NEW)
        if self.family_consistency:
            overall_score = self.family_consistency.get('overall', {}).get('score', 0.0)
            lines.append("FAMILY CONSISTENCY:")
            lines.append(f"  Overall score: {overall_score:.3f}")
            
            pattern_score = self.family_consistency.get('pattern_preservation', {}).get('score', 0.0)
            fidelity_score = self.family_consistency.get('generation_fidelity', {}).get('score', 0.0)
            degree_score = self.family_consistency.get('degree_consistency', {}).get('score', 0.0)
            
            lines.append(f"  Pattern preservation: {pattern_score:.3f}")
            lines.append(f"  Generation fidelity: {fidelity_score:.3f}")
            lines.append(f"  Degree consistency: {degree_score:.3f}")
            lines.append("")
        
        # Community signals summary (NEW)
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
        
        # Results summary
        for task, task_results in self.results.items():
            lines.append(f"TASK: {task.upper()}")
            lines.append("-" * 40)
            
            is_regression = self.config.is_regression.get(task, False)
            primary_metric = 'r2' if is_regression else 'f1_macro'
            
            best_score = 0.0
            best_model = None
            
            for model_name, model_results in task_results.items():
                if 'test_metrics' in model_results and model_results['test_metrics']:
                    test_metrics = model_results['test_metrics']
                    score = test_metrics.get(primary_metric, 0.0)
                    train_time = model_results.get('train_time', 0.0)
                    
                    lines.append(f"  {model_name.upper()}:")
                    lines.append(f"    {primary_metric.upper()}: {score:.4f}")
                    lines.append(f"    Training time: {train_time:.2f}s")
                    
                    if score > best_score:
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

    def run_pretraining_only(
        self, 
        family_id: Optional[str] = None,
        use_existing_family: bool = False
    ) -> Dict[str, Any]:
        """Run pre-training only, generating and saving graphs for later fine-tuning."""
        
        print("="*80)
        print("ENHANCED SSL PRE-TRAINING (GRAPHS SAVED FOR LATER)")
        print("="*80)
        
        pipeline_start = time.time()
        results = {'config': asdict(self.config)}
        
        # Step 1: Generate or load graph family
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
            family_graphs, family_id = self.family_manager.generate_and_save_family(family_id)
            
            # Create splits
            n_actual_pretraining, n_warmup, n_finetuning = self.config.get_graph_splits()
            graph_splits = {
                'pretraining': family_graphs[:n_actual_pretraining],
                'warmup': family_graphs[n_actual_pretraining:n_actual_pretraining + n_warmup],
                'finetuning': family_graphs[n_actual_pretraining + n_warmup:]
            }
            results['used_existing_family'] = False
        
        results['family_id'] = family_id
        results['graph_splits_sizes'] = {k: len(v) for k, v in graph_splits.items()}
        
        print(f"Graph splits: {results['graph_splits_sizes']}")
        
        # Step 2: Hyperparameter optimization (if enabled)
        hyperopt_results = None
        if self.config.optimize_hyperparams:
            print(f"\nStep 2: Hyperparameter Optimization")
            print("-" * 50)
            
            warmup_graphs = graph_splits['warmup']
            print(f"Using {len(warmup_graphs)} warmup graphs for hyperopt")
            
            task = create_ssl_task(self.config)
            if isinstance(task, list):
                task = task[0]  # Use first task for optimization
            
            hyperopt_results = self._optimize_hyperparameters(warmup_graphs, task)
            results['hyperopt_results'] = hyperopt_results
        else:
            print("\nSkipping hyperparameter optimization")
        
        # Step 3: Pre-training on main graphs
        print(f"\nStep 3: Pre-training")
        print("-" * 50)
        
        pretraining_graphs = graph_splits['pretraining']
        print(f"Using {len(pretraining_graphs)} graphs for pre-training")
        
        optimized_params = None
        if hyperopt_results:
            optimized_params = hyperopt_results['best_params']
        
        model, training_results = self._train_model(pretraining_graphs, optimized_params)
        results.update(training_results)
        
        # Step 4: Save pre-trained model with family reference
        print(f"\nStep 4: Saving pre-trained model")
        print("-" * 50)
        
        # Include family information in model metadata
        enhanced_metadata = {
            'family_id': family_id,
            'family_total_graphs': len(family_graphs),
            'finetuning_graphs_available': len(graph_splits['finetuning']),
            'pretraining_graphs_used': len(pretraining_graphs),
            'warmup_graphs_used': len(graph_splits['warmup'])
        }
        
        model_id = self.model_saver.save_model(
            model=model,
            config=self.config,
            training_history=training_results['training_history'],
            metrics=training_results['final_metrics'],
            hyperopt_results=hyperopt_results,
            enhanced_metadata=enhanced_metadata  # Add family info
        )
        
        results['model_id'] = model_id
        
        # Final summary
        total_time = time.time() - pipeline_start
        results['total_time'] = total_time
        
        print(f"\n" + "="*80)
        print("PRE-TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Model ID: {model_id}")
        print(f"Graph family ID: {family_id}")
        print(f"Graphs available for fine-tuning: {len(graph_splits['finetuning'])}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"✓ Ready for fine-tuning with: python run_ssl_experiments.py --mode finetune --model_id {model_id}")
        
        return results
    
    def _optimize_hyperparameters(self, warmup_graphs: List, task: SelfSupervisedTask) -> Dict[str, Any]:
        """Optimize hyperparameters using warmup graphs."""
        
        print(f"Optimizing hyperparameters using {len(warmup_graphs)} warmup graphs")
        
        # Create warmup dataloader
        warmup_loader = create_pretraining_dataloader(
            warmup_graphs, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Get input dimension
        sample_batch = next(iter(warmup_loader))
        input_dim = sample_batch.x.shape[1]
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            
            # Sample hyperparameters
            suggested_config = self._sample_hyperparameters(trial)
            
            # Create temporary config
            temp_config = PreTrainingConfig(**{**asdict(self.config), **suggested_config})
            temp_task = create_ssl_task(temp_config)
            
            if isinstance(temp_task, list):
                temp_task = temp_task[0]
            
            model = temp_task.create_model(input_dim).to(self.device)
            
            # Quick training
            max_epochs = min(50, self.config.epochs // 4)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=suggested_config['learning_rate'],
                weight_decay=suggested_config.get('weight_decay', 1e-5)
            )
            
            best_metric = float('-inf')
            patience_counter = 0
            patience = 10
            
            for epoch in range(max_epochs):
                model.train()
                for batch in warmup_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    loss = temp_task.compute_loss(model, batch)
                    loss.backward()
                    optimizer.step()
                
                # Quick evaluation
                if epoch % 5 == 0:
                    eval_metrics = temp_task.evaluate(model, warmup_loader)
                    
                    if self.config.pretraining_task == "link_prediction":
                        metric = eval_metrics.get('auc', 0.0)
                    else:
                        metric = eval_metrics.get('accuracy', 0.0)
                    
                    if metric > best_metric:
                        best_metric = metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
            
            return best_metric
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.optimization_timeout
        )
        
        print(f"Hyperopt completed: Best metric = {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for optimization."""
        params = {}
        
        # Common parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        params['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 256, step=32)
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Task-specific parameters
        # if self.config.pretraining_task in ["link_prediction", "both"]:
        #     params['negative_sampling_ratio'] = trial.suggest_float('negative_sampling_ratio', 0.5, 2.0)
        
        if self.config.pretraining_task in ["contrastive", "both"]:
            params['contrastive_temperature'] = trial.suggest_float('contrastive_temperature', 0.01, 0.2)
            params['corruption_rate'] = trial.suggest_float('corruption_rate', 0.1, 0.5)
        
        return params
    
    def _train_model(self, pretraining_graphs: List, optimized_params: Optional[Dict] = None) -> Tuple[torch.nn.Module, Dict]:
        """Train model on pretraining graphs."""
        
        # Update config with optimized parameters
        if optimized_params:
            for key, value in optimized_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            print(f"Using optimized parameters: {optimized_params}")
        
        # Create dataloader
        train_loader = create_pretraining_dataloader(
            pretraining_graphs,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.x.shape[1]
        
        # Create task and model
        task = create_ssl_task(self.config)
        if isinstance(task, list):
            task = task[0]
        
        model = task.create_model(input_dim).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_metric': []
        }
        
        best_metric = float('-inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"Training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                loss = task.compute_loss(model, batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            training_history['train_loss'].append(avg_train_loss)
            
            # Evaluation every 10 epochs
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                eval_metrics = task.evaluate(model, train_loader)
                eval_loss = eval_metrics['loss']
                
                if self.config.pretraining_task == "link_prediction":
                    primary_metric = eval_metrics.get('auc', 0.0)
                else:
                    primary_metric = eval_metrics.get('accuracy', 0.0)
                
                training_history['eval_loss'].append(eval_loss)
                training_history['eval_metric'].append(primary_metric)
                
                # Model selection
                if primary_metric > best_metric:
                    best_metric = primary_metric
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Logging
                if epoch % 50 == 0:
                    print(f"Epoch {epoch:4d}: Loss: {avg_train_loss:.4f}, Metric: {primary_metric:.4f}")
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = task.evaluate(model, train_loader)
        
        print(f"Training completed! Best metric: {best_metric:.4f}")
        
        return model, {
            'training_history': training_history,
            'final_metrics': final_metrics,
            'best_metric': best_metric,
            'total_epochs': epoch + 1
        }
    
class PreTrainingExperiment:
    """High-level experiment runner for multiple pre-training configurations."""
    
    def __init__(self, base_output_dir: str = "ssl_experiments"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
    
    def run_pretraining_sweep(
        self,
        graph_family: List,
        gnn_types: List[str] = ["gcn", "sage"],
        pretraining_tasks: List[str] = ["link_prediction", "contrastive"],
        base_config: Optional[PreTrainingConfig] = None
    ) -> Dict[str, Any]:
        """Run pre-training experiments across multiple configurations."""
        
        if base_config is None:
            base_config = PreTrainingConfig()
        
        all_results = {}
        experiment_start = time.time()
        
        print("="*100)
        print("MULTI-CONFIGURATION PRE-TRAINING SWEEP")
        print("="*100)
        print(f"GNN Types: {gnn_types}")
        print(f"Pre-training Tasks: {pretraining_tasks}")
        print(f"Total configurations: {len(gnn_types) * len(pretraining_tasks)}")
        
        for gnn_type in gnn_types:
            for task in pretraining_tasks:
                config_name = f"{gnn_type}_{task}"
                print(f"\n{'='*50}")
                print(f"CONFIGURATION: {config_name}")
                print(f"{'='*50}")
                
                # Create configuration for this run
                config = PreTrainingConfig(
                    **{**asdict(base_config), 
                       'gnn_type': gnn_type,
                       'pretraining_task': task,
                       'output_dir': os.path.join(self.base_output_dir, config_name),
                       'experiment_name': f"sweep_{config_name}"
                    }
                )
                
                try:
                    # Run pre-training
                    runner = PreTrainingRunner(config)
                    results = runner.run_full_pipeline(graph_family, save_model=True)
                    
                    all_results[config_name] = {
                        'status': 'success',
                        'results': results,
                        'config': asdict(config)
                    }
                    
                    print(f"✓ {config_name} completed successfully")
                    
                except Exception as e:
                    print(f"✗ {config_name} failed: {str(e)}")
                    logger.error(f"Error in {config_name}: {str(e)}", exc_info=True)
                    
                    all_results[config_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'config': asdict(config)
                    }
        
        # Save sweep results
        total_sweep_time = time.time() - experiment_start
        sweep_summary = {
            'sweep_results': all_results,
            'total_time': total_sweep_time,
            'successful_configs': [k for k, v in all_results.items() if v['status'] == 'success'],
            'failed_configs': [k for k, v in all_results.items() if v['status'] == 'failed'],
            'timestamp': time.time()
        }
        
        # Save summary
        summary_path = os.path.join(self.base_output_dir, "sweep_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        
        print(f"\n{'='*100}")
        print("SWEEP COMPLETED")
        print(f"{'='*100}")
        print(f"Total time: {total_sweep_time:.2f} seconds")
        print(f"Successful: {len(sweep_summary['successful_configs'])}")
        print(f"Failed: {len(sweep_summary['failed_configs'])}")
        print(f"Results saved to: {summary_path}")
        
        return sweep_summary

def run_pretraining_with_saved_graphs(
    gnn_type: str = "gcn",
    pretraining_task: str = "link_prediction",
    n_graphs: int = 50,
    n_extra_graphs_for_finetuning: int = 30,
    output_dir: str = "pretrained_models",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for pre-training with saved graphs."""
    
    config = EnhancedPreTrainingConfig(
        gnn_type=gnn_type,
        pretraining_task=pretraining_task,
        n_graphs=n_graphs,
        n_extra_graphs_for_finetuning=n_extra_graphs_for_finetuning,
        output_dir=output_dir,
        **kwargs
    )
    
    runner = PreTrainingRunner(config)
    return runner.run_pretraining_only()

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

def load_pretrained_model(self, model_id: str, model_type: str = 'gnn'):
    """Load a pre-trained model for fine-tuning."""
    from experiments.inductive.data import PreTrainedModelSaver
    
    model_saver = PreTrainedModelSaver("pretrained_models")
    model, metadata = model_saver.load_model(model_id, self.device)
    
    print(f"Loaded pre-trained model: {model_id}")
    print(f"  Pre-training task: {metadata['config']['pretraining_task']}")
    print(f"  Architecture: {metadata['architecture']}")
    
    return model, metadata

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
        print(f"📁 Using explicitly specified graph family: {family_id}")
    
    # Method 2: Auto-load family from pre-trained model
    elif config.auto_load_family and config.pretrained_model_id:
        try:
            from experiments.inductive.self_supervised_task import PreTrainedModelSaver
            
            model_saver = PreTrainedModelSaver(config.pretrained_model_dir)
            _, metadata = model_saver.load_model(config.pretrained_model_id)
            
            family_id = metadata.get('family_id')
            if family_id:
                print(f"📁 Auto-loaded graph family from model: {family_id}")
            else:
                print(f"⚠️  Pre-trained model has no associated graph family")
                return None
                
        except Exception as e:
            print(f"⚠️  Failed to auto-load graph family: {e}")
            return None
    
    # Method 3: No family specified - generate new graphs
    else:
        print(f"📁 No graph family specified - will generate new graphs")
        return None
    
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
            
            print(f"✅ Loaded {len(finetuning_graphs)} fine-tuning graphs from family {family_id}")
            print(f"   Total family size: {len(family_graphs)} graphs")
            print(f"   Family metadata: {family_metadata.get('creation_timestamp', 'Unknown')}")
            
            return finetuning_graphs
            
        except Exception as e:
            print(f"❌ Failed to load graph family {family_id}: {e}")
            return None
    
    return None

def create_model_from_pretrained(self, pretrained_model, metadata, output_dim: int, is_regression: bool):
    """Create a fine-tuning model from pre-trained model."""
    
    # Extract encoder from pre-trained model
    if hasattr(pretrained_model, 'encoder'):
        encoder = pretrained_model.encoder
    else:
        encoder = pretrained_model
    
    # Create new classification/regression head
    if is_regression:
        head = torch.nn.Linear(encoder.output_dim, output_dim)
    else:
        head = torch.nn.Sequential(
            torch.nn.Linear(encoder.output_dim, output_dim),
            torch.nn.LogSoftmax(dim=-1)
        )
    
    # Combine into new model
    class FineTuningModel(torch.nn.Module):
        def __init__(self, encoder, head, freeze_encoder=False):
            super().__init__()
            self.encoder = encoder
            self.head = head
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        def forward(self, x, edge_index=None):
            if edge_index is not None:
                embeddings = self.encoder(x, edge_index)
            else:
                embeddings = self.encoder(x)
            return self.head(embeddings)
    
    return FineTuningModel(encoder, head)