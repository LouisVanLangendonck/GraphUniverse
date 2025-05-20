"""
Enhanced experiment class with hyperparameter optimization support.

This module updates the Experiment class to incorporate hyperparameter optimization.
"""

import os
import json
import logging
import gc
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from experiments.core.data import prepare_data
from experiments.core.config import ExperimentConfig
from experiments.core.models import GNNModel, MLPModel, SklearnModel
from experiments.core.training import train_and_evaluate, optimize_hyperparameters
from experiments.core.metrics import model_performance_summary
from mmsb.model import GraphUniverse, GraphSample
from utils.parameter_analysis import analyze_graph_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set CUDA_LAUNCH_BLOCKING for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Experiment:
    """Class for running a single experiment with hyperparameter optimization."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration."""
        self.config = config
        
        # Initialize device
        if config.force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = self._setup_cuda(config.device_id)
        
        # Set random seeds
        self._set_seed(config.seed, self.device)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.output_dir, "config.json"))
        
        # Initialize results
        self.results = {}
    
    def _cleanup_cuda(self):
        """Clean up CUDA resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _setup_cuda(self, device_id: int = 0) -> torch.device:
        """Initialize CUDA and return appropriate device."""
        try:
            # Clean up any existing CUDA state
            self._cleanup_cuda()
            
            if not torch.cuda.is_available():
                return torch.device("cpu")
            
            # Check CUDA device
            if device_id >= torch.cuda.device_count():
                return torch.device("cpu")
            
            # Set device
            device = torch.device(f"cuda:{device_id}")
            
            # Test CUDA with a simple operation
            test_tensor = torch.zeros(1, device=device)
            test_tensor += 1
            
            # Set CUDA device
            torch.cuda.set_device(device_id)
            
            return device
                
        except Exception as e:
            return torch.device("cpu")

    def _set_seed(self, seed: int, device: torch.device) -> None:
        """Set random seeds for reproducibility."""
        try:
            # First set CPU seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Then handle CUDA seeds if available
            if device.type == "cuda":
                try:
                    # Clear CUDA cache before setting seeds
                    self._cleanup_cuda()
                    
                    # Set CUDA seeds
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    
                    # Additional CUDA settings for reproducibility
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
        except Exception:
            pass
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        try:
            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(42)
            
            # Generate graph
            print("\nGenerating graph...")
            graph_sample = self.generate_graph()
            
            # Prepare data
            print("\nPreparing data for tasks...")
            task_data = prepare_data(graph_sample, self.config)
            
            # Train and evaluate models for each task
            model_results = {}
            for task in self.config.tasks:
                print(f"\nTraining models for task: {task}")
                
                # Set is_regression based on task type
                is_regression = self.config.is_regression[task]
                
                # Get data for this task
                data = task_data[task]
                
                # Train GNN models if enabled
                if self.config.run_gnn:
                    for gnn_type in self.config.gnn_types:
                        print(f"\nTraining {gnn_type.upper()} model...")
                        try:
                            # Get output dimension based on task type
                            if is_regression:
                                output_dim = data['labels'].shape[1]  # For regression, use the number of target values
                            else:
                                output_dim = data['num_classes']  # For classification, use number of classes
                            
                            model = GNNModel(
                                input_dim=data['features'].shape[1],
                                hidden_dim=self.config.hidden_dim,
                                output_dim=output_dim,
                                num_layers=self.config.num_layers,
                                dropout=self.config.dropout,
                                gnn_type=gnn_type,
                                is_regression=is_regression
                            )
                            
                            if self.config.optimize_hyperparams:
                                print(f"Optimizing hyperparameters for {gnn_type.upper()}...")
                                hyperopt_results = optimize_hyperparameters(
                                    model_creator=lambda **kwargs: GNNModel(**kwargs),
                                    features=data['features'],
                                    edge_index=data['edge_index'],
                                    labels=data['labels'],
                                    train_idx=data['train_idx'],
                                    val_idx=data['val_idx'],
                                    test_idx=data['test_idx'],
                                    model_type="gnn",
                                    gnn_type=gnn_type,
                                    n_trials=self.config.n_trials,
                                    max_epochs=self.config.epochs,
                                    timeout=self.config.optimization_timeout,
                                    device=self.device,
                                    is_regression=is_regression
                                )
                                
                                # Update model with best parameters
                                if hyperopt_results and "best_params" in hyperopt_results:
                                    best_params = hyperopt_results["best_params"]
                                    model = GNNModel(
                                        input_dim=data['features'].shape[1],
                                        hidden_dim=best_params.get("hidden_dim", self.config.hidden_dim),
                                        output_dim=output_dim,
                                        num_layers=best_params.get("num_layers", self.config.num_layers),
                                        dropout=best_params.get("dropout", self.config.dropout),
                                        gnn_type=gnn_type,
                                        is_regression=is_regression,
                                        residual=best_params.get("residual", False),
                                        norm_type=best_params.get("norm_type", "none"),
                                        agg_type=best_params.get("agg_type", "mean"),
                                        heads=best_params.get("heads", 1) if gnn_type == "gat" else None,
                                        concat_heads=best_params.get("concat_heads", True) if gnn_type == "gat" else None
                                    )
                            
                            results = train_and_evaluate(model, data, self.config, is_regression)
                            # if self.config.optimize_hyperparams and hyperopt_results:
                            #     results['hyperopt_results'] = hyperopt_results
                            model_results[f"{task}--{gnn_type}"] = results
                        except Exception as e:
                            print(f"Error in experiment: {str(e)}")
                            print("Traceback:")
                            import traceback
                            traceback.print_exc()
                            continue
                
                # Train MLP model if enabled
                if self.config.run_mlp:
                    print("\nTraining MLP model...")
                    try:
                        # Get output dimension based on task type
                        if is_regression:
                            output_dim = data['labels'].shape[1]  # For regression, use the number of target values
                        else:
                            output_dim = data['num_classes']  # For classification, use number of classes
                        
                        model = MLPModel(
                            input_dim=data['features'].shape[1],
                            hidden_dim=self.config.hidden_dim,
                            output_dim=output_dim,
                            num_layers=self.config.num_layers,
                            dropout=self.config.dropout,
                            is_regression=is_regression
                        )
                        
                        if self.config.optimize_hyperparams:
                            print("Optimizing hyperparameters for MLP...")
                            hyperopt_results = optimize_hyperparameters(
                                model_creator=lambda **kwargs: MLPModel(**kwargs),
                                features=data['features'],
                                edge_index=None,
                                labels=data['labels'],
                                train_idx=data['train_idx'],
                                val_idx=data['val_idx'],
                                test_idx=data['test_idx'],
                                model_type="mlp",
                                n_trials=self.config.n_trials,
                                max_epochs=self.config.epochs,
                                timeout=self.config.optimization_timeout,
                                device=self.device,
                                is_regression=is_regression
                            )
                            
                            # Update model with best parameters
                            if hyperopt_results and "best_params" in hyperopt_results:
                                best_params = hyperopt_results["best_params"]
                                model = MLPModel(
                                    input_dim=data['features'].shape[1],
                                    hidden_dim=best_params.get("hidden_dim", self.config.hidden_dim),
                                    output_dim=output_dim,
                                    num_layers=best_params.get("num_layers", self.config.num_layers),
                                    dropout=best_params.get("dropout", self.config.dropout),
                                    is_regression=is_regression
                                )
                        
                        results = train_and_evaluate(model, data, self.config, is_regression)
                        # if self.config.optimize_hyperparams and hyperopt_results:
                        #     results['hyperopt_results'] = hyperopt_results
                        model_results[f"{task}--mlp"] = results
                    except Exception as e:
                        print(f"Error in experiment: {str(e)}")
                        print("Traceback:")
                        import traceback
                        traceback.print_exc()
                
                # Train Random Forest model if enabled
                if self.config.run_rf:
                    print("\nTraining Random Forest model...")
                    try:
                        # Get output dimension based on task type
                        if is_regression:
                            output_dim = data['labels'].shape[1]  # For regression, use the number of target values
                        else:
                            output_dim = data['num_classes']  # For classification, use number of classes
                        
                        model = SklearnModel(
                            input_dim=data['features'].shape[1],
                            output_dim=output_dim,
                            is_regression=is_regression
                        )
                        
                        if self.config.optimize_hyperparams:
                            print("Optimizing hyperparameters for Random Forest...")
                            hyperopt_results = optimize_hyperparameters(
                                model_creator=lambda **kwargs: SklearnModel(**kwargs),
                                features=data['features'],
                                edge_index=None,
                                labels=data['labels'],
                                train_idx=data['train_idx'],
                                val_idx=data['val_idx'],
                                test_idx=data['test_idx'],
                                model_type="rf",
                                n_trials=self.config.n_trials,
                                max_epochs=self.config.epochs,
                                timeout=self.config.optimization_timeout,
                                device=self.device,
                                is_regression=is_regression
                            )
                            
                            # Update model with best parameters
                            if hyperopt_results and "best_params" in hyperopt_results:
                                best_params = hyperopt_results["best_params"]
                                model = SklearnModel(
                                    input_dim=data['features'].shape[1],
                                    output_dim=output_dim,
                                    is_regression=is_regression,
                                    n_estimators=best_params.get("n_estimators", 100),
                                    max_depth=best_params.get("max_depth", None),
                                    min_samples_split=best_params.get("min_samples_split", 2),
                                    min_samples_leaf=best_params.get("min_samples_leaf", 1)
                                )
                        
                        results = train_and_evaluate(model, data, self.config, is_regression)
                        # if self.config.optimize_hyperparams and hyperopt_results:
                        #     results['hyperopt_results'] = hyperopt_results
                        model_results[f"{task}--rf"] = results
                    except Exception as e:
                        print(f"Error in experiment: {str(e)}")
                        print("Traceback:")
                        import traceback
                        traceback.print_exc()
            
            # Save results
            print("\nSaving results...")
            self._save_results(graph_sample, model_results)
            
            return {
                "graph_sample": graph_sample,
                "model_results": model_results
            }
            
        except Exception as e:
            print(f"Error in experiment: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self, graph_sample: GraphSample, model_results: Dict[str, Any]) -> None:
        """Save experiment results in JSON format."""
        try:
            # Save model results directly
            model_results_path = os.path.join(self.output_dir, "model_results.json")
            with open(model_results_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            # Save graph sample
            graph_sample_path = os.path.join(self.output_dir, "graph_sample.pkl")
            import pickle
            with open(graph_sample_path, 'wb') as f:
                pickle.dump(graph_sample, f)
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            
            # Save error information
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            error_path = os.path.join(self.output_dir, "error_info.json")
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)

    def generate_graph(self) -> GraphSample:
        """
        Generate a graph sample using the MMSB model based on the experiment configuration.
        
        Returns:
            GraphSample: The generated graph sample
        """
        try:
            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(42)
            
            # Generate graph
            print("\nStarting experiment...")
            print("Initializing universe...")
            universe = GraphUniverse(
                K=self.config.num_communities,
                feature_dim=self.config.feature_dim,
                block_structure=self.config.block_structure,
                edge_density=self.config.edge_density,
                homophily=self.config.homophily,
                randomness_factor=self.config.randomness_factor,
                regimes_per_community=self.config.regimes_per_community,
                intra_community_regime_similarity=self.config.intra_community_regime_similarity,
                inter_community_regime_similarity=self.config.inter_community_regime_similarity
            )
            
            print("Generating graph sample...")
            # Select communities (random for now, could be improved)
            communities = list(range(self.config.num_communities))
            
            # Prepare method-specific config_model_params
            config_model_params = None
            use_configuration_model = False
            degree_distribution = None
            power_law_exponent = None
            target_avg_degree = None
            triangle_enhancement = 0.0  # Could be exposed as a config param
            
            if self.config.distribution_type == "power_law":
                use_configuration_model = True
                degree_distribution = "power_law"
                power_law_exponent = self.config.power_law_exponent
                target_avg_degree = self.config.power_law_target_avg_degree
                config_model_params = {
                    'power_law_exponent': power_law_exponent,
                    'target_avg_degree': target_avg_degree
                }
            elif self.config.distribution_type == "exponential":
                use_configuration_model = True
                degree_distribution = "exponential"
                config_model_params = {
                    'rate': self.config.exponential_rate,
                    'target_avg_degree': self.config.exponential_target_avg_degree
                }
                power_law_exponent = None
                target_avg_degree = self.config.exponential_target_avg_degree
            elif self.config.distribution_type == "uniform":
                use_configuration_model = True
                degree_distribution = "uniform"
                config_model_params = {
                    'min_factor': self.config.uniform_min_factor,
                    'max_factor': self.config.uniform_max_factor,
                    'target_avg_degree': self.config.uniform_target_avg_degree
                }
                power_law_exponent = None
                target_avg_degree = self.config.uniform_target_avg_degree
            else:
                use_configuration_model = False
                degree_distribution = None
                config_model_params = None
                power_law_exponent = None
                target_avg_degree = None
            
            # Create graph sample with all required parameters
            graph_sample = GraphSample(
                universe=universe,
                communities=communities,
                n_nodes=self.config.num_nodes,
                min_component_size=self.config.min_component_size,
                degree_heterogeneity=self.config.degree_heterogeneity,
                edge_noise=self.config.edge_noise,
                feature_regime_balance=self.config.feature_regime_balance,
                target_homophily=self.config.homophily,
                target_density=self.config.edge_density,
                use_configuration_model=use_configuration_model,
                degree_distribution=degree_distribution,
                power_law_exponent=power_law_exponent,
                target_avg_degree=target_avg_degree,
                triangle_enhancement=triangle_enhancement,
                max_mean_community_deviation=self.config.max_mean_community_deviation,
                max_max_community_deviation=self.config.max_max_community_deviation,
                max_parameter_search_attempts=self.config.max_parameter_search_attempts,
                parameter_search_range=self.config.parameter_search_range,
                min_edge_density=0.001,  # Could be exposed as a config param
                max_retries=self.config.max_retries,
                seed=self.config.seed,
                config_model_params=config_model_params
            )
            
            # Print timing information
            print("\nGraph Generation Timing:")
            for step, time_taken in graph_sample.timing_info.items():
                print(f"  {step}: {time_taken:.3f}s")
            
            # Calculate real graph properties
            print("\nCalculating graph properties...")
            real_graph_properties = analyze_graph_parameters(
                graph=graph_sample.graph,
                community_labels=graph_sample.community_labels,
                communities=list(range(self.config.num_communities))
            )
            
            # Store real graph properties in the graph sample
            graph_sample.real_graph_properties = real_graph_properties
            
            return graph_sample
            
        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")
            raise