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
from experiments.core.training import train_gnn_model, train_mlp_model, train_sklearn_model
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
        """Run the experiment."""
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
            # --- New: Select communities (random for now, could be improved) ---
            communities = list(range(self.config.num_communities))
            # --- New: Prepare method-specific config_model_params ---
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
            # --- New: Pass all required parameters to GraphSample ---
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
            
            # Prepare data for all tasks
            print("\nPreparing data for tasks...")
            task_data = prepare_data(
                graph_sample=graph_sample,
                config=self.config,
                feature_type=self.config.feature_type
            )
            
            # Store results
            self.results = {
                "graph_sample": graph_sample,
                "task_data": task_data,
                "real_graph_properties": real_graph_properties,
                "model_results": {},
                "timing_info": graph_sample.timing_info  # Store timing information
            }
            
            # Run models with hyperparameter optimization
            print("\nRunning models...")
            try:
                self._run_models()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print("\nFATAL: CUDA Error detected!")
                    print("Error message:", str(e))
                    print("\nStack trace:")
                    import traceback
                    traceback.print_exc()
                    print("\nExperiment stopped due to CUDA error.")
                    import sys
                    sys.exit(1)  # Exit with error code
                raise
            
            # Save results
            print("\nSaving results...")
            self._save_results()
            
            return self.results
            
        except Exception as e:
            print(f"Error in experiment: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_models(self) -> None:
        """Run all models for each task with hyperparameter optimization if enabled."""
        try:
            # Move data to device for each task
            if self.device.type == "cuda":
                for task_name, data in self.results["task_data"].items():
                    data["features"] = data["features"].to(self.device)
                    data["edge_index"] = data["edge_index"].to(self.device)
                    data["labels"] = data["labels"].to(self.device)
                    data["train_idx"] = data["train_idx"].to(self.device)
                    data["val_idx"] = data["val_idx"].to(self.device)
                    data["test_idx"] = data["test_idx"].to(self.device)
            
            # Train models for each task
            model_results = {}
            
            for task_name, data in self.results["task_data"].items():
                print(f"\nTraining models for {task_name} prediction task...")
                task_results = {}
                
                if self.config.run_gnn:
                    for gnn_type in self.config.gnn_types:
                        print(f"\nTraining {gnn_type.upper()} model...")
                        try:
                            model = GNNModel(
                                input_dim=data["features"].shape[1],
                                hidden_dim=64,
                                output_dim=data["num_classes"],
                                gnn_type=gnn_type
                            ).to(self.device)
                            
                            # Print model architecture for debugging
                            print(f"\nModel architecture for {gnn_type.upper()}:")
                            print(model)
                            
                            # Print tensor shapes for debugging
                            print("\nTensor shapes:")
                            print(f"Features: {data['features'].shape}")
                            print(f"Edge index: {data['edge_index'].shape}")
                            print(f"Labels: {data['labels'].shape}")
                            print(f"Train idx: {data['train_idx'].shape}")
                            
                            result = train_gnn_model(
                                model, 
                                data["features"], 
                                data["edge_index"], 
                                data["labels"],
                                data["train_idx"], 
                                data["val_idx"], 
                                data["test_idx"],
                                epochs=self.config.epochs,
                                patience=self.config.patience,
                                optimize=self.config.optimize_hyperparams,
                                n_trials=self.config.n_trials,
                                timeout=self.config.opt_timeout
                            )
                            task_results[f"{gnn_type.upper()}"] = result
                            
                            # Log hyperparameter optimization results if available
                            if 'hyperopt_results' in result:
                                print(f"Best hyperparameters for {gnn_type.upper()}: {result['hyperopt_results']['best_params']}")
                                print(f"Best validation score: {result['hyperopt_results']['best_value']:.4f}")
                                
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                print(f"\nCUDA Error in {gnn_type.upper()} model:")
                                print(f"Error message: {str(e)}")
                                print("\nStack trace:")
                                import traceback
                                traceback.print_exc()
                                print("\nModel state at error:")
                                print(f"Model type: {gnn_type}")
                                print(f"Input dimension: {data['features'].shape[1]}")
                                print(f"Hidden dimension: 64")
                                print(f"Output dimension: {data['num_classes']}")
                                print(f"Device: {self.device}")
                                raise  # Re-raise to stop the experiment
                            else:
                                raise
                
                if self.config.run_mlp:
                    print("\nTraining MLP model...")
                    model = MLPModel(
                        input_dim=data["features"].shape[1],
                        hidden_dim=64,
                        output_dim=data["num_classes"],
                        num_layers=2,
                        dropout=0.5
                    ).to(self.device)
                    
                    result = train_mlp_model(
                        model, 
                        data["features"], 
                        data["labels"],
                        data["train_idx"], 
                        data["val_idx"], 
                        data["test_idx"],
                        epochs=self.config.epochs,
                        patience=self.config.patience,
                        optimize=self.config.optimize_hyperparams,
                        n_trials=self.config.n_trials,
                        timeout=self.config.opt_timeout
                    )
                    task_results["MLP"] = result
                    
                    # Log hyperparameter optimization results if available
                    if 'hyperopt_results' in result:
                        print(f"Best hyperparameters for MLP: {result['hyperopt_results']['best_params']}")
                        print(f"Best validation score: {result['hyperopt_results']['best_value']:.4f}")
                
                if self.config.run_rf:
                    print("\nTraining Random Forest model...")
                    model = SklearnModel(
                        input_dim=data["features"].shape[1],
                        output_dim=data["num_classes"]
                    )
                    result = train_sklearn_model(
                        model, 
                        data["features"], 
                        data["labels"],
                        data["train_idx"], 
                        data["val_idx"], 
                        data["test_idx"],
                        optimize=self.config.optimize_hyperparams,
                        n_trials=self.config.n_trials,
                        timeout=self.config.opt_timeout
                    )
                    task_results["RandomForest"] = result
                    
                    # Log hyperparameter optimization results if available
                    if 'hyperopt_results' in result:
                        print(f"Best hyperparameters for RandomForest: {result['hyperopt_results']['best_params']}")
                        print(f"Best validation score: {result['hyperopt_results']['best_value']:.4f}")
                
                # Store results for this task
                model_results[task_name] = task_results
            
            # Store all model results
            self.results["model_results"] = model_results
            
        except Exception as e:
            print(f"\nError in _run_models: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self) -> None:
        """Save experiment results."""
        try:
            # Create a serializable version of model results
            serializable_results = {}
            for task_name, task_results in self.results["model_results"].items():
                serializable_task = {}
                for model_name, result in task_results.items():
                    serializable_model = {
                        "test_acc": result.get("test_acc", 0),
                        "train_time": result.get("train_time", 0),
                        "metrics": result.get("metrics", {}),
                        "history": result.get("history", {})
                    }
                    
                    # Add hyperparameter optimization results if available
                    if 'hyperopt_results' in result:
                        serializable_model["hyperopt_results"] = {
                            "best_params": result["hyperopt_results"].get("best_params", {}),
                            "best_value": result["hyperopt_results"].get("best_value", 0.0),
                            "n_trials": result["hyperopt_results"].get("n_trials", 0)
                        }
                    
                    serializable_task[model_name] = serializable_model
                serializable_results[task_name] = serializable_task
            
            # Save model results
            model_results_path = os.path.join(self.output_dir, "model_results.json")
            with open(model_results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Save real graph properties
            real_props_path = os.path.join(self.output_dir, "real_graph_properties.json")
            with open(real_props_path, 'w') as f:
                json.dump(self.results["real_graph_properties"], f, indent=2)
            
            # Save task-specific information
            for task_name, data in self.results["task_data"].items():
                task_info = {}
                
                # Save task-specific metadata
                if task_name == "regime" and "rules" in data:
                    task_info["rules"] = [str(rule) for rule in data["rules"]]
                elif task_name == "role" and "role_analyzer" in data:
                    role_analyzer = data["role_analyzer"]
                    task_info["role_definitions"] = role_analyzer.interpret_roles()
                
                if task_info:
                    task_path = os.path.join(self.output_dir, f"{task_name}_info.json")
                    with open(task_path, 'w') as f:
                        json.dump(task_info, f, indent=2)
            
            # Save graph sample
            graph_sample_path = os.path.join(self.output_dir, "graph_sample.pkl")
            import pickle
            with open(graph_sample_path, 'wb') as f:
                pickle.dump(self.results["graph_sample"], f)
            
        except Exception as e:
            raise