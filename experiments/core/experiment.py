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
            universe = GraphUniverse(
                K=self.config.num_communities,
                feature_dim=self.config.feature_dim,
                block_structure=self.config.block_structure,
                edge_density=self.config.edge_density,
                homophily=self.config.homophily,
                randomness_factor=self.config.randomness_factor,
                mixed_membership=self.config.mixed_membership,
                regimes_per_community=self.config.regimes_per_community,
                intra_community_regime_similarity=self.config.intra_community_regime_similarity,
                inter_community_regime_similarity=self.config.inter_community_regime_similarity
            )
            
            # Generate graph sample
            graph_sample = GraphSample(
                universe=universe,
                communities=list(range(self.config.num_communities)),
                n_nodes=self.config.num_nodes,
                min_component_size=self.config.min_component_size,
                degree_heterogeneity=self.config.degree_heterogeneity,
                edge_noise=self.config.edge_noise,
                indirect_influence=self.config.indirect_influence,
                feature_regime_balance=self.config.feature_regime_balance
            )
            
            # Calculate real graph properties
            real_graph_properties = analyze_graph_parameters(
                graph=graph_sample.graph,
                membership_vectors=graph_sample.membership_vectors,
                communities=list(range(self.config.num_communities))
            )
            
            # Store real graph properties in the graph sample
            graph_sample.real_graph_properties = real_graph_properties
            
            # Prepare data for all tasks
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
                "model_results": {}
            }
            
            # Run models with hyperparameter optimization
            self._run_models()
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
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
                        model = GNNModel(
                            input_dim=data["features"].shape[1],
                            hidden_dim=64,
                            output_dim=data["num_classes"],
                            gnn_type=gnn_type
                        ).to(self.device)
                        
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