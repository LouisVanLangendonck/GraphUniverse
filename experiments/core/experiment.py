"""
Experiment configuration and execution.

This module provides classes for configuring and running MMSB graph learning experiments.
"""

import os
import json
import logging
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from experiments.core.data import prepare_data
from experiments.core.models import GNNModel, MLPModel, SklearnModel
from experiments.core.training import train_gnn_model, train_mlp_model, train_sklearn_model
from experiments.core.metrics import model_performance_summary
from mmsb.model import GraphUniverse, GraphSample
from utils.parameter_analysis import analyze_graph_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_cuda():
    """Clean up CUDA resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CUDA cache cleared")

def setup_cuda(device_id: int = 0) -> torch.device:
    """Initialize CUDA and return appropriate device."""
    try:
        # Clean up any existing CUDA state
        cleanup_cuda()
        
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        
        # Check CUDA device
        if device_id >= torch.cuda.device_count():
            logger.warning(f"CUDA device {device_id} not available. Falling back to CPU.")
            return torch.device("cpu")
        
        # Set device
        device = torch.device(f"cuda:{device_id}")
        
        # Test CUDA with a simple operation
        try:
            test_tensor = torch.zeros(1, device=device)
            test_tensor += 1
            logger.info(f"CUDA device {device_id} initialized successfully")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(device_id)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(device_id) / 1024**2:.2f} MB")
            logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(device_id) / 1024**2:.2f} MB")
            
            # Set CUDA device
            torch.cuda.set_device(device_id)
            
            return device
        except Exception as e:
            logger.error(f"CUDA test failed: {str(e)}")
            return torch.device("cpu")
            
    except Exception as e:
        logger.error(f"CUDA initialization failed: {str(e)}")
        return torch.device("cpu")

def set_seed(seed: int, device: torch.device) -> None:
    """Set random seeds for reproducibility."""
    try:
        # First set CPU seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Then handle CUDA seeds if available
        if device.type == "cuda":
            try:
                # Clear CUDA cache before setting seeds
                cleanup_cuda()
                
                # Set CUDA seeds
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                # Additional CUDA settings for reproducibility
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Test CUDA seeding
                test_tensor = torch.randn(1, device=device)
                logger.info(f"CUDA seeding successful on device {device}")
            except Exception as e:
                logger.error(f"CUDA seeding failed: {str(e)}")
                # Don't raise here, continue with CPU seeding
                
        logger.info(f"Random seeds set to {seed} on device {device}")
    except Exception as e:
        logger.error(f"Error setting seeds: {str(e)}")
        # Don't raise here, continue with whatever seeding worked

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Graph generation parameters
    num_communities: int = 5
    num_nodes: int = 100
    feature_dim: int = 32
    edge_density: float = 0.1
    inter_community_density: float = 0.05
    homophily: float = 0.5
    feature_signal: float = 0.5
    randomness_factor: float = 0.1
    overlap_density: float = 0.2
    min_connection_strength: float = 0.05
    min_component_size: int = 5
    degree_heterogeneity: float = 0.5
    indirect_influence: float = 0.1
    block_structure: str = "assortative"
    overlap_structure: str = "random"
    edge_noise: float = 0.0
    feature_type: str = "generated"
    
    # Model parameters
    gnn_types: List[str] = field(default_factory=lambda: ['gat', 'gcn', 'sage'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    
    # Training parameters
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    patience: int = 150  # Add default matching run_multiple_experiments.py
    epochs: int = 500    # Add default matching run_multiple_experiments.py
    
    # Output parameters
    output_dir: str = "results"
    device_id: int = 0  # Default to first CUDA device
    force_cpu: bool = False  # Option to force CPU usage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save config to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class Experiment:
    """Class for running a single experiment."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration."""
        self.config = config
        
        # Initialize device
        if config.force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forcing CPU usage as requested")
        else:
            self.device = setup_cuda(config.device_id)
            logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        set_seed(config.seed, self.device)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.output_dir, "config.json"))
        
        # Initialize results
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        try:
            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(42)
            logger.info(f"Random seeds set to 42 on device {self.device}")
            
            # Generate graph
            logger.info("Starting experiment...")
            logger.info("Creating graph universe...")
            universe = GraphUniverse(
                K=self.config.num_communities,
                feature_dim=self.config.feature_dim,
                block_structure=self.config.block_structure,
                edge_density=self.config.edge_density,
                inter_community_density=self.config.inter_community_density,
                feature_signal=self.config.feature_signal
            )
            
            # Generate graph sample
            logger.info("Creating graph sample...")
            graph_sample = GraphSample(
                universe=universe,
                communities=list(range(self.config.num_communities)),
                n_nodes=self.config.num_nodes,
                min_component_size=self.config.min_component_size,
                degree_heterogeneity=self.config.degree_heterogeneity,
                edge_noise=self.config.edge_noise,
                indirect_influence=self.config.indirect_influence
            )
            
            # Calculate real graph properties
            logger.info("Calculating real graph properties...")
            real_graph_properties = analyze_graph_parameters(
                graph=graph_sample.graph,
                membership_vectors=graph_sample.membership_vectors,
                communities=list(range(self.config.num_communities))
            )
            
            # Store real graph properties in the graph sample
            graph_sample.real_graph_properties = real_graph_properties
            
            # Print real graph properties
            logger.info("\nReal Graph Properties:")
            for key, value in real_graph_properties.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Prepare data
            logger.info("Preparing data...")
            features, edge_index, labels, train_idx, val_idx, test_idx, num_classes = prepare_data(
                graph_sample=graph_sample,
                config=self.config,
                feature_type=self.config.feature_type
            )
            
            # Print class distribution
            logger.info("\nClass Distribution:")
            logger.info(f"Total: {torch.bincount(labels)}")
            logger.info(f"Train: {torch.bincount(labels[train_idx])}")
            logger.info(f"Val: {torch.bincount(labels[val_idx])}")
            logger.info(f"Test: {torch.bincount(labels[test_idx])}")
            
            # Check for class imbalance
            train_dist = torch.bincount(labels[train_idx]).float() / len(train_idx)
            
            # Store results
            self.results = {
                "graph_sample": graph_sample,
                "features": features,
                "edge_index": edge_index,
                "labels": labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "num_classes": num_classes,
                "real_graph_properties": real_graph_properties
            }
            
            # Run models
            self._run_models()
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}")
            raise
    
    def _run_models(self) -> None:
        """Run all models for the experiment."""
        try:
            # Move data to device
            if self.device.type == "cuda":
                logger.info("Moving data to CUDA device...")
                self.results["features"] = self.results["features"].to(self.device)
                self.results["edge_index"] = self.results["edge_index"].to(self.device)
                self.results["labels"] = self.results["labels"].to(self.device)
                self.results["train_idx"] = self.results["train_idx"].to(self.device)
                self.results["val_idx"] = self.results["val_idx"].to(self.device)
                self.results["test_idx"] = self.results["test_idx"].to(self.device)
            
            # Train models
            model_results = {}
            
            if self.config.run_gnn:
                for gnn_type in self.config.gnn_types:
                    logger.info(f"Training {gnn_type.upper()} model...")
                    model = GNNModel(
                        input_dim=self.results["features"].shape[1],
                        hidden_dim=64,
                        output_dim=self.results["num_classes"],
                        gnn_type=gnn_type
                    ).to(self.device)
                    
                    result = train_gnn_model(
                        model, self.results["features"], self.results["edge_index"], self.results["labels"],
                        self.results["train_idx"], self.results["val_idx"], self.results["test_idx"],
                        epochs=self.config.epochs,
                        patience=self.config.patience
                    )
                    model_results[f"{gnn_type.upper()}"] = result
            
            if self.config.run_mlp:
                logger.info("Training MLP model...")
                model = MLPModel(
                    input_dim=self.results["features"].shape[1],
                    hidden_dim=64,
                    output_dim=self.results["num_classes"],
                    num_layers=2,
                    dropout=0.5
                ).to(self.device)
                
                result = train_mlp_model(
                    model, self.results["features"], self.results["labels"],
                    self.results["train_idx"], self.results["val_idx"], self.results["test_idx"],
                    epochs=self.config.epochs,
                    patience=self.config.patience
                )
                model_results["MLP"] = result
            
            if self.config.run_rf:
                logger.info("Training Random Forest model...")
                model = SklearnModel(
                    input_dim=self.results["features"].shape[1],
                    output_dim=self.results["num_classes"]
                )
                result = train_sklearn_model(
                    model, self.results["features"], self.results["labels"],
                    self.results["train_idx"], self.results["val_idx"], self.results["test_idx"]
                )
                model_results["RandomForest"] = result
            
            # Store model results
            self.results["model_results"] = model_results
            
        except Exception as e:
            logger.error(f"Error running models: {str(e)}")
            raise
    
    def _save_results(self) -> None:
        """Save experiment results."""
        try:
            # Create a serializable version of model results
            serializable_results = {}
            for model_name, result in self.results["model_results"].items():
                serializable_results[model_name] = {
                    "test_acc": result.get("test_acc", 0),
                    "train_time": result.get("train_time", 0),
                    "metrics": result.get("metrics", {}),
                    "history": result.get("history", {})
                }
            
            # Save model results
            model_results_path = os.path.join(self.output_dir, "model_results.json")
            with open(model_results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Save real graph properties
            real_props_path = os.path.join(self.output_dir, "real_graph_properties.json")
            with open(real_props_path, 'w') as f:
                json.dump(self.results["real_graph_properties"], f, indent=2)
            
            # Save graph sample
            graph_sample_path = os.path.join(self.output_dir, "graph_sample.pkl")
            import pickle
            with open(graph_sample_path, 'wb') as f:
                pickle.dump(self.results["graph_sample"], f)
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise 