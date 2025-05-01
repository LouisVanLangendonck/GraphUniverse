"""
Configuration class for MMSB graph learning experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
import json
import os

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Graph generation parameters
    num_communities: int = 5
    num_nodes: int = 100
    feature_dim: int = 32
    edge_density: float = 0.1  # Overall target edge density
    homophily: float = 0.8  # Controls ratio between intra/inter probabilities (0=equal, 1=max)
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
    patience: int = 150
    epochs: int = 500
    
    # Hyperparameter optimization parameters
    optimize_hyperparams: bool = True
    n_trials: int = 15  # Number of hyperparameter optimization trials
    opt_timeout: int = 300  # Timeout in seconds for each model's optimization
    
    # Output parameters
    output_dir: str = "results"
    device_id: int = 0  # Default to first CUDA device
    force_cpu: bool = False  # Option to force CPU usage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}
    
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