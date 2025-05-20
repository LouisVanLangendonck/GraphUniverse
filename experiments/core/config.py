"""
Configuration for graph learning experiments.

This module defines the configuration classes and default parameters for experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
import json
import os

@dataclass
class ExperimentConfig:
    """Configuration for graph learning experiments."""
    
    # Task configuration
    tasks: List[str] = field(default_factory=lambda: ['community', 'k_hop_community_counts'])
    khop_community_counts_k: int = 2  # Number of hops for community counts task
    
    # Model configuration
    gnn_types: List[str] = field(default_factory=lambda: ['gcn', 'gat', 'sage'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    
    # Training configuration
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    batch_size: int = 32
    
    # Model architecture
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    
    # Hyperparameter optimization
    optimize_hyperparams: bool = False
    n_trials: int = 10
    optimization_timeout: int = 300  # Timeout in seconds for hyperparameter optimization
    
    # Output configuration
    output_dir: str = 'results'
    
    # Task-specific parameters
    is_regression: Dict[str, bool] = field(default_factory=lambda: {
        'community': False,
        'k_hop_community_counts': True
    })
    
    # Regression-specific parameters
    regression_loss: str = 'mse'  # 'mse' or 'mae'
    regression_metrics: List[str] = field(default_factory=lambda: ['mse', 'rmse', 'mae', 'r2'])
    
    # Graph generation parameters
    num_communities: int = 5
    num_nodes: int = 100
    feature_dim: int = 32
    edge_density: float = 0.1  # Overall target edge density
    homophily: float = 0.8  # Controls ratio between intra/inter probabilities (0=equal, 1=max)
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
    mixed_membership: bool = False  # Default to non-mixed membership
    
    # Feature regime parameters
    regimes_per_community: int = 2
    intra_community_regime_similarity: float = 0.2  # Default to low intra-community similarity
    inter_community_regime_similarity: float = 0.9  # Default to high inter-community similarity
    feature_regime_balance: float = 0.3  # Default to unbalanced regimes
    
    # Task parameters
    regime_task_min_hop: int = 1
    regime_task_max_hop: int = 3
    regime_task_n_labels: int = 4
    regime_task_min_support: float = 0.1
    regime_task_max_rules_per_label: int = 3
    role_task_max_motif_size: int = 3
    role_task_n_roles: int = 5
    
    # Training parameters
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    distribution_type: str = "standard"
    power_law_exponent: Optional[float] = None
    power_law_target_avg_degree: Optional[float] = None
    exponential_rate: Optional[float] = None
    exponential_target_avg_degree: Optional[float] = None
    uniform_min_factor: Optional[float] = None
    uniform_max_factor: Optional[float] = None
    uniform_target_avg_degree: Optional[float] = None
    max_mean_community_deviation: float = 0.1
    max_max_community_deviation: float = 0.2
    parameter_search_range: float = 0.2
    max_parameter_search_attempts: int = 20
    max_retries: int = 10
    
    # Output parameters
    device_id: int = 0  # Default to first CUDA device
    force_cpu: bool = False  # Option to force CPU usage
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate tasks
        valid_tasks = ['community', 'k_hop_community_counts']
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
        
        # Validate GNN types
        valid_gnn_types = ['gcn', 'gat', 'sage']
        for gnn_type in self.gnn_types:
            if gnn_type not in valid_gnn_types:
                raise ValueError(f"Invalid GNN type: {gnn_type}. Must be one of {valid_gnn_types}")
        
        # Validate regression loss
        valid_losses = ['mse', 'mae']
        if self.regression_loss not in valid_losses:
            raise ValueError(f"Invalid regression loss: {self.regression_loss}. Must be one of {valid_losses}")
        
        # Validate regression metrics
        valid_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in self.regression_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid regression metric: {metric}. Must be one of {valid_metrics}")
    
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