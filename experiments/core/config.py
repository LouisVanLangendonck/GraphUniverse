"""
Clean configuration for transductive graph learning experiments.
Based on the inductive experiment configuration but adapted for transductive learning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import os


@dataclass
class TransductiveExperimentConfig:
    """Clean configuration for transductive graph learning experiments."""
    
    # === EXPERIMENT SETUP ===
    output_dir: str = "transductive_results"
    seed: int = 42
    device_id: int = 0
    force_cpu: bool = False

    # === GRAPH GENERATION ===
    num_nodes: int = 100
    num_communities: int = 5
    
    # === UNIVERSE PARAMETERS ===
    universe_K: int = 10
    universe_feature_dim: int = 32
    universe_edge_density: float = 0.1
    universe_homophily: float = 0.8
    universe_randomness_factor: float = 0.0
    
    # === FEATURE GENERATION ===
    cluster_count_factor: float = 1.0
    center_variance: float = 1.0
    cluster_variance: float = 0.1
    assignment_skewness: float = 0.0
    community_exclusivity: float = 1.0
    degree_center_method: str = "linear"  # "linear", "random", "shuffled"
    
    # === GRAPH VARIATION ===
    homophily_range: Tuple[float, float] = (0.0, 0.2)
    density_range: Tuple[float, float] = (0.0, 0.2)
    degree_heterogeneity: float = 0.5
    edge_noise: float = 0.1
    
    # === GENERATION METHOD SELECTION ===
    use_dccc_sbm: bool = False  # If False, uses standard DC-SBM
    
    # === DCCC-SBM PARAMETERS ===
    community_imbalance_range: Tuple[float, float] = (0.0, 0.3)
    degree_separation_range: Tuple[float, float] = (0.0, 1.0)
    degree_distribution: str = "power_law"  # "power_law", "exponential", "uniform", "standard"
    
    # Degree distribution specific parameters
    power_law_exponent_range: Tuple[float, float] = (2.0, 3.5)
    power_law_x_min: float = 1.0
    exponential_rate_range: Tuple[float, float] = (0.3, 1.0)
    uniform_min_factor_range: Tuple[float, float] = (0.3, 0.7)
    uniform_max_factor_range: Tuple[float, float] = (1.3, 2.0)
    
    # === GENERATION CONSTRAINTS ===
    max_parameter_search_attempts: int = 20
    parameter_search_range: float = 0.5
    max_retries: int = 10
    min_edge_density: float = 0.005
    disable_deviation_limiting: bool = False
    max_mean_community_deviation: float = 0.10
    max_max_community_deviation: float = 0.20
    min_component_size: int = 10
    
    # === TRANSDUCTIVE DATA SPLITS ===
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    max_training_nodes: Optional[int] = None  # Maximum number of nodes to use for training
    
    # === TASKS ===
    tasks: List[str] = field(default_factory=lambda: ['community'])
    is_regression: Dict[str, bool] = field(default_factory=lambda: {'community': False, 'k_hop_community_counts': True})
    khop_community_counts_k: int = 2

    # === METAPATH TASK SETTINGS ===
    enable_metapath_tasks: bool = False
    metapath_k_values: List[int] = field(default_factory=lambda: [4, 5])
    metapath_require_loop: bool = True
    metapath_degree_weight: float = 0.3
    max_community_participation: float = 0.95
    
    # === MODELS ===
    gnn_types: List[str] = field(default_factory=lambda: ['gcn', 'sage', 'gat', 'fagcn'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    
    # === TRAINING ===
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    batch_size: int = 32  # Not used in transductive but kept for compatibility
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    # Special parameter for FAGCN
    eps: float = 0.2

    # === GRAPH TRANSFORMER CONFIGURATION ===
    transformer_types: List[str] = field(default_factory=lambda: ['graphormer'])
    run_transformers: bool = False
    
    # Transformer-specific parameters
    transformer_num_heads: int = 8
    transformer_max_nodes: int = 200
    transformer_max_path_length: int = 10
    transformer_precompute_encodings: bool = True
    transformer_cache_encodings: bool = True
    
    # GraphGPS specific
    local_gnn_type: str = "gcn"
    global_model_type: str = "transformer"
    transformer_prenorm: bool = True
    
    # === HYPERPARAMETER OPTIMIZATION ===
    optimize_hyperparams: bool = False
    n_trials: int = 20
    optimization_timeout: int = 600
    
    # === ANALYSIS ===
    collect_signal_metrics: bool = True
    
    # Regression-specific parameters
    regression_loss: str = 'mae'  # 'mse' or 'mae'
    regression_metrics: List[str] = field(default_factory=lambda: ['mae', 'mse', 'rmse', 'r2'])
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate split ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate DCCC-SBM parameters
        if self.use_dccc_sbm:
            valid_distributions = ["standard", "power_law", "exponential", "uniform"]
            if self.degree_distribution not in valid_distributions:
                raise ValueError(f"Invalid degree distribution: {self.degree_distribution}")
        
        if self.num_communities > self.universe_K:
            raise ValueError("num_communities cannot exceed universe_K")
    
    def get_splits(self) -> Tuple[int, int, int]:
        """Calculate number of nodes for each split."""
        n_train = int(self.num_nodes * self.train_ratio)
        n_val = int(self.num_nodes * self.val_ratio)
        n_test = self.num_nodes - n_train - n_val
        return n_train, n_val, n_test
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransductiveExperimentConfig':
        """Create config from dictionary."""
        # Convert lists back to tuples where needed
        tuple_fields = [
            'homophily_range', 'density_range', 'community_imbalance_range',
            'degree_separation_range', 'power_law_exponent_range', 
            'exponential_rate_range', 'uniform_min_factor_range',
            'uniform_max_factor_range'
        ]
        
        processed_dict = config_dict.copy()
        for field in tuple_fields:
            if field in processed_dict and isinstance(processed_dict[field], list):
                processed_dict[field] = tuple(processed_dict[field])
        
        return cls(**processed_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TransductiveExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)