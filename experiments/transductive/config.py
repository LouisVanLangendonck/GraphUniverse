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
    k_fold: int = 3  # Number of folds for cross-validation

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
    center_variance: float = 0.1
    cluster_variance: float = 0.5
    assignment_skewness: float = 0.0
    community_exclusivity: float = 1.0
    degree_center_method: str = "linear"  # "linear", "random", "shuffled"
    
    # === GRAPH VARIATION ===
    homophily_range: Tuple[float, float] = (0.0, 0.0)
    density_range: Tuple[float, float] = (0.0, 0.0)
    degree_heterogeneity: float = 0.5
    edge_noise: float = 0.1
    
    # === GENERATION METHOD SELECTION ===
    use_dccc_sbm: bool = False  # If False, uses standard DC-SBM
    
    # === DCCC-SBM PARAMETERS ===
    community_imbalance_range: Tuple[float, float] = (0.0, 0.0)
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
    parameter_search_range: float = 0.0  # Always 0 for single graph
    max_retries: int = 10
    min_edge_density: float = 0.005
    disable_deviation_limiting: bool = False
    max_mean_community_deviation: float = 0.90
    max_max_community_deviation: float = 0.90
    min_component_size: int = 4
    
    # === TRANSDUCTIVE DATA SPLITS ===
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    # Fallback options:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # === TASKS ===
    tasks: List[str] = field(default_factory=lambda: ['community', 'k_hop_community_counts_k1', 'k_hop_community_counts_k2'])
    is_regression: Dict[str, bool] = field(default_factory=lambda: {
        'community': False,
        'k_hop_community_counts_k1': True,
        'k_hop_community_counts_k2': True,
        'k_hop_community_counts_k3': True,
        'k_hop_community_counts_k4': True,
        'k_hop_community_counts_k5': True,
        'k_hop_community_counts_k6': True,
        'k_hop_community_counts_k7': True,
        'k_hop_community_counts_k8': True,
        'k_hop_community_counts_k9': True,
        'k_hop_community_counts_k10': True
    })
    is_graph_level_tasks: Dict[str, bool] = field(default_factory=lambda: {
        'community': False,
        'k_hop_community_counts_k1': False,
        'k_hop_community_counts_k2': False,
        'k_hop_community_counts_k3': False,
        'k_hop_community_counts_k4': False,
        'k_hop_community_counts_k5': False,
        'k_hop_community_counts_k6': False,
        'k_hop_community_counts_k7': False,
        'k_hop_community_counts_k8': False,
        'k_hop_community_counts_k9': False,
        'k_hop_community_counts_k10': False
    })
    khop_community_counts_k: int = 2

    # === METAPATH TASK SETTINGS ===
    enable_metapath_tasks: bool = False
    metapath_k_values: List[int] = field(default_factory=lambda: [4, 5])
    metapath_require_loop: bool = True
    metapath_degree_weight: float = 0.3
    max_community_participation: float = 0.95
    
    # === MODELS ===
    gnn_types: List[str] = field(default_factory=lambda: ['gcn', 'sage', 'gat', 'fagcn', 'gin'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    run_neural_sheaf: bool = False
    sheaf_type: str = "diagonal" # "diagonal", "bundle", "general"
    sheaf_d: int = 2
    differentiate_with_and_without_PE: bool = True
    
    # === TRAINING ===
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 100
    batch_size: int = 32  # Not used in transductive but kept for compatibility
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    # Special parameter for FAGCN
    eps: float = 0.2

    # === GRAPH TRANSFORMER CONFIGURATION ===
    transformer_types: List[str] = field(default_factory=lambda: ['graphgps'])
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
    
    # === POSITIONAL ENCODING ===
    pe_type: str = 'laplacian' # Choose from None, 'laplacian', 'degree', 'rwse'
    max_pe_dim: int = 8
    
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
        # Validate range parameters for single graph
        for rng in [self.homophily_range, self.density_range, self.community_imbalance_range, self.degree_separation_range]:
            if rng[0] != 0.0 or rng[1] != 0.0:
                print("Warning: All range parameters should be (0.0, 0.0) for single-graph transductive experiments.")
        # Ensure is_regression and is_graph_level_tasks are always set
        if self.is_regression is None:
            self.is_regression = {
                'community': False,
                'k_hop_community_counts_k1': True,
                'k_hop_community_counts_k2': True,
                'k_hop_community_counts_k3': True,
                'k_hop_community_counts_k4': True,
                'k_hop_community_counts_k5': True,
                'k_hop_community_counts_k6': True,
                'k_hop_community_counts_k7': True,
                'k_hop_community_counts_k8': True,
                'k_hop_community_counts_k9': True,
                'k_hop_community_counts_k10': True
            }
        if self.is_graph_level_tasks is None:
            self.is_graph_level_tasks = {
                'community': False,
                'k_hop_community_counts_k1': False,
                'k_hop_community_counts_k2': False,
                'k_hop_community_counts_k3': False,
                'k_hop_community_counts_k4': False,
                'k_hop_community_counts_k5': False,
                'k_hop_community_counts_k6': False,
                'k_hop_community_counts_k7': False,
                'k_hop_community_counts_k8': False,
                'k_hop_community_counts_k9': False,
                'k_hop_community_counts_k10': False
            }
    
    def get_splits(self) -> Tuple[int, int, int]:
        """Calculate number of nodes for each split."""
        if self.n_val is not None and self.n_test is not None:
            n_val = self.n_val
            n_test = self.n_test
            n_train = self.num_nodes - n_val - n_test
        else:
            n_train = int(self.num_nodes * self.train_ratio)
            n_val = int(self.num_nodes * self.val_ratio)
            n_test = self.num_nodes - n_train - n_val

        print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
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