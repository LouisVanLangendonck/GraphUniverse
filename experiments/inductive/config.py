"""
Clean configuration for inductive graph learning experiments.
Removes old inheritance and focuses only on DC-SBM and DCCC-SBM methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import os


@dataclass
class InductiveExperimentConfig:
    """Clean configuration for inductive graph learning experiments."""
    
    # === EXPERIMENT SETUP ===
    output_dir: str = "inductive_results"
    seed: int = 42
    device_id: int = 0
    force_cpu: bool = False

    # === SSL FINE-TUNING CONFIGURATION ===
    use_pretrained: bool = False
    pretrained_model_dir: str = "ssl_experiments"
    pretrained_model_id: Optional[str] = None  # Specific model to use
    graph_family_id: Optional[str] = None      # Specific family to use
    graph_family_dir: str = "graph_families"   # Family directory
    auto_load_family: bool = True              # Auto-load family from model
    freeze_encoder: bool = False
    compare_pretrained: bool = False
    fine_tune_lr_multiplier: float = 0.1
    
    # === GRAPH FAMILY GENERATION ===
    n_graphs: int = 50
    min_n_nodes: int = 80
    max_n_nodes: int = 120
    min_communities: int = 3
    max_communities: int = 7
    
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
    
    # === GRAPH FAMILY VARIATION ===
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
    
    # === DATA SPLITS ===
    train_graph_ratio: float = 0.5
    val_graph_ratio: float = 0.25
    test_graph_ratio: float = 0.25
    inductive_mode: str = "graph_level"  # "graph_level" or "mixed"
    
    # === TASKS ===
    tasks: List[str] = field(default_factory=lambda: ['community'])
    is_regression: Dict[str, bool] = field(default_factory=lambda: {'community': False})
    khop_community_counts_k: int = 2

    # === METAPATH TASK SETTINGS ===
    enable_metapath_tasks: bool = False
    metapath_k_values: List[int] = field(default_factory=lambda: [4, 5])  # 4+ for proper loops
    metapath_require_loop: bool = True
    metapath_degree_weight: float = 0.3
    max_community_participation: float = 0.95
    n_candidates_per_k: int = 30
    
    # === MODELS ===
    gnn_types: List[str] = field(default_factory=lambda: ['gcn', 'sage'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    
    # === TRAINING ===
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    batch_size: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    
    # === HYPERPARAMETER OPTIMIZATION ===
    optimize_hyperparams: bool = False
    n_trials: int = 20
    optimization_timeout: int = 600
    
    # === ANALYSIS ===
    min_family_consistency: float = 0.1
    require_consistency_check: bool = False
    collect_family_stats: bool = True
    collect_signal_metrics: bool = True  # NEW: collect community signals
    save_individual_graphs: bool = False
    
    # Regression-specific parameters
    regression_loss: str = 'mae'  # 'mse' or 'mae'
    regression_metrics: List[str] = field(default_factory=lambda: ['mae', 'mse', 'rmse', 'r2'])
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate graph split ratios
        total_ratio = self.train_graph_ratio + self.val_graph_ratio + self.test_graph_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Graph split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate inductive mode
        if self.inductive_mode not in ["graph_level", "mixed"]:
            raise ValueError(f"Invalid inductive mode: {self.inductive_mode}")
        
        # Validate DCCC-SBM parameters
        if self.use_dccc_sbm:
            valid_distributions = ["standard", "power_law", "exponential", "uniform"]
            if self.degree_distribution not in valid_distributions:
                raise ValueError(f"Invalid degree distribution: {self.degree_distribution}")
        
        # Validate minimum requirements
        if self.n_graphs < 10:
            raise ValueError("Need at least 10 graphs for meaningful inductive experiments")
        
        if self.max_communities > self.universe_K:
            raise ValueError("max_communities cannot exceed universe_K")
    
    def get_graph_splits(self) -> Tuple[int, int, int]:
        """Calculate number of graphs for each split."""
        n_train = int(self.n_graphs * self.train_graph_ratio)
        n_val = int(self.n_graphs * self.val_graph_ratio)
        n_test = self.n_graphs - n_train - n_val
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InductiveExperimentConfig':
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
    def load(cls, filepath: str) -> 'InductiveExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

@dataclass
class SSLInductiveConfig(InductiveExperimentConfig):
    """Extended inductive config with SSL pre-training options."""
    
    # === SSL PRE-TRAINING OPTIONS ===
    use_pretrained_models: bool = False
    pretrained_model_dir: str = "pretrained_models"
    
    # Pre-training configuration (if training from scratch)
    pretrain_from_scratch: bool = False
    pretraining_tasks: List[str] = field(default_factory=lambda: ["link_prediction"])
    pretraining_gnn_types: List[str] = field(default_factory=lambda: ["gcn", "sage"])
    
    # Fine-tuning options
    freeze_encoder: bool = False  # Whether to freeze pre-trained encoder during fine-tuning
    fine_tune_lr_multiplier: float = 0.1  # Learning rate multiplier for fine-tuning
    
    # Model matching
    auto_match_pretrained: bool = True  # Automatically match pre-trained models to downstream tasks
    fallback_to_random: bool = True  # Fall back to random initialization if no pre-trained model found

@dataclass
class PreTrainingConfig:
    """Configuration for self-supervised pre-training."""
    
    # === EXPERIMENT SETUP ===
    output_dir: str = "pretrained_models"
    experiment_name: str = "ssl_pretraining"
    seed: int = 42
    device_id: int = 0
    force_cpu: bool = False
    
    # === PRE-TRAINING TASK ===
    pretraining_task: str = "link_prediction"  # "link_prediction", "contrastive", "both"

    # === GRAPH FAMILY PERSISTENCE ===
    n_extra_graphs_for_finetuning: int = 30  # Extra graphs to generate for later fine-tuning
    save_graph_family: bool = True  # Save the entire graph family
    graph_family_dir: str = "graph_families"  # Directory to save graph families
    
    # === PRE-TRAINING DATA SPLIT ===
    pretraining_graph_ratio: float = 0.7  # Ratio of total graphs used for pre-training
    warmup_graph_ratio: float = 0.3  # Ratio of pre-training graphs used for hyperopt warmup
    
    # === FAMILY GENERATION ===
    # These will be used to generate the graph family
    n_graphs: int = 50  # Base number of graphs for pre-training
    min_n_nodes: int = 80
    max_n_nodes: int = 120
    min_communities: int = 3
    max_communities: int = 7
    universe_K: int = 10
    universe_feature_dim: int = 32
    universe_edge_density: float = 0.1
    universe_homophily: float = 0.8
    use_dccc_sbm: bool = False
    degree_distribution: str = "power_law"
    
    # === MODEL CONFIGURATION ===
    gnn_type: str = "gcn"  # "gcn", "sage", "gat"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # === TRAINING PARAMETERS ===
    epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    patience: int = 50
    
    # === HYPERPARAMETER OPTIMIZATION ===
    optimize_hyperparams: bool = True
    n_warmup_graphs: int = 5  # Number of graphs for hyperparameter tuning
    n_trials: int = 20
    optimization_timeout: int = 1200  # 20 minutes
    
    # === TASK-SPECIFIC PARAMETERS ===
    # Link prediction
    negative_sampling_ratio: float = 1.0
    link_pred_loss: str = "bce"  # "bce", "margin"
    
    # Contrastive learning (Deep Graph InfoMax style)
    contrastive_temperature: float = 0.07
    corruption_type: str = "feature_shuffle"  # "feature_shuffle", "edge_dropout"
    corruption_rate: float = 0.2
    
    # === SAVING CONFIGURATION ===
    save_checkpoints: bool = True
    checkpoint_frequency: int = 50  # Save every N epochs
    save_best_only: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        valid_tasks = ["link_prediction", "contrastive", "both"]
        if self.pretraining_task not in valid_tasks:
            raise ValueError(f"pretraining_task must be one of {valid_tasks}")
        
        valid_gnn_types = ["gcn", "sage", "gat"]
        if self.gnn_type not in valid_gnn_types:
            raise ValueError(f"gnn_type must be one of {valid_gnn_types}")
        
        if self.n_extra_graphs_for_finetuning < 0:
            raise ValueError("n_extra_graphs_for_finetuning must be >= 0")
        
        if not (0.0 < self.pretraining_graph_ratio <= 1.0):
            raise ValueError("pretraining_graph_ratio must be in (0, 1]")
        
        if not (0.0 < self.warmup_graph_ratio <= 1.0):
            raise ValueError("warmup_graph_ratio must be in (0, 1]")
        
    def get_total_graphs(self) -> int:
        """Get total number of graphs that will be generated."""
        return self.n_graphs + self.n_extra_graphs_for_finetuning
    
    def get_graph_splits(self) -> Tuple[int, int, int]:
        """Get (pretraining, warmup, finetuning) graph counts."""
        total_pretraining = self.n_graphs
        n_warmup = int(total_pretraining * self.warmup_graph_ratio)
        n_actual_pretraining = total_pretraining - n_warmup
        n_finetuning = self.n_extra_graphs_for_finetuning
        
        return n_actual_pretraining, n_warmup, n_finetuning
