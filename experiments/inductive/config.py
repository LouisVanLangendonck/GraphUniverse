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
    use_parallel_training: bool = True
    max_parallel_gpu_jobs: Optional[int] = None 
    cross_task_parallelization: bool = False
    
    # Performance optimization settings
    use_mixed_precision: bool = False

    # === SSL FINE-TUNING CONFIGURATION ===
    use_pretrained: bool = False
    pretrained_model_dir: str = "ssl_experiments"
    pretrained_model_id: Optional[str] = None  # Specific model to use
    graph_family_id: Optional[str] = None      # Specific family to use
    graph_family_dir: str = "graph_families"   # Family directory
    auto_load_family: bool = True              # Auto-load family from model
    freeze_encoder: bool = False
    only_pretrained_experiments: bool = False
    max_train_graphs_for_finetuning: int = 10
    minimum_train_graphs_to_cover_k: bool = True
    calculate_silhouette_score: bool = True
    
    # === GRAPH FAMILY GENERATION ===
    n_graphs: int = 50
    min_n_nodes: int = 80
    max_n_nodes: int = 120
    min_communities: int = 3
    max_communities: int = 7
    distributional_shift_in_eval: bool = True # Whether to shift the validation set
    distributional_shift_test_only: bool = False # Whether to shift the test set only
    distributional_shift_in_eval_type: str = 'unseen_community_combinations' # Type of distributional shift to apply in evaluation
    distributional_shift_in_eval_homophily_shift: float = 0.15
    distributional_shift_in_eval_density_shift: float = 0.1
    distributional_shift_in_eval_n_nodes_shift: int = 100

    # === UNIVERSE PARAMETERS ===
    universe_K: int = 10
    universe_feature_dim: int = 32
    universe_edge_density: float = 0.1
    universe_homophily: float = 0.8
    universe_randomness_factor: float = 0.0
    community_density_variation: float = 0.2
    community_cooccurrence_homogeneity: float = 0.7
    triangle_density: float = 0.0
    triangle_community_relation_homogeneity: float = 0.7
    
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
    use_dccc_sbm: bool = True  # If False, uses standard DC-SBM
    
    # === DCCC-SBM PARAMETERS ===
    community_imbalance_range: Tuple[float, float] = (0.0, 0.2)
    degree_separation_range: Tuple[float, float] = (0.5, 1.0)
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
    n_repetitions: int = 3  # Number of random seed repetitions instead of k-fold
    train_graph_ratio: float = 0.5
    val_graph_ratio: float = 0.25
    test_graph_ratio: float = 0.25
    inductive_mode: str = "graph_level"  # "graph_level" or "mixed"
    
    # === TASKS ===
    tasks: List[str] = field(default_factory=lambda: ['community', 'triangle_count'])
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
        'k_hop_community_counts_k10': True, 
        'triangle_count': True
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
        'k_hop_community_counts_k10': False, 
        'triangle_count': True
    })
    khop_community_counts_k: int = 2

    # === METAPATH TASK SETTINGS ===
    enable_metapath_tasks: bool = False
    metapath_k_values: List[int] = field(default_factory=lambda: [4, 5])  # 4+ for proper loops
    metapath_require_loop: bool = True
    metapath_degree_weight: float = 0.3
    max_community_participation: float = 0.95
    n_candidates_per_k: int = 30
    
    # === MODELS ===
    gnn_types: List[str] = field(default_factory=lambda: ['gcn', 'sage', 'gat', 'fagcn', 'gin'])
    run_gnn: bool = True
    run_mlp: bool = True
    run_rf: bool = True
    differentiate_with_and_without_PE: bool = True
    
    # === TRAINING ===
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    optimized_patience: Optional[int] = None  # Store optimized patience value if available
    batch_size: int = 1
    hidden_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.5
    # Special parameter for FAGCN
    eps: float = 0.2

    # === NEURAL SHEAF DIFFUSION ===
    run_neural_sheaf: bool = False
    sheaf_type: str = "diagonal" # "diagonal", "bundle", "general"
    sheaf_d: int = 2

    # === GRAPH TRANSFORMER CONFIGURATION ===
    transformer_types: List[str] = field(default_factory=lambda: ['graphgps'])
    run_transformers: bool = False
    
    # Transformer-specific parameters
    transformer_num_heads: int = 8
    transformer_max_nodes: int = 200
    transformer_max_path_length: int = 10
    transformer_precompute_encodings: bool = True
    transformer_cache_encodings: bool = False
    
    # GraphGPS specific
    local_gnn_type: str = "gcn"
    global_model_type: str = "transformer"
    transformer_prenorm: bool = True
    pe_type: List[str] = 'laplacian' # Choose from None, 'laplacian', 'degree', 'rwse'
    max_pe_dim: int = 8
    precompute_pe: bool = True
    pe_norm_type: str = None # 'layer', 'batch', 'instance', 'graph', None
    
    # === HYPERPARAMETER OPTIMIZATION ===
    optimize_hyperparams: bool = False
    n_trials: int = 20
    trial_epochs: int = 30
    optimization_timeout: int = 600
    
    # === ANALYSIS ===
    min_family_consistency: float = 0.1
    require_consistency_check: bool = False
    collect_family_stats: bool = True
    collect_signal_metrics: bool = True
    save_individual_graphs: bool = False
    
    # Regression-specific parameters
    regression_loss: str = 'mse'  # 'mse' or 'mae'
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
class PreTrainingConfig:
    """Configuration for self-supervised pre-training experiments."""
    
    # === EXPERIMENT SETUP ===
    output_dir: str = "ssl_experiments"
    experiment_name: str = "ssl_experiment"
    seed: int = 42
    device_id: int = 0
    force_cpu: bool = False

    # === AMP CONFIGURATION ===
    use_mixed_precision: bool = False
    
    # === PRE-TRAINING TASK ===
    pretraining_task: str = "link_prediction"  # "link_prediction" or "dgi" or "graphcl"
    
    # === DGI SPECIFIC PARAMETERS ===
    dgi_corruption_type: str = "feature_shuffle"  # "feature_shuffle", "edge_dropout", "subgraph"
    dgi_noise_std: float = 0.1
    dgi_perturb_rate: float = 0.1
    dgi_corruption_rate: float = 0.2
    num_corruptions: int = 1  # Number of corrupted versions to generate
    use_infonce: bool = False  # Whether to use InfoNCE loss instead of BCE
    temperature: float = 0.1  # Temperature parameter for InfoNCE loss

    # === GRAPHMAE SPECIFIC PARAMETERS ===
    graphmae_mask_rate: float = 0.5
    graphmae_replace_rate: float = 0.1
    graphmae_gamma: float = 2.0
    graphmae_decoder_type: str = "gnn"
    graphmae_decoder_gnn_type: str = "gcn"
    
    # === GRAPHCL SPECIFIC PARAMETERS ===
    augmentation_types: List[str] = field(default_factory=lambda: ["edge_dropout", "feature_dropout", "subgraph"])
    num_augmentations: int = 2  # Number of augmentations to apply per view
    
    # === GRAPH FAMILY PERSISTENCE ===
    n_extra_graphs_for_finetuning: int = 10
    save_graph_family: bool = True
    graph_family_dir: str = "graph_families"
    
    # === GRAPH SPLIT RATIOS ===
    pretraining_graph_ratio: float = 0.8  # Ratio of graphs used for pre-training
    warmup_graph_ratio: float = 0.2  # Ratio of pre-training graphs used for warmup
    
    # === FAMILY GENERATION ===
    n_graphs: int = 50
    min_n_nodes: int = 80
    max_n_nodes: int = 120
    min_communities: int = 3
    max_communities: int = 7
    universe_K: int = 10
    universe_feature_dim: int = 32
    universe_edge_density: float = 0.1
    universe_homophily: float = 0.8
    universe_randomness_factor: float = 0.8
    use_dccc_sbm: bool = False
    degree_distribution: str = "power_law"
    cluster_count_factor: float = 1.0
    center_variance: float = 1.0
    cluster_variance: float = 0.1
    community_density_variation: float = 0.2
    community_cooccurrence_homogeneity: float = 0.6
    triangle_density: float = 0.0
    triangle_community_relation_homogeneity: float = 0.5

    assignment_skewness: float = 0.0
    community_exclusivity: float = 1.0
    degree_center_method: str = "linear"
    degree_heterogeneity: float = 0.5
    edge_noise: float = 0.1
    max_parameter_search_attempts: int = 20
    parameter_search_range: float = 0.5
    max_retries: int = 20
    min_edge_density: float = 0.005
    disable_deviation_limiting: bool = False
    max_mean_community_deviation: float = 0.15
    max_max_community_deviation: float = 0.15
    min_component_size: int = 10
    homophily_range: Tuple[float, float] = (0.0, 0.2)
    density_range: Tuple[float, float] = (0.0, 0.1)
    community_imbalance_range: Tuple[float, float] = (0.0, 0.3)
    degree_separation_range: Tuple[float, float] = (0.0, 1.0)
    power_law_exponent_range: Tuple[float, float] = (2.0, 3.5)
    exponential_rate_range: Tuple[float, float] = (0.3, 1.0)
    uniform_min_factor_range: Tuple[float, float] = (0.3, 0.7)
    uniform_max_factor_range: Tuple[float, float] = (1.3, 2.0)
    
    # === MODEL CONFIGURATION ===
    model_type: str = "gnn"  # "gnn" or "transformer"
    gnn_type: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # === GRAPH TRANSFORMER CONFIGURATION ===
    transformer_type: str = "graphormer"  # "graphormer" or "graphgps"
    run_transformers: bool = False
    transformer_num_heads: int = 8
    transformer_max_nodes: int = 200
    transformer_max_path_length: int = 10
    transformer_precompute_encodings: bool = True
    transformer_cache_encodings: bool = True
    local_gnn_type: str = "gcn"
    global_model_type: str = "transformer"
    transformer_prenorm: bool = True
    residual: bool = True  # Whether to use residual connections in transformer
    norm_type: str = "layer"  # Type of normalization to use ("layer", "none")
    agg_type: str = "sum"  # Type of aggregation to use ("mean", "max", or "sum")
    
    # GAT-specific parameters
    heads: int = 1  # Number of attention heads for GAT
    concat_heads: bool = True  # Whether to concatenate or average attention heads

    # FAGCN-specific parameters
    eps: float = 0.2
    
    # === TRAINING PARAMETERS ===
    epochs: int = 300
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    batch_size: int = 32
    patience: int = 50
    
    # === HYPERPARAMETER OPTIMIZATION ===
    optimize_hyperparams: bool = True
    n_trials: int = 20
    optimization_timeout: int = 1200
    
    # === TASK-SPECIFIC PARAMETERS ===
    negative_sampling_ratio: float = 1.0
    link_pred_loss: str = "bce"

    # Older deprecated parameters
    contrastive_temperature: float = 0.1
    corruption_rate: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        valid_tasks = ["link_prediction", "dgi", "graphcl"]
        if self.pretraining_task not in valid_tasks:
            raise ValueError(f"pretraining_task must be one of {valid_tasks}")
        
        # Validate DGI parameters
        if self.pretraining_task == "dgi":
            valid_corruption_types = ["feature_shuffle", "edge_dropout", "feature_noise", "edge_perturbation", "feature_dropout"]
            if self.dgi_corruption_type not in valid_corruption_types:
                raise ValueError(f"corruption_type must be one of {valid_corruption_types}")
            if self.num_corruptions < 1:
                raise ValueError("num_corruptions must be >= 1")
            if self.temperature <= 0:
                raise ValueError("temperature must be > 0")
        
        # Validate GraphCL parameters
        if self.pretraining_task == "graphcl":
            valid_augmentation_types = ["edge_dropout", "feature_dropout", "subgraph"]
            for aug_type in self.augmentation_types:
                if aug_type not in valid_augmentation_types:
                    raise ValueError(f"augmentation_type must be one of {valid_augmentation_types}")
            if self.num_augmentations < 1:
                raise ValueError("num_augmentations must be >= 1")

        if self.pretraining_task == "graphmae":
            if self.graphmae_mask_rate <= 0 or self.graphmae_mask_rate >= 1:
                raise ValueError("graphmae_mask_rate must be in (0, 1)")
            if self.graphmae_replace_rate <= 0 or self.graphmae_replace_rate >= 1:
                raise ValueError("graphmae_replace_rate must be in (0, 1)")
            if self.graphmae_gamma <= 0:
                raise ValueError("graphmae_gamma must be > 0")
        
        # Set model_type based on run_transformers
        if self.run_transformers:
            self.model_type = "transformer"
        
        valid_model_types = ["gnn", "transformer"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}")
        
        valid_gnn_types = ["gcn", "sage", "gat", "fagcn", "gin"]
        if self.gnn_type not in valid_gnn_types:
            raise ValueError(f"gnn_type must be one of {valid_gnn_types}")
        
        if self.n_extra_graphs_for_finetuning < 1:
            raise ValueError("n_extra_graphs_for_finetuning must be >= 1")
        
        if not (0.0 < self.pretraining_graph_ratio <= 1.0):
            raise ValueError("pretraining_graph_ratio must be in (0, 1]")
        
        if not (0.0 < self.warmup_graph_ratio <= 1.0):
            raise ValueError("warmup_graph_ratio must be in (0, 1]")
        
        # Validate DCCC-SBM parameters
        if self.use_dccc_sbm:
            valid_distributions = ["standard", "power_law", "exponential", "uniform"]
            if self.degree_distribution not in valid_distributions:
                raise ValueError(f"Invalid degree distribution: {self.degree_distribution}")
    
    def get_total_graphs(self) -> int:
        """Get total number of graphs that will be generated."""
        return self.n_graphs + self.n_extra_graphs_for_finetuning
    
    def get_graph_splits(self, n_generated_graphs: int = None) -> Tuple[int, int, int]:
        """Get (pretraining, warmup, finetuning) graph counts."""
        if n_generated_graphs is None:
            total_pretraining = self.n_graphs
            print(f"n_generated_graphs: {n_generated_graphs}")
        else:
            print(f"n_generated_graphs: {n_generated_graphs}")
            total_pretraining = n_generated_graphs
        n_warmup = int(total_pretraining * self.warmup_graph_ratio)
        n_finetuning = self.n_extra_graphs_for_finetuning
        n_actual_pretraining = total_pretraining - n_warmup - n_finetuning
        
        return n_actual_pretraining, n_warmup, n_finetuning
    
    def to_graph_family_params(self) -> Dict[str, Any]:
        """Convert to parameters for GraphFamilyGenerator."""
        return {
            'n_graphs': self.get_total_graphs(),
            'min_n_nodes': self.min_n_nodes,
            'max_n_nodes': self.max_n_nodes,
            'min_communities': self.min_communities,
            'max_communities': self.max_communities,
            'min_component_size': self.min_component_size,
            'homophily_range': self.homophily_range,
            'density_range': self.density_range,
            'use_dccc_sbm': self.use_dccc_sbm,
            'community_imbalance_range': self.community_imbalance_range,
            'degree_separation_range': self.degree_separation_range,
            'degree_distribution': self.degree_distribution,
            'power_law_exponent_range': self.power_law_exponent_range,
            'exponential_rate_range': self.exponential_rate_range,
            'uniform_min_factor_range': self.uniform_min_factor_range,
            'uniform_max_factor_range': self.uniform_max_factor_range,
            'degree_heterogeneity': self.degree_heterogeneity,
            'edge_noise': self.edge_noise,
            'max_parameter_search_attempts': self.max_parameter_search_attempts,
            'parameter_search_range': self.parameter_search_range,
            'max_retries': self.max_retries,
            'min_edge_density': self.min_edge_density,
            'disable_deviation_limiting': self.disable_deviation_limiting,
            'max_mean_community_deviation': self.max_mean_community_deviation,
            'max_max_community_deviation': self.max_max_community_deviation,
            'seed': self.seed
        }
    
    def to_universe_params(self) -> Dict[str, Any]:
        """Convert to parameters for GraphUniverse."""
        return {
            'K': self.universe_K,
            'feature_dim': self.universe_feature_dim,
            'edge_density': self.universe_edge_density,
            'homophily': self.universe_homophily,
            'randomness_factor': self.universe_randomness_factor,
            'cluster_count_factor': self.cluster_count_factor,
            'center_variance': self.center_variance,
            'cluster_variance': self.cluster_variance,
            'assignment_skewness': self.assignment_skewness,
            'community_exclusivity': self.community_exclusivity,
            'degree_center_method': self.degree_center_method,
            'community_density_variation': self.community_density_variation,
            'community_cooccurrence_homogeneity': self.community_cooccurrence_homogeneity,
            'triangle_density': self.triangle_density,
            'triangle_community_relation_homogeneity': self.triangle_community_relation_homogeneity,
            'seed': self.seed
        }
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreTrainingConfig':
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
    def load(cls, filepath: str) -> 'PreTrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


