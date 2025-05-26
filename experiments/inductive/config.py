"""
Configuration for inductive graph learning experiments.

This module defines configuration for experiments using graph families
where models are trained on some graphs and tested on others.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
import os
from experiments.core.config import ExperimentConfig

@dataclass
class InductiveExperimentConfig(ExperimentConfig):
    """Configuration for inductive graph learning experiments using graph families."""
    
    # Graph family generation parameters
    n_graphs: int = 50  # Total number of graphs to generate
    min_n_nodes: int = 80
    max_n_nodes: int = 120
    min_communities: int = 3
    max_communities: int = 7
    
    # Graph family variation parameters
    homophily_range: tuple = (0.0, 0.2)  # Range around universe homophily
    density_range: tuple = (0.0, 0.2)    # Range around universe density
    
    # Graph generation method
    use_dccc_sbm: bool = False  # If False, uses standard DC-SBM
    
    # DCCC-SBM parameters (only used if use_dccc_sbm=True)
    community_imbalance_range: tuple = (0.0, 0.3)  # Range for community size imbalance
    degree_separation_range: tuple = (0.0, 1.0)    # Range for degree distribution separation
    
    # Degree distribution type for DCCC-SBM
    degree_distribution: str = "standard"  # "standard", "power_law", "exponential", "uniform"
    
    # Power law distribution parameters (used if degree_distribution="power_law")
    power_law_exponent_range: tuple = (2.0, 3.5)
    power_law_x_min: float = 1.0
    
    # Exponential distribution parameters (used if degree_distribution="exponential")
    exponential_rate_range: tuple = (0.3, 1.0)
    
    # Uniform distribution parameters (used if degree_distribution="uniform")
    uniform_min_factor_range: tuple = (0.3, 0.7)
    uniform_max_factor_range: tuple = (1.3, 2.0)
    
    # Target average degree range for all DCCC distributions
    dccc_target_avg_degree_range: tuple = (2.0, 10.0)
    
    # Fixed parameters for all graphs in family
    degree_heterogeneity: float = 0.5
    edge_noise: float = 0.1
    target_avg_degree: Optional[float] = None
    triangle_enhancement: float = 0.0
    
    # Graph generation constraints
    max_parameter_search_attempts: int = 20
    parameter_search_range: float = 0.5
    max_retries: int = 5
    min_edge_density: float = 0.005
    disable_deviation_limiting: bool = False
    max_mean_community_deviation: float = 0.15
    max_max_community_deviation: float = 0.3
    
    # Data split configuration
    train_graph_ratio: float = 0.6  # Fraction of graphs for training
    val_graph_ratio: float = 0.2    # Fraction of graphs for validation
    test_graph_ratio: float = 0.2   # Fraction of graphs for testing
    
    # Inductive learning mode
    inductive_mode: str = "graph_level"  # "graph_level" or "mixed"
    # - "graph_level": completely separate graphs for train/val/test
    # - "mixed": some overlap allowed but focus on generalization
    
    # Node-level splits within each graph (if needed)
    use_node_splits: bool = False  # Whether to also split nodes within graphs
    node_train_ratio: float = 0.8  # Only used if use_node_splits=True
    node_val_ratio: float = 0.1
    node_test_ratio: float = 0.1
    
    # Family consistency requirements
    min_family_consistency: float = 0.5  # Minimum consistency score required
    require_consistency_check: bool = True
    
    # Universe parameters (for family generation)
    universe_K: int = 10  # Number of communities in universe
    universe_feature_dim: int = 32
    universe_block_structure: str = "assortative"
    universe_edge_density: float = 0.1
    universe_homophily: float = 0.8
    universe_randomness_factor: float = 0.0
    
    # Feature generation parameters for universe
    cluster_count_factor: float = 1.0
    center_variance: float = 1.0
    cluster_variance: float = 0.1
    assignment_skewness: float = 0.0
    community_exclusivity: float = 1.0
    degree_center_method: str = "linear"
    
    # Experiment control
    collect_family_stats: bool = True
    save_individual_graphs: bool = False  # Whether to save each graph separately
    
    def __post_init__(self):
        """Validate inductive configuration after initialization."""
        super().__post_init__()
        
        # Validate graph split ratios
        total_ratio = self.train_graph_ratio + self.val_graph_ratio + self.test_graph_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Graph split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate node split ratios if used
        if self.use_node_splits:
            node_total = self.node_train_ratio + self.node_val_ratio + self.node_test_ratio
            if abs(node_total - 1.0) > 1e-6:
                raise ValueError(f"Node split ratios must sum to 1.0, got {node_total}")
        
        # Validate inductive mode
        valid_modes = ["graph_level", "mixed"]
        if self.inductive_mode not in valid_modes:
            raise ValueError(f"Invalid inductive mode: {self.inductive_mode}. Must be one of {valid_modes}")
        
        # Validate DCCC-SBM parameters
        if self.use_dccc_sbm:
            valid_distributions = ["standard", "power_law", "exponential", "uniform"]
            if self.degree_distribution not in valid_distributions:
                raise ValueError(f"Invalid degree distribution: {self.degree_distribution}. Must be one of {valid_distributions}")
            
            # Validate distribution-specific parameter ranges
            if self.degree_distribution == "power_law":
                if self.power_law_exponent_range[0] <= 1.0:
                    raise ValueError("Power law exponent must be > 1.0")
                if self.power_law_x_min <= 0:
                    raise ValueError("Power law x_min must be > 0")
            
            if self.degree_distribution == "exponential":
                if self.exponential_rate_range[0] <= 0:
                    raise ValueError("Exponential rate must be > 0")
            
            if self.degree_distribution == "uniform":
                if self.uniform_min_factor_range[0] <= 0:
                    raise ValueError("Uniform min factor must be > 0")
                if self.uniform_max_factor_range[0] <= 0:
                    raise ValueError("Uniform max factor must be > 0")
            
        # Validate minimum requirements
        if self.n_graphs < 10:
            raise ValueError("Need at least 10 graphs for meaningful inductive experiments")
        
        if self.max_communities > self.universe_K:
            raise ValueError("max_communities cannot exceed universe_K")
    
    def get_graph_splits(self) -> tuple:
        """Calculate number of graphs for each split."""
        n_train = int(self.n_graphs * self.train_graph_ratio)
        n_val = int(self.n_graphs * self.val_graph_ratio)
        n_test = self.n_graphs - n_train - n_val  # Remainder goes to test
        
        return n_train, n_val, n_test