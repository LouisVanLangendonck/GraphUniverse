"""
Graph family generation utilities for MMSB graphs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from mmsb.model import GraphUniverse, GraphSample

class GraphFamilyGenerator:
    """
    Generates families of graphs sampled from a graph universe.
    Each graph in the family is a subgraph of the universe with its own community structure.
    """
    
    def __init__(
        self,
        universe: Optional[GraphUniverse] = None,
        K: int = 100,
        feature_dim: int = 64,
        block_structure: str = "assortative",
        overlap_structure: str = "modular",
        edge_density: float = 0.1,
        homophily: float = 0.8,
        randomness_factor: float = 0.0,
        intra_community_regime_similarity: float = 0.8,
        inter_community_regime_similarity: float = 0.2,
        regimes_per_community: int = 2,
        # New parameters for allowed deviations
        homophily_range: float = 0.0,  # Allowed deviation from universe homophily
        density_range: float = 0.0,    # Allowed deviation from universe density
        # New parameters for configuration model
        use_configuration_model: bool = False,
        degree_distribution: str = "power_law",
        power_law_exponent_min: float = 1.5,  # Minimum power law exponent
        power_law_exponent_max: float = 3.0,  # Maximum power law exponent
        target_avg_degree_min: float = 2.0,   # Minimum target average degree
        target_avg_degree_max: float = 20.0,  # Maximum target average degree
        # Deviation constraints from GraphSample
        max_mean_community_deviation: float = 0.1,  # Maximum allowed mean deviation from community structure
        max_max_community_deviation: float = 0.2,   # Maximum allowed maximum deviation from community structure
        max_parameter_search_attempts: int = 10,    # Maximum number of parameter search attempts
        parameter_search_range: float = 0.2,        # How aggressively to search parameter space
        min_edge_density: float = 0.005,           # Minimum acceptable edge density
        max_retries: int = 5,                      # Maximum number of retries if graph is too sparse
        seed: Optional[int] = None
    ):
        """
        Initialize the graph family generator.
        
        Args:
            universe: Optional GraphUniverse instance. If None, creates a new one.
            K: Number of communities in the universe
            feature_dim: Dimension of node features
            block_structure: Type of block structure ("assortative", "disassortative", "core-periphery", "hierarchical")
            overlap_structure: Structure of community overlaps ("modular", "hierarchical", "hub-spoke")
            edge_density: Overall edge density
            homophily: Strength of within-community connections
            randomness_factor: Amount of random noise in edge probabilities
            intra_community_regime_similarity: How similar regimes within same community should be (0-1)
            inter_community_regime_similarity: How similar regimes between communities should be (0-1)
            regimes_per_community: Number of feature regimes per community
            homophily_range: Allowed deviation from universe homophily
            density_range: Allowed deviation from universe density
            use_configuration_model: Whether to use configuration model-like edge generation
            degree_distribution: Type of degree distribution ("power_law", "log_normal", "uniform")
            power_law_exponent_min: Minimum power law exponent
            power_law_exponent_max: Maximum power law exponent
            target_avg_degree_min: Minimum target average degree
            target_avg_degree_max: Maximum target average degree
            max_mean_community_deviation: Maximum allowed mean deviation from community structure
            max_max_community_deviation: Maximum allowed maximum deviation from community structure
            max_parameter_search_attempts: Maximum number of parameter search attempts
            parameter_search_range: How aggressively to search parameter space
            min_edge_density: Minimum acceptable edge density
            max_retries: Maximum number of retries if graph is too sparse
            seed: Random seed for reproducibility
        """
        # Create universe if not provided
        if universe is None:
            self.universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                block_structure=block_structure,
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                regimes_per_community=regimes_per_community,
                seed=seed
            )
        else:
            self.universe = universe
        
        # Store parameters
        self.homophily_range = homophily_range
        self.density_range = density_range
        self.use_configuration_model = use_configuration_model
        self.degree_distribution = degree_distribution
        self.power_law_exponent_min = power_law_exponent_min
        self.power_law_exponent_max = power_law_exponent_max
        self.target_avg_degree_min = target_avg_degree_min
        self.target_avg_degree_max = target_avg_degree_max
        self.max_mean_community_deviation = max_mean_community_deviation
        self.max_max_community_deviation = max_max_community_deviation
        self.max_parameter_search_attempts = max_parameter_search_attempts
        self.parameter_search_range = parameter_search_range
        self.min_edge_density = min_edge_density
        self.max_retries = max_retries
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

    def _sample_target_parameters(self) -> Tuple[float, float]:
        """
        Sample target homophily and density from allowed ranges.
        
        Returns:
            Tuple of (target_homophily, target_density)
        """
        # Sample homophily from range
        if isinstance(self.homophily_range, tuple):
            homophily_min = max(0.0, self.universe.homophily - self.homophily_range[1])
            homophily_max = min(1.0, self.universe.homophily + self.homophily_range[1])
        else:
            homophily_min = max(0.0, self.universe.homophily - self.homophily_range)
            homophily_max = min(1.0, self.universe.homophily + self.homophily_range)
        target_homophily = np.random.uniform(homophily_min, homophily_max)
        
        # Sample density from range
        if isinstance(self.density_range, tuple):
            density_min = max(0.0, self.universe.edge_density - self.density_range[1])
            density_max = min(1.0, self.universe.edge_density + self.density_range[1])
        else:
            density_min = max(0.0, self.universe.edge_density - self.density_range)
            density_max = min(1.0, self.universe.edge_density + self.density_range)
        target_density = np.random.uniform(density_min, density_max)
        
        return target_homophily, target_density

    def generate(
        self,
        n_graphs: int,
        min_communities: int,
        max_communities: int,
        min_nodes: int,
        max_nodes: int,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        sampling_method: str = "random",
        min_component_size: int = 0,
        feature_regime_balance: float = 0.5,
        seed: Optional[int] = None
    ) -> List[GraphSample]:
        """
        Generate a family of graphs.
        
        Args:
            n_graphs: Number of graphs to generate
            min_communities: Minimum number of communities per graph
            max_communities: Maximum number of communities per graph
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            degree_heterogeneity: Controls degree variability (0=homogeneous, 1=highly skewed)
            edge_noise: Random noise in edge probabilities
            sampling_method: Method for sampling communities ("random", "similar", "diverse", "correlated")
            min_component_size: Minimum size for connected components
            feature_regime_balance: How evenly feature regimes are distributed
            seed: Random seed for reproducibility
            
        Returns:
            List of generated GraphSample instances
        """
        if seed is not None:
            np.random.seed(seed)
            
        graphs = []
        attempts = 0
        max_attempts = n_graphs * 3  # Allow up to 3x the target number of attempts
        
        while len(graphs) < n_graphs and attempts < max_attempts:
            try:
                # Sample number of communities
                n_communities = np.random.randint(min_communities, max_communities + 1)
                
                # Sample communities
                communities = self.universe.sample_connected_community_subset(
                    size=n_communities,
                    method=sampling_method
                )
                
                # Sample number of nodes
                n_nodes = np.random.randint(min_nodes, max_nodes + 1)
                
                # Sample target parameters
                target_homophily, target_density = self._sample_target_parameters()
                
                # Sample power law exponent and target average degree if using configuration model
                power_law_exponent = None
                target_avg_degree = None
                if self.use_configuration_model:
                    power_law_exponent = np.random.uniform(
                        self.power_law_exponent_min,
                        self.power_law_exponent_max
                    )
                    target_avg_degree = np.random.uniform(
                        self.target_avg_degree_min,
                        self.target_avg_degree_max
                    )
                
                # Create graph sample
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    min_component_size=min_component_size,
                    degree_heterogeneity=degree_heterogeneity,
                    edge_noise=edge_noise,
                    feature_regime_balance=feature_regime_balance,
                    target_homophily=target_homophily,
                    target_density=target_density,
                    use_configuration_model=self.use_configuration_model,
                    degree_distribution=self.degree_distribution,
                    power_law_exponent=power_law_exponent,
                    target_avg_degree=target_avg_degree,
                    seed=seed + len(graphs) if seed is not None else None
                )
                
                # Store deviation constraints in the graph instance
                if self.use_configuration_model:
                    graph.max_mean_community_deviation = self.max_mean_community_deviation
                    graph.max_max_community_deviation = self.max_max_community_deviation
                    graph.max_parameter_search_attempts = self.max_parameter_search_attempts
                    graph.parameter_search_range = self.parameter_search_range
                    graph.min_edge_density = self.min_edge_density
                    graph.max_retries = self.max_retries
                
                graphs.append(graph)
                
            except ValueError as e:
                print(f"Warning: Failed to generate graph {len(graphs) + 1}: {str(e)}")
                attempts += 1
                continue
            
            attempts += 1
        
        if len(graphs) < n_graphs:
            print(f"Warning: Could only generate {len(graphs)} out of {n_graphs} graphs after {attempts} attempts")
        
        return graphs 