"""
Graph family generation utilities for MMSB graphs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from mmsb.model import GraphUniverse, GraphSample
from tqdm import tqdm
import time

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
        shared_regime_ratio: float = 0.0,
        shared_regime_mode: str = 'average',
        homophily_range: float = 0.0,
        density_range: float = 0.0,
        method_distribution: Optional[Dict[str, float]] = None,
        standard_method_params: Optional[Dict[str, Any]] = None,
        config_model_params: Optional[Dict[str, Any]] = None,
        max_mean_community_deviation: float = None,
        max_max_community_deviation: float = None,
        max_parameter_search_attempts: int = None,
        parameter_search_range: float = None,
        min_edge_density: float = None,
        max_retries: int = None,
        triangle_enhancement: float = 0.0,
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
            shared_regime_ratio: Proportion of regimes shared between communities (0-1)
            shared_regime_mode: How to merge shared regimes ('first', 'second', or 'average')
            homophily_range: Allowed deviation from universe homophily
            density_range: Allowed deviation from universe density
            method_distribution: Dictionary mapping method names to their probabilities
                               e.g. {"standard": 0.5, "power_law": 0.3, "uniform": 0.2}
            standard_method_params: Parameters for standard method generation
            config_model_params: Parameters for configuration model generation
            max_mean_community_deviation: Maximum allowed mean deviation from community structure
            max_max_community_deviation: Maximum allowed maximum deviation from community structure
            max_parameter_search_attempts: Maximum number of parameter search attempts
            parameter_search_range: How aggressively to search parameter space
            min_edge_density: Minimum acceptable edge density
            max_retries: Maximum number of retries if graph is too sparse
            triangle_enhancement: Triangle enhancement factor
            seed: Random seed for reproducibility
        """
        # Store all parameters that need to be passed to GraphSample
        self.max_mean_community_deviation = max_mean_community_deviation
        self.max_max_community_deviation = max_max_community_deviation
        self.max_parameter_search_attempts = max_parameter_search_attempts
        self.parameter_search_range = parameter_search_range
        self.min_edge_density = min_edge_density
        self.max_retries = max_retries
        self.triangle_enhancement = triangle_enhancement

        # Validate required parameters
        required_params = [
            ('max_mean_community_deviation', max_mean_community_deviation),
            ('max_max_community_deviation', max_max_community_deviation),
            ('max_parameter_search_attempts', max_parameter_search_attempts),
            ('parameter_search_range', parameter_search_range),
            ('min_edge_density', min_edge_density),
            ('max_retries', max_retries)
        ]
        missing_params = [name for name, value in required_params if value is None]
        if missing_params:
            raise ValueError(f"Missing required parameters from app.py: {', '.join(missing_params)}")

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
                shared_regime_ratio=shared_regime_ratio,
                shared_regime_mode=shared_regime_mode,
                seed=seed
            )
        else:
            self.universe = universe
        
        # Store parameters
        self.homophily_range = homophily_range
        self.density_range = density_range
        
        # Set up method distribution
        if method_distribution is None:
            self.method_distribution = {"standard": 1.0}  # Default to standard method
        else:
            # Normalize probabilities
            total = sum(method_distribution.values())
            self.method_distribution = {k: v/total for k, v in method_distribution.items()}
        
        # Set up method parameters
        self.standard_method_params = standard_method_params or {}
        
        # Set up configuration model parameters
        if config_model_params is None:
            config_model_params = {
                "power_law": {
                    "exponent_min": 1.5,
                    "exponent_max": 3.0,
                    "target_avg_degree_min": 2.0,
                    "target_avg_degree_max": 20.0
                },
                "exponential": {
                    "rate_min": 0.1,  # Lower rate = more spread out
                    "rate_max": 1.0,  # Higher rate = more concentrated
                    "target_avg_degree_min": 2.0,
                    "target_avg_degree_max": 20.0
                },
                "uniform": {
                    "min_factor": 0.5,
                    "max_factor": 1.5,
                    "target_avg_degree_min": 2.0,
                    "target_avg_degree_max": 20.0
                }
            }
        self.config_model_params = config_model_params
        
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

    def _sample_generation_method(self) -> Tuple[str, Dict[str, Any]]:
        """
        Sample a generation method and its parameters based on the method distribution.
        
        Returns:
            Tuple of (method_name, method_params)
        """
        # Sample method based on distribution
        method = np.random.choice(
            list(self.method_distribution.keys()),
            p=list(self.method_distribution.values())
        )
        if method == "standard":
            return method, self.standard_method_params
        # For configuration model methods, always sample from the full global range
        params = self.config_model_params[method].copy() if method in self.config_model_params else {}
        if method == "power_law":
            params["power_law_exponent"] = np.random.uniform(0.5, 4.0)
            params["target_avg_degree"] = np.random.uniform(1.0, 20.0)
        elif method == "exponential":
            params["rate"] = np.random.uniform(0.05, 2.0)
            params["target_avg_degree"] = np.random.uniform(1.0, 20.0)
        elif method == "uniform":
            params["min_factor"] = np.random.uniform(0.1, 1.0)
            params["max_factor"] = np.random.uniform(params["min_factor"], 2.0)
            params["target_avg_degree"] = np.random.uniform(1.0, 20.0)
        return method, params

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
        seed: Optional[int] = None,
        show_progress: bool = False,
        show_timing: bool = False
    ) -> list:
        """
        Generate a family of graphs.
        Returns a list of (graph, n_attempts) tuples.
        """
        import numpy as np
        if seed is not None:
            np.random.seed(seed)
        results = []
        attempts = 0
        max_attempts = n_graphs * 3
        iterator = range(n_graphs)
        if show_progress:
            iterator = tqdm(iterator, desc='Generating graphs in family', leave=False)
        # Timing accumulators
        timings = {k: [] for k in [
            'community_sampling', 'node_sampling', 'target_param_sampling',
            'method_sampling', 'graph_construction', 'total']}
        for _ in iterator:
            t0 = time.time()
            n_attempts_for_this_graph = 0
            while n_attempts_for_this_graph < 10:  # up to 10 tries per graph
                try:
                    t1 = time.time()
                    # Sample number of communities
                    n_communities = np.random.randint(min_communities, max_communities + 1)
                    # Sample communities
                    t2 = time.time()
                    communities = self.universe.sample_connected_community_subset(
                        size=n_communities,
                        method=sampling_method
                    )
                    t3 = time.time()
                    # Sample number of nodes
                    n_nodes = np.random.randint(min_nodes, max_nodes + 1)
                    t4 = time.time()
                    # Sample target parameters
                    target_homophily, target_density = self._sample_target_parameters()
                    t5 = time.time()
                    # Sample generation method and its parameters
                    method, method_params = self._sample_generation_method()
                    t6 = time.time()
                    # Create graph sample with ALL required parameters
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
                        use_configuration_model=(method in ["power_law", "exponential", "uniform"]),
                        degree_distribution=method if method in ["power_law", "exponential", "uniform"] else None,
                        power_law_exponent=method_params.get("power_law_exponent"),
                        target_avg_degree=method_params.get("target_avg_degree"),
                        triangle_enhancement=self.triangle_enhancement,
                        max_mean_community_deviation=self.max_mean_community_deviation,
                        max_max_community_deviation=self.max_max_community_deviation,
                        max_parameter_search_attempts=self.max_parameter_search_attempts,
                        parameter_search_range=self.parameter_search_range,
                        min_edge_density=self.min_edge_density,
                        max_retries=self.max_retries,
                        seed=seed + len(results) if seed is not None else None,
                        config_model_params=method_params if method in ["power_law", "exponential", "uniform"] else None
                    )
                    t7 = time.time()
                    timings['community_sampling'].append(t2-t1)
                    timings['node_sampling'].append(t4-t3)
                    timings['target_param_sampling'].append(t5-t4)
                    timings['method_sampling'].append(t6-t5)
                    timings['graph_construction'].append(t7-t6)
                    timings['total'].append(t7-t0)
                    results.append((graph, n_attempts_for_this_graph + 1))
                    break
                except ValueError as e:
                    n_attempts_for_this_graph += 1
                    attempts += 1
                    if n_attempts_for_this_graph >= 10:
                        print(f"Warning: Failed to generate graph {len(results) + 1} after 10 attempts: {str(e)}")
                        break
            attempts += 1
        if len(results) < n_graphs:
            print(f"Warning: Could only generate {len(results)} out of {n_graphs} graphs after {attempts} attempts")
        # Print timing summary if requested
        if show_progress or show_timing:
            print("Timing summary for this family:")
            for k in ['community_sampling', 'node_sampling', 'target_param_sampling', 'method_sampling', 'graph_construction', 'total']:
                arr = np.array(timings[k])
                if len(arr) > 0:
                    print(f"  {k:25s}: mean={arr.mean()*1000:.1f}ms, std={arr.std()*1000:.1f}ms")
        return results 