"""
Graph family generation utilities for MMSB graphs.

This module provides functionality for generating families of graphs
sampled from a single graph universe, with varying parameters across
the family members.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from graph_universe.model import GraphUniverse, GraphSample
from tqdm import tqdm
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from itertools import combinations
import networkx as nx

class GraphFamilyGenerator:
    """
    Generates families of graphs sampled from a graph universe.
    Each graph in the family is a subgraph of the universe with its own community structure.
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        min_n_nodes: int,
        max_n_nodes: int,
        min_communities: int = 2,
        max_communities: int = None,
        min_component_size: int = 10,
        homophily_range: Tuple[float, float] = (0.0, 0.4),  # Homophily range
        avg_degree_range: Tuple[float, float] = (1.0, 3.0),    # Average degree range
        # density_range: Tuple[float, float] = (0.1, 0.15),    # Density range

        # Whether to use DCCC-SBM or standard DC-SBM
        use_dccc_sbm: bool = False,
        
        # Community co-occurrence homogeneity
        community_cooccurrence_homogeneity: float = 1.0,

        # Deviation limiting
        disable_deviation_limiting: bool = False,
        max_mean_community_deviation: float = 0.10,
        max_max_community_deviation: float = 0.15,
        min_edge_density: float = 0.005,

        # DCCC distribution-specific parameter ranges
        degree_distribution: str = "standard",
        power_law_exponent_range: Tuple[float, float] = (2.0, 3.5),
        exponential_rate_range: Tuple[float, float] = (0.3, 1.0),
        uniform_min_factor_range: Tuple[float, float] = (0.3, 0.7),
        uniform_max_factor_range: Tuple[float, float] = (1.3, 2.0),
        degree_separation_range: Tuple[float, float] = (0.5, 0.5),    # Range for degree separation
        degree_signal_calc_method: str = "standard", # How to calculate degree signal

        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        degree_heterogeneity: float = 0.5,

        # Fixed parameters for all graphs in family
        max_retries: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize the graph family generator.
        
        Args:
            universe: GraphUniverse to sample from
            n_graphs: Number of graphs to generate in the family
            min_n_nodes: Minimum number of nodes per graph
            max_n_nodes: Maximum number of nodes per graph
            min_communities: Minimum number of communities per graph
            max_communities: Maximum number of communities per graph (defaults to universe.K)
            min_component_size: Minimum size for connected components
            homophily_range: Tuple of (min_homophily, max_homophily) in graph family
            density_range: Tuple of (min_density, max_density) in graph family

            # Whether to use DCCC-SBM or standard DC-SBM
            use_dccc_sbm: Whether to use DCCC-SBM model

            # DCCC distribution-specific parameters
            degree_separation_range: Range for degree distribution separation (min, max)
            degree_signal_calc_method: How to calculate degree signal
            disable_deviation_limiting: Whether to disable deviation checks
            max_mean_community_deviation: Maximum allowed mean community deviation
            max_max_community_deviation: Maximum allowed max community deviation
            min_edge_density: Minimum edge density for graphs
            degree_distribution: Degree distribution type ("standard", "power_law", "exponential", "uniform")
            power_law_exponent_range: Range for power law exponent (min, max)
            exponential_rate_range: Range for exponential distribution rate (min, max)
            uniform_min_factor_range: Range for uniform distribution min factor (min, max)
            uniform_max_factor_range: Range for uniform distribution max factor (min, max)
            
            # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
            degree_heterogeneity: Fixed degree heterogeneity for all graphs

            # Fixed parameters for all graphs in family
            max_retries: Maximum retries for graph generation
            seed: Random seed for reproducibility
        """
        self.universe = universe
        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes
        self.min_communities = min_communities
        self.max_communities = max_communities if max_communities is not None else universe.K
        self.min_component_size = min_component_size
        self.homophily_range = homophily_range
        self.avg_degree_range = avg_degree_range
        # self.density_range = density_range

        # Whether to use DCCC-SBM or standard DC-SBM
        self.use_dccc_sbm = use_dccc_sbm

        # DCCC distribution-specific parameters
        self.degree_separation_range = degree_separation_range # Range for degree separation
        self.degree_signal_calc_method = degree_signal_calc_method # How to calculate degree signal
        
        # Community co-occurrence homogeneity
        self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity

        # Deviation limiting
        self.disable_deviation_limiting = disable_deviation_limiting
        self.max_mean_community_deviation = max_mean_community_deviation
        self.max_max_community_deviation = max_max_community_deviation
        self.min_edge_density = min_edge_density
        
        # DCCC distribution parameters
        self.degree_distribution = degree_distribution
        self.power_law_exponent_range = power_law_exponent_range
        self.exponential_rate_range = exponential_rate_range
        self.uniform_min_factor_range = uniform_min_factor_range
        self.uniform_max_factor_range = uniform_max_factor_range

        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        self.degree_heterogeneity = degree_heterogeneity
        
        # Fixed parameters for all graphs
        self.max_retries = max_retries
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Validate parameters
        self._validate_parameters()
        
        # Storage for generated graphs and metadata
        self.graphs: List[GraphSample] = []
        self.generation_metadata: List[Dict[str, Any]] = []
        self.generation_stats: Dict[str, Any] = {}
    
    def generate_family(
        self,
        n_graphs: int,
        show_progress: bool = True,
        collect_stats: bool = True,
        max_attempts_per_graph: int = 10,
        timeout_minutes: float = 5.0,
        allowed_community_combinations: Optional[List[List[int]]] = None
    ) -> List[GraphSample]:
        """
        Generate a family of graphs from the universe.
        
        Args:
            show_progress: Whether to show progress bar
            collect_stats: Whether to collect generation statistics
            max_attempts_per_graph: Maximum attempts per graph before giving up
            timeout_minutes: Maximum time in minutes to spend generating graphs
            allowed_community_combinations: Optional list of lists of community indices to be sampled from the universe
        Returns:
            List of generated GraphSample objects
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        self.graphs = []
        self.community_labels_per_graph = []
        self.generation_metadata = []
        failed_graphs = 0

        # Every time this function is called we need to reset Seed
        self.seed = np.random.randint(0, 1000000)
        np.random.seed(self.seed)
        
        # Progress bar setup
        if show_progress:
            pbar = tqdm(total=n_graphs, desc="Generating graph family")

        
        while len(self.graphs) < n_graphs:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                warnings.warn(f"Timeout reached after {timeout_minutes} minutes. Generated {len(self.graphs)} graphs instead of {n_graphs}")
                break
                
            graph_generated = False
            attempts = 0
            
            while not graph_generated and attempts < max_attempts_per_graph:
                attempts += 1
                
                try:

                    # Get a random seed for this graph
                    graph_seed = np.random.randint(0, 1000000)

                    # Sample parameters for this graph
                    params = self._sample_graph_parameters()

                    if allowed_community_combinations is not None:
                        sampled_community_combination_index = np.random.randint(0, len(allowed_community_combinations))
                        sampled_community_combination = allowed_community_combinations[sampled_community_combination_index]
                    
                    # Create graph sample
                    graph_sample = GraphSample(
                        # Give GraphUniverse object to sample from
                        universe=self.universe,

                        # Graph Sample specific parameters
                        num_communities=params['n_communities'],
                        n_nodes=params['n_nodes'],
                        min_component_size=self.min_component_size,
                        target_homophily=params['target_homophily'],
                        target_average_degree=params['target_average_degree'],
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        max_mean_community_deviation=self.max_mean_community_deviation,
                        max_max_community_deviation=self.max_max_community_deviation,
                        min_edge_density=self.min_edge_density,
                        max_retries=self.max_retries, # Retries of single graph generation

                        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
                        degree_heterogeneity=self.degree_heterogeneity,

                        # DCCC-SBM parameters
                        use_dccc_sbm=self.use_dccc_sbm,
                        degree_separation=params.get('degree_separation', 0.5),
                        dccc_global_degree_params=params.get('dccc_global_degree_params', {}),
                        degree_signal_calc_method=self.degree_signal_calc_method, # How to calculate degree signal
                        disable_deviation_limiting=self.disable_deviation_limiting,

                        # Random seed
                        seed=graph_seed,

                        # Optional Parameter for user-defined communites to be sampled (Such that NO unseen communities are sampled for val or test)
                        user_defined_communities=sampled_community_combination if allowed_community_combinations is not None else None
                    )
                    
                    # Store graph and metadata
                    self.graphs.append(graph_sample)
                    self.community_labels_per_graph.append(np.unique(graph_sample.community_labels_universe_level, axis=0))
                    metadata = {
                        'graph_id': len(self.graphs) - 1,
                        'attempts': attempts,
                        'final_n_nodes': graph_sample.n_nodes,
                        'final_n_edges': graph_sample.graph.number_of_edges(),
                        'actual_density': graph_sample.graph.number_of_edges() / (graph_sample.n_nodes * (graph_sample.n_nodes - 1) / 2) if graph_sample.n_nodes > 1 else 0,
                        'generation_method': graph_sample.generation_method,
                        'timing_info': graph_sample.timing_info.copy() if hasattr(graph_sample, 'timing_info') else {},
                        **params  # Include sampled parameters
                    }
                    
                    self.generation_metadata.append(metadata)
                    graph_generated = True
                    
                    if show_progress:
                        pbar.update(1)
                    
                except Exception as e:
                    if attempts == max_attempts_per_graph:
                        warnings.warn(f"Failed to generate graph after {attempts} attempts: {e}")
                        failed_graphs += 1
                        # Add empty metadata for failed graph
                        self.generation_metadata.append({
                            'graph_id': len(self.graphs),
                            'attempts': attempts,
                            'failed': True,
                            'error': str(e)
                        })
                    # Continue to next attempt
        
        # Use set of sorted tuples for uniqueness
        unique_set_of_community_combinations = {tuple(sorted(arr)) for arr in self.community_labels_per_graph}

        # Convert back to list of lists
        unique_list_of_community_combinations = [list(tup) for tup in unique_set_of_community_combinations]
        self.seen_community_combinations = unique_list_of_community_combinations
        
        # print(f"Seen community combinations: {self.seen_community_combinations}")

        if show_progress:
            pbar.close()
        
        # Collect generation statistics
        if collect_stats:
            self._collect_generation_stats(start_time, failed_graphs, n_graphs)
        
        return self.graphs

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.min_n_nodes <= 0:
            raise ValueError("min_n_nodes must be positive")
        if self.max_n_nodes < self.min_n_nodes:
            raise ValueError("max_n_nodes must be >= min_n_nodes")
        if self.min_communities < 1:
            raise ValueError("min_communities must be >= 1")
        if self.max_communities > self.universe.K:
            raise ValueError(f"max_communities cannot exceed universe size ({self.universe.K})")
        if self.max_communities < self.min_communities:
            raise ValueError("max_communities must be >= min_communities")
        
        # Validate ranges
        if len(self.homophily_range) != 2 or self.homophily_range[0] > self.homophily_range[1]:
            raise ValueError("homophily_range must be a tuple (min, max) with min <= max")
        if self.homophily_range[0] < 0.0 or self.homophily_range[1] > 1.0:
            raise ValueError("homophily_range values must be between 0.0 and 1.0")
        if len(self.avg_degree_range) != 2 or self.avg_degree_range[0] > self.avg_degree_range[1]:
            raise ValueError("avg_degree_range must be a tuple (min, max) with min <= max")
        if self.avg_degree_range[0] < 0.0 or self.avg_degree_range[1] > 100.0:
            raise ValueError("avg_degree_range values must be between 0.0 and 100.0")
        if len(self.degree_separation_range) != 2 or self.degree_separation_range[0] > self.degree_separation_range[1]:
            raise ValueError("degree_separation_range must be a tuple (min, max) with min <= max")
        
        # Validate DCCC distribution parameter ranges
        if len(self.power_law_exponent_range) != 2 or self.power_law_exponent_range[0] > self.power_law_exponent_range[1]:
            raise ValueError("power_law_exponent_range must be a tuple (min, max) with min <= max")
        if len(self.exponential_rate_range) != 2 or self.exponential_rate_range[0] > self.exponential_rate_range[1]:
            raise ValueError("exponential_rate_range must be a tuple (min, max) with min <= max")
        if len(self.uniform_min_factor_range) != 2 or self.uniform_min_factor_range[0] > self.uniform_min_factor_range[1]:
            raise ValueError("uniform_min_factor_range must be a tuple (min, max) with min <= max")
        if len(self.uniform_max_factor_range) != 2 or self.uniform_max_factor_range[0] > self.uniform_max_factor_range[1]:
            raise ValueError("uniform_max_factor_range must be a tuple (min, max) with min <= max")
        
        # Validate that distribution ranges make sense
        if self.power_law_exponent_range[0] <= 1.0:
            raise ValueError("power_law_exponent_range values must be > 1.0")
        if self.exponential_rate_range[0] <= 0.0:
            raise ValueError("exponential_rate_range values must be > 0.0")
        if self.uniform_min_factor_range[0] <= 0.0:
            raise ValueError("uniform_min_factor_range values must be > 0.0")
        if self.uniform_max_factor_range[0] <= 0.0:
            raise ValueError("uniform_max_factor_range values must be > 0.0")
    
    def _sample_graph_parameters(self) -> Dict[str, Any]:
        """
        Sample parameters for a single graph within the specified ranges.
        
        Returns:
            Dictionary of sampled parameters
        """
        # Sample number of nodes
        n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        
        # Sample number of communities
        n_communities = np.random.randint(self.min_communities, self.max_communities + 1)
        
        # Sample target homophily
        target_homophily = np.random.uniform(self.homophily_range[0], self.homophily_range[1])

        # Sample target average degree
        target_average_degree = np.random.uniform(self.avg_degree_range[0], self.avg_degree_range[1])
        
        # # Sample target density
        # target_density = np.random.uniform(self.density_range[0], self.density_range[1])
        
        # Sample DCCC parameters if using DCCC-SBM
        dccc_params = {}
        if self.use_dccc_sbm:
            # Sample degree separation
            degree_separation = np.random.uniform(
                self.degree_separation_range[0], 
                self.degree_separation_range[1]
            )
            
            # Sample distribution-specific parameters
            distribution_params = {}
            if self.degree_distribution == "power_law":
                power_law_exponent = np.random.uniform(
                    self.power_law_exponent_range[0],
                    self.power_law_exponent_range[1]
                )
                distribution_params = {
                    "exponent": power_law_exponent,
                    "x_min": 1.0
                }
            elif self.degree_distribution == "exponential":
                rate = np.random.uniform(
                    self.exponential_rate_range[0],
                    self.exponential_rate_range[1]
                )
                distribution_params = {"rate": rate}
            elif self.degree_distribution == "uniform":
                min_factor = np.random.uniform(
                    self.uniform_min_factor_range[0],
                    self.uniform_min_factor_range[1]
                )
                max_factor = np.random.uniform(
                    self.uniform_max_factor_range[0],
                    self.uniform_max_factor_range[1]
                )
                # Ensure max_factor > min_factor
                if max_factor <= min_factor:
                    max_factor = min_factor + 0.1
                distribution_params = {
                    "min_degree": min_factor,
                    "max_degree": max_factor
                }
            
            dccc_params = {
                "degree_separation": degree_separation,
                "dccc_global_degree_params": distribution_params,
                "power_law_exponent": distribution_params.get("exponent", None)
            }
        
        # Combine all parameters
        params = {
            'n_nodes': n_nodes,
            'target_homophily': target_homophily,
            'target_average_degree': target_average_degree,
            'n_communities': n_communities,
        }
        
        # Add DCCC parameters
        params.update(dccc_params)
        
        return params
    
    def _generate_single_graph(
        self,
        max_attempts: int = 10,
        timeout_minutes: float = 5.0
    ) -> GraphSample:
        """
        Generate a single graph with progress tracking.
        
        Args:
            max_attempts: Maximum attempts per graph before giving up
            timeout_minutes: Maximum time in minutes to spend generating this graph
            
        Returns:
            Generated GraphSample object
            
        Raises:
            Exception: If graph generation fails after max_attempts
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                raise Exception(f"Timeout reached after {timeout_minutes} minutes")
            
            try:
                # Get a random seed for this graph
                graph_seed = np.random.randint(0, 1000000)
                
                # Sample parameters for this graph
                params = self._sample_graph_parameters()
                
                # Debug: Print parameters being used
                print(f"Attempt {attempts}: Generating graph with params: {params}")
                
                # Create graph sample
                print(f"Attempt {attempts}: Starting GraphSample initialization...")
                
                try:
                    graph_sample = GraphSample(
                        # Give GraphUniverse object to sample from
                        universe=self.universe,

                        # Graph Sample specific parameters
                        num_communities=params['n_communities'],
                        n_nodes=params['n_nodes'],
                        min_component_size=self.min_component_size,
                        target_homophily=params['target_homophily'],
                        target_average_degree=params['target_average_degree'],
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        max_mean_community_deviation=self.max_mean_community_deviation,
                        max_max_community_deviation=self.max_max_community_deviation,
                        min_edge_density=self.min_edge_density,
                        max_retries=self.max_retries, # Retries of single graph generation

                        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
                        degree_heterogeneity=self.degree_heterogeneity,

                        # DCCC-SBM parameters
                        use_dccc_sbm=self.use_dccc_sbm,
                        degree_separation=params.get('degree_separation', 0.5),
                        dccc_global_degree_params=params.get('dccc_global_degree_params', {}),
                        degree_signal_calc_method=self.degree_signal_calc_method, # How to calculate degree signal
                        disable_deviation_limiting=self.disable_deviation_limiting,

                        # Random seed
                        seed=graph_seed,

                        # Optional Parameter for user-defined communites to be sampled
                        user_defined_communities=None
                    )
                    print(f"Attempt {attempts}: GraphSample initialization completed successfully")
                except Exception as e:
                    print(f"Attempt {attempts}: GraphSample initialization failed: {str(e)}")
                    raise e
                
                print(f"Successfully generated graph with {graph_sample.n_nodes} nodes")
                return graph_sample
                
            except Exception as e:
                print(f"Attempt {attempts} failed: {str(e)}")
                if attempts == max_attempts:
                    raise Exception(f"Failed to generate graph after {attempts} attempts: {e}")
                # Continue to next attempt
 
    def _collect_generation_stats(self, start_time: float, failed_graphs: int, n_graphs: int) -> None:
        """Collect statistics about the generation process."""
        total_time = time.time() - start_time
        successful_graphs = len(self.graphs)
        
        # Basic stats
        self.generation_stats = {
            'total_time': total_time,
            'successful_graphs': successful_graphs,
            'failed_graphs': failed_graphs,
            'success_rate': successful_graphs / n_graphs if n_graphs > 0 else 0,
            'avg_time_per_graph': total_time / n_graphs if n_graphs > 0 else 0
        }
        
        if successful_graphs > 0:
            # Get statistics from successful graphs
            successful_metadata = [m for m in self.generation_metadata if not m.get('failed', False)]
            
            node_counts = [m['final_n_nodes'] for m in successful_metadata]
            edge_counts = [m['final_n_edges'] for m in successful_metadata]
            densities = [m['actual_density'] for m in successful_metadata]
            attempts = [m['attempts'] for m in successful_metadata]
            
            self.generation_stats.update({
                'node_count_stats': {
                    'mean': np.mean(node_counts),
                    'std': np.std(node_counts),
                    'min': np.min(node_counts),
                    'max': np.max(node_counts)
                },
                'edge_count_stats': {
                    'mean': np.mean(edge_counts),
                    'std': np.std(edge_counts),
                    'min': np.min(edge_counts),
                    'max': np.max(edge_counts)
                },
                'density_stats': {
                    'mean': np.mean(densities),
                    'std': np.std(densities),
                    'min': np.min(densities),
                    'max': np.max(densities)
                },
                'attempts_stats': {
                    'mean': np.mean(attempts),
                    'std': np.std(attempts),
                    'min': np.min(attempts),
                    'max': np.max(attempts)
                }
            })
    
    def save_family(self, filepath: str, include_graphs: bool = True, n_graphs: int = 0) -> None:
        """
        Save the graph family to file.
        
        Args:
            filepath: Path to save file
            include_graphs: Whether to include graph objects (large files)
        """
        import pickle
        
        save_data = {
            'generation_metadata': self.generation_metadata,
            'generation_stats': self.generation_stats,
            'parameters': {
                'n_graphs': n_graphs,
                'min_n_nodes': self.min_n_nodes,
                'max_n_nodes': self.max_n_nodes,
                'min_communities': self.min_communities,
                'max_communities': self.max_communities,
                'homophily_range': self.homophily_range,
                'density_range': self.density_range,
                'use_dccc_sbm': self.use_dccc_sbm,
                'degree_separation_range': self.degree_separation_range,
                'degree_distribution': self.degree_distribution,
                'power_law_exponent_range': self.power_law_exponent_range,
                'exponential_rate_range': self.exponential_rate_range,
                'uniform_min_factor_range': self.uniform_min_factor_range,
                'uniform_max_factor_range': self.uniform_max_factor_range
            }
        }
        
        if include_graphs:
            save_data['graphs'] = self.graphs
            
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def __len__(self) -> int:
        """Return number of successfully generated graphs."""
        return len(self.graphs)
    
    def __getitem__(self, index: int) -> GraphSample:
        """Get a specific graph from the family."""
        return self.graphs[index]
    
    def __iter__(self):
        """Iterate over graphs in the family."""
        return iter(self.graphs)
    
    def analyze_graph_family_properties(self) -> Dict[str, Any]:
        """Analyze the properties of the graph family once generated."""
        if self.graphs is None:
            raise ValueError("No graphs in family. Please generate family first before analyzing properties.")
        
        properties = {
            'n_graphs': len(self.graphs),
            'node_counts': [],
            'edge_counts': [],
            'densities': [],
            'avg_degrees': [],
            'clustering_coefficients': [],
            'community_counts': [],
            'homophily_levels': [],
            'nr_of_triangles': [],
            'generation_methods': []
        }
        
        for graph in self.graphs:
            properties['node_counts'].append(graph.n_nodes)
            properties['edge_counts'].append(graph.graph.number_of_edges())
            
            if graph.n_nodes > 1:
                density = graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2)
                properties['densities'].append(density)
            else:
                properties['densities'].append(0.0)
            
            if graph.n_nodes > 0:
                avg_degree = sum(dict(graph.graph.degree()).values()) / graph.n_nodes
                properties['avg_degrees'].append(avg_degree)
            else:
                properties['avg_degrees'].append(0.0)
            
            try:
                clustering = nx.average_clustering(graph.graph)
                properties['clustering_coefficients'].append(clustering)
            except:
                properties['clustering_coefficients'].append(0.0)
            
            properties['community_counts'].append(len(np.unique(graph.community_labels)))
            
            # Calculate homophily level
            if graph.n_nodes > 0 and graph.graph.number_of_edges() > 0:
                # Count edges between nodes of same community
                same_community_edges = 0
                for u, v in graph.graph.edges():
                    if graph.community_labels[u] == graph.community_labels[v]:
                        same_community_edges += 1
                homophily = same_community_edges / graph.graph.number_of_edges()
                properties['homophily_levels'].append(homophily)
            else:
                properties['homophily_levels'].append(0.0)
            
            # Calculate number of triangles
            if graph.n_nodes > 0:
                triangle_values = list(nx.triangles(graph.graph).values())
                properties['nr_of_triangles'].append(np.sum(triangle_values)/3)
            else:
                properties['nr_of_triangles'].append(0.0)
            
            # Track generation method
            if hasattr(graph, 'generation_method'):
                properties['generation_methods'].append(graph.generation_method)
        
        # Calculate statistics and convert to native Python types
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels', 'nr_of_triangles']:
            values = properties[key]
            if values:
                properties[f'{key}_mean'] = float(np.mean(values))
                properties[f'{key}_std'] = float(np.std(values))
                properties[f'{key}_min'] = float(np.min(values))
                properties[f'{key}_max'] = float(np.max(values))
            else:
                properties[f'{key}_mean'] = 0.0
                properties[f'{key}_std'] = 0.0
                properties[f'{key}_min'] = 0.0
                properties[f'{key}_max'] = 0.0
        
        # Add generation method summary
        if properties['generation_methods']:
            from collections import Counter
            method_counts = Counter(properties['generation_methods'])
            properties['generation_method_distribution'] = dict(method_counts)
        
        return properties
   
    def analyze_graph_family_learning_signals(self) -> Dict[str, Any]:
        """Analyze the learning signals of the graph family."""
        if not self.graphs:
            raise ValueError("No graphs in family. Please generate family first before analyzing learning signals.")
        
        # Calculate the learning signals
        results = {}
        
        # 1. Pattern preservation (do communities RELATIVELY connect more to the communities they are supposed to connect to?)
        try:
            pattern_corrs = self._calculate_pattern_consistency()
            results['pattern_preservation'] = pattern_corrs

        except Exception as e:
            results['pattern_preservation'] = []

        # 2. Generation fidelity (do graphs match their scaled probability targets (P_sub)?)
        try:
            generation_fidelity = self._calculate_generation_fidelity()
            results['generation_fidelity'] = generation_fidelity
        except Exception as e:
            results['generation_fidelity'] = []

        # 3. Degree consistency (do actual node degrees correlate with universe degree centers?)
        try:
            degree_consistency = self._calculate_degree_consistency()
            results['degree_consistency'] = degree_consistency
        except Exception as e:
            results['degree_consistency'] = []

        return results

    def _calculate_pattern_consistency(self) -> List[float]:
        """
        Measure whether relative community connection patterns are preserved.
        Uses rank correlation to focus on structural patterns rather than absolute values.
        """
        pattern_correlations = []
        universe_P = self.universe.P

        for graph in self.graphs:
            try:
                # Extract relevant submatrix from universe
                P_sub = graph.P_sub
                
                # Get actual connections
                actual_edge_probabilities, community_sizes, connection_counts = graph.calculate_actual_probability_matrix()

                # Calculate rank correlation per row comparing the expected and actual edge probabilities and average them
                if len(P_sub) > 1 and np.std(P_sub) > 0 and np.std(actual_edge_probabilities) > 0:
                    correlations = []
                    for i in range(len(P_sub)):
                        correlation, _ = spearmanr(P_sub[i], actual_edge_probabilities[i])
                        if not np.isnan(correlation):
                            correlations.append(correlation)
                    pattern_correlations.append(np.mean(correlations))
            except Exception as e:
                warnings.warn(f"Error in pattern consistency calculation for graph: {e}")
                continue
        
        return pattern_correlations

    def _calculate_generation_fidelity(self) -> List[float]:
            """
            Measure how well graphs match their scaled probability targets (P_sub).
            """
            fidelity_scores = []
            
            for graph in self.graphs:
                try:
                    # Use the graph's own scaled P_sub as reference
                    expected_matrix = graph.P_sub
                    
                    # Get actual connections
                    actual_analysis = graph.analyze_community_connections()
                    actual_matrix = actual_analysis['actual_matrix']
                    
                    # Calculate correlation between expected and actual
                    expected_flat = expected_matrix.flatten()
                    actual_flat = actual_matrix.flatten()
                    
                    if len(expected_flat) > 1:
                        if np.std(expected_flat) > 0 and np.std(actual_flat) > 0:
                            correlation, _ = pearsonr(expected_flat, actual_flat)
                            if not np.isnan(correlation):
                                fidelity_scores.append(correlation)
                except Exception as e:
                    warnings.warn(f"Error in generation fidelity calculation for graph: {e}")
                    continue
            
            if not fidelity_scores:
                return []
            
            return fidelity_scores
    
    def _calculate_degree_consistency(self) -> List[float]:
        """
        Compare actual node degrees to expected degrees based on universe degree centers.
        """
        consistency_scores = []
        
        degree_centers = self.universe.degree_centers
        for graph in self.graphs:
            try:
                # Get actual degrees per community
                actual_degrees_per_community = np.zeros(len(graph.communities))

                for node_idx in range(graph.n_nodes):
                    community_id = graph.community_labels[node_idx]
                    degree = graph.graph.degree[node_idx]

                    actual_degrees_per_community[community_id] += degree
                    
                
                # Get ranking of actual degrees per community
                actual_degrees_per_community_ranking = np.argsort(actual_degrees_per_community)
                
                # Get expected ranking according to degree centers
                expected_ranking = np.argsort(degree_centers[graph.communities])
                
                # Calculate correlation between actual and expected rankings
                correlation, _ = spearmanr(actual_degrees_per_community_ranking, expected_ranking)
                consistency_scores.append(correlation)

            except Exception as e:
                warnings.warn(f"Error in degree consistency calculation for graph: {e}")
                continue
        
        if not consistency_scores:
            return []
        
        return consistency_scores


class FamilyConsistencyAnalyzer:
    """
    Analyzes consistency focusing on structural patterns rather than absolute values.
    
    This class provides three main consistency metrics:
    1. Pattern Preservation: How well rank ordering of connections is preserved vs universe
    2. Cross-Graph Similarity: How similar graphs are to each other within the family
    3. Generation Fidelity: How well graphs match their intended generation targets
    """
    
    def __init__(self, family_graphs: List, universe):
        """
        Initialize the consistency analyzer.
        
        Args:
            family_graphs: List of GraphSample objects from the same family
            universe: GraphUniverse object that generated the family
        """
        self.family_graphs = family_graphs
        self.universe = universe
        self.results = None
    
    def analyze_consistency(self) -> Dict[str, Any]:
        """
        Comprehensive consistency analysis.
        
        Returns:
            Dictionary with consistency metrics and interpretations
        """
        if not self.family_graphs:
            return {'error': 'No graphs provided for analysis'}
        
        results = {}
        
        # 1. Pattern preservation (vs universe structure)
        try:
            pattern_consistency, pattern_corrs = self._calculate_pattern_consistency()
            results['pattern_preservation'] = {
                'score': pattern_consistency,
                'individual_correlations': pattern_corrs,
                'std': np.std(pattern_corrs) if pattern_corrs else 0.0,
                'interpretation': self._interpret_score(pattern_consistency, 'pattern'),
                'description': 'How well rank ordering of community connections is preserved relative to universe'
            }
        except Exception as e:
            results['pattern_preservation'] = {'error': str(e)}
        
        # 2. Generation fidelity (actual vs intended)
        try:
            generation_consistency, gen_scores = self._calculate_generation_fidelity()
            results['generation_fidelity'] = {
                'score': generation_consistency,
                'individual_scores': gen_scores,
                'std': np.std(gen_scores) if gen_scores else 0.0,
                'interpretation': self._interpret_score(generation_consistency, 'fidelity'),
                'description': 'How well graphs match their scaled probability targets (P_sub)'
            }
        except Exception as e:
            results['generation_fidelity'] = {'error': str(e)}
        
        # 3. Degree consistency
        try:
            degree_consistency, degree_scores = self._calculate_degree_consistency()
            results['degree_consistency'] = {
                'score': degree_consistency,
                'individual_scores': degree_scores,
                'std': np.std(degree_scores) if degree_scores else 0.0,
                'interpretation': self._interpret_score(degree_consistency, 'degree'),
                'description': 'How well actual node degrees correlate with universe degree centers'
            }
        except Exception as e:
            results['degree_consistency'] = {'error': str(e)}
        
        
        # 5. Co-occurrence consistency
        try:
            cooccurrence_analysis = self.analyze_cooccurrence()
            if 'correlation' in cooccurrence_analysis:
                results['cooccurrence_consistency'] = {
                    'score': cooccurrence_analysis['correlation'],
                    'difference_matrix': cooccurrence_analysis['difference_matrix'],
                    'interpretation': self._interpret_score(cooccurrence_analysis['correlation'], 'cooccurrence'),
                    'description': 'How well community co-occurrence patterns are preserved'
                }
        except Exception as e:
            results['cooccurrence_consistency'] = {'error': str(e)}
        
        # 6. Overall consistency (weighted average of successful metrics)
        try:
            overall_score = self._calculate_overall_consistency(results)
            results['overall'] = {
                'score': overall_score,
                'interpretation': self._interpret_score(overall_score, 'overall'),
                'description': 'Weighted average of all consistency metrics'
            }
        except Exception as e:
            results['overall'] = {'error': str(e)}
        
        # 7. Community coverage analysis
        try:
            results['community_coverage'] = self._analyze_community_coverage()
        except Exception as e:
            results['community_coverage'] = {'error': str(e)}
        
        self.results = results
        return results
    
    def _calculate_pattern_consistency(self) -> Tuple[float, List[float]]:
        """
        Measure whether relative community connection patterns are preserved.
        Uses rank correlation to focus on structural patterns rather than absolute values.
        """
        pattern_correlations = []
        universe_P = self.universe.P
        
        for graph in self.family_graphs:
            try:
                # Extract relevant submatrix from universe
                community_indices = graph.communities
                universe_submatrix = universe_P[np.ix_(community_indices, community_indices)]
                
                # Get actual connections
                actual_analysis = graph.analyze_community_connections()
                actual_matrix = actual_analysis['actual_matrix']
                
                # Compare rank orderings (upper triangle only to avoid redundancy)
                k = len(community_indices)
                if k < 2:
                    continue
                    
                triu_indices = np.triu_indices(k, k=0)  # Include diagonal
                
                universe_pattern = universe_submatrix[triu_indices]
                actual_pattern = actual_matrix[triu_indices]
                
                # Calculate rank correlation
                if len(universe_pattern) > 1 and np.std(universe_pattern) > 0 and np.std(actual_pattern) > 0:
                    correlation, _ = spearmanr(universe_pattern, actual_pattern)
                    if not np.isnan(correlation):
                        pattern_correlations.append(correlation)
            except Exception as e:
                warnings.warn(f"Error in pattern consistency calculation for graph: {e}")
                continue
        
        if not pattern_correlations:
            return 0.0, []
        
        return np.mean(pattern_correlations), pattern_correlations
    
    def _calculate_generation_fidelity(self) -> Tuple[float, List[float]]:
        """
        Measure how well graphs match their scaled probability targets (P_sub).
        """
        fidelity_scores = []
        
        for graph in self.family_graphs:
            try:
                # Use the graph's own scaled P_sub as reference
                expected_matrix = graph.P_sub
                
                # Get actual connections
                actual_analysis = graph.analyze_community_connections()
                actual_matrix = actual_analysis['actual_matrix']
                
                # Calculate correlation between expected and actual
                expected_flat = expected_matrix.flatten()
                actual_flat = actual_matrix.flatten()
                
                if len(expected_flat) > 1:
                    if np.std(expected_flat) > 0 and np.std(actual_flat) > 0:
                        correlation, _ = pearsonr(expected_flat, actual_flat)
                        if not np.isnan(correlation):
                            fidelity_scores.append(correlation)
            except Exception as e:
                warnings.warn(f"Error in generation fidelity calculation for graph: {e}")
                continue
        
        if not fidelity_scores:
            return 0.0, []
        
        return np.mean(fidelity_scores), fidelity_scores
    
    def _calculate_degree_consistency(self) -> Tuple[float, List[float]]:
        """
        Compare actual node degrees to expected degrees based on universe degree centers.
        """
        consistency_scores = []
        
        for graph in self.family_graphs:
            try:
                # Get actual degrees
                actual_degrees = np.array([d for _, d in graph.graph.degree()])
                
                # Get expected degrees based on universe centers
                expected_degrees = []
                for node_idx in range(graph.n_nodes):
                    community_local_idx = graph.community_labels[node_idx]
                    universe_community = graph.communities[community_local_idx]
                    expected_degree = self.universe.degree_centers[universe_community]
                    expected_degrees.append(expected_degree)
                
                expected_degrees = np.array(expected_degrees)
                
                # Calculate correlation
                if len(actual_degrees) > 1:
                    if np.std(actual_degrees) > 0 and np.std(expected_degrees) > 0:
                        correlation, _ = pearsonr(actual_degrees, expected_degrees)
                        if not np.isnan(correlation):
                            consistency_scores.append(correlation)
            except Exception as e:
                warnings.warn(f"Error in degree consistency calculation for graph: {e}")
                continue
        
        if not consistency_scores:
            return 0.0, []
        
        return np.mean(consistency_scores), consistency_scores
 
    def _calculate_overall_consistency(self, results: Dict) -> float:
        """Calculate weighted average of all successful consistency metrics."""
        scores = []
        weights = []
        
        # Define weights for different metrics
        metric_weights = {
            'pattern_preservation': 0.3,
            'generation_fidelity': 0.3,
            'degree_consistency': 0.15,
            'cooccurrence_consistency': 0.1
        }
        
        for metric, weight in metric_weights.items():
            if metric in results and 'score' in results[metric]:
                score = results[metric]['score']
                if not np.isnan(score):
                    scores.append(score)
                    weights.append(weight)
        
        if not scores:
            return 0.0
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return np.average(scores, weights=weights)
    
    def _analyze_community_coverage(self) -> Dict[str, Any]:
        """Analyze how communities are distributed across the family."""
        if not self.family_graphs:
            return {}
        
        # Count community usage
        community_usage = {}
        all_communities = set()
        
        for graph in self.family_graphs:
            all_communities.update(graph.communities)
            for comm in graph.communities:
                # Convert numpy int to native Python int
                comm_key = int(comm)
                community_usage[comm_key] = community_usage.get(comm_key, 0) + 1
        
        total_graphs = len(self.family_graphs)
        
        # Calculate statistics
        usage_counts = list(community_usage.values())
        
        # Convert numpy types to native Python types
        return {
            'total_unique_communities': int(len(all_communities)),
            'universe_communities': int(self.universe.K),
            'coverage_fraction': float(len(all_communities) / self.universe.K),
            'community_usage': community_usage,
            'avg_usage_per_community': float(np.mean(usage_counts) if usage_counts else 0),
            'usage_std': float(np.std(usage_counts) if usage_counts else 0),
            'min_usage': int(min(usage_counts) if usage_counts else 0),
            'max_usage': int(max(usage_counts) if usage_counts else 0),
            'communities_in_all_graphs': [int(comm) for comm, count in community_usage.items() if count == total_graphs],
            'rarely_used_communities': [int(comm) for comm, count in community_usage.items() if count == 1]
        }
    
    def _interpret_score(self, score: float, metric_type: str) -> str:
        """Provide human-readable interpretation of consistency scores."""
        if np.isnan(score):
            return "Unable to calculate - insufficient data"
        
        interpretations = {
            'pattern': {
                0.8: "Excellent pattern preservation - rank ordering strongly maintained",
                0.6: "Good pattern preservation - structural relationships mostly maintained", 
                0.4: "Moderate pattern preservation - some structural similarity remains",
                0.2: "Poor pattern preservation - limited structural similarity",
                0.0: "Very poor pattern preservation - little structural relationship"
            },
            'fidelity': {
                0.8: "Excellent generation fidelity - graphs closely match targets",
                0.6: "Good generation fidelity - graphs reasonably match targets",
                0.4: "Moderate generation fidelity - some deviation from targets",
                0.2: "Poor generation fidelity - significant deviation from targets", 
                0.0: "Very poor generation fidelity - graphs don't match targets"
            },
            'degree': {
                0.8: "Strong degree-community relationship - degrees follow universe centers well",
                0.6: "Moderate degree-community relationship - some correlation with centers",
                0.4: "Weak degree-community relationship - limited correlation with centers",
                0.2: "Very weak degree-community relationship - little correlation with centers",
                0.0: "No degree-community relationship - no correlation with centers"
            },
            'cooccurrence': {
                0.8: "Strong co-occurrence pattern preservation - community relationships well maintained",
                0.6: "Good co-occurrence pattern preservation - community relationships mostly maintained",
                0.4: "Moderate co-occurrence pattern preservation - some community relationships preserved",
                0.2: "Weak co-occurrence pattern preservation - limited community relationship preservation",
                0.0: "Very weak co-occurrence pattern preservation - little community relationship preserved"
            },
            'overall': {
                0.8: "High overall consistency - family preserves universe structure well",
                0.6: "Moderate overall consistency - family shows some structural preservation",
                0.4: "Low overall consistency - family shows significant structural variation",
                0.2: "Very low overall consistency - family shows high structural diversity",
                0.0: "Minimal overall consistency - family shows very high structural diversity"
            }
        }
        
        thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]
        for threshold in thresholds:
            if score >= threshold:
                return interpretations[metric_type][threshold]
        
        return "Score out of expected range"
    
    def analyze_cooccurrence(self) -> Dict[str, Any]:
        """
        Simple analysis of community co-occurrence patterns in a graph family.
        
        Returns:
            Dictionary with correlation and difference matrix
        """
        from collections import defaultdict
        
        # Count how often each community pair appears together
        pair_counts = defaultdict(int)
        individual_counts = defaultdict(int)
        total_graphs = len(self.family_graphs)
        
        # Count occurrences
        for graph in self.family_graphs:
            communities = set(graph.communities)
            
            # Count individual community appearances
            for comm in communities:
                individual_counts[comm] += 1
            
            # Count pair co-occurrences
            comm_list = sorted(list(communities))
            for i in range(len(comm_list)):
                for j in range(i + 1, len(comm_list)):
                    pair = tuple(sorted([comm_list[i], comm_list[j]]))
                    pair_counts[pair] += 1
        
        # Calculate expected vs actual co-occurrence rates
        K = self.universe.K
        expected_matrix = self.universe.community_cooccurrence_matrix
        actual_matrix = np.zeros((K, K))
        
        # Fill actual co-occurrence matrix
        for (i, j), count in pair_counts.items():
            # Convert to probability (how often they co-occur when both are present)
            joint_appearances = count
            i_appearances = individual_counts[i]
            j_appearances = individual_counts[j]
            
            if i_appearances > 0 and j_appearances > 0:
                # Probability they appear together given at least one appears
                actual_prob = joint_appearances / min(i_appearances, j_appearances)
                actual_matrix[i, j] = actual_prob
                actual_matrix[j, i] = actual_prob
        
        # Set diagonal to 1 (communities always co-occur with themselves)
        np.fill_diagonal(actual_matrix, 1.0)
        
        # Calculate correlation between expected and actual
        # Only use upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(expected_matrix, dtype=bool), k=1)
        expected_flat = expected_matrix[mask]
        actual_flat = actual_matrix[mask]
        
        correlation = np.corrcoef(expected_flat, actual_flat)[0, 1] if len(expected_flat) > 1 else 0.0
        
        # Calculate difference matrix
        diff_matrix = actual_matrix - expected_matrix
        
        return {
            'correlation': correlation,
            'difference_matrix': diff_matrix
        }
    
    def create_consistency_dashboard(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a comprehensive visualization dashboard of consistency metrics.
        
        Args:
            figsize: Figure size for the dashboard
            
        Returns:
            Matplotlib figure object
        """
        # Commented out to prevent memory issues with too many open figures
        return None
        
        # if self.results is None:
        #     raise ValueError("Must run analyze_consistency() first")
        
        # fig = plt.figure(figsize=figsize)
        
        # # Create grid layout
        # gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # # 1. Overall consistency scores (top left)
        # ax1 = fig.add_subplot(gs[0, 0])
        # self._plot_consistency_scores(ax1)
        
        # # 2. Individual score distributions (top right)
        # ax2 = fig.add_subplot(gs[0, 1])
        # self._plot_score_distributions(ax2)
        
        # # 3. Community coverage (middle left)
        # ax3 = fig.add_subplot(gs[1, 0])
        # self._plot_community_coverage(ax3)
        
        # # 4. Pattern preservation details (middle right)
        # ax4 = fig.add_subplot(gs[1, 1])
        # self._plot_individual_scores(ax4, 'pattern_preservation', 'Pattern Preservation')
        
        # # 5. Generation fidelity details (bottom left)
        # ax5 = fig.add_subplot(gs[2, 0])
        # self._plot_individual_scores(ax5, 'generation_fidelity', 'Generation Fidelity')
        
        # # 6. Degree consistency details (bottom right)
        # ax6 = fig.add_subplot(gs[2, 1])
        # self._plot_individual_scores(ax6, 'degree_consistency', 'Degree Consistency')
        
        # # 7. Triangle consistency details (bottom left)
        # ax7 = fig.add_subplot(gs[3, 0])
        # self._plot_individual_scores(ax7, 'triangle_consistency', 'Triangle Consistency')
        
        # # 8. Co-occurrence consistency details (bottom right)
        # ax8 = fig.add_subplot(gs[3, 1])
        # self._plot_individual_scores(ax8, 'cooccurrence_consistency', 'Co-occurrence Consistency')
        
        # plt.suptitle('Graph Family Consistency Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # return fig
    
    def _plot_consistency_scores(self, ax):
        """Plot overall consistency scores as a bar chart."""
        # Commented out to prevent memory issues with too many open figures
        return
        
        # scores = []
        # labels = []
        # colors = []
        
        # metrics = ['pattern_preservation', 'generation_fidelity', 'degree_consistency', 'overall']
        # metric_labels = ['Pattern\nPreservation', 'Generation\nFidelity', 'Degree\nConsistency', 'Overall']
        
        # for i, metric in enumerate(metrics):
        #     if metric in self.results and 'score' in self.results[metric]:
        #         score = self.results[metric]['score']
        #         if not np.isnan(score):
        #             scores.append(score)
        #             labels.append(metric_labels[i])
        #             # Color based on score
        #             if score >= 0.8:
        #                 colors.append('green')
        #             elif score >= 0.6:
        #                 colors.append('orange')
        #             else:
        #                 colors.append('red')
        
        # if scores:
        #     bars = ax.bar(labels, scores, color=colors, alpha=0.7)
        #     ax.set_ylim(0, 1)
        #     ax.set_ylabel('Consistency Score')
        #     ax.set_title('Consistency Metrics Overview')
        #     ax.tick_params(axis='x', rotation=45)
            
        #     # Add value labels on bars
        #     for bar, score in zip(bars, scores):
        #         height = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #                f'{score:.3f}', ha='center', va='bottom')
        # else:
        #     ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_score_distributions(self, ax):
        """Plot distributions of individual scores."""
        # Commented out to prevent memory issues with too many open figures
        return
        
        # all_scores = []
        # labels = []
        
        # metrics = ['pattern_preservation', 'generation_fidelity', 'degree_consistency']
        
        # for metric in metrics:
        #     if metric in self.results and 'individual_correlations' in self.results[metric]:
        #         scores = self.results[metric]['individual_correlations']
        #     elif metric in self.results and 'individual_scores' in self.results[metric]:
        #         scores = self.results[metric]['individual_scores']
        #     else:
        #         continue
            
        #     if scores:
        #         all_scores.extend(scores)
        #         labels.extend([metric.replace('_', ' ').title()] * len(scores))
        
        # if all_scores:
        #     # Create violin plot
        #     data_dict = {}
        #     for score, label in zip(all_scores, labels):
        #         if label not in data_dict:
        #             data_dict[label] = []
        #         data_dict[label].append(score)
            
        #     positions = []
        #     data_for_violin = []
        #     tick_labels = []
            
        #     for i, (label, scores) in enumerate(data_dict.items()):
        #         positions.append(i)
        #         data_for_violin.append(scores)
        #         tick_labels.append(label)
            
        #     parts = ax.violinplot(data_for_violin, positions=positions)
        #     ax.set_xticks(positions)
        #     ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        #     ax.set_ylabel('Individual Scores')
        #     ax.set_title('Score Distributions')
        #     ax.set_ylim(-1, 1)
        # else:
        #     ax.text(0.5, 0.5, 'No individual scores available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_community_coverage(self, ax):
        """Plot community coverage statistics."""
        # Commented out to prevent memory issues with too many open figures
        return
        
        # if 'community_coverage' in self.results:
        #     coverage = self.results['community_coverage']
            
        #     # Create pie chart of community usage
        #     if 'community_usage' in coverage:
        #         usage_counts = list(coverage['community_usage'].values())
        #         if usage_counts:
        #             # Group by usage frequency
        #             usage_freq = {}
        #             for count in usage_counts:
        #                 usage_freq[count] = usage_freq.get(count, 0) + 1
                
        #             if len(usage_freq) > 1:
        #                 labels = [f'Used {k} times' for k in usage_freq.keys()]
        #                 sizes = list(usage_freq.values())
        #                 ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        #                 ax.set_title('Community Usage Distribution')
        #             else:
        #                 ax.text(0.5, 0.5, f'All communities used\n{list(usage_freq.keys())[0]} times', 
        #                        ha='center', va='center', transform=ax.transAxes)
            
        #     # Add coverage statistics as text
        #     coverage_text = f"Coverage: {coverage.get('coverage_fraction', 0):.1%}\n"
        #     coverage_text += f"Unique communities: {coverage.get('total_unique_communities', 0)}\n"
        #     coverage_text += f"Universe total: {coverage.get('universe_communities', 0)}"
            
        #     ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes, 
        #            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # else:
        #     ax.text(0.5, 0.5, 'No coverage data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_individual_scores(self, ax, metric_key, title):
        """Plot individual scores for a specific metric."""
        # Commented out to prevent memory issues with too many open figures
        return
        
        # if metric_key in self.results:
        #     result = self.results[metric_key]
            
        #     # Get individual scores
        #     scores = None
        #     if 'individual_correlations' in result:
        #         scores = result['individual_correlations']
        #     elif 'individual_scores' in result:
        #         scores = result['individual_scores'] 
        #     elif 'pairwise_correlations' in result:
        #         scores = result['pairwise_correlations']
            
        #     if scores:
        #         x_values = range(len(scores))
        #         ax.plot(x_values, scores, 'o-', alpha=0.7)
        #         ax.axhline(y=np.mean(scores), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(scores):.3f}')
        #         ax.set_xlabel('Graph/Pair Index')
        #         ax.set_ylabel('Score')
        #         ax.set_title(f'{title}\n(Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f})')
        #         ax.legend()
        #         ax.grid(True, alpha=0.3)
        #     else:
        #         ax.text(0.5, 0.5, f'No {title.lower()} data', ha='center', va='center', transform=ax.transAxes)
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report of the consistency analysis.
        
        Returns:
            Formatted string report
        """
        if self.results is None:
            return "No analysis results available. Run analyze_consistency() first."
        
        report = "GRAPH FAMILY CONSISTENCY ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall summary
        if 'overall' in self.results:
            overall = self.results['overall']
            report += f"OVERALL CONSISTENCY: {overall['score']:.3f}\n"
            report += f"Interpretation: {overall['interpretation']}\n\n"
        
        # Individual metrics
        metrics = ['pattern_preservation', 'generation_fidelity', 'degree_consistency']
        metric_names = ['Pattern Preservation', 'Generation Fidelity', 'Degree Consistency']
        
        for metric, name in zip(metrics, metric_names):
            if metric in self.results and 'score' in self.results[metric]:
                result = self.results[metric]
                report += f"{name.upper()}:\n"
                report += f"  Score: {result['score']:.3f} (±{result.get('std', 0):.3f})\n"
                report += f"  {result['description']}\n"
                report += f"  Interpretation: {result['interpretation']}\n\n"
        
        return report

    def analyze_family_overview(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of the graph family providing overview of properties.
        
        Returns:
            Dictionary with overview analysis including ranges, distributions, and statistics
        """
        if not self.graphs:
            return {'error': 'No graphs available for analysis'}
        
        # Use existing analysis function
        try:
            from experiments.inductive.data import analyze_graph_family_properties
            properties = analyze_graph_family_properties(self.graphs)
        except ImportError:
            # Fallback if the import fails
            properties = self._analyze_graph_family_properties_fallback()
        
        # Add additional analysis
        overview = {
            'basic_stats': properties,
            'property_ranges': {},
            'distributions': {},
            'correlations': {},
            'generation_metadata': self.generation_metadata if hasattr(self, 'generation_metadata') else []
        }
        
        # Calculate property ranges
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels']:
            if key in properties and properties[key]:
                values = properties[key]
                overview['property_ranges'][key] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }
        
        # Analyze target vs actual properties
        if self.generation_metadata:
            target_vs_actual = self._analyze_target_vs_actual()
            overview['target_vs_actual'] = target_vs_actual
        
        # Analyze community coverage
        community_analysis = self._analyze_community_coverage()
        overview['community_analysis'] = community_analysis
        
        return overview
    
    def _analyze_graph_family_properties_fallback(self) -> Dict[str, Any]:
        """Fallback method to analyze graph family properties when the import fails."""
        if not self.graphs:
            return {}
        
        properties = {
            'node_counts': [],
            'edge_counts': [],
            'densities': [],
            'avg_degrees': [],
            'clustering_coefficients': [],
            'community_counts': [],
            'homophily_levels': [],
            'generation_method_distribution': {}
        }
        
        for graph in self.graphs:
            # Node count
            properties['node_counts'].append(graph.n_nodes)
            
            # Edge count
            edge_count = graph.graph.number_of_edges()
            properties['edge_counts'].append(edge_count)
            
            # Density
            if graph.n_nodes > 1:
                density = edge_count / (graph.n_nodes * (graph.n_nodes - 1) / 2)
            else:
                density = 0.0
            properties['densities'].append(density)
            
            # Average degree
            if graph.n_nodes > 0:
                avg_degree = 2 * edge_count / graph.n_nodes
            else:
                avg_degree = 0.0
            properties['avg_degrees'].append(avg_degree)
            
            # Clustering coefficient (simplified)
            try:
                clustering = nx.average_clustering(graph.graph)
            except:
                clustering = 0.0
            properties['clustering_coefficients'].append(clustering)
            
            # Community count
            properties['community_counts'].append(len(graph.communities))
            
            # Homophily level (simplified calculation)
            try:
                homophily = self._calculate_simple_homophily(graph)
            except:
                homophily = 0.0
            properties['homophily_levels'].append(homophily)
            
            # Generation method
            method = getattr(graph, 'generation_method', 'unknown')
            properties['generation_method_distribution'][method] = properties['generation_method_distribution'].get(method, 0) + 1
        
        # Calculate summary statistics
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels']:
            if properties[key]:
                values = properties[key]
                properties[f'{key}_mean'] = float(np.mean(values))
                properties[f'{key}_std'] = float(np.std(values))
                properties[f'{key}_min'] = float(np.min(values))
                properties[f'{key}_max'] = float(np.max(values))
        
        return properties
    
    def _calculate_simple_homophily(self, graph) -> float:
        """Calculate a simple homophily measure for a graph."""
        try:
            # Count edges within communities vs between communities
            within_edges = 0
            total_edges = graph.graph.number_of_edges()
            
            for edge in graph.graph.edges():
                node1, node2 = edge
                comm1 = graph.community_labels[node1]
                comm2 = graph.community_labels[node2]
                if comm1 == comm2:
                    within_edges += 1
            
            if total_edges > 0:
                return within_edges / total_edges
            else:
                return 0.0
        except:
            return 0.0
    
    def _analyze_target_vs_actual(self) -> Dict[str, Any]:
        """Analyze how well graphs match their target parameters."""
        if not self.generation_metadata:
            return {}
        
        analysis = {
            'homophily': {'targets': [], 'actuals': [], 'differences': []},
            'density': {'targets': [], 'actuals': [], 'differences': []},
            'communities': {'targets': [], 'actuals': [], 'differences': []}
        }
        
        for i, metadata in enumerate(self.generation_metadata):
            if metadata.get('failed', False):
                continue
                
            # Homophily analysis
            if 'target_homophily' in metadata and 'homophily_levels' in self.analyze_family_overview()['basic_stats']:
                target_hom = metadata['target_homophily']
                actual_hom = self.analyze_family_overview()['basic_stats']['homophily_levels'][i]
                analysis['homophily']['targets'].append(target_hom)
                analysis['homophily']['actuals'].append(actual_hom)
                analysis['homophily']['differences'].append(abs(target_hom - actual_hom))
            
            # Density analysis
            if 'target_density' in metadata and 'densities' in self.analyze_family_overview()['basic_stats']:
                target_den = metadata['target_density']
                actual_den = self.analyze_family_overview()['basic_stats']['densities'][i]
                analysis['density']['targets'].append(target_den)
                analysis['density']['actuals'].append(actual_den)
                analysis['density']['differences'].append(abs(target_den - actual_den))
            
            # Community count analysis
            if 'n_communities' in metadata and 'community_counts' in self.analyze_family_overview()['basic_stats']:
                target_comm = metadata['n_communities']
                actual_comm = self.analyze_family_overview()['basic_stats']['community_counts'][i]
                analysis['communities']['targets'].append(target_comm)
                analysis['communities']['actuals'].append(actual_comm)
                analysis['communities']['differences'].append(abs(target_comm - actual_comm))
        
        # Calculate summary statistics
        for property_name in analysis:
            if analysis[property_name]['differences']:
                analysis[property_name]['mean_difference'] = float(np.mean(analysis[property_name]['differences']))
                analysis[property_name]['std_difference'] = float(np.std(analysis[property_name]['differences']))
                analysis[property_name]['max_difference'] = float(np.max(analysis[property_name]['differences']))
                analysis[property_name]['correlation'] = float(np.corrcoef(analysis[property_name]['targets'], analysis[property_name]['actuals'])[0, 1]) if len(analysis[property_name]['targets']) > 1 else 0.0
        
        return analysis
    
    def _analyze_community_coverage(self) -> Dict[str, Any]:
        """Analyze community coverage across the family."""
        if not self.graphs:
            return {}
        
        # Count community usage
        community_usage = {}
        all_communities = set()
        
        for graph in self.graphs:
            all_communities.update(graph.communities)
            for comm in graph.communities:
                comm_key = int(comm)
                community_usage[comm_key] = community_usage.get(comm_key, 0) + 1
        
        total_graphs = len(self.graphs)
        universe_communities = self.universe.K if hasattr(self, 'universe') else len(all_communities)
        
        return {
            'total_unique_communities': int(len(all_communities)),
            'universe_communities': int(universe_communities),
            'coverage_fraction': float(len(all_communities) / universe_communities),
            'community_usage': community_usage,
            'avg_usage_per_community': float(np.mean(list(community_usage.values())) if community_usage else 0),
            'usage_std': float(np.std(list(community_usage.values())) if community_usage else 0),
            'min_usage': int(min(community_usage.values()) if community_usage else 0),
            'max_usage': int(max(community_usage.values()) if community_usage else 0),
            'communities_in_all_graphs': [int(comm) for comm, count in community_usage.items() if count == total_graphs],
            'rarely_used_communities': [int(comm) for comm, count in community_usage.items() if count == 1]
        }
    
    def create_family_overview_dashboard(self, figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Create a comprehensive dashboard showing family overview.
        
        Args:
            figsize: Figure size for the dashboard
            
        Returns:
            Matplotlib figure object
        """
        if not self.graphs:
            # Create empty figure with message
            fig = plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, 'No graphs available for analysis',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    fontsize=16)
            return fig
        
        overview = self.analyze_family_overview()
        
        # Create figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Property ranges overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_property_ranges(ax1, overview)
        
        # 2. Homophily vs Density scatter (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_homophily_density_scatter(ax2, overview)
        
        # 3. Node count vs Edge count scatter (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_node_edge_scatter(ax3, overview)
        
        # 4. Target vs Actual analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_target_vs_actual(ax4, overview)
        
        # 5. Community coverage (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_community_coverage(ax5, overview)
        
        # 6. Property distributions (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_property_distributions(ax6, overview)
        
        # 7. Generation method distribution (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_generation_methods(ax7, overview)
        
        # 8. Clustering coefficient vs Community count (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_clustering_communities(ax8, overview)
        
        # 9. Summary statistics table (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary_table(ax9, overview)
        
        plt.suptitle('Graph Family Overview Dashboard', fontsize=16, fontweight='bold')
        
        return fig
    
    def _plot_property_ranges(self, ax, overview):
        """Plot property ranges as box plots."""
        if 'property_ranges' not in overview:
            ax.text(0.5, 0.5, 'No property data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Select key properties to plot
        key_properties = ['node_counts', 'edge_counts', 'densities', 'homophily_levels', 'avg_degrees']
        available_properties = [p for p in key_properties if p in overview['basic_stats']]
        
        if not available_properties:
            ax.text(0.5, 0.5, 'No property data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create box plot data
        data = []
        labels = []
        for prop in available_properties:
            if prop in overview['basic_stats'] and overview['basic_stats'][prop]:
                data.append(overview['basic_stats'][prop])
                labels.append(prop.replace('_', ' ').title())
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_title('Property Ranges')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_homophily_density_scatter(self, ax, overview):
        """Plot homophily vs density scatter plot."""
        if 'homophily_levels' in overview['basic_stats'] and 'densities' in overview['basic_stats']:
            homophily = overview['basic_stats']['homophily_levels']
            densities = overview['basic_stats']['densities']
            
            if homophily and densities:
                ax.scatter(homophily, densities, alpha=0.7, s=50)
                ax.set_xlabel('Homophily')
                ax.set_ylabel('Density')
                ax.set_title('Homophily vs Density')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(homophily) > 1:
                    z = np.polyfit(homophily, densities, 1)
                    p = np.poly1d(z)
                    ax.plot(homophily, p(homophily), "r--", alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No homophily/density data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_node_edge_scatter(self, ax, overview):
        """Plot node count vs edge count scatter plot."""
        if 'node_counts' in overview['basic_stats'] and 'edge_counts' in overview['basic_stats']:
            nodes = overview['basic_stats']['node_counts']
            edges = overview['basic_stats']['edge_counts']
            
            if nodes and edges:
                ax.scatter(nodes, edges, alpha=0.7, s=50)
                ax.set_xlabel('Node Count')
                ax.set_ylabel('Edge Count')
                ax.set_title('Nodes vs Edges')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(nodes) > 1:
                    z = np.polyfit(nodes, edges, 1)
                    p = np.poly1d(z)
                    ax.plot(nodes, p(nodes), "r--", alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No node/edge data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_target_vs_actual(self, ax, overview):
        """Plot target vs actual parameter comparison."""
        if 'target_vs_actual' not in overview:
            ax.text(0.5, 0.5, 'No target vs actual data', ha='center', va='center', transform=ax.transAxes)
            return
        
        target_vs_actual = overview['target_vs_actual']
        
        # Plot homophily comparison
        if 'homophily' in target_vs_actual and target_vs_actual['homophily']['targets']:
            targets = target_vs_actual['homophily']['targets']
            actuals = target_vs_actual['homophily']['actuals']
            
            ax.scatter(targets, actuals, alpha=0.7, s=50, label='Homophily')
            
            # Add perfect correlation line
            min_val = min(min(targets), min(actuals))
            max_val = max(max(targets), max(actuals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect')
            
            ax.set_xlabel('Target')
            ax.set_ylabel('Actual')
            ax.set_title('Target vs Actual Parameters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No target vs actual data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_community_coverage(self, ax, overview):
        """Plot community coverage analysis."""
        if 'community_analysis' not in overview:
            ax.text(0.5, 0.5, 'No community data', ha='center', va='center', transform=ax.transAxes)
            return
        
        community_analysis = overview['community_analysis']
        
        if 'community_usage' in community_analysis:
            usage_counts = list(community_analysis['community_usage'].values())
            if usage_counts:
                ax.hist(usage_counts, bins=min(10, len(set(usage_counts))), alpha=0.7, edgecolor='black')
                ax.set_xlabel('Usage Count')
                ax.set_ylabel('Number of Communities')
                ax.set_title('Community Usage Distribution')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No community usage data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No community data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_property_distributions(self, ax, overview):
        """Plot distributions of key properties."""
        if 'basic_stats' not in overview:
            ax.text(0.5, 0.5, 'No property data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot density distribution
        if 'densities' in overview['basic_stats'] and overview['basic_stats']['densities']:
            densities = overview['basic_stats']['densities']
            ax.hist(densities, bins=min(10, len(set(densities))), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Density')
            ax.set_ylabel('Frequency')
            ax.set_title('Density Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No density data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_generation_methods(self, ax, overview):
        """Plot generation method distribution."""
        if 'basic_stats' in overview and 'generation_method_distribution' in overview['basic_stats']:
            method_dist = overview['basic_stats']['generation_method_distribution']
            if method_dist:
                methods = list(method_dist.keys())
                counts = list(method_dist.values())
                
                bars = ax.bar(methods, counts, alpha=0.7)
                ax.set_xlabel('Generation Method')
                ax.set_ylabel('Count')
                ax.set_title('Generation Method Distribution')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                           str(count), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No generation method data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No generation method data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_clustering_communities(self, ax, overview):
        """Plot clustering coefficient vs community count."""
        if ('clustering_coefficients' in overview['basic_stats'] and 
            'community_counts' in overview['basic_stats']):
            clustering = overview['basic_stats']['clustering_coefficients']
            communities = overview['basic_stats']['community_counts']
            
            if clustering and communities:
                ax.scatter(communities, clustering, alpha=0.7, s=50)
                ax.set_xlabel('Community Count')
                ax.set_ylabel('Clustering Coefficient')
                ax.set_title('Clustering vs Communities')
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No clustering/community data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_summary_table(self, ax, overview):
        """Plot summary statistics table."""
        ax.axis('tight')
        ax.axis('off')
        
        if 'basic_stats' in overview:
            stats = overview['basic_stats']
            
            # Create table data
            table_data = []
            table_data.append(['Metric', 'Mean', 'Std', 'Min', 'Max'])
            
            key_metrics = ['node_counts', 'edge_counts', 'densities', 'homophily_levels', 'avg_degrees']
            for metric in key_metrics:
                if f'{metric}_mean' in stats:
                    table_data.append([
                        metric.replace('_', ' ').title(),
                        f"{stats[f'{metric}_mean']:.3f}",
                        f"{stats[f'{metric}_std']:.3f}",
                        f"{stats[f'{metric}_min']:.3f}",
                        f"{stats[f'{metric}_max']:.3f}"
                    ])
            
            if len(table_data) > 1:
                table = ax.table(cellText=table_data, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Color header row
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
            else:
                ax.text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=ax.transAxes)
    
