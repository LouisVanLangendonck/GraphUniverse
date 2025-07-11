"""
Graph family generation utilities for MMSB graphs.

This module provides functionality for generating families of graphs
sampled from a single graph universe, with varying parameters across
the family members.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from mmsb.model import GraphUniverse, GraphSample
from tqdm import tqdm
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from itertools import combinations

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
        feature_regime_balance: float = 0.5,
        homophily_range: Tuple[float, float] = (0.0, 0.2),  # Range around universe homophily
        density_range: Tuple[float, float] = (0.0, 0.2),    # Range around universe density
        use_dccc_sbm: bool = False,
        community_imbalance_range: Tuple[float, float] = (0.0, 0.0),  # Range for community imbalance
        degree_separation_range: Tuple[float, float] = (0.5, 0.5),    # Range for degree separation
        degree_method: str = "standard",
        # Community density variation
        community_density_variation: float = 0.0,
        # Community co-occurrence homogeneity
        community_cooccurrence_homogeneity: float = 1.0,
        # Deviation limiting
        disable_deviation_limiting: bool = False,
        max_mean_community_deviation: float = 0.15,
        max_max_community_deviation: float = 0.3,
        min_edge_density: float = 0.005,
        # DCCC distribution-specific parameter ranges
        degree_distribution: str = "standard",
        power_law_exponent_range: Tuple[float, float] = (2.0, 3.5),
        exponential_rate_range: Tuple[float, float] = (0.3, 1.0),
        uniform_min_factor_range: Tuple[float, float] = (0.3, 0.7),
        uniform_max_factor_range: Tuple[float, float] = (1.3, 2.0),
        # Triangle density parameters
        triangle_density: float = 0.0,
        triangle_community_relation_homogeneity: float = 1.0,
        # Fixed parameters for all graphs in family
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.1,
        target_avg_degree: Optional[float] = None,
        max_parameter_search_attempts: int = 20,
        parameter_search_range: float = 0.5,
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
            feature_regime_balance: Feature regime balance parameter
            homophily_range: Tuple of (min_offset, max_offset) around universe homophily
            density_range: Tuple of (min_offset, max_offset) around universe density
            use_dccc_sbm: Whether to use DCCC-SBM model
            community_imbalance_range: Range for community size imbalance (min, max)
            degree_separation_range: Range for degree distribution separation (min, max)
            degree_method: Degree generation method
            disable_deviation_limiting: Whether to disable deviation checks
            max_mean_community_deviation: Maximum allowed mean community deviation
            max_max_community_deviation: Maximum allowed max community deviation
            min_edge_density: Minimum edge density for graphs
            degree_distribution: Degree distribution type ("standard", "power_law", "exponential", "uniform")
            power_law_exponent_range: Range for power law exponent (min, max)
            exponential_rate_range: Range for exponential distribution rate (min, max)
            uniform_min_factor_range: Range for uniform distribution min factor (min, max)
            uniform_max_factor_range: Range for uniform distribution max factor (min, max)
            degree_heterogeneity: Fixed degree heterogeneity for all graphs
            edge_noise: Fixed edge noise level for all graphs
            target_avg_degree: Target average degree
            max_parameter_search_attempts: Max attempts for parameter search
            parameter_search_range: Range for parameter search
            max_retries: Maximum retries for graph generation
            seed: Random seed for reproducibility
        """
        self.universe = universe
        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes
        self.min_communities = min_communities
        self.max_communities = max_communities if max_communities is not None else universe.K
        self.min_component_size = min_component_size
        self.feature_regime_balance = feature_regime_balance
        self.homophily_range = homophily_range
        self.density_range = density_range
        self.use_dccc_sbm = use_dccc_sbm
        self.community_imbalance_range = community_imbalance_range
        self.degree_separation_range = degree_separation_range
        self.degree_method = degree_method
        # Community density variation
        self.community_density_variation = community_density_variation
        # Community co-occurrence homogeneity
        self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity
        # Triangle density
        self.triangle_density = triangle_density
        # Triangle community relation homogeneity
        self.triangle_community_relation_homogeneity = triangle_community_relation_homogeneity

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
        
        # Fixed parameters for all graphs
        self.degree_heterogeneity = degree_heterogeneity
        self.edge_noise = edge_noise
        self.target_avg_degree = target_avg_degree
        self.max_parameter_search_attempts = max_parameter_search_attempts
        self.parameter_search_range = parameter_search_range
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
        if len(self.density_range) != 2 or self.density_range[0] > self.density_range[1]:
            raise ValueError("density_range must be a tuple (min, max) with min <= max")
        if len(self.community_imbalance_range) != 2 or self.community_imbalance_range[0] > self.community_imbalance_range[1]:
            raise ValueError("community_imbalance_range must be a tuple (min, max) with min <= max")
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
        
        # Sample target homophily (relative to universe homophily)
        homophily_offset = np.random.uniform(self.homophily_range[0], self.homophily_range[1])
        pos_or_neg = np.random.choice([-1, 1])
        target_homophily = np.clip(
            self.universe.homophily + homophily_offset*pos_or_neg,
            0.0, 1.0
        )
        
        # Sample target density (relative to universe density)
        density_offset = np.random.uniform(self.density_range[0], self.density_range[1])
        pos_or_neg = np.random.choice([-1, 1])
        target_density = np.clip(
            self.universe.edge_density + density_offset*pos_or_neg,
            self.min_edge_density, 1.0
        )
        
        # Sample DCCC parameters if using DCCC-SBM
        dccc_params = {}
        if self.use_dccc_sbm:
            # Sample community imbalance
            community_imbalance = np.random.uniform(
                self.community_imbalance_range[0], 
                self.community_imbalance_range[1]
            )
            
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
                "community_imbalance": community_imbalance,
                "degree_separation": degree_separation,
                "dccc_global_degree_params": distribution_params,
                "power_law_exponent": distribution_params.get("exponent", None)
            }
        
        # # Sample community subset
        # communities = self.universe.sample_connected_community_subset(
        #     size=n_communities,
        #     existing_communities=None
        # )
        
        # Combine all parameters
        params = {
            'n_nodes': n_nodes,
            'target_homophily': target_homophily,
            'target_density': target_density,
            'n_communities': n_communities,
            'homophily_offset': homophily_offset,
            'density_offset': density_offset
        }
        
        # Add DCCC parameters
        params.update(dccc_params)
        
        return params
    
    def generate_family(
        self,
        n_graphs: int,
        show_progress: bool = True,
        collect_stats: bool = True,
        max_attempts_per_graph: int = 5,
        timeout_minutes: float = 2.0,
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
                    # Sample parameters for this graph
                    params = self._sample_graph_parameters()

                    if allowed_community_combinations is not None:
                        sampled_community_combination_index = np.random.randint(0, len(allowed_community_combinations))
                        sampled_community_combination = allowed_community_combinations[sampled_community_combination_index]
                    
                    # Create graph sample
                    graph_sample = GraphSample(
                        universe=self.universe,
                        num_communities=params['n_communities'],
                        n_nodes=params['n_nodes'],
                        min_component_size=self.min_component_size,
                        degree_heterogeneity=self.degree_heterogeneity,
                        edge_noise=self.edge_noise,
                        feature_regime_balance=self.feature_regime_balance,
                        target_homophily=params['target_homophily'],
                        target_density=params['target_density'],
                        use_configuration_model=False,  # Not allowing config models for now
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        target_avg_degree=self.target_avg_degree,
                        max_mean_community_deviation=self.max_mean_community_deviation,
                        max_max_community_deviation=self.max_max_community_deviation,
                        max_parameter_search_attempts=self.max_parameter_search_attempts,
                        parameter_search_range=self.parameter_search_range,
                        min_edge_density=self.min_edge_density,
                        max_retries=self.max_retries,
                        use_dccc_sbm=self.use_dccc_sbm,
                        community_imbalance=params.get('community_imbalance', 0.0),
                        degree_separation=params.get('degree_separation', 0.5),
                        dccc_global_degree_params=params.get('dccc_global_degree_params', {}),
                        degree_method=self.degree_method,
                        disable_deviation_limiting=self.disable_deviation_limiting,
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
    
    def get_family_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of the generated graph family.
        
        Returns:
            DataFrame with graph properties and metadata
        """
        if not self.graphs:
            return pd.DataFrame()
        
        # Extract data for DataFrame
        data = []
        for i, (graph, metadata) in enumerate(zip(self.graphs, self.generation_metadata)):
            if metadata.get('failed', False):
                continue
                
            row = {
                'graph_id': i,
                'n_nodes': graph.n_nodes,
                'n_edges': graph.graph.number_of_edges(),
                'n_communities': len(graph.communities),
                'density': metadata['actual_density'],
                'target_homophily': metadata.get('target_homophily', None),
                'target_density': metadata.get('target_density', None),
                'homophily_offset': metadata.get('homophily_offset', None),
                'density_offset': metadata.get('density_offset', None),
                'generation_method': metadata.get('generation_method', 'unknown'),
                'attempts': metadata.get('attempts', 1),
                'total_generation_time': metadata.get('timing_info', {}).get('total', None)
            }
            
            # Add community-specific information
            community_sizes = np.bincount(graph.community_labels)
            row.update({
                'min_community_size': np.min(community_sizes),
                'max_community_size': np.max(community_sizes),
                'community_size_std': np.std(community_sizes)
            })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_family_diversity(self) -> Dict[str, Any]:
        """
        Analyze the diversity of the generated graph family.
        
        Returns:
            Dictionary with diversity metrics
        """
        if not self.graphs:
            return {}
        
        # Collect graph properties
        node_counts = [g.n_nodes for g in self.graphs]
        edge_counts = [g.graph.number_of_edges() for g in self.graphs]
        community_counts = [len(g.communities) for g in self.graphs]
        
        # Calculate diversity metrics
        diversity_metrics = {
            'node_count_range': (min(node_counts), max(node_counts)),
            'edge_count_range': (min(edge_counts), max(edge_counts)),
            'community_count_range': (min(community_counts), max(community_counts)),
            'node_count_cv': np.std(node_counts) / np.mean(node_counts) if np.mean(node_counts) > 0 else 0,
            'edge_count_cv': np.std(edge_counts) / np.mean(edge_counts) if np.mean(edge_counts) > 0 else 0,
            'community_count_cv': np.std(community_counts) / np.mean(community_counts) if np.mean(community_counts) > 0 else 0
        }
        
        # Community overlap analysis
        all_communities = set()
        for graph in self.graphs:
            all_communities.update(graph.communities)
        
        community_usage = {}
        for comm in all_communities:
            community_usage[comm] = sum(1 for graph in self.graphs if comm in graph.communities)
        
        diversity_metrics.update({
            'total_unique_communities': len(all_communities),
            'community_usage_distribution': community_usage,
            'avg_communities_per_graph': np.mean(community_counts),
            'community_reuse_rate': np.mean(list(community_usage.values())) / len(self.graphs) if self.graphs else 0
        })
        
        return diversity_metrics
    
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
                'community_imbalance_range': self.community_imbalance_range,
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
        
        # 4. Triangle consistency
        try:
            triangle_analysis = self.calculate_triangle_signal_strength()
            if 'triangle_signal_strength' in triangle_analysis:
                results['triangle_consistency'] = {
                    'score': triangle_analysis['triangle_signal_strength'],
                    'mean_correlation': triangle_analysis['mean_correlation'],
                    'mean_triangle_density': triangle_analysis['mean_triangle_density'],
                    'std': triangle_analysis['std_correlation'],
                    'interpretation': self._interpret_score(triangle_analysis['triangle_signal_strength'], 'triangle'),
                    'description': 'How well triangle patterns are preserved across the family'
                }
        except Exception as e:
            results['triangle_consistency'] = {'error': str(e)}
        
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
    
    def calculate_triangle_signal_strength(self) -> Dict[str, Any]:
        """
        Calculate triangle signal strength for the graph family.
        
        Returns:
            Dictionary with triangle signal strength score and basic metrics
        """
        if not self.family_graphs:
            return {"error": "No graphs in family"}
        
        correlations = []
        triangle_densities = []
        
        for graph in self.family_graphs:
            try:
                # Get triangle analysis
                triangle_analysis = graph.analyze_triangles()
                
                # Store correlation if valid
                correlation = triangle_analysis.get("triangle_propensity_correlation", 0.0)
                if not np.isnan(correlation):
                    correlations.append(correlation)
                
                # Calculate triangle density
                total_triangles = triangle_analysis.get("total_triangles", 0)
                n_nodes = graph.n_nodes
                max_possible = n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6 if n_nodes >= 3 else 1
                triangle_density = total_triangles / max_possible
                triangle_densities.append(triangle_density)
                
            except Exception:
                continue
        
        if not correlations or not triangle_densities:
            return {"error": "No valid triangle data"}
        
        # Calculate main score: average correlation weighted by triangle presence
        mean_correlation = np.mean(correlations)
        mean_density = np.mean(triangle_densities)
        
        # Triangle signal strength = correlation quality * density presence
        density_factor = min(1.0, mean_density * 1000)  # Scale up small densities
        triangle_signal_strength = max(0.0, mean_correlation) * density_factor
        
        return {
            "triangle_signal_strength": float(triangle_signal_strength),
            "mean_correlation": float(mean_correlation),
            "mean_triangle_density": float(mean_density),
            "std_correlation": float(np.std(correlations)),
            "n_graphs": len(correlations)
        }

    def _calculate_overall_consistency(self, results: Dict) -> float:
        """Calculate weighted average of all successful consistency metrics."""
        scores = []
        weights = []
        
        # Define weights for different metrics
        metric_weights = {
            'pattern_preservation': 0.3,
            'generation_fidelity': 0.3,
            'degree_consistency': 0.15,
            'triangle_consistency': 0.15,
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
            'triangle': {
                0.8: "Strong triangle pattern preservation - triangle structures well maintained",
                0.6: "Good triangle pattern preservation - triangle structures mostly maintained",
                0.4: "Moderate triangle pattern preservation - some triangle patterns preserved",
                0.2: "Weak triangle pattern preservation - limited triangle pattern preservation",
                0.0: "Very weak triangle pattern preservation - little triangle structure preserved"
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