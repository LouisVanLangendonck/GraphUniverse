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

class GraphFamilyGenerator:
    """
    Generates families of graphs sampled from a graph universe.
    Each graph in the family is a subgraph of the universe with its own community structure.
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        n_graphs: int,
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
        # Fixed parameters for all graphs in family
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.1,
        target_avg_degree: Optional[float] = None,
        triangle_enhancement: float = 0.0,
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
            triangle_enhancement: Triangle formation enhancement
            max_parameter_search_attempts: Max attempts for parameter search
            parameter_search_range: Range for parameter search
            max_retries: Maximum retries for graph generation
            seed: Random seed for reproducibility
        """
        self.universe = universe
        self.n_graphs = n_graphs
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
        self.triangle_enhancement = triangle_enhancement
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
        if self.n_graphs <= 0:
            raise ValueError("n_graphs must be positive")
        
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
        target_homophily = np.clip(
            self.universe.homophily + homophily_offset,
            0.0, 1.0
        )
        
        # Sample target density (relative to universe density)
        density_offset = np.random.uniform(self.density_range[0], self.density_range[1])
        target_density = np.clip(
            self.universe.edge_density + density_offset,
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
        
        # Sample community subset
        communities = self.universe.sample_connected_community_subset(
            size=n_communities,
            existing_communities=None
        )
        
        # Combine all parameters
        params = {
            'n_nodes': n_nodes,
            'communities': communities,
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
        show_progress: bool = True,
        collect_stats: bool = True,
        max_attempts_per_graph: int = 5
    ) -> List[GraphSample]:
        """
        Generate a family of graphs from the universe.
        
        Args:
            show_progress: Whether to show progress bar
            collect_stats: Whether to collect generation statistics
            max_attempts_per_graph: Maximum attempts per graph before giving up
            
        Returns:
            List of generated GraphSample objects
        """
        start_time = time.time()
        self.graphs = []
        self.generation_metadata = []
        failed_graphs = 0
        
        # Progress bar setup
        iterator = range(self.n_graphs)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating graph family")
        
        for graph_idx in iterator:
            graph_generated = False
            attempts = 0
            
            while not graph_generated and attempts < max_attempts_per_graph:
                attempts += 1
                
                try:
                    # Sample parameters for this graph
                    params = self._sample_graph_parameters()
                    
                    # Create graph sample
                    graph_sample = GraphSample(
                        universe=self.universe,
                        communities=params['communities'],
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
                        triangle_enhancement=self.triangle_enhancement,
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
                        disable_deviation_limiting=self.disable_deviation_limiting
                    )
                    
                    # Store graph and metadata
                    self.graphs.append(graph_sample)
                    
                    metadata = {
                        'graph_id': graph_idx,
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
                    
                except Exception as e:
                    if attempts == max_attempts_per_graph:
                        warnings.warn(f"Failed to generate graph {graph_idx} after {attempts} attempts: {e}")
                        failed_graphs += 1
                        # Add empty metadata for failed graph
                        self.generation_metadata.append({
                            'graph_id': graph_idx,
                            'attempts': attempts,
                            'failed': True,
                            'error': str(e)
                        })
                    # Continue to next attempt
        
        # Collect generation statistics
        if collect_stats:
            self._collect_generation_stats(start_time, failed_graphs)
        
        return self.graphs
    
    def _collect_generation_stats(self, start_time: float, failed_graphs: int) -> None:
        """Collect statistics about the generation process."""
        total_time = time.time() - start_time
        successful_graphs = len(self.graphs)
        
        # Basic stats
        self.generation_stats = {
            'total_time': total_time,
            'successful_graphs': successful_graphs,
            'failed_graphs': failed_graphs,
            'success_rate': successful_graphs / self.n_graphs if self.n_graphs > 0 else 0,
            'avg_time_per_graph': total_time / self.n_graphs if self.n_graphs > 0 else 0
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
    
    def save_family(self, filepath: str, include_graphs: bool = True) -> None:
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
                'n_graphs': self.n_graphs,
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