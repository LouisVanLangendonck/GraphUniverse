"""
Sampling utilities for generating graph instances from the universe.

This module provides specialized sampling functions for generating
diverse graph datasets with controlled properties.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Callable
import sys
import os

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mmsb.model import GraphUniverse, GraphSample, MMSBBenchmark


class GraphSampler:
    """
    Handles specialized sampling strategies for graph generation.
    
    This class provides methods to generate graph instances with 
    specific properties, controlled distribution shifts, and
    diverse community structures.
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        seed: Optional[int] = None
    ):
        """
        Initialize the sampler with a graph universe.
        
        Args:
            universe: The graph universe to sample from
            seed: Random seed for reproducibility
        """
        self.universe = universe
        
        if seed is not None:
            np.random.seed(seed)
    
    def sample_graphs_by_community_pattern(
        self,
        n_graphs: int,
        community_patterns: List[List[int]],
        min_nodes: int = 100,
        max_nodes: int = 500,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        min_component_size: int = 0,
        indirect_influence: float = 0.1
    ) -> List[GraphSample]:
        """
        Sample graphs with specific community patterns.
        
        Args:
            n_graphs: Number of graphs to generate
            community_patterns: List of community subset patterns to sample from
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            degree_heterogeneity: Controls degree variability
            edge_noise: Random noise added to edge probabilities
            min_component_size: Minimum size for a component to be kept
            indirect_influence: How strongly co-memberships influence edge formation
            
        Returns:
            List of generated graph samples
        """
        graphs = []
        
        # Generate graphs by sampling from provided patterns
        for i in range(n_graphs):
            # Select a pattern
            pattern_idx = i % len(community_patterns)
            communities = community_patterns[pattern_idx]
            
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph sample
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                min_component_size=min_component_size,
                indirect_influence=indirect_influence
            )
            
            graphs.append(graph)
            
        return graphs
    
    def sample_graphs_with_increasing_complexity(
        self,
        n_graphs: int,
        min_communities: int = 3,
        max_communities: int = 20,
        min_nodes: int = 100,
        max_nodes: int = 1000,
        min_degree_heterogeneity: float = 0.2,
        max_degree_heterogeneity: float = 0.8
    ) -> List[GraphSample]:
        """
        Sample graphs with gradually increasing complexity.
        
        Args:
            n_graphs: Number of graphs to generate
            min_communities: Minimum number of communities
            max_communities: Maximum number of communities
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            min_degree_heterogeneity: Minimum degree heterogeneity
            max_degree_heterogeneity: Maximum degree heterogeneity
            
        Returns:
            List of generated graph samples sorted by complexity
        """
        graphs = []
        
        # Generate parameter values with increasing complexity
        complexity_steps = np.linspace(0, 1, n_graphs)
        
        for i, step in enumerate(complexity_steps):
            # Scale parameters based on complexity
            n_communities = int(min_communities + step * (max_communities - min_communities))
            n_nodes = int(min_nodes + step * (max_nodes - min_nodes))
            degree_heterogeneity = min_degree_heterogeneity + step * (max_degree_heterogeneity - min_degree_heterogeneity)
            
            # Sample communities
            if i < n_graphs // 3:
                # First third: mostly assortative communities
                method = "random"
                similarity_bias = 0.5
            elif i < 2 * n_graphs // 3:
                # Middle third: mixed communities
                method = "correlated"
                similarity_bias = 0.0
            else:
                # Last third: more diverse and challenging communities
                method = "diverse"
                similarity_bias = -0.5
                
            communities = self.universe.sample_community_subset(
                size=n_communities,
                method=method,
                similarity_bias=similarity_bias
            )
            
            # Generate graph sample
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=0.05 * step,  # Gradually increase noise
                min_component_size=0,
                indirect_influence=0.1
            )
            
            graphs.append(graph)
            
        return graphs
    
    def sample_graphs_with_community_evolution(
        self,
        n_graphs: int,
        initial_communities: List[int],
        evolution_type: str = "drift",
        min_nodes: int = 100,
        max_nodes: int = 500,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        min_component_size: int = 0,
        indirect_influence: float = 0.1
    ) -> List[GraphSample]:
        """
        Sample graphs with evolving community structure.
        
        Args:
            n_graphs: Number of graphs to generate
            initial_communities: Initial set of communities
            evolution_type: Type of evolution ("drift", "merge_split", "birth_death")
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            degree_heterogeneity: Controls degree variability
            edge_noise: Random noise added to edge probabilities
            min_component_size: Minimum size for a component to be kept
            indirect_influence: How strongly co-memberships influence edge formation
            
        Returns:
            List of generated graph samples showing community evolution
        """
        graphs = []
        current_communities = initial_communities.copy()
        
        for i in range(n_graphs):
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph with current communities
            graph = GraphSample(
                universe=self.universe,
                communities=current_communities,
                n_nodes=n_nodes,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                min_component_size=min_component_size,
                indirect_influence=indirect_influence
            )
            
            graphs.append(graph)
            
            # Evolve communities based on evolution type
            if evolution_type == "drift":
                # Randomly add or remove one community
                if np.random.random() < 0.5 and len(current_communities) > 1:
                    current_communities.pop(np.random.randint(len(current_communities)))
                else:
                    new_comm = np.random.choice(self.universe.K)
                    while new_comm in current_communities:
                        new_comm = np.random.choice(self.universe.K)
                    current_communities.append(new_comm)
            elif evolution_type == "merge_split":
                # Randomly merge or split communities
                if np.random.random() < 0.5 and len(current_communities) > 1:
                    # Merge two random communities
                    idx1, idx2 = np.random.choice(len(current_communities), 2, replace=False)
                    current_communities.pop(max(idx1, idx2))
                else:
                    # Split a random community into two
                    comm = np.random.choice(current_communities)
                    new_comm = np.random.choice(self.universe.K)
                    while new_comm in current_communities:
                        new_comm = np.random.choice(self.universe.K)
                    current_communities.append(new_comm)
            elif evolution_type == "birth_death":
                # Randomly add or remove communities
                if np.random.random() < 0.5 and len(current_communities) > 1:
                    current_communities.pop(np.random.randint(len(current_communities)))
                else:
                    new_comm = np.random.choice(self.universe.K)
                    while new_comm in current_communities:
                        new_comm = np.random.choice(self.universe.K)
                    current_communities.append(new_comm)
            
        return graphs
    
    def sample_correlated_graphs(
        self,
        n_graphs: int,
        base_communities: List[int],
        correlation_strength: float = 0.7,
        min_communities: int = 5,
        max_communities: int = 15,
        min_nodes: int = 100,
        max_nodes: int = 500
    ) -> List[GraphSample]:
        """
        Sample graphs with correlation to a base community set.
        
        Args:
            n_graphs: Number of graphs to generate
            base_communities: Base set of communities to correlate with
            correlation_strength: How strongly to correlate (0-1)
            min_communities: Minimum communities per graph
            max_communities: Maximum communities per graph
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            
        Returns:
            List of generated graph samples with correlation structure
        """
        graphs = []
        
        for i in range(n_graphs):
            # Sample number of communities
            n_communities = np.random.randint(min_communities, max_communities + 1)
            
            # Sample number of shared communities
            n_shared = int(correlation_strength * n_communities)
            n_shared = min(n_shared, len(base_communities))
            n_shared = max(1, n_shared)  # At least one shared community
            
            # Sample shared communities
            if n_shared < len(base_communities):
                shared_communities = np.random.choice(base_communities, size=n_shared, replace=False).tolist()
            else:
                shared_communities = base_communities.copy()
            
            # Sample additional communities
            n_additional = n_communities - n_shared
            
            if n_additional > 0:
                remaining_comms = set(range(self.universe.K)) - set(shared_communities)
                
                if len(remaining_comms) >= n_additional:
                    additional_communities = np.random.choice(
                        list(remaining_comms),
                        size=n_additional,
                        replace=False
                    ).tolist()
                else:
                    additional_communities = list(remaining_comms)
                
                # Combine communities
                communities = shared_communities + additional_communities
            else:
                communities = shared_communities
            
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph sample
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                degree_heterogeneity=0.5,
                edge_noise=0.0,
                min_component_size=0,
                indirect_influence=0.1
            )
            
            graphs.append(graph)
            
        return graphs
    
    def sample_transfer_learning_benchmark(
        self,
        n_pretrain: int,
        n_transfer: int,
        transfer_modes: List[str] = ["new_combinations", "rare_communities", "novel_communities"],
        min_communities: int = 5,
        max_communities: int = 15,
        min_nodes: int = 100,
        max_nodes: int = 500,
        transfer_difficulty: float = 0.5
    ) -> Dict[str, List[GraphSample]]:
        """
        Sample a complete benchmark dataset for transfer learning.
        
        Args:
            n_pretrain: Number of pretraining graphs
            n_transfer: Number of transfer graphs
            transfer_modes: List of transfer modes to include
            min_communities: Minimum communities per graph
            max_communities: Maximum communities per graph
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            transfer_difficulty: How challenging the transfer should be (0-1)
            
        Returns:
            Dictionary with "pretrain" and "transfer" graph lists
        """
        # Create benchmark
        benchmark = MMSBBenchmark(
            K=self.universe.K,
            feature_dim=self.universe.feature_dim,
            block_structure="hierarchical",
            overlap_structure="modular",
            edge_density=0.1,
            inter_community_density=0.01,
            overlap_density=0.2
        )
        
        # Generate pretraining graphs
        pretrain_graphs = benchmark.generate_pretraining_graphs(
            n_graphs=n_pretrain,
            min_communities=min_communities,
            max_communities=max_communities,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            avg_memberships=1.5,
            membership_concentration=5.0,
            degree_heterogeneity=0.5,
            edge_noise=0.0,
            sampling_method="random"
        )
        
        # Generate transfer graphs for each mode
        transfer_graphs = []
        
        # Calculate how many graphs per mode
        n_per_mode = n_transfer // len(transfer_modes)
        remainder = n_transfer % len(transfer_modes)
        
        for i, mode in enumerate(transfer_modes):
            # Add extras to first few modes
            mode_count = n_per_mode + (1 if i < remainder else 0)
            
            mode_graphs = benchmark.generate_transfer_graphs(
                n_graphs=mode_count,
                reference_graphs=pretrain_graphs,
                transfer_mode=mode,
                transfer_difficulty=transfer_difficulty,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                avg_memberships=1.5
            )
            
            transfer_graphs.extend(mode_graphs)
        
        return {
            "pretrain": pretrain_graphs,
            "transfer": transfer_graphs,
            "universe": benchmark.universe
        }
    
    def sample_diverse_difficulty_benchmark(
        self,
        n_graphs: int,
        difficulty_levels: int = 4,
        min_communities: int = 3,
        max_communities: int = 20,
        min_nodes: int = 100,
        max_nodes: int = 1000
    ) -> Dict[str, List[GraphSample]]:
        """
        Sample graphs with increasing levels of difficulty.
        
        Args:
            n_graphs: Total number of graphs to generate
            difficulty_levels: Number of distinct difficulty levels
            min_communities: Minimum communities in easiest graphs
            max_communities: Maximum communities in hardest graphs
            min_nodes: Minimum nodes in easiest graphs
            max_nodes: Maximum nodes in hardest graphs
            
        Returns:
            Dictionary mapping difficulty level to list of graphs
        """
        result = {}
        
        # Calculate graphs per level
        graphs_per_level = n_graphs // difficulty_levels
        remainder = n_graphs % difficulty_levels
        
        for level in range(difficulty_levels):
            level_count = graphs_per_level + (1 if level < remainder else 0)
            
            # Scale parameters based on difficulty
            difficulty = level / (difficulty_levels - 1)  # 0 to 1
            
            level_min_comm = min_communities + int(difficulty * (max_communities - min_communities) * 0.5)
            level_max_comm = min_communities + int(difficulty * (max_communities - min_communities))
            
            level_min_nodes = min_nodes + int(difficulty * (max_nodes - min_nodes) * 0.5)
            level_max_nodes = min_nodes + int(difficulty * (max_nodes - min_nodes))
            
            overlap = 1.0 + difficulty * 1.5  # 1.0 to 2.5
            heterogeneity = 0.3 + difficulty * 0.4  # 0.3 to 0.7
            
            # For harder levels, increase diversity of community selection
            if level < difficulty_levels // 2:
                method = "similar"
                similarity_bias = 0.5 - level * 0.5  # 0.5 to 0.0
            else:
                method = "diverse"
                similarity_bias = -0.5 * (level - difficulty_levels // 2) / (difficulty_levels // 2)  # 0.0 to -0.5
            
            # Generate graphs for this level
            level_graphs = []
            
            for i in range(level_count):
                # Sample number of communities
                n_communities = np.random.randint(level_min_comm, level_max_comm + 1)
                
                # Sample communities
                communities = self.universe.sample_community_subset(
                    size=n_communities,
                    method=method,
                    similarity_bias=similarity_bias
                )
                
                # Sample number of nodes
                n_nodes = np.random.randint(level_min_nodes, level_max_nodes + 1)
                
                # Generate graph
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=heterogeneity,
                    edge_noise=0.05 * difficulty,  # 0.0 to 0.05
                    min_component_size=0,
                    indirect_influence=0.1
                )
                
                level_graphs.append(graph)
            
            result[f"difficulty_{level+1}"] = level_graphs
        
        return result
    
    def sample_graphs_by_structural_property(
        self,
        n_graphs: int,
        property_type: str = "density",
        min_value: float = 0.01,
        max_value: float = 0.2,
        min_communities: int = 5,
        max_communities: int = 15,
        min_nodes: int = 100,
        max_nodes: int = 500
    ) -> Dict[float, GraphSample]:
        """
        Sample graphs with controlled structural properties.
        
        Args:
            n_graphs: Number of graphs to generate
            property_type: Property to control ("density", "clustering", "community_overlap")
            min_value: Minimum value of the property
            max_value: Maximum value of the property
            min_communities: Minimum communities per graph
            max_communities: Maximum communities per graph
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            
        Returns:
            Dictionary mapping property value to graph
        """
        result = {}
        
        # Create evenly spaced property values
        property_values = np.linspace(min_value, max_value, n_graphs)
        
        for i, value in enumerate(property_values):
            # Sample number of communities
            n_communities = np.random.randint(min_communities, max_communities + 1)
            
            # Sample communities
            communities = self.universe.sample_community_subset(
                size=n_communities,
                method="random"
            )
            
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            if property_type == "density":
                # Control edge density by scaling community edge probabilities
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=0.5,
                    edge_noise=0.0,
                    min_component_size=0,
                    indirect_influence=0.1,
                    # Control density through degree factors
                    # A larger scale factor increases overall edge density
                    scale_factor=value * 10.0  # Scale appropriately
                )
                
            elif property_type == "clustering":
                # Control clustering coefficient through edge pattern
                # This is harder to control directly, so adjust using heuristics
                # Higher concentration parameter leads to more homogeneous memberships
                # which generally increases clustering
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=0.3,
                    edge_noise=0.0,
                    min_component_size=0,
                    indirect_influence=0.1,
                    # Higher concentration increases clustering
                    membership_concentration=value * 20.0,
                )
                
            elif property_type == "community_overlap":
                # Control average community memberships per node
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=0.5,
                    edge_noise=0.0,
                    min_component_size=0,
                    indirect_influence=0.1,
                    # Directly control overlap
                    avg_memberships=value * (max_communities / 2.0),
                )
                
            else:
                raise ValueError(f"Unknown property type: {property_type}")
            
            result[value] = graph
        
        return result