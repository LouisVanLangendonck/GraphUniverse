"""
Graph family generation utilities for MMSB graphs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from mmsb.model import GraphUniverse, GraphSample

class GraphFamilyGenerator:
    """
    Generates families of graph instances sampled from a common universe.
    Each family consists of a set of graphs with similar characteristics.
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
        seed: Optional[int] = None
    ):
        """
        Initialize the graph family generator.
        
        Args:
            universe: Optional existing GraphUniverse to use. If None, creates a new one.
            K: Total number of communities in the universe (if creating new)
            feature_dim: Dimension of node features (if creating new)
            block_structure: Type of block structure ("assortative")
            overlap_structure: Type of community overlap structure ("modular")
            edge_density: Overall edge density (if creating new)
            homophily: Strength of within-community connections (if creating new)
            randomness_factor: Amount of random noise in edge probabilities (if creating new)
            intra_community_regime_similarity: How similar regimes within same community should be (0-1)
            inter_community_regime_similarity: How similar regimes between communities should be (0-1)
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
        """
        # Store parameters
        if universe is not None:
            self.universe = universe
        else:
            # Initialize the universe
            self.universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                block_structure=block_structure,
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
                regimes_per_community=regimes_per_community,
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                seed=seed
            )
            
            # Generate community co-membership matrix
            self.universe.community_co_membership = self.universe.generate_community_co_membership_matrix(
                overlap_density=0.1,
                structure=overlap_structure
            )

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
        Generate a family of graphs with similar characteristics.
        
        Args:
            n_graphs: Number of graphs to generate
            min_communities: Minimum communities per graph
            max_communities: Maximum communities per graph
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            degree_heterogeneity: Controls variability in node degrees
            edge_noise: Random noise in edge probabilities
            sampling_method: How to sample communities ("random", "similar", "diverse", "correlated")
            min_component_size: Minimum size for connected components
            feature_regime_balance: How evenly feature regimes are distributed
            seed: Random seed for reproducibility
            
        Returns:
            List of graph samples
        """
        if seed is not None:
            np.random.seed(seed)
        
        graphs = []
        for i in range(n_graphs):
            # Sample number of communities and nodes
            n_communities = np.random.randint(min_communities, max_communities + 1)
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Sample communities
            if sampling_method == "random":
                communities = self.universe.sample_community_subset(
                    size=n_communities,
                    method=sampling_method
                )
            else:
                # Use enhanced connected sampling for other methods
                communities = self.universe.sample_connected_community_subset(
                    size=n_communities,
                    method=sampling_method
                )
            
            # Generate graph
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                min_component_size=min_component_size,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                feature_regime_balance=feature_regime_balance,
                seed=seed + i if seed is not None else None
            )
            
            graphs.append(graph)
        
        return graphs 