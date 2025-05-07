"""
Mixed-Membership Stochastic Block Model (MMSB) with overlapping communities.

This module implements a generative framework for creating graph instances
sampled from subsets of a core "graph universe" defined by a master
stochastic block model.

The model supports:
- Degree correction
- Feature generation conditioned on community membership
- Systematic sampling of subgraphs from a larger universe

References:
- Karrer & Newman (2011). Stochastic blockmodels and community structure in networks.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from mmsb.feature_regimes import FeatureRegimeGenerator, NeighborhoodFeatureAnalyzer, FeatureRegimeLabelGenerator
import time

def sample_connected_community_subset(
    P: np.ndarray,
    size: int,
    method: str = "random",
    similarity_bias: float = 0.0,
    max_attempts: int = 10,
    existing_communities: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> List[int]:
    """
    Sample a subset of communities that are well-connected to each other.
    
    Args:
        P: Probability matrix (K × K)
        size: Number of communities to sample
        method: Sampling method ("random", "similar", "diverse", "correlated")
        similarity_bias: Controls bias towards similar communities (positive) or diverse (negative)
        max_attempts: Maximum number of attempts to find a connected subset
        existing_communities: Optional list of communities to condition on
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled community indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    K = P.shape[0]
    size = min(size, K)  # Ensure we don't sample more than available
    
    # Calculate average inter-community probability from the scaled probability matrix
    off_diag_mask = ~np.eye(K, dtype=bool)
    avg_inter_comm_prob = np.mean(P[off_diag_mask])
    
    # Track failures for debugging
    failures = {
        "no_strong_connections": [],
        "insufficient_connections": 0,
        "random_additions_needed": 0
    }
    
    # If we have existing communities, use them as seeds
    if existing_communities:
        seeds = existing_communities
        remaining_size = size - len(seeds)
    else:
        # Sample initial seed communities
        n_seeds = size // 2
        seeds = np.random.choice(K, size=n_seeds, replace=False).tolist()
        remaining_size = size - n_seeds
    
    # Initialize result with seeds
    result = set(seeds)
    
    # First round: find strongly connected partners for each seed
    for seed in seeds:
        # Find all communities with strong connections to this seed
        strong_connections = [
            j for j in range(K)
            if j not in result and P[seed, j] >= avg_inter_comm_prob
        ]
        
        if strong_connections:
            # Randomly select one strongly connected community
            partner = np.random.choice(strong_connections)
            result.add(partner)
            remaining_size -= 1
        else:
            failures["no_strong_connections"].append(seed)
    
    # Second round: if we still need more communities
    if remaining_size > 0:
        # Find communities with strong connections to any existing community
        potential_additions = []
        for j in range(K):
            if j not in result:
                # Check connection strength to all existing communities
                max_connection = max(P[j, i] for i in result)
                if max_connection >= avg_inter_comm_prob:
                    potential_additions.append(j)
        
        # If we found potential additions, randomly select from them
        if potential_additions:
            n_to_add = min(remaining_size, len(potential_additions))
            additions = np.random.choice(potential_additions, size=n_to_add, replace=False)
            result.update(additions)
            remaining_size -= n_to_add
        else:
            failures["insufficient_connections"] = remaining_size
    
    # Final round: if we still need more communities, add random ones
    if remaining_size > 0:
        remaining = set(range(K)) - result
        if remaining:
            final_additions = np.random.choice(list(remaining), size=min(remaining_size, len(remaining)), replace=False)
            result.update(final_additions)
            failures["random_additions_needed"] = len(final_additions)
    
    # Only print debug information if there were issues
    if failures["no_strong_connections"] or failures["insufficient_connections"] or failures["random_additions_needed"]:
        print("\nCommunity sampling issues:")
        if failures["no_strong_connections"]:
            print(f"  Seeds with no strong connections: {failures['no_strong_connections']}")
        if failures["insufficient_connections"]:
            print(f"  Missing connections for {failures['insufficient_connections']} communities")
        if failures["random_additions_needed"]:
            print(f"  Had to add {failures['random_additions_needed']} random communities")
        print(f"  Connection strength threshold (avg inter-comm prob): {avg_inter_comm_prob:.4f}")
        print(f"  Final community set size: {len(result)}")
    
    return list(result)

class GraphUniverse:
    """
    Represents a generative universe for graph instances sampled from a master stochastic block model.
    Each instance is a subgraph of the universe, with its own community structure and features.
    """
    
    def __init__(
        self,
        K: int,
        P: Optional[np.ndarray] = None,
        feature_dim: int = 0,
        intra_community_regime_similarity: float = 0.8,
        inter_community_regime_similarity: float = 0.2,
        block_structure: str = "assortative",
        edge_density: float = 0.1,
        homophily: float = 0.8,
        randomness_factor: float = 0.0,
        regimes_per_community: int = 2,
        seed: Optional[int] = None
    ):
        """
        Initialize the graph universe.
        
        Args:
            K: Total number of community types
            P: Optional edge probability matrix (K × K)
            feature_dim: Dimension of node features
            intra_community_regime_similarity: How similar regimes within same community should be (0-1)
            inter_community_regime_similarity: How similar regimes between communities should be (0-1)
            block_structure: Type of block structure ("assortative", "disassortative", "core-periphery", "hierarchical")
            edge_density: Overall edge density
            homophily: Strength of within-community connections
            randomness_factor: Amount of random noise in edge probabilities
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
        """
        self.K = K
        self.feature_dim = feature_dim
        self.regimes_per_community = regimes_per_community
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate or validate probability matrix
        if P is not None:
            if P.shape != (K, K):
                raise ValueError(f"Probability matrix must be {K}×{K}")
            self.P = P
        else:
            self.P = self._generate_probability_matrix(
                K, block_structure, edge_density, homophily, randomness_factor
            )
        
        # Initialize feature regime generator if features are enabled
        if feature_dim > 0:
            self.regime_generator = FeatureRegimeGenerator(
                universe_K=K,
                feature_dim=feature_dim,
                regimes_per_community=regimes_per_community,
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                feature_variance=0.1,
                seed=seed
            )
            self.regime_prototypes = self.regime_generator.regime_prototypes
        else:
            self.regime_generator = None
            self.regime_prototypes = None
        
        # Store parameters
        self.edge_density = edge_density
        self.homophily = homophily
        self.block_structure = block_structure
        self.randomness_factor = randomness_factor
        self.feature_variance = 0.1
        self.feature_similarity_matrix = None

    def _generate_probability_matrix(
        self, 
        K: int, 
        structure: str, 
        edge_density: float,
        homophily: float = 0.8,
        randomness_factor: float = 0.0
    ) -> np.ndarray:
        """
        Generate an assortative probability matrix based on edge density and homophily parameters.
        
        Args:
            K: Number of communities
            structure: Must be "assortative" (other structures not supported)
            edge_density: Overall edge density
            homophily: Strength of within-community connections (0=random, 1=maximum homophily)
            randomness_factor: Amount of random noise in edge probabilities
            
        Returns:
            K × K probability matrix
        """
        if structure != "assortative":
            raise ValueError("Only assortative structure is currently supported")
            
        P = np.zeros((K, K))
        
        # Calculate inter-community probability
        if homophily < 1.0:
            q = edge_density * K / ((K-1)*(1-homophily) + homophily*K)
            p = homophily * q * (K-1) / (1-homophily)
        else:
            # Maximum homophily - all edges within communities
            p = edge_density * K
            q = 0.0
        
        # Set probabilities WITHOUT capping - keep theoretical values
        np.fill_diagonal(P, p)
        P[~np.eye(K, dtype=bool)] = q
        
        # Add randomness if requested, but don't clip results
        if randomness_factor > 0:
            noise = np.random.uniform(-randomness_factor/2, randomness_factor/2, size=(K, K))
            P = P * (1 + noise)
            # Only clipping values at 0.0 (no negative values)
            P[P < 0.0] = 0.0

        return P
    
    def _generate_feature_similarity_matrix(
        self,
        K: int,
        mixing_strength: float,
        feature_structure: str = "distinct"
    ) -> np.ndarray:
        """
        Generate a matrix controlling feature similarity/mixing between communities.
        
        Args:
            K: Number of communities
            mixing_strength: Strength of feature mixing (0=no mixing, 1=strong mixing)
            feature_structure: Type of feature structure
            
        Returns:
            K × K feature similarity matrix
        """
        if feature_structure == "distinct":
            # Low similarity between communities
            similarity = np.eye(K)  # Identity matrix (no mixing by default)
            
            # Add some off-diagonal similarity proportional to mixing_strength
            if mixing_strength > 0:
                off_diag = np.random.uniform(0, mixing_strength, size=(K, K))
                np.fill_diagonal(off_diag, 1.0)  # Keep diagonal at 1.0
                similarity = off_diag
                
        elif feature_structure == "hierarchical":
            # Hierarchical similarity based on community relationships
            similarity = np.eye(K)
            
            # Determine hierarchy levels
            levels = int(np.log2(K)) + 1
            
            # For each pair of communities, calculate similarity based on hierarchy
            for i in range(K):
                bin_i = bin(i)[2:].zfill(levels)
                
                for j in range(K):
                    if i == j:
                        continue
                        
                    bin_j = bin(j)[2:].zfill(levels)
                    common_prefix = 0
                    
                    # Count common prefix length
                    for b_i, b_j in zip(bin_i, bin_j):
                        if b_i == b_j:
                            common_prefix += 1
                        else:
                            break
                    
                    # Similarity based on common prefix and mixing strength
                    similarity[i, j] = (common_prefix / levels) * mixing_strength
                    
        elif feature_structure == "correlated":
            # Correlated structure where nearby communities have similar features
            similarity = np.eye(K)
            
            # Create a distance-based correlation
            for i in range(K):
                for j in range(K):
                    if i != j:
                        # Distance in community index space (circular)
                        dist = min(abs(i - j), K - abs(i - j))
                        # Convert to similarity: closer = more similar
                        similarity[i, j] = mixing_strength * np.exp(-dist / (K / 4))
                        
        elif feature_structure == "random":
            # Random similarity structure
            similarity = np.random.uniform(0, mixing_strength, size=(K, K))
            np.fill_diagonal(similarity, 1.0)  # Self-similarity is always 1.0
            
        else:
            # Default: identity matrix (no mixing)
            similarity = np.eye(K)
            
        return similarity
    
    def sample_community_subset(
        self, 
        size: int, 
        method: str = "random",
        similarity_bias: float = 0.0,
        existing_communities: Optional[List[int]] = None
    ) -> List[int]:
        """
        Sample a subset of communities from the universe.
        
        Args:
            size: Number of communities to sample
            method: Sampling method ("random", "similar", "diverse", "correlated")
            similarity_bias: Controls bias towards similar communities (positive) or diverse (negative)
            existing_communities: For transfer learning, optionally condition on existing communities
            
        Returns:
            List of sampled community indices
        """
        size = min(size, self.K)  # Ensure we don't sample more than available
        
        if method == "random":
            # Simple random sampling
            return np.random.choice(self.K, size=size, replace=False).tolist()
            
        elif method == "similar" or method == "diverse":
            # Sample based on similarity in the probability matrix
            if existing_communities is None:
                # Start with a random community
                communities = [np.random.choice(self.K)]
                remaining = set(range(self.K)) - set(communities)
            else:
                # Start with specified communities
                communities = list(existing_communities)
                remaining = set(range(self.K)) - set(communities)
            
            # Compute similarities based on probability patterns
            while len(communities) < size and remaining:
                # Compute average similarity to existing communities
                similarities = np.zeros(self.K)
                for k in remaining:
                    for c in communities:
                        row_sim = np.corrcoef(self.P[k, :], self.P[c, :])[0, 1]
                        col_sim = np.corrcoef(self.P[:, k], self.P[:, c])[0, 1]
                        similarities[k] += (row_sim + col_sim) / 2
                    
                    similarities[k] /= len(communities)
                
                # Zero out communities already selected
                similarities[communities] = -np.inf
                
                # For diverse sampling, invert similarities
                if method == "diverse":
                    similarities = -similarities
                
                # Apply similarity bias
                if similarity_bias != 0:
                    probs = np.exp(similarity_bias * similarities)
                    probs[communities] = 0
                    probs = probs / probs.sum()
                    next_community = np.random.choice(self.K, p=probs)
                else:
                    # Just pick the most similar/diverse
                    next_community = np.argmax(similarities)
                
                # Only remove if the community is in the remaining set
                if next_community in remaining:
                    communities.append(next_community)
                    remaining.remove(next_community)
                else:
                    # If not in remaining, try to find another community
                    remaining_list = list(remaining)
                    if remaining_list:
                        next_community = np.random.choice(remaining_list)
                        communities.append(next_community)
                        remaining.remove(next_community)
        
        elif method == "correlated":
            # Use the co-membership matrix to generate correlated samples
            if existing_communities is None:
                first = np.random.choice(self.K)
                communities = [first]
            else:
                communities = list(existing_communities)
                
            while len(communities) < size:
                # Compute sampling probabilities based on co-membership
                co_probs = np.zeros(self.K)
                for c in communities:
                    co_probs += self.community_co_membership[c, :]
                
                # Zero out already selected communities
                co_probs[communities] = 0
                
                # Normalize to get probabilities
                if co_probs.sum() > 0:
                    co_probs = co_probs / co_probs.sum()
                    next_community = np.random.choice(self.K, p=co_probs)
                else:
                    # If all co-probabilities are zero, sample randomly from remaining
                    remaining = list(set(range(self.K)) - set(communities))
                    next_community = np.random.choice(remaining)
                
                communities.append(next_community)
                
            return communities
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
            
    # def generate_community_co_membership_matrix(
    #     self, 
    #     overlap_density: float = 0.1,
    #     structure: str = "modular"
    # ) -> np.ndarray:
    #     """
    #     Generate a community co-membership matrix to model overlapping communities.
        
    #     Args:
    #         overlap_density: Overall density of overlaps between communities
    #         structure: Structure of overlaps ("modular", "hierarchical", "hub-spoke")
            
    #     Returns:
    #         K × K co-membership probability matrix
    #     """
    #     K = self.K
    #     co_membership = np.eye(K)  # Diagonal is always 1 (self-membership)
        
    #     if structure == "modular":
    #         # Create modules of overlapping communities
    #         module_size = max(2, K // 5)  # Module size
    #         for i in range(0, K, module_size):
    #             module_end = min(i + module_size, K)
    #             module_indices = np.arange(i, module_end)
                
    #             # Communities within same module have higher overlap probability
    #             for j in module_indices:
    #                 for k in module_indices:
    #                     if j != k:
    #                         co_membership[j, k] = overlap_density
                            
    #     elif structure == "hierarchical":
    #         # Hierarchical structure with stronger overlaps between related communities
    #         levels = int(np.log2(K)) + 1
    #         for i in range(K):
    #             for j in range(i+1, K):
    #                 # Compute overlap based on hierarchical distance
    #                 bin_i, bin_j = bin(i)[2:].zfill(levels), bin(j)[2:].zfill(levels)
    #                 common_prefix = 0
    #                 for b_i, b_j in zip(bin_i, bin_j):
    #                     if b_i == b_j:
    #                         common_prefix += 1
    #                     else:
    #                         break
                    
    #                 # Probability decreases with hierarchical distance
    #                 hier_distance = levels - common_prefix
    #                 overlap_prob = overlap_density * (0.5 ** (hier_distance - 1))
    #                 co_membership[i, j] = co_membership[j, i] = overlap_prob
                    
    #     elif structure == "hub-spoke":
    #         # Some communities are hubs that overlap with many others
    #         num_hubs = max(1, K // 10)
    #         hub_indices = np.random.choice(K, size=num_hubs, replace=False)
            
    #         # Hub-spoke connections
    #         for hub in hub_indices:
    #             for i in range(K):
    #                 if i != hub:
    #                     co_membership[hub, i] = co_membership[i, hub] = overlap_density
                        
    #         # Hub-hub connections (stronger)
    #         for i in hub_indices:
    #             for j in hub_indices:
    #                 if i != j:
    #                     co_membership[i, j] = overlap_density * 2
                        
    #     else:
    #         # Random uniform co-membership
    #         for i in range(K):
    #             for j in range(i+1, K):
    #                 p = np.random.uniform(0, overlap_density)
    #                 co_membership[i, j] = co_membership[j, i] = p
                    
    #     return co_membership

    def sample_connected_community_subset(
        self,
        size: int,
        method: str = "random",
        similarity_bias: float = 0.0,
        max_attempts: int = 10,
        existing_communities: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Sample a subset of communities that are well-connected to each other.
        
        Args:
            size: Number of communities to sample
            method: Sampling method ("random", "similar", "diverse", "correlated")
            similarity_bias: Controls bias towards similar communities (positive) or diverse (negative)
            max_attempts: Maximum number of attempts to find a connected subset
            existing_communities: Optional list of communities to condition on
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled community indices
        """
        if seed is not None:
            np.random.seed(seed)
        
        K = self.K
        size = min(size, K)  # Ensure we don't sample more than available
        
        # Calculate average inter-community probability from probability matrix
        off_diag_mask = ~np.eye(K, dtype=bool)
        avg_inter_comm_prob = np.mean(self.P[off_diag_mask])
        
        # Track failures for debugging
        failures = {
            "no_strong_connections": [],
            "insufficient_connections": 0,
            "random_additions_needed": 0
        }
        
        # If we have existing communities, use them as seeds
        if existing_communities:
            seeds = existing_communities
            remaining_size = size - len(seeds)
        else:
            # Sample initial seed communities
            n_seeds = size // 2
            seeds = np.random.choice(K, size=n_seeds, replace=False).tolist()
            remaining_size = size - n_seeds
        
        # Initialize result with seeds
        result = set(seeds)
        
        # First round: find strongly connected partners for each seed
        for seed in seeds:
            # Find all communities with strong connections to this seed
            strong_connections = [
                j for j in range(K)
                if j not in result and self.P[seed, j] >= avg_inter_comm_prob
            ]
            
            if strong_connections:
                # Randomly select one strongly connected community
                partner = np.random.choice(strong_connections)
                result.add(partner)
                remaining_size -= 1
            else:
                failures["no_strong_connections"].append(seed)
        
        # Second round: if we still need more communities
        if remaining_size > 0:
            # Find communities with strong connections to any existing community
            potential_additions = []
            for j in range(K):
                if j not in result:
                    # Check connection strength to all existing communities
                    max_connection = max(self.P[j, i] for i in result)
                    if max_connection >= avg_inter_comm_prob:
                        potential_additions.append(j)
            
            # If we found potential additions, randomly select from them
            if potential_additions:
                n_to_add = min(remaining_size, len(potential_additions))
                additions = np.random.choice(potential_additions, size=n_to_add, replace=False)
                result.update(additions)
                remaining_size -= n_to_add
            else:
                failures["insufficient_connections"] = remaining_size
        
        # Final round: if we still need more communities, add random ones
        if remaining_size > 0:
            remaining = set(range(K)) - result
            if remaining:
                final_additions = np.random.choice(list(remaining), size=min(remaining_size, len(remaining)), replace=False)
                result.update(final_additions)
                failures["random_additions_needed"] = len(final_additions)
        
        # Only print debug information if there were issues
        if failures["no_strong_connections"] or failures["insufficient_connections"] or failures["random_additions_needed"]:
            print("\nCommunity sampling issues:")
            if failures["no_strong_connections"]:
                print(f"  Seeds with no strong connections: {failures['no_strong_connections']}")
            if failures["insufficient_connections"]:
                print(f"  Missing connections for {failures['insufficient_connections']} communities")
            if failures["random_additions_needed"]:
                print(f"  Had to add {failures['random_additions_needed']} random communities")
            print(f"  Connection strength threshold (avg inter-comm prob): {avg_inter_comm_prob:.4f}")
            print(f"  Final community set size: {len(result)}")
        
        return list(result)

class GraphSample:
    """
    Represents a single graph instance sampled from the GraphUniverse.
    
    This modified version implements:
    1. Filtering of components smaller than min_component_size
    2. All graph properties reflect the filtered state
    3. Feature regime generation and analysis
    4. Neighborhood feature analysis
    5. Optional configuration model-like edge generation with configurable degree distributions
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        communities: List[int],
        n_nodes: int,
        min_component_size: int = 0,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        feature_regime_balance: float = 0.5,
        target_homophily: Optional[float] = None,
        target_density: Optional[float] = None,
        use_configuration_model: bool = False,
        degree_distribution: str = "power_law",
        power_law_exponent: Optional[float] = None,
        target_avg_degree: Optional[float] = None,
        triangle_enhancement: float = 0.0,  # Added back with default value
        seed: Optional[int] = None
    ):
        """
        Initialize and generate a graph sample.
        
        Args:
            universe: The graph universe to sample from
            communities: List of community indices to include
            n_nodes: Number of nodes in the graph
            min_component_size: Minimum size for connected components
            degree_heterogeneity: Controls degree variability (0=homogeneous, 1=highly skewed)
            edge_noise: Random noise in edge probabilities
            feature_regime_balance: How evenly feature regimes are distributed
            target_homophily: Target homophily level (if None, use universe value)
            target_density: Target edge density (if None, use universe value)
            use_configuration_model: Whether to use configuration model-like edge generation
            degree_distribution: Type of degree distribution ("power_law", "log_normal", "uniform")
            power_law_exponent: Exponent for power law distribution (lower = more skewed)
            target_avg_degree: Target average degree (if None, calculated from density)
            triangle_enhancement: How much to enhance triangle formation (0-1)
            seed: Random seed for reproducibility
        """
        # Dictionary to store timing information
        self.timing_info = {}
        total_start = time.time()

        self.universe = universe
        self.communities = sorted(communities)
        self.original_n_nodes = n_nodes
        self.min_component_size = min_component_size
        
        # Store feature generation parameters
        self.feature_regime_balance = feature_regime_balance
        
        # Store target parameters (use universe values if not specified)
        self.target_homophily = target_homophily if target_homophily is not None else universe.homophily
        self.target_density = target_density if target_density is not None else universe.edge_density
        
        # Store configuration model parameters
        self.use_configuration_model = use_configuration_model
        self.degree_distribution = degree_distribution
        self.power_law_exponent = power_law_exponent
        self.target_avg_degree = target_avg_degree
        self.triangle_enhancement = triangle_enhancement  # Store triangle enhancement parameter
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Time: Extract and scale probability matrix
        start = time.time()
        # Extract the submatrix of the probability matrix for these communities
        K_sub = len(communities)
        self.P_sub = np.zeros((K_sub, K_sub))
        for i, ci in enumerate(communities):
            for j, cj in enumerate(communities):
                self.P_sub[i, j] = universe.P[ci, cj]

        # Scale the probability matrix
        self.P_sub = self._scale_probability_matrix(
            self.P_sub, 
            self.target_density,
            self.target_homophily
        )
        self.timing_info['probability_matrix'] = time.time() - start
        
        # Time: Generate memberships
        start = time.time()
        self.community_labels = self._generate_memberships(n_nodes, K_sub)  # Now returns 1D array
        self.timing_info['memberships'] = time.time() - start
        
        # Time: Generate degree factors
        start = time.time()
        if self.use_configuration_model:
            self.degree_factors = self._generate_degree_factors_configuration(
                n_nodes,
                None,  # degree_heterogeneity not used in configuration model
                self.degree_distribution,
                self.power_law_exponent,
                self.target_avg_degree
            )
        else:
            self.degree_factors = self._generate_degree_factors(n_nodes, degree_heterogeneity)
        self.timing_info['degree_factors'] = time.time() - start
        
        # Time: Generate edges
        start = time.time()
        if self.use_configuration_model:
            # Use default values for deviation constraints if not set
            max_mean_community_deviation = getattr(self, 'max_mean_community_deviation', 0.1)
            max_max_community_deviation = getattr(self, 'max_max_community_deviation', 0.2)
            max_parameter_search_attempts = getattr(self, 'max_parameter_search_attempts', 10)
            parameter_search_range = getattr(self, 'parameter_search_range', 0.2)
            min_edge_density = getattr(self, 'min_edge_density', 0.005)
            max_retries = getattr(self, 'max_retries', 5)
            
            self.adjacency = self._generate_edges_configuration(
                self.community_labels,
                self.P_sub,
                self.degree_factors,
                edge_noise,
                min_edge_density=min_edge_density,
                max_retries=max_retries,
                max_mean_community_deviation=max_mean_community_deviation,
                max_max_community_deviation=max_max_community_deviation,
                max_parameter_search_attempts=max_parameter_search_attempts,
                parameter_search_range=parameter_search_range
            )
        else:
            self.adjacency = self._generate_edges(
                self.community_labels,
                self.P_sub,
                self.degree_factors,
                edge_noise
            )
        self.timing_info['edge_generation'] = time.time() - start
        
        # Time: Component filtering
        start = time.time()
        # Create initial NetworkX graph
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)
        
        # Find connected components
        components = list(nx.connected_components(temp_graph))
        components.sort(key=len, reverse=True)
        
        # Track deleted components
        self.deleted_components = []
        self.deleted_node_types = {}
        
        # Filter components based on size
        if min_component_size > 0:
            kept_components = []
            for comp in components:
                if len(comp) >= min_component_size:
                    kept_components.append(comp)
                else:
                    self.deleted_components.append(comp)
                    # Track community distribution of deleted nodes
                    for node in comp:
                        primary_comm = self.communities[self.community_labels[node]]  # Use community labels directly
                        if primary_comm not in self.deleted_node_types:
                            self.deleted_node_types[primary_comm] = 0
                        self.deleted_node_types[primary_comm] += 1
        else:
            kept_components = components
        self.timing_info['component_filtering'] = time.time() - start
        
        # Time: Graph reconstruction
        start = time.time()
        if kept_components:
            # Create union of all kept components
            kept_nodes = sorted(list(set().union(*kept_components)))
            
            # Create mapping from old to new indices
            self.node_map = {old: new for new, old in enumerate(kept_nodes)}
            self.reverse_node_map = {new: old for old, new in self.node_map.items()}
            
            # Create new graph with remapped nodes
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(len(kept_nodes)))
            
            # Add edges with remapped indices
            for comp in kept_components:
                for u, v in temp_graph.subgraph(comp).edges():
                    self.graph.add_edge(self.node_map[u], self.node_map[v])
            
            # Update node count
            self.n_nodes = self.graph.number_of_nodes()
            
            # Update community labels and degree factors with new indices
            self.community_labels = self.community_labels[kept_nodes]
            self.degree_factors = self.degree_factors[kept_nodes]
            
            # Update adjacency matrix
            self.adjacency = nx.adjacency_matrix(self.graph)
        else:
            # If no components meet the size threshold, keep an empty graph
            self.graph = nx.Graph()
            self.n_nodes = 0
            self.community_labels = np.array([], dtype=int)
            self.degree_factors = np.zeros(0)
            self.adjacency = sp.csr_matrix((0, 0))
            self.features = None if universe.feature_dim > 0 else None
            self.node_map = {}
            self.reverse_node_map = {}
            self.node_labels = np.zeros(0, dtype=int)
            self.node_regimes = None
            self.neighborhood_analyzer = None
            self.label_generator = None
        self.timing_info['graph_reconstruction'] = time.time() - start
        
        # Time: Feature generation
        start = time.time()
        if universe.feature_dim > 0:
            # Assign nodes to feature regimes using community labels directly
            self.node_regimes = universe.regime_generator.assign_node_regimes(
                self.community_labels,
                regime_balance=self.feature_regime_balance
            )
            
            # Generate features based on regimes
            self.features = universe.regime_generator.generate_node_features(
                self.node_regimes
            )
            
            # Initialize these as None - they will be computed on demand
            self.neighborhood_analyzer = None
            self.label_generator = None
            self.node_labels = None
        else:
            self.features = None
            self.node_regimes = None
            self.neighborhood_analyzer = None
            self.label_generator = None
            self.node_labels = None
        self.timing_info['feature_generation'] = time.time() - start
        
        # Time: Node attributes
        start = time.time()
        self._add_node_attributes()
        self.timing_info['node_attributes'] = time.time() - start
        
        # Store total time
        self.timing_info['total'] = time.time() - total_start

    def _add_node_attributes(self):
        """Add node attributes to the graph."""
        for i, node in enumerate(self.graph.nodes()):
            # Get community label directly
            community_idx = self.community_labels[i]
            primary_comm = self.communities[community_idx]
            
            # Create node attributes dictionary
            node_attrs = {
                "community": int(primary_comm),  # Store the actual community ID
                "degree_factor": float(self.degree_factors[i])
            }
            
            # Add features if available
            if self.features is not None:
                node_attrs["features"] = self.features[i].tolist()
                
            # Add regime information if available
            if self.node_regimes is not None:
                node_attrs["feature_regime"] = int(self.node_regimes[i])
                
            # Update node attributes
            nx.set_node_attributes(self.graph, {node: node_attrs})
    
    def extract_parameters(self) -> Dict[str, float]:
        """
        Extract key parameters from this graph sample.
        
        Returns:
            Dictionary of graph parameters
        """
        from utils.parameter_analysis import analyze_graph_parameters
        
        # Analyze parameters
        params = analyze_graph_parameters(
            self.graph,
            self.community_labels,  # Pass community labels directly
            self.communities
        )
        
        return params

    def _generate_memberships(self, n_nodes: int, K_sub: int) -> np.ndarray:
        """
        Generate community assignments for nodes.
        Each node belongs to exactly one community.
        
        Args:
            n_nodes: Number of nodes
            K_sub: Number of communities in this subgraph
            
        Returns:
            Array of community labels (indices)
        """
        # Directly assign each node to a random community
        return np.random.choice(K_sub, size=n_nodes)

    def _generate_degree_factors(self, n_nodes: int, heterogeneity: float) -> np.ndarray:
        """
        Generate degree correction factors for nodes.
        
        Args:
            n_nodes: Number of nodes
            heterogeneity: Controls degree variability (0=homogeneous, 1=highly skewed)
            
        Returns:
            Array of degree correction factors
        """
        if heterogeneity == 0:
            # Homogeneous degrees
            return np.ones(n_nodes)
        
        # Generate factors from a power-law distribution
        # Interpolate between homogeneous (exponent→∞) and heterogeneous (exponent→2)
        exponent = 2 + 8 * (1 - heterogeneity)
        
        # Sample from power law
        factors = np.random.pareto(exponent, size=n_nodes) + 1
        
        # Normalize to keep expected edge count unchanged
        factors = factors / factors.mean()
        
        return factors

    def _generate_edges(
        self, 
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        min_edge_density: float = 0.005,
        max_retries: int = 5
    ) -> sp.spmatrix:
        """
        Generate edges with minimum density guarantee.
        Uses vectorized operations for faster edge generation with community labels.
        
        Args:
            community_labels: Node community assignments (indices)
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors
            noise: Edge noise level
            min_edge_density: Minimum acceptable edge density
            max_retries: Maximum number of retries if graph is too sparse
            
        Returns:
            Sparse adjacency matrix
        """
        n_nodes = len(community_labels)
        
        for attempt in range(max_retries):
            # Create node pairs using meshgrid
            i_nodes, j_nodes = np.triu_indices(n_nodes, k=1)
            
            # Get community pairs for all node pairs at once
            comm_i = community_labels[i_nodes]
            comm_j = community_labels[j_nodes]
            
            # Get base probabilities from P matrix
            edge_probs = P_sub[comm_i, comm_j]
            
            # Apply degree correction
            edge_probs *= degree_factors[i_nodes] * degree_factors[j_nodes]
            
            # Add noise if specified
            if noise > 0:
                edge_probs *= (1 + np.random.uniform(-noise, noise, size=len(edge_probs)))
                edge_probs = np.clip(edge_probs, 0, 1)
            
            # Sample edges
            edges = np.random.random(len(edge_probs)) < edge_probs
            
            # Get the edges that were sampled
            rows = i_nodes[edges]
            cols = j_nodes[edges]
            
            # Create data for both directions (undirected graph)
            all_rows = np.concatenate([rows, cols])
            all_cols = np.concatenate([cols, rows])
            all_data = np.ones(len(all_rows))
            
            # Create sparse adjacency matrix
            adj = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
            
            # Calculate actual edge density
            actual_density = len(all_rows) / (n_nodes * (n_nodes - 1))
            
            if actual_density >= min_edge_density:
                return adj
            
            # If density is too low, increase edge probabilities
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Graph too sparse (density={actual_density:.4f}). Retrying with adjusted probabilities...")
                P_sub = P_sub * 2  # Double the connection probabilities
                noise = noise * 0.5  # Reduce noise to maintain signal
            else:
                print(f"Warning: Could not achieve minimum edge density after {max_retries} attempts.")
                print(f"Final density: {actual_density:.4f}")
                return adj
        
        return adj

    def _scale_probability_matrix(
        self, 
        P_sub: np.ndarray, 
        target_density: Optional[float] = None, 
        target_homophily: Optional[float] = None
    ) -> np.ndarray:
        """
        Scale a probability matrix to achieve target density and homophily
        while preserving relative probabilities within communities.
        """
        n = P_sub.shape[0]
        P_scaled = P_sub.copy()
        
        # Use instance target values if not specified
        target_density = target_density if target_density is not None else self.target_density
        target_homophily = target_homophily if target_homophily is not None else self.target_homophily
        
        # Create masks for diagonal and off-diagonal elements
        diagonal_mask = np.eye(n, dtype=bool)
        off_diagonal_mask = ~diagonal_mask
        
        # Get current values - no clipping yet
        diagonal_elements = P_sub[diagonal_mask]
        off_diagonal_elements = P_sub[off_diagonal_mask]
        
        # Calculate current sums
        diagonal_sum = np.sum(diagonal_elements)
        off_diagonal_sum = np.sum(off_diagonal_elements)
        total_sum = diagonal_sum + off_diagonal_sum
        
        # Calculate target sums
        target_total_sum = target_density * n * n  # Total probability mass
        target_diagonal_sum = target_homophily * target_total_sum
        target_off_diagonal_sum = target_total_sum - target_diagonal_sum
        
        # Calculate scaling factors
        diagonal_scale = 1.0
        off_diagonal_scale = 1.0
        
        if diagonal_sum > 0:
            diagonal_scale = target_diagonal_sum / diagonal_sum
        
        if off_diagonal_sum > 0:
            off_diagonal_scale = target_off_diagonal_sum / off_diagonal_sum
        
        # Apply scaling
        P_scaled[diagonal_mask] *= diagonal_scale
        P_scaled[off_diagonal_mask] *= off_diagonal_scale
        
        # Handle special cases where there are no diagonal or off-diagonal elements
        if diagonal_sum == 0 and target_diagonal_sum > 0:
            # No existing diagonal elements, but we need some
            P_scaled[diagonal_mask] = target_diagonal_sum / n
        
        if off_diagonal_sum == 0 and target_off_diagonal_sum > 0:
            # No existing off-diagonal elements, but we need some
            P_scaled[off_diagonal_mask] = target_off_diagonal_sum / (n * n - n)
        
        # NOW ensure all probabilities are in [0, 1] for actual graph generation
        P_scaled = np.clip(P_scaled, 0, 1)
        
        # Recalculate actual values after clipping
        actual_diagonal_sum = np.sum(P_scaled[diagonal_mask])
        actual_total_sum = np.sum(P_scaled)
        
        # If clipping significantly affected our targets, do one final density adjustment
        actual_density = actual_total_sum / (n * n)
        if abs(actual_density - target_density) > 1e-3:
            density_correction = target_density / actual_density
            P_scaled = P_scaled * density_correction
            P_scaled = np.clip(P_scaled, 0, 1)  # Clip again after final adjustment
        
        return P_scaled

    def _generate_features(
        self,
        memberships: np.ndarray,
        prototypes: np.ndarray,
        feature_signal: float,
    ) -> np.ndarray:
        """
        Generate node features based on community memberships.
        
        Args:
            memberships: Node-community membership vectors
            prototypes: Feature prototypes for the universe communities
            feature_signal: How strongly features correlate with community membership
            
        Returns:
            Node feature matrix
        """
        n_nodes = memberships.shape[0]
        feature_dim = prototypes.shape[1]
        
        # Get prototypes for selected communities
        community_prototypes = prototypes[self.communities]
        
        # Generate random noise features
        noise = np.random.normal(0, 1, size=(n_nodes, feature_dim))
        # Normalize noise
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        
        # Generate community-based features
        community_features = memberships @ community_prototypes
        # Normalize community features
        community_features = community_features / np.linalg.norm(community_features, axis=1, keepdims=True)
        
        # Combine signal and noise based on feature_signal parameter
        # Square feature_signal to make the effect more pronounced at low values
        signal_weight = np.sqrt(feature_signal)  # This makes the transition more gradual
        features = signal_weight * community_features + (1 - signal_weight) * noise
        
        # Normalize final features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / norms
        
        return features

    def _create_networkx_graph(self) -> nx.Graph:
        """
        Create a NetworkX graph representation.
        
        Returns:
            NetworkX graph with node attributes
        """
        # Convert adjacency matrix to NetworkX graph
        G = nx.from_scipy_sparse_array(self.adjacency)
        
        # Add node attributes
        for i in range(self.n_nodes):
            node_attrs = {
                "community": int(self.community_labels[i]),  # Store the actual community ID
                "degree_factor": float(self.degree_factors[i])
            }
            
            # Add features if available
            if self.features is not None:
                node_attrs["features"] = self.features[i].tolist()
                
            # Update node attributes
            nx.set_node_attributes(G, {i: node_attrs})
            
        return G

    def generate_feature_regimes(
        self,
        n_regimes: int,
        regime_strength: Optional[float] = None,
        regime_overlap: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate feature regimes for the graph.
        
        Args:
            n_regimes: Number of feature regimes to generate
            regime_strength: How strongly features correlate with regime membership (0=random, 1=perfect)
            regime_overlap: How much overlap between regimes (0=no overlap, 1=complete overlap)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - regime_assignments: Node-regime membership matrix
            - regime_features: Feature prototypes for each regime
            - regime_correlation: Correlation matrix between regimes
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Use instance parameters if not specified
        regime_strength = regime_strength if regime_strength is not None else self.feature_regime_strength
        regime_overlap = regime_overlap if regime_overlap is not None else self.feature_regime_overlap
        
        # Generate regime prototypes
        feature_dim = self.features.shape[1]
        regime_features = np.random.normal(0, 1, size=(n_regimes, feature_dim))
        regime_features = regime_features / np.linalg.norm(regime_features, axis=1, keepdims=True)
        
        # Generate regime assignments
        regime_assignments = np.random.dirichlet(
            np.ones(n_regimes) * (1 - regime_overlap) + regime_overlap,
            size=self.n_nodes
        )
        
        # Generate regime correlation matrix
        regime_correlation = np.eye(n_regimes) * (1 - regime_overlap) + regime_overlap
        
        return {
            "regime_assignments": regime_assignments,
            "regime_features": regime_features,
            "regime_correlation": regime_correlation
        }

    def analyze_neighborhood_features(
        self,
        hop_distance: int = 1,
        feature_aggregation: str = "mean",
        include_self: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Analyze feature regimes in k-hop neighborhoods.
        
        Args:
            hop_distance: Number of hops to consider
            feature_aggregation: How to aggregate features ("mean", "max", "min")
            include_self: Whether to include the node itself in the analysis
            
        Returns:
            Dictionary containing:
            - frequency_vectors: Matrix of regime frequencies in neighborhoods
            - feature_vectors: Matrix of aggregated feature vectors
        """
        if self.neighborhood_analyzer is None:
            raise ValueError("Feature regimes not enabled for this graph")
            
        # Get frequency vectors
        frequency_vectors = self.neighborhood_analyzer.get_all_frequency_vectors(hop_distance)
        
        # Get feature vectors
        feature_vectors = np.zeros((self.n_nodes, self.universe.feature_dim))
        
        for node in range(self.n_nodes):
            # Get k-hop neighbors
            neighbors = self.neighborhood_analyzer.neighborhoods[hop_distance][node]
            
            if include_self:
                neighbors = [node] + neighbors
                
            if not neighbors:
                continue
                
            # Get features for neighbors
            neighbor_features = self.features[neighbors]
            
            # Aggregate based on method
            if feature_aggregation == "mean":
                feature_vectors[node] = np.mean(neighbor_features, axis=0)
            elif feature_aggregation == "max":
                feature_vectors[node] = np.max(neighbor_features, axis=0)
            elif feature_aggregation == "min":
                feature_vectors[node] = np.min(neighbor_features, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {feature_aggregation}")
        
        return {
            "frequency_vectors": frequency_vectors,
            "feature_vectors": feature_vectors
        }
    
    def generate_balanced_labels(
        self,
        n_classes: int,
        balance_ratio: float = 0.5,
        feature_influence: float = 0.7,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate balanced node labels based on neighborhood feature regimes.
        
        Args:
            n_classes: Number of label classes to generate
            balance_ratio: Target balance ratio between classes
            feature_influence: How much to weight feature information vs. topology
            seed: Random seed
            
        Returns:
            Array of node labels
        """
        if self.neighborhood_analyzer is None:
            raise ValueError("Feature regimes not enabled for this graph")
            
        # Get frequency vectors
        frequency_vectors = self.neighborhood_analyzer.get_all_frequency_vectors(1)
        
        # Create label generator
        label_generator = FeatureRegimeLabelGenerator(
            frequency_vectors=frequency_vectors,
            n_labels=n_classes,
            balance_tolerance=1.0 - balance_ratio,
            seed=seed
        )
        
        # Get labels
        labels = label_generator.get_node_labels()
        
        # Update node attributes
        for i, node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]["label"] = int(labels[i])
        
        return labels

    def compute_neighborhood_features(self, max_hops: int = 3) -> None:
        """
        Compute neighborhood features on demand.
        This is separated from initialization to make graph generation faster.
        
        Args:
            max_hops: Maximum number of hops to analyze
        """
        if self.universe.feature_dim > 0 and self.neighborhood_analyzer is None:
            start = time.time()
            self.neighborhood_analyzer = NeighborhoodFeatureAnalyzer(
                graph=self.graph,
                node_regimes=self.node_regimes,
                total_regimes=len(self.communities) * self.universe.regimes_per_community,
                max_hops=max_hops
            )
            if hasattr(self, 'timing_info'):
                self.timing_info['neighborhood_analysis'] = time.time() - start

    def compute_node_labels(self, balance_tolerance: float = 0.1) -> None:
        """
        Compute node labels on demand.
        This is separated from initialization to make graph generation faster.
        
        Args:
            balance_tolerance: Maximum allowed class imbalance (0-1)
        """
        if self.universe.feature_dim > 0 and self.label_generator is None:
            # Ensure neighborhood features are computed
            if self.neighborhood_analyzer is None:
                self.compute_neighborhood_features()
            
            start = time.time()
            # Generate balanced labels based on neighborhood features
            self.label_generator = FeatureRegimeLabelGenerator(
                frequency_vectors=self.neighborhood_analyzer.get_all_frequency_vectors(1),
                n_labels=len(self.communities),
                balance_tolerance=balance_tolerance,
                seed=None  # Use current random state
            )
            
            # Get node labels
            self.node_labels = self.label_generator.get_node_labels()
            if hasattr(self, 'timing_info'):
                self.timing_info['label_generation'] = time.time() - start

    def _generate_degree_factors_configuration(
        self,
        n_nodes: int,
        heterogeneity: float,  # Keep parameter for compatibility but don't use it
        distribution: str,
        power_law_exponent: float,
        target_avg_degree: Optional[float]
    ) -> np.ndarray:
        """
        Generate degree factors for configuration model with specified distribution.
        
        Args:
            n_nodes: Number of nodes
            heterogeneity: Not used in configuration model (kept for compatibility)
            distribution: Type of degree distribution ("power_law", "log_normal", "uniform")
            power_law_exponent: Exponent for power law distribution
            target_avg_degree: Target average degree (if None, calculated from density)
            
        Returns:
            Array of degree factors
        """
        if target_avg_degree is None:
            # Calculate from target density
            target_avg_degree = self.target_density * (n_nodes - 1)
        
        if distribution == "power_law":
            # Generate from power law distribution
            # Use power_law_exponent directly without heterogeneity adjustment
            factors = np.random.pareto(power_law_exponent - 1, size=n_nodes) + 1
            
        elif distribution == "log_normal":
            # Generate from log-normal distribution
            # Use fixed sigma for log-normal
            sigma = 1.0
            factors = np.random.lognormal(mean=0, sigma=sigma, size=n_nodes)
            
        elif distribution == "uniform":
            # Generate from uniform distribution
            # Use fixed range for uniform
            min_factor = 0.5
            max_factor = 1.5
            factors = np.random.uniform(min_factor, max_factor, size=n_nodes)
            
        else:
            raise ValueError(f"Unknown degree distribution: {distribution}")
        
        # Normalize to match target average degree
        factors = factors / factors.mean() * target_avg_degree
        
        # Ensure minimum degree of 1
        factors = np.maximum(factors, 1)
        
        return factors

    def _generate_edges_configuration(
        self,
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        min_edge_density: float = 0.005,
        max_retries: int = 5,
        max_mean_community_deviation: float = 0.1,
        max_max_community_deviation: float = 0.2,
        max_parameter_search_attempts: int = 10,
        parameter_search_range: float = 0.2
    ) -> sp.spmatrix:
        """
        Generate edges using a configuration model-like approach while respecting community structure.
        This method tries to match target degrees while maintaining community structure within specified deviation bounds.
        Community structure is prioritized over degree matching.
        
        Args:
            community_labels: Node community assignments (indices)
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors
            noise: Edge noise level
            min_edge_density: Minimum acceptable edge density
            max_retries: Maximum number of retries if graph is too sparse
            max_mean_community_deviation: Maximum allowed mean deviation from community structure (0-1)
            max_max_community_deviation: Maximum allowed maximum deviation from community structure (0-1)
            max_parameter_search_attempts: Maximum number of parameter search attempts
            parameter_search_range: How aggressively to search parameter space (0.05=small variations, 1.0=full random)
            
        Returns:
            Sparse adjacency matrix
        """
        n_nodes = len(community_labels)
        best_adj = None
        best_deviation = float('inf')
        best_mean_deviation = float('inf')
        best_max_deviation = float('inf')
        best_degree_params = None
        best_degree_stats = None
        
        # Define parameter search ranges based on search_range
        if self.degree_distribution == "power_law":
            # Calculate ranges based on search_range
            min_exponent = max(1.5, self.power_law_exponent - 2.0 * parameter_search_range)
            max_exponent = min(3.0, self.power_law_exponent + 2.0 * parameter_search_range)
            
            min_target = max(2, self.target_avg_degree * (1 - parameter_search_range))
            max_target = min(20, self.target_avg_degree * (1 + parameter_search_range))
            
            # Generate parameter combinations
            if parameter_search_range >= 0.8:  # For very aggressive search
                # Random sampling across the full range
                exponents = np.random.uniform(min_exponent, max_exponent, max_parameter_search_attempts)
                targets = np.random.uniform(min_target, max_target, max_parameter_search_attempts)
            else:
                # Linear spacing for more controlled search
                exponents = np.linspace(min_exponent, max_exponent, max_parameter_search_attempts)
                targets = np.linspace(min_target, max_target, max_parameter_search_attempts)
        else:
            # For other distributions, try different scaling factors
            if parameter_search_range >= 0.8:  # For very aggressive search
                scale_range = np.random.uniform(0.5, 2.0, max_parameter_search_attempts)
            else:
                scale_range = np.linspace(1 - parameter_search_range, 1 + parameter_search_range, max_parameter_search_attempts)
        
        # Try different parameter combinations
        for attempt in range(max_parameter_search_attempts):
            # Adjust degree factors based on attempt number
            if self.degree_distribution == "power_law":
                # Try different combinations of power law exponent and target average degree
                current_exponent = exponents[attempt]
                current_target = targets[attempt]
                
                # Generate new degree factors with current parameters
                adjusted_factors = self._generate_degree_factors_configuration(
                    n_nodes,
                    None,  # degree_heterogeneity not used in configuration model
                    self.degree_distribution,
                    current_exponent,
                    current_target
                )
                
                # Store current parameters for logging
                current_params = {
                    'power_law_exponent': current_exponent,
                    'target_avg_degree': current_target
                }
            else:
                # For other distributions, try different scaling factors
                scale = scale_range[attempt]
                adjusted_factors = degree_factors * scale
                current_params = {'scale_factor': scale}
            
            # Generate graph with current parameters
            adj = self._generate_edges_configuration_single(
                community_labels,
                P_sub,
                adjusted_factors,
                noise,
                min_edge_density,
                max_retries
            )
            
            if adj is None:
                continue
            
            # Create temporary graph for community analysis
            temp_graph = nx.from_scipy_sparse_array(adj)
            
            # Calculate both mean and maximum deviations
            mean_deviation = self._calculate_community_deviation(
                temp_graph,
                community_labels,
                P_sub
            )
            
            max_deviation = self._calculate_max_community_deviation(
                temp_graph,
                community_labels,
                P_sub
            )
            
            # Check if this is the best attempt so far
            # Weight community deviation more heavily than degree deviation
            total_deviation = 3 * mean_deviation + max_deviation
            
            # Only update best if both deviation constraints are better or equal
            if ((max_deviation < best_max_deviation or 
                 (max_deviation == best_max_deviation and mean_deviation < best_mean_deviation)) and
                max_deviation <= max_max_community_deviation and
                mean_deviation <= max_mean_community_deviation):
                best_adj = adj
                best_deviation = total_deviation
                best_mean_deviation = mean_deviation
                best_max_deviation = max_deviation
                best_degree_params = adjusted_factors.copy()
                best_degree_stats = {
                    'parameters': current_params,
                    'mean_degree': np.mean(adjusted_factors),
                    'std_degree': np.std(adjusted_factors),
                    'min_degree': np.min(adjusted_factors),
                    'max_degree': np.max(adjusted_factors)
                }
                
                # If we're within acceptable bounds, we can stop
                if (max_deviation <= max_max_community_deviation and 
                    mean_deviation <= max_mean_community_deviation):
                    print(f"\nFound solution with parameters:")
                    print(f"  {current_params}")
                    print(f"  Mean deviation: {mean_deviation:.4f}")
                    print(f"  Max deviation: {max_deviation:.4f}")
                    print(f"  Degree stats: mean={np.mean(adjusted_factors):.2f}, std={np.std(adjusted_factors):.2f}")
                    return adj
        
        # If we get here, we couldn't find a solution within bounds
        if best_adj is not None:
            print(f"\nWarning: Could not find solution within specified deviation bounds.")
            print(f"Best achieved: max_deviation={best_max_deviation:.4f}, "
                  f"mean_deviation={best_mean_deviation:.4f}")
            print(f"Best parameters found:")
            print(f"  {best_degree_stats['parameters']}")
            print(f"  Degree stats: mean={best_degree_stats['mean_degree']:.2f}, "
                  f"std={best_degree_stats['std_degree']:.2f}, "
                  f"min={best_degree_stats['min_degree']:.2f}, "
                  f"max={best_degree_stats['max_degree']:.2f}")
            
            # If either deviation constraint is not met, raise an error
            if best_max_deviation > max_max_community_deviation:
                raise ValueError(f"Could not generate graph with maximum community deviation below {max_max_community_deviation}. "
                               f"Best achieved: {best_max_deviation:.4f}")
            if best_mean_deviation > max_mean_community_deviation:
                raise ValueError(f"Could not generate graph with mean community deviation below {max_mean_community_deviation}. "
                               f"Best achieved: {best_mean_deviation:.4f}")
            
            return best_adj
        else:
            raise ValueError("Could not generate graph with specified constraints")

    def _calculate_community_deviation(
        self,
        graph: nx.Graph,
        community_labels: np.ndarray,
        P_sub: np.ndarray
    ) -> float:
        """
        Calculate mean absolute deviation between actual and expected community connection patterns.
        This represents the average deviation across all community pairs.
        """
        n_communities = len(self.communities)
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            community_sizes[label] += 1
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            actual_matrix[comm_i, comm_j] += 1
            actual_matrix[comm_j, comm_i] += 1  # Undirected graph
        
        # Calculate actual probabilities
        for i in range(n_communities):
            for j in range(n_communities):
                if i == j:
                    # Within community: divide by n(n-1)/2
                    n = community_sizes[i]
                    if n > 1:
                        actual_matrix[i, j] = actual_matrix[i, j] / (n * (n - 1))
                else:
                    # Between communities: divide by n1*n2
                    actual_matrix[i, j] = actual_matrix[i, j] / (community_sizes[i] * community_sizes[j])
        
        # Calculate mean deviation
        deviation_matrix = np.abs(actual_matrix - P_sub)
        return np.mean(deviation_matrix)

    def _calculate_max_community_deviation(
        self,
        graph: nx.Graph,
        community_labels: np.ndarray,
        P_sub: np.ndarray
    ) -> float:
        """
        Calculate maximum absolute deviation between actual and expected community connection patterns.
        This represents the worst-case deviation for any community pair.
        """
        n_communities = len(self.communities)
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            community_sizes[label] += 1
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            actual_matrix[comm_i, comm_j] += 1
            actual_matrix[comm_j, comm_i] += 1  # Undirected graph
        
        # Calculate actual probabilities
        for i in range(n_communities):
            for j in range(n_communities):
                if i == j:
                    # Within community: divide by n(n-1)/2
                    n = community_sizes[i]
                    if n > 1:
                        actual_matrix[i, j] = actual_matrix[i, j] / (n * (n - 1))
                else:
                    # Between communities: divide by n1*n2
                    actual_matrix[i, j] = actual_matrix[i, j] / (community_sizes[i] * community_sizes[j])
        
        # Calculate maximum deviation
        deviation_matrix = np.abs(actual_matrix - P_sub)
        return np.max(deviation_matrix)

    def _generate_edges_configuration_single(
        self,
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        min_edge_density: float = 0.005,
        max_retries: int = 5
    ) -> Optional[sp.spmatrix]:
        """
        Single attempt at generating edges using configuration model.
        This is the core edge generation logic, separated from the parameter search.
        Community structure is strictly maintained.
        """
        n_nodes = len(community_labels)
        
        # Calculate target degrees based on degree factors
        target_degrees = np.round(degree_factors).astype(int)
        
        # Ensure even sum of degrees
        if np.sum(target_degrees) % 2 == 1:
            target_degrees[np.random.randint(0, n_nodes)] += 1
        
        # Initialize edge list and remaining degrees
        edges = []
        remaining_degrees = target_degrees.copy()
        
        # Calculate connection probabilities between all node pairs
        conn_probs = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Calculate community-based probability
                comm_i = community_labels[i]
                comm_j = community_labels[j]
                prob = P_sub[comm_i, comm_j]
                
                # Add noise if specified
                if noise > 0:
                    prob *= (1 + np.random.uniform(-noise, noise))
                    prob = np.clip(prob, 0, 1)
                
                conn_probs[i, j] = conn_probs[j, i] = prob
        
        # While there are nodes with remaining degree
        while np.sum(remaining_degrees) > 1:
            # Choose node with highest remaining degree
            node_i = np.argmax(remaining_degrees)
            
            if remaining_degrees[node_i] == 0:
                break
            
            # Calculate sampling probabilities for potential partners
            # Only consider nodes with remaining degree
            valid_partners = remaining_degrees > 0
            valid_partners[node_i] = False  # No self-loops
            
            if not np.any(valid_partners):
                break
                
            sampling_probs = conn_probs[node_i] * valid_partners
            sampling_probs = sampling_probs / np.sum(sampling_probs)
            
            # Apply triangle enhancement if specified
            if self.triangle_enhancement > 0 and edges:
                triangle_boost = np.zeros(n_nodes)
                for edge in edges:
                    if edge[0] == node_i:
                        for other_edge in edges:
                            if other_edge[0] == edge[1]:
                                triangle_boost[other_edge[1]] += self.triangle_enhancement
                            elif other_edge[1] == edge[1]:
                                triangle_boost[other_edge[0]] += self.triangle_enhancement
                    elif edge[1] == node_i:
                        for other_edge in edges:
                            if other_edge[0] == edge[0]:
                                triangle_boost[other_edge[1]] += self.triangle_enhancement
                            elif other_edge[1] == edge[0]:
                                triangle_boost[other_edge[0]] += self.triangle_enhancement
                
                sampling_probs *= (1 + triangle_boost)
                sampling_probs = sampling_probs / np.sum(sampling_probs)
            
            # Sample partner node
            node_j = np.random.choice(n_nodes, p=sampling_probs)
            
            # Add edge
            edges.append((node_i, node_j))
            
            # Update remaining degrees
            remaining_degrees[node_i] -= 1
            remaining_degrees[node_j] -= 1
        
        # Create sparse adjacency matrix
        if edges:
            rows, cols = zip(*edges)
            data = np.ones(len(edges))
            adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            
            # Make symmetric (undirected graph)
            adj = adj + adj.T
            adj.data = np.ones_like(adj.data)  # Ensure no double edges
            
            # Calculate actual edge density
            actual_density = adj.nnz / (n_nodes * (n_nodes - 1))
            
            # If density is too low, try again with adjusted probabilities
            if actual_density < min_edge_density and max_retries > 0:
                return self._generate_edges_configuration_single(
                    community_labels,
                    P_sub * 2,  # Double the connection probabilities
                    degree_factors,
                    noise * 0.5,  # Reduce noise to maintain signal
                    min_edge_density,
                    max_retries - 1
                )
            
            return adj
        
        return None

    def analyze_community_connections(self) -> Dict[str, Any]:
        """
        Analyze the actual community connection patterns in the generated graph
        and compare them with the scaled probability matrix used for generation.
        
        Returns:
            Dictionary containing:
            - actual_matrix: Actual community-community connection probabilities
            - expected_matrix: Expected probabilities from scaled matrix
            - deviation_matrix: Absolute difference between actual and expected
            - mean_deviation: Mean absolute deviation
            - max_deviation: Maximum absolute deviation
            - community_sizes: Number of nodes in each community
            - connection_counts: Raw number of edges between communities
            - degree_analysis: Dictionary containing degree distribution analysis
        """
        n_communities = len(self.communities)
        
        # Calculate actual connection probabilities
        actual_matrix = np.zeros((n_communities, n_communities))
        connection_counts = np.zeros((n_communities, n_communities), dtype=int)
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in self.community_labels:
            community_sizes[label] += 1
        
        # Count edges between communities
        for i, j in self.graph.edges():
            comm_i = self.community_labels[i]
            comm_j = self.community_labels[j]
            connection_counts[comm_i, comm_j] += 1
            connection_counts[comm_j, comm_i] += 1  # Undirected graph
        
        # Calculate actual probabilities
        for i in range(n_communities):
            for j in range(n_communities):
                if i == j:
                    # Within community: divide by n(n-1)/2
                    n = community_sizes[i]
                    if n > 1:
                        actual_matrix[i, j] = connection_counts[i, j] / (n * (n - 1))
                else:
                    # Between communities: divide by n1*n2
                    actual_matrix[i, j] = connection_counts[i, j] / (community_sizes[i] * community_sizes[j])
        
        # Calculate deviation
        deviation_matrix = np.abs(actual_matrix - self.P_sub)
        mean_deviation = np.mean(deviation_matrix)
        max_deviation = np.max(deviation_matrix)
        
        # Analyze degree distribution
        actual_degrees = np.array([d for n, d in self.graph.degree()])
        
        # Get the best found parameters from the parameter search
        if hasattr(self, 'best_degree_stats'):
            best_params = self.best_degree_stats['parameters']
            target_degrees = np.round(self.best_degree_params).astype(int)
            used_power_law_exponent = best_params.get('power_law_exponent', self.power_law_exponent)
            used_target_avg_degree = best_params.get('target_avg_degree', self.target_avg_degree)
            used_scale_factor = best_params.get('scale_factor', 1.0)
        else:
            # Fallback to original parameters if no search was performed
            target_degrees = np.round(self.degree_factors).astype(int)
            used_power_law_exponent = self.power_law_exponent
            used_target_avg_degree = self.target_avg_degree
            used_scale_factor = 1.0
        
        # Calculate degree statistics
        degree_analysis = {
            'actual_degrees': actual_degrees,
            'target_degrees': target_degrees,
            'mean_actual_degree': np.mean(actual_degrees),
            'mean_target_degree': np.mean(target_degrees),
            'std_actual_degree': np.std(actual_degrees),
            'std_target_degree': np.std(target_degrees),
            'degree_deviation': np.mean(np.abs(actual_degrees - target_degrees) / target_degrees),
            'degree_correlation': np.corrcoef(actual_degrees, target_degrees)[0, 1],
            'used_parameters': {
                'power_law_exponent': used_power_law_exponent,
                'target_avg_degree': used_target_avg_degree,
                'scale_factor': used_scale_factor
            }
        }
        
        # Verify that max deviation is within constraints
        if hasattr(self, 'max_max_community_deviation') and max_deviation > self.max_max_community_deviation:
            print(f"Warning: Maximum community deviation ({max_deviation:.4f}) exceeds constraint ({self.max_max_community_deviation:.4f})")
            # Try to regenerate the graph with stricter constraints
            if hasattr(self, '_generate_edges_configuration'):
                print("Attempting to regenerate graph with stricter constraints...")
                self.adjacency = self._generate_edges_configuration(
                    self.community_labels,
                    self.P_sub,
                    self.degree_factors,
                    edge_noise=0.0,  # Use original parameters
                    min_edge_density=0.005,
                    max_retries=5,
                    max_mean_community_deviation=self.max_mean_community_deviation,
                    max_max_community_deviation=self.max_max_community_deviation,
                    max_parameter_search_attempts=10,
                    parameter_search_range=0.2
                )
                # Recreate the graph with new adjacency
                self.graph = nx.from_scipy_sparse_array(self.adjacency)
                # Recursively call analyze_community_connections to get updated analysis
                return self.analyze_community_connections()
        
        return {
            "actual_matrix": actual_matrix,
            "expected_matrix": self.P_sub,
            "deviation_matrix": deviation_matrix,
            "mean_deviation": mean_deviation,
            "max_deviation": max_deviation,
            "community_sizes": community_sizes,
            "connection_counts": connection_counts,
            "degree_analysis": degree_analysis
        }
