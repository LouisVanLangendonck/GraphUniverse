"""
Mixed-Membership Stochastic Block Model (MMSB) with overlapping communities.

This module implements a generative framework for creating graph instances
sampled from subsets of a core "graph universe" defined by a master
stochastic block model.

The model supports:
- Overlapping community memberships
- Degree correction
- Feature generation conditioned on community membership
- Systematic sampling of subgraphs from a larger universe

References:
- Airoldi et al. (2008). Mixed Membership Stochastic Blockmodels.
- Karrer & Newman (2011). Stochastic blockmodels and community structure in networks.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from .feature_regimes import FeatureRegimeGenerator, NeighborhoodFeatureAnalyzer, FeatureRegimeLabelGenerator


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
        intra_community_regime_similarity: float = 0.8,  # New parameter
        inter_community_regime_similarity: float = 0.2,  # New parameter
        community_co_membership: Optional[np.ndarray] = None,
        block_structure: str = "assortative",
        edge_density: float = 0.1,
        homophily: float = 0.8,
        randomness_factor: float = 0.0,
        mixed_membership: bool = True,
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
            community_co_membership: Optional co-membership probability matrix (K × K)
            block_structure: Type of block structure ("assortative", "disassortative", "core-periphery", "hierarchical")
            edge_density: Overall edge density
            homophily: Strength of within-community connections
            randomness_factor: Amount of random noise in edge probabilities
            mixed_membership: Whether nodes can belong to multiple communities
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
        """
        self.K = K
        self.feature_dim = feature_dim
        self.mixed_membership = mixed_membership
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
        
        # Generate or validate co-membership matrix
        if community_co_membership is not None:
            if community_co_membership.shape != (K, K):
                raise ValueError(f"Co-membership matrix must be {K}×{K}")
            self.community_co_membership = community_co_membership
        else:
            self.community_co_membership = self.generate_community_co_membership_matrix()
        
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
        
        self.edge_density = edge_density
        self.homophily = homophily
        self.block_structure = block_structure
        
        self.feature_variance = 0.1
        
        self.feature_similarity_matrix = None

    def _generate_probability_matrix(
        self, 
        K: int, 
        structure: str, 
        edge_density: float,
        homophily: float = 0.8,  # Controls ratio between intra/inter probabilities
        randomness_factor: float = 0.0
    ) -> np.ndarray:
        """
        Generate a structured probability matrix based on the specified pattern with optional randomness.
        
        Args:
            K: Number of communities
            structure: Type of block structure
            edge_density: Overall target edge density
            homophily: Controls ratio between intra/inter probabilities (0-1)
                0 = no homophily (equal probabilities)
                1 = maximum homophily (all edges within communities)
            randomness_factor: Amount of random noise to add (0.0-1.0)
            
        Returns:
            K × K probability matrix
        """
        P = np.zeros((K, K))
        
        if structure == "assortative":
            # Calculate intra and inter probabilities based on homophily and target density
            # Let x be intra prob and y be inter prob
            # We want: x * f + y * (1-f) = d where:
            # f = fraction of possible edges that are intra-community = 1/K
            # d = target density
            # And we want: x = y * (1 + h/(1-h)) where h = homophily
            
            f = 1/K  # Fraction of possible edges that are intra-community
            d = edge_density  # Target density
            
            if homophily < 1.0:
                # Calculate ratio between intra and inter probabilities
                ratio = (1 + homophily/(1-homophily))
                
                # Solve for y (inter probability)
                # y * (f*ratio + (1-f)) = d
                inter_prob = d / (f*ratio + (1-f))
                intra_prob = ratio * inter_prob
                
                # Ensure probabilities are valid
                if intra_prob > 1.0:
                    intra_prob = 1.0
                    inter_prob = (d - f) / (1-f)
            else:
                # Maximum homophily - all edges within communities
                intra_prob = min(1.0, d/f)
                inter_prob = 0.0
            
            # Set probabilities
            np.fill_diagonal(P, intra_prob)
            P[~np.eye(K, dtype=bool)] = inter_prob
            
        elif structure == "disassortative":
            # For disassortative structure, we invert the homophily
            return self._generate_probability_matrix(
                K, "assortative", edge_density, 1-homophily, randomness_factor
            )
            
        elif structure == "core-periphery":
            # First community is the core, others are periphery
            core_size = max(1, K // 10)
            
            # Core-core connections
            P[:core_size, :core_size] = edge_density
            
            # Core-periphery connections
            P[core_size:, :core_size] = edge_density / 2
            P[:core_size, core_size:] = edge_density / 2
            
            # Periphery-periphery connections
            P[core_size:, core_size:] = edge_density / 4
            
        elif structure == "hierarchical":
            # Create hierarchical block structure
            levels = int(np.log2(K)) + 1
            for i in range(K):
                for j in range(K):
                    # Compute hierarchical distance (in binary representation)
                    bin_i, bin_j = bin(i)[2:].zfill(levels), bin(j)[2:].zfill(levels)
                    common_prefix = 0
                    for b_i, b_j in zip(bin_i, bin_j):
                        if b_i == b_j:
                            common_prefix += 1
                        else:
                            break
                    
                    # Probability decreases with hierarchical distance
                    hier_distance = levels - common_prefix
                    P[i, j] = edge_density * (0.5 ** hier_distance)
        
        elif structure == "random_blocks":
            # Completely random block structure with some constraints
            # Diagonal blocks have higher probabilities on average
            for i in range(K):
                for j in range(K):
                    if i == j:  # Diagonal blocks
                        P[i, j] = np.random.uniform(edge_density/2, edge_density)
                    else:
                        # Off-diagonal blocks have varying probabilities
                        # but generally lower than diagonal blocks
                        P[i, j] = np.random.uniform(edge_density/4, edge_density/2)
                    
        else:
            # Random uniform probabilities
            P = np.random.uniform(low=edge_density/2, high=edge_density, size=(K, K))
        
        # Add randomness if requested
        if randomness_factor > 0:
            # Generate random noise with values in [-randomness_factor/2, randomness_factor/2]
            noise = np.random.uniform(-randomness_factor/2, randomness_factor/2, size=(K, K))
            
            # Apply noise to probabilities
            P = P * (1 + noise)
            
            # Ensure probabilities stay within valid range [0, 1]
            P = np.clip(P, 0, 1)
            
            # For assortative structure, ensure diagonal still has higher values on average
            if structure == "assortative":
                # Calculate average diagonal and off-diagonal values
                diag_avg = np.mean(np.diag(P))
                off_diag_avg = (np.sum(P) - np.sum(np.diag(P))) / (K*K - K)
                
                # If the average off-diagonal is higher than diagonal, swap them
                if off_diag_avg > diag_avg:
                    diag_scale = edge_density / diag_avg
                    off_diag_scale = (edge_density/2) / off_diag_avg
                    
                    # Apply scaling
                    for i in range(K):
                        for j in range(K):
                            if i == j:
                                P[i, j] *= diag_scale
                            else:
                                P[i, j] *= off_diag_scale
                    
                    # Re-clip to ensure valid range
                    P = np.clip(P, 0, 1)
        
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
            
    def generate_community_co_membership_matrix(
        self, 
        overlap_density: float = 0.1,
        structure: str = "modular"
    ) -> np.ndarray:
        """
        Generate a community co-membership matrix to model overlapping communities.
        
        Args:
            overlap_density: Overall density of overlaps between communities
            structure: Structure of overlaps ("modular", "hierarchical", "hub-spoke")
            
        Returns:
            K × K co-membership probability matrix
        """
        K = self.K
        co_membership = np.eye(K)  # Diagonal is always 1 (self-membership)
        
        if structure == "modular":
            # Create modules of overlapping communities
            module_size = max(2, K // 5)  # Module size
            for i in range(0, K, module_size):
                module_end = min(i + module_size, K)
                module_indices = np.arange(i, module_end)
                
                # Communities within same module have higher overlap probability
                for j in module_indices:
                    for k in module_indices:
                        if j != k:
                            co_membership[j, k] = overlap_density
                            
        elif structure == "hierarchical":
            # Hierarchical structure with stronger overlaps between related communities
            levels = int(np.log2(K)) + 1
            for i in range(K):
                for j in range(i+1, K):
                    # Compute overlap based on hierarchical distance
                    bin_i, bin_j = bin(i)[2:].zfill(levels), bin(j)[2:].zfill(levels)
                    common_prefix = 0
                    for b_i, b_j in zip(bin_i, bin_j):
                        if b_i == b_j:
                            common_prefix += 1
                        else:
                            break
                    
                    # Probability decreases with hierarchical distance
                    hier_distance = levels - common_prefix
                    overlap_prob = overlap_density * (0.5 ** (hier_distance - 1))
                    co_membership[i, j] = co_membership[j, i] = overlap_prob
                    
        elif structure == "hub-spoke":
            # Some communities are hubs that overlap with many others
            num_hubs = max(1, K // 10)
            hub_indices = np.random.choice(K, size=num_hubs, replace=False)
            
            # Hub-spoke connections
            for hub in hub_indices:
                for i in range(K):
                    if i != hub:
                        co_membership[hub, i] = co_membership[i, hub] = overlap_density
                        
            # Hub-hub connections (stronger)
            for i in hub_indices:
                for j in hub_indices:
                    if i != j:
                        co_membership[i, j] = overlap_density * 2
                        
        else:
            # Random uniform co-membership
            for i in range(K):
                for j in range(i+1, K):
                    p = np.random.uniform(0, overlap_density)
                    co_membership[i, j] = co_membership[j, i] = p
                    
        return co_membership

    def sample_connected_community_subset(
        self,
        size: int,
        method: str = "random",
        similarity_bias: float = 0.0,
        min_connection_strength: float = 0.05,
        max_attempts: int = 10,
        existing_communities: Optional[List[int]] = None
    ) -> List[int]:
        """
        Enhanced version that first samples seed communities and then finds their strongly connected partners.
        This approach better ensures that each community has at least one strong connection to another community.
        
        Args:
            size: Number of communities to sample
            method: Sampling method ("random", "similar", "diverse", "correlated")
            similarity_bias: Controls bias towards similar communities (positive) or diverse (negative)
            min_connection_strength: Minimum probability threshold for considering communities connected
            max_attempts: Maximum number of sampling attempts to find a suitable subset
            existing_communities: For transfer learning, optionally condition on existing communities
            
        Returns:
            List of sampled community indices
        """
        size = min(size, self.K)
        
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
            seeds = self.sample_community_subset(
                size=n_seeds,
                method=method,
                similarity_bias=similarity_bias
            )
            remaining_size = size - n_seeds
        
        # Initialize result with seeds
        result = set(seeds)
        
        # First round: find strongly connected partners for each seed
        for seed in seeds:
            # Find all communities with strong connections to this seed
            strong_connections = [
                j for j in range(self.K)
                if j not in result and self.P[seed, j] >= min_connection_strength
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
            for j in range(self.K):
                if j not in result:
                    # Check connection strength to all existing communities
                    max_connection = max(self.P[j, i] for i in result)
                    if max_connection >= min_connection_strength:
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
            remaining = set(range(self.K)) - result
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
            print(f"  Connection strength threshold: {min_connection_strength}")
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
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        communities: List[int],
        n_nodes: int,
        min_component_size: int = 0,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        indirect_influence: float = 0.1,
        feature_regime_balance: float = 0.5,  # Renamed from feature_subtype_mixing
        seed: Optional[int] = None
    ):
        """
        Initialize and generate a graph sample.
        
        Args:
            universe: Parent universe
            communities: List of community indices to include
            n_nodes: Number of nodes
            min_component_size: Minimum size of connected components to keep
            degree_heterogeneity: Controls degree distribution (0=uniform, 1=heterogeneous)
            edge_noise: Amount of random noise in edge probabilities
            indirect_influence: How strongly co-memberships influence edge formation
            feature_regime_balance: How evenly regimes are distributed within communities
            seed: Random seed for reproducibility
        """
        self.universe = universe
        self.communities = sorted(communities)
        self.original_n_nodes = n_nodes
        self.min_component_size = min_component_size
        
        # Store feature generation parameters
        self.feature_regime_balance = feature_regime_balance
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Extract the submatrix of the probability matrix for these communities
        K_sub = len(communities)
        P_sub = np.zeros((K_sub, K_sub))
        for i, ci in enumerate(communities):
            for j, cj in enumerate(communities):
                P_sub[i, j] = universe.P[ci, cj]
        
        # Generate membership vectors for all original nodes
        self.membership_vectors = self._generate_memberships(n_nodes, K_sub)
        
        # Generate degree correction factors for all original nodes
        self.degree_factors = self._generate_degree_factors(n_nodes, degree_heterogeneity)
        
        # Generate edges with indirect influence parameter
        self.adjacency = self._generate_edges(
            self.membership_vectors, 
            P_sub, 
            self.degree_factors, 
            edge_noise,
            indirect_influence
        )
        
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
                        primary_comm = self.communities[np.argmax(self.membership_vectors[node])]
                        if primary_comm not in self.deleted_node_types:
                            self.deleted_node_types[primary_comm] = 0
                        self.deleted_node_types[primary_comm] += 1
        else:
            kept_components = components
            
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
            
            # Update membership vectors and degree factors with new indices
            self.membership_vectors = self.membership_vectors[kept_nodes]
            self.degree_factors = self.degree_factors[kept_nodes]
            
            # Update adjacency matrix
            self.adjacency = nx.adjacency_matrix(self.graph)
            
            # Generate features if enabled
            if universe.feature_dim > 0:
                # Assign nodes to feature regimes
                primary_communities = np.argmax(self.membership_vectors, axis=1)
                self.node_regimes = universe.regime_generator.assign_node_regimes(
                    primary_communities,
                    regime_balance=self.feature_regime_balance
                )
                
                # Generate features based on regimes
                self.features = universe.regime_generator.generate_node_features(
                    self.node_regimes
                )
                
                # Initialize neighborhood analyzer
                self.neighborhood_analyzer = NeighborhoodFeatureAnalyzer(
                    graph=self.graph,
                    node_regimes=self.node_regimes,
                    total_regimes=len(communities) * universe.regimes_per_community,
                    max_hops=3
                )
                
                # Generate balanced labels based on neighborhood features
                self.label_generator = FeatureRegimeLabelGenerator(
                    frequency_vectors=self.neighborhood_analyzer.get_all_frequency_vectors(1),
                    n_labels=len(communities),
                    balance_tolerance=0.1,
                    seed=seed
                )
                
                # Get node labels
                self.node_labels = self.label_generator.get_node_labels()
            else:
                self.features = None
                self.node_regimes = None
                self.neighborhood_analyzer = None
                self.label_generator = None
                
            # Add node attributes including primary community labels
            self._add_node_attributes()
            
        else:
            # If no components meet the size threshold, keep an empty graph
            self.graph = nx.Graph()
            self.n_nodes = 0
            self.membership_vectors = np.zeros((0, K_sub))
            self.degree_factors = np.zeros(0)
            self.adjacency = sp.csr_matrix((0, 0))
            self.features = None if universe.feature_dim > 0 else None
            self.node_map = {}
            self.reverse_node_map = {}
            self.node_labels = np.zeros(0, dtype=int)
            self.node_regimes = None
            self.neighborhood_analyzer = None
            self.label_generator = None

    def _add_node_attributes(self):
        """Add node attributes to the graph."""
        for i, node in enumerate(self.graph.nodes()):
            # Calculate primary community for this node
            primary_comm = self.communities[np.argmax(self.membership_vectors[i])]
            
            # Create node attributes dictionary
            node_attrs = {
                "memberships": {self.communities[j]: float(self.membership_vectors[i, j]) 
                               for j in range(len(self.communities)) 
                               if self.membership_vectors[i, j] > 0},
                "degree_factor": float(self.degree_factors[i]),
                "primary_community": int(primary_comm)  # Add primary community as integer
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
            self.membership_vectors,
            self.communities
        )
        
        return params

    def _generate_memberships(
        self, 
        n_nodes: int, 
        K_sub: int, 
        avg_memberships: float = None,  # Made optional as it's no longer used
        concentration: float = None      # Made optional as it's no longer used
    ) -> np.ndarray:
        """
        Generate memberships strictly respecting universe co-membership rules.
        If mixed_membership is False, each node belongs to exactly one community.
        
        Args:
            n_nodes: Number of nodes
            K_sub: Number of communities in this subgraph
            avg_memberships: Deprecated, kept for backwards compatibility
            concentration: Deprecated, kept for backwards compatibility
            
        Returns:
            n_nodes × K_sub matrix of membership probabilities
        """
        if avg_memberships is not None or concentration is not None:
            print("[WARNING] avg_memberships and concentration parameters are deprecated. "
                  "Memberships are now generated based on universe co-membership rules.")
        
        membership = np.zeros((n_nodes, K_sub))
        
        if not self.universe.mixed_membership:
            # Simple case: each node belongs to exactly one community
            primary_communities = np.random.choice(K_sub, size=n_nodes)
            membership[np.arange(n_nodes), primary_communities] = 1.0
            return membership
        
        # Get the relevant submatrix of co-membership probabilities
        co_membership_sub = np.zeros((K_sub, K_sub))
        for i, ci in enumerate(self.communities):
            for j, cj in enumerate(self.communities):
                co_membership_sub[i, j] = self.universe.community_co_membership[ci, cj]
        
        for i in range(n_nodes):
            # Always start with one primary community
            primary_idx = np.random.choice(K_sub)
            # Primary membership strength is high but not always 1.0
            membership[i, primary_idx] = np.random.beta(8, 2)  # Strongly skewed toward high values
            
            # For each other possible community, check co-membership probability
            for j in range(K_sub):
                if j != primary_idx:
                    # Sample based on co-membership probability
                    if np.random.random() < co_membership_sub[primary_idx, j]:
                        # If co-membership occurs, add with a weight sampled from Beta
                        # Beta(2,5) gives right-skewed distribution - most secondary memberships
                        # will be moderate to low, with few high values
                        membership[i, j] = np.random.beta(2, 5) * membership[i, primary_idx]
            
            # Normalize membership vector for this node
            if np.sum(membership[i]) > 0:  # Avoid division by zero
                membership[i] = membership[i] / np.sum(membership[i])
        
        return membership
    
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
    
    def _calculate_edge_probability_vectorized(
        self,
        node_i: int,
        node_j: int,
        memberships: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        indirect_influence: float = 0.1,  # New parameter to control indirect effect strength
        precomputed: Optional[Dict] = None
    ) -> float:
        """
        Vectorized version of edge probability calculation.
        Uses pre-computation and numpy operations for speed.
        
        Args:
            indirect_influence: Controls how much indirect connections influence the final probability (0-1)
        """
        # Fast path for non-mixed membership case
        if not self.universe.mixed_membership:
            # Get primary communities (indices where membership is 1)
            comm_i = np.argmax(memberships[node_i])
            comm_j = np.argmax(memberships[node_j])
            # Direct probability from P matrix
            edge_prob = P_sub[comm_i, comm_j]
            # Apply degree correction
            edge_prob *= degree_factors[node_i] * degree_factors[node_j]
            # Add noise if specified
            if noise > 0:
                edge_prob *= (1 + np.random.uniform(-noise, noise))
            return np.clip(edge_prob, 0, 1)
            
        # Use pre-computed matrices if available
        if precomputed is None:
            precomputed = {}
        
        # Direct connections (fast matrix operation)
        direct_prob = memberships[node_i] @ P_sub @ memberships[node_j]
        
        # Skip indirect connections if indirect_influence is 0
        if indirect_influence <= 0:
            edge_prob = direct_prob
        else:
            # Indirect connections through co-membership
            if 'indirect_matrix' not in precomputed:
                # Pre-compute the indirect connection matrix once
                K = len(self.communities)
                indirect_matrix = np.zeros((K, K))
                
                # Create masks for non-selected communities
                mask = np.ones(self.universe.K, dtype=bool)
                mask[self.communities] = False
                
                # Vectorized computation of indirect connections
                for ci, comm_i in enumerate(self.communities):
                    for cj, comm_j in enumerate(self.communities):
                        if ci != cj:  # Prevent self-reinforcement through co-membership
                            # Get co-membership strengths for both communities
                            co_mem_i = self.universe.community_co_membership[comm_i]
                            co_mem_j = self.universe.community_co_membership[comm_j]
                            
                            # Only consider connections through non-selected communities
                            valid_intermediates = mask & (co_mem_i > 0) & (co_mem_j > 0)
                            
                            if np.any(valid_intermediates):
                                # Calculate indirect connection strength
                                # Scale by the actual co-membership strengths
                                intermediates = (
                                    co_mem_i[valid_intermediates] *
                                    co_mem_j[valid_intermediates] *
                                    self.universe.P[valid_intermediates, :][:, valid_intermediates]
                                )
                                
                                # Take average of intermediate connections, weighted by co-membership strengths
                                indirect_matrix[ci, cj] = np.mean(intermediates)
                
                precomputed['indirect_matrix'] = indirect_matrix
            
            # Use pre-computed indirect matrix (fast matrix operation)
            indirect_prob = memberships[node_i] @ precomputed['indirect_matrix'] @ memberships[node_j]
            
            # Scale indirect probability by influence factor
            indirect_prob *= indirect_influence
            
            # Combine probabilities
            edge_prob = direct_prob + indirect_prob
            
        # Apply degree correction
        edge_prob *= degree_factors[node_i] * degree_factors[node_j]
        
        # Add noise if specified
        if noise > 0:
            edge_prob *= (1 + np.random.uniform(-noise, noise))
        
        return np.clip(edge_prob, 0, 1)

    def _generate_edges(
        self, 
        memberships: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        indirect_influence: float = 0.1,  # Add parameter here too
        min_edge_density: float = 0.01,  # Minimum acceptable edge density
        max_retries: int = 3  # Maximum number of retries
    ) -> sp.spmatrix:
        """
        Generate edges with minimum density guarantee.
        
        Args:
            memberships: Node-community membership matrix
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors
            noise: Edge noise level
            indirect_influence: Strength of indirect community influence
            min_edge_density: Minimum acceptable edge density
            max_retries: Maximum number of retries if graph is too sparse
            
        Returns:
            Sparse adjacency matrix
        """
        n_nodes = memberships.shape[0]
        
        for attempt in range(max_retries):
            # Sparse matrix for efficient storage
            rows, cols, data = [], [], []
            
            # For efficiency, precompute the membership edge probability tensor
            MP = memberships @ P_sub  # n×k
            
            # Compute edge probabilities efficiently
            for i in range(n_nodes):
                # For each node i, compute probabilities to all nodes j
                non_zero_i = np.nonzero(memberships[i])[0]
                
                if len(non_zero_i) > 0:
                    # This computes the sum over a,b efficiently for all j at once
                    weighted_prob = (MP @ memberships.T)[i, :]
                    
                    # Apply degree correction
                    edge_probs = degree_factors[i] * degree_factors * weighted_prob
                    
                    # Add noise if specified
                    if noise > 0:
                        edge_probs += np.random.uniform(-noise, noise, size=n_nodes) * edge_probs
                        edge_probs = np.clip(edge_probs, 0, 1)
                    
                    # Sample edges (upper triangular part only to avoid duplicates)
                    for j in range(i+1, n_nodes):
                        if np.random.random() < edge_probs[j]:
                            rows.append(i)
                            cols.append(j)
                            data.append(1)
                            
                            # Add the reverse edge for undirected graph
                            rows.append(j)
                            cols.append(i)
                            data.append(1)
            
            # Create sparse adjacency matrix
            adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            
            # Calculate actual edge density
            actual_density = len(data) / (n_nodes * (n_nodes - 1))
            
            if actual_density >= min_edge_density:
                return adj
            
            # If density is too low, increase edge probabilities
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Graph too sparse (density={actual_density:.4f}). Retrying with adjusted probabilities...")
                # Increase edge probabilities by a factor
                P_sub = P_sub * 2  # Double the connection probabilities
                noise = noise * 0.5  # Reduce noise to maintain signal
            else:
                print(f"Warning: Could not achieve minimum edge density after {max_retries} attempts.")
                print(f"Final density: {actual_density:.4f}")
                return adj
        
        return adj

    def _generate_edges_with_component_filtering(
        self, 
        memberships: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float,
        max_retries: int = 3
    ) -> sp.spmatrix:
        """
        Generate edges with component filtering, ensuring at least one valid component.
        
        Args:
            memberships: Node-community membership matrix
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors
            noise: Edge noise level
            max_retries: Maximum number of retries if no valid components found
            
        Returns:
            Sparse adjacency matrix
        """
        # First try: Use specified min_component_size
        adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
        G = nx.from_scipy_sparse_array(adj)
        
        # Get all components
        components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in components]
        
        # Check if any components meet the minimum size
        valid_components = [c for c in components if len(c) >= self.min_component_size]
        
        if not valid_components:
            # If no valid components, try keeping the largest component
            largest_component = max(components, key=len)
            if len(largest_component) >= 5:  # Minimum acceptable size
                # Create new graph with only the largest component
                G = G.subgraph(largest_component).copy()
                # Update memberships and degree_factors
                valid_nodes = sorted(largest_component)
                memberships = memberships[valid_nodes]
                degree_factors = degree_factors[valid_nodes]
                # Generate new adjacency matrix
                adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
            else:
                # If largest component is too small, retry with lower min_component_size
                for retry in range(max_retries):
                    # Reduce min_component_size by half each time
                    reduced_size = max(2, self.min_component_size // 2)
                    print(f"Retry {retry + 1}: No valid components found. Trying with min_component_size={reduced_size}")
                    
                    # Generate new edges
                    adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
                    G = nx.from_scipy_sparse_array(adj)
                    
                    # Get valid components with reduced size
                    valid_components = [c for c in nx.connected_components(G) 
                                      if len(c) >= reduced_size]
                    
                    if valid_components:
                        # Keep all valid components
                        valid_nodes = sorted(set().union(*valid_components))
                        G = G.subgraph(valid_nodes).copy()
                        # Update memberships and degree_factors
                        memberships = memberships[valid_nodes]
                        degree_factors = degree_factors[valid_nodes]
                        # Generate final adjacency matrix
                        adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
                        break
                
                if not valid_components:
                    # If still no valid components after retries, keep largest component
                    largest_component = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_component).copy()
                    valid_nodes = sorted(largest_component)
                    memberships = memberships[valid_nodes]
                    degree_factors = degree_factors[valid_nodes]
                    adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
        else:
            # Keep all valid components
            valid_nodes = sorted(set().union(*valid_components))
            G = G.subgraph(valid_nodes).copy()
            # Update memberships and degree_factors
            memberships = memberships[valid_nodes]
            degree_factors = degree_factors[valid_nodes]
            # Generate final adjacency matrix
            adj = self._generate_edges(memberships, P_sub, degree_factors, noise)
        
        # Update the graph's node count
        self.n_nodes = adj.shape[0]
        
        # Store the valid nodes for reference
        self.valid_nodes = valid_nodes
        
        return adj

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
                "memberships": {self.communities[j]: float(self.membership_vectors[i, j]) 
                               for j in range(len(self.communities)) 
                               if self.membership_vectors[i, j] > 0},
                "degree_factor": float(self.degree_factors[i])
            }
            
            # Add primary community label (for visualization and evaluation)
            primary_comm = self.communities[np.argmax(self.membership_vectors[i])]
            node_attrs["primary_community"] = int(primary_comm)
            
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


class MMSBBenchmark:
    """
    Generates benchmarks of graph instances for pretraining and transfer learning.
    Each benchmark consists of a set of graphs sampled from a common universe.
    """
    
    def __init__(
        self,
        K: int = 100,
        feature_dim: int = 64,
        feature_signal: float = 1.0,
        block_structure: str = "hierarchical",
        overlap_structure: str = "modular",
        edge_density: float = 0.1,
        homophily: float = 0.8,
        randomness_factor: float = 0.0,
        feature_regime_strength: float = 0.8,  # New parameter
        feature_regime_overlap: float = 0.2,   # New parameter
        regimes_per_community: int = 2,        # New parameter
        seed: Optional[int] = None
    ):
        """
        Initialize the benchmark generator.
        
        Args:
            K: Total number of communities in the universe
            feature_dim: Dimension of node features
            feature_signal: Correlation strength of features with community membership
            block_structure: Type of block structure ("assortative", "disassortative", "core-periphery", "hierarchical")
            overlap_structure: Type of community overlap structure ("modular", "random", "hierarchical")
            edge_density: Overall edge density
            homophily: Strength of within-community connections
            randomness_factor: Amount of random noise in edge probabilities
            feature_regime_strength: How strongly features correlate with regimes
            feature_regime_overlap: How much regimes overlap between communities
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
        """
        # Initialize the universe
        self.universe = GraphUniverse(
            K=K,
            feature_dim=feature_dim,
            feature_signal=feature_signal,
            block_structure=block_structure,
            edge_density=edge_density,
            homophily=homophily,
            randomness_factor=randomness_factor,
            feature_structure="distinct",
            feature_subtypes_per_community=1,
            feature_mixing_strength=0.5,
            feature_similarity_control=0.7,
            mixed_membership=True,
            feature_regime_strength=feature_regime_strength,
            feature_regime_overlap=feature_regime_overlap,
            regimes_per_community=regimes_per_community,
            seed=seed
        )
        
        # Generate community co-membership matrix
        self.universe.community_co_membership = self.universe.generate_community_co_membership_matrix(
            overlap_density=0.1,
            structure=overlap_structure
        )
        
        # Store parameters
        self.K = K
        self.feature_dim = feature_dim
        self.feature_signal = feature_signal
        self.block_structure = block_structure
        self.overlap_structure = overlap_structure
        self.edge_density = edge_density
        self.homophily = homophily
        self.randomness_factor = randomness_factor
        self.feature_regime_strength = feature_regime_strength
        self.feature_regime_overlap = feature_regime_overlap
        self.regimes_per_community = regimes_per_community
        self.seed = seed
        
        # Add connected community sampling to universe
        self.add_connected_community_sampling_to_universe()
    
    def generate_pretraining_graphs(
        self,
        n_graphs: int,
        min_communities: int = 5,
        max_communities: int = 15,
        min_nodes: int = 100,
        max_nodes: int = 500,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        sampling_method: str = "random",
        similarity_bias: float = 0.0,
        min_connection_strength: float = 0.05,
        min_component_size: int = 0,
        indirect_influence: float = 0.1,
        feature_regime_strength: Optional[float] = None,  # New parameter
        feature_regime_overlap: Optional[float] = None,   # New parameter
        regimes_per_community: Optional[int] = None,      # New parameter
        seed: Optional[int] = None
    ) -> List[GraphSample]:
        """
        Generate a set of graphs for pretraining.
        
        Args:
            n_graphs: Number of graphs to generate
            min_communities: Minimum number of communities per graph
            max_communities: Maximum number of communities per graph
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            degree_heterogeneity: Controls degree distribution (0=uniform, 1=heterogeneous)
            edge_noise: Amount of random noise in edge generation
            sampling_method: Method for sampling communities ("random", "similar", "dissimilar")
            similarity_bias: Bias toward similar/dissimilar communities
            min_connection_strength: Minimum connection strength between communities
            min_component_size: Minimum size of connected components to keep
            indirect_influence: Strength of indirect influence in edge generation
            feature_regime_strength: How strongly features correlate with regimes
            feature_regime_overlap: How much regimes overlap between communities
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
            
        Returns:
            List of generated GraphSample instances
        """
        # Use universe parameters if not specified
        if feature_regime_strength is None:
            feature_regime_strength = self.feature_regime_strength
        if feature_regime_overlap is None:
            feature_regime_overlap = self.feature_regime_overlap
        if regimes_per_community is None:
            regimes_per_community = self.regimes_per_community
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        graphs = []
        for i in range(n_graphs):
            # Sample number of communities
            n_communities = np.random.randint(min_communities, max_communities + 1)
            
            # Sample communities based on method
            if sampling_method == "random":
                communities = self.universe.sample_community_subset(
                    n_communities,
                    method="random",
                    similarity_bias=similarity_bias
                )
            elif sampling_method == "connected":
                communities = self.universe.sample_connected_community_subset(
                    n_communities,
                    method="random",
                    similarity_bias=similarity_bias,
                    min_connection_strength=min_connection_strength
                )
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}")
                
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                min_component_size=min_component_size,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                indirect_influence=indirect_influence,
                feature_regime_balance=0.5,  # Renamed from feature_subtype_mixing
                seed=seed + i if seed is not None else None
            )
            
            graphs.append(graph)
            
        return graphs
    
    def generate_transfer_graphs(
        self,
        n_graphs: int,
        reference_graphs: List[GraphSample],
        transfer_mode: str = "new_combinations",
        transfer_difficulty: float = 0.5,
        min_nodes: int = 100,
        max_nodes: int = 500,
        degree_heterogeneity: float = 0.5,
        edge_noise: float = 0.0,
        min_connection_strength: float = 0.05,
        min_component_size: int = 0,
        indirect_influence: float = 0.1,
        feature_regime_strength: Optional[float] = None,  # New parameter
        feature_regime_overlap: Optional[float] = None,   # New parameter
        regimes_per_community: Optional[int] = None,      # New parameter
        seed: Optional[int] = None
    ) -> List[GraphSample]:
        """
        Generate transfer learning graphs based on reference graphs.
        
        Args:
            n_graphs: Number of graphs to generate
            reference_graphs: List of reference graphs to base transfer on
            transfer_mode: Type of transfer ("new_combinations", "new_communities", "new_features")
            transfer_difficulty: How different transfer graphs should be (0=easy, 1=hard)
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            degree_heterogeneity: Controls degree distribution (0=uniform, 1=heterogeneous)
            edge_noise: Amount of random noise in edge generation
            min_connection_strength: Minimum connection strength between communities
            min_component_size: Minimum size of connected components to keep
            indirect_influence: Strength of indirect influence in edge generation
            feature_regime_strength: How strongly features correlate with regimes
            feature_regime_overlap: How much regimes overlap between communities
            regimes_per_community: Number of feature regimes per community
            seed: Random seed for reproducibility
            
        Returns:
            List of generated GraphSample instances
        """
        # Use universe parameters if not specified
        if feature_regime_strength is None:
            feature_regime_strength = self.feature_regime_strength
        if feature_regime_overlap is None:
            feature_regime_overlap = self.feature_regime_overlap
        if regimes_per_community is None:
            regimes_per_community = self.regimes_per_community
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Extract reference communities
        reference_communities = set()
        for graph in reference_graphs:
            reference_communities.update(graph.communities)
        reference_communities = sorted(list(reference_communities))
        
        graphs = []
        for i in range(n_graphs):
            if transfer_mode == "new_combinations":
                # Sample new combinations of reference communities
                n_communities = np.random.randint(
                    max(2, int(len(reference_communities) * (1 - transfer_difficulty))),
                    len(reference_communities) + 1
                )
                communities = np.random.choice(
                    reference_communities,
                    size=n_communities,
                    replace=False
                ).tolist()
                
            elif transfer_mode == "new_communities":
                # Sample new communities not in reference set
                available_communities = [c for c in range(self.K) if c not in reference_communities]
                n_communities = np.random.randint(
                    max(2, int(len(available_communities) * (1 - transfer_difficulty))),
                    len(available_communities) + 1
                )
                communities = np.random.choice(
                    available_communities,
                    size=n_communities,
                    replace=False
                ).tolist()
                
            elif transfer_mode == "new_features":
                # Use same communities but different feature regimes
                n_communities = np.random.randint(
                    max(2, int(len(reference_communities) * (1 - transfer_difficulty))),
                    len(reference_communities) + 1
                )
                communities = np.random.choice(
                    reference_communities,
                    size=n_communities,
                    replace=False
                ).tolist()
                
                # Adjust feature regime parameters
                feature_regime_strength = self.feature_regime_strength * (1 - transfer_difficulty)
                feature_regime_overlap = self.feature_regime_overlap * (1 + transfer_difficulty)
                
            else:
                raise ValueError(f"Unknown transfer mode: {transfer_mode}")
                
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                min_component_size=min_component_size,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                indirect_influence=indirect_influence,
                feature_regime_balance=0.5,  # Renamed from feature_subtype_mixing
                seed=seed + i if seed is not None else None
            )
            
            graphs.append(graph)
            
        return graphs

    def generate_node_features(
        self,
        graph: GraphSample,
        feature_signal: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate node features for a given graph.
        
        Args:
            graph: Graph sample to generate features for
            feature_signal: Optional override for feature signal strength
                (if None, uses universe's feature_signal)
            
        Returns:
            Node feature matrix
        """
        if self.universe.feature_dim == 0:
            return None
        
        # Use the graph's _generate_features method with either provided or universe's feature_signal
        signal = feature_signal if feature_signal is not None else self.universe.feature_signal
        
        features = graph._generate_features(
            graph.membership_vectors,
            self.universe.feature_prototypes,
            signal
        )
        
        return features
    
    def generate_node_labels(
        self,
        graph: GraphSample,
        label_type: str = "primary_community",
        label_noise: float = 0.0
    ) -> np.ndarray:
        """
        Generate node labels for evaluation tasks.
        
        Args:
            graph: Graph sample to generate labels for
            label_type: Type of label to generate
                "primary_community": Primary community of each node
                "multi_label": Binary indicators for community membership
                "regression": Continuous value derived from memberships
            label_noise: Probability of random label noise
            
        Returns:
            Node labels
        """
        n_nodes = graph.n_nodes
        
        if label_type == "primary_community":
            # For each node, get the community with highest membership weight
            labels = np.zeros(n_nodes, dtype=int)
            
            for i in range(n_nodes):
                if label_noise > 0 and np.random.random() < label_noise:
                    # Assign random label
                    labels[i] = np.random.choice(graph.communities)
                else:
                    # Assign primary community
                    labels[i] = graph.communities[np.argmax(graph.membership_vectors[i])]
            
            return labels
            
        elif label_type == "multi_label":
            # For each node, generate binary indicators for community membership
            # thresholded by some minimum membership weight
            threshold = 0.2  # Minimum membership weight to count
            labels = np.zeros((n_nodes, len(graph.communities)), dtype=int)
            
            for i in range(n_nodes):
                if label_noise > 0 and np.random.random() < label_noise:
                    # Random multi-label
                    n_labels = np.random.randint(1, len(graph.communities) + 1)
                    random_communities = np.random.choice(
                        len(graph.communities), size=n_labels, replace=False
                    )
                    labels[i, random_communities] = 1
                else:
                    # Actual multi-label based on memberships
                    labels[i] = (graph.membership_vectors[i] > threshold).astype(int)
            
            return labels
            
        elif label_type == "regression":
            # Generate a continuous label as a weighted sum of community-specific values
            community_values = np.random.normal(0, 1, size=len(graph.communities))
            
            # Generate raw values as weighted combinations of community values
            raw_values = graph.membership_vectors @ community_values
            
            # Add noise if requested
            if label_noise > 0:
                raw_values += np.random.normal(0, label_noise, size=n_nodes)
                
            # Normalize to [0, 1] range
            min_val, max_val = raw_values.min(), raw_values.max()
            if max_val > min_val:
                labels = (raw_values - min_val) / (max_val - min_val)
            else:
                labels = np.zeros(n_nodes)
                
            return labels
            
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    
    def analyze_benchmark_parameters(
        self,
        pretrain_graphs: List[GraphSample],
        transfer_graphs: List[GraphSample]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze parameters for benchmark graph families.
        
        Args:
            pretrain_graphs: List of pretraining graphs
            transfer_graphs: List of transfer graphs
            
        Returns:
            Tuple of (pretrain_parameters_df, transfer_parameters_df)
        """
        from utils.parameter_analysis import analyze_graph_family
        
        # Analyze parameters
        pretrain_df = analyze_graph_family(pretrain_graphs)
        transfer_df = analyze_graph_family(transfer_graphs)
        
        return pretrain_df, transfer_df

    def split_train_val_test(
        self,
        graph: GraphSample,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Split nodes into training, validation and test sets.
        
        Args:
            graph: Graph to split
            train_ratio: Fraction of nodes for training
            val_ratio: Fraction of nodes for validation
            test_ratio: Fraction of nodes for testing
            stratify: Whether to stratify by primary community
            
        Returns:
            Dictionary with train/val/test indices
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n_nodes = graph.n_nodes
        indices = np.arange(n_nodes)
        
        if stratify:
            # Get primary community for each node
            labels = self.generate_node_labels(graph, label_type="primary_community")
            
            # Get unique communities and their counts
            unique_communities = np.unique(labels)
            
            # Split each community separately to maintain distribution
            train_idx, val_idx, test_idx = [], [], []
            
            for comm in unique_communities:
                comm_indices = indices[labels == comm]
                n_comm = len(comm_indices)
                
                # Calculate sizes
                n_train = int(train_ratio * n_comm)
                n_val = int(val_ratio * n_comm)
                
                # Shuffle indices
                np.random.shuffle(comm_indices)
                
                # Split
                train_idx.extend(comm_indices[:n_train])
                val_idx.extend(comm_indices[n_train:n_train+n_val])
                test_idx.extend(comm_indices[n_train+n_val:])
                
            # Convert to arrays
            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            test_idx = np.array(test_idx)
            
        else:
            # Simple random split
            np.random.shuffle(indices)
            n_train = int(train_ratio * n_nodes)
            n_val = int(val_ratio * n_nodes)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            
        return {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }
    
    def save_graph_data(
        self, 
        graph: GraphSample,
        directory: str,
        prefix: str = "graph",
        format: str = "networkx"
    ) -> None:
        """
        Save a graph and its metadata to disk.
        
        Args:
            graph: Graph to save
            directory: Directory to save in
            prefix: Filename prefix
            format: Output format (networkx, pyg, dgl)
        """
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if format == "networkx":
            import pickle
            # Save NetworkX graph
            with open(os.path.join(directory, f"{prefix}.gpickle"), "wb") as f:
                pickle.dump(graph.graph, f)
                
            # Save metadata
            metadata = {
                "universe_K": self.universe.K,
                "communities": graph.communities,
                "n_nodes": graph.n_nodes,
                "n_edges": graph.graph.number_of_edges(),
                "avg_degree": 2 * graph.graph.number_of_edges() / graph.n_nodes,
                "feature_dim": self.universe.feature_dim if self.universe.feature_dim > 0 else 0
            }
            
            with open(os.path.join(directory, f"{prefix}_meta.pkl"), "wb") as f:
                pickle.dump(metadata, f)
                
        elif format == "pyg":
            try:
                import torch
                from torch_geometric.data import Data
                
                # Convert to PyG format
                if graph.graph.number_of_edges() > 0:
                    edge_index = np.array(list(graph.graph.edges())).T
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                else:
                    # Handle empty graph case
                    edge_index = torch.tensor([[], []], dtype=torch.long)
                
                # Node features
                if graph.features is not None:
                    x = torch.tensor(graph.features, dtype=torch.float)
                else:
                    x = torch.ones((graph.n_nodes, 1), dtype=torch.float)
                    
                # Node labels (primary community)
                y = self.generate_node_labels(graph, label_type="primary_community")
                y = torch.tensor(y, dtype=torch.long)
                
                # Create PyG data object
                data = Data(x=x, edge_index=edge_index, y=y)
                
                # Add membership vectors as additional attribute
                data.memberships = torch.tensor(graph.membership_vectors, dtype=torch.float)
                
                # Save data
                torch.save(data, os.path.join(directory, f"{prefix}.pt"))
                
            except ImportError:
                print("PyTorch Geometric not installed. Falling back to NetworkX format.")
                self.save_graph_data(graph, directory, prefix, "networkx")
                
        elif format == "dgl":
            try:
                import torch
                import dgl
                
                # Create DGL graph
                g = dgl.graph(([], []))
                g.add_nodes(graph.n_nodes)
                
                # Add edges
                edges = list(graph.graph.edges())
                src, dst = zip(*edges)
                g.add_edges(src, dst)
                
                # Add node features
                if graph.features is not None:
                    g.ndata["feat"] = torch.tensor(graph.features, dtype=torch.float)
                else:
                    g.ndata["feat"] = torch.ones((graph.n_nodes, 1), dtype=torch.float)
                    
                # Add labels
                labels = self.generate_node_labels(graph, label_type="primary_community")
                g.ndata["label"] = torch.tensor(labels, dtype=torch.long)
                
                # Add membership vectors
                g.ndata["memberships"] = torch.tensor(graph.membership_vectors, dtype=torch.float)
                
                # Save graph
                dgl.save_graphs(os.path.join(directory, f"{prefix}.dgl"), [g])
                
            except ImportError:
                print("DGL not installed. Falling back to NetworkX format.")
                self.save_graph_data(graph, directory, prefix, "networkx")
                
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_feature_challenge_benchmark(
        self,
        n_nodes: int,
        n_communities: int,
        n_features: int,
        n_graphs: int,
        noise: float = 0.0,
        feature_noise: float = 0.0,
        feature_correlation: float = 0.5,
        feature_sparsity: float = 0.0,
        feature_type: str = "continuous",
        feature_challenge_type: str = "random",
        feature_challenge_strength: float = 0.5,
        seed: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_format: str = "npz"
    ) -> Dict[str, Any]:
        """
        Generate a benchmark dataset for testing feature robustness.
        
        Args:
            n_nodes: Number of nodes per graph
            n_communities: Number of communities
            n_features: Number of node features
            n_graphs: Number of graphs to generate
            noise: Random noise level for edge probabilities
            feature_noise: Random noise level for node features
            feature_correlation: Correlation between features and communities
            feature_sparsity: Sparsity level of node features
            feature_type: Type of features ("continuous" or "binary")
            feature_challenge_type: Type of feature challenge ("random", "targeted", "adversarial")
            feature_challenge_strength: Strength of the feature challenge (0 to 1)
            seed: Random seed for reproducibility
            save_dir: Directory to save the benchmark data
            save_format: Format to save the data ("npz" or "pickle")
            
        Returns:
            Dictionary containing the benchmark data
        """
        # ... existing code ...