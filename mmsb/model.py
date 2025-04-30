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


class GraphUniverse:
    """
    Represents the core generative universe from which graph instances are sampled.
    
    Attributes:
        K (int): Total number of community types in the universe
        P (np.ndarray): K×K edge probability matrix between communities
        feature_dim (int): Dimension of node features (if features are enabled)
        feature_signal (float): How strongly features correlate with community membership (0=random, 1=perfect)
        community_co_membership (np.ndarray): Probability of communities co-occurring (shape: K × K)
        feature_subtypes (List[np.ndarray]): Subtype feature prototypes for each community (if enabled)
        feature_similarity_matrix (np.ndarray): Controls feature similarity between communities
    """
    
    def __init__(
        self,
        K: int,
        P: Optional[np.ndarray] = None,
        feature_dim: int = 0,
        feature_signal: float = 1.0,
        community_co_membership: Optional[np.ndarray] = None,
        block_structure: str = "assortative",
        edge_density: float = 0.1,
        inter_community_density: float = 0.01,
        randomness_factor: float = 0.0,
        feature_structure: str = "distinct",
        feature_subtypes_per_community: int = 1,
        feature_mixing_strength: float = 0.5,
        feature_similarity_control: float = 0.7,
    ):
        """
        Initialize the GraphUniverse.
        
        Args:
            K: Number of communities in the universe
            P: Edge probability matrix (K × K). If None, generated based on block_structure
            feature_dim: Dimension of node features. If 0, no features are generated
            feature_signal: How strongly features correlate with community membership (0=random, 1=perfect)
            community_co_membership: Probability matrix for community co-membership
            block_structure: Structure of the probability matrix if P is None
            edge_density: Overall edge density within communities
            inter_community_density: Edge density between different communities
            randomness_factor: Amount of random variation to add to the probability matrix (0.0-1.0)
            feature_structure: Structure for feature prototypes ("distinct", "correlated", "hierarchical", "random")
            feature_subtypes_per_community: Number of feature subtypes within each community
            feature_mixing_strength: How strongly features are mixed across communities (0=no mixing, 1=strong mixing)
            feature_similarity_control: How similar/different features are between communities (0=very different, 1=very similar)
        """
        self.K = K
        
        # Initialize probability matrix if not provided
        if P is None:
            self.P = self._generate_probability_matrix(
                K, block_structure, edge_density, inter_community_density, randomness_factor
            )
        else:
            assert P.shape == (K, K), f"P should be a {K}×{K} matrix"
            self.P = P
            
        # Feature parameters
        self.feature_dim = feature_dim
        self.feature_signal = feature_signal
        
        # Generate base feature prototypes if features are enabled
        if feature_dim > 0:
            # Generate random vectors for each community
            prototypes = np.random.normal(0, 1, size=(K, feature_dim))
            
            if feature_dim >= K:
                # If we have enough dimensions, make them orthogonal using QR decomposition
                q, _ = np.linalg.qr(prototypes.T)
                self.feature_prototypes = q.T
            else:
                # If we don't have enough dimensions, just normalize the random vectors
                norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
                self.feature_prototypes = prototypes / norms
            
            # Verify we have the right number of prototypes
            assert self.feature_prototypes.shape[0] == K, f"Expected {K} prototypes, got {self.feature_prototypes.shape[0]}"
        else:
            self.feature_prototypes = None
                
        # Community co-membership probabilities
        if community_co_membership is None:
            self.community_co_membership = np.eye(K)  # Default: no overlap
        else:
            assert community_co_membership.shape == (K, K)
            self.community_co_membership = community_co_membership
            
        self.feature_variance = 0.1
        
        # Feature generation parameters
        self.feature_structure = feature_structure
        self.feature_subtypes_per_community = feature_subtypes_per_community
        self.feature_mixing_strength = feature_mixing_strength
        self.feature_similarity_control = feature_similarity_control
        
        if feature_dim > 0:
            # Generate feature similarity matrix
            self.feature_similarity_matrix = self._generate_feature_similarity_matrix(
                K, feature_mixing_strength, feature_structure
            )
        else:
            self.feature_similarity_matrix = None

    def _generate_probability_matrix(
        self, 
        K: int, 
        structure: str, 
        edge_density: float, 
        inter_density: float,
        randomness_factor: float = 0.0
    ) -> np.ndarray:
        """
        Generate a structured probability matrix based on the specified pattern with optional randomness.
        
        Args:
            K: Number of communities
            structure: Type of block structure
            edge_density: Edge density within communities
            inter_density: Edge density between communities
            randomness_factor: Amount of random noise to add (0.0-1.0)
            
        Returns:
            K × K probability matrix
        """
        P = np.zeros((K, K))
        
        if structure == "assortative":
            # Higher probabilities within communities, lower between
            np.fill_diagonal(P, edge_density)
            P[~np.eye(K, dtype=bool)] = inter_density
            
        elif structure == "disassortative":
            # Lower probabilities within communities, higher between
            np.fill_diagonal(P, inter_density)
            P[~np.eye(K, dtype=bool)] = edge_density
            
        elif structure == "core-periphery":
            # First community is the core, others are periphery
            core_size = max(1, K // 10)
            
            # Core-core connections
            P[:core_size, :core_size] = edge_density
            
            # Core-periphery connections
            P[core_size:, :core_size] = edge_density / 2
            P[:core_size, core_size:] = edge_density / 2
            
            # Periphery-periphery connections
            P[core_size:, core_size:] = inter_density
            
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
                        P[i, j] = np.random.uniform(inter_density, edge_density)
                    else:
                        # Off-diagonal blocks have varying probabilities
                        # but generally lower than diagonal blocks
                        P[i, j] = np.random.uniform(inter_density/2, edge_density/2)
                    
        else:
            # Random uniform probabilities
            P = np.random.uniform(low=inter_density, high=edge_density, size=(K, K))
        
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
                    off_diag_scale = inter_density / off_diag_avg
                    
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
        feature_correlation_with_topology: float = 0.0,
        feature_subtype_mixing: float = 0.5,
        feature_noise_heterogeneity: float = 0.0,
        feature_sparsity: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize and generate a graph sample.
        """
        self.universe = universe
        self.communities = sorted(communities)
        self.original_n_nodes = n_nodes
        self.min_component_size = min_component_size
        
        # Store feature generation parameters
        self.feature_correlation_with_topology = feature_correlation_with_topology
        self.feature_subtype_mixing = feature_subtype_mixing
        self.feature_noise_heterogeneity = feature_noise_heterogeneity
        self.feature_sparsity = feature_sparsity
        
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
                if hasattr(universe, 'feature_prototypes') and universe.feature_prototypes is not None:
                    self.features = self._generate_features(
                        self.membership_vectors,
                        universe.feature_prototypes,
                        universe.feature_signal
                    )
                else:
                    self.features = None
            else:
                self.features = None
                
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
        # Use pre-computed matrices if available
        if precomputed is None:
            precomputed = {}
        
        # Direct connections (fast matrix operation)
        direct_prob = memberships[node_i] @ P_sub @ memberships[node_j]
        
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
        
        # Combine probabilities and apply corrections
        edge_prob = direct_prob + indirect_prob
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
        indirect_influence: float = 0.1  # Add parameter here too
    ) -> sp.spmatrix:
        """
        Generate graph edges using vectorized probability calculation.
        Uses batching and pre-computation for speed.
        """
        n_nodes = memberships.shape[0]
        rows, cols, data = [], [], []
        
        # Pre-compute matrices for indirect connections
        precomputed = {}
        
        # Track statistics
        total_pairs = 0
        direct_connections = 0
        indirect_connections = 0
        
        # Process in batches for better memory usage
        batch_size = min(1000, n_nodes)
        for i in range(n_nodes):
            for j_start in range(i + 1, n_nodes, batch_size):
                j_end = min(j_start + batch_size, n_nodes)
                
                for j in range(j_start, j_end):
                    total_pairs += 1
                    
                    # Calculate edge probability
                    edge_prob = self._calculate_edge_probability_vectorized(
                        i, j, memberships, P_sub, degree_factors, noise, 
                        indirect_influence, precomputed
                    )
                    
                    # Sample edge
                    if np.random.random() < edge_prob:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([1, 1])
                        
                        # Track connection type
                        direct_prob = memberships[i] @ P_sub @ memberships[j]
                        if direct_prob > edge_prob / 2:
                            direct_connections += 1
                        else:
                            indirect_connections += 1
        
        # Create sparse adjacency matrix
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        return adj

    def _generate_edges_with_component_filtering(
        self, 
        memberships: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float
    ) -> sp.spmatrix:
        """
        Generate graph edges based on mixed-membership SBM with degree correction,
        filtering out small connected components instead of forcing connectivity.
        
        Args:
            memberships: Node-community membership vectors (n_nodes × n_communities)
            P_sub: Community edge probability submatrix
            degree_factors: Degree correction factors for nodes
            noise: Random noise level for edge probabilities
            
        Returns:
            Sparse adjacency matrix with small components removed
        """
        n_nodes = memberships.shape[0]
        
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


class MMSBBenchmark:
    """
    Generates a benchmark of graph instances for pretraining and transfer learning.
    
    This modified version implements:
    1. Better community connectivity assessment during sampling
    2. Component filtering instead of forced connectivity
    
    Attributes:
        universe (GraphUniverse): The graph universe from which instances are sampled
        graphs (List[GraphSample]): Generated graph instances
    """
    
    def __init__(
        self,
        K: int = 100,
        feature_dim: int = 64,
        feature_signal: float = 1.0,  # Add feature_signal parameter
        block_structure: str = "hierarchical",
        overlap_structure: str = "modular",
        edge_density: float = 0.1,
        inter_community_density: float = 0.01,
        overlap_density: float = 0.2,
        randomness_factor: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the benchmark generator.
        
        Args:
            K: Number of communities in the universe
            feature_dim: Dimension of node features (0 for no features)
            feature_signal: How strongly features correlate with community membership (0=random, 1=perfect)
            block_structure: Structure of the edge probability matrix
            overlap_structure: Structure of community overlaps
            edge_density: Edge density within communities
            inter_community_density: Edge density between communities
            overlap_density: Density of community overlaps
            randomness_factor: Amount of random variation in edge probabilities (0.0-1.0)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize the graph universe
        self.universe = GraphUniverse(
            K=K,
            feature_dim=feature_dim,
            feature_signal=feature_signal,  # Pass feature_signal
            block_structure=block_structure,
            edge_density=edge_density,
            inter_community_density=inter_community_density,
            randomness_factor=randomness_factor
        )
        
        # Generate community co-membership matrix
        co_membership = self.universe.generate_community_co_membership_matrix(
            overlap_density=overlap_density,
            structure=overlap_structure
        )
        self.universe.community_co_membership = co_membership
        
        # Initialize empty list of graphs
        self.graphs = []
    
    # Add the new community sampling method to the universe (monkey patching)
    def add_connected_community_sampling_to_universe(self):
        """Add the connected community sampling method to the universe object."""
        import types
        self.universe.sample_connected_community_subset = types.MethodType(
            self.sample_connected_community_subset, self.universe
        )
    
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
        min_component_size: int = 0,  # New parameter
        indirect_influence: float = 0.1  # Add indirect_influence parameter
    ) -> List[GraphSample]:
        """
        Generate a set of graphs for pretraining.
        
        Args:
            n_graphs: Number of graphs to generate
            min_communities: Minimum communities per graph
            max_communities: Maximum communities per graph
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            degree_heterogeneity: Controls degree variability
            edge_noise: Random noise added to edge probabilities
            sampling_method: Method for sampling community subsets
            similarity_bias: Bias towards similar or diverse communities
            min_connection_strength: Minimum edge probability between communities
            min_component_size: Minimum size for a component to be kept (0 keeps all components)
            indirect_influence: How strongly co-memberships influence edge formation (0-1)
            
        Returns:
            List of generated graph samples with components filtered by size
        """
        # Add the connected community sampling method if needed
        if not hasattr(self.universe, 'sample_connected_community_subset'):
            self.add_connected_community_sampling_to_universe()
            
        graphs = []
        failed_attempts = []
        
        for i in range(n_graphs):
            # Sample number of communities for this graph
            n_communities = np.random.randint(min_communities, max_communities + 1)
            
            # Sample subset of communities based on connection strength requirement
            try:
                if min_connection_strength > 0:
                    communities = self.universe.sample_connected_community_subset(
                        size=n_communities,
                        method=sampling_method,
                        similarity_bias=similarity_bias,
                        min_connection_strength=min_connection_strength
                    )
                else:
                    communities = self.universe.sample_community_subset(
                        size=n_communities,
                        method=sampling_method,
                        similarity_bias=similarity_bias
                    )
                
                if communities is None:
                    failed_attempts.append({
                        "graph_num": i + 1,
                        "error": "Community sampling returned None",
                        "n_communities": n_communities,
                        "sampling_method": sampling_method,
                        "min_connection_strength": min_connection_strength
                    })
                    continue
                    
            except Exception as e:
                failed_attempts.append({
                    "graph_num": i + 1,
                    "error": f"Community sampling failed: {str(e)}",
                    "n_communities": n_communities,
                    "sampling_method": sampling_method,
                    "min_connection_strength": min_connection_strength
                })
                continue
            
            # Sample number of nodes
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Generate graph with these communities
            try:
                graph = GraphSample(
                    universe=self.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=degree_heterogeneity,
                    edge_noise=edge_noise,
                    min_component_size=min_component_size,
                    indirect_influence=indirect_influence,
                    seed=None
                )
                
                # Only add graphs that have nodes after filtering
                if graph.n_nodes > 0:
                    graphs.append(graph)
                else:
                    failed_attempts.append({
                        "graph_num": i + 1,
                        "error": "Graph had 0 nodes after filtering",
                        "n_communities": len(communities),
                        "original_nodes": n_nodes,
                        "communities": communities
                    })
                    
            except Exception as e:
                failed_attempts.append({
                    "graph_num": i + 1,
                    "error": f"Graph creation failed: {str(e)}",
                    "n_communities": len(communities),
                    "n_nodes": n_nodes,
                    "communities": communities
                })
        
        # Only print debug information if there were failures
        if failed_attempts:
            print(f"\nFailed to generate {len(failed_attempts)} out of {n_graphs} graphs")
            print("\nFailure details:")
            for attempt in failed_attempts:
                print(f"\nGraph {attempt['graph_num']}:")
                for key, value in attempt.items():
                    if key != 'graph_num':
                        print(f"  {key}: {value}")
        
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
        min_component_size: int = 0,  # New parameter
        indirect_influence: float = 0.1  # Add indirect_influence parameter
    ) -> List[GraphSample]:
        """
        Generate transfer learning graphs with controlled distributional shift.
        
        Args:
            n_graphs: Number of transfer graphs to generate
            reference_graphs: List of already generated "training" graphs
            transfer_mode: Type of transfer ("new_combinations", "rare_communities", "novel_communities")
            transfer_difficulty: Controls how challenging the transfer task is (0-1)
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            degree_heterogeneity: Controls degree variability
            edge_noise: Random noise added to edge probabilities
            min_connection_strength: Minimum edge probability between communities
            min_component_size: Minimum size for a component to be kept (0 keeps all components)
            indirect_influence: How strongly co-memberships influence edge formation (0-1)
            
        Returns:
            List of transfer graph samples with components filtered by size
        """
        # Add the connected community sampling method if needed
        if not hasattr(self.universe, 'sample_connected_community_subset'):
            self.add_connected_community_sampling_to_universe()
            
        # Analyze community distribution in reference graphs
        community_counts = np.zeros(self.universe.K)
        for graph in reference_graphs:
            for comm in graph.communities:
                community_counts[comm] += 1
                
        # Normalize to get frequencies
        if len(reference_graphs) > 0:
            community_freq = community_counts / len(reference_graphs)
        else:
            community_freq = np.ones(self.universe.K) / self.universe.K
            
        transfer_graphs = []
        
        while len(transfer_graphs) < n_graphs:
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            if transfer_mode == "new_combinations":
                # Generate graphs with new combinations of seen communities
                # Select some common and some rare communities
                n_communities = len(reference_graphs[0].communities) if reference_graphs else 10
                
                # Sort communities by frequency
                sorted_communities = np.argsort(community_freq)
                
                # Select a mix of common and rare communities
                # Higher transfer_difficulty means more rare communities
                common_ratio = 1.0 - transfer_difficulty
                n_common = max(1, int(common_ratio * n_communities))
                n_rare = n_communities - n_common
                
                common_pool = sorted_communities[-int(self.universe.K * 0.3):]  # Top 30%
                rare_pool = sorted_communities[:int(self.universe.K * 0.7)]     # Bottom 70%
                
                common_communities = np.random.choice(common_pool, size=n_common, replace=False)
                rare_communities = np.random.choice(rare_pool, size=n_rare, replace=False)
                
                communities = np.concatenate([common_communities, rare_communities]).tolist()
                
            elif transfer_mode == "rare_communities":
                # Focus on communities that were rare in training
                n_communities = len(reference_graphs[0].communities) if reference_graphs else 10
                
                # Get the rarest communities
                rarest_idx = np.argsort(community_freq)
                num_rare = int(self.universe.K * transfer_difficulty)
                rare_pool = rarest_idx[:num_rare]
                
                # Ensure we have enough communities to sample from
                if len(rare_pool) < n_communities:
                    remaining = np.setdiff1d(np.arange(self.universe.K), rare_pool)
                    rare_pool = np.concatenate([rare_pool, remaining])
                
                communities = np.random.choice(rare_pool, size=min(n_communities, len(rare_pool)), replace=False).tolist()
                
            elif transfer_mode == "novel_communities":
                # Include communities never or rarely seen during training
                n_communities = len(reference_graphs[0].communities) if reference_graphs else 10
                
                # Sort communities by frequency
                sorted_communities = np.argsort(community_freq)
                
                # Take some unseen/very rare communities
                n_novel = max(1, int(transfer_difficulty * n_communities))
                novel_communities = sorted_communities[:n_novel].tolist()
                
                # Take some familiar communities
                n_familiar = n_communities - n_novel
                familiar_pool = sorted_communities[n_novel:]
                familiar_communities = np.random.choice(familiar_pool, size=min(n_familiar, len(familiar_pool)), replace=False).tolist()
                
                communities = novel_communities + familiar_communities
                
            else:
                # Default to a random subset
                n_communities = len(reference_graphs[0].communities) if reference_graphs else 10
                communities = self.universe.sample_community_subset(
                    size=n_communities,
                    method="random"
                )
            
            # Check and enhance connectivity if needed
            if min_connection_strength > 0:
                communities = self.universe.sample_connected_community_subset(
                    size=len(communities),
                    method="correlated",
                    min_connection_strength=min_connection_strength,
                    existing_communities=communities[:min(3, len(communities))]
                )
            
            # Generate graph with these communities
            graph = GraphSample(
                universe=self.universe,
                communities=communities,
                n_nodes=n_nodes,
                degree_heterogeneity=degree_heterogeneity,
                edge_noise=edge_noise,
                min_component_size=min_component_size,  # Add parameter
                indirect_influence=indirect_influence,  # Add indirect_influence parameter
                seed=None
            )
            
            # Only add graphs that have nodes after filtering
            if graph.n_nodes > 0:
                transfer_graphs.append(graph)
            
        return transfer_graphs

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