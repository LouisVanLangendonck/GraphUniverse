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
from mmsb.feature_regimes import (
    SimplifiedFeatureGenerator,
    NeighborhoodFeatureAnalyzer,
    FeatureClusterLabelGenerator
)
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import community as community_louvain
import warnings
from itertools import combinations

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
        block_structure: str = "assortative",
        edge_density: float = 0.1,
        homophily: float = 0.8,
        randomness_factor: float = 0.0,
        # Feature generation parameters
        cluster_count_factor: float = 1.0,  # Number of clusters relative to communities
        center_variance: float = 1.0,       # Separation between cluster centers
        cluster_variance: float = 0.1,      # Spread within each cluster
        assignment_skewness: float = 0.0,   # If some clusters are used more frequently
        community_exclusivity: float = 1.0, # How exclusively clusters map to communities
        # Degree center parameters
        degree_center_method: str = "linear",  # How to generate degree centers ("linear", "random", "shuffled")
        seed: Optional[int] = None,
        # Community density parameters
        community_density_variation: float = 0.0,      # 0-1: amount of density variation
        # Community co-occurrence homogeneity
        community_cooccurrence_homogeneity: float = 1.0,  # 0-1: how homogeneous the co-occurrence of communities is

        # Triangle density parameters
        triangle_density: float = 0.0,
        triangle_community_relation_homogeneity: float = 1.0,
    ):
        """
        Initialize a graph universe with K communities and optional feature generation.
        
        Args:
            K: Number of communities
            P: Optional probability matrix (if None, will be generated)
            feature_dim: Dimension of node features
            block_structure: Type of block structure ("assortative", "disassortative", "mixed")
            edge_density: Base edge density
            homophily: Strength of community-based edge formation
            randomness_factor: Amount of randomness in edge formation
            cluster_count_factor: Number of clusters relative to communities (0.1 to 4.0)
            center_variance: Separation between cluster centers
            cluster_variance: Spread within each cluster
            assignment_skewness: If some clusters are used more frequently (0.0 to 1.0)
            community_exclusivity: How exclusively clusters map to communities (0.0 to 1.0)
            degree_center_method: How to generate degree centers ("linear", "random", "shuffled")
            community_density_variation: Amount of density variation (automatically coupled to degree centers. So higher density variation means higher densities for nodes with higher degree variation)
            community_cooccurrence_homogeneity: Controls community co-occurrence patterns (0-1)
                1.0 = homogeneous (all communities equally likely to co-occur)
                0.0 = heterogeneous (some communities prefer to co-occur with specific others)
            seed: Random seed for reproducibility
        """
        self.K = K
        self.feature_dim = feature_dim
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided probability matrix
        if P is None:
            self.P = self._generate_probability_matrix(
                K, block_structure, edge_density, homophily, randomness_factor
            )
        else:
            self.P = P
            
        # Initialize feature generator if features are enabled
        if feature_dim > 0:
            self.feature_generator = SimplifiedFeatureGenerator(
                universe_K=K,
                feature_dim=feature_dim,
                cluster_count_factor=cluster_count_factor,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
                assignment_skewness=assignment_skewness,
                community_exclusivity=community_exclusivity,
                seed=seed
            )
        else:
            self.feature_generator = None
        
        # Store parameters
        self.edge_density = edge_density
        self.homophily = homophily
        self.block_structure = block_structure
        self.randomness_factor = randomness_factor
        self.feature_variance = 0.1
        self.feature_similarity_matrix = None
        # Store regime parameters
        self.intra_community_regime_similarity = 0.8
        self.inter_community_regime_similarity = 0.2

        # Store community density parameters
        self.community_density_variation = community_density_variation

        # Generate degree centers based on method
        if degree_center_method == "linear":
            # Linear spacing from -1 to 1
            self.degree_centers = np.linspace(-1, 1, K)
        elif degree_center_method == "random":
            # Random uniform distribution
            self.degree_centers = np.random.uniform(-1, 1, K)
        elif degree_center_method == "shuffled":
            # Linear spacing but shuffled
            self.degree_centers = np.linspace(-1, 1, K)
            np.random.shuffle(self.degree_centers)
        else:
            raise ValueError(f"Unknown degree center method: {degree_center_method}")
    
        # Apply community density variations if requested
        if community_density_variation > 0:
            self.P = self._apply_community_density_variation(
                self.P, 
                community_density_variation,
            )
        
        # Generate community co-occurrence matrix
        self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity
        self.community_cooccurrence_matrix = self._generate_cooccurrence_matrix(K, community_cooccurrence_homogeneity, seed)

        # Triangle density parameters
        self.triangle_density = triangle_density
        self.triangle_community_relation_homogeneity = triangle_community_relation_homogeneity

        # Generate community triangle propensities
        self.community_triangle_propensities = self._generate_community_triangle_propensities()

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
        
        # Add randomness if requested, but only clip negative values to zero (do not clip to one)
        if randomness_factor > 0:
            min_val = min(p, q)
            noise = np.random.normal(0, randomness_factor * min_val, size=(K, K))
            P = P + noise
            # Only clip negative values to zero
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
    
    def sample_connected_community_subset(
        self,
        size: int,
        seed: Optional[int] = None,
        use_cooccurrence: bool = True
    ) -> List[int]:
        """
        Sample a subset of communities using co-occurrence patterns.
        
        Args:
            size: Number of communities to sample
            existing_communities: Optional list of communities to condition on
            seed: Random seed for reproducibility
            use_cooccurrence: Whether to use co-occurrence matrix for sampling
            
        Returns:
            List of sampled community indices
        """
        if seed is not None:
            np.random.seed(seed)
        
        K = self.K
        size = min(size, K)
        
        if not use_cooccurrence or self.community_cooccurrence_homogeneity == 1.0:
            # Random sampling
            return np.random.choice(self.K, size=size, replace=False).tolist()
        
        # Start with a random seed community
        first_community = np.random.choice(K)
        result = {first_community}
        remaining_size = size - 1
        
        # Iteratively add communities based on co-occurrence probabilities
        while remaining_size > 0 and len(result) < K:
            # Calculate sampling probabilities based on co-occurrence with existing communities
            remaining_communities = list(set(range(K)) - result)
            if not remaining_communities:
                break
            
            # For each remaining community, calculate its average co-occurrence with selected ones
            cooccurrence_scores = np.zeros(len(remaining_communities))
            for i, candidate in enumerate(remaining_communities):
                # Average co-occurrence probability with all selected communities
                avg_cooccurrence = np.mean([
                    self.community_cooccurrence_matrix[candidate, selected] 
                    for selected in result
                ])
                cooccurrence_scores[i] = avg_cooccurrence
            
            # Convert scores to probabilities
            if np.sum(cooccurrence_scores) > 0:
                probabilities = cooccurrence_scores / np.sum(cooccurrence_scores)
            else:
                # Fallback to uniform if all scores are zero
                probabilities = np.ones(len(remaining_communities)) / len(remaining_communities)
            
            # Sample next community
            next_idx = np.random.choice(len(remaining_communities), p=probabilities)
            next_community = remaining_communities[next_idx]
            result.add(next_community)
            remaining_size -= 1
        
        return list(result)
    
    def _apply_community_density_variation(
        self,
        P: np.ndarray,
        community_density_variation: float = 0.0
    ) -> np.ndarray:
        """
        Apply degree-coupled community density variations to the universe P matrix.
        Communities with higher degree centers get higher overall connection probabilities.
        
        Args:
            P: Universe probability matrix (K x K)
            community_density_variation: Amount of variation (0-1)
            
        Returns:
            Modified probability matrix with community density variations
        """
        if community_density_variation == 0.0:
            return P
        
        K = P.shape[0]
        
        # Generate density multipliers based on degree center hierarchy
        if np.std(self.degree_centers) > 0:
            # Normalize degree centers to [-1, 1] range
            normalized_centers = (self.degree_centers - np.mean(self.degree_centers)) / np.std(self.degree_centers)
            normalized_centers = np.clip(normalized_centers, -1, 1)
            
            # Create multipliers: high degree centers → high density
            density_multipliers = 1.0 + community_density_variation * normalized_centers
        else:
            # If all degree centers are the same, no variation
            density_multipliers = np.ones(K)
        
        # Ensure multipliers are positive and reasonable
        density_multipliers = np.clip(density_multipliers, 0.3, 2.0)
        
        # Apply multipliers to each community's row (outgoing connections)
        P_modified = P.copy()
        for i in range(K):
            P_modified[i, :] *= density_multipliers[i]
        
        # Store the multipliers for analysis
        self.community_density_multipliers = density_multipliers
        
        # DON'T clip to [0,1] here - let GraphSample handle that during scaling
        # This preserves the relative differences in the signal
        
        return P_modified

    def _generate_cooccurrence_matrix(self, K: int, homogeneity: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate symmetric community co-occurrence matrix.
        
        Args:
            K: Number of communities
            homogeneity: 1.0 = uniform co-occurrence, 0.0 = heterogeneous patterns
            seed: Random seed
            
        Returns:
            Symmetric K x K co-occurrence probability matrix
        """
        if seed is not None:
            np.random.seed(seed)
        
        if homogeneity == 1.0:
            # Perfectly homogeneous - all pairs equally likely
            matrix = np.ones((K, K)) / K
            np.fill_diagonal(matrix, 1.0)  # Self-occurrence is always 1
            return matrix
        
        # Generate heterogeneous matrix
        # Start with base uniform probability
        base_prob = 1.0 / K
        
        # Generate random variations
        # Use narrow normal distribution for heterogeneous patterns
        variance = (1.0 - homogeneity) * 0.5  # Scale variance with heterogeneity
        
        # Generate upper triangle of matrix (excluding diagonal)
        upper_triangle = np.random.normal(base_prob, variance, size=(K, K))
        
        # Make it symmetric
        matrix = np.triu(upper_triangle, k=1) + np.triu(upper_triangle, k=1).T
        
        # Set diagonal to 1.0 (self-occurrence)
        np.fill_diagonal(matrix, 1.0)
        
        # Ensure all values are positive and reasonable
        matrix = np.clip(matrix, 0.01, 1.0)
        
        # Normalize rows to maintain proper probabilities
        # Each row should sum to something reasonable relative to K
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        matrix = matrix / row_sums * K * base_prob * 2  # Scale to reasonable range
        matrix = np.clip(matrix, 0.01, 1.0)
        
        return matrix

    def _generate_community_triangle_propensities(self) -> np.ndarray:
        """Generate triangle propensity for each community."""
        if self.triangle_community_relation_homogeneity == 1.0:
            # Homogeneous: all communities equally likely
            return np.ones(self.K)
        else:
            # Heterogeneous: some communities much more triangle-prone
            # Use Dirichlet to create skewed distribution
            alpha = self.triangle_community_relation_homogeneity * 10 + 0.1
            propensities = np.random.dirichlet(np.ones(self.K) * alpha) * self.K
            return propensities
    
class GraphSample:
    """
    Represents a single graph instance sampled from the GraphUniverse.
    
    This modified version implements both standard DC-SBM and the new
    Degree-Community-Coupled Corrected SBM (DCCC-SBM).
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        num_communities: int,
        n_nodes: int,
        min_component_size: int,
        degree_heterogeneity: float,
        edge_noise: float,
        feature_regime_balance: float,
        target_homophily: Optional[float],
        target_density: Optional[float],
        use_configuration_model: bool,
        degree_distribution: str,
        power_law_exponent: Optional[float],
        target_avg_degree: Optional[float],
        max_mean_community_deviation: float,
        max_max_community_deviation: float,
        max_parameter_search_attempts: int,
        parameter_search_range: float,
        min_edge_density: float,
        max_retries: int,
        seed: Optional[int] = None,
        exact_filtered_graph: Optional[Dict[str, Any]] = None,
        config_model_params: Optional[dict] = None,
        # DCCC-SBM parameters
        use_dccc_sbm: bool = False,
        community_imbalance: float = 0.0,
        degree_separation: float = 0.5,
        dccc_global_degree_params: Optional[dict] = None,
        degree_method: str = "standard",
        disable_deviation_limiting: bool = False,
        # Optional Parameter for user-defined communuties to be sampled
        user_defined_communities: Optional[List[int]] = None
    ):
        """
        Initialize and generate a graph sample.
        
        Added DCCC-SBM parameters:
        - use_dccc_sbm: Whether to use the Distribution-Community-Coupled Corrected SBM
        - community_imbalance: Controls how imbalanced community sizes are (0-1)
        - degree_distribution_overlap: Controls how much degree distributions overlap (0-1)
        - dccc_global_degree_params: Parameters for the global degree distribution
        """
        # Store additional DCCC-SBM parameters
        self.use_dccc_sbm = use_dccc_sbm
        self.community_imbalance = community_imbalance
        self.degree_separation = degree_separation
        self.dccc_global_degree_params = dccc_global_degree_params or {}
        self.disable_deviation_limiting = disable_deviation_limiting  # Store the parameter
        
        # Original initialization code with modifications...
        self.timing_info = {}
        total_start = time.time()

        self.universe = universe

        # Sample communities from universe
        if user_defined_communities is not None:
            self.communities = user_defined_communities
        else:
            self.communities = universe.sample_connected_community_subset(
                num_communities,
                seed=seed,
                use_cooccurrence=True
            )
        
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

        # Store community deviation parameters as instance attributes
        self.max_mean_community_deviation = max_mean_community_deviation
        self.max_max_community_deviation = max_max_community_deviation
        self.min_edge_density = min_edge_density
        self.max_retries = max_retries

        # Create mapping between local community indices and universe community IDs
        self.community_id_mapping = {i: comm_id for i, comm_id in enumerate(sorted(self.communities))}
        self.reverse_community_id_mapping = {comm_id: i for i, comm_id in self.community_id_mapping.items()}

        # Initialize generation method and parameters
        self.generation_method = "standard"
        self.generation_params = {
            "degree_heterogeneity": degree_heterogeneity,
            "edge_noise": edge_noise,
            "max_mean_community_deviation": max_mean_community_deviation,
            "max_max_community_deviation": max_max_community_deviation
        }
        
        # If DCCC-SBM is enabled, update generation method
        if use_dccc_sbm:
            self.generation_method = "dccc_sbm"
            self.generation_params.update({
                "community_imbalance": community_imbalance,
                "degree_separation": degree_separation,
                "degree_distribution_type": degree_distribution,
            })
            if degree_distribution == "power_law":
                self.generation_params["power_law_exponent"] = power_law_exponent
                
        elif use_configuration_model and degree_distribution in ["power_law", "exponential", "uniform"]:
            self.generation_method = degree_distribution
            if degree_distribution == "power_law":
                self.generation_params = {
                    "power_law_exponent": power_law_exponent,
                    "target_avg_degree": target_avg_degree,
                    "max_mean_community_deviation": max_mean_community_deviation,
                    "max_max_community_deviation": max_max_community_deviation
                }
            elif degree_distribution == "exponential":
                self.generation_params = {
                    "rate": getattr(self, 'rate', 0.5),  # This one can have a default as it's method-specific
                    "target_avg_degree": target_avg_degree,
                    "max_mean_community_deviation": max_mean_community_deviation,
                    "max_max_community_deviation": max_max_community_deviation
                }
            elif degree_distribution == "uniform":
                self.generation_params = {
                    "min_factor": getattr(self, 'min_factor', 0.5),  # This one can have a default as it's method-specific
                    "max_factor": getattr(self, 'max_factor', 1.5),  # This one can have a default as it's method-specific
                    "target_avg_degree": target_avg_degree,
                    "max_mean_community_deviation": max_mean_community_deviation,
                    "max_max_community_deviation": max_max_community_deviation
                }

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Time: Extract and scale probability matrix
        start = time.time()
        # Extract the submatrix of the probability matrix for these communities
        K_sub = len(self.communities)
        self.P_sub = np.zeros((K_sub, K_sub))
        for i, ci in enumerate(sorted(self.communities)):
            for j, cj in enumerate(sorted(self.communities)):
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
        if self.use_dccc_sbm:
            # For DCCC-SBM, use the imbalanced membership generation
            self.community_labels = self._generate_memberships_with_imbalance(
                n_nodes, K_sub, self.community_imbalance)
        else:
            # Standard membership generation
            self.community_labels = self._generate_memberships(n_nodes, K_sub)  # Now returns 1D array

        # Create a new array that maps the community labels to the universe community IDs
        self.community_labels_universe_level = np.array([self.community_id_mapping[idx] for idx in self.community_labels])
        
        self.timing_info['memberships'] = time.time() - start
        
        # Time: Generate degree factors
        start = time.time()
        if self.use_dccc_sbm:
            # For DCCC-SBM, generate community-coupled degree factors
            global_degree_params = {}
            if degree_distribution == "power_law":
                global_degree_params = {
                    "exponent": power_law_exponent,
                    "x_min": 1.0
                }
            elif degree_distribution == "exponential":
                global_degree_params = {
                    "rate": getattr(self, 'rate', 0.5)
                }
            elif degree_distribution == "uniform":
                global_degree_params = {
                    "min_degree": getattr(self, 'min_factor', 0.5),
                    "max_degree": getattr(self, 'max_factor', 1.5)
                }
                
            # Update with any user-provided parameters
            if dccc_global_degree_params:
                global_degree_params.update(dccc_global_degree_params)
                
            # Store the degree method
            self.degree_method = degree_method
            
            # Generate community-specific degree factors using improved method
            self.degree_factors = self._generate_community_degree_factors_improved(
                self.community_labels,
                degree_distribution,
                degree_separation,  # Changed from degree_distribution_overlap
                global_degree_params
            )
            edge_noise = 0.0
        elif self.use_configuration_model:
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
            self.adjacency = self._generate_edges_configuration(
                self.community_labels,
                self.P_sub,
                self.degree_factors,
                edge_noise,
                min_edge_density=self.min_edge_density,
                max_retries=self.max_retries,
                max_mean_community_deviation=self.max_mean_community_deviation,
                max_max_community_deviation=self.max_max_community_deviation,
                max_parameter_search_attempts=self.max_parameter_search_attempts,
                parameter_search_range=self.parameter_search_range
            )
        else:
            # Standard DC-SBM edge generation
            self.adjacency = self._generate_edges(
                self.community_labels,
                self.P_sub,
                self.degree_factors,
                edge_noise
            )
        self.timing_info['edge_generation'] = time.time() - start

        # Time: Generate triangles
        if self.universe.triangle_density > 0:
            start = time.time()
            self._enhance_triangles()
            self.timing_info['triangle_enhancement'] = time.time() - start

        # Create initial NetworkX graph
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)
        
        # Time: Component filtering
        start = time.time()
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
                        primary_comm = self.communities[self.community_labels[node]]
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

            # Update community labels at universe level
            self.community_labels_universe_level = np.array([self.community_id_mapping[idx] for idx in self.community_labels])
            
            # Update adjacency matrix
            self.adjacency = nx.adjacency_matrix(self.graph)
            
            # Now check deviations after all filtering and reconstruction is done
            if self.n_nodes > 0:  # Only check if we have nodes
                deviations = self._calculate_community_deviations(
                    self.graph,
                    self.community_labels,
                    self.P_sub
                )
                mean_deviation = deviations["mean_deviation"]
                max_deviation = deviations["max_deviation"]
                
                if not self.disable_deviation_limiting:
                    if mean_deviation > self.max_mean_community_deviation:
                        raise ValueError(f"Graph exceeds mean community deviation limit: {mean_deviation:.4f} > {self.max_mean_community_deviation:.4f}")
                    if max_deviation > self.max_max_community_deviation:
                        raise ValueError(f"Graph exceeds maximum community deviation limit: {max_deviation:.4f} > {self.max_max_community_deviation:.4f}")
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
                self.node_clusters = None
                self.neighborhood_analyzer = None
                self.label_generator = None
        self.timing_info['graph_reconstruction'] = time.time() - start
        
        # Time: Feature generation
        start = time.time()
        if universe.feature_dim > 0:
            # Get community assignments and map to universe community IDs
            local_community_assignments = self.community_labels
            universe_community_assignments = np.array(self.community_labels_universe_level)
            
            # Generate node clusters based on universe community assignments
            self.node_clusters = universe.feature_generator.assign_node_clusters(universe_community_assignments)
            
            # Generate features based on node clusters
            self.features = universe.feature_generator.generate_node_features(self.node_clusters)
            
            # Initialize these as None - they will be computed on demand
            self.neighborhood_analyzer = None
            self.label_generator = None
            self.node_labels = None
        else:
            self.features = None
            self.node_clusters = None
            self.neighborhood_analyzer = None
            self.label_generator = None
            self.node_labels = None
        self.timing_info['feature_generation'] = time.time() - start
        
        # Store total time
        self.timing_info['total'] = time.time() - total_start

    def _add_node_attributes(self):
        """Add node attributes to the graph."""
        for i, node in enumerate(self.graph.nodes()):
            # Get community label and map to universe community ID
            local_comm_idx = self.community_labels[i]
            universe_comm_id = self.community_id_mapping[local_comm_idx]
            
            # Create node attributes dictionary
            node_attrs = {
                "community": int(universe_comm_id),  # Store the actual universe community ID
                "degree_factor": float(self.degree_factors[i])
            }
            
            # Add features if available
            if self.features is not None:
                node_attrs["features"] = self.features[i].tolist()
                
            # Add cluster information if available
            if self.node_clusters is not None:
                node_attrs["feature_cluster"] = int(self.node_clusters[i])
                
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

    def _generate_memberships_with_imbalance(
        self, n_nodes: int, K_sub: int, community_imbalance: float = 0.0
    ) -> np.ndarray:
        """
        Generate community assignments with controlled imbalance.
        
        Args:
            n_nodes: Number of nodes
            K_sub: Number of communities
            community_imbalance: Controls how imbalanced communities are (0-1)
                0 = perfectly balanced, 1 = maximally imbalanced
                
        Returns:
            Array of community labels
        """
        if community_imbalance == 0.0:
            # Balanced communities - equal probability
            probs = np.ones(K_sub) / K_sub
        else:
            # Generate imbalanced distribution using Dirichlet with concentration parameter
            # Lower alpha = more imbalanced
            alpha = max(0.01, (1.0 - community_imbalance) * 10)  # Map 0-1 to 10-0.01
            probs = np.random.dirichlet(np.ones(K_sub) * alpha)
        
        # Sample from categorical distribution
        labels = np.random.choice(K_sub, size=n_nodes, p=probs)
        
        return labels

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
    
    def _generate_community_degree_factors_improved(
        self,
        community_labels: np.ndarray,
        degree_distribution_type: str,
        degree_separation: float,
        global_degree_params: dict,
    ) -> np.ndarray:
        n_nodes = len(community_labels)
        degree_factors = np.zeros(n_nodes)

        # 1. Sample global degree distribution
        if degree_distribution_type == "power_law":
            exponent = global_degree_params.get("exponent", 2.5)
            raw_degrees = np.random.pareto(exponent, size=n_nodes) + 1
        elif degree_distribution_type == "exponential":
            rate = global_degree_params.get("rate", 1.0)
            raw_degrees = np.random.exponential(scale=1/rate, size=n_nodes)
        elif degree_distribution_type == "uniform":
            low = global_degree_params.get("min_degree", 1.0)
            high = global_degree_params.get("max_degree", 10.0)
            raw_degrees = np.random.uniform(low, high, size=n_nodes)
        else:
            raise ValueError("Unknown distribution type")

        sorted_degrees = np.sort(raw_degrees)
        assigned_indices = np.zeros(n_nodes, dtype=bool)

        # Get unique communities and their universe IDs
        local_comm_ids = np.unique(community_labels)
        K = len(local_comm_ids)
        total_nodes = n_nodes

        # Get universe degree centers for our communities
        universe_degree_centers = np.array([
            self.universe.degree_centers[self.community_id_mapping[local_comm_id]]
            for local_comm_id in local_comm_ids
        ])

        # Order communities by universe degree center (lowest to highest)
        comm_order = np.argsort(universe_degree_centers)
        ordered_comms = local_comm_ids[comm_order]

        # Decide window width:
        # At separation=0: full width; at 1: just enough for community size
        min_window_fraction = 1.0 / K  # Minimum window size (just fits community size)
        window_fraction = min_window_fraction + (1.0 - min_window_fraction) * (1 - degree_separation)
        window_width = int(np.round(window_fraction * total_nodes))

        # Center positions along the degree spectrum (spread equally between 0 and n_nodes-1)
        if K == 1:
            centers = [total_nodes // 2]
        else:
            centers = np.linspace(window_width // 2, total_nodes - window_width // 2, K, dtype=int)

        # For each community, assign its nodes to available degree indices within its window
        for comm_idx, comm in enumerate(ordered_comms):
            nodes = np.where(community_labels == comm)[0]
            n_comm = len(nodes)

            # Determine window bounds
            c = centers[comm_idx]
            window_start = max(0, c - window_width // 2)
            window_end = min(total_nodes, c + window_width // 2)
            available_in_window = [i for i in range(window_start, window_end) if not assigned_indices[i]]

            # If not enough available in window (can happen with large overlap/small separation), expand outwards
            if len(available_in_window) < n_comm:
                extras_needed = n_comm - len(available_in_window)
                # Pick random from the rest
                extra_indices = [i for i in range(total_nodes) if not assigned_indices[i] and i not in available_in_window]
                if extras_needed > 0 and len(extra_indices) >= extras_needed:
                    available_in_window += list(np.random.choice(extra_indices, size=extras_needed, replace=False))
                else:
                    # Fallback: take whatever is left, even if fewer
                    available_in_window += extra_indices

            # Randomly assign indices in window to nodes
            chosen_indices = np.random.choice(available_in_window, size=n_comm, replace=False)
            for node, deg_idx in zip(nodes, chosen_indices):
                degree_factors[node] = sorted_degrees[deg_idx]
                assigned_indices[deg_idx] = True

        # Normalize to mean 1 for consistency
        degree_factors = degree_factors / degree_factors.mean()
        return degree_factors

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
        Generate node features using the new SimplifiedFeatureGenerator.
        
        Args:
            memberships: Node community assignments
            prototypes: Not used in new implementation
            feature_signal: Not used in new implementation
            
        Returns:
            np.ndarray: Node features
        """
        if self.universe.feature_generator is None:
            return np.zeros((len(memberships), 0))
            
        # Get community assignments for each node
        community_assignments = np.argmax(memberships, axis=1)
        
        # Generate node clusters based on community assignments
        node_clusters = self.universe.feature_generator.assign_node_clusters(community_assignments)
        
        # Generate features based on node clusters
        features = self.universe.feature_generator.generate_node_features(node_clusters)
        
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
        label_generator = FeatureClusterLabelGenerator(
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
                node_regimes=self.node_clusters,  # Use node_clusters instead of node_regimes
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
            self.label_generator = FeatureClusterLabelGenerator(
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
        """
        if target_avg_degree is None:
            # Calculate from target density
            target_avg_degree = self.target_density * (n_nodes - 1)
        if distribution == "power_law":
            # Generate from power law distribution
            # Use power_law_exponent directly without heterogeneity adjustment
            exponent = self.config_model_params.get('power_law_exponent', power_law_exponent)
            factors = np.random.pareto(exponent - 1, size=n_nodes) + 1
        elif distribution == "exponential":
            # Generate from exponential distribution
            rate = self.config_model_params.get('rate', 0.5)
            factors = np.random.exponential(1/rate, size=n_nodes) + 1  # Add 1 to ensure minimum degree of 1
        elif distribution == "uniform":
            # Generate from uniform distribution
            min_factor = self.config_model_params.get('min_factor', 0.5)
            max_factor = self.config_model_params.get('max_factor', 1.5)
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
        min_edge_density: float = None,  # Remove default values
        max_retries: int = None,
        max_mean_community_deviation: float = None,  # Remove default values
        max_max_community_deviation: float = None,
        max_parameter_search_attempts: int = None,
        parameter_search_range: float = None
    ) -> Optional[sp.spmatrix]:
        """
        Generate edges using the configuration model approach.
        """
        # ... existing code until deviation check ...

        # 3. NOW check deviations on filtered graph
        if not self.disable_deviation_limiting:
            deviations = self._calculate_community_deviations(
                filtered_graph,
                filtered_labels,
                P_sub
            )
            mean_deviation = deviations["mean_deviation"]
            max_deviation = deviations["max_deviation"]
            
            # 4. If deviations are within limits, we've found a valid graph
            if mean_deviation <= max_mean_community_deviation and max_deviation <= max_max_community_deviation:
                # Store the EXACT graph and all its properties
                self.graph = filtered_graph.copy()  # Make a deep copy
                self.community_labels = filtered_labels.copy()
                self.degree_factors = adjusted_factors[kept_nodes].copy()
                self.adjacency = nx.adjacency_matrix(filtered_graph)
                self.generation_method = self.degree_distribution
                self.generation_params = current_params.copy()
                self.node_map = node_map.copy()
                self.reverse_node_map = {new: old for old, new in node_map.items()}
                
                return self.adjacency
        else:
            # Skip deviation check and accept the graph
            self.graph = filtered_graph.copy()  # Make a deep copy
            self.community_labels = filtered_labels.copy()
            self.degree_factors = adjusted_factors[kept_nodes].copy()
            self.adjacency = nx.adjacency_matrix(filtered_graph)
            self.generation_method = self.degree_distribution
            self.generation_params = current_params.copy()
            self.node_map = node_map.copy()
            self.reverse_node_map = {new: old for old, new in node_map.items()}
            
            return self.adjacency

    def _generate_edges_configuration_single(
        self,
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        noise: float = 0.0,
        min_edge_density: float = None,  # Remove default value
        max_retries: int = None  # Remove default value
    ) -> Optional[sp.spmatrix]:
        """
        Single attempt at generating edges using configuration model.
        Uses global parameters from instance for constraints.
        """
        # Use instance parameters instead of defaults
        min_edge_density = self.min_edge_density if min_edge_density is None else min_edge_density
        max_retries = self.max_retries if max_retries is None else max_retries

        # Calculate target degrees based on degree factors
        target_degrees = np.round(degree_factors).astype(int)
        
        # Ensure even sum of degrees
        if np.sum(target_degrees) % 2 == 1:
            target_degrees[np.random.randint(0, len(target_degrees))] += 1
        
        # Initialize edge list and remaining degrees
        edges = []
        remaining_degrees = target_degrees.copy()
        
        # Calculate connection probabilities between all node pairs
        conn_probs = np.zeros((len(target_degrees), len(target_degrees)))
        for i in range(len(target_degrees)):
            for j in range(i+1, len(target_degrees)):
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
            total_prob = np.sum(sampling_probs)
            if total_prob == 0:
                # Fallback: choose a valid partner uniformly at random
                sampling_probs = valid_partners.astype(float)
                sampling_probs = sampling_probs / np.sum(sampling_probs)
            else:
                sampling_probs = sampling_probs / total_prob
            
           
            # Sample partner node
            node_j = np.random.choice(len(target_degrees), p=sampling_probs)
            
            # Add edge
            edges.append((node_i, node_j))
            
            # Update remaining degrees
            remaining_degrees[node_i] -= 1
            remaining_degrees[node_j] -= 1
        
        # Create sparse adjacency matrix
        if edges:
            rows, cols = zip(*edges)
            data = np.ones(len(edges))
            adj = sp.csr_matrix((data, (rows, cols)), shape=(len(target_degrees), len(target_degrees)))
            
            # Make symmetric (undirected graph)
            adj = adj + adj.T
            adj.data = np.ones_like(adj.data)  # Ensure no double edges
            
            # Calculate actual edge density
            actual_density = adj.nnz / (len(target_degrees) * (len(target_degrees) - 1))
            
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

    def _enhance_triangles(self) -> None:
        """Enhance triangle formation through probabilistic edge addition."""
        if self.universe.triangle_density == 0:
            return
        
        # Store initial adjacency matrix before triangle enhancement
        self.initial_adjacency = self.adjacency.copy()
        
        # Get community triangle propensities for our communities
        community_propensities = np.array([
            self.universe.community_triangle_propensities[comm_id] 
            for comm_id in self.communities
        ])
        
        # Use vectorized version that does EXACTLY the same thing
        self._add_triangles_vectorized(community_propensities)

    def _add_triangles_vectorized(self, community_propensities: np.ndarray) -> None:
        """Vectorized version of the original triangle enhancement logic."""
        
        adj_dense = self.adjacency.toarray()
        n_nodes = len(self.community_labels)
        
        # Generate ALL possible triangles at once
        triangles = np.array(list(combinations(range(n_nodes), 3)))
        
        if len(triangles) == 0:
            return
        
        # Extract indices for vectorized operations
        i_indices = triangles[:, 0]
        j_indices = triangles[:, 1] 
        k_indices = triangles[:, 2]
        
        # Check existing edges for all triangles at once
        edge_ij = adj_dense[i_indices, j_indices]
        edge_jk = adj_dense[j_indices, k_indices]
        edge_ik = adj_dense[i_indices, k_indices]
        
        # Count existing edges per triangle
        edges_exist = edge_ij + edge_jk + edge_ik
        
        # Filter: only triangles that are not complete (< 3 edges) but have 2-paths (>= 2 edges)
        valid_mask = (edges_exist >= 2) & (edges_exist < 3)
        valid_triangles = triangles[valid_mask]
        
        if len(valid_triangles) == 0:
            return
        
        # Extract communities for valid triangles
        valid_i = valid_triangles[:, 0]
        valid_j = valid_triangles[:, 1]
        valid_k = valid_triangles[:, 2]
        
        comm_i = self.community_labels[valid_i]
        comm_j = self.community_labels[valid_j]
        comm_k = self.community_labels[valid_k]
        
        # Calculate average propensities vectorized
        propensities_i = community_propensities[comm_i]
        propensities_j = community_propensities[comm_j]
        propensities_k = community_propensities[comm_k]
        
        avg_propensities = (propensities_i + propensities_j + propensities_k) / 3
        
        # Calculate triangle probabilities
        triangle_probs = self.universe.triangle_density * avg_propensities / len(self.communities)
        
        # Sample which triangles to complete
        random_vals = np.random.random(len(triangle_probs))
        selected_mask = random_vals < triangle_probs
        selected_triangles = valid_triangles[selected_mask]
        
        if len(selected_triangles) == 0:
            return
        
        # For selected triangles, add missing edges
        for triangle in selected_triangles:
            i, j, k = triangle
            comm_i = self.community_labels[i]
            comm_j = self.community_labels[j]
            comm_k = self.community_labels[k]
            
            # Add missing edges (same logic as original)
            if adj_dense[i, j] == 0:
                if self._should_connect_communities(comm_i, comm_j):
                    adj_dense[i, j] = adj_dense[j, i] = 1
            
            if adj_dense[j, k] == 0:
                if self._should_connect_communities(comm_j, comm_k):
                    adj_dense[j, k] = adj_dense[k, j] = 1
            
            if adj_dense[i, k] == 0:
                if self._should_connect_communities(comm_i, comm_k):
                    adj_dense[i, k] = adj_dense[k, i] = 1
        
        # Convert back to sparse
        self.adjacency = sp.csr_matrix(adj_dense)

    def _should_connect_communities(self, comm_a: int, comm_b: int) -> bool:
        """Simple check if communities should be connected."""
        return self.P_sub[comm_a, comm_b] > 0.01

    def analyze_community_connections(self) -> Dict[str, Any]:
        """
        Analyze community connection patterns and deviations from expected probabilities.
        
        Returns:
            Dictionary with analysis results
        """
        n_communities = len(self.communities)
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        connection_counts = np.zeros((n_communities, n_communities), dtype=int)
        
        # Count nodes in each community
        for label in self.community_labels:
            community_sizes[label] += 1
        
        # Count edges between communities
        for i, j in self.graph.edges():
            comm_i = self.community_labels[i]
            comm_j = self.community_labels[j]
            actual_matrix[comm_i, comm_j] += 1
            actual_matrix[comm_j, comm_i] += 1  # Undirected graph
            connection_counts[comm_i, comm_j] += 1
            connection_counts[comm_j, comm_i] += 1
        
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
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:
                        actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
        
        # Calculate deviations
        deviation_matrix = np.abs(actual_matrix - self.P_sub)
        mean_deviation = np.mean(deviation_matrix)
        max_deviation = np.max(deviation_matrix)
        
        # Calculate average degrees per community
        avg_degrees = np.zeros(n_communities)
        for i in range(n_communities):
            community_nodes = np.where(self.community_labels == i)[0]
            if len(community_nodes) > 0:
                avg_degrees[i] = np.mean([self.graph.degree(n) for n in community_nodes])
        
        # Calculate community densities
        densities = np.zeros(n_communities)
        for i in range(n_communities):
            n = community_sizes[i]
            if n > 1:
                densities[i] = actual_matrix[i, i]
        
        # Degree analysis
        degree_analysis = {
            "community_avg_degrees": avg_degrees.tolist(),
            "community_densities": densities.tolist()
        }
        
        if max_deviation > self.max_max_community_deviation:
            # Keep this warning as it indicates a real problem if it ever occurs
            print(f"\nWarning: Inconsistency detected - graph was generated with constraints but now exceeds them.")
            print(f"This should not happen. Please report this as a bug.")
            print(f"Current max deviation: {max_deviation:.4f}")
            print(f"Original constraint: {self.max_max_community_deviation:.4f}")
            print(f"Generation method: {self.generation_method}")
            print(f"Generation parameters: {self.generation_params}")
        
        return {
            "actual_matrix": actual_matrix,
            "expected_matrix": self.P_sub,
            "deviation_matrix": deviation_matrix,
            "mean_deviation": float(mean_deviation),
            "max_deviation": float(max_deviation),
            "community_sizes": community_sizes,
            "connection_counts": connection_counts,
            "degree_analysis": degree_analysis,
            "constraints": {  # Keep constraints in output for verification
                "max_mean_deviation": self.max_mean_community_deviation,
                "max_max_deviation": self.max_max_community_deviation
            },
            "avg_degrees": avg_degrees,  # Add average degrees per community
            "densities": densities  # Add community densities
        }

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
        n_communities = P_sub.shape[0]  # Use P_sub shape instead of self.communities
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            if 0 <= label < n_communities:  # Add bounds check
                community_sizes[label] += 1
            else:
                raise ValueError(f"Invalid community label {label} for {n_communities} communities")
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            if 0 <= comm_i < n_communities and 0 <= comm_j < n_communities:  # Add bounds check
                actual_matrix[comm_i, comm_j] += 1
                actual_matrix[comm_j, comm_i] += 1  # Undirected graph
            else:
                raise ValueError(f"Invalid community labels {comm_i}, {comm_j} for {n_communities} communities")
        
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
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:  # Add check for zero community sizes
                        actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
        
        # Calculate deviations
        deviation_matrix = np.abs(actual_matrix - P_sub)
        mean_deviation = np.mean(deviation_matrix)
        max_deviation = np.max(deviation_matrix)
        
        return {
            "mean_deviation": mean_deviation,
            "max_deviation": max_deviation
        }

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
        n_communities = P_sub.shape[0]  # Use P_sub shape instead of self.communities
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            if 0 <= label < n_communities:  # Add bounds check
                community_sizes[label] += 1
            else:
                raise ValueError(f"Invalid community label {label} for {n_communities} communities")
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            if 0 <= comm_i < n_communities and 0 <= comm_j < n_communities:  # Add bounds check
                actual_matrix[comm_i, comm_j] += 1
                actual_matrix[comm_j, comm_i] += 1  # Undirected graph
            else:
                raise ValueError(f"Invalid community labels {comm_i}, {comm_j} for {n_communities} communities")
        
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
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:  # Add check for zero community sizes
                        actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
        
        # Calculate maximum deviation
        deviation_matrix = np.abs(actual_matrix - P_sub)
        return np.max(deviation_matrix)

    def _calculate_community_deviations(
        self,
        graph: nx.Graph,
        community_labels: np.ndarray,
        P_sub: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate community deviations for a given graph and community labels.
        
        Args:
            graph: NetworkX graph
            community_labels: Array of community labels
            P_sub: Expected probability matrix
            
        Returns:
            Dictionary with mean and max deviations
        """
        n_communities = P_sub.shape[0]  # Use P_sub shape instead of self.communities
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            if 0 <= label < n_communities:  # Add bounds check
                community_sizes[label] += 1
            else:
                raise ValueError(f"Invalid community label {label} for {n_communities} communities")
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            if 0 <= comm_i < n_communities and 0 <= comm_j < n_communities:  # Add bounds check
                actual_matrix[comm_i, comm_j] += 1
                actual_matrix[comm_j, comm_i] += 1  # Undirected graph
            else:
                raise ValueError(f"Invalid community labels {comm_i}, {comm_j} for {n_communities} communities")
        
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
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:  # Add check for zero community sizes
                        actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
        
        # Calculate deviations
        deviation_matrix = np.abs(actual_matrix - P_sub)
        mean_deviation = np.mean(deviation_matrix)
        max_deviation = np.max(deviation_matrix)
        
        return {
            "mean_deviation": mean_deviation,
            "max_deviation": max_deviation
        }

    def calculate_feature_signal(self, random_state: int = 42) -> float:
        """
        Calculate Feature Signal using Random Forest classifier and macro F1 score.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Feature signal ∈ [0, 1], or 0.0 if no features available
        """
        if self.features is None or len(self.features) == 0:
            return 0.0
            
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        
        # Get features and labels
        X = self.features
        y = self.community_labels
        
        # Split data ensuring all communities are represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3,
            stratify=y,
            random_state=random_state
        )
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state
        )
        
        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate macro F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        
        return float(f1)

    def calculate_structure_signal(self, divergence_metric='kl'):
        """
        Calculate Structure Signal: Edge Probability Row Divergence
        
        Args:
            divergence_metric: Divergence metric to use ('kl', 'js', or 'cosine')
            
        Returns:
            Dictionary mapping community index to structure signal
        """
        import numpy as np
        from scipy.spatial.distance import cosine
        from scipy.special import kl_div, rel_entr
        
        # Get unique communities and create a mapping from label to index
        unique_communities = np.unique(self.community_labels)
        n_communities = len(unique_communities)
        label_to_index = {label: idx for idx, label in enumerate(unique_communities)}
        
        # Initialize community-community edge probability matrix
        edge_probs = np.zeros((n_communities, n_communities))
        
        # Count nodes in each community
        community_sizes = np.zeros(n_communities, dtype=int)
        for label in self.community_labels:
            community_sizes[label_to_index[label]] += 1
        
        # Count edges between communities
        for i, j in self.graph.edges():
            comm_i = label_to_index[self.community_labels[i]]
            comm_j = label_to_index[self.community_labels[j]]
            edge_probs[comm_i, comm_j] += 1
            edge_probs[comm_j, comm_i] += 1  # Undirected graph
        
        # Calculate actual probabilities
        for i in range(n_communities):
            for j in range(n_communities):
                if i == j:
                    # Within community: divide by n(n-1)/2
                    n = community_sizes[i]
                    if n > 1:
                        edge_probs[i, j] = edge_probs[i, j] / (n * (n - 1))
                else:
                    # Between communities: divide by n1*n2
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:
                        edge_probs[i, j] = edge_probs[i, j] / (n1 * n2)
        
        # Normalize rows to form distributions
        row_sums = np.sum(edge_probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        normalized_probs = edge_probs / row_sums
        
        # Initialize results
        structure_signals = {}
        
        # Define JS divergence function
        def js_divergence(p, q):
            # Add small epsilon to avoid log(0)
            p = p + 1e-10
            q = q + 1e-10
            # Normalize
            p = p / np.sum(p)
            q = q / np.sum(q)
            m = 0.5 * (p + q)
            return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
        
        # For each community
        for i in range(n_communities):
            if community_sizes[i] <= 1:  # Skip if only one node or none
                continue
                
            # Initialize minimum divergence as infinity
            min_divergence = float('inf')
            
            # For each other community
            for j in range(n_communities):
                if i == j or community_sizes[j] <= 1:
                    continue
                    
                # Calculate divergence based on specified metric
                if divergence_metric == 'kl':
                    # KL divergence (with small epsilon to avoid division by zero)
                    p = normalized_probs[i] + 1e-10
                    q = normalized_probs[j] + 1e-10
                    divergence = np.sum(rel_entr(p, q))
                elif divergence_metric == 'js':
                    # JS divergence
                    divergence = js_divergence(normalized_probs[i], normalized_probs[j])
                else:  # cosine
                    # Cosine distance
                    divergence = cosine(normalized_probs[i], normalized_probs[j])
                
                # Update minimum divergence
                min_divergence = min(min_divergence, divergence)
            
            # Store structure signal for this community (if valid)
            if min_divergence != float('inf'):
                structure_signals[int(unique_communities[i])] = float(min_divergence)
            else:
                structure_signals[int(unique_communities[i])] = 0.0
        
        return structure_signals

    def calculate_degree_signal(self,
                            method: str = "naive_bayes",
                            metric: str = "accuracy", 
                            cv_folds: int = 5,
                            random_state: int = 42) -> float:
        """
        Calculate Degree Signal using degree-based classification.
        
        Args:
            method: Classification method ("naive_bayes", "quantile_binning")
            metric: Evaluation metric ("accuracy", "nmi", "ari")
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            
        Returns:
            Degree signal ∈ [0, 1]
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        
        # Get node degrees - ensure proper ordering
        degrees = np.array([self.graph.degree(i) for i in range(self.graph.number_of_nodes())])
        
        # Ensure we have the right number of community labels
        min_len = min(len(degrees), len(self.community_labels))
        degrees = degrees[:min_len]
        ground_truth = self.community_labels[:min_len]
        
        # Reshape degrees for sklearn
        degrees_reshaped = degrees.reshape(-1, 1)
        
        if method == "naive_bayes":
            classifier = GaussianNB()
            
            if metric == "accuracy":
                scores = cross_val_score(
                    classifier, degrees_reshaped, ground_truth, 
                    cv=cv_folds, scoring='accuracy'
                )
                return np.mean(scores)
            else:
                # For NMI/ARI, use cross-validation with custom scoring
                scores = []
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                for train_idx, test_idx in skf.split(degrees_reshaped, ground_truth):
                    classifier.fit(degrees_reshaped[train_idx], ground_truth[train_idx])
                    predicted = classifier.predict(degrees_reshaped[test_idx])
                    
                    if metric == "nmi":
                        score = normalized_mutual_info_score(ground_truth[test_idx], predicted)
                    elif metric == "ari":
                        score = adjusted_rand_score(ground_truth[test_idx], predicted)
                    
                    scores.append(score)
                
                return np.mean(scores)
        
        elif method == "quantile_binning":
            n_communities = len(np.unique(ground_truth))
            
            # Create quantile bins based on degree distribution
            quantiles = np.linspace(0, 1, n_communities + 1)
            bin_edges = np.quantile(degrees, quantiles)
            
            # Handle edge case where all degrees are the same
            if len(np.unique(bin_edges)) == 1:
                return 0.0
            
            # Assign degrees to bins
            predicted_bins = np.digitize(degrees, bin_edges) - 1
            predicted_bins = np.clip(predicted_bins, 0, n_communities - 1)
            
            if metric == "accuracy":
                return self._calculate_best_accuracy(ground_truth, predicted_bins)
            elif metric == "nmi":
                return normalized_mutual_info_score(ground_truth, predicted_bins)
            elif metric == "ari":
                return adjusted_rand_score(ground_truth, predicted_bins)
        
        else:
            raise ValueError(f"Unknown degree classification method: {method}")

    def calculate_triangle_community_signal(self) -> float:
        """Measure correlation between community and triangle participation."""
        triangle_counts = []
        community_labels = []
        
        for node in self.graph.nodes():
            # Count triangles involving this node
            triangles = 0
            neighbors = list(self.graph.neighbors(node))
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if self.graph.has_edge(neighbor1, neighbor2):
                        triangles += 1
            
            triangle_counts.append(triangles)
            community_labels.append(self.community_labels[node])
        
        # Correlation between community and triangle participation
        return np.corrcoef(community_labels, triangle_counts)[0, 1]

    def calculate_community_signals(self,
                                structure_metric: str = 'kl',
                                degree_method: str = "naive_bayes",
                                degree_metric: str = "accuracy",
                                cv_folds: int = 5,
                                random_state: int = 42) -> Dict[str, Any]:
        """
        Calculate all community-related signal metrics using the unified approach.
        
        Args:
            structure_metric: Divergence metric for structure signal ('kl', 'js', 'cosine')
            degree_method: Method for degree signal ("naive_bayes", "quantile_binning")
            degree_metric: Evaluation metric for degree signal ("accuracy", "nmi", "ari")
            cv_folds: Number of cross-validation folds for degree signal
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with signal values and summary statistics
        """
        signals = {}
        
        # Structure signal
        try:
            structure_signals = self.calculate_structure_signal(divergence_metric=structure_metric)
            signals['structure_signal'] = structure_signals
            signals['mean_structure_signal'] = float(np.mean(list(structure_signals.values())))
            signals['min_structure_signal'] = float(np.min(list(structure_signals.values())))
            signals['max_structure_signal'] = float(np.max(list(structure_signals.values())))
        except Exception as e:
            warnings.warn(f"Failed to calculate structure signal: {e}")
            signals['structure_signal'] = 0.0
            signals['mean_structure_signal'] = 0.0
            signals['min_structure_signal'] = 0.0
            signals['max_structure_signal'] = 0.0
        
        # Feature signal (only if features are available)
        if self.features is not None and len(self.features) > 0:
            try:
                feature_signal = self.calculate_feature_signal(random_state=random_state)
                signals['feature_signal'] = feature_signal
            except Exception as e:
                warnings.warn(f"Failed to calculate feature signal: {e}")
                signals['feature_signal'] = 0.0
        else:
            signals['feature_signal'] = None
        
        # Degree signal
        try:
            degree_signal = self.calculate_degree_signal(
                method=degree_method,
                metric=degree_metric,
                cv_folds=cv_folds,
                random_state=random_state
            )
            signals['degree_signal'] = degree_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate degree signal: {e}")
            signals['degree_signal'] = 0.0
        
        # Triangle signal
        try:
            triangle_signal = self.calculate_triangle_community_signal()
            signals['triangle_signal'] = triangle_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate triangle signal: {e}")
            signals['triangle_signal'] = 0.0

        # Calculate summary statistics (excluding None values)
        valid_signals = [v for v in [signals.get('feature_signal'), signals.get('degree_signal')] if v is not None]
        if valid_signals:
            signals['mean_signal'] = float(np.mean(valid_signals))
            signals['min_signal'] = float(np.min(valid_signals))
            signals['max_signal'] = float(np.max(valid_signals))
            signals['std_signal'] = float(np.std(valid_signals))
        
        # Add metadata
        signals['method_info'] = {
            'structure_metric': structure_metric,
            'degree_method': degree_method,
            'degree_metric': degree_metric,
            'triangle_signal': triangle_signal,
            'cv_folds': cv_folds
        }
        
        return signals

    def _calculate_best_accuracy(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """
        Helper method: Calculate best possible accuracy by finding optimal label mapping.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            
        Returns:
            Best achievable accuracy ∈ [0, 1]
        """
        from scipy.optimize import linear_sum_assignment
        
        # Get unique labels
        true_unique = np.unique(true_labels)
        pred_unique = np.unique(predicted_labels)
        
        # Create confusion matrix
        confusion_matrix = np.zeros((len(pred_unique), len(true_unique)))
        
        for i, pred_label in enumerate(pred_unique):
            for j, true_label in enumerate(true_unique):
                confusion_matrix[i, j] = np.sum(
                    (predicted_labels == pred_label) & (true_labels == true_label)
                )
        
        # Find optimal assignment using Hungarian algorithm
        row_idx, col_idx = linear_sum_assignment(-confusion_matrix)
        
        # Calculate accuracy with optimal mapping
        correct_assignments = confusion_matrix[row_idx, col_idx].sum()
        total_assignments = len(true_labels)
        
        return correct_assignments / total_assignments if total_assignments > 0 else 0.0

    def update_feature_generator(
        self,
        cluster_count_factor: float = 1.0,
        center_variance: float = 1.0,
        cluster_variance: float = 0.1,
        assignment_skewness: float = 0.0,
        community_exclusivity: float = 1.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Update the feature generator with new parameters.
        
        Args:
            cluster_count_factor: Number of clusters relative to communities
            center_variance: Separation between cluster centers
            cluster_variance: Spread within each cluster
            assignment_skewness: If some clusters are used more frequently
            community_exclusivity: How exclusively clusters map to communities
            seed: Random seed for reproducibility
        """
        if self.feature_dim > 0:
            self.feature_generator = SimplifiedFeatureGenerator(
                universe_K=self.K,
                feature_dim=self.feature_dim,
                cluster_count_factor=cluster_count_factor,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
                assignment_skewness=assignment_skewness,
                community_exclusivity=community_exclusivity,
                seed=seed
            )

    def regenerate_features(self) -> None:
        """
        Regenerate node features using the current feature generator parameters.
        This should be called after updating feature generator parameters.
        """
        if self.universe.feature_generator is None:
            return
            
        # Get community assignments and map to universe community IDs
        local_community_assignments = self.community_labels
        universe_community_assignments = np.array([self.community_id_mapping[idx] for idx in local_community_assignments])
        
        # Generate node clusters based on universe community assignments
        self.node_clusters = self.universe.feature_generator.assign_node_clusters(universe_community_assignments)
        
        # Generate features based on node clusters
        self.features = self.universe.feature_generator.generate_node_features(self.node_clusters)
        
        # Update node attributes in the graph
        for i in range(self.n_nodes):
            self.graph.nodes[i]['features'] = self.features[i].tolist()

    def analyze_triangles(self) -> Dict[str, Any]:
        """Analyze triangles in the graph.
        
        Returns:
            Dict containing:
            - total_triangles: Total number of triangles in the graph
            - triangles_per_community: Dict mapping community ID to number of triangles
            - additional_triangles_per_community: Dict mapping community ID to number of additionally created triangles
            - triangle_propensity_correlation: Correlation between community triangle counts and propensities
        """
        # Convert to dense for easier manipulation
        adj_dense = self.adjacency.toarray()
        n_nodes = len(self.community_labels)
        
        # Count triangles per community (using universe community IDs)
        triangles_per_community = {comm: 0 for comm in self.communities}
        additional_triangles_per_community = {comm: 0 for comm in self.communities}
        total_triangles = 0
        total_additional_triangles = 0
        
        # Get community propensities for triangle assignment
        # Map universe community IDs to their propensities
        community_propensities = {
            comm_id: self.universe.community_triangle_propensities[comm_id]
            for comm_id in self.communities
        }
        
        # For each potential triangle (i, j, k)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                for k in range(j + 1, n_nodes):
                    # Check if triangle exists in final graph
                    if adj_dense[i, j] and adj_dense[j, k] and adj_dense[i, k]:
                        total_triangles += 1
                        
                        # Get communities involved (using universe community IDs)
                        comm_i = self.community_labels_universe_level[i]
                        comm_j = self.community_labels_universe_level[j]
                        comm_k = self.community_labels_universe_level[k]
                        
                        # Find the community with highest propensity among the three
                        comm_props = [
                            (comm_i, community_propensities[comm_i]),
                            (comm_j, community_propensities[comm_j]),
                            (comm_k, community_propensities[comm_k])
                        ]
                        primary_comm = max(comm_props, key=lambda x: x[1])[0]
                        
                        # Count triangle only for the primary community
                        triangles_per_community[primary_comm] += 1
                        
                        # Check if this was an additional triangle
                        if hasattr(self, 'initial_adjacency'):
                            initial_adj_dense = self.initial_adjacency.toarray()
                            if not (initial_adj_dense[i, j] and initial_adj_dense[j, k] and initial_adj_dense[i, k]):
                                total_additional_triangles += 1
                                additional_triangles_per_community[primary_comm] += 1
        
        # Get community propensities for our communities
        community_propensities_array = np.array([
            self.universe.community_triangle_propensities[comm_id] 
            for comm_id in self.communities
        ])
        
        # Calculate correlation between triangle counts and propensities
        triangle_counts = np.array([triangles_per_community[comm] for comm in self.communities])
        correlation = np.corrcoef(triangle_counts, community_propensities_array)[0, 1]
        
        return {
            "total_triangles": total_triangles,
            "total_additional_triangles": total_additional_triangles,
            "triangles_per_community": triangles_per_community,
            "additional_triangles_per_community": additional_triangles_per_community,
            "triangle_propensity_correlation": correlation
        }
