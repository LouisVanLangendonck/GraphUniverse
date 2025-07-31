import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from graph_universe.feature_regimes import (
    SimplifiedFeatureGenerator,
    NeighborhoodFeatureAnalyzer,
    FeatureClusterLabelGenerator
)
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
from itertools import combinations
import signal

class GraphUniverse:
    """
    Represents a generative universe for graph instances sampled from a master "pseudo" stochastic block model.
    The GraphSample class will randomly sub-sample from these global universe properties.
    """
    
    def __init__(
        self,
        K: int,
        P: Optional[np.ndarray] = None,
        feature_dim: int = 0,
        inter_community_variance: float = 0.0,

        # Feature generation parameters
        cluster_count_factor: float = 1.0,  # Number of clusters relative to communities
        center_variance: float = 1.0,       # Separation between cluster centers
        cluster_variance: float = 0.1,      # Spread within each cluster
        assignment_skewness: float = 0.0,   # If some clusters are used more frequently
        community_exclusivity: float = 1.0, # How exclusively clusters map to communities

        # Degree center parameters
        degree_center_method: str = "linear",  # How to generate degree centers ("linear", "random", "shuffled")
        seed: Optional[int] = None,

        # Community co-occurrence homogeneity
        community_cooccurrence_homogeneity: float = 1.0,  # 0-1: how homogeneous the co-occurrence of communities is
    ):
        """
        Initialize a graph universe with K communities and optional feature generation.
        
        Args:
            K: Number of communities
            P: Optional probability matrix (if None, will be generated)
            feature_dim: Dimension of node features
            inter_community_variance: Amount of variance in the inter-community probabilities
            cluster_count_factor: Number of clusters relative to communities (0.1 to 4.0)
            center_variance: Separation between cluster centers
            cluster_variance: Spread within each cluster
            assignment_skewness: If some clusters are used more frequently (0.0 to 1.0)
            community_exclusivity: How exclusively clusters map to communities (0.0 to 1.0)
            degree_center_method: How to generate degree centers ("linear", "random", "shuffled")
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
            self.P = self._generate_inter_community_variance_matrix(
                K, inter_community_variance
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
        self.inter_community_variance = inter_community_variance
        self.feature_variance = 0.1
        self.feature_similarity_matrix = None
        # Store regime parameters
        self.intra_community_regime_similarity = 0.8
        self.inter_community_regime_similarity = 0.2

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
            
        # Generate community co-occurrence matrix
        self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity
        self.community_cooccurrence_matrix = self._generate_cooccurrence_matrix(K, community_cooccurrence_homogeneity, seed)

    def _generate_inter_community_variance_matrix(
        self, 
        K: int, 
        inter_community_variance: float = 0.0
    ) -> np.ndarray:
        """
        Generate a pseudo-probability matrix that gives the RELATIVE probabilities BETWEEN different communities (the intra-community prob / homophily is decided by the GraphFamily object and scaled accordingly in a GraphSample object)
        
        Args:
            K: Number of communities
            inter_community_variance: Amount of variance in the inter-community probabilities
            
        Returns:
            K × K probability matrix
        """
            
        P = np.ones((K, K))
        
        # Add the inter-community variance if requested
        if inter_community_variance > 0:
            noise = np.random.normal(0, inter_community_variance*2, size=(K, K))
            P = P + noise
            # Clip to be between 0 and 2
            P[P < 0.0] = 0.0
            P[P > 2.0] = 2.0

        # Have the P to be symmetric (just set all values of lower triangle to be the same as the upper triangle)
        for i in range(K):
            for j in range(i+1, K):
                P[i, j] = P[j, i]

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
            # Sample community one by one and always check for a new candidate that is has a non-zero probabilty connection to the existing communities
            result = [np.random.choice(self.K)]
            while len(result) < size:
                new_community = np.random.choice(self.K)
                # Check if self.P[new_community, result] is non-zero
                if np.sum(self.P[new_community, result]) > 0 and new_community not in result:
                    result.append(new_community)
            return result
        
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
    
    def _generate_cooccurrence_matrix(self, K: int, homogeneity: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate symmetric community co-occurrence matrix.
        
        Args:
            K: Number of communities
            homogeneity: 1.0 = uniform co-occurrence, 0.0 = heterogeneous patterns
            seed: Random seed
            
        Returns:
            Symmetric K x K co-occurrence probability matrix (diagonal is always 0)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if homogeneity == 1.0:
            # Perfectly homogeneous - all pairs equally likely
            matrix = np.ones((K, K)) / K
            np.fill_diagonal(matrix, 0.0)  # Self-occurrence is always 0
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
        
        # Set diagonal to 0.0 (self-occurrence is always 0)
        np.fill_diagonal(matrix, 0.0)
        
        # Ensure all values are positive and reasonable
        matrix = np.clip(matrix, 0.01, 1.0)
        
        # Normalize rows to maintain proper probabilities
        # Each row should sum to something reasonable relative to K
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        matrix = matrix / row_sums * K * base_prob * 2  # Scale to reasonable range
        matrix = np.clip(matrix, 0.01, 1.0)
        
        # Ensure diagonal stays at 0 after normalization
        np.fill_diagonal(matrix, 0.0)
        
        return matrix

class GraphSample:
    """
    Represents a single graph instance sampled from the GraphUniverse.
    
    This modified version implements both standard DC-SBM and the new
    Degree-Community-Coupled Corrected SBM (DCCC-SBM).
    """
    
    def __init__(
        self,
        # Give GraphUniverse object to sample from
        universe: GraphUniverse,

        # Graph Sample specific parameters
        num_communities: int,
        n_nodes: int,
        min_component_size: int,
        target_homophily: float,
        target_average_degree: float,
        # target_density: float,
        degree_distribution: str,
        power_law_exponent: Optional[float],
        max_mean_community_deviation: float,
        max_max_community_deviation: float,
        min_edge_density: float,
        max_retries: int,

        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        degree_heterogeneity: float,

        # DCCC-SBM parameters
        use_dccc_sbm: bool = True,
        degree_separation: float = 0.5,
        dccc_global_degree_params: Optional[dict] = None,
        degree_signal_calc_method: str = "standard",
        disable_deviation_limiting: bool = False,

        # Random seed
        seed: Optional[int] = None,

        # Optional Parameter for user-defined communuties to be sampled
        user_defined_communities: Optional[List[int]] = None
    ):
        """
        Initialize and generate a graph sample from the GraphUniverse.
        """

        # Store the GraphUniverse object
        self.universe = universe

        # Store additional DCCC-SBM parameters
        self.use_dccc_sbm = use_dccc_sbm
        self.degree_separation = degree_separation
        self.dccc_global_degree_params = dccc_global_degree_params or {}
        self.disable_deviation_limiting = disable_deviation_limiting  # Store the parameter
        
        # Original initialization code with modifications...
        self.timing_info = {}
        total_start = time.time()
        
        # Add timeout mechanism
        TIMEOUT_SECONDS = 60
        
        def check_timeout():
            if time.time() - total_start > TIMEOUT_SECONDS:
                raise TimeoutError(f"GraphSample initialization timed out after {TIMEOUT_SECONDS} seconds")
        
        try:
            # Sample communities from universe or use user-defined communities
            if user_defined_communities is not None:
                self.communities = user_defined_communities
            else:
                check_timeout()
                self.communities = universe.sample_connected_community_subset(
                    num_communities,
                    seed=seed,
                    use_cooccurrence=True
                )
            
            # Store the number of nodes
            self.original_n_nodes = n_nodes
            self.min_component_size = min_component_size # Minimum size of a connected component
            
            # Store target parameters
            self.target_homophily = target_homophily
            # self.target_density = target_density
            self.target_average_degree = target_average_degree
            
            # Degree distribution parameters
            self.degree_distribution = degree_distribution
            self.power_law_exponent = power_law_exponent

            # Store community deviation parameters as instance attributes
            self.max_mean_community_deviation = max_mean_community_deviation
            self.max_max_community_deviation = max_max_community_deviation
            self.min_edge_density = min_edge_density
            self.max_retries = max_retries

            # Create mapping between local community indices and universe community IDs
            self.community_id_mapping = {i: comm_id for i, comm_id in enumerate(self.communities)}
            self.reverse_community_id_mapping = {comm_id: i for i, comm_id in self.community_id_mapping.items()}

            # Initialize generation method and parameters
            self.generation_method = "standard"
            self.generation_params = {
                "degree_heterogeneity": degree_heterogeneity,
                "max_mean_community_deviation": max_mean_community_deviation,
                "max_max_community_deviation": max_max_community_deviation
            }
            
            # If DCCC-SBM is enabled, update generation method
            if self.use_dccc_sbm:
                self.generation_method = "dccc_sbm"
                self.generation_params.update({
                    "degree_separation": degree_separation,
                    "degree_distribution_type": degree_distribution,
                })
                if degree_distribution == "power_law":
                    self.generation_params["power_law_exponent"] = power_law_exponent
                

            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
            
            check_timeout()
            
            # Time: Extract and scale probability matrix
            start = time.time()

            # Extract the submatrix of the probability matrix for these communities
            K_sub = len(self.communities) # Number of communities in the sample
            self.P_sub = np.zeros((K_sub, K_sub)) # Initialize the probability matrix for the sample
            for i, ci in enumerate(sorted(self.communities)):
                for j, cj in enumerate(sorted(self.communities)):
                    self.P_sub[i, j] = universe.P[ci, cj]

            # Scale the probability matrix
            self.P_sub = self._scale_probability_matrix(
                self.P_sub, 
                self.target_average_degree,
                self.target_homophily,
                self.original_n_nodes
            )
            self.timing_info['probability_matrix'] = time.time() - start
            
            check_timeout()
            
            # Time: Generate memberships
            start = time.time()

            # Uniform membership generation
            self.community_labels = self._generate_memberships(n_nodes, K_sub)  # Now returns 1D array

            # Create a new array that maps the community labels to the universe community IDs
            self.community_labels_universe_level = np.array([self.community_id_mapping[idx] for idx in self.community_labels])
            
            # Store membership generation time
            self.timing_info['memberships'] = time.time() - start
            
            check_timeout()
            
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
                self.degree_signal_calc_method = degree_signal_calc_method
                
                # Generate community-specific degree factors using improved method
                self.degree_factors = self._generate_community_degree_factors_improved(
                    self.community_labels,
                    degree_distribution,
                    degree_separation,
                    global_degree_params
                )

            else:
                self.degree_factors = self._generate_degree_factors(n_nodes, degree_heterogeneity)
            self.timing_info['degree_factors'] = time.time() - start
            
            check_timeout()
            
            # Time: Generate edges
            start = time.time()
            self.adjacency = self._generate_edges(
                self.community_labels,
                self.P_sub,
                self.degree_factors,
            )
            self.timing_info['edge_generation'] = time.time() - start

            check_timeout()

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
            
            # Filter components based on size and have a maximum of num_communities unique components
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

            if len(kept_components) > num_communities:
                raise ValueError(f"Number of unique components exceeds number of communities: {len(kept_components)} > {num_communities}.")

            self.timing_info['component_filtering'] = time.time() - start
            
            check_timeout()
            
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
            
            check_timeout()
            
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

        except TimeoutError:
            raise TimeoutError("GraphSample initialization timed out")
  
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
    
    def _generate_community_degree_factors_improved(
        self,
        community_labels: np.ndarray,
        degree_distribution_type: str,
        degree_separation: float,
        global_degree_params: dict,
    ) -> np.ndarray:
        n_nodes = len(community_labels)
        
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

        # 2. Sort degrees
        sorted_degrees = np.sort(raw_degrees)
        
        # 3. Get universe degree centers for our communities
        K = len(self.communities)
        universe_degree_centers = np.array([
            self.universe.degree_centers[self.community_id_mapping[local_comm_id]]
            for local_comm_id in range(K)
        ])
        
        # 4. Order communities by universe degree center (lowest to highest)
        comm_order = np.argsort(universe_degree_centers)
        ordered_degree_centers = universe_degree_centers[comm_order]
        
        # 5. Map universe degree centers to positions preserving relative distances
        if K == 1:
            community_means = [n_nodes // 2]
        else:
            # Scale the actual degree center values to [0, n_nodes-1] range
            min_center = np.min(ordered_degree_centers)
            max_center = np.max(ordered_degree_centers)
            
            if max_center == min_center:
                # All centers are the same, spread evenly
                community_means = np.linspace(0, n_nodes - 1, K)
            else:
                # Scale preserving relative distances
                scaled_centers = (ordered_degree_centers - min_center) / (max_center - min_center)
                community_means = scaled_centers * (n_nodes - 1)
        
        # 6. Calculate standard deviation based on degree_separation
        if degree_separation == 1.0:
            # For perfect separation, make std tight enough to avoid overlap
            if K > 1:
                min_distance = np.min(np.diff(community_means))
                community_std = max(1.0, min_distance / 6)  # 3*std = half the minimum distance
            else:
                community_std = 1.0
        else:
            # Linear interpolation between tight and wide
            max_std = n_nodes / 3  # Wide case
            min_std = 1.0 if K == 1 else max(1.0, np.min(np.diff(community_means)) / 6)  # Tight case
            community_std = min_std + (1 - degree_separation) * (max_std - min_std)
        
        # 7. Create available positions array and sample for each node
        available_positions = list(range(n_nodes))
        degree_factors = np.zeros(n_nodes)
        
        # 8. For each node, sample from its community's truncated normal distribution
        for node_idx in range(n_nodes):
            community_local_idx = community_labels[node_idx]
            community_rank = np.where(comm_order == community_local_idx)[0][0]
            
            # Get the mean for this community's distribution
            mean_pos = community_means[community_rank]
            std_pos = community_std
            
            if not available_positions:
                # This shouldn't happen, but fallback
                degree_factors[node_idx] = sorted_degrees[node_idx % n_nodes]
                continue
            
            # Sample from truncated normal distribution over available positions
            if len(available_positions) == 1:
                # Only one position left, take it
                chosen_pos = available_positions[0]
            else:
                # Calculate probabilities for each available position based on normal distribution
                probabilities = np.array([
                    np.exp(-0.5 * ((pos - mean_pos) / std_pos) ** 2)
                    for pos in available_positions
                ])
                probabilities = probabilities / np.sum(probabilities)  # Normalize
                
                # Sample from available positions based on probabilities
                chosen_idx = np.random.choice(len(available_positions), p=probabilities)
                chosen_pos = available_positions[chosen_idx]
            
            # Assign the degree and remove position from available
            degree_factors[node_idx] = sorted_degrees[chosen_pos]
            available_positions.remove(chosen_pos)
        
        # 9. Normalize to mean 1 to not influence edge density
        degree_factors = degree_factors / degree_factors.mean()
        
        return degree_factors

    def _generate_edges(
        self, 
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
        min_edge_density: float = 0.001,
        max_retries: int = 5
    ) -> sp.spmatrix:
        """
        Generate edges with minimum density guarantee.
        Uses vectorized operations for faster edge generation with community labels.
        
        Args:
            community_labels: Node community assignments (indices)
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors
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
            else:
                print(f"Warning: Could not achieve minimum edge density after {max_retries} attempts.")
                print(f"Final density: {actual_density:.4f}")
                return adj
        
        return adj

    def _scale_probability_matrix(
        self, 
        P_sub: np.ndarray, 
        target_avg_degree: Optional[float] = None, 
        target_homophily: Optional[float] = None,
        n_nodes: Optional[int] = None
    ) -> np.ndarray:
        n = P_sub.shape[0]
        P_scaled = P_sub.copy()
        
        # Use instance target values if not specified
        target_avg_degree = target_avg_degree if target_avg_degree is not None else self.target_average_degree
        target_homophily = target_homophily if target_homophily is not None else self.target_homophily
        n_nodes = n_nodes if n_nodes is not None else self.n_nodes
        
        # Convert target average degree to equivalent density
        # avg_degree = 2 * edges / n_nodes
        # density = edges / (n_nodes * (n_nodes - 1) / 2)
        # Therefore: density = avg_degree / (n_nodes - 1)
        target_density = target_avg_degree / (n_nodes - 1)
        
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
        
        # # Recalculate actual values after clipping
        # actual_diagonal_sum = np.sum(P_scaled[diagonal_mask])
        # actual_total_sum = np.sum(P_scaled)
        
        # # If clipping significantly affected our targets, do one final density adjustment
        # actual_density = actual_total_sum / (n * n)
        # if abs(actual_density - target_density) > 1e-3:
        #     density_correction = target_density / actual_density
        #     P_scaled = P_scaled * density_correction
        #     P_scaled = np.clip(P_scaled, 0, 1)  # Clip again after final adjustment
        
        return P_scaled

    def calculate_actual_probability_matrix(self) -> np.ndarray:
        """
        Calculate the actual probability matrix for the graph.
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
        
        return actual_matrix, community_sizes, connection_counts

    def analyze_community_connections(self) -> Dict[str, Any]:
        """
        Analyze community connection patterns and deviations from expected probabilities.
        
        Returns:
            Dictionary with analysis results
        """
        n_communities = len(self.communities)
        actual_matrix, community_sizes, connection_counts = self.calculate_actual_probability_matrix()
        
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

    def calculate_structure_signal(self, random_state: int = 42) -> float:
        """
        Calculate Structure Signal using RF classifier by using as input of node the community label counts of its 1-hop neighbors, concatted by 2-hop neighbors and 3-hop neighbors.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from collections import Counter
        
        # For each node, get the community label counts of its 1-hop neighbors
        node_structural_features = []
        labels = []
        n_communities = len(self.communities)
        
        for node in range(self.graph.number_of_nodes()):
            # Get nodes at exact distance 1 using single-source shortest path
            sp_lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=1)
            one_hop_nodes = [n for n, dist in sp_lengths.items() if dist == 1]
            one_hop_community_labels = [self.community_labels[n] for n in one_hop_nodes]

            # Get nodes at exact distance 2 using single-source shortest path
            sp_lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=2)
            two_hop_nodes = [n for n, dist in sp_lengths.items() if dist == 2]
            two_hop_community_labels = [self.community_labels[n] for n in two_hop_nodes]

            # Get nodes at exact distance 3 using single-source shortest path
            sp_lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=3)
            three_hop_nodes = [n for n, dist in sp_lengths.items() if dist == 3]
            three_hop_community_labels = [self.community_labels[n] for n in three_hop_nodes]

            # Count community labels for each hop distance
            one_hop_counts = Counter(one_hop_community_labels)
            two_hop_counts = Counter(two_hop_community_labels)
            three_hop_counts = Counter(three_hop_community_labels)
            
            # Create feature vector with counts for each community at each hop distance
            feature_vector = []
            for comm in range(n_communities):
                feature_vector.append(one_hop_counts.get(comm, 0))
            for comm in range(n_communities):
                feature_vector.append(two_hop_counts.get(comm, 0))
            for comm in range(n_communities):
                feature_vector.append(three_hop_counts.get(comm, 0))
            
            node_structural_features.append(feature_vector)
            labels.append(self.community_labels[node])

        # Split data ensuring all communities are represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            node_structural_features,
            labels,
            test_size=0.3,
            stratify=labels,
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
        
        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        return float(f1)

    def calculate_degree_signal(self, random_state: int = 42) -> float:
        """
        Calculate Degree Signal using degree-based classification.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            
        Returns:
            Degree signal ∈ [0, 1]
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        
        # Get node degrees and community labels and use RF classifier to calculate degree signal
        degrees = np.array([self.graph.degree(i) for i in range(self.graph.number_of_nodes())])
        community_labels = self.community_labels
        
        # Reshape degrees to 2D array for sklearn (n_samples, n_features)
        degrees_2d = degrees.reshape(-1, 1)
        
        # Split data ensuring all communities are represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            degrees_2d,
            community_labels,
            test_size=0.3,
            stratify=community_labels,
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
        
        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        return float(f1)

    def calculate_triangle_community_signal(self, random_state: int = 42) -> float:
        """Measure how well we can predict triangle participation based on community labels using RF classifier."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        
        # Get triangle counts and community labels
        triangle_counts = []
        community_labels = []

        # Use nx.triangles to get triangle counts
        triangle_counts = list(nx.triangles(self.graph).values())
        community_labels = self.community_labels
        
        # Reshape triangle counts to 2D array for sklearn (n_samples, n_features)
        triangle_counts_2d = np.array(triangle_counts).reshape(-1, 1)
        
        # Split data ensuring all communities are represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            triangle_counts_2d,
            community_labels,
            test_size=0.3,
            stratify=community_labels,
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
        
        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        return float(f1)

    def calculate_community_signals(self,
                                random_state: int = 42) -> Dict[str, Any]:
        """
        Calculate all community-related signal metrics using the unified approach.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with signal values and summary statistics
        """
        signals = {}
        
        # Structure signal
        try:
            structure_signal = self.calculate_structure_signal(random_state=random_state)
            signals['structure_signal'] = structure_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate structure signal: {e}")
            signals['structure_signal'] = 0.0
        
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
                random_state=random_state
            )
            signals['degree_signal'] = degree_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate degree signal: {e}")
            signals['degree_signal'] = 0.0
        
        # Triangle signal
        try:
            triangle_signal = self.calculate_triangle_community_signal(random_state=random_state)
            signals['triangle_signal'] = triangle_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate triangle signal: {e}")
            signals['triangle_signal'] = 0.0
        
        return signals
