import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from graph_universe.feature_regimes import (
    SimplifiedFeatureGenerator
)
import time
import warnings
from scipy.stats import spearmanr, pearsonr
from itertools import combinations
from tqdm import tqdm

class GraphUniverse:
    """
    Represents a generative universe for graph instances sampled from a master "pseudo" stochastic block model.
    The GraphSample class will randomly sub-sample from these global universe properties.
    """
    
    def __init__(
        self,
        # Main Universe Parameters
        K: int,
        P: Optional[np.ndarray] = None,

        # Only used if use_dccc_sbm is True
        edge_probability_variance: float = 0.5, # 0-1: how much variance in the edge probabilities

        # Feature generation parameters
        feature_dim: int = 0,
        center_variance: float = 1.0,       # Separation between cluster centers
        cluster_variance: float = 0.1,      # Spread within each cluster

        # Degree center parameters
        degree_center_method: str = "linear",  # How to generate degree centers ("linear", "random", "constant")
        seed: Optional[int] = None,

        # Community co-occurrence homogeneity
        community_cooccurrence_homogeneity: float = 1.0,  # 0-1: how homogeneous the co-occurrence of communities is
    ):
        """
        Initialize a graph universe with K communities and optional feature generation.
        
        Args:
            K: Number of communities
            P: Optional probability matrix (if None, will be generated)
            edge_probability_variance: Amount of variance in the edge probabilities
            feature_dim: Dimension of node features
            center_variance: Separation between cluster centers
            cluster_variance: Spread within each cluster
            degree_center_method: How to generate degree centers ("linear", "random", "constant")
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
            self.P = self._generate_edge_probability_variance_matrix(
                K, edge_probability_variance
            )
        else:
            self.P = P
            
        # Initialize feature generator if features are enabled
        if feature_dim > 0:
            self.feature_generator = SimplifiedFeatureGenerator(
                universe_K=K,
                feature_dim=feature_dim,
                cluster_count_factor=1.0,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
                assignment_skewness=0.0,
                community_exclusivity=1.0,
                seed=seed
            )
        else:
            self.feature_generator = None
        
        # Store parameters
        self.edge_probability_variance = edge_probability_variance
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
        elif degree_center_method == "constant":
            # Constant value
            self.degree_centers = np.zeros(K)
        else:
            raise ValueError(f"Unknown degree center method: {degree_center_method}")
            
        # Generate community co-occurrence matrix
        self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity
        self.community_cooccurrence_matrix = self._generate_cooccurrence_matrix(K, community_cooccurrence_homogeneity, seed)

    def _generate_edge_probability_variance_matrix(
        self, 
        K: int, 
        edge_probability_variance: float = 0.0
    ) -> np.ndarray:
        """
        Generate a pseudo-probability matrix that gives the RELATIVE probabilities BETWEEN different communities (the intra-community prob / homophily is decided by the GraphFamily object and scaled accordingly in a GraphSample object)
        
        Args:
            K: Number of communities
            edge_probability_variance: Amount of variance in the edge probabilities
            
        Returns:
            K × K probability matrix
        """
            
        P = np.ones((K, K))
        
        # Add the edge probability variance if requested
        if edge_probability_variance > 0:
            noise = np.random.normal(0, edge_probability_variance*2, size=(K, K))
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
        Sample a subset of communities using co-occurrence patterns if cooccurrence_homogeneity is not 1.0.
        
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
        target_homophily: float,
        target_average_degree: float,
        
        degree_distribution: str,
        power_law_exponent: Optional[float],
        max_mean_community_deviation: float,
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
            
            # Store target parameters
            self.target_homophily = target_homophily
            self.target_average_degree = target_average_degree
            
            # Degree distribution parameters
            self.degree_distribution = degree_distribution
            self.power_law_exponent = power_law_exponent

            # Store community deviation parameters as instance attributes
            self.max_mean_community_deviation = max_mean_community_deviation
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
                self.original_n_nodes,
                self.use_dccc_sbm,
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

            self.adjacency = self._connect_disconnected_components(components)

            # Update the number of nodes
            self.graph = nx.from_scipy_sparse_array(self.adjacency)
            self.n_nodes = self.graph.number_of_nodes()

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
  
    def _connect_disconnected_components(self, components: list[set[int]]) -> None:
        """
        Connect disconnected components of the graph iteratively, starting with the smallest component.
        Uses deviation analysis to find optimal connections that bring the actual probability matrix
        closer to the expected P_sub matrix.
        """
        if len(components) <= 1:
            return  # Already connected or empty graph
        
        # Create a copy of the graph to work with
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)
        
        # Sort components by size (smallest first)
        components = sorted(components, key=len)
        
        while len(components) > 1:
            # Get the smallest component
            smallest_component = components[0]
            other_components = components[1:]
            
            # Calculate current deviation matrix for the whole graph
            current_analysis = self._calculate_community_deviations_with_matrix(temp_graph, self.community_labels, self.P_sub)
            current_deviation_matrix = current_analysis['deviation_matrix']
            current_actual_matrix = current_analysis['actual_matrix']
            
            # Find the best connection to make
            best_connection = self._find_best_connection(
                smallest_component, 
                other_components, 
                current_deviation_matrix, 
                current_actual_matrix
            )
            
            if best_connection is None:
                # No valid connection found, remove the smallest component
                components.pop(0)
                continue
            
            # Make the connection
            node_from, node_to = best_connection
            temp_graph.add_edge(node_from, node_to)
            
            # Recalculate components
            new_components = list(nx.connected_components(temp_graph))
            components = sorted(new_components, key=len)
        
        # return the adjacency matrix of the connected graph
        return nx.adjacency_matrix(temp_graph)
        
    def _calculate_community_deviations_with_matrix(self, graph: nx.Graph, community_labels: np.ndarray, P_sub: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate community deviations and return the actual and deviation matrices.
        Uses normalized actual probability matrix for more accurate deviation calculations.
        
        Args:
            graph: NetworkX graph
            community_labels: Array of community labels
            P_sub: Expected probability matrix
            
        Returns:
            Dictionary with actual_matrix and deviation_matrix
        """
        n_communities = P_sub.shape[0]
        actual_matrix = np.zeros((n_communities, n_communities))
        community_sizes = np.zeros(n_communities, dtype=int)
        
        # Count nodes in each community
        for label in community_labels:
            if 0 <= label < n_communities:
                community_sizes[label] += 1
        
        # Count edges between communities
        for i, j in graph.edges():
            comm_i = community_labels[i]
            comm_j = community_labels[j]
            if 0 <= comm_i < n_communities and 0 <= comm_j < n_communities:
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
                    n1, n2 = community_sizes[i], community_sizes[j]
                    if n1 > 0 and n2 > 0:
                        actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
        
        # Normalize actual matrix to match P_sub total mass
        p_sub_total_mass = np.sum(P_sub)
        actual_total_mass = np.sum(actual_matrix)
        
        if actual_total_mass > 0 and p_sub_total_mass > 0:
            # Scale actual matrix to match P_sub total mass
            normalization_factor = p_sub_total_mass / actual_total_mass
            actual_matrix = actual_matrix * normalization_factor
        
        # Calculate deviation matrix
        deviation_matrix = actual_matrix - P_sub
        
        return {
            'actual_matrix': actual_matrix,
            'deviation_matrix': deviation_matrix
        }

    def _find_best_connection(self, smallest_component: set[int], other_components: list[set[int]], 
                             deviation_matrix: np.ndarray, actual_matrix: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the best connection between the smallest component and other components.
        
        Args:
            smallest_component: Set of nodes in the smallest component
            other_components: List of other component node sets
            deviation_matrix: Current deviation matrix (actual - expected)
            actual_matrix: Current actual probability matrix
            
        Returns:
            Tuple of (node_from, node_to) for the best connection, or None if no valid connection
        """
        best_connection = None
        best_score = float('-inf')
        
        # Get communities in the smallest component
        smallest_communities = set(self.community_labels[node] for node in smallest_component)
        
        # Check all possible connections from smallest component to other components
        for other_component in other_components:
            other_communities = set(self.community_labels[node] for node in other_component)
            
            # Check all community pairs between smallest and other component
            for comm_small in smallest_communities:
                for comm_other in other_communities:
                    # Skip if P_sub[comm_small, comm_other] is 0 (no connection allowed)
                    if self.P_sub[comm_small, comm_other] <= 0:
                        continue
                    
                    # Calculate potential improvement score
                    current_deviation = deviation_matrix[comm_small, comm_other]
                    
                    # Prefer negative deviations (actual < expected) - these are good to increase
                    if current_deviation < 0:
                        score = -current_deviation  # Higher score for more negative deviation
                    else:
                        # For positive deviations, prefer smaller ones
                        score = -abs(current_deviation)
                    
                    # If this is better than current best, update
                    if score > best_score:
                        best_score = score
                        # Find a node from each community to connect
                        node_from = self._find_node_in_community(smallest_component, comm_small)
                        node_to = self._find_node_in_community(other_component, comm_other)
                        
                        if node_from is not None and node_to is not None:
                            best_connection = (node_from, node_to)
        
        return best_connection

    def _find_node_in_community(self, component: set[int], community: int) -> Optional[int]:
        """
        Find a node in the given component that belongs to the specified community.
        
        Args:
            component: Set of node indices
            community: Community label
            
        Returns:
            Node index if found, None otherwise
        """
        for node in component:
            if self.community_labels[node] == community:
                return node
        return None

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
            raw_degrees = (np.random.pareto(exponent, size=n_nodes) + 1) ** 1.5 # Slightly more skewed to balance normalization effect
            # raw_degrees = np.random.pareto(exponent, size=n_nodes) + 1
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
                # Scale preserving relative distances - use ordered_degree_centers for scaling
                scaled_centers = (ordered_degree_centers - min_center) / (max_center - min_center)
                # Map back to original community order
                community_means = np.zeros(K)
                for i, comm_idx in enumerate(comm_order):
                    community_means[comm_idx] = scaled_centers[i] * (n_nodes - 1)
        
        # 6. Calculate standard deviation based on degree_separation
        # if degree_separation == 1.0:
        #     # For perfect separation, make std tight enough to avoid overlap
        #     if K > 1:
        #         min_distance = np.min(np.diff(community_means))
        #         community_std = max(1.0, min_distance / 6)  # 3*std = half the minimum distance
        #     else:
        #         community_std = 1.0
        # else:
        #     # Linear interpolation between tight and wide
        max_std = n_nodes / 3  # Wide case
        min_std = 1.0 if K == 1 else max(1.0, np.min(np.diff(community_means)) / 6)  # Tight case
        community_std = min_std + max((1 - degree_separation), 0.1) * (max_std - min_std)
        
        # 7. Create available positions array and sample for each node
        available_positions = list(range(n_nodes))
        degree_factors = np.zeros(n_nodes)
        
        # 8. For each node, sample from its community's truncated normal distribution
        for node_idx in range(n_nodes):
            community_local_idx = community_labels[node_idx]
            
            # Get the mean for this community's distribution (use original community index)
            mean_pos = community_means[community_local_idx]
            std_pos = community_std
            
            # if not available_positions:
            #     # This shouldn't happen, but fallback
            #     degree_factors[node_idx] = sorted_degrees[node_idx % n_nodes]
            #     continue
            
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
        n_nodes: Optional[int] = None,
        use_dccc_sbm: bool = True
    ) -> np.ndarray:
        # Get the number of communities
        n = P_sub.shape[0]

        # If using standard dc_sbm, the inter-community structure is FLAT (uniform). Set the P_sub to ones all and let it be scaled
        if not use_dccc_sbm:
            P_sub = np.ones((n, n))

        # Make copy to avoid modifying the original matrix
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
        Normalizes the actual matrix to match the total probability mass of P_sub
        for more accurate deviation calculations.
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
        
        # Normalize actual matrix to match P_sub total mass
        p_sub_total_mass = np.sum(self.P_sub)
        actual_total_mass = np.sum(actual_matrix)
        
        if actual_total_mass > 0 and p_sub_total_mass > 0:
            # Scale actual matrix to match P_sub total mass
            normalization_factor = p_sub_total_mass / actual_total_mass
            actual_matrix = actual_matrix * normalization_factor
        
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
        
        return {
            "actual_matrix": actual_matrix,
            "expected_matrix": self.P_sub,
            "deviation_matrix": deviation_matrix,
            "mean_deviation": float(mean_deviation),
            "community_sizes": community_sizes,
            "connection_counts": connection_counts,
            "constraints": {  # Keep constraints in output for verification
                "max_mean_deviation": self.max_mean_community_deviation,
            }
        }

    def _calculate_community_deviations(
        self,
        graph: nx.Graph,
        community_labels: np.ndarray,
        P_sub: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate community deviations for a given graph and community labels.
        Uses normalized actual probability matrix for more accurate deviation calculations.
        
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
        
        # Normalize actual matrix to match P_sub total mass
        p_sub_total_mass = np.sum(P_sub)
        actual_total_mass = np.sum(actual_matrix)
        
        if actual_total_mass > 0 and p_sub_total_mass > 0:
            # Scale actual matrix to match P_sub total mass
            normalization_factor = p_sub_total_mass / actual_total_mass
            actual_matrix = actual_matrix * normalization_factor
        
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
            homophily_range: Tuple of (min_homophily, max_homophily) in graph family
            density_range: Tuple of (min_density, max_density) in graph family

            # Whether to use DCCC-SBM or standard DC-SBM
            use_dccc_sbm: Whether to use DCCC-SBM model

            # DCCC distribution-specific parameters
            degree_separation_range: Range for degree distribution separation (min, max)
            degree_signal_calc_method: How to calculate degree signal
            disable_deviation_limiting: Whether to disable deviation checks
            max_mean_community_deviation: Maximum allowed mean target scaled edge probability community deviation
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
        starting_time_new_graph = start_time
        timeout_seconds = timeout_minutes * 60
        self.graphs = []
        self.community_labels_per_graph = []
        self.generation_metadata = []
        self.graph_generation_times = []
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
                        target_homophily=params['target_homophily'],
                        target_average_degree=params['target_average_degree'],
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        max_mean_community_deviation=self.max_mean_community_deviation,
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

                    graph_generation_time = time.time() - starting_time_new_graph
                    self.graph_generation_times.append(graph_generation_time)
                    starting_time_new_graph = time.time()
                    
                    if show_progress:
                        pbar.update(1)
                    
                except Exception as e:
                    print(f"Failed to generate graph after {attempts} attempts: {e}")
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
                        target_homophily=params['target_homophily'],
                        target_average_degree=params['target_average_degree'],
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        max_mean_community_deviation=self.max_mean_community_deviation,
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
                # print(f"Attempt {attempts} failed: {str(e)}")
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
            'generation_methods': [],
            'degree_distributions': [],
            'degree_distribution_power_law_exponents': [],
            'tail_ratio_95': [],
            'tail_ratio_99': [],
            'graph_generation_times': self.graph_generation_times
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
            
            # Extract degree distribution and fit power law exponent
            if graph.n_nodes > 0:
                degrees = list(dict(graph.graph.degree()).values())
                # Fit power law exponent
                power_law_exponent = self._fit_power_law_exponent(degrees)
                tail_metrics = self._calculate_degree_tail_metrics(degrees)
                properties['tail_ratio_95'].append(tail_metrics['tail_ratio_95'])
                properties['tail_ratio_99'].append(tail_metrics['tail_ratio_99'])
                properties['degree_distributions'].append(degrees)
                properties['degree_distribution_power_law_exponents'].append(power_law_exponent)
            else:
                properties['degree_distributions'].append([])
                properties['degree_distribution_power_law_exponents'].append(0.0)

        # Calculate statistics and convert to native Python types
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels', 'nr_of_triangles', 'graph_generation_times']:
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
   
    def analyze_graph_family_signals(self) -> Dict[str, Any]:
        """Analyze the signals of the graph family."""
        if not self.graphs:
            raise ValueError("No graphs in family. Please generate family first before analyzing signals.")
        
        # Calculate the signals
        signals = {
            'feature_signal': [],
            'degree_signal': [],
            'triangle_signal': [],
            'structure_signal': []
        }
        
        for graph in self.graphs:
            signals['feature_signal'].append(graph.calculate_feature_signal())
            signals['degree_signal'].append(graph.calculate_degree_signal())
            signals['triangle_signal'].append(graph.calculate_triangle_community_signal())
            signals['structure_signal'].append(graph.calculate_structure_signal())
        
        return signals

    def analyze_graph_family_consistency(self) -> Dict[str, Any]:
        """Analyze the consistency of the graph family."""
        if not self.graphs:
            raise ValueError("No graphs in family. Please generate family first before analyzing consistency.")
        
        # Calculate the consistency
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

        # 4. Co-occurrence consistency (do communities co-occur in the family as expected?)
        try:
            cooccurrence_consistency = self._calculate_cooccurrence_consistency()
            results['cooccurrence_consistency'] = cooccurrence_consistency
        except Exception as e:
            results['cooccurrence_consistency'] = 0.0

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

                # Get the degree centers for the communities in the graph
                community_degree_centers = degree_centers[graph.communities]

                # Check if degree centers are constant (all same value)
                if np.std(community_degree_centers) == 0:
                    # For constant degree centers, measure degree homogeneity instead
                    # Calculate coefficient of variation of actual degrees per community
                    if np.mean(actual_degrees_per_community) > 0:
                        cv = np.std(actual_degrees_per_community) / np.mean(actual_degrees_per_community)
                        # Convert to a "consistency" score: lower CV = higher consistency
                        # Use 1 / (1 + cv) to get a score between 0 and 1
                        consistency_score = 1.0 / (1.0 + cv)
                    else:
                        consistency_score = 0.0
                else:
                    # Calculate the correlation between the actual degrees and the community degree centers
                    correlation, _ = pearsonr(actual_degrees_per_community, community_degree_centers)
                    consistency_score = correlation if not np.isnan(correlation) else 0.0
                
                consistency_scores.append(consistency_score)
                    
            except Exception as e:
                warnings.warn(f"Error in degree consistency calculation for graph: {e}")
                continue
        
        if not consistency_scores:
            return []
        
        return consistency_scores

    def _calculate_cooccurrence_consistency(self) -> List[float]:
        """
        Measure how well community co-occurrence patterns are preserved.
        """        
        # Calculate how often communities co-occur in the family
        cooccurrence_counts_matrix = np.zeros((self.universe.K, self.universe.K))
        for graph in self.graphs:
            for i in range(len(graph.communities)):
                for j in range(i+1, len(graph.communities)):
                    cooccurrence_counts_matrix[graph.community_id_mapping[i], graph.community_id_mapping[j]] += 1
                    cooccurrence_counts_matrix[graph.community_id_mapping[j], graph.community_id_mapping[i]] += 1

        # Calculate how correlated the cooccurrence counts are with the universe cooccurrence matrix
        correlation, _ = pearsonr(cooccurrence_counts_matrix.flatten(), self.universe.community_cooccurrence_matrix.flatten())
        
        return correlation
        
    def _fit_power_law_exponent(self, degrees: List[int]) -> float:
        """
        Fit a power law exponent to a degree distribution using discrete MLE.
        This method is specifically designed for discrete network degrees.
        
        Args:
            degrees: List of node degrees
            
        Returns:
            float: Power law exponent alpha, or 0 if fitting fails or alpha not in [1, 20]
        """
        if not degrees or len(degrees) < 2:
            return 0.0
            
        # Convert to numpy array and filter out zeros
        degrees_array = np.array(degrees, dtype=int)
        degrees_array = degrees_array[degrees_array > 0]
        
        if len(degrees_array) < 2:
            return 0.0
        
        k_min = np.min(degrees_array)
        n = len(degrees_array)
        
        def negative_log_likelihood(alpha):
            if alpha <= 1.0:
                return np.inf
            try:
                # For discrete power law, the log-likelihood is:
                # L = -alpha * sum(log(k_i)) - n * log(zeta(alpha, k_min))
                # We approximate zeta(alpha, k_min) for computational efficiency
                
                # Approximation: zeta(alpha, k_min) ≈ sum_{k=k_min}^{k_max} k^(-alpha)
                k_max = max(100, np.max(degrees_array) * 2)  # Reasonable upper bound
                k_range = np.arange(k_min, k_max + 1)
                zeta_approx = np.sum(k_range**(-alpha))
                
                if zeta_approx <= 0:
                    return np.inf
                    
                log_likelihood = -alpha * np.sum(np.log(degrees_array)) - n * np.log(zeta_approx)
                return -log_likelihood  # Return negative for minimization
                
            except (OverflowError, ZeroDivisionError, ValueError):
                return np.inf
        
        try:
            # Find MLE estimate using bounded optimization
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(negative_log_likelihood, bounds=(1.01, 10.0), method='bounded')
            if result.success and 1.0 < result.x < 20.0:
                return result.x
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_degree_tail_metrics(self, degrees: List[int]) -> Dict[str, float]:
        """Calculate tail-based degree metrics instead of power law fitting"""        
        if len(degrees) == 0:
            return {'tail_ratio': 0.0, 'cv': 0.0}
        
        tail_95 = np.percentile(degrees, 95)
        tail_99 = np.percentile(degrees, 99) 
        mean_degree = np.mean(degrees)
        
        return {
            'tail_ratio_95': tail_95 / mean_degree if mean_degree > 0 else 0.0,
            'tail_ratio_99': tail_99 / mean_degree if mean_degree > 0 else 0.0,
            'coefficient_variation': np.std(degrees) / mean_degree if mean_degree > 0 else 0.0,
            'max_degree': np.max(degrees),
            'mean_degree': mean_degree
        }
