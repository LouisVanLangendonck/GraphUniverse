import time
import warnings
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.data as pyg
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected

from graph_universe.graph_universe import GraphUniverse


class PositionalEncodingComputer:
    """Compute various types of positional encodings for graphs."""

    def __init__(self, max_pe_dim: int = 10, pe_types: list[str] | None = None):
        """
        Initialize PE computer.

        Args:
            max_pe_dim: Maximum PE dimension
            pe_types: List of PE types to compute ['laplacian', 'degree', 'rwse']
        """
        self.max_pe_dim = max_pe_dim
        self.pe_types = pe_types or ["laplacian"]

    def compute_degree_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Degree-based PE."""
        from torch_geometric.utils import degree

        degrees = degree(edge_index[0], num_nodes=num_nodes).float()
        pe = torch.zeros(num_nodes, self.max_pe_dim)

        for i in range(min(self.max_pe_dim, 8)):
            pe[:, i] = (degrees ** (i / 4.0)) / (1 + degrees ** (i / 4.0))

        return pe

    def compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Random Walk Structural Encoding - landing probabilities after k steps."""
        try:
            from torch_geometric.utils import degree

            # Get node degrees
            degrees = degree(edge_index[0], num_nodes=num_nodes).float()

            # Handle isolated nodes
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)

            # Create adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0

            # Transition matrix: P[i,j] = A[i,j] / degree[i]
            P = adj / degrees.unsqueeze(1)

            # Compute powers of transition matrix for different walk lengths
            rwse = torch.zeros(num_nodes, self.max_pe_dim)
            P_power = torch.eye(num_nodes)  # P^0 = I

            for k in range(self.max_pe_dim):
                if k > 0:
                    P_power = P_power @ P  # P^k

                # Use diagonal entries (return probabilities) as features
                rwse[:, k] = P_power.diag()

            return rwse

        except Exception as e:
            print(f"Warning: RWSE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)

    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Laplacian Positional Encoding using eigenvectors."""
        try:
            # Handle empty/trivial graphs
            if edge_index.shape[1] == 0 or num_nodes <= 1:
                return torch.zeros(num_nodes, self.max_pe_dim)

            # Get normalized Laplacian
            edge_index_lap, edge_weight = get_laplacian(
                edge_index, edge_weight=None, normalization="sym", num_nodes=num_nodes
            )

            # Convert to scipy sparse matrix
            L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)

            # Compute eigenvalues/eigenvectors
            k = min(self.max_pe_dim, num_nodes - 2)
            if k <= 0:
                return torch.zeros(num_nodes, self.max_pe_dim)

            try:
                eigenvals, eigenvecs = eigsh(
                    L,
                    k=k,
                    which="SM",  # Smallest eigenvalues
                    return_eigenvectors=True,
                    tol=1e-6,
                )
            except ArpackNoConvergence:
                # Fallback for small graphs
                L_dense = L.toarray()
                eigenvals, eigenvecs = np.linalg.eigh(L_dense)
                idx = np.argsort(eigenvals)
                eigenvecs = eigenvecs[:, idx[1 : k + 1]]  # Skip first (constant) eigenvector

            # Handle sign ambiguity
            for i in range(eigenvecs.shape[1]):
                if eigenvecs[0, i] < 0:
                    eigenvecs[:, i] *= -1

            # Pad or truncate to max_pe_dim
            if eigenvecs.shape[1] < self.max_pe_dim:
                pad_width = self.max_pe_dim - eigenvecs.shape[1]
                eigenvecs = np.pad(eigenvecs, ((0, 0), (0, pad_width)), mode="constant")
            else:
                eigenvecs = eigenvecs[:, : self.max_pe_dim]

            return torch.tensor(eigenvecs, dtype=torch.float32)

        except Exception as e:
            print(f"Warning: Laplacian PE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)

    def compute_all_pe(self, edge_index: torch.Tensor, num_nodes: int) -> dict[str, torch.Tensor]:
        """Compute all requested PE types."""
        pe_dict = {}

        for pe_type in self.pe_types:
            if pe_type == "laplacian":
                pe = self.compute_laplacian_pe(edge_index, num_nodes)
                pe_dict["laplacian_pe"] = pe

            elif pe_type == "degree":
                pe = self.compute_degree_pe(edge_index, num_nodes)
                pe_dict["degree_pe"] = pe

            elif pe_type == "rwse":
                pe = self.compute_rwse(edge_index, num_nodes)
                pe_dict["rwse_pe"] = pe

            else:
                raise ValueError(f"Invalid PE type: {pe_type}")

        return pe_dict


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
        power_law_exponent: float | None,
        max_mean_community_deviation: float,
        # Whether to use the poisson version of the DC-SBM (i.e. the edge count matrix is scaled to match the target average degree)
        poisson_version: bool = False,
        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        degree_heterogeneity: float | None = None,
        # DCCC-SBM parameters
        use_dccc_sbm: bool = True,
        degree_separation: float = 0.5,
        dccc_global_degree_params: dict | None = None,
        enable_deviation_limiting: bool = False,
        # Random seed
        seed: int | None = None,
        # Optional Parameter for user-defined communuties to be sampled
        user_defined_communities: list[int] | None = None,
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
        self.enable_deviation_limiting = enable_deviation_limiting  # Store the parameter

        # Original initialization code with modifications...
        self.timing_info = {}
        total_start = time.time()

        # Add timeout mechanism
        TIMEOUT_SECONDS = 60

        def check_timeout():
            if time.time() - total_start > TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"GraphSample initialization timed out after {TIMEOUT_SECONDS} seconds"
                )

        try:
            # Sample communities from universe or use user-defined communities
            if user_defined_communities is not None:
                self.communities = user_defined_communities
            else:
                check_timeout()
                self.communities = universe.sample_connected_community_subset(
                    num_communities, seed=seed, use_cooccurrence=True
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

            # Create mapping between local community indices and universe community IDs
            self.community_id_mapping = dict(enumerate(self.communities))
            self.reverse_community_id_mapping = {
                comm_id: i for i, comm_id in self.community_id_mapping.items()
            }

            # Initialize generation method and parameters
            self.generation_method = "standard"
            self.generation_params = {
                "degree_heterogeneity": degree_heterogeneity,
                "max_mean_community_deviation": max_mean_community_deviation,
            }

            # If DCCC-SBM is enabled, update generation method
            if self.use_dccc_sbm:
                self.generation_method = "dccc_sbm"
                self.generation_params.update(
                    {
                        "degree_separation": degree_separation,
                        "degree_distribution_type": degree_distribution,
                    }
                )
                if degree_distribution == "power_law":
                    self.generation_params["power_law_exponent"] = power_law_exponent

            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)

            check_timeout()

            # Time: Extract and scale probability matrix
            start = time.time()

            # Extract the submatrix of the probability matrix for these communities
            K_sub = len(self.communities)  # Number of communities in the sample
            self.P_sub = np.zeros(
                (K_sub, K_sub)
            )  # Initialize the probability matrix for the sample
            for i, ci in enumerate(sorted(self.communities)):
                for j, cj in enumerate(sorted(self.communities)):
                    self.P_sub[i, j] = universe.P[ci, cj]

            # Scale the probability matrix
            if poisson_version:
                self.P_sub = self._scale_edge_propensity_to_edge_count(
                    self.P_sub,
                    self.target_average_degree,
                    self.target_homophily,
                    self.original_n_nodes,
                    self.use_dccc_sbm,
                )
            else:
                self.P_sub = self._scale_edge_propensity_to_edge_probability(
                    self.P_sub,
                    self.target_average_degree,
                    self.target_homophily,
                    self.original_n_nodes,
                    self.use_dccc_sbm,
                )
            self.timing_info["propensity_matrix_scaling"] = time.time() - start

            check_timeout()

            # Time: Generate memberships
            start = time.time()

            # Uniform membership generation
            self.community_labels = self._generate_memberships(
                n_nodes, K_sub
            )  # Now returns 1D array

            # Create a new array that maps the community labels to the universe community IDs
            self.community_labels_universe_level = np.array(
                [self.community_id_mapping[idx] for idx in self.community_labels]
            )

            # Store membership generation time
            self.timing_info["memberships"] = time.time() - start

            check_timeout()

            # Time: Generate degree factors
            start = time.time()
            if self.use_dccc_sbm:
                # For DCCC-SBM, generate community-coupled degree factors
                global_degree_params = {}
                if degree_distribution == "power_law":
                    global_degree_params = {"exponent": power_law_exponent, "x_min": 1.0}
                elif degree_distribution == "exponential":
                    global_degree_params = {"rate": getattr(self, "rate", 0.5)}
                elif degree_distribution == "uniform":
                    global_degree_params = {
                        "min_degree": getattr(self, "min_factor", 0.5),
                        "max_degree": getattr(self, "max_factor", 1.5),
                    }

                # Update with any user-provided parameters
                if dccc_global_degree_params:
                    global_degree_params.update(dccc_global_degree_params)

                # Generate community-specific degree factors
                self.degree_factors = self._generate_community_degree_factors(
                    self.community_labels,
                    degree_distribution,
                    degree_separation,
                    global_degree_params,
                    poisson_version,
                )

            else:
                self.degree_factors = self._generate_degree_factors(n_nodes, degree_heterogeneity)
            self.timing_info["degree_factors"] = time.time() - start

            check_timeout()

            # Time: Generate edges
            start = time.time()
            if poisson_version:
                self.adjacency = self._generate_edges_poisson(
                    self.community_labels,
                    self.P_sub,
                    self.degree_factors,
                )
            else:
                self.adjacency = self._generate_edges(
                    self.community_labels,
                    self.P_sub,
                    self.degree_factors,
                )
            self.timing_info["edge_generation"] = time.time() - start

            check_timeout()

            # Create initial NetworkX graph
            # print(self.adjacency)
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

            if self.enable_deviation_limiting:
                deviations = self._calculate_community_deviations(
                    self.graph, self.community_labels, self.P_sub
                )
                mean_deviation = deviations["mean_deviation"]

                if mean_deviation > self.max_mean_community_deviation:
                    raise ValueError(
                        f"Graph exceeds mean community deviation limit: {mean_deviation:.4f} > {self.max_mean_community_deviation:.4f}"
                    )

            check_timeout()

            # Time: Feature generation
            start = time.time()
            if universe.feature_dim > 0:
                # Get community assignments and map to universe community IDs
                universe_community_assignments = np.array(self.community_labels_universe_level)

                # Generate node clusters based on universe community assignments
                self.node_clusters = universe.feature_generator.assign_node_clusters(
                    universe_community_assignments
                )

                # Generate features based on node clusters
                self.features = universe.feature_generator.generate_node_features(
                    self.node_clusters
                )

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
            self.timing_info["feature_generation"] = time.time() - start

            # Store total time
            self.timing_info["total"] = time.time() - total_start

        except TimeoutError as err:
            raise TimeoutError("GraphSample initialization timed out") from err

    def _connect_disconnected_components(self, components: list[set[int]]) -> sp.spmatrix:
        """
        Connect disconnected components of the graph iteratively, starting with the smallest component.
        Uses deviation analysis to find optimal connections that bring the actual probability matrix
        closer to the expected P_sub matrix.
        """
        if len(components) <= 1:
            # Already connected or empty graph; return current adjacency unchanged
            return self.adjacency

        # Create a copy of the graph to work with
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)

        # Sort components by size (smallest first)
        components = sorted(components, key=len)

        while len(components) > 1:
            # Get the smallest component
            smallest_component = components[0]
            other_components = components[1:]

            # Calculate current deviation matrix for the whole graph
            current_analysis = self._calculate_community_deviations_with_matrix(
                temp_graph, self.community_labels, self.P_sub
            )
            current_deviation_matrix = current_analysis["deviation_matrix"]
            current_actual_matrix = current_analysis["actual_matrix"]

            # Find the best connection to make
            best_connection = self._find_best_connection(
                smallest_component,
                other_components,
                current_deviation_matrix,
                current_actual_matrix,
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

    def _calculate_community_deviations_with_matrix(
        self, graph: nx.Graph, community_labels: np.ndarray, P_sub: np.ndarray
    ) -> dict[str, np.ndarray]:
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

        return {"actual_matrix": actual_matrix, "deviation_matrix": deviation_matrix}

    def _find_best_connection(
        self,
        smallest_component: set[int],
        other_components: list[set[int]],
        deviation_matrix: np.ndarray,
        actual_matrix: np.ndarray,
    ) -> tuple[int, int] | None:
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
        best_score = float("-inf")

        # Get communities in the smallest component
        smallest_communities = {self.community_labels[node] for node in smallest_component}

        # Check all possible connections from smallest component to other components
        for other_component in other_components:
            other_communities = {self.community_labels[node] for node in other_component}

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

    def _find_node_in_community(self, component: set[int], community: int) -> int | None:
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

    def _generate_degree_factors(
        self, n_nodes: int, heterogeneity: float, poisson_version: bool
    ) -> np.ndarray:
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

    def _generate_community_degree_factors(
        self,
        community_labels: np.ndarray,
        degree_distribution_type: str,
        degree_separation: float,
        global_degree_params: dict,
        poisson_version: bool = True,
    ) -> np.ndarray:
        n_nodes = len(community_labels)

        # 1. Sample global degree distribution
        if degree_distribution_type == "power_law":
            exponent = global_degree_params.get("exponent", 2.5)
            raw_degrees = (
                np.random.pareto(exponent, size=n_nodes) + 1
            )  # ** 1.5 (if added the scaler: Slightly more skewed to balance normalization effect
            # raw_degrees = np.random.pareto(exponent, size=n_nodes) + 1
        elif degree_distribution_type == "exponential":
            rate = global_degree_params.get("rate", 1.0)
            raw_degrees = np.random.exponential(scale=1 / rate, size=n_nodes)
        elif degree_distribution_type == "uniform":
            low = global_degree_params.get("min_degree", 1.0)
            high = global_degree_params.get("max_degree", 10.0)
            raw_degrees = np.random.uniform(low, high, size=n_nodes)
        else:
            raise ValueError("Unknown distribution type")

        # 2. Sort degrees
        sorted_degrees = np.sort(raw_degrees)

        # 3. Get universe community-degree propensity vector for our communities
        k = len(self.communities)
        universe_degree_centers = np.array(
            [
                self.universe.community_degree_propensity_vector[
                    self.community_id_mapping[local_comm_id]
                ]
                for local_comm_id in range(k)
            ]
        )

        # 4. Order communities by universe degree center (lowest to highest)
        comm_order = np.argsort(universe_degree_centers)
        ordered_degree_centers = universe_degree_centers[comm_order]

        # 5. Map universe degree centers to positions preserving relative distances
        if k == 1:
            community_means = [n_nodes // 2]
        else:
            # Scale the actual degree center values to [0, n_nodes-1] range
            min_center = np.min(ordered_degree_centers)
            max_center = np.max(ordered_degree_centers)

            if max_center == min_center:
                # All centers are the same, spread evenly
                community_means = np.linspace(0, n_nodes - 1, k)
            else:
                # Scale preserving relative distances - use ordered_degree_centers for scaling
                scaled_centers = (ordered_degree_centers - min_center) / (max_center - min_center)
                # Map back to original community order
                community_means = np.zeros(k)
                for i, comm_idx in enumerate(comm_order):
                    community_means[comm_idx] = scaled_centers[i] * (n_nodes - 1)

        max_std = n_nodes  # Wide case
        min_std = 1.0 if k == 1 else max(1.0, np.min(np.diff(community_means)) / 6)  # Tight case
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

            # Sample from truncated normal distribution over available positions
            if len(available_positions) == 1:
                # Only one position left, take it
                chosen_pos = available_positions[0]
            else:
                # Calculate probabilities for each available position based on normal distribution
                probabilities = np.array(
                    [
                        np.exp(-0.5 * ((pos - mean_pos) / std_pos) ** 2)
                        for pos in available_positions
                    ]
                )
                probabilities = probabilities / np.sum(probabilities)  # Normalize

                # Sample from available positions based on probabilities
                chosen_idx = np.random.choice(len(available_positions), p=probabilities)
                chosen_pos = available_positions[chosen_idx]

            # Assign the degree and remove position from available
            degree_factors[node_idx] = sorted_degrees[chosen_pos]
            available_positions.remove(chosen_pos)

        if poisson_version:
            # Degree factor sum-to-1 normalization to minimize effect on expected edge count PER COMMUNITY
            for community_local_idx in range(k):
                community_nodes = np.where(community_labels == community_local_idx)[0]
                community_degree_factors = degree_factors[community_nodes]
                community_degree_factors_sum = np.sum(community_degree_factors)
                degree_factors[community_nodes] = (
                    community_degree_factors / community_degree_factors_sum
                )

        else:
            degree_factors_mean = np.mean(degree_factors)
            degree_factors = degree_factors / degree_factors_mean
            # # Degree factor mean normalization to minimize effect on expected edge count PER COMMUNITY
            # for community_local_idx in range(k):
            #     community_nodes = np.where(community_labels == community_local_idx)[0]
            #     community_degree_factors = degree_factors[community_nodes]
            #     community_degree_factors_mean = np.mean(community_degree_factors)
            #     degree_factors[community_nodes] = community_degree_factors / community_degree_factors_mean

        return degree_factors

    def _generate_edges_poisson(
        self,
        community_labels: np.ndarray,
        Lambda_sub: np.ndarray,
        degree_factors: np.ndarray,
        simple_graph: bool = True,
    ) -> sp.spmatrix:
        """
        Generate edges using Poisson DC-SBM.

        Args:
            community_labels: Node community assignments (indices)
            Lambda_sub: Community-community expected edge count matrix
            degree_factors: Node degree factors
            simple_graph: If True, collapse multiedges to single edges

        Returns:
            Sparse adjacency matrix
        """
        n_nodes = len(community_labels)

        # Upper-triangular node pairs
        i_nodes, j_nodes = np.triu_indices(n_nodes, k=1)
        comm_i = community_labels[i_nodes]
        comm_j = community_labels[j_nodes]

        # Poisson mean parameter
        lam = Lambda_sub[comm_i, comm_j] * degree_factors[i_nodes] * degree_factors[j_nodes]

        # Sample edge counts
        edge_counts = np.random.poisson(lam)

        if simple_graph:
            mask = edge_counts > 0
            rows, cols = i_nodes[mask], j_nodes[mask]
            data = np.ones(len(rows) * 2)
        else:
            # Keep multiplicities: repeat indices according to count
            rows = np.repeat(i_nodes, edge_counts)
            cols = np.repeat(j_nodes, edge_counts)
            data = np.ones(len(rows) * 2)

        # Make undirected adjacency
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])

        adj = sp.csr_matrix((data, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
        return adj

    def _generate_edges(
        self,
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
    ) -> sp.spmatrix:
        """
        Generate edges with minimum density guarantee.
        Uses vectorized operations for faster edge generation with community labels.

        Args:
            community_labels: Node community assignments (indices)
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors

        Returns:
            Sparse adjacency matrix
        """
        n_nodes = len(community_labels)

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

        return adj

    def _scale_edge_propensity_to_edge_count(
        self,
        P_sub: np.ndarray,
        target_avg_degree: float,
        target_homophily: float,
        n_nodes: int,
        use_dccc_sbm: bool = True,
    ) -> np.ndarray:
        """
        Scale edge propensity matrix so entries represent expected edge counts (Poisson DC-SBM form).
        """
        n = P_sub.shape[0]
        if not use_dccc_sbm:
            P_sub = np.ones((n, n))

        P_scaled = P_sub.copy()

        # Total expected edges in the graph
        target_total_edges = n_nodes * target_avg_degree

        # Diagonal vs off-diagonal masks
        diagonal_mask = np.eye(n, dtype=bool)
        off_diagonal_mask = ~diagonal_mask

        diagonal_sum = np.sum(P_sub[diagonal_mask])
        off_diagonal_sum = np.sum(P_sub[off_diagonal_mask])

        # Split total edge budget by homophily
        target_diag_sum = target_homophily * target_total_edges
        target_off_sum = (1 - target_homophily) * target_total_edges

        # Scale so that diagonals/off-diagonals match
        diag_scale = target_diag_sum / diagonal_sum if diagonal_sum > 0 else 0
        off_scale = target_off_sum / off_diagonal_sum if off_diagonal_sum > 0 else 0

        P_scaled[diagonal_mask] *= diag_scale
        P_scaled[off_diagonal_mask] *= off_scale

        return P_scaled

    def _scale_edge_propensity_to_edge_probability(
        self,
        P_sub: np.ndarray,
        target_avg_degree: float | None = None,
        target_homophily: float | None = None,
        n_nodes: int | None = None,
        use_dccc_sbm: bool = True,
    ) -> np.ndarray:
        n = P_sub.shape[0]

        # If using standard dc_sbm, the inter-community structure is FLAT (uniform). Set the P_sub to ones all and let it be scaled
        if not use_dccc_sbm:
            P_sub = np.ones((n, n))

        # Make copy to avoid modifying the original matrix
        P_scaled = P_sub.copy()

        # Convert target average degree to equivalent density
        # avg_degree = (2 * edges / n_nodes) / 2 -> avg_degree = edges / n_nodes -> extra /2 because undirected graph
        # density = edges / (n_nodes * (n_nodes - 1) / 2)
        # Therefore: density = 2 * avg_degree / (n_nodes - 1)
        target_density = target_avg_degree / (n_nodes - 1)
        # These extra *2 factors are more internal to the practicalities of the graph libraries and not included in paper formulas. Empirically validated though.

        # Create masks for diagonal and off-diagonal elements
        diagonal_mask = np.eye(n, dtype=bool)
        off_diagonal_mask = ~diagonal_mask

        # Get current values - no clipping yet
        diagonal_elements = P_sub[diagonal_mask]
        off_diagonal_elements = P_sub[off_diagonal_mask]

        # Calculate current sums
        diagonal_sum = np.sum(diagonal_elements)
        off_diagonal_sum = np.sum(off_diagonal_elements)

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

        # Now ensure all probabilities are in [0, 1] for actual graph generation
        P_scaled = np.clip(P_scaled, 0, 1)
        # print(P_scaled)

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

    def to_pyg_graph(
        self,
        tasks: str | list[str],
        pe_types: list[str] | None = None,
        pe_dim: int = 10,
    ) -> pyg.data.Data:
        """
        Convert the graph to a PyG graph including all specified tasks as properties.
        Possible tasks are:
        - "community_detection"
        - "triangle_counting"
        - "k_hop_community_counts_k{N}" (where N is the hop count, e.g., "k_hop_community_counts_k2")

        Args:
            tasks: Single task string or list of task strings to include
            pe_types: List of positional encoding types to compute
            pe_dim: Dimension of the positional encodings

        Returns:
            PyG Data object with task results and positional encodings stored as properties
        """
        if pe_types is None:
            pe_types = ["laplacian", "degree", "rwse"]
        self.pe_types = pe_types

        # Handle single task input
        if isinstance(tasks, str):
            tasks = [tasks]

        graph = self.graph
        edges = list(graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        # Use features from GraphSample if available, else raise error
        if hasattr(self, "features") and self.features is not None:
            features = torch.tensor(self.features, dtype=torch.float)
        else:
            raise ValueError("Features are not available for the graph")

        # Create base Data object
        data = Data(x=features, edge_index=edge_index)

        # Process each task and add as property
        for task in tasks:
            if task == "community_detection":
                if (
                    hasattr(self, "community_labels_universe_level")
                    and self.community_labels_universe_level is not None
                ):
                    data.community_detection = torch.tensor(
                        self.community_labels_universe_level, dtype=torch.long
                    )
                else:
                    raise ValueError("Community labels are not available for the graph")

            elif task == "triangle_counting":
                # Triangle counting (via networkx then to tensor)
                triangle_count = self.count_triangles_graph()
                data.triangle_counting = torch.tensor(triangle_count, dtype=torch.float)

            elif task.startswith("k_hop_community_counts_k"):
                # Extract k value from task name
                k = int(task.split("k")[-1])
                # K-hop community counting - universe-indexed
                k_hop_counts = self.compute_khop_community_counts_universe_indexed(k)
                k_hop_counts_binary = (k_hop_counts > 0).float()
                task_binary = task + "_binary"
                setattr(data, task, k_hop_counts)
                setattr(data, task_binary, k_hop_counts_binary)

            else:
                raise ValueError(f"Unknown task: {task}")

        # Set y to the first task's result for backward compatibility (if only one task)
        if len(tasks) == 1:
            data.y = getattr(data, tasks[0])

        # Compute positional encodings
        # pe_dict = self.compute_positional_encodings(pyg_graph=data, pe_types=pe_types, pe_dim=pe_dim)
        # for pe_type, pe in pe_dict.items():
        #     setattr(data, pe_type, pe)

        return data

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

        return actual_matrix, community_sizes, connection_counts

    def analyze_community_connections(self) -> dict[str, Any]:
        """
        Analyze community connection patterns and deviations from expected probabilities.

        Returns:
            Dictionary with analysis results
        """
        actual_matrix, community_sizes, connection_counts = (
            self.calculate_actual_probability_matrix()
        )

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
            },
        }

    def _calculate_community_deviations(
        self, graph: nx.Graph, community_labels: np.ndarray, P_sub: np.ndarray
    ) -> dict[str, float]:
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
                raise ValueError(
                    f"Invalid community labels {comm_i}, {comm_j} for {n_communities} communities"
                )

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

        # # Normalize actual matrix to match P_sub total mass
        # p_sub_total_mass = np.sum(P_sub)
        # actual_total_mass = np.sum(actual_matrix)

        # if actual_total_mass > 0 and p_sub_total_mass > 0:
        #     # Scale actual matrix to match P_sub total mass
        #     normalization_factor = p_sub_total_mass / actual_total_mass
        #     actual_matrix = actual_matrix * normalization_factor

        # Calculate deviations
        deviation_matrix = np.abs(actual_matrix - P_sub)
        mean_deviation = np.mean(deviation_matrix)
        max_deviation = np.max(deviation_matrix)

        return {"mean_deviation": mean_deviation, "max_deviation": max_deviation}

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
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        # Get features and labels
        X = self.features
        y = self.community_labels

        # Split data ensuring all communities are represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )

        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate macro F1 score
        f1 = f1_score(y_test, y_pred, average="macro")

        return float(f1)

    def calculate_structure_signal(self, random_state: int = 42) -> float:
        """
        Calculate Structure Signal using RF classifier by using as input of node the community label counts of its 1-hop neighbors, concatted by 2-hop neighbors and 3-hop neighbors.
        """
        from collections import Counter

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

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
            random_state=random_state,
        )

        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average="macro")
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
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

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
            random_state=random_state,
        )

        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average="macro")
        return float(f1)

    def calculate_triangle_community_signal(self, random_state: int = 42) -> float:
        """Measure how well we can predict triangle participation based on community labels using RF classifier."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

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
            random_state=random_state,
        )

        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate f1 score
        f1 = f1_score(y_test, y_pred, average="macro")
        return float(f1)

    def calculate_community_signals(self, random_state: int = 42) -> dict[str, Any]:
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
            signals["structure_signal"] = structure_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate structure signal: {e}", stacklevel=2)
            signals["structure_signal"] = 0.0

        # Feature signal (only if features are available)
        if self.features is not None and len(self.features) > 0:
            try:
                feature_signal = self.calculate_feature_signal(random_state=random_state)
                signals["feature_signal"] = feature_signal
            except Exception as e:
                warnings.warn(f"Failed to calculate feature signal: {e}", stacklevel=2)
                signals["feature_signal"] = 0.0
        else:
            signals["feature_signal"] = None

        # Degree signal
        try:
            degree_signal = self.calculate_degree_signal(random_state=random_state)
            signals["degree_signal"] = degree_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate degree signal: {e}", stacklevel=2)
            signals["degree_signal"] = 0.0

        # Triangle signal
        try:
            triangle_signal = self.calculate_triangle_community_signal(random_state=random_state)
            signals["triangle_signal"] = triangle_signal
        except Exception as e:
            warnings.warn(f"Failed to calculate triangle signal: {e}", stacklevel=2)
            signals["triangle_signal"] = 0.0

        return signals

    def count_triangles_graph(self) -> int:
        # Use networkx's triangles function which is more reliable
        triangle_counts = nx.triangles(self.graph)
        # Sum all triangle counts and divide by 3 (each triangle is counted 3 times)
        total_triangles = sum(triangle_counts.values()) // 3
        return total_triangles

    def compute_khop_community_counts_universe_indexed(self, k: int) -> torch.Tensor:
        """
        Compute k-hop community counts (only nodes at exactly k-hops) with universe indexing.
        """
        n_nodes = self.graph.number_of_nodes()
        counts = np.zeros((n_nodes, self.universe.K))

        for node in range(n_nodes):
            # Get nodes at exact distance k using single-source shortest path
            sp_lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=k)
            khop_nodes = [n for n, dist in sp_lengths.items() if dist == k]

            for neighbor in khop_nodes:
                local_comm = self.community_labels[neighbor]
                # Map local community index to universe community ID
                if local_comm in self.community_id_mapping:
                    universe_comm = self.community_id_mapping[local_comm]
                    counts[node, universe_comm] += 1
                else:
                    raise ValueError(f"Community {local_comm} not in community_id_mapping")

        return torch.tensor(counts, dtype=torch.float)

    def compute_positional_encodings(
        self,
        pyg_graph: pyg.data.Data,
        pe_types: list[str] | None = None,
        pe_dim: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Compute positional encodings for the graph."""
        if pe_types is None:
            pe_types = ["laplacian", "degree", "rwse"]
        self.pe_types = pe_types

        pe_computer = PositionalEncodingComputer(max_pe_dim=pe_dim, pe_types=pe_types)
        return pe_computer.compute_all_pe(pyg_graph.edge_index, len(pyg_graph.x))
