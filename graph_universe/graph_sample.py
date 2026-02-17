import time
import warnings
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.data as pyg
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from graph_universe.graph_universe import GraphUniverse


class GraphSample:
    """
    Represents a single graph instance sampled from the GraphUniverse.
    """

    def __init__(
        self,
        # Give GraphUniverse object to sample from
        universe: GraphUniverse,
        num_communities: int,
        n_nodes: int,
        target_homophily: float,
        target_average_degree: float,
        power_law_exponent: float | None,
        degree_separation: float = 0.5,
        global_degree_params: dict | None = None,
        seed: int | None = None,
        user_defined_communities: list[int] | None = None,
    ):
        """
        Initialize and generate a graph sample from the GraphUniverse.
        """

        self.universe = universe
        self.degree_separation = degree_separation
        self.global_degree_params = global_degree_params or {}

        self.timing_info = {}
        total_start = time.time()

        if user_defined_communities is not None:
            self.communities = user_defined_communities
        else:
            self.communities = universe.sample_connected_community_subset(
                num_communities, seed=seed
            )

        self.original_n_nodes = n_nodes
        self.target_homophily = target_homophily
        self.target_average_degree = target_average_degree
        self.power_law_exponent = power_law_exponent

        self.community_id_mapping = dict(enumerate(self.communities))
        self.reverse_community_id_mapping = {
            comm_id: i for i, comm_id in self.community_id_mapping.items()
        }

        self.generation_params = {
            "degree_separation": degree_separation,
            "power_law_exponent": power_law_exponent,
        }

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

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
        self.P_sub = self._scale_edge_propensity_to_edge_probability(
            self.P_sub,
            self.target_average_degree,
            self.target_homophily,
            self.original_n_nodes,
        )
        self.timing_info["propensity_matrix_scaling"] = time.time() - start

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

        start = time.time()
        global_degree_params = {"exponent": power_law_exponent, "x_min": 1.0}

        if self.global_degree_params:
            global_degree_params.update(self.global_degree_params)

        self.degree_factors = self._generate_community_degree_factors(
            self.community_labels,
            degree_separation,
            global_degree_params,
        )
        self.timing_info["degree_factors"] = time.time() - start

        # Time: Generate edges
        start = time.time()
        self.adjacency = self._generate_edges(
            self.community_labels,
            self.P_sub,
            self.degree_factors,
        )
        self.timing_info["edge_generation"] = time.time() - start

        # Create initial NetworkX graph
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)

        # Time: Component filtering
        start = time.time()
        # Find connected components
        components = list(nx.connected_components(temp_graph))
        components.sort(key=len, reverse=True)

        # Enforce connectivity
        self.adjacency = self._connect_disconnected_components(components)

        # Update the number of nodes
        self.graph = nx.from_scipy_sparse_array(self.adjacency)
        self.n_nodes = self.graph.number_of_nodes()

        # Store component connection time
        self.timing_info["component_connection"] = time.time() - start

        # Time: Feature generation
        start = time.time()
        if universe.feature_dim > 0:
            # how many graphs were generated before this one
            if seed is not None:
                # Use graph seed + 1000000 to derive feature seed (avoid collision with graph structure seed)
                feature_seed = seed + 1000000
                universe.feature_generator.reset_rng(feature_seed)

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

    def _connect_disconnected_components(self, components: list[set[int]]) -> sp.spmatrix:
        """
        Connect disconnected components of the graph iteratively, starting with the smallest component.
        Uses deviation analysis to find optimal connections that bring the actual probability matrix
        closer to the expected P_sub matrix.

        For large graphs (>1000 nodes), uses a fast approximation method.
        """
        if len(components) <= 1:
            # Already connected or empty graph; return current adjacency unchanged
            return self.adjacency

        # For large graphs, use fast approximation
        if self.original_n_nodes > 1000:
            return self._connect_disconnected_components_fast(components)

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

            # Make the connection
            node_from, node_to = best_connection
            temp_graph.add_edge(node_from, node_to)

            # Recalculate components
            new_components = list(nx.connected_components(temp_graph))
            components = sorted(new_components, key=len)

        # return the adjacency matrix of the connected graph
        return nx.adjacency_matrix(temp_graph)

    def _connect_disconnected_components_fast(self, components: list[set[int]]) -> sp.spmatrix:
        """
        Fast approximation for connecting disconnected components in large graphs (>1000 nodes).

        This is slightly less optimal than the full method but much faster.
        """
        if len(components) <= 1:
            return self.adjacency

        # Create a copy of the graph to work with
        temp_graph = nx.from_scipy_sparse_array(self.adjacency)

        # Calculate deviation matrix once at the start
        initial_analysis = self._calculate_community_deviations_with_matrix(
            temp_graph, self.community_labels, self.P_sub
        )
        deviation_matrix = initial_analysis["deviation_matrix"]

        # Sort components by size (largest first for fast method)
        components = sorted(components, key=len, reverse=True)
        largest_component = components[0]

        # Connect all other components to the largest component
        for component in components[1:]:
            # Find best connection between this component and largest component
            best_connection = self._find_best_connection_fast(
                component,
                largest_component,
                deviation_matrix,
            )

            # Make the connection
            node_from, node_to = best_connection
            temp_graph.add_edge(node_from, node_to)

            # Update largest component (merge the newly connected component)
            largest_component = largest_component.union(component)

        return nx.adjacency_matrix(temp_graph)

    def _find_best_connection_fast(
        self,
        component_a: set[int],
        component_b: set[int],
        deviation_matrix: np.ndarray,
    ) -> tuple[int, int]:
        """
        Fast approximation for finding the best connection between two components.

        Args:
            component_a: First component (smaller)
            component_b: Second component (larger)
            deviation_matrix: Current deviation matrix

        Returns:
            Tuple of (node_from_a, node_to_b)
        """
        # Get community distributions in each component
        communities_a = {}  # {community_id: [list of nodes]}
        communities_b = {}

        for node in component_a:
            comm = self.community_labels[node]
            if comm not in communities_a:
                communities_a[comm] = []
            communities_a[comm].append(node)

        for node in component_b:
            comm = self.community_labels[node]
            if comm not in communities_b:
                communities_b[comm] = []
            communities_b[comm].append(node)

        # Calculate connection scores for each community pair
        connection_scores = []  # (score, comm_a, comm_b)

        for comm_a in communities_a:
            for comm_b in communities_b:
                # Score based on P_sub and deviation
                p_sub_value = self.P_sub[comm_a, comm_b]

                if p_sub_value > 0:
                    # Allowed connection - prioritize if deviation is negative (under-connected)
                    deviation = deviation_matrix[comm_a, comm_b]
                    if deviation < 0:
                        # Under-connected: high priority
                        score = p_sub_value * (1 - deviation)
                    else:
                        # Over-connected: lower priority but still allowed
                        score = p_sub_value * 0.5
                else:
                    # Disallowed connection: very low score
                    score = 0.01

                connection_scores.append((score, comm_a, comm_b))

        if not connection_scores:
            # Fallback: just pick any two nodes
            node_a = next(iter(component_a))
            node_b = next(iter(component_b))
            return (node_a, node_b)

        # Sample community pair based on scores
        scores = np.array([s[0] for s in connection_scores])
        scores = scores / np.sum(scores)  # Normalize to probabilities

        chosen_idx = np.random.choice(len(connection_scores), p=scores)
        _, chosen_comm_a, chosen_comm_b = connection_scores[chosen_idx]

        # Within chosen communities, sample nodes based on degree factors
        # Prefer higher-degree nodes for better connectivity
        nodes_a = communities_a[chosen_comm_a]
        nodes_b = communities_b[chosen_comm_b]

        # Get degree factors for these nodes
        degree_factors_a = np.array([self.degree_factors[n] for n in nodes_a])
        degree_factors_b = np.array([self.degree_factors[n] for n in nodes_b])

        # Normalize to probabilities
        probs_a = degree_factors_a / np.sum(degree_factors_a)
        probs_b = degree_factors_b / np.sum(degree_factors_b)

        # Sample nodes
        node_a = np.random.choice(nodes_a, p=probs_a)
        node_b = np.random.choice(nodes_b, p=probs_b)

        return (node_a, node_b)

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
    ):
        """
        Connect disconnected components of the graph iteratively, starting with the smallest component.
        Uses deviation analysis to find optimal connections that bring the actual probability matrix
        closer to the expected P_sub matrix.
        """
        # Get communities in the smallest component
        smallest_communities = {self.community_labels[node] for node in smallest_component}

        # Track allowed and disallowed connections separately
        allowed_connections = []  # (score, node_from, node_to)
        disallowed_connections = []  # (score, node_from, node_to)

        # Check all possible connections
        for other_component in other_components:
            other_communities = {self.community_labels[node] for node in other_component}

            for comm_small in smallest_communities:
                for comm_other in other_communities:
                    node_from = self._find_node_in_community(smallest_component, comm_small)
                    node_to = self._find_node_in_community(other_component, comm_other)

                    if node_from is None or node_to is None:
                        continue

                    # Score the connection
                    if self.P_sub[comm_small, comm_other] > 0:
                        # Allowed connection - score by deviation
                        current_deviation = deviation_matrix[comm_small, comm_other]
                        if current_deviation < 0:
                            score = -current_deviation
                        else:
                            score = -abs(current_deviation)
                        allowed_connections.append((score, node_from, node_to))
                    else:
                        # Disallowed connection - score by minimal damage
                        score = -actual_matrix[comm_small, comm_other]
                        disallowed_connections.append((score, node_from, node_to))

        # Prefer allowed connections, fall back to disallowed
        if allowed_connections:
            best = max(allowed_connections, key=lambda x: x[0])
            return (best[1], best[2])
        elif disallowed_connections:
            best = max(disallowed_connections, key=lambda x: x[0])
            return (best[1], best[2])
        else:
            # Ultimate fallback - just connect any two nodes
            raise ValueError("No valid connections found")

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

    def _generate_community_degree_factors(
        self,
        community_labels: np.ndarray,
        degree_separation: float,
        global_degree_params: dict,
    ) -> np.ndarray:
        n_nodes = len(community_labels)

        if n_nodes > 1000:
            return self._generate_community_degree_factors_fast(
                community_labels,
                degree_separation,
                global_degree_params,
            )

        exponent = global_degree_params.get("exponent", 2.5)
        raw_degrees = np.random.pareto(exponent, size=n_nodes) + 1

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
        # Use numpy array for efficiency (avoids O(n) list.remove operations)
        available_positions = np.arange(n_nodes)
        degree_factors = np.zeros(n_nodes)

        # 8. For each node, sample from its community's truncated normal distribution
        for node_idx in range(n_nodes):
            community_local_idx = community_labels[node_idx]

            # Get the mean for this community's distribution (use original community index)
            mean_pos = community_means[community_local_idx]
            std_pos = community_std

            # Sample from truncated normal distribution over available positions
            n_available = len(available_positions)
            if n_available == 1:
                # Only one position left, take it
                chosen_idx = 0
                chosen_pos = available_positions[0]
            else:
                positions_array = available_positions  # Already numpy array
                probabilities = np.exp(-0.5 * ((positions_array - mean_pos) / std_pos) ** 2)
                probabilities = probabilities / np.sum(probabilities)  # Normalize

                # Sample from available positions based on probabilities
                chosen_idx = np.random.choice(n_available, p=probabilities)
                chosen_pos = available_positions[chosen_idx]

            degree_factors[node_idx] = sorted_degrees[chosen_pos]
            available_positions = np.delete(available_positions, chosen_idx)

        degree_factors_mean = np.mean(degree_factors)
        degree_factors = degree_factors / degree_factors_mean

        return degree_factors

    def _generate_community_degree_factors_fast(
        self,
        community_labels: np.ndarray,
        degree_separation: float,
        global_degree_params: dict,
    ) -> np.ndarray:
        """
        Fast approximation for community degree factors for large graphs.
        """
        n_nodes = len(community_labels)
        k = len(np.unique(community_labels))

        community_id_mapping = dict(enumerate(sorted(set(community_labels))))
        universe_degree_centers = np.array(
            [
                self.universe.community_degree_propensity_vector[
                    community_id_mapping[local_comm_id]
                ]
                for local_comm_id in range(k)
            ]
        )

        if k > 1:
            min_center = np.min(universe_degree_centers)
            max_center = np.max(universe_degree_centers)
            if max_center > min_center:
                normalized_centers = (universe_degree_centers - min_center) / (max_center - min_center)
            else:
                normalized_centers = np.ones(k) * 0.5
        else:
            normalized_centers = np.array([0.5])

        exponent = global_degree_params.get("exponent", 2.5)
        base_degrees = np.random.pareto(exponent, size=n_nodes) + 1

        degree_factors = np.zeros(n_nodes)
        for node_idx in range(n_nodes):
            comm_idx = community_labels[node_idx]
            scale = 1.0 + degree_separation * (normalized_centers[comm_idx] - 0.5) * 2
            degree_factors[node_idx] = base_degrees[node_idx] * scale

        degree_factors_mean = np.mean(degree_factors)
        if degree_factors_mean > 0:
            degree_factors = degree_factors / degree_factors_mean

        return degree_factors

    def _generate_edges(
        self,
        community_labels: np.ndarray,
        P_sub: np.ndarray,
        degree_factors: np.ndarray,
    ) -> sp.spmatrix:
        """
        Generate edges with minimum density guarantee.
        Uses community-aware generation for efficiency on large graphs.

        Args:
            community_labels: Node community assignments (indices)
            P_sub: Community-community probability matrix
            degree_factors: Node degree factors

        Returns:
            Sparse adjacency matrix
        """
        n_nodes = len(community_labels)

        # For smaller graphs, use the original fast vectorized method
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

    def _scale_edge_propensity_to_edge_probability(
        self,
        P_sub: np.ndarray,
        target_avg_degree: float | None = None,
        target_homophily: float | None = None,
        n_nodes: int | None = None,
    ) -> np.ndarray:
        n = P_sub.shape[0]

        # Make copy to avoid modifying the original matrix
        P_scaled = P_sub.copy()

        # Convert target average degree to equivalent density
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

        if diagonal_sum == 0 and target_diagonal_sum > 0:
            P_scaled[diagonal_mask] = target_diagonal_sum / n

        if off_diagonal_sum == 0 and target_off_diagonal_sum > 0 and n > 1:
            P_scaled[off_diagonal_mask] = target_off_diagonal_sum / (n * n - n)

        # Now ensure all probabilities are in [0, 1] for actual graph generation
        P_scaled = np.clip(P_scaled, 0, 1)

        return P_scaled

    def to_pyg_graph(
        self,
        task: str,
    ) -> pyg.data.Data:
        """
        Convert the graph to a PyG graph including the specified task as a property.
        Available tasks are:
        - "community_detection"
        - "triangle_counting"

        Args:
            task: String of task to include as property on the graph.

        Returns:
            PyG Data object with task results stored as properties
        """
        if task not in ["community_detection", "triangle_counting"]:
            raise ValueError(f"Invalid task specified: {task}")
        self.task = task

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
        if task == "community_detection":
            if (
                hasattr(self, "community_labels_universe_level")
                and self.community_labels_universe_level is not None
            ):
                data.y = torch.tensor(
                    self.community_labels_universe_level, dtype=torch.long
                )
            else:
                raise ValueError("Community labels are not available for the graph")

        elif task == "triangle_counting":
            triangle_count = self.count_triangles_graph()
            data.y = torch.tensor(triangle_count, dtype=torch.float)

        else:
            raise ValueError(f"Unknown task: {task}")

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
        (
            actual_matrix,
            community_sizes,
            connection_counts,
        ) = self.calculate_actual_probability_matrix()

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
            "constraints": {},  # No constraints needed
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
