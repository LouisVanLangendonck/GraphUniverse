"""Test GraphSample class."""


import networkx as nx
import numpy as np
import pytest
import torch

from graph_universe import GraphUniverse
from graph_universe.graph_sample import GraphSample


class TestGraphSample:
    """Test GraphSample class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create a simple universe for testing
        self.K = 5
        self.feature_dim = 10
        self.seed = 42

        self.universe = GraphUniverse(
            K=self.K,
            edge_propensity_variance=0.5,
            feature_dim=self.feature_dim,
            center_variance=1.0,
            cluster_variance=0.1,
            seed=self.seed,
        )

        # Create a simple graph sample
        self.n_nodes = 30
        self.num_communities = 3
        self.target_homophily = 0.3
        self.target_average_degree = 2.5
        self.power_law_exponent = 2.5

        self.graph_sample = GraphSample(
            universe=self.universe,
            num_communities=self.num_communities,
            n_nodes=self.n_nodes,
            target_homophily=self.target_homophily,
            target_average_degree=self.target_average_degree,
            power_law_exponent=self.power_law_exponent,
            seed=self.seed,
        )

    def teardown_method(self):
        """Clean up after each test."""
        del self.universe
        del self.graph_sample

    # ============================================================
    # Initialization Tests
    # ============================================================

    def test_initialization_basic(self):
        """Test basic initialization with default parameters."""
        graph = self.graph_sample

        # Check basic properties
        assert graph.universe == self.universe
        assert graph.original_n_nodes == self.n_nodes
        assert graph.target_homophily == self.target_homophily
        assert graph.target_average_degree == self.target_average_degree
        assert graph.power_law_exponent == self.power_law_exponent
        assert len(graph.communities) == self.num_communities

        # Check that graph is created
        assert graph.graph is not None
        assert isinstance(graph.graph, nx.Graph)
        assert graph.n_nodes > 0
        assert graph.graph.number_of_edges() > 0

        # Check that features are created
        assert graph.features is not None
        assert graph.features.shape == (graph.n_nodes, self.feature_dim)

        # Check that community labels are created
        assert graph.community_labels is not None
        assert len(graph.community_labels) == graph.n_nodes
        assert graph.community_labels_universe_level is not None
        assert len(graph.community_labels_universe_level) == graph.n_nodes

    def test_initialization_with_user_defined_communities(self):
        """Test initialization with user-defined communities."""
        user_communities = [0, 2, 4]  # Specific communities to use

        graph = GraphSample(
            universe=self.universe,
            num_communities=len(user_communities),
            n_nodes=self.n_nodes,
            target_homophily=self.target_homophily,
            target_average_degree=self.target_average_degree,
            power_law_exponent=self.power_law_exponent,
            seed=self.seed,
            user_defined_communities=user_communities,
        )

        # Check that the specified communities are used
        assert set(graph.communities) == set(user_communities)
        assert all(label in user_communities for label in graph.community_labels_universe_level)

    def test_initialization_timing_info(self):
        """Test that timing information is recorded."""
        graph = self.graph_sample

        # Check that timing info is recorded
        assert hasattr(graph, "timing_info")
        assert isinstance(graph.timing_info, dict)

        # Check that key timing steps are recorded
        assert "propensity_matrix_scaling" in graph.timing_info
        assert "memberships" in graph.timing_info
        assert "degree_factors" in graph.timing_info
        assert "edge_generation" in graph.timing_info
        assert "feature_generation" in graph.timing_info
        assert "total" in graph.timing_info

    # ============================================================
    # Connectivity Tests
    # ============================================================

    def test_graph_is_connected(self):
        """Test that the generated graph is always connected."""
        # Check the current graph is connected
        assert nx.is_connected(self.graph_sample.graph), "Graph should be connected"

        # Generate multiple graphs with different seeds to ensure they're all connected
        for seed in range(5):
            graph = GraphSample(
                universe=self.universe,
                num_communities=self.num_communities,
                n_nodes=self.n_nodes,
                target_homophily=self.target_homophily,
                target_average_degree=self.target_average_degree,
                power_law_exponent=self.power_law_exponent,
                seed=seed,
            )
            assert nx.is_connected(graph.graph), f"Graph with seed {seed} should be connected"

    def test_connect_disconnected_components(self):
        """Test the _connect_disconnected_components method."""
        # Create a disconnected graph with two components
        G = nx.Graph()

        # First component: nodes 0-4
        for i in range(5):
            G.add_node(i)
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

        # Second component: nodes 5-9
        for i in range(5, 10):
            G.add_node(i)
        G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 5)])

        # Set up a test graph sample
        graph = self.graph_sample
        graph.adjacency = nx.adjacency_matrix(G)
        graph.community_labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

        # Call the method to connect components
        connected_adj = graph._connect_disconnected_components([set(range(5)), set(range(5, 10))])

        # Create a new graph from the connected adjacency matrix
        connected_graph = nx.from_scipy_sparse_array(connected_adj)

        # Check that the graph is now connected
        assert nx.is_connected(connected_graph), "Graph should be connected after connecting components"

        # Should have added at least one edge between components
        component1_nodes = set(range(5))
        component2_nodes = set(range(5, 10))

        cross_component_edges = [
            (u, v) for u, v in connected_graph.edges()
            if (u in component1_nodes and v in component2_nodes) or
               (u in component2_nodes and v in component1_nodes)
        ]

        assert len(cross_component_edges) >= 1, "Should have added at least one edge between components"

    def test_find_best_connection(self):
        """Test the _find_best_connection method."""
        # Set up a simple test case
        graph = self.graph_sample
        graph.community_labels = np.array([0, 0, 1, 1, 2, 2])
        graph.P_sub = np.array([
            [1.0, 0.5, 0.0],  # Community 0 connects to 0 and 1
            [0.5, 1.0, 0.5],  # Community 1 connects to 0, 1, and 2
            [0.0, 0.5, 1.0],  # Community 2 connects to 1 and 2
        ])

        # Set up components
        smallest_component = {0, 1}  # Nodes in community 0
        other_components = [{2, 3, 4, 5}]  # Nodes in communities 1 and 2

        # Calculate deviation matrix for test
        test_graph = nx.Graph()
        test_graph.add_nodes_from(range(6))
        test_graph.add_edges_from([(0, 1), (2, 3), (3, 4), (4, 5), (5, 2)])

        analysis = graph._calculate_community_deviations_with_matrix(
            test_graph, graph.community_labels, graph.P_sub
        )
        deviation_matrix = analysis["deviation_matrix"]
        actual_matrix = analysis["actual_matrix"]

        # Find best connection
        connection = graph._find_best_connection(
            smallest_component,
            other_components,
            deviation_matrix,
            actual_matrix,
        )

        # Should return a tuple of node indices
        assert isinstance(connection, tuple)
        assert len(connection) == 2

        # First node should be from smallest component
        assert connection[0] in smallest_component

        # Second node should be from other component
        assert connection[1] in other_components[0]

        # Should prefer connections between communities that should connect
        # In this case, community 0 should connect to community 1 but not 2
        assert (graph.community_labels[connection[0]] == 0 and
                graph.community_labels[connection[1]] == 1) or \
               (graph.community_labels[connection[0]] == 1 and
                graph.community_labels[connection[1]] == 0)

    # ============================================================
    # Edge Generation Tests
    # ============================================================

    def test_scale_edge_propensity_to_edge_probability(self):
        """Test scaling of edge propensity to edge probability."""
        # Create a test propensity matrix
        P_sub = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])

        # Scale the matrix
        P_scaled = self.graph_sample._scale_edge_propensity_to_edge_probability(
            P_sub,
            target_avg_degree=3.0,
            target_homophily=0.7,
            n_nodes=50,
        )

        # Check that the matrix is still symmetric
        assert np.allclose(P_scaled, P_scaled.T), "Scaled matrix should be symmetric"

        # Check that values are in valid range [0, 1]
        assert np.all(P_scaled >= 0.0), "All probabilities should be >= 0"
        assert np.all(P_scaled <= 1.0), "All probabilities should be <= 1"

        # Check that diagonal elements have higher values (due to homophily)
        diag_mean = np.mean(np.diag(P_scaled))
        off_diag_mean = np.mean(P_scaled[~np.eye(P_scaled.shape[0], dtype=bool)])
        assert diag_mean > off_diag_mean, "Diagonal elements should have higher values due to homophily"

    def test_generate_memberships(self):
        """Test generation of community memberships."""
        n_nodes = 100
        K_sub = 3

        # Generate memberships
        memberships = self.graph_sample._generate_memberships(n_nodes, K_sub)

        # Check shape and values
        assert len(memberships) == n_nodes
        assert all(0 <= m < K_sub for m in memberships)

        # Check that all communities have at least one member (with high probability)
        # This might occasionally fail due to randomness, but very unlikely with n_nodes=100
        unique_communities = np.unique(memberships)
        assert len(unique_communities) == K_sub, "All communities should have at least one member"

    def test_generate_community_degree_factors(self):
        """Test generation of community degree factors."""
        # Set up test data
        community_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        degree_separation = 0.5
        global_degree_params = {"exponent": 2.5, "x_min": 1.0}

        degree_factors = self.graph_sample._generate_community_degree_factors(
            community_labels,
            degree_separation,
            global_degree_params,
        )

        # Check shape and values
        assert len(degree_factors) == len(community_labels)
        assert np.all(degree_factors > 0), "All degree factors should be positive"

        # Check that mean is approximately 1.0 (normalization)
        assert np.isclose(np.mean(degree_factors), 1.0, atol=0.1), "Mean degree factor should be approximately 1.0"

        # Check that communities have different degree distributions based on universe propensity
        # This is harder to test directly, but we can check that there's variation between communities
        community0_factors = degree_factors[community_labels == 0]
        community1_factors = degree_factors[community_labels == 1]
        community2_factors = degree_factors[community_labels == 2]

        # With degree separation > 0, there should be differences between communities
        assert not np.isclose(np.mean(community0_factors), np.mean(community1_factors), atol=0.01) or \
               not np.isclose(np.mean(community1_factors), np.mean(community2_factors), atol=0.01) or \
               not np.isclose(np.mean(community0_factors), np.mean(community2_factors), atol=0.01)

    def test_generate_edges(self):
        """Test edge generation."""
        # Set up test data
        n_nodes = 20
        community_labels = np.random.randint(0, 3, size=n_nodes)
        P_sub = np.array([
            [0.8, 0.2, 0.1],
            [0.2, 0.7, 0.3],
            [0.1, 0.3, 0.9],
        ])
        degree_factors = np.ones(n_nodes)  # Uniform degree factors for simplicity

        # Generate edges
        adjacency = self.graph_sample._generate_edges(
            community_labels,
            P_sub,
            degree_factors,
        )

        # Check that adjacency matrix is symmetric
        assert (adjacency != adjacency.T).nnz == 0, "Adjacency matrix should be symmetric"

        # Check that diagonal is all zeros (no self-loops)
        assert np.all(adjacency.diagonal() == 0), "Diagonal should be all zeros (no self-loops)"

        # Check that there are edges
        assert adjacency.nnz > 0, "Should have generated some edges"

    # ============================================================
    # Feature and Signal Tests
    # ============================================================

    def test_calculate_feature_signal(self):
        """Test calculation of feature signal."""
        # Calculate feature signal
        signal = self.graph_sample.calculate_feature_signal()

        # Should return a value between 0 and 1
        assert 0.0 <= signal <= 1.0, "Feature signal should be between 0 and 1"

    def test_calculate_degree_signal(self):
        """Test calculation of degree signal."""
        # Calculate degree signal
        signal = self.graph_sample.calculate_degree_signal()

        # Should return a value between 0 and 1
        assert 0.0 <= signal <= 1.0, "Degree signal should be between 0 and 1"

    def test_calculate_structure_signal(self):
        """Test calculation of structure signal."""
        # Calculate structure signal
        signal = self.graph_sample.calculate_structure_signal()

        # Should return a value between 0 and 1
        assert 0.0 <= signal <= 1.0, "Structure signal should be between 0 and 1"

    def test_calculate_triangle_community_signal(self):
        """Test calculation of triangle community signal."""
        # Calculate triangle signal
        signal = self.graph_sample.calculate_triangle_community_signal()

        # Should return a value between 0 and 1
        assert 0.0 <= signal <= 1.0, "Triangle signal should be between 0 and 1"

    def test_calculate_community_signals(self):
        """Test calculation of all community signals."""
        # Calculate all signals
        signals = self.graph_sample.calculate_community_signals()

        # Check that all expected signals are present
        assert "feature_signal" in signals
        assert "degree_signal" in signals
        assert "structure_signal" in signals
        assert "triangle_signal" in signals

        # Check that all signals are between 0 and 1
        for key, value in signals.items():
            if value is not None:  # Some signals might be None if calculation failed
                assert 0.0 <= value <= 1.0, f"{key} should be between 0 and 1"

    # ============================================================
    # PyG Conversion Tests
    # ============================================================

    def test_to_pyg_graph(self):
        """Test conversion to PyG graph."""
        # Convert to PyG graph with community detection task
        pyg_graph = self.graph_sample.to_pyg_graph("community_detection")

        # Check that graph has expected attributes
        assert hasattr(pyg_graph, "x")  # Node features
        assert hasattr(pyg_graph, "edge_index")  # Edge indices
        assert hasattr(pyg_graph, "y")  # Task labels

        # Check shapes
        assert pyg_graph.x.shape == (self.graph_sample.n_nodes, self.feature_dim)
        assert pyg_graph.edge_index.shape[0] == 2  # 2 rows for source and target nodes
        assert len(pyg_graph.y) == self.graph_sample.n_nodes
        assert pyg_graph.y.dtype == torch.long  # Community labels are long

    def test_to_pyg_graph_with_triangle_counting_task(self):
        """Test conversion to PyG graph with triangle counting task."""
        # Convert to PyG graph with triangle counting task
        task = "triangle_counting"
        pyg_graph = self.graph_sample.to_pyg_graph(task)

        # Check that graph has the y attribute with triangle count
        assert hasattr(pyg_graph, "y")
        assert pyg_graph.y.dtype == torch.float
        assert pyg_graph.y.numel() == 1  # Scalar value

    def test_to_pyg_graph_different_tasks_same_structure(self):
        """Test that different tasks produce the same graph structure, only different labels."""
        # Convert the same graph with two different tasks
        task1 = "community_detection"
        task2 = "triangle_counting"

        pyg_graph1 = self.graph_sample.to_pyg_graph(task1)
        pyg_graph2 = self.graph_sample.to_pyg_graph(task2)

        # Node features should be identical
        assert torch.allclose(pyg_graph1.x, pyg_graph2.x)

        # Edge indices should be identical
        assert torch.equal(pyg_graph1.edge_index, pyg_graph2.edge_index)

        # Number of nodes and edges should be the same
        assert pyg_graph1.num_nodes == pyg_graph2.num_nodes
        assert pyg_graph1.num_edges == pyg_graph2.num_edges

        # But y should be different (different task targets)
        # Task 1 (community_detection) produces vector of length num_nodes
        # Task 2 (triangle_counting) produces a scalar
        assert pyg_graph1.y.shape != pyg_graph2.y.shape
        assert pyg_graph1.y.shape == (self.graph_sample.n_nodes,)
        assert pyg_graph2.y.numel() == 1

    def test_to_pyg_graph_invalid_task(self):
        """Test conversion with invalid task raises error."""
        with pytest.raises(ValueError, match="Invalid task specified"):
            self.graph_sample.to_pyg_graph("invalid_task")

    # ============================================================
    # Community Analysis Tests
    # ============================================================

    def test_calculate_actual_probability_matrix(self):
        """Test calculation of actual probability matrix."""
        # Calculate actual probability matrix
        actual_matrix, community_sizes, connection_counts = self.graph_sample.calculate_actual_probability_matrix()

        # Check shapes
        n_communities = len(self.graph_sample.communities)
        assert actual_matrix.shape == (n_communities, n_communities)
        assert len(community_sizes) == n_communities
        assert connection_counts.shape == (n_communities, n_communities)

        # Check that matrix is symmetric
        assert np.allclose(actual_matrix, actual_matrix.T), "Actual probability matrix should be symmetric"

        # Check that community sizes sum to total nodes
        assert np.sum(community_sizes) == self.graph_sample.n_nodes

    def test_analyze_community_connections(self):
        """Test analysis of community connections."""
        # Analyze community connections
        analysis = self.graph_sample.analyze_community_connections()

        # Check that analysis contains expected components
        assert "actual_matrix" in analysis
        assert "expected_matrix" in analysis
        assert "deviation_matrix" in analysis
        assert "mean_deviation" in analysis
        assert "community_sizes" in analysis
        assert "connection_counts" in analysis

        # Check shapes
        n_communities = len(self.graph_sample.communities)
        assert analysis["actual_matrix"].shape == (n_communities, n_communities)
        assert analysis["expected_matrix"].shape == (n_communities, n_communities)
        assert analysis["deviation_matrix"].shape == (n_communities, n_communities)

        # Check that deviation matrix is absolute difference between actual and expected
        assert np.allclose(
            analysis["deviation_matrix"],
            np.abs(analysis["actual_matrix"] - analysis["expected_matrix"])
        )

    def test_calculate_community_deviations(self):
        """Test calculation of community deviations."""
        # Create a simple test graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)])
        community_labels = np.array([0, 0, 0, 1, 1, 1])
        P_sub = np.array([[0.8, 0.2], [0.2, 0.7]])

        # Calculate deviations
        deviations = self.graph_sample._calculate_community_deviations(G, community_labels, P_sub)

        # Check that deviations are calculated
        assert "mean_deviation" in deviations
        assert "max_deviation" in deviations

        # Values should be non-negative
        assert deviations["mean_deviation"] >= 0
        assert deviations["max_deviation"] >= 0

        # Max deviation should be >= mean deviation
        assert deviations["max_deviation"] >= deviations["mean_deviation"]

    # ============================================================
    # Triangle and K-hop Tests
    # ============================================================

    def test_count_triangles_graph(self):
        """Test counting triangles in the graph."""
        # Count triangles
        triangle_count = self.graph_sample.count_triangles_graph()

        # Should be a non-negative integer
        assert isinstance(triangle_count, int)
        assert triangle_count >= 0

        # Verify with networkx's triangle counting
        nx_triangles = sum(nx.triangles(self.graph_sample.graph).values()) // 3
        assert triangle_count == nx_triangles

    # ============================================================
    # Positional Encoding Tests
    # ============================================================

    # Positional encoding tests removed - PositionalEncodingComputer class has been removed from the codebase

    # ============================================================
    # Additional Edge Case Tests
    # ============================================================

    def test_connect_disconnected_components_already_connected(self):
        """Test that connect_disconnected_components handles already connected graphs."""
        import networkx as nx

        G = nx.from_scipy_sparse_array(self.graph_sample.adjacency)
        components = list(nx.connected_components(G))

        if len(components) > 1:
            result = self.graph_sample._connect_disconnected_components(components)
            assert result.shape == self.graph_sample.adjacency.shape
        else:
            assert len(components) == 1

    def test_find_best_connection_fast(self):
        """Test the fast connection finding method."""
        # Create two small components
        component_a = {0, 1, 2}
        component_b = {3, 4, 5}

        # Create a simple deviation matrix
        deviation_matrix = np.zeros((self.K, self.K))

        # Find best connection
        node_a, node_b = self.graph_sample._find_best_connection_fast(
            component_a, component_b, deviation_matrix
        )

        # Check that nodes are from different components
        assert node_a in component_a
        assert node_b in component_b

    def test_calculate_community_deviations_with_matrix(self):
        """Test community deviation calculation with matrix."""
        import networkx as nx

        G = nx.from_scipy_sparse_array(self.graph_sample.adjacency)

        result = self.graph_sample._calculate_community_deviations_with_matrix(
            G, self.graph_sample.community_labels, self.graph_sample.P_sub
        )

        assert "deviation_matrix" in result
        assert result["deviation_matrix"].shape == self.graph_sample.P_sub.shape

    def test_graph_with_single_community(self):
        """Test graph generation with only one community."""
        single_comm_graph = GraphSample(
            universe=self.universe,
            num_communities=1,
            n_nodes=10,
            target_homophily=0.3,
            target_average_degree=2.0,
            power_law_exponent=2.5,
            seed=42,
        )

        assert single_comm_graph.n_nodes == 10
        assert len(single_comm_graph.communities) == 1
        assert len(np.unique(single_comm_graph.community_labels)) == 1

    def test_graph_with_very_low_homophily(self):
        """Test graph generation with very low homophily."""
        low_homophily_graph = GraphSample(
            universe=self.universe,
            num_communities=3,
            n_nodes=30,
            target_homophily=0.0,
            target_average_degree=2.0,
            power_law_exponent=2.5,
            seed=42,
        )

        assert low_homophily_graph.n_nodes == 30
        assert low_homophily_graph.target_homophily == 0.0

    def test_graph_with_high_homophily(self):
        """Test graph generation with very high homophily."""
        high_homophily_graph = GraphSample(
            universe=self.universe,
            num_communities=3,
            n_nodes=30,
            target_homophily=0.9,
            target_average_degree=2.0,
            power_law_exponent=2.5,
            seed=42,
        )

        assert high_homophily_graph.n_nodes == 30
        assert high_homophily_graph.target_homophily == 0.9



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
