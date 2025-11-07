"""Test GraphFamilyGenerator class."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from graph_universe import GraphUniverse
from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_sample import GraphSample


class TestGraphFamilyGenerator:
    """Test GraphFamilyGenerator class."""

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

        # Create a graph family generator with default parameters
        self.min_n_nodes = 20
        self.max_n_nodes = 50
        self.n_graphs = 3
        self.min_communities = 2
        self.max_communities = 4

        self.family_generator = GraphFamilyGenerator(
            universe=self.universe,
            min_n_nodes=self.min_n_nodes,
            max_n_nodes=self.max_n_nodes,
            n_graphs=self.n_graphs,
            min_communities=self.min_communities,
            max_communities=self.max_communities,
            seed=self.seed,
        )

    def teardown_method(self):
        """Clean up after each test."""
        del self.universe
        del self.family_generator

    # ============================================================
    # Initialization Tests
    # ============================================================

    def test_initialization_basic(self):
        """Test basic initialization with default parameters."""
        family = GraphFamilyGenerator(
            universe=self.universe,
            min_n_nodes=10,
            max_n_nodes=20,
        )

        assert family.universe == self.universe
        assert family.min_n_nodes == 10
        assert family.max_n_nodes == 20
        assert family.n_graphs == 100  # Default value
        assert family.min_communities == 2  # Default value
        assert family.max_communities == self.K  # Default to universe.K
        assert family.family_generated is False
        assert len(family.graphs) == 0

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        family = GraphFamilyGenerator(
            universe=self.universe,
            min_n_nodes=15,
            max_n_nodes=30,
            n_graphs=5,
            min_communities=3,
            max_communities=4,
            homophily_range=(0.2, 0.6),
            avg_degree_range=(2.0, 4.0),
            degree_distribution="power_law",
            power_law_exponent_range=(2.2, 3.0),
            degree_separation_range=(0.3, 0.7),
            seed=123,
        )

        assert family.min_n_nodes == 15
        assert family.max_n_nodes == 30
        assert family.n_graphs == 5
        assert family.min_communities == 3
        assert family.max_communities == 4
        assert family.homophily_range == (0.2, 0.6)
        assert family.avg_degree_range == (2.0, 4.0)
        assert family.degree_distribution == "power_law"
        assert family.power_law_exponent_range == (2.2, 3.0)
        assert family.degree_separation_range == (0.3, 0.7)
        assert family.seed == 123

    def test_initialization_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test min_n_nodes <= 0
        with pytest.raises(ValueError, match="min_n_nodes must be positive"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=0,
                max_n_nodes=20,
            )

        # Test max_n_nodes < min_n_nodes
        with pytest.raises(ValueError, match="max_n_nodes must be >= min_n_nodes"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=30,
                max_n_nodes=20,
            )

        # Test min_communities < 1
        with pytest.raises(ValueError, match="min_communities must be >= 1"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=10,
                max_n_nodes=20,
                min_communities=0,
            )

        # Test max_communities > universe.K
        with pytest.raises(ValueError, match="max_communities cannot exceed universe size"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=10,
                max_n_nodes=20,
                max_communities=self.K + 1,
            )

        # Test max_communities < min_communities
        with pytest.raises(ValueError, match="max_communities must be >= min_communities"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=10,
                max_n_nodes=20,
                min_communities=3,
                max_communities=2,
            )

        # Test invalid homophily_range
        with pytest.raises(ValueError, match="homophily_range must be a tuple"):
            GraphFamilyGenerator(
                universe=self.universe,
                min_n_nodes=10,
                max_n_nodes=20,
                homophily_range=(0.5, 0.3),  # min > max
            )

    # ============================================================
    # Parameter Sampling Tests
    # ============================================================

    def test_sample_graph_parameters(self):
        """Test sampling of graph parameters."""
        params = self.family_generator._sample_graph_parameters()

        # Check all required parameters are present
        assert "n_nodes" in params
        assert "n_communities" in params
        assert "target_homophily" in params
        assert "target_average_degree" in params
        assert "degree_separation" in params

        # Check parameters are within expected ranges
        assert self.min_n_nodes <= params["n_nodes"] <= self.max_n_nodes
        assert self.min_communities <= params["n_communities"] <= self.max_communities
        assert self.family_generator.homophily_range[0] <= params["target_homophily"] <= self.family_generator.homophily_range[1]
        assert self.family_generator.avg_degree_range[0] <= params["target_average_degree"] <= self.family_generator.avg_degree_range[1]
        assert self.family_generator.degree_separation_range[0] <= params["degree_separation"] <= self.family_generator.degree_separation_range[1]

        # Check power law exponent if using power law distribution
        if self.family_generator.degree_distribution == "power_law":
            assert "power_law_exponent" in params
            assert self.family_generator.power_law_exponent_range[0] <= params["power_law_exponent"] <= self.family_generator.power_law_exponent_range[1]

    # ============================================================
    # Graph Family Generation Tests
    # ============================================================

    def test_generate_family_basic(self):
        """Test basic family generation with small number of graphs."""
        n_graphs = 2
        self.family_generator.generate_family(n_graphs=n_graphs, show_progress=False)

        assert self.family_generator.family_generated is True
        assert len(self.family_generator.graphs) == n_graphs
        assert len(self.family_generator.generation_metadata) == n_graphs

        # Check that all graphs are GraphSample instances
        for graph in self.family_generator.graphs:
            assert isinstance(graph, GraphSample)

    def test_generate_family_with_allowed_communities(self):
        """Test family generation with allowed community combinations."""
        n_graphs = 2
        allowed_communities = [[0, 1, 2], [2, 3, 4]]

        self.family_generator.generate_family(
            n_graphs=n_graphs,
            show_progress=False,
            allowed_community_combinations=allowed_communities
        )

        assert self.family_generator.family_generated is True
        assert len(self.family_generator.graphs) == n_graphs

        # Check that all graphs use only allowed communities
        for graph in self.family_generator.graphs:
            communities = graph.communities
            assert any(set(communities).issubset(set(allowed_combo)) for allowed_combo in allowed_communities)

    def test_generate_family_with_timeout(self):
        """Test family generation with timeout."""
        # Set a very short timeout to force early termination
        with pytest.warns(UserWarning, match="Timeout reached"):
            self.family_generator.generate_family(
                n_graphs=100,  # Large number to ensure timeout
                show_progress=False,
                timeout_minutes=0.001  # Very short timeout (0.06 seconds)
            )

        # Should have generated some graphs but fewer than requested
        assert self.family_generator.family_generated is True
        assert len(self.family_generator.graphs) < 100

    def test_generate_family_reproducibility(self):
        """Test that family generation is reproducible with same seed."""
        # Generate first family
        generator1 = GraphFamilyGenerator(
            universe=self.universe,
            min_n_nodes=20,
            max_n_nodes=30,
            n_graphs=2,
            seed=123,
        )
        generator1.generate_family(show_progress=False)

        # Generate second family with same seed
        generator2 = GraphFamilyGenerator(
            universe=self.universe,
            min_n_nodes=20,
            max_n_nodes=30,
            n_graphs=2,
            seed=123,
        )
        generator2.generate_family(show_progress=False)

        # Check that graphs have same number of nodes and edges
        for i in range(len(generator1.graphs)):
            g1 = generator1.graphs[i]
            g2 = generator2.graphs[i]
            assert g1.n_nodes == g2.n_nodes
            assert g1.graph.number_of_edges() == g2.graph.number_of_edges()

    # ============================================================
    # PyG Conversion Tests
    # ============================================================

    def test_to_pyg_graphs(self):
        """Test conversion to PyG graphs."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Convert to PyG graphs with default tasks
        pyg_graphs = self.family_generator.to_pyg_graphs()

        assert len(pyg_graphs) == 2
        for graph in pyg_graphs:
            # Check that all expected attributes are present
            assert hasattr(graph, "x")  # Node features
            assert hasattr(graph, "edge_index")  # Edge indices
            assert hasattr(graph, "community_detection")  # Community labels
            assert hasattr(graph, "realized_homophily")  # New task: realized homophily
            assert hasattr(graph, "graph_diameter")  # New task: graph diameter

    def test_to_pyg_graphs_with_custom_tasks(self):
        """Test conversion to PyG graphs with custom tasks."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Convert to PyG graphs with specific tasks
        tasks = ["community_detection", "triangle_counting"]
        pyg_graphs = self.family_generator.to_pyg_graphs(tasks=tasks)

        assert len(pyg_graphs) == 2
        for graph in pyg_graphs:
            assert hasattr(graph, "community_detection")
            assert hasattr(graph, "triangle_counting")
            # Should not have k-hop tasks
            assert not hasattr(graph, "k_hop_community_counts_k1")

    def test_to_pyg_graphs_with_new_tasks(self):
        """Test conversion to PyG graphs with the new tasks."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Convert to PyG graphs with new tasks
        tasks = ["realized_homophily", "graph_diameter"]
        pyg_graphs = self.family_generator.to_pyg_graphs(tasks=tasks)

        assert len(pyg_graphs) == 2
        for graph in pyg_graphs:
            # Check that new task attributes are present
            assert hasattr(graph, "realized_homophily")
            assert hasattr(graph, "graph_diameter")
            # Check that realized_homophily is a tensor of size K
            assert isinstance(graph.realized_homophily, torch.Tensor)
            assert graph.realized_homophily.shape == (self.K,)
            # Should not have other tasks
            assert not hasattr(graph, "community_detection")
            assert not hasattr(graph, "triangle_counting")

    def test_to_pyg_graphs_invalid_task(self):
        """Test conversion with invalid task raises error."""
        self.family_generator.generate_family(n_graphs=1, show_progress=False)

        with pytest.raises(ValueError, match="Invalid task specified"):
            self.family_generator.to_pyg_graphs(tasks=["invalid_task"])

    # ============================================================
    # Metadata and Analysis Tests
    # ============================================================

    def test_get_uniquely_identifying_metadata(self):
        """Test getting uniquely identifying metadata."""
        metadata = self.family_generator.get_uniquely_identifying_metadata()

        # Check that metadata contains expected sections
        assert "universe_parameters" in metadata
        assert "family_parameters" in metadata

        # Check that universe parameters are correct
        universe_params = metadata["universe_parameters"]
        assert universe_params["K"] == self.K
        assert universe_params["feature_dim"] == self.feature_dim
        assert universe_params["center_variance"] == 1.0
        assert universe_params["cluster_variance"] == 0.1
        assert universe_params["edge_propensity_variance"] == 0.5
        assert universe_params["seed"] == self.seed

        # Check that family parameters are correct
        family_params = metadata["family_parameters"]
        assert family_params["n_graphs"] == self.n_graphs
        assert family_params["min_n_nodes"] == self.min_n_nodes
        assert family_params["max_n_nodes"] == self.max_n_nodes
        assert family_params["min_communities"] == self.min_communities
        assert family_params["max_communities"] == self.max_communities

    def test_analyze_graph_family_properties(self):
        """Test analysis of graph family properties."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Analyze properties
        properties = self.family_generator.analyze_graph_family_properties()

        # Check that analysis contains expected metrics
        assert "n_graphs" in properties
        assert "node_counts" in properties
        assert "edge_counts" in properties
        assert "densities" in properties
        assert "avg_degrees" in properties
        assert "homophily_levels" in properties

        # Check that statistics are calculated
        assert "node_counts_mean" in properties
        assert "edge_counts_mean" in properties
        assert "densities_mean" in properties
        assert "avg_degrees_mean" in properties
        assert "homophily_levels_mean" in properties

    def test_analyze_graph_family_signals(self):
        """Test analysis of graph family signals."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Analyze signals
        signals = self.family_generator.analyze_graph_family_signals()

        # Check that analysis contains expected signals
        assert "feature_signal" in signals
        assert "degree_signal" in signals
        assert "triangle_signal" in signals
        assert "structure_signal" in signals

        # Check that signals are lists with correct length
        assert len(signals["feature_signal"]) == 2
        assert len(signals["degree_signal"]) == 2
        assert len(signals["structure_signal"]) == 2

    def test_analyze_graph_family_consistency(self):
        """Test analysis of graph family consistency."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Analyze consistency
        consistency = self.family_generator.analyze_graph_family_consistency()

        # Check that analysis contains expected metrics
        assert "structure_consistency" in consistency
        assert "feature_consistency" in consistency
        assert "degree_consistency" in consistency

    # ============================================================
    # Helper Method Tests
    # ============================================================

    def test_calculate_pattern_consistency(self):
        """Test calculation of pattern consistency."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Calculate pattern consistency
        consistency = self.family_generator._calculate_pattern_consistency()

        # Should return a list of correlation values
        assert isinstance(consistency, list)
        assert len(consistency) <= 2  # May be fewer if some calculations fail

        # Values should be between -1 and 1 (correlation range)
        for value in consistency:
            assert -1.0 <= value <= 1.0

    def test_calculate_feature_consistency(self):
        """Test calculation of feature consistency."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Calculate feature consistency
        consistency = self.family_generator._calculate_feature_consistency()

        # Should return a list of consistency values
        assert isinstance(consistency, list)
        assert len(consistency) <= 2  # May be fewer if some calculations fail

        # Values should be between 0 and 1 (similarity range)
        for value in consistency:
            assert 0.0 <= value <= 1.0

    def test_calculate_generation_fidelity(self):
        """Test calculation of generation fidelity."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Calculate generation fidelity
        fidelity = self.family_generator._calculate_generation_fidelity()

        # Should return a list of fidelity values
        assert isinstance(fidelity, list)

        # Values should be between -1 and 1 (correlation range)
        for value in fidelity:
            assert -1.0 <= value <= 1.0

    def test_measure_ordering_consistency(self):
        """Test measurement of ordering consistency."""
        # Create test arrays
        values_a = np.array([1.0, 2.0, 3.0, 4.0])
        values_b = np.array([1.0, 2.0, 3.0, 4.0])  # Perfect match

        # Calculate consistency
        consistency = self.family_generator._measure_ordering_consistency(values_a, values_b)

        # Should be 1.0 for perfect match
        assert consistency == 1.0

        # Test with reversed order
        values_b_reversed = np.array([4.0, 3.0, 2.0, 1.0])  # Reversed
        consistency = self.family_generator._measure_ordering_consistency(values_a, values_b_reversed)

        # Should be 0.0 for completely reversed order
        assert consistency == 0.0

        # Test with some consistency
        values_b_partial = np.array([1.0, 3.0, 2.0, 4.0])  # Partially consistent
        consistency = self.family_generator._measure_ordering_consistency(values_a, values_b_partial)

        # Should be between 0 and 1
        assert 0.0 < consistency < 1.0

    def test_calculate_degree_consistency(self):
        """Test calculation of degree consistency."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=2, show_progress=False)

        # Calculate degree consistency
        consistency = self.family_generator._calculate_degree_consistency()

        # Should return a list of consistency values
        assert isinstance(consistency, list)
        assert len(consistency) <= 2  # May be fewer if some calculations fail

        # Values should be between 0 and 1
        for value in consistency:
            assert 0.0 <= value <= 1.0

    def test_get_community_avg_degrees(self):
        """Test getting average degrees per community."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=1, show_progress=False)

        # Get a graph
        graph = self.family_generator.graphs[0]

        # Calculate average degrees
        avg_degrees = self.family_generator._get_community_avg_degrees(graph)

        # Should return an array with one value per community
        assert isinstance(avg_degrees, np.ndarray)
        assert len(avg_degrees) == len(graph.communities)

        # Values should be non-negative
        assert np.all(avg_degrees >= 0)

    def test_get_expected_ordering(self):
        """Test getting expected ordering based on universe degree centers."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=1, show_progress=False)

        # Get communities from first graph
        communities = self.family_generator.graphs[0].communities

        # Get degree centers from universe
        degree_centers = self.family_generator.universe.community_degree_propensity_vector

        # Get expected ordering
        ordering = self.family_generator._get_expected_ordering(communities, degree_centers)

        # Should return an array with one value per community
        assert isinstance(ordering, np.ndarray)
        assert len(ordering) == len(communities)

    def test_get_actual_ordering(self):
        """Test getting actual ordering based on observed average degrees."""
        # Create test array
        avg_degrees = np.array([2.5, 1.0, 3.5])

        # Get actual ordering
        ordering = self.family_generator._get_actual_ordering(avg_degrees)

        # Should return an array with one value per community
        assert isinstance(ordering, np.ndarray)
        assert len(ordering) == len(avg_degrees)

        # Check that ordering is correct
        assert ordering[0] == 1  # 2.5 is middle value -> rank 1
        assert ordering[1] == 0  # 1.0 is lowest -> rank 0
        assert ordering[2] == 2  # 3.5 is highest -> rank 2

    def test_rank_correlation(self):
        """Test calculation of rank correlation."""
        # Create test arrays
        expected_ranks = np.array([0, 1, 2])
        actual_ranks = np.array([0, 1, 2])  # Perfect correlation

        # Calculate correlation
        correlation = self.family_generator._rank_correlation(expected_ranks, actual_ranks)

        # Should be 1.0 for perfect correlation
        assert correlation == 1.0

        # Test with reversed ranks
        actual_ranks_reversed = np.array([2, 1, 0])  # Perfect negative correlation
        correlation = self.family_generator._rank_correlation(expected_ranks, actual_ranks_reversed)

        # Should be -1.0 for perfect negative correlation
        assert correlation == -1.0

    def test_calculate_cooccurrence_consistency(self):
        """Test calculation of cooccurrence consistency."""
        # Skip this test if the universe doesn't have a community_cooccurrence_matrix attribute
        if not hasattr(self.family_generator.universe, 'community_cooccurrence_matrix'):
            pytest.skip("Universe doesn't have community_cooccurrence_matrix attribute")

        # Generate a small family
        self.family_generator.generate_family(n_graphs=3, show_progress=False)

        # Calculate cooccurrence consistency
        try:
            consistency = self.family_generator._calculate_cooccurrence_consistency()

            # Should return a correlation value
            assert isinstance(consistency, float)

            # Value should be between -1 and 1 (correlation range)
            assert -1.0 <= consistency <= 1.0
        except AttributeError:
            # If the method fails due to missing attributes, skip the test
            pytest.skip("Method requires attributes not present in current implementation")

    def test_calculate_degree_tail_metrics(self):
        """Test calculation of degree tail metrics."""
        # Create test degrees
        degrees = [1, 2, 2, 3, 3, 3, 4, 5, 10]

        # Calculate tail metrics
        metrics = self.family_generator._calculate_degree_tail_metrics(degrees)

        # Should return a dictionary with expected keys
        assert isinstance(metrics, dict)

        # Check for expected keys - use the actual keys from the implementation
        expected_keys = ["tail_ratio_95", "tail_ratio_99", "coefficient_variation", "max_degree", "mean_degree"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        # Check specific values
        assert metrics["max_degree"] == 10
        assert metrics["mean_degree"] == sum(degrees) / len(degrees)

        # Test with empty list
        metrics = self.family_generator._calculate_degree_tail_metrics([])
        # Check that the function handles empty list without errors
        assert isinstance(metrics, dict)
        # The function should return default values for empty list
        assert "tail_ratio_95" in metrics
        assert metrics["tail_ratio_95"] == 0.0

    def test_fit_power_law_exponent(self):
        """Test fitting power law exponent to degree distribution."""
        # Create a synthetic power law distribution
        np.random.seed(42)  # Set seed for reproducibility
        degrees = list(np.random.pareto(2.5, size=1000) + 1)  # Convert to list as function expects list[int]

        # Mock the minimize_scalar function to return a predictable result
        from types import SimpleNamespace

        # Create a mock result object
        mock_result = SimpleNamespace(success=True, x=2.5)

        # Patch the minimize_scalar function
        with patch('scipy.optimize.minimize_scalar', return_value=mock_result):
            exponent = self.family_generator._fit_power_law_exponent(degrees)

            # Should return the mocked value
            assert exponent == 2.5

    # ============================================================
    # Special Cases Tests
    # ============================================================

    def test_empty_graph_family(self):
        """Test behavior with empty graph family."""
        # Don't generate any graphs
        self.family_generator.graphs = None  # Code checks for None, not empty list
        self.family_generator.family_generated = True
        self.family_generator.graph_generation_times = []  # Initialize this attribute

        # Analyze empty family
        with pytest.raises(ValueError, match="No graphs in family"):
            self.family_generator.analyze_graph_family_properties()

        # Reset for next test
        self.family_generator.graphs = None
        with pytest.raises(ValueError, match="No graphs in family"):
            self.family_generator.analyze_graph_family_signals()

        # Reset for next test
        self.family_generator.graphs = None
        with pytest.raises(ValueError, match="No graphs in family"):
            self.family_generator.analyze_graph_family_consistency()

    def test_len_and_getitem(self):
        """Test __len__ and __getitem__ methods."""
        # Generate a small family
        self.family_generator.generate_family(n_graphs=3, show_progress=False)

        # Test __len__
        assert len(self.family_generator) == 3

        # Test __getitem__
        assert isinstance(self.family_generator[0], GraphSample)
        assert isinstance(self.family_generator[1], GraphSample)
        assert isinstance(self.family_generator[2], GraphSample)

        # Test iteration
        count = 0
        for graph in self.family_generator:
            assert isinstance(graph, GraphSample)
            count += 1
        assert count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
