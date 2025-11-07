"""Test GraphUniverse class."""

import numpy as np
import pytest

from graph_universe import GraphUniverse


class TestGraphUniverse:
    """Test GraphUniverse class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Basic universe with default parameters
        self.K = 5
        self.feature_dim = 10
        self.seed = 42

        # Create a simple universe for testing
        self.universe = GraphUniverse(
            K=self.K,
            edge_propensity_variance=0.5,
            feature_dim=self.feature_dim,
            center_variance=1.0,
            cluster_variance=0.1,
            seed=self.seed,
        )

    def teardown_method(self):
        """Clean up after each test."""
        del self.universe

    # ============================================================
    # Initialization Tests
    # ============================================================

    def test_initialization_basic(self):
        """Test basic initialization with default parameters."""
        universe = GraphUniverse(K=3, seed=42)

        assert universe.K == 3
        assert universe.feature_dim == 0
        assert universe.seed == 42
        assert universe.P is not None
        assert universe.P.shape == (3, 3)
        assert universe.feature_generator is None  # No features when feature_dim=0

    def test_initialization_with_features(self):
        """Test initialization with feature generation enabled."""
        universe = GraphUniverse(K=4, feature_dim=8, seed=42)

        assert universe.feature_dim == 8
        assert universe.feature_generator is not None

    def test_initialization_without_features(self):
        """Test initialization without feature generation enabled."""
        universe = GraphUniverse(K=4, seed=42, feature_dim=0)
        assert universe.feature_generator is None

    def test_initialization_with_custom_P(self):
        """Test initialization with pre-defined propensity matrix."""
        custom_P = np.array([[1.0, 0.5], [0.5, 1.0]])
        universe = GraphUniverse(K=2, P=custom_P, seed=42)

        assert np.array_equal(universe.P, custom_P)

    def test_initialization_stores_parameters(self):
        """Test that all parameters are correctly stored."""
        edge_var = 0.3
        center_var = 2.0
        cluster_var = 0.05

        universe = GraphUniverse(
            K=5,
            edge_propensity_variance=edge_var,
            feature_dim=10,
            center_variance=center_var,
            cluster_variance=cluster_var,
            seed=123,
        )

        assert universe.edge_propensity_variance == edge_var
        assert universe.center_variance == center_var
        assert universe.cluster_variance == cluster_var
        assert universe.seed == 123

    # ============================================================
    # Edge Propensity Matrix Tests
    # ============================================================

    def test_propensity_matrix_shape(self):
        """Test that propensity matrix has correct shape."""
        assert self.universe.P.shape == (self.K, self.K)

    def test_propensity_matrix_symmetric(self):
        """Test that propensity matrix is symmetric."""
        P = self.universe.P
        assert np.allclose(P, P.T), "Propensity matrix should be symmetric"

    def test_propensity_matrix_values_in_range(self):
        """Test that propensity values are within valid range [0, 2]."""
        P = self.universe.P
        assert np.all(P >= 0.0), "All propensities should be >= 0"
        assert np.all(P <= 2.0), "All propensities should be <= 2"

    def test_propensity_matrix_no_variance(self):
        """Test propensity matrix with zero variance is all ones."""
        universe = GraphUniverse(K=4, edge_propensity_variance=0.0, seed=42)
        expected = np.ones((4, 4))

        assert np.allclose(universe.P, expected)

    def test_propensity_matrix_with_variance(self):
        """Test propensity matrix with variance has variation."""
        universe = GraphUniverse(K=4, edge_propensity_variance=0.5, seed=42)

        # Should not be exactly all ones
        assert not np.allclose(universe.P, np.ones((4, 4)))

        # But should still be symmetric
        assert np.allclose(universe.P, universe.P.T)

    def test_generate_edge_propensity_matrix_directly(self):
        """Test the _generate_edge_propensity_matrix method directly."""
        K = 3
        P = self.universe._generate_edge_propensity_matrix(K, edge_propensity_variance=0.0)

        assert P.shape == (K, K)
        assert np.allclose(P, np.ones((K, K)))

    # ============================================================
    # Community Degree Propensity Vector Tests
    # ============================================================

    def test_community_degree_propensity_vector_exists(self):
        """Test that community degree propensity vector is initialized."""
        assert hasattr(self.universe, "community_degree_propensity_vector")
        assert self.universe.community_degree_propensity_vector is not None

    def test_community_degree_propensity_vector_shape(self):
        """Test community degree propensity vector has correct length."""
        assert len(self.universe.community_degree_propensity_vector) == self.K

    def test_community_degree_propensity_vector_range(self):
        """Test community degree propensity values are in [-1, 1]."""
        vec = self.universe.community_degree_propensity_vector
        assert np.all(vec >= -1.0) and np.all(vec <= 1.0)

    # ============================================================
    # Reproducibility Tests (Seed)
    # ============================================================

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces identical results."""
        universe1 = GraphUniverse(K=5, edge_propensity_variance=0.5, seed=123)
        universe2 = GraphUniverse(K=5, edge_propensity_variance=0.5, seed=123)

        assert np.allclose(universe1.P, universe2.P)
        assert np.allclose(
            universe1.community_degree_propensity_vector,
            universe2.community_degree_propensity_vector,
        )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        universe1 = GraphUniverse(K=5, edge_propensity_variance=0.5, seed=111)
        universe2 = GraphUniverse(K=5, edge_propensity_variance=0.5, seed=222)

        # Should be different (with very high probability)
        assert not np.allclose(universe1.P, universe2.P)

    def test_none_seed_works(self):
        """Test that None seed doesn't crash."""
        universe = GraphUniverse(K=3, seed=None)
        assert universe.P is not None

    # ============================================================
    # Sample Connected Community Subset Tests
    # ============================================================

    def test_sample_connected_community_subset_basic(self):
        """Test basic community subset sampling."""
        size = 3
        communities = self.universe.sample_connected_community_subset(size=size, seed=42)

        assert len(communities) == size
        assert all(0 <= c < self.K for c in communities)
        assert len(set(communities)) == len(communities)  # All unique

    def test_sample_connected_community_subset_reproducible(self):
        """Test that sampling with same seed is reproducible."""
        communities1 = self.universe.sample_connected_community_subset(size=3, seed=100)
        communities2 = self.universe.sample_connected_community_subset(size=3, seed=100)

        assert communities1 == communities2

    def test_sample_connected_community_subset_size_larger_than_K(self):
        """Test sampling when requested size exceeds K."""
        # Request more communities than available
        communities = self.universe.sample_connected_community_subset(size=self.K + 5, seed=42)

        # Should cap at K
        assert len(communities) <= self.K

    def test_sample_connected_community_subset_size_one(self):
        """Test sampling a single community."""
        communities = self.universe.sample_connected_community_subset(size=1, seed=42)

        assert len(communities) == 1
        assert 0 <= communities[0] < self.K

    def test_sample_connected_community_subset_connectivity(self):
        """Test that sampled communities have non-zero edge propensity."""
        communities = self.universe.sample_connected_community_subset(size=3, seed=42)

        # Each community should have non-zero connection to at least one other
        for i, comm in enumerate(communities[1:], 1):  # Skip first one
            # Check connection to previous communities
            connections = [self.universe.P[comm, prev] for prev in communities[:i]]
            assert any(conn > 0 for conn in connections), (
                f"Community {comm} has no connection to previous communities"
            )

    def test_sample_connected_community_subset_all_unique(self):
        """Test that all sampled communities are unique."""
        communities = self.universe.sample_connected_community_subset(size=self.K, seed=42)

        assert len(set(communities)) == len(communities)

    def test_sample_connected_community_use_cooccurrence_parameter(self):
        """Test that use_cooccurrence parameter doesn't crash."""
        # Test both True and False
        communities_true = self.universe.sample_connected_community_subset(
            size=3, seed=42, use_cooccurrence=True
        )
        communities_false = self.universe.sample_connected_community_subset(
            size=3, seed=42, use_cooccurrence=False
        )

        # Both should work (currently they do the same thing based on your code)
        assert len(communities_true) == 3
        assert len(communities_false) == 3

    # ============================================================
    # Edge Cases
    # ============================================================

    def test_initialization_K_equals_one(self):
        """Test with single community."""
        universe = GraphUniverse(K=1, seed=42)

        assert universe.P.shape == (1, 1)
        assert len(universe.community_degree_propensity_vector) == 1

    def test_sample_connected_community_zero_size(self):
        """Test sampling with size zero."""
        # Note: Your current implementation doesn't handle this,
        # but it's good to document expected behavior
        communities = self.universe.sample_connected_community_subset(size=0, seed=42)

        # Could be empty or might have issues - test what actually happens
        # This test helps you decide if you need to handle this edge case
        assert isinstance(communities, list)

    def test_large_K_performance(self):
        """Test that large K values don't crash (performance test)."""
        # This is more of a sanity check than strict unit test
        universe = GraphUniverse(K=100, seed=42)

        assert universe.P.shape == (100, 100)

        # Should still be able to sample
        communities = universe.sample_connected_community_subset(size=10, seed=42)
        assert len(communities) == 10


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v"])
