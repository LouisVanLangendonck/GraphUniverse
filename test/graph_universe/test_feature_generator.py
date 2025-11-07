"""Test FeatureGenerator class."""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pytest

from graph_universe import FeatureGenerator


class TestFeatureGenerator:
    """Test FeatureGenerator class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.universe_K = 5
        self.feature_dim = 10
        self.seed = 42

        self.generator = FeatureGenerator(
            universe_K=self.universe_K,
            feature_dim=self.feature_dim,
            center_variance=1.0,
            cluster_variance=0.1,
            seed=self.seed,
        )

    def teardown_method(self):
        """Clean up after each test."""
        del self.generator

    # ============================================================
    # Initialization Tests
    # ============================================================

    def test_initialization_basic(self):
        """Test basic initialization with default parameters."""
        generator = FeatureGenerator(universe_K=3, feature_dim=8, seed=42)

        assert generator.universe_K == 3
        assert generator.feature_dim == 8
        assert generator.n_clusters == 3  # Should equal universe_K
        assert generator.seed == 42

    def test_initialization_stores_parameters(self):
        """Test that all parameters are correctly stored."""
        center_var = 2.0
        cluster_var = 0.05

        generator = FeatureGenerator(
            universe_K=5,
            feature_dim=10,
            center_variance=center_var,
            cluster_variance=cluster_var,
            seed=123,
        )

        assert generator.center_variance == center_var
        assert generator.cluster_variance == cluster_var

    def test_initialization_without_seed(self):
        """Test initialization without seed doesn't crash."""
        generator = FeatureGenerator(universe_K=3, feature_dim=5, seed=None)

        assert generator.cluster_centers is not None
        assert generator.cluster_centers.shape == (3, 5)

    def test_initialization_n_clusters_equals_universe_K(self):
        """Test that n_clusters equals universe_K (one-to-one mapping)."""
        generator = FeatureGenerator(universe_K=7, feature_dim=10, seed=42)

        assert generator.n_clusters == 7

    # ============================================================
    # Cluster Centers Tests
    # ============================================================

    def test_cluster_centers_shape(self):
        """Test cluster centers have correct shape."""
        assert self.generator.cluster_centers.shape == (
            self.universe_K,
            self.feature_dim,
        )

    def test_cluster_centers_not_all_same(self):
        """Test that cluster centers are different from each other."""
        centers = self.generator.cluster_centers

        # Check that not all centers are identical
        for i in range(len(centers) - 1):
            assert not np.allclose(centers[i], centers[i + 1])

    def test_cluster_centers_reproducible_with_seed(self):
        """Test that same seed produces identical cluster centers."""
        gen1 = FeatureGenerator(
            universe_K=5, feature_dim=10, center_variance=1.0, seed=42
        )
        gen2 = FeatureGenerator(
            universe_K=5, feature_dim=10, center_variance=1.0, seed=42
        )

        assert np.allclose(gen1.cluster_centers, gen2.cluster_centers)

    def test_cluster_centers_independent_of_cluster_variance(self):
        """
        CRITICAL: Cluster centers should be identical regardless of cluster_variance
        when seed and center_variance are the same.
        """
        gen_low = FeatureGenerator(
            universe_K=5,
            feature_dim=10,
            center_variance=1.0,
            cluster_variance=0.01,  # Low variance
            seed=42,
        )
        gen_high = FeatureGenerator(
            universe_K=5,
            feature_dim=10,
            center_variance=1.0,
            cluster_variance=1.0,  # High variance
            seed=42,
        )

        # Centers should be EXACTLY the same
        assert np.allclose(
            gen_low.cluster_centers, gen_high.cluster_centers
        ), "Cluster centers should not depend on cluster_variance"

    def test_center_variance_affects_separation(self):
        """
        Test that higher center_variance leads to greater separation between centers.
        Measures average pairwise distances between cluster centers.
        """

        def calculate_avg_center_distance(centers):
            """Calculate average pairwise Euclidean distance between centers."""
            n = len(centers)
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            return np.mean(distances)

        # Test with increasing center variance
        variances = [0.1, 0.5, 1.0, 2.0, 5.0]
        avg_distances = []

        for var in variances:
            gen = FeatureGenerator(
                universe_K=5, feature_dim=10, center_variance=var, seed=42
            )
            avg_dist = calculate_avg_center_distance(gen.cluster_centers)
            avg_distances.append(avg_dist)

        # Check that distances are monotonically increasing with variance
        for i in range(len(avg_distances) - 1):
            assert avg_distances[i] < avg_distances[i + 1], (
                f"Expected distances to increase with variance, but "
                f"{avg_distances[i]:.3f} >= {avg_distances[i+1]:.3f} "
                f"for variances {variances[i]} and {variances[i+1]}"
            )

    def test_cosine_distance_scaling_with_center_variance(self):
        """
        Test that cosine distances between centers increase with center_variance.
        Cosine distance = 1 - cosine_similarity.
        """

        def calculate_avg_cosine_distance(centers):
            """Calculate average pairwise cosine distance."""
            n = len(centers)
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Normalize vectors
                    v1 = centers[i] / (np.linalg.norm(centers[i]) + 1e-10)
                    v2 = centers[j] / (np.linalg.norm(centers[j]) + 1e-10)
                    # Cosine similarity
                    cos_sim = np.dot(v1, v2)
                    # Cosine distance
                    cos_dist = 1 - cos_sim
                    distances.append(cos_dist)
            return np.mean(distances)

        # Test with increasing center variance
        variances = [0.1, 0.5, 1.0, 2.0]
        cosine_distances = []

        for var in variances:
            gen = FeatureGenerator(
                universe_K=5, feature_dim=20, center_variance=var, seed=42
            )
            cos_dist = calculate_avg_cosine_distance(gen.cluster_centers)
            cosine_distances.append(cos_dist)

        # Cosine distances should generally increase with variance
        # (more separation means less similarity)
        for i in range(len(cosine_distances) - 1):
            assert cosine_distances[i] <= cosine_distances[i + 1], (
                f"Expected cosine distances to increase with variance, but "
                f"{cosine_distances[i]:.3f} > {cosine_distances[i+1]:.3f}"
            )

    # ============================================================
    # Community-Cluster Mapping Tests
    # ============================================================

    def test_community_cluster_mapping_is_identity(self):
        """Test that community-cluster mapping is one-to-one (identity matrix)."""
        expected = np.eye(self.universe_K)
        assert np.allclose(self.generator.community_cluster_probs, expected)

    def test_community_cluster_mapping_shape(self):
        """Test community-cluster probability matrix has correct shape."""
        assert self.generator.community_cluster_probs.shape == (
            self.universe_K,
            self.universe_K,
        )

    def test_community_cluster_probs_sum_to_one(self):
        """Test that each community's cluster probabilities sum to 1."""
        for comm_idx in range(self.universe_K):
            prob_sum = np.sum(self.generator.community_cluster_probs[comm_idx])
            assert np.isclose(prob_sum, 1.0)

    # ============================================================
    # Node Cluster Assignment Tests
    # ============================================================

    def test_assign_node_clusters_basic(self):
        """Test basic cluster assignment for nodes."""
        community_assignments = np.array([0, 0, 1, 1, 2])

        node_clusters = self.generator.assign_node_clusters(community_assignments)

        assert len(node_clusters) == 5
        assert all(0 <= c < self.universe_K for c in node_clusters)

    def test_assign_node_clusters_deterministic_with_seed(self):
        """Test that cluster assignment is deterministic with same seed."""
        community_assignments = np.array([0, 1, 2, 0, 1, 2])

        gen1 = FeatureGenerator(universe_K=3, feature_dim=5, seed=42)
        gen2 = FeatureGenerator(universe_K=3, feature_dim=5, seed=42)

        # Need to set seed again before assignment due to internal randomness
        np.random.seed(42)
        clusters1 = gen1.assign_node_clusters(community_assignments)

        np.random.seed(42)
        clusters2 = gen2.assign_node_clusters(community_assignments)

        assert np.array_equal(clusters1, clusters2)

    def test_assign_node_clusters_one_to_one_mapping(self):
        """
        Test that with identity mapping, community ID == cluster ID.
        """
        community_assignments = np.array([0, 1, 2, 3, 4, 0, 1, 2])

        node_clusters = self.generator.assign_node_clusters(community_assignments)

        # With identity matrix, each community should map to same cluster ID
        assert np.array_equal(node_clusters, community_assignments)

    def test_assign_node_clusters_updates_stats(self):
        """Test that cluster assignment updates statistics correctly."""
        community_assignments = np.array([0, 0, 1, 1, 2, 2, 2])

        self.generator.assign_node_clusters(community_assignments)

        # Check stats were created
        assert "cluster_counts" in self.generator.cluster_stats
        assert "community_distribution" in self.generator.cluster_stats

        # Check cluster counts
        cluster_counts = self.generator.cluster_stats["cluster_counts"]
        assert len(cluster_counts) == self.universe_K
        assert np.sum(cluster_counts) == 7  # Total nodes

    def test_assign_node_clusters_all_clusters_valid(self):
        """Test that all assigned clusters are within valid range."""
        community_assignments = np.random.randint(
            0, self.universe_K, size=100
        )

        node_clusters = self.generator.assign_node_clusters(community_assignments)

        assert all(0 <= cluster < self.universe_K for cluster in node_clusters)

    def test_assign_node_clusters_empty_array(self):
        """Test cluster assignment with empty array."""
        community_assignments = np.array([], dtype=int)

        node_clusters = self.generator.assign_node_clusters(community_assignments)

        assert len(node_clusters) == 0

    # ============================================================
    # Feature Generation Tests
    # ============================================================

    def test_generate_node_features_shape(self):
        """Test that generated features have correct shape."""
        node_clusters = np.array([0, 0, 1, 1, 2, 2])

        features = self.generator.generate_node_features(node_clusters)

        assert features.shape == (6, self.feature_dim)

    def test_generate_node_features_not_all_zeros(self):
        """Test that generated features are not all zeros."""
        node_clusters = np.array([0, 1, 2, 0, 1])

        features = self.generator.generate_node_features(node_clusters)

        assert not np.allclose(features, 0)

    def test_generate_node_features_deterministic_with_seed(self):
        """Test that feature generation is deterministic with seed."""
        node_clusters = np.array([0, 1, 2, 0, 1, 2])

        gen1 = FeatureGenerator(universe_K=3, feature_dim=5, seed=42)
        gen2 = FeatureGenerator(universe_K=3, feature_dim=5, seed=42)

        np.random.seed(100)
        features1 = gen1.generate_node_features(node_clusters)

        np.random.seed(100)
        features2 = gen2.generate_node_features(node_clusters)

        assert np.allclose(features1, features2)

    def test_generate_node_features_empty_clusters(self):
        """Test feature generation with empty cluster array."""
        node_clusters = np.array([], dtype=int)

        features = self.generator.generate_node_features(node_clusters)

        assert features.shape == (0, self.feature_dim)

    def test_cluster_variance_affects_feature_spread(self):
        """
        CRITICAL: Higher cluster_variance should make features harder to classify.
        Measure intra-cluster variance to verify.
        """

        def calculate_intra_cluster_variance(features, clusters):
            """Calculate average variance within each cluster."""
            variances = []
            for cluster_id in np.unique(clusters):
                cluster_features = features[clusters == cluster_id]
                if len(cluster_features) > 1:
                    # Variance from cluster center
                    var = np.mean(np.var(cluster_features, axis=0))
                    variances.append(var)
            return np.mean(variances) if variances else 0

        # Create node assignments
        node_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)  # 90 nodes

        # Test with different cluster variances
        variances_to_test = [0.01, 0.1, 0.5, 1.0]
        intra_cluster_vars = []

        for cluster_var in variances_to_test:
            gen = FeatureGenerator(
                universe_K=3,
                feature_dim=10,
                center_variance=1.0,  # Keep same
                cluster_variance=cluster_var,
                seed=42,
            )

            np.random.seed(100)
            features = gen.generate_node_features(node_clusters)
            intra_var = calculate_intra_cluster_variance(features, node_clusters)
            intra_cluster_vars.append(intra_var)

        # Verify intra-cluster variance increases with cluster_variance
        for i in range(len(intra_cluster_vars) - 1):
            assert intra_cluster_vars[i] < intra_cluster_vars[i + 1], (
                f"Expected increasing intra-cluster variance, but "
                f"{intra_cluster_vars[i]:.4f} >= {intra_cluster_vars[i+1]:.4f}"
            )

    def test_cluster_variance_preserves_cluster_centers(self):
        """
        CRITICAL: Changing cluster_variance should NOT change the actual
        cluster centers, only the spread around them.
        """
        # Same seed and center_variance, different cluster_variance
        gen_tight = FeatureGenerator(
            universe_K=3,
            feature_dim=10,
            center_variance=2.0,
            cluster_variance=0.01,
            seed=42,
        )
        gen_loose = FeatureGenerator(
            universe_K=3,
            feature_dim=10,
            center_variance=2.0,
            cluster_variance=1.0,
            seed=42,
        )

        # Cluster centers should be identical
        assert np.allclose(gen_tight.cluster_centers, gen_loose.cluster_centers)

        # But generated features should have different spreads
        node_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 5)

        np.random.seed(200)
        features_tight = gen_tight.generate_node_features(node_clusters)

        np.random.seed(200)
        features_loose = gen_loose.generate_node_features(node_clusters)

        # Calculate spreads (standard deviations)
        std_tight = np.std(features_tight, axis=0).mean()
        std_loose = np.std(features_loose, axis=0).mean()

        # Loose should have higher spread
        assert std_loose > std_tight

    def test_features_centered_around_cluster_centers(self):
        """Test that generated features are approximately centered around cluster centers."""
        node_clusters = np.array([0] * 100 + [1] * 100 + [2] * 100)

        np.random.seed(42)
        features = self.generator.generate_node_features(node_clusters)

        # Check that mean of features for cluster 0 is close to center 0
        for cluster_id in range(3):
            cluster_mask = node_clusters == cluster_id
            cluster_features = features[cluster_mask]
            mean_features = np.mean(cluster_features, axis=0)
            expected_center = self.generator.cluster_centers[cluster_id]

            # Should be close (within tolerance due to randomness and sample size)
            assert np.allclose(mean_features, expected_center, atol=0.2)

    def test_features_follow_multivariate_normal(self):
        """Test that features approximately follow multivariate normal distribution."""
        node_clusters = np.array([0] * 1000)  # Many samples for one cluster

        np.random.seed(42)
        features = self.generator.generate_node_features(node_clusters)

        # Test normality using simple statistics
        # Mean should be close to cluster center
        mean_features = np.mean(features, axis=0)
        expected_center = self.generator.cluster_centers[0]
        assert np.allclose(mean_features, expected_center, atol=0.1)

        # Covariance should be close to identity * cluster_variance
        cov_features = np.cov(features, rowvar=False)
        expected_cov = np.eye(self.feature_dim) * self.generator.cluster_variance
        assert np.allclose(cov_features, expected_cov, atol=0.05)

    # ============================================================
    # Edge Cases
    # ============================================================

    def test_single_community(self):
        """Test with single community."""
        gen = FeatureGenerator(universe_K=1, feature_dim=5, seed=42)

        assert gen.n_clusters == 1
        assert gen.cluster_centers.shape == (1, 5)

        community_assignments = np.array([0, 0, 0, 0])
        node_clusters = gen.assign_node_clusters(community_assignments)
        assert all(c == 0 for c in node_clusters)

        features = gen.generate_node_features(node_clusters)
        assert features.shape == (4, 5)

    def test_single_feature_dimension(self):
        """Test with single feature dimension."""
        gen = FeatureGenerator(universe_K=3, feature_dim=1, seed=42)

        assert gen.cluster_centers.shape == (3, 1)

        node_clusters = np.array([0, 1, 2])
        features = gen.generate_node_features(node_clusters)
        assert features.shape == (3, 1)

    def test_large_feature_dimension(self):
        """Test with large feature dimension."""
        gen = FeatureGenerator(universe_K=3, feature_dim=100, seed=42)

        assert gen.cluster_centers.shape == (3, 100)

        node_clusters = np.array([0, 1, 2])
        features = gen.generate_node_features(node_clusters)
        assert features.shape == (3, 100)

    def test_extreme_variances(self):
        """Test with extreme variance values."""
        # Very small center variance
        gen_small = FeatureGenerator(
            universe_K=3, feature_dim=5, center_variance=0.001, seed=42
        )
        assert gen_small.cluster_centers is not None

        # Very large center variance
        gen_large = FeatureGenerator(
            universe_K=3, feature_dim=5, center_variance=100.0, seed=42
        )
        assert gen_large.cluster_centers is not None

        # Very small cluster variance
        gen_tight = FeatureGenerator(
            universe_K=3, feature_dim=5, cluster_variance=0.001, seed=42
        )
        node_clusters = np.array([0, 0, 0])
        features = gen_tight.generate_node_features(node_clusters)
        assert features.shape == (3, 5)

    # ============================================================
    # Integration Tests
    # ============================================================

    def test_full_pipeline(self):
        """Test complete feature generation pipeline."""
        # Create generator
        gen = FeatureGenerator(
            universe_K=4, feature_dim=8, center_variance=1.0, cluster_variance=0.1, seed=42
        )

        # Assign communities to nodes
        community_assignments = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        # Get node clusters
        node_clusters = gen.assign_node_clusters(community_assignments)

        # Generate features
        features = gen.generate_node_features(node_clusters)

        # Verify end-to-end
        assert features.shape == (8, 8)
        assert not np.allclose(features, 0)
        assert gen.cluster_stats["cluster_counts"].sum() == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
