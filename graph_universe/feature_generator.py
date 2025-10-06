import numpy as np


class FeatureGenerator:
    """
    Feature generator using multivariate Gaussian clusters.

    Features are generated with controllable inter-cluster and intra-cluster variance.
    Each community is assigned to exactly one cluster.

    Args:
        universe_K: Total number of communities in the universe
        feature_dim: Dimension of node features
        cluster_count_factor: Controls number of clusters relative to communities
                                (0.1 = few clusters, 1.0 = same as communities, 4.0 = many clusters)
        center_variance: Controls separation between cluster centers
        cluster_variance: Controls spread within each cluster
        assignment_skewness: Kept for backward compatibility (not used)
        community_exclusivity: Kept for backward compatibility (not used)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        universe_K: int,
        feature_dim: int,
        cluster_count_factor: float = 1.0,  # Number of clusters relative to communities (0.1 to 4.0)
        center_variance: float = 1.0,  # Separation between cluster centers
        cluster_variance: float = 0.1,  # Spread within each cluster
        assignment_skewness: float = 0.0,  # Kept for backward compatibility (not used)
        community_exclusivity: float = 1.0,  # Kept for backward compatibility (not used)
        seed: int | None = None,
    ):
        self.universe_K = universe_K
        self.feature_dim = feature_dim
        self.center_variance = center_variance
        self.cluster_variance = cluster_variance

        # Kept for backward compatibility
        self.assignment_skewness = 0.0
        self.community_exclusivity = 1.0

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Determine number of clusters
        self.n_clusters = max(1, int(universe_K * cluster_count_factor))

        # Generate cluster centers
        self.cluster_centers = self._generate_cluster_centers()

        # Create simple one-to-one community-cluster mapping
        self.community_cluster_probs = self._create_community_cluster_mapping()

        # Track assignment statistics
        self.cluster_stats = {}

    def _generate_cluster_centers(self) -> np.ndarray:
        """
        Generate cluster centers with controlled separation.

        Returns:
            Matrix of cluster centers (n_clusters x feature_dim)
        """
        # Generate centers from multivariate normal distribution with specified variance
        centers = np.random.normal(
            0, self.center_variance, size=(self.n_clusters, self.feature_dim)
        )
        return centers

    def _create_community_cluster_mapping(self) -> np.ndarray:
        """
        Create simple one-to-one mapping between communities and clusters.
        Each community is assigned to exactly one cluster.

        Returns:
            Matrix of community-cluster probabilities (universe_K x n_clusters)
        """
        probs = np.zeros((self.universe_K, self.n_clusters))

        # Simple assignment: each community gets one cluster
        # If more communities than clusters, multiple communities share clusters
        # If more clusters than communities, some clusters may not be used
        for comm_idx in range(self.universe_K):
            cluster_idx = comm_idx % self.n_clusters
            probs[comm_idx, cluster_idx] = 1.0

        return probs

    def assign_node_clusters(self, community_assignments: np.ndarray) -> np.ndarray:
        """
        Assign nodes to feature clusters based on their community assignments.

        Args:
            community_assignments: Array of community IDs for each node

        Returns:
            Array of cluster IDs for each node
        """
        n_nodes = len(community_assignments)
        node_clusters = np.zeros(n_nodes, dtype=int)

        # Reset cluster stats
        self.cluster_stats = {
            "cluster_counts": np.zeros(self.n_clusters, dtype=int),
            "community_distribution": np.zeros((self.n_clusters, self.universe_K), dtype=int),
        }

        # For each node
        for i in range(n_nodes):
            comm_id = community_assignments[i]

            # Sample cluster from community's distribution
            node_clusters[i] = np.random.choice(
                self.n_clusters, p=self.community_cluster_probs[comm_id]
            )

            # Track stats
            cluster = node_clusters[i]
            self.cluster_stats["cluster_counts"][cluster] += 1
            self.cluster_stats["community_distribution"][cluster, comm_id] += 1

        # Log some basic statistics
        # print(f"Cluster assignment stats:")
        # print(f"  Most frequent cluster: {np.argmax(self.cluster_stats['cluster_counts'])} "
        #       f"with {np.max(self.cluster_stats['cluster_counts'])} nodes")
        # print(f"  Least frequent cluster: {np.argmin(self.cluster_stats['cluster_counts'])} "
        #       f"with {np.min(self.cluster_stats['cluster_counts'])} nodes")

        return node_clusters

    def generate_node_features(self, node_clusters: np.ndarray) -> np.ndarray:
        """
        Generate node features based on assigned clusters.

        Args:
            node_clusters: Array of cluster IDs for each node

        Returns:
            Node feature matrix (n_nodes x feature_dim)
        """
        n_nodes = len(node_clusters)
        features = np.zeros((n_nodes, self.feature_dim))

        # Generate features for all nodes of each cluster at once
        for cluster_id in np.unique(node_clusters):
            # Get nodes with this cluster
            cluster_mask = node_clusters == cluster_id
            n_cluster_nodes = np.sum(cluster_mask)

            if n_cluster_nodes == 0:
                continue

            # Get cluster center
            center = self.cluster_centers[cluster_id]

            # Covariance matrix
            cov = np.eye(self.feature_dim) * self.cluster_variance

            # Generate features from multivariate normal
            cluster_features = np.random.multivariate_normal(
                mean=center, cov=cov, size=n_cluster_nodes
            )

            # Assign to nodes with this cluster
            features[cluster_mask] = cluster_features

        return features

    def analyze_cluster_community_relationship(self):
        """
        Analyze how clusters are distributed across communities.
        Simplified version that returns basic stats about cluster usage.

        Returns:
            Dictionary with analysis metrics
        """
        if not hasattr(self, "cluster_stats") or not self.cluster_stats:
            return {"error": "No nodes have been assigned to clusters yet"}

        # Calculate cluster balance
        cluster_counts = self.cluster_stats["cluster_counts"]

        return {
            "community_exclusivity": 1.0,  # Always 1.0 in simplified version
            "cluster_skewness": 0.0,  # Always 0.0 in simplified version
            "cluster_counts": cluster_counts.tolist(),
        }
