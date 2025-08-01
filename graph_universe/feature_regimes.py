import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

class SimplifiedFeatureGenerator:
    """
    Simplified feature generator using multivariate Gaussian clusters.
    Features are generated with controllable inter-cluster and intra-cluster variance,
    and flexible assignment of clusters to communities.
    """
    
    def __init__(
        self,
        universe_K: int,
        feature_dim: int,
        cluster_count_factor: float = 1.0,  # Number of clusters relative to communities (0.1 to 4.0)
        center_variance: float = 1.0,       # Separation between cluster centers 
        cluster_variance: float = 0.1,      # Spread within each cluster
        assignment_skewness: float = 0.0,   # If some clusters are used more frequently (0.0 to 1.0)
        community_exclusivity: float = 1.0, # How exclusively clusters map to communities (0.0 to 1.0)
        seed: Optional[int] = None
    ):
        """
        Initialize the feature generator.
        
        Args:
            universe_K: Total number of communities in the universe
            feature_dim: Dimension of node features
            cluster_count_factor: Controls number of clusters relative to communities
                                 (0.1 = few clusters, 1.0 = same as communities, 4.0 = many clusters)
            center_variance: Controls separation between cluster centers
            cluster_variance: Controls spread within each cluster
            assignment_skewness: Controls if some clusters are used more frequently
                                (0.0 = balanced, 1.0 = highly skewed)
            community_exclusivity: Controls how exclusively clusters map to communities
                                  (0.0 = shared across communities, 1.0 = exclusive to communities)
            seed: Random seed for reproducibility
        """
        self.universe_K = universe_K
        self.feature_dim = feature_dim
        self.center_variance = center_variance
        self.cluster_variance = cluster_variance
        self.assignment_skewness = assignment_skewness
        self.community_exclusivity = community_exclusivity
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Determine number of clusters
        self.n_clusters = max(1, int(universe_K * cluster_count_factor))
        print(f"Creating {self.n_clusters} feature clusters for {universe_K} communities")
        
        # Generate cluster centers
        self.cluster_centers = self._generate_cluster_centers()
        
        # Create community-cluster mapping
        self.community_cluster_probs = self._create_community_cluster_mapping()
        
        # Track assignment statistics
        self.cluster_stats = {}
    
    def _generate_cluster_centers(self) -> np.ndarray:
        """
        Generate cluster centers with controlled separation.
        
        Returns:
            Matrix of cluster centers (n_clusters × feature_dim)
        """
        # Generate centers from multivariate normal distribution
        centers = np.random.normal(0, self.center_variance, size=(self.n_clusters, self.feature_dim))
        
        # Normalize to unit sphere for consistent scaling
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        centers = centers / norms
        
        return centers
    
    def _create_community_cluster_mapping(self) -> np.ndarray:
        """
        Create mapping probabilities between communities and clusters.
        Handles both n_clusters >= n_communities and n_clusters < n_communities cases.
        Ensures every community is assigned at least one cluster.
        Skewness is applied to community popularity (number of clusters per community).
        Returns:
            Matrix of community-cluster probabilities (universe_K × n_clusters)
        """
        probs = np.zeros((self.universe_K, self.n_clusters))
        base_prob = (1.0 - self.community_exclusivity) / self.n_clusters
        probs.fill(base_prob)

        # --- Determine how many clusters each community should get ---
        min_clusters_per_community = 1
        max_extra_clusters = max(0, self.n_clusters - self.universe_K)

        # Skewed popularity: how many clusters each community wants
        if self.assignment_skewness > 0:
            alpha = 1.0 / self.assignment_skewness
            raw_popularity = np.random.exponential(scale=alpha, size=self.universe_K)
        else:
            raw_popularity = np.ones(self.universe_K)
        # Normalize to sum to 1
        popularity = raw_popularity / np.sum(raw_popularity)

        # Each community gets at least one cluster, distribute remaining clusters by popularity
        extra_clusters = np.round(popularity * max_extra_clusters).astype(int) if max_extra_clusters > 0 else np.zeros(self.universe_K, dtype=int)
        # Adjust to ensure sum is correct
        while np.sum(extra_clusters) < max_extra_clusters:
            extra_clusters[np.argmax(popularity)] += 1
        while np.sum(extra_clusters) > max_extra_clusters:
            extra_clusters[np.argmax(extra_clusters)] -= 1
        clusters_per_community = min_clusters_per_community + extra_clusters

        # --- Assignment matrix ---
        # Case 1: More clusters than communities (assign clusters to communities)
        if self.n_clusters >= self.universe_K:
            available_clusters = list(range(self.n_clusters))
            np.random.shuffle(available_clusters)
            cluster_ptr = 0
            for comm_idx in range(self.universe_K):
                n_assign = clusters_per_community[comm_idx]
                assigned = []
                for _ in range(n_assign):
                    if cluster_ptr >= self.n_clusters:
                        # If we run out, assign randomly
                        cluster = np.random.choice(range(self.n_clusters))
                    else:
                        cluster = available_clusters[cluster_ptr]
                        cluster_ptr += 1
                    assigned.append(cluster)
                    probs[comm_idx, cluster] += self.community_exclusivity / n_assign
        else:
            # Case 2: Fewer clusters than communities (assign communities to clusters)
            # Each community must be assigned to at least one cluster
            # Distribute communities over clusters as evenly as possible, with skewness
            # First, assign each community to a cluster (round-robin)
            for comm_idx in range(self.universe_K):
                cluster = comm_idx % self.n_clusters
                probs[comm_idx, cluster] += self.community_exclusivity / clusters_per_community[comm_idx]
            # Now, for communities that want more clusters, assign additional clusters
            for comm_idx in range(self.universe_K):
                n_assign = clusters_per_community[comm_idx]
                already_assigned = {comm_idx % self.n_clusters}
                if n_assign > 1:
                    # Assign additional clusters randomly (but not the already assigned one)
                    possible_clusters = list(set(range(self.n_clusters)) - already_assigned)
                    if possible_clusters:
                        chosen = np.random.choice(possible_clusters, size=n_assign-1, replace=len(possible_clusters)<(n_assign-1))
                        for cluster in chosen:
                            probs[comm_idx, cluster] += self.community_exclusivity / n_assign
        # Normalize each community's probabilities to sum to 1
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
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
            'cluster_counts': np.zeros(self.n_clusters, dtype=int),
            'community_distribution': np.zeros((self.n_clusters, self.universe_K), dtype=int)
        }
        
        # For each node
        for i in range(n_nodes):
            comm_id = community_assignments[i]
            
            # Sample cluster from community's distribution
            node_clusters[i] = np.random.choice(
                self.n_clusters,
                p=self.community_cluster_probs[comm_id]
            )
            
            # Track stats
            cluster = node_clusters[i]
            self.cluster_stats['cluster_counts'][cluster] += 1
            self.cluster_stats['community_distribution'][cluster, comm_id] += 1
        
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
            Node feature matrix (n_nodes × feature_dim)
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
                mean=center,
                cov=cov,
                size=n_cluster_nodes
            )
            
            # Assign to nodes with this cluster
            features[cluster_mask] = cluster_features
        
        return features
    
    def analyze_cluster_community_relationship(self):
        """
        Analyze how clusters are distributed across communities.
        
        Returns:
            Dictionary with analysis metrics
        """
        if not hasattr(self, 'cluster_stats') or not self.cluster_stats:
            return {"error": "No nodes have been assigned to clusters yet"}
        
        community_dist = self.cluster_stats['community_distribution']
        
        # Calculate entropy for each cluster's community distribution
        entropies = []
        for cluster_idx in range(self.n_clusters):
            dist = community_dist[cluster_idx]
            if np.sum(dist) > 0:
                # Normalize to probabilities
                dist = dist / np.sum(dist)
                # Calculate entropy (-sum(p*log(p)))
                entropy = -np.sum(dist * np.log2(dist + 1e-10))
                # Normalize by max possible entropy
                max_entropy = np.log2(self.universe_K)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                entropies.append(normalized_entropy)
        
        # Overall metrics
        avg_entropy = np.mean(entropies) if entropies else 0
        exclusivity = 1.0 - avg_entropy  # High entropy = low exclusivity
        
        # Calculate cluster balance
        cluster_counts = self.cluster_stats['cluster_counts']
        total_nodes = np.sum(cluster_counts)
        expected_count = total_nodes / self.n_clusters
        gini = 0.0
        
        if total_nodes > 0:
            # Calculate Gini coefficient for cluster size inequality
            cluster_fractions = cluster_counts / total_nodes
            sorted_fractions = np.sort(cluster_fractions)
            cumsum = np.cumsum(sorted_fractions)
            # Gini formula
            gini = 1 - 2 * np.sum((cumsum - sorted_fractions / 2) / self.n_clusters)
        
        return {
            "community_exclusivity": exclusivity,
            "cluster_skewness": gini,
            "cluster_entropies": entropies,
            "cluster_counts": cluster_counts.tolist(),
        }

