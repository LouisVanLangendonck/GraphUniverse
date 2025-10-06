import numpy as np

from .feature_generator import FeatureGenerator


class GraphUniverse:
    """
    Represents a generative universe for graph instances sampled from a master "pseudo" stochastic block model.
    The GraphSample class will randomly sub-sample from these global universe properties.

    Initialize a graph universe with K communities and optional feature generation.

    Args:
        K: Number of communities
        edge_propensity_variance: Amount of variance in the edge propensities
        feature_dim: Dimension of node features
        center_variance: Separation between cluster centers
        cluster_variance: Spread within each cluster
        seed: Random seed for reproducibility
        P: Optional propensity matrix (if None, will be generated)
    """

    def __init__(
        self,
        K: int,
        edge_propensity_variance: float = 0.5,  # 0-1: how much variance in the edge propensities. Only used if use_dccc_sbm is True
        feature_dim: int = 0,
        center_variance: float = 1.0,  # Separation between cluster centers
        cluster_variance: float = 0.1,  # Spread within each cluster
        seed: int | None = 42,  # Random seed
        P: np.ndarray
        | None = None,  # If we want to use a pre-defined probability matrix, we can pass it in here
    ):
        self.K = K
        self.feature_dim = feature_dim
        self.center_variance = center_variance
        self.cluster_variance = cluster_variance
        self.seed = seed

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Generate or use provided propensity matrix
        if P is None:
            self.P = self._generate_edge_propensity_matrix(K, edge_propensity_variance)
        else:
            self.P = P

        # Initialize feature generator if features are enabled
        if feature_dim > 0:
            self.feature_generator = FeatureGenerator(
                universe_K=K,
                feature_dim=feature_dim,
                cluster_count_factor=1.0,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
                assignment_skewness=0.0,
                community_exclusivity=1.0,
                seed=seed,
            )
        else:
            self.feature_generator = None

        # Store parameters
        self.edge_propensity_variance = edge_propensity_variance

        # Generate community-degree propensity vector based on method
        self.community_degree_propensity_vector = np.random.uniform(-1, 1, K)

    def _generate_edge_propensity_matrix(
        self, K: int, edge_propensity_variance: float = 0.0
    ) -> np.ndarray:
        """
        Generate an edge propensity matrix that gives the RELATIVE propensities BETWEEN different communities (the intra-community prob / homophily is decided by the GraphFamily object and scaled accordingly in a GraphSample object)

        Args:
            K: Number of communities
            edge_propensity_variance: Amount of variance in the edge propensities

        Returns:
            K x K propensity matrix
        """

        P = np.ones((K, K))

        # Add the edge propensity variance if requested
        if edge_propensity_variance > 0:
            noise = np.random.normal(0, edge_propensity_variance * 2, size=(K, K))
            P = P + noise
            # Clip to be between 0 and 2
            P[P < 0.0] = 0.0
            P[P > 2.0] = 2.0

        # Have the P to be symmetric (just set all values of lower triangle to be the same as the upper triangle)
        for i in range(K):
            for j in range(i + 1, K):
                P[i, j] = P[j, i]

        return P

    def sample_connected_community_subset(
        self, size: int, seed: int | None = None, use_cooccurrence: bool = True
    ) -> list[int]:
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

        # if not use_cooccurrence or self.community_cooccurrence_homogeneity == 1.0:
        # Sample community one by one and always check for a new candidate that is has a non-zero probabilty connection to the existing communities
        result = [np.random.choice(self.K)]
        while len(result) < size:
            new_community = np.random.choice(self.K)
            # Check if self.P[new_community, result] is non-zero
            if np.sum(self.P[new_community, result]) > 0 and new_community not in result:
                result.append(new_community)
        return result

        # # Start with a random seed community
        # first_community = np.random.choice(K)
        # result = {first_community}
        # remaining_size = size - 1

        # # Iteratively add communities based on co-occurrence probabilities
        # while remaining_size > 0 and len(result) < K:

        #     # Calculate sampling probabilities based on co-occurrence with existing communities
        #     remaining_communities = list(set(range(K)) - result)
        #     if not remaining_communities:
        #         break

        #     # For each remaining community, calculate its average co-occurrence with selected ones
        #     cooccurrence_scores = np.zeros(len(remaining_communities))
        #     for i, candidate in enumerate(remaining_communities):
        #         # Average co-occurrence probability with all selected communities
        #         avg_cooccurrence = np.mean([
        #             self.community_cooccurrence_matrix[candidate, selected]
        #             for selected in result
        #         ])
        #         cooccurrence_scores[i] = avg_cooccurrence

        #     # Convert scores to probabilities
        #     if np.sum(cooccurrence_scores) > 0:
        #         probabilities = cooccurrence_scores / np.sum(cooccurrence_scores)
        #     else:
        #         # Fallback to uniform if all scores are zero
        #         probabilities = np.ones(len(remaining_communities)) / len(remaining_communities)

        #     # Sample next community
        #     next_idx = np.random.choice(len(remaining_communities), p=probabilities)
        #     next_community = remaining_communities[next_idx]
        #     result.add(next_community)
        #     remaining_size -= 1

        # return list(result)
