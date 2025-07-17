"""
Feature regime generation and analysis for MMSB graphs.

This module provides functionality for:
1. Generating multiple feature regimes per community
2. Analyzing k-hop neighborhood feature distributions
3. Creating balanced node labels based on feature regimes
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

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

class NeighborhoodFeatureAnalyzer:
    """
    Analyzes feature distributions in k-hop neighborhoods.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        node_clusters: np.ndarray,
        total_clusters: int,
        max_hops: int = 3
    ):
        """
        Initialize the analyzer.
        
        Args:
            graph: NetworkX graph
            node_clusters: Array of cluster IDs for each node
            total_clusters: Total number of clusters
            max_hops: Maximum number of hops to analyze
        """
        self.graph = graph
        self.node_clusters = node_clusters
        self.total_clusters = total_clusters
        self.max_hops = max_hops
        
        # Compute neighborhoods
        self.neighborhoods = self._compute_neighborhoods()
        
        # Compute frequency vectors for each hop
        self.frequency_vectors = {}
        for k in range(1, max_hops + 1):
            self.frequency_vectors[k] = self._compute_frequency_vectors(k)
    
    def _compute_neighborhoods(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Compute k-hop neighborhoods for each node.
        
        Returns:
            Dictionary mapping hop distance to node neighborhoods
        """
        neighborhoods = {}
        
        # For each hop distance
        for k in range(1, self.max_hops + 1):
            node_neighbors = {}
            
            # For each node
            for node in self.graph.nodes():
                # Get k-hop neighbors using NetworkX
                neighbors = set()
                for nbr in nx.single_source_shortest_path_length(self.graph, node, cutoff=k).keys():
                    if nbr != node:  # Exclude self
                        neighbors.add(nbr)
                node_neighbors[node] = sorted(list(neighbors))
                
            neighborhoods[k] = node_neighbors
            
        return neighborhoods
    
    def _compute_frequency_vectors(self, k: int) -> np.ndarray:
        """
        Compute frequency vectors for k-hop neighborhoods.
        
        Args:
            k: Hop distance
            
        Returns:
            Matrix of frequency vectors (n_nodes × total_clusters)
        """
        n_nodes = len(self.graph)
        freq_vectors = np.zeros((n_nodes, self.total_clusters))
        
        # For each node
        for node in range(n_nodes):
            # Get k-hop neighbors
            neighbors = self.neighborhoods[k][node]
            
            if not neighbors:
                continue
                
            # Count cluster frequencies
            cluster_counts = Counter([self.node_clusters[nbr] for nbr in neighbors])
            
            # Convert to frequency vector
            for cluster, count in cluster_counts.items():
                freq_vectors[node, cluster] = count / len(neighbors)
                
        return freq_vectors
    
    def get_frequency_vector(
        self,
        node: int,
        k: int
    ) -> np.ndarray:
        """
        Get frequency vector for a specific node and hop distance.
        
        Args:
            node: Node ID
            k: Hop distance
            
        Returns:
            Frequency vector for the node's k-hop neighborhood
        """
        if k not in self.frequency_vectors:
            raise ValueError(f"Frequency vectors not computed for hop distance {k}")
            
        return self.frequency_vectors[k][node]
    
    def get_all_frequency_vectors(
        self,
        k: int
    ) -> np.ndarray:
        """
        Get frequency vectors for all nodes at a specific hop distance.
        
        Args:
            k: Hop distance
            
        Returns:
            Matrix of frequency vectors (n_nodes × total_clusters)
        """
        if k not in self.frequency_vectors:
            raise ValueError(f"Frequency vectors not computed for hop distance {k}")
            
        return self.frequency_vectors[k]

class FeatureClusterLabelGenerator:
    """
    Generates balanced node labels based on feature cluster distributions in neighborhoods.
    """
    
    def __init__(
        self,
        frequency_vectors: np.ndarray,
        n_labels: int = 4,
        balance_tolerance: float = 0.1,
        max_iterations: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the label generator.
        
        Args:
            frequency_vectors: Matrix of cluster frequencies in neighborhoods (n_nodes × n_clusters)
            n_labels: Number of label classes to generate
            balance_tolerance: Maximum allowed class imbalance (0-1)
            max_iterations: Maximum number of iterations for balancing
            seed: Random seed for reproducibility
        """
        self.frequency_vectors = frequency_vectors
        self.n_labels = n_labels
        self.balance_tolerance = balance_tolerance
        self.max_iterations = max_iterations
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Generate balanced labels
        self.labels, self.label_rules = self._generate_balanced_labels()
    
    def _generate_balanced_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate balanced node labels based on cluster frequencies.
        
        Returns:
            Tuple of (labels, label_rules)
        """
        n_nodes = len(self.frequency_vectors)
        
        # Initialize labels randomly
        labels = np.random.randint(0, self.n_labels, size=n_nodes)
        
        # Track label counts
        label_counts = np.zeros(self.n_labels)
        for label in labels:
            label_counts[label] += 1
            
        # Target count per label
        target_count = n_nodes / self.n_labels
        
        # Iteratively balance labels
        for _ in range(self.max_iterations):
            # Check if balanced enough
            max_deviation = np.max(np.abs(label_counts - target_count)) / target_count
            if max_deviation <= self.balance_tolerance:
                break
                
            # Find most imbalanced labels
            deviations = np.abs(label_counts - target_count)
            most_imbalanced = np.argsort(deviations)[-2:]  # Get 2 most imbalanced
            
            # Get nodes with these labels
            nodes_to_swap = []
            for label in most_imbalanced:
                nodes = np.where(labels == label)[0]
                nodes_to_swap.extend(nodes)
                
            # Try swapping labels between these nodes
            for i in range(0, len(nodes_to_swap), 2):
                if i + 1 >= len(nodes_to_swap):
                    break
                    
                node1, node2 = nodes_to_swap[i], nodes_to_swap[i + 1]
                label1, label2 = labels[node1], labels[node2]
                
                # Check if swap improves balance
                old_deviation = np.abs(label_counts[label1] - target_count) + np.abs(label_counts[label2] - target_count)
                
                # Swap labels
                labels[node1], labels[node2] = label2, label1
                label_counts[label1] -= 1
                label_counts[label2] -= 1
                label_counts[label2] += 1
                label_counts[label1] += 1
                
                # Check new deviation
                new_deviation = np.abs(label_counts[label1] - target_count) + np.abs(label_counts[label2] - target_count)
                
                # Revert if no improvement
                if new_deviation >= old_deviation:
                    labels[node1], labels[node2] = label1, label2
                    label_counts[label1] -= 1
                    label_counts[label2] -= 1
                    label_counts[label2] += 1
                    label_counts[label1] += 1
        
        # Extract rules for each label
        label_rules = self._extract_label_rules()
        
        return labels, label_rules
    
    def _extract_label_rules(self) -> List[str]:
        """
        Extract rules that characterize each label class.
        
        Returns:
            List of rule descriptions
        """
        rules = []
        
        # For each label
        for label in range(self.n_labels):
            # Get nodes with this label
            label_nodes = np.where(self.labels == label)[0]
            
            if len(label_nodes) == 0:
                rules.append(f"Label {label}: No nodes assigned")
                continue
                
            # Get cluster frequencies for these nodes
            label_freqs = self.frequency_vectors[label_nodes]
            
            # Find most distinctive clusters
            mean_freqs = np.mean(label_freqs, axis=0)
            std_freqs = np.std(label_freqs, axis=0)
            
            # Get clusters with high mean and low std
            distinctive = np.where((mean_freqs > 0.3) & (std_freqs < 0.2))[0]
            
            if len(distinctive) == 0:
                rules.append(f"Label {label}: No distinctive clusters found")
                continue
                
            # Create rule description
            rule_parts = []
            for cluster in distinctive:
                rule_parts.append(f"cluster {cluster} ({mean_freqs[cluster]:.2f})")
                
            rules.append(f"Label {label}: High frequency of " + ", ".join(rule_parts))
            
        return rules
    
    def get_node_labels(self) -> np.ndarray:
        """
        Get the generated node labels.
        
        Returns:
            Array of node labels
        """
        return self.labels

class GenerativeRuleBasedLabeler:
    """
    Generates transferable rules based on k-hop neighborhood feature regime distributions.
    Each rule is a logical condition on regime frequencies at specific hop distances.
    """
    
    def __init__(
        self,
        n_labels: int,
        min_support: float = 0.1,
        max_rules_per_label: int = 3,
        min_hop: int = 1,
        max_hop: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialize the rule-based labeler.
        
        Args:
            n_labels: Number of distinct labels to generate
            min_support: Minimum fraction of nodes a rule should apply to
            max_rules_per_label: Maximum number of rules per label
            min_hop: Minimum hop distance to consider for rules
            max_hop: Maximum hop distance to consider for rules
            seed: Random seed for reproducibility
        """
        self.n_labels = n_labels
        self.min_support = min_support
        self.max_rules_per_label = max_rules_per_label
        self.min_hop = min_hop
        self.max_hop = max_hop
        self.rules = []  # List of (rule_fn, label, rule_str, hop) tuples
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_rule(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> Tuple[Optional[Callable], Optional[str], Optional[int]]:
        """
        Generate a single rule.
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance to frequency vectors
            
        Returns:
            Tuple of (rule_function, rule_description, hop_distance) or (None, None, None) if no valid rule found
        """
        # Pick random hop distance within specified range
        hop = np.random.randint(self.min_hop, self.max_hop + 1)
        freq_vectors = freq_vectors_by_hop[hop]
        
        # Get non-zero frequencies for each regime
        regime_freqs = {}
        for regime in range(freq_vectors.shape[1]):
            freqs = freq_vectors[:, regime]
            non_zero_freqs = freqs[freqs > 0.01]  # Consider frequencies above 1%
            if len(non_zero_freqs) > 0:
                regime_freqs[regime] = non_zero_freqs
        
        if not regime_freqs:
            return None, None, None
        
        # Select regimes that have significant presence
        valid_regimes = [r for r, f in regime_freqs.items() if len(f) >= int(freq_vectors.shape[0] * self.min_support)]
        if not valid_regimes:
            return None, None, None
        
        # Generate 1-2 conditions
        n_conditions = np.random.randint(1, 3)
        if len(valid_regimes) < n_conditions:
            n_conditions = len(valid_regimes)
        
        selected_regimes = np.random.choice(valid_regimes, size=n_conditions, replace=False)
        
        conditions = []
        rule_str_parts = []
        
        for regime in selected_regimes:
            freqs = regime_freqs[regime]
            condition_type = np.random.choice(['threshold', 'range'])
            
            if condition_type == 'threshold':
                # Use more meaningful thresholds based on data distribution
                threshold = np.percentile(freqs, np.random.choice([25, 50, 75]))
                op = np.random.choice(['>', '<'])
                
                # Create condition with proper closure
                if op == '>':
                    conditions.append(lambda x, r=regime, t=threshold: x[r] > t)
                else:
                    conditions.append(lambda x, r=regime, t=threshold: x[r] < t)
                
                rule_str_parts.append(f"Regime {regime} {op} {threshold:.2f}")
                
            elif condition_type == 'range':
                # Use interquartile range for more meaningful bounds
                low = np.percentile(freqs, 25)
                high = np.percentile(freqs, 75)
                
                # Create condition with proper closure
                conditions.append(lambda x, r=regime, l=low, h=high: l <= x[r] <= h)
                rule_str_parts.append(f"{low:.2f} ≤ Regime {regime} ≤ {high:.2f}")
        
        if not conditions:
            return None, None, None
        
        # Combine conditions with AND
        rule_fn = lambda x: all(c(x) for c in conditions)
        rule_str = f"At {hop}-hop: " + " AND ".join(rule_str_parts)
        
        return rule_fn, rule_str, hop

    def generate_rules(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> List[Tuple[Callable, int, str, int]]:
        """
        Generate rules for each label based on frequency vectors.
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance (int) to frequency vectors
            
        Returns:
            List of tuples (rule_function, label, rule_description, hop_distance)
        """
        # Validate input
        if not freq_vectors_by_hop:
            raise ValueError("freq_vectors_by_hop cannot be empty")
        
        # Ensure all required hop distances are present
        required_hops = set(range(self.min_hop, self.max_hop + 1))
        available_hops = set(freq_vectors_by_hop.keys())
        if not required_hops.issubset(available_hops):
            raise ValueError(f"Missing frequency vectors for hop distances {required_hops - available_hops}")
        
        # Get dimensions from first frequency vector
        n_nodes = freq_vectors_by_hop[self.min_hop].shape[0]
        
        # Initialize rules list and tracking of covered nodes
        rules = []
        covered_nodes = np.zeros(n_nodes, dtype=bool)
        
        # Generate rules for each label (except the last one which will be the "rest" class)
        for label in range(self.n_labels - 1):
            label_rules = []
            max_attempts = 20  # Increased attempts for better rule generation
            
            while len(label_rules) < self.max_rules_per_label and max_attempts > 0:
                rule_fn, rule_str, hop = self.generate_rule(freq_vectors_by_hop)
                
                if rule_fn is not None:
                    # Apply rule to get nodes that match
                    matching_nodes = np.array([rule_fn(freq_vectors_by_hop[hop][i]) 
                                            for i in range(n_nodes)])
                    
                    # Only consider uncovered nodes for support calculation
                    uncovered_matches = matching_nodes & ~covered_nodes
                    support = np.mean(uncovered_matches)
                    
                    if support >= self.min_support:
                        # Check overlap with existing rules for this label
                        overlap = False
                        for existing_rule, _, _ in label_rules:
                            existing_matches = np.array([existing_rule(freq_vectors_by_hop[hop][i])
                                                       for i in range(n_nodes)])
                            if np.mean(matching_nodes & existing_matches) > 0.5 * support:
                                overlap = True
                                break
                        
                        if not overlap:
                            label_rules.append((rule_fn, rule_str, hop))
                            covered_nodes |= matching_nodes  # Mark these nodes as covered
                            if len(label_rules) >= self.max_rules_per_label:
                                break
                
                max_attempts -= 1
            
            # Add rules for this label
            for rule_fn, rule_str, hop in label_rules:
                rules.append((rule_fn, label, rule_str, hop))
            
            # If no rules were generated for this label, try simpler rules
            if not label_rules:
                print(f"Warning: No rules generated for label {label}. Trying simpler rules...")
                # Add a simple threshold-based rule
                for hop in range(self.min_hop, self.max_hop + 1):
                    freq_vectors = freq_vectors_by_hop[hop]
                    for regime in range(freq_vectors.shape[1]):
                        threshold = np.median(freq_vectors[:, regime])
                        if threshold > 0:
                            rule_fn = lambda x, r=regime, t=threshold: x[r] > t
                            rule_str = f"At {hop}-hop: Regime {regime} > {threshold:.2f}"
                            rules.append((rule_fn, label, rule_str, hop))
                            break
                    if len(rules) > 0 and rules[-1][1] == label:
                        break
        
        # Ensure we have at least one rule
        if not rules:
            print("Warning: No rules generated. Adding default rule...")
            # Add a simple default rule for label 0
            hop = self.min_hop
            rule_fn = lambda x: True  # Always matches
            rule_str = "Default rule"
            rules.append((rule_fn, 0, rule_str, hop))
        
        self.rules = rules
        return rules

    def apply_rules(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rules to generate node labels.
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance (int) to frequency vectors
            
        Returns:
            Tuple of (node_labels, applied_rules) where applied_rules[i] is the index
            of the rule that generated the label for node i (or None if no rule matched)
        """
        n_nodes = freq_vectors_by_hop[self.min_hop].shape[0]
        labels = np.full(n_nodes, self.n_labels - 1)  # Initialize all nodes to "rest" class
        applied_rules = np.full(n_nodes, None, dtype=object)
        
        # Apply each rule in order
        for rule_idx, (rule_fn, label, _, hop) in enumerate(self.rules):
            # Only apply rule to unlabeled nodes
            for i in range(n_nodes):
                if labels[i] == self.n_labels - 1:  # Node still has "rest" class label
                    if rule_fn(freq_vectors_by_hop[hop][i]):
                        labels[i] = label
                        applied_rules[i] = rule_idx
        
        return labels, applied_rules

def graphsample_to_pyg(graph_sample):
    """
    Convert a GraphSample to a PyTorch Geometric Data object.
    Uses node features from the GraphSample if available, otherwise uses identity features.
    Labels are set to community_labels.
    """
    graph = graph_sample.graph
    n_nodes = graph.number_of_nodes()
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # Use features from GraphSample if available, else identity
    if hasattr(graph_sample, 'features') and graph_sample.features is not None:
        features = torch.tensor(graph_sample.features, dtype=torch.float)
    else:
        features = torch.eye(n_nodes, dtype=torch.float)

    # Use community_labels as y
    if hasattr(graph_sample, 'community_labels') and graph_sample.community_labels is not None:
        y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
    else:
        y = torch.zeros(n_nodes, dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=y)
    return data 