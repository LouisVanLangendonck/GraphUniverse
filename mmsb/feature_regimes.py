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

class FeatureRegimeGenerator:
    """
    Handles feature regime generation for communities.
    Each community has multiple feature regimes, with controlled similarity within and between communities.
    """
    
    def __init__(
        self,
        universe_K: int,
        feature_dim: int,
        regimes_per_community: int = 2,
        intra_community_regime_similarity: float = 0.8,
        inter_community_regime_similarity: float = 0.2,
        feature_variance: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the feature regime generator.
        
        Args:
            universe_K: Total number of communities in the universe
            feature_dim: Dimension of node features
            regimes_per_community: Number of feature regimes per community
            intra_community_regime_similarity: How similar regimes within same community should be (0-1)
            inter_community_regime_similarity: How similar regimes between communities should be (0-1)
            feature_variance: Variance of features within each regime
            seed: Random seed for reproducibility
        """
        self.universe_K = universe_K
        self.feature_dim = feature_dim
        self.regimes_per_community = regimes_per_community
        self.intra_similarity = intra_community_regime_similarity
        self.inter_similarity = inter_community_regime_similarity
        self.feature_variance = feature_variance
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create the mapping from regime_id to (community_id, regime_idx)
        self.regime_map = {}
        regime_id = 0
        for comm_id in range(universe_K):
            for regime_idx in range(regimes_per_community):
                self.regime_map[regime_id] = (comm_id, regime_idx)
                regime_id += 1
        
        # Pre-compute base components for feature generation
        self._precompute_feature_components()
        
        # Generate regime prototypes
        self.regime_prototypes = self._generate_regime_prototypes()
        
        # Pre-compute feature distributions for each regime
        self._precompute_feature_distributions()
    
    def _precompute_feature_components(self):
        """Pre-compute base components for feature generation."""
        # Generate global base (shared across all communities)
        self.global_base = np.random.normal(0, 1, size=self.feature_dim)
        self.global_base = self.global_base / np.linalg.norm(self.global_base)
        
        # Generate community bases
        self.community_bases = np.random.normal(0, 1, size=(self.universe_K, self.feature_dim))
        self.community_bases = self.community_bases / np.linalg.norm(self.community_bases, axis=1, keepdims=True)
        
        # Calculate weights for components
        self.global_weight = self.inter_similarity
        remaining_weight = 1.0 - self.global_weight
        self.community_weight = remaining_weight * self.intra_similarity
        self.unique_weight = remaining_weight * (1.0 - self.intra_similarity)
    
    def _precompute_feature_distributions(self):
        """Pre-compute feature distributions for each regime."""
        total_regimes = self.universe_K * self.regimes_per_community
        
        # Store mean vectors and covariance matrices for each regime
        self.regime_distributions = {}
        
        for regime_id in range(total_regimes):
            comm_id, regime_idx = self.regime_map[regime_id]
            
            # Get prototype for this regime
            prototype = self.regime_prototypes[regime_id]
            
            # Create covariance matrix based on feature variance
            # We use a diagonal covariance matrix for efficiency
            covariance = np.eye(self.feature_dim) * self.feature_variance
            
            self.regime_distributions[regime_id] = {
                'mean': prototype,
                'covariance': covariance
            }
    
    def _generate_regime_prototypes(self) -> np.ndarray:
        """
        Generate feature regimes with smooth control over similarities.
        Uses pre-computed components for efficiency.
        """
        total_regimes = self.universe_K * self.regimes_per_community
        regime_prototypes = np.zeros((total_regimes, self.feature_dim))
        
        regime_id = 0
        for comm_id in range(self.universe_K):
            comm_base = self.community_bases[comm_id]
            
            for _ in range(self.regimes_per_community):
                # Generate unique component for this regime
                unique_base = np.random.normal(0, 1, size=self.feature_dim)
                unique_base = unique_base / np.linalg.norm(unique_base)
                
                # Combine components with pre-computed weights
                prototype = (self.global_weight * self.global_base +
                           self.community_weight * comm_base +
                           self.unique_weight * unique_base)
                
                # Normalize the final prototype
                prototype = prototype / np.linalg.norm(prototype)
                regime_prototypes[regime_id] = prototype
                regime_id += 1
        
        return regime_prototypes
    
    def generate_node_features(
        self,
        node_regimes: np.ndarray
    ) -> np.ndarray:
        """
        Generate node features based on assigned regimes.
        Uses pre-computed distributions for efficiency.
        
        Args:
            node_regimes: Array of regime IDs for each node
            
        Returns:
            Node feature matrix (n_nodes × feature_dim)
        """
        n_nodes = len(node_regimes)
        features = np.zeros((n_nodes, self.feature_dim))
        
        # Generate features for all nodes of each regime at once for efficiency
        for regime_id in np.unique(node_regimes):
            # Get nodes with this regime
            regime_mask = node_regimes == regime_id
            n_regime_nodes = np.sum(regime_mask)
            
            if n_regime_nodes == 0:
                continue
            
            # Get pre-computed distribution for this regime
            dist = self.regime_distributions[regime_id]
            
            # Generate features from distribution
            regime_features = np.random.multivariate_normal(
                mean=dist['mean'],
                cov=dist['covariance'],
                size=n_regime_nodes
            )
            
            # Normalize features
            regime_features = regime_features / np.linalg.norm(regime_features, axis=1, keepdims=True)
            
            # Assign to nodes with this regime
            features[regime_mask] = regime_features
        
        return features
    
    def assign_node_regimes(
        self,
        community_assignments: np.ndarray,
        regime_balance: float = 0.5
    ) -> np.ndarray:
        """
        Assign nodes to feature regimes based on their community assignments.
        
        Args:
            community_assignments: Array of community IDs for each node
            regime_balance: Controls how evenly regimes are distributed within communities (0-1)
                0: One regime dominates
                1: Even distribution across regimes
                
        Returns:
            Array of regime IDs for each node
        """
        n_nodes = len(community_assignments)
        node_regimes = np.zeros(n_nodes, dtype=int)
        
        # For each community, assign nodes to regimes
        for comm_id in range(self.universe_K):
            # Get nodes in this community
            comm_nodes = np.where(community_assignments == comm_id)[0]
            
            if len(comm_nodes) == 0:
                continue
                
            # Calculate probabilities for each regime
            if regime_balance == 1.0:
                # Perfect balance - equal probability
                probs = np.ones(self.regimes_per_community) / self.regimes_per_community
            else:
                # Uneven distribution - one regime is more common
                # Lower regime_balance means more skewed distribution
                alpha = 1.0 / (1.0 - regime_balance) if regime_balance < 1.0 else 1.0
                probs = np.random.dirichlet(np.ones(self.regimes_per_community) * alpha)
            
            # Assign regimes to nodes
            regime_indices = np.random.choice(
                self.regimes_per_community,
                size=len(comm_nodes),
                p=probs
            )
            
            # Convert to global regime IDs
            global_regime_ids = comm_id * self.regimes_per_community + regime_indices
            node_regimes[comm_nodes] = global_regime_ids
        
        return node_regimes

    def test_regime_continuity(
        self,
        n_steps: int = 100,
        test_regime_pair: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        """
        Test the continuity of regime generation by measuring distances between regimes
        as similarity parameters change.
        
        Args:
            n_steps: Number of steps to test
            test_regime_pair: Tuple of regime indices to compare
            
        Returns:
            Array of distances between the test regimes at each step
        """
        distances = np.zeros((n_steps, n_steps))
        
        # Test over range of both similarity parameters
        inter_range = np.linspace(0, 1, n_steps)
        intra_range = np.linspace(0, 1, n_steps)
        
        for i, inter_sim in enumerate(inter_range):
            for j, intra_sim in enumerate(intra_range):
                # Store current parameters
                old_inter = self.inter_similarity
                old_intra = self.intra_similarity
                
                # Set test parameters
                self.inter_similarity = inter_sim
                self.intra_similarity = intra_sim
                
                # Generate regimes
                prototypes = self._generate_regime_prototypes()
                
                # Measure distance between test regimes
                regime1, regime2 = test_regime_pair
                dist = np.linalg.norm(prototypes[regime1] - prototypes[regime2])
                distances[i, j] = dist
                
                # Restore parameters
                self.inter_similarity = old_inter
                self.intra_similarity = old_intra
        
        return distances

class NeighborhoodFeatureAnalyzer:
    """
    Analyzes feature regimes in k-hop neighborhoods of nodes.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        node_regimes: np.ndarray,
        total_regimes: int,
        max_hops: int = 3
    ):
        """
        Initialize the neighborhood analyzer.
        
        Args:
            graph: NetworkX graph
            node_regimes: Array of regime IDs for each node
            total_regimes: Total number of regimes in the universe
            max_hops: Maximum number of hops to analyze
        """
        self.graph = graph
        self.node_regimes = node_regimes
        self.total_regimes = total_regimes
        self.max_hops = max_hops
        
        # Pre-compute neighborhoods for efficiency
        self.neighborhoods = self._compute_neighborhoods()
        
        # Pre-compute frequency vectors
        self.frequency_vectors = {}
        for k in range(1, max_hops + 1):
            self.frequency_vectors[k] = self._compute_frequency_vectors(k)
    
    def _compute_neighborhoods(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Compute k-hop neighborhoods for all nodes and hop distances.
        
        Returns:
            Dictionary mapping hop distance to node neighborhoods
        """
        neighborhoods = {}
        
        # For each hop distance
        for k in range(1, self.max_hops + 1):
            neighborhoods[k] = {}
            
            # For each node
            for node in self.graph.nodes():
                # Use NetworkX to find k-hop neighborhood
                if k == 1:
                    # 1-hop is direct neighbors
                    neighbors = list(self.graph.neighbors(node))
                else:
                    # k-hop uses shortest paths
                    neighbors = []
                    for other in self.graph.nodes():
                        if other != node:
                            try:
                                path_length = nx.shortest_path_length(self.graph, node, other)
                                if path_length == k:
                                    neighbors.append(other)
                            except nx.NetworkXNoPath:
                                # No path exists
                                pass
                
                neighborhoods[k][node] = neighbors
        
        return neighborhoods
    
    def _compute_frequency_vectors(self, k: int) -> np.ndarray:
        """
        Compute frequency vectors for k-hop neighborhoods.
        
        Args:
            k: Hop distance
            
        Returns:
            Matrix of frequency vectors (n_nodes × total_regimes)
        """
        n_nodes = self.graph.number_of_nodes()
        frequencies = np.zeros((n_nodes, self.total_regimes))
        
        # For each node
        for node in self.graph.nodes():
            # Get k-hop neighbors
            neighbors = self.neighborhoods[k][node]
            
            if not neighbors:
                continue
                
            # Count regimes in neighborhood
            regime_counts = Counter([self.node_regimes[nbr] for nbr in neighbors])
            
            # Convert to frequency vector
            for regime, count in regime_counts.items():
                frequencies[node, regime] = count / len(neighbors)
        
        return frequencies
    
    def get_frequency_vector(
        self,
        node: int,
        k: int
    ) -> np.ndarray:
        """
        Get the frequency vector for a node's k-hop neighborhood.
        
        Args:
            node: Node index
            k: Hop distance
            
        Returns:
            Frequency vector
        """
        return self.frequency_vectors[k][node]
    
    def get_all_frequency_vectors(
        self,
        k: int
    ) -> np.ndarray:
        """
        Get frequency vectors for all nodes at k hops.
        
        Args:
            k: Hop distance
            
        Returns:
            Matrix of frequency vectors
        """
        return self.frequency_vectors[k]

class FeatureRegimeLabelGenerator:
    """
    Generates node labels based on neighborhood feature regime distributions.
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
            frequency_vectors: Matrix of neighborhood frequency vectors
            n_labels: Number of node labels to generate
            balance_tolerance: Maximum allowed class imbalance (0-1)
            max_iterations: Maximum clustering iterations to achieve balance
            seed: Random seed
        """
        self.frequency_vectors = frequency_vectors
        self.n_labels = n_labels
        self.balance_tolerance = balance_tolerance
        self.max_iterations = max_iterations
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate labels
        self.node_labels, self.cluster_centers = self._generate_balanced_labels()
        
        # Extract interpretable rules
        self.label_rules = self._extract_label_rules()
    
    def _generate_balanced_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate balanced labels using k-means clustering with iterations to improve balance.
        
        Returns:
            Tuple of (node labels, cluster centers)
        """
        n_nodes = self.frequency_vectors.shape[0]
        
        # Scale features for better clustering
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(self.frequency_vectors)
        
        # Initialize with basic k-means
        best_labels = None
        best_balance = 0.0
        best_centers = None
        
        for iteration in range(self.max_iterations):
            # Run k-means with different initialization
            kmeans = KMeans(
                n_clusters=self.n_labels,
                n_init=10,
                random_state=self.seed + iteration if self.seed else None
            )
            labels = kmeans.fit_predict(scaled_vectors)
            
            # Calculate class balance
            class_counts = np.bincount(labels, minlength=self.n_labels)
            expected_count = n_nodes / self.n_labels
            
            # Calculate balance as 1 - (max deviation / expected)
            max_deviation = np.max(np.abs(class_counts - expected_count))
            balance = 1.0 - (max_deviation / expected_count)
            
            # Keep best result
            if best_labels is None or balance > best_balance:
                best_labels = labels
                best_balance = balance
                best_centers = kmeans.cluster_centers_
            
            # Stop if balance is good enough
            if balance >= (1.0 - self.balance_tolerance):
                break
        
        # Transform the centers back to original scale
        original_centers = scaler.inverse_transform(best_centers)
        
        return best_labels, original_centers
    
    def _extract_label_rules(self) -> List[str]:
        """
        Extract interpretable rules for each label class.
        
        Returns:
            List of rule strings for each label
        """
        rules = []
        
        for label in range(self.n_labels):
            # Get center for this label
            center = self.cluster_centers[label]
            
            # Find dominant regimes (top 2)
            top_regimes = np.argsort(center)[-2:][::-1]
            top_values = center[top_regimes]
            
            # Create rule
            rule = f"Label {label}: "
            rule_parts = []
            
            for regime, value in zip(top_regimes, top_values):
                if value > 0.1:  # Only include significant regime contributions
                    rule_parts.append(f"Regime {regime} ({value:.0%})")
            
            rule += " + ".join(rule_parts)
            rules.append(rule)
        
        return rules
    
    def get_node_labels(self) -> np.ndarray:
        """
        Get the generated node labels.
        
        Returns:
            Array of node labels
        """
        return self.node_labels

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