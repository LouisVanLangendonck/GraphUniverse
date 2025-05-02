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
        
        # Generate regime prototypes
        self.regime_prototypes = self._generate_regime_prototypes()
    
    def _generate_regime_prototypes(self) -> np.ndarray:
        """
        Generate feature regimes with smooth control over similarities.
        
        Each regime's prototype is a weighted combination of:
        1. Global component (shared across all communities)
        2. Community component (shared within community)
        3. Unique component (specific to regime)
        
        The weights are determined by inter_similarity and intra_similarity:
        - High inter_similarity → strong global component
        - High intra_similarity → strong community component
        - Remaining weight goes to unique component
        
        Returns:
            Array of regime prototypes (universe_K * regimes_per_community × feature_dim)
        """
        total_regimes = self.universe_K * self.regimes_per_community
        regime_prototypes = np.zeros((total_regimes, self.feature_dim))
        
        # Generate normalized random vectors for each component
        global_base = np.random.normal(0, 1, size=self.feature_dim)
        global_base = global_base / np.linalg.norm(global_base)
        
        community_bases = np.random.normal(0, 1, size=(self.universe_K, self.feature_dim))
        community_bases = community_bases / np.linalg.norm(community_bases, axis=1, keepdims=True)
        
        # Calculate weights for each component
        # As inter_similarity increases, global weight increases
        global_weight = self.inter_similarity
        
        # Remaining influence is split between community and unique components
        # As intra_similarity increases, community weight increases relative to unique weight
        remaining_weight = 1.0 - global_weight
        community_weight = remaining_weight * self.intra_similarity
        unique_weight = remaining_weight * (1.0 - self.intra_similarity)
        
        # Generate regimes
        regime_id = 0
        for comm_id in range(self.universe_K):
            comm_base = community_bases[comm_id]
            
            for _ in range(self.regimes_per_community):
                # Generate unique component for this regime
                unique_base = np.random.normal(0, 1, size=self.feature_dim)
                unique_base = unique_base / np.linalg.norm(unique_base)
                
                # Combine components with weights
                prototype = (global_weight * global_base +
                           community_weight * comm_base +
                           unique_weight * unique_base)
                
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
        
        Args:
            node_regimes: Array of regime IDs for each node
            
        Returns:
            Node feature matrix (n_nodes × feature_dim)
        """
        n_nodes = len(node_regimes)
        features = np.zeros((n_nodes, self.feature_dim))
        
        for i in range(n_nodes):
            regime_id = node_regimes[i]
            prototype = self.regime_prototypes[regime_id]
            
            # Generate random noise
            noise = np.random.normal(0, self.feature_variance, size=self.feature_dim)
            noise = noise / np.linalg.norm(noise)
            
            # Combine prototype and noise
            features[i] = prototype + self.feature_variance * noise
            
            # Normalize
            features[i] = features[i] / np.linalg.norm(features[i])
            
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
            for node_idx in comm_nodes:
                # Sample regime for this node
                regime_idx = np.random.choice(self.regimes_per_community, p=probs)
                
                # Convert to global regime ID
                global_regime_id = comm_id * self.regimes_per_community + regime_idx
                node_regimes[node_idx] = global_regime_id
        
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
    
    def generate_rule(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> Tuple[Callable, str, int]:
        """
        Generate a single rule.
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance to frequency vectors
            
        Returns:
            Tuple of (rule_function, rule_description, hop_distance)
        """
        # Pick random hop distance within specified range
        hop = np.random.randint(self.min_hop, self.max_hop + 1)
        freq_vectors = freq_vectors_by_hop[hop]
        
        # Rest of the rule generation remains the same
        n_conditions = np.random.randint(1, 4)
        regimes = np.random.choice(freq_vectors.shape[1], size=n_conditions, replace=False)
        
        conditions = []
        rule_str_parts = []
        for regime in regimes:
            freqs = freq_vectors[:, regime]
            non_zero_freqs = freqs[freqs > 0]
            
            if len(non_zero_freqs) == 0:
                continue
                
            condition_type = np.random.choice(['threshold', 'range', 'ratio'])
            
            if condition_type == 'threshold':
                threshold = np.random.choice(np.percentile(non_zero_freqs, [25, 50, 75]))
                op = np.random.choice(['>', '<'])
                conditions.append(lambda x, r=regime, t=threshold, o=op: 
                                eval(f"x[r] {o} {t}"))
                rule_str_parts.append(f"Regime {regime} {op} {threshold:.2f}")
                
            elif condition_type == 'range':
                low = np.percentile(non_zero_freqs, 25)
                high = np.percentile(non_zero_freqs, 75)
                conditions.append(lambda x, r=regime, l=low, h=high: 
                                l <= x[r] <= h)
                rule_str_parts.append(f"{low:.2f} ≤ Regime {regime} ≤ {high:.2f}")
                
            elif condition_type == 'ratio':
                other_regime = np.random.choice([r for r in range(freq_vectors.shape[1]) 
                                               if r != regime])
                ratio = np.random.choice([0.5, 1.0, 2.0])
                conditions.append(lambda x, r1=regime, r2=other_regime, rat=ratio:
                                x[r1] > rat * x[r2] if x[r2] > 0 else False)
                rule_str_parts.append(f"Regime {regime} > {ratio}×Regime {other_regime}")
        
        if not conditions:
            return self.generate_rule(freq_vectors_by_hop)
        
        rule_fn = lambda x: all(c(x) for c in conditions)
        rule_str = f"At {hop}-hop: " + " AND ".join(rule_str_parts)
        
        return rule_fn, rule_str, hop
    
    def generate_rules(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> List[Tuple[Callable, int, str, int]]:
        """
        Generate a set of rules and assign them to labels.
        Makes sure rules have sufficient support and don't overlap too much.
        Reserves last label for 'rest' class.
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance to frequency vectors
            
        Returns:
            List of (rule_function, label, rule_description, hop_distance) tuples
        """
        rules = []
        nodes_covered = set()
        n_nodes = freq_vectors_by_hop[self.min_hop].shape[0]
        max_attempts = 100
        
        available_labels = list(range(self.n_labels - 1))
        
        for attempt in range(max_attempts):
            if len(rules) >= len(available_labels) * self.max_rules_per_label:
                break
                
            rule_fn, rule_str, hop = self.generate_rule(freq_vectors_by_hop)
            
            # Check which nodes this rule applies to
            applicable_nodes = set()
            for node in range(n_nodes):
                if rule_fn(freq_vectors_by_hop[hop][node]):
                    applicable_nodes.add(node)
            
            # Check support
            support = len(applicable_nodes) / n_nodes
            if support < self.min_support:
                continue
                
            if applicable_nodes:
                overlap = len(applicable_nodes & nodes_covered) / len(applicable_nodes)
                if overlap > 0.5:
                    continue
                
                label_counts = Counter(label for _, label, _, _ in rules)
                label = min(available_labels, key=lambda l: label_counts[l])
                
                rules.append((rule_fn, label, rule_str, hop))
                nodes_covered.update(applicable_nodes)
        
        self.rules = rules
        return rules
    
    def apply_rules(self, freq_vectors_by_hop: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply rules to generate labels for nodes.
        If multiple rules apply, use the most specific one (most conditions).
        If no rules apply, assign to the 'rest' label (n_labels - 1).
        
        Args:
            freq_vectors_by_hop: Dict mapping hop distance to frequency vectors
            
        Returns:
            Tuple of (node_labels, applied_rule_descriptions)
        """
        n_nodes = freq_vectors_by_hop[self.min_hop].shape[0]
        labels = np.full(n_nodes, -1)
        applied_rules = [None] * n_nodes
        
        available_labels = list(range(self.n_labels - 1))
        
        for node in range(n_nodes):
            matching_rules = []
            
            for rule_fn, label, rule_str, hop in self.rules:
                if label in available_labels:
                    if rule_fn(freq_vectors_by_hop[hop][node]):
                        matching_rules.append((rule_str.count('AND'), label, rule_str))
            
            if matching_rules:
                n_conditions, label, rule_str = max(matching_rules, key=lambda x: x[0])
                labels[node] = label
                applied_rules[node] = rule_str
            else:
                labels[node] = self.n_labels - 1
                applied_rules[node] = "No rule matched - assigned to 'rest' class"
        
        return labels, applied_rules 