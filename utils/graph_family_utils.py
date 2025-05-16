import numpy as np
import math
import random
import networkx as nx

# DeviationCache for caching successful parameter sets
class DeviationCache:
    """Cache for distribution parameters that successfully meet deviation constraints"""
    def __init__(self):
        self.cache = {
            "power_law": [],
            "exponential": [],
            "uniform": [],
            "standard": []
        }
    def add_success(self, dist_type, params, deviation, target_stats):
        self.cache[dist_type].append({
            "params": params,
            "deviation": deviation,
            "target_stats": target_stats
        })
    def find_similar_params(self, dist_type, target_stats, max_entries=3):
        if not self.cache[dist_type]:
            return None
        similarities = []
        for entry in self.cache[dist_type]:
            sim_score = self._calc_similarity(target_stats, entry["target_stats"])
            similarities.append((sim_score, entry))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:max_entries]]
    def _calc_similarity(self, target1, target2):
        common_keys = set(target1.keys()) & set(target2.keys())
        if not common_keys:
            return 0.0
        dist = 0.0
        for key in common_keys:
            range_factor = 1.0
            if "homophily" in key or "density" in key:
                range_factor = 1.0
            elif "nodes" in key or "communities" in key:
                range_factor = 100.0
            norm_diff = abs(target1[key] - target2[key]) / range_factor
            dist += norm_diff**2
        dist = math.sqrt(dist / len(common_keys))
        similarity = max(0.0, 1.0 - dist)
        return similarity
    def get_success_rate(self, dist_type):
        return len(self.cache[dist_type])

# DistributionDiversifier for balanced distribution type selection
class DistributionDiversifier:
    """Manages distribution type selection to ensure diversity"""
    def __init__(self, target_counts=None):
        self.dist_types = ["power_law", "exponential", "uniform", "standard"]
        self.current_counts = {d: 0 for d in self.dist_types}
        if target_counts is None:
            self.target_counts = {d: 25 for d in self.dist_types}
        else:
            self.target_counts = target_counts
        self.deviation_stats = {d: [] for d in self.dist_types}
    def select_distribution(self, deviation_cache, exploration_weight=0.3):
        diversity_scores = {}
        for dist in self.dist_types:
            if self.current_counts[dist] >= self.target_counts[dist]:
                diversity_scores[dist] = 0.0
            else:
                shortfall = self.target_counts[dist] - self.current_counts[dist]
                diversity_scores[dist] = shortfall / self.target_counts[dist]
        success_rates = {}
        for dist in self.dist_types:
            n_successes = deviation_cache.get_success_rate(dist)
            success_rates[dist] = max(0.01, float(n_successes))
        total_success = sum(success_rates.values())
        success_scores = {d: s/total_success for d, s in success_rates.items()}
        scores = {}
        for dist in self.dist_types:
            scores[dist] = (exploration_weight * diversity_scores[dist] +
                           (1 - exploration_weight) * success_scores[dist])
        total_score = sum(scores.values())
        if total_score > 0:
            probs = {d: s/total_score for d, s in scores.items()}
            chosen_dist = np.random.choice(
                list(probs.keys()),
                p=[probs[d] for d in self.dist_types]
            )
        else:
            chosen_dist = np.random.choice(self.dist_types)
        return chosen_dist
    def record_success(self, dist_type, deviation):
        self.current_counts[dist_type] += 1
        self.deviation_stats[dist_type].append(deviation)
    def get_stats(self):
        return {
            "counts": self.current_counts,
            "targets": self.target_counts,
            "avg_deviations": {
                d: np.mean(devs) if devs else 0.0 
                for d, devs in self.deviation_stats.items()
            }
        }

# Helper functions for degree factor generation

def generate_power_law_factors(n_nodes, exponent, target_avg_degree):
    factors = np.random.pareto(exponent - 1, size=n_nodes) + 1
    factors = factors / factors.mean() * target_avg_degree
    factors = np.maximum(factors, 1.0)
    return factors

def generate_exponential_factors(n_nodes, rate, target_avg_degree):
    factors = np.random.exponential(1/rate, size=n_nodes) + 1
    factors = factors / factors.mean() * target_avg_degree
    factors = np.maximum(factors, 1.0)
    return factors

def generate_uniform_factors(n_nodes, min_factor, max_factor, target_avg_degree):
    factors = np.random.uniform(min_factor, max_factor, size=n_nodes)
    factors = factors / factors.mean() * target_avg_degree
    factors = np.maximum(factors, 1.0)
    return factors

# Helper for edge generation (simplified, undirected)
def generate_edges_with_config(community_labels, P_sub, degree_factors):
    n_nodes = len(community_labels)
    i_nodes, j_nodes = np.triu_indices(n_nodes, k=1)
    comm_i = community_labels[i_nodes]
    comm_j = community_labels[j_nodes]
    edge_probs = P_sub[comm_i, comm_j]
    edge_probs *= degree_factors[i_nodes] * degree_factors[j_nodes]
    edge_probs = np.clip(edge_probs, 0, 1)
    edges = np.random.random(len(edge_probs)) < edge_probs
    rows = i_nodes[edges]
    cols = j_nodes[edges]
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.ones(len(all_rows))
    try:
        import scipy.sparse as sp
        adj = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
        return adj
    except ImportError:
        return None

def calculate_community_structure_deviation(graph, community_labels, P_sub):
    n_communities = P_sub.shape[0]
    actual_matrix = np.zeros((n_communities, n_communities))
    community_sizes = np.zeros(n_communities, dtype=int)
    for label in community_labels:
        community_sizes[label] += 1
    for i, j in graph.edges():
        comm_i = community_labels[i]
        comm_j = community_labels[j]
        actual_matrix[comm_i, comm_j] += 1
        actual_matrix[comm_j, comm_i] += 1
    for i in range(n_communities):
        for j in range(n_communities):
            if i == j:
                n = community_sizes[i]
                if n > 1:
                    actual_matrix[i, j] = actual_matrix[i, j] / (n * (n - 1))
            else:
                n1, n2 = community_sizes[i], community_sizes[j]
                if n1 > 0 and n2 > 0:
                    actual_matrix[i, j] = actual_matrix[i, j] / (n1 * n2)
    deviation_matrix = np.abs(actual_matrix - P_sub)
    mean_deviation = np.mean(deviation_matrix)
    max_deviation = np.max(deviation_matrix)
    return {"mean_deviation": mean_deviation, "max_deviation": max_deviation}

def generate_test_graph(dist_type, config_params, target_stats):
    try:
        K = target_stats.get("K", 20)
        n_nodes = target_stats.get("n_nodes", 100)
        n_communities = target_stats.get("n_communities", 5)
        edge_density = target_stats.get("edge_density", 0.1)
        homophily = target_stats.get("homophily", 0.8)
        communities = list(range(n_communities))
        community_labels = np.random.choice(n_communities, size=n_nodes)
        P_sub = np.zeros((n_communities, n_communities))
        np.fill_diagonal(P_sub, homophily * edge_density * n_communities)
        P_sub[~np.eye(n_communities, dtype=bool)] = (1 - homophily) * edge_density * n_communities / (n_communities - 1)
        if dist_type == "power_law":
            degree_factors = generate_power_law_factors(
                n_nodes, 
                config_params["power_law_exponent"],
                config_params["target_avg_degree"]
            )
        elif dist_type == "exponential":
            degree_factors = generate_exponential_factors(
                n_nodes, 
                config_params["rate"],
                config_params["target_avg_degree"]
            )
        elif dist_type == "uniform":
            degree_factors = generate_uniform_factors(
                n_nodes, 
                config_params["min_factor"],
                config_params["max_factor"],
                config_params["target_avg_degree"]
            )
        else:
            degree_factors = np.ones(n_nodes)
        adjacency = generate_edges_with_config(
            community_labels, P_sub, degree_factors
        )
        if adjacency is None:
            return None
        import networkx as nx
        temp_graph = nx.from_scipy_sparse_array(adjacency)
        deviation = calculate_community_structure_deviation(
            temp_graph, community_labels, P_sub
        )
        return deviation
    except Exception as e:
        print(f"Error in test graph generation: {str(e)}")
        return None

def convert_params_to_config(dist_type, params):
    if dist_type == "power_law":
        return {"power_law_exponent": params[0], "target_avg_degree": params[1]}
    elif dist_type == "exponential":
        return {"rate": params[0], "target_avg_degree": params[1]}
    elif dist_type == "uniform":
        return {"min_factor": params[0], "max_factor": params[1], "target_avg_degree": params[2]}
    else:
        return {}

def create_family_params(target_stats, dist_type, config_params):
    method_distribution = {dist_type: 1.0}
    # Set defaults for all required parameters
    defaults = {
        # Universe/family-level
        "block_structure": "assortative",
        "overlap_structure": "modular",
        "edge_density": 0.1,
        "homophily": 0.8,
        "randomness_factor": 0.0,
        "intra_community_regime_similarity": 0.8,
        "inter_community_regime_similarity": 0.2,
        "regimes_per_community": 2,
        "homophily_range": 0.0,
        "density_range": 0.0,
        "method_distribution": method_distribution,
        "standard_method_params": {"degree_heterogeneity": 0.5, "edge_noise": 0.0},
        "config_model_params": {dist_type: config_params},
        "max_mean_community_deviation": 0.1,
        "max_max_community_deviation": 0.15,
        "max_parameter_search_attempts": 10,
        "parameter_search_range": 0.9,
        "min_edge_density": 0.005,
        "max_retries": 3,
        "triangle_enhancement": 0.0,
        "seed": np.random.randint(0, 100000),
        # Instance-level
        "min_communities": 3,
        "max_communities": 8,
        "min_nodes": 40,
        "max_nodes": 200,
        "degree_heterogeneity": 0.5,
        "edge_noise": 0.0,
        "sampling_method": "random",
        "min_component_size": 2,
        "feature_regime_balance": 0.5,
    }
    # Start with defaults, then update with target_stats, then config_params, then any explicit overrides
    family_params = defaults.copy()
    family_params.update(target_stats)
    family_params.update({
        "method_distribution": method_distribution,
        "config_model_params": {dist_type: config_params},
    })
    # Ensure min/max communities and nodes are valid
    K = family_params["K"]
    n_nodes = family_params["n_nodes"]
    n_communities = family_params["n_communities"]
    family_params["min_communities"] = max(3, min(n_communities, K//5))
    family_params["max_communities"] = max(family_params["min_communities"]+1, min(8, K//2))
    family_params["min_nodes"] = max(10, min(n_nodes, 100))
    family_params["max_nodes"] = max(family_params["min_nodes"]+1, 200)
    # Pass through degree_heterogeneity and edge_noise if present in config
    if "degree_heterogeneity" in config_params:
        family_params["degree_heterogeneity"] = config_params["degree_heterogeneity"]
    if "edge_noise" in config_params:
        family_params["edge_noise"] = config_params["edge_noise"]
    return family_params

def measure_family_deviation(family_data):
    graphs = family_data['graphs']
    deviations = []
    for graph in graphs:
        analysis = graph.analyze_community_connections()
        deviations.append(analysis["mean_deviation"])
    return np.mean(deviations) if deviations else 1.0 