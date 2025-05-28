"""
Metapath-based node classification for graph learning tasks.

This module implements sophisticated metapath discovery and node labeling
based on theoretical edge probabilities and degree distributions from DCCC-SBM.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
import time


class MetapathAnalyzer:
    """
    Discovers significant metapaths and creates node classification tasks.
    
    This class finds the most probable k-length metapaths using theoretical
    edge probabilities and degree distributions, then labels nodes based on
    their participation in these metapaths.
    """
    
    def __init__(
        self,
        P_sub: np.ndarray,
        community_labels: np.ndarray,
        degree_factors: Optional[np.ndarray] = None,
        min_prob_threshold: float = 0.01,
        degree_weight: float = 0.3,
        allow_immediate_return: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the metapath analyzer.
        
        Args:
            P_sub: Community-community edge probability matrix
            community_labels: Node community assignments
            degree_factors: Optional degree factors for DCCC-SBM weighting
            min_prob_threshold: Minimum probability threshold for valid transitions
            degree_weight: Weight for degree factor influence (0.0 = ignore degrees, 1.0 = only degrees)
            allow_immediate_return: Whether to allow returning to start community before k steps
            verbose: Whether to print detailed information
        """
        self.P_sub = P_sub
        self.community_labels = community_labels
        self.degree_factors = degree_factors
        self.min_prob_threshold = min_prob_threshold
        self.degree_weight = degree_weight
        self.allow_immediate_return = allow_immediate_return
        self.verbose = verbose
        
        self.n_communities = P_sub.shape[0]
        self.n_nodes = len(community_labels)
        
        # Validate inputs
        self._validate_inputs()
        
        # Compute community-level degree statistics for DCCC-SBM integration
        self.community_degree_stats = self._compute_community_degree_stats()
        
        # Storage for discovered metapaths
        self.discovered_metapaths = {}
        self.node_metapath_participation = {}
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.P_sub.shape[0] != self.P_sub.shape[1]:
            raise ValueError("P_sub must be square")
        
        if len(set(self.community_labels)) > self.n_communities:
            raise ValueError("More unique community labels than communities in P_sub")
        
        if self.degree_factors is not None and len(self.degree_factors) != self.n_nodes:
            raise ValueError("degree_factors length must match number of nodes")
        
        if not 0 <= self.degree_weight <= 1:
            raise ValueError("degree_weight must be between 0 and 1")
    
    def _compute_community_degree_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Compute degree statistics for each community.
        
        Returns:
            Dictionary mapping community to degree statistics
        """
        stats = {}
        
        for comm in range(self.n_communities):
            # Get nodes in this community
            comm_nodes = np.where(self.community_labels == comm)[0]
            
            if len(comm_nodes) == 0:
                stats[comm] = {'mean_degree_factor': 1.0, 'attraction_weight': 1.0}
                continue
            
            if self.degree_factors is not None:
                # Use actual degree factors
                comm_degree_factors = self.degree_factors[comm_nodes]
                mean_degree = np.mean(comm_degree_factors)
                attraction_weight = mean_degree  # Higher degree = more attractive
            else:
                # Default case without degree factors
                mean_degree = 1.0
                attraction_weight = 1.0
            
            stats[comm] = {
                'mean_degree_factor': mean_degree,
                'attraction_weight': attraction_weight,
                'node_count': len(comm_nodes)
            }
        
        return stats
   
    def _is_valid_loop(self, metapath: List[int]) -> bool:
        """
        Check if a metapath forms a valid loop.
        A valid loop requires:
        1. First and last communities are the same
        2. At least 4 nodes (3 edges) in the path
        """
        if len(metapath) < 4:  # Need at least 4 communities for 3 edges
            return False
        return metapath[0] == metapath[-1]

    def create_node_labels_from_metapath(
        self,
        graph: nx.Graph,
        metapath: List[int],
        label_name: str = "metapath_participation"
    ) -> np.ndarray:
        """
        Create node labels with 3-class system for loops.
        Updated to distinguish actual node loops vs community loops.
        """
        if len(metapath) < 2:
            raise ValueError("Metapath must have at least 2 communities")
        
        # Check if this could form loops (community-level check)
        is_potential_loop = self._is_valid_loop(metapath)
        
        if is_potential_loop:
            # For potential loops, create 3-class labels
            labels = np.zeros(self.n_nodes, dtype=int)
            non_loop_participating = set()
            loop_participating = set()
        else:
            # For non-loops, create binary labels
            labels = np.zeros(self.n_nodes, dtype=int)
            participating_nodes = set()
        
        # Find all instances of the metapath in the actual graph
        metapath_instances = self._find_metapath_instances_in_graph(graph, metapath)
        
        # Separate actual NODE loops from non-loops
        if is_potential_loop:
            loop_instances = []
            non_loop_instances = []
            
            for instance in metapath_instances:
                # Check for actual NODE loop (same node ID at start and end)
                if len(instance) >= 4 and instance[0] == instance[-1]:  # Same NODE ID
                    loop_instances.append(instance)
                    loop_participating.update(instance)
                else:
                    non_loop_instances.append(instance)
                    non_loop_participating.update(instance)
            
            # Set labels: 1 for non-loop, 2 for actual node loop participation
            for node in non_loop_participating:
                if node not in loop_participating:  # Only non-loop
                    labels[node] = 1
            for node in loop_participating:
                labels[node] = 2  # Actual node loop takes precedence
                
            participating_nodes = non_loop_participating | loop_participating
            
        else:
            # Mark all participating nodes for non-loops
            for instance in metapath_instances:
                participating_nodes.update(instance)
            
            # Set binary labels
            for node in participating_nodes:
                labels[node] = 1
        
        # Rest of the method remains the same...
        community_participation = self._analyze_community_participation(metapath, participating_nodes)
        max_participation_rate = max(stats['participation_rate'] for stats in community_participation.values()) if community_participation else 0.0
        
        task_info = {
            'metapath': metapath,
            'participating_nodes': participating_nodes,
            'n_participating': len(participating_nodes),
            'n_instances': len(metapath_instances),
            'participation_rate': len(participating_nodes) / self.n_nodes,
            'community_participation': community_participation,
            'is_loop_task': is_potential_loop,
            'max_community_participation': max_participation_rate
        }
        
        if is_potential_loop:
            task_info.update({
                'loop_instances': len([inst for inst in metapath_instances if len(inst) >= 4 and inst[0] == inst[-1]]),
                'non_loop_instances': len([inst for inst in metapath_instances if not (len(inst) >= 4 and inst[0] == inst[-1])]),
                'loop_participating_nodes': len(loop_participating),
                'non_loop_participating_nodes': len(non_loop_participating)
            })
        
        self.node_metapath_participation[label_name] = task_info
        
        if self.verbose:
            print(f"Metapath participation statistics:")
            print(f"  Metapath: {' -> '.join(map(str, metapath))}")
            print(f"  Participating nodes: {len(participating_nodes)}/{self.n_nodes} "
                f"({100 * len(participating_nodes) / self.n_nodes:.1f}%)")
            print(f"  Metapath instances found: {len(metapath_instances)}")
            if is_potential_loop:
                print(f"  Loop instances (4+ nodes): {task_info['loop_instances']}")
                print(f"  Non-loop instances: {task_info['non_loop_instances']}")
            print(f"  Community participation rates:")
            for comm, stats in community_participation.items():
                print(f"    Community {comm}: {stats['participating']}/{stats['total']} "
                    f"({100 * stats['participation_rate']:.1f}%)")
            
            if max_participation_rate >= 1.0:
                print(f"  WARNING: Community has 100% participation - task may be too easy")
        
        return labels

    def _analyze_community_participation(
        self,
        metapath: List[int],
        participating_nodes: set
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze what percentage of each community's nodes participate in the metapath.
        
        Args:
            metapath: List of communities in the metapath
            participating_nodes: Set of nodes that participate in metapath instances
            
        Returns:
            Dictionary mapping community to participation statistics
        """
        community_participation = {}
        
        # Get unique communities in metapath
        metapath_communities = set(metapath)
        
        for comm in metapath_communities:
            # Find all nodes in this community
            community_nodes = set(np.where(self.community_labels == comm)[0])
            
            # Find participating nodes in this community
            participating_in_comm = community_nodes & participating_nodes
            
            # Calculate statistics
            total_nodes = len(community_nodes)
            participating_count = len(participating_in_comm)
            participation_rate = participating_count / total_nodes if total_nodes > 0 else 0.0
            
            community_participation[comm] = {
                'total': total_nodes,
                'participating': participating_count,
                'non_participating': total_nodes - participating_count,
                'participation_rate': participation_rate,
                'participating_nodes': participating_in_comm,
                'non_participating_nodes': community_nodes - participating_in_comm
            }
        
        return community_participation
    
    def _find_metapath_instances_in_graph(
        self,
        graph: nx.Graph,
        metapath: List[int]
    ) -> List[List[int]]:
        """
        Find all instances of a metapath in the actual graph.
        
        Args:
            graph: NetworkX graph
            metapath: List of communities forming the metapath
            
        Returns:
            List of node sequences that form instances of the metapath
        """
        instances = []
        k = len(metapath)
        
        # For each starting node in the first community of the metapath
        start_community = metapath[0]
        start_nodes = np.where(self.community_labels == start_community)[0]
        
        for start_node in start_nodes:
            # Use DFS to find all valid paths of length k starting from this node
            valid_paths = self._dfs_metapath_search(
                graph, start_node, metapath, 0, [start_node], set([start_node])
            )
            instances.extend(valid_paths)
        
        return instances
    
    def _dfs_metapath_search(
        self,
        graph: nx.Graph,
        current_node: int,
        metapath: List[int],
        metapath_index: int,
        current_path: List[int],
        visited: set
    ) -> List[List[int]]:
        """
        DFS search for metapath instances.
        Updated to handle actual node loops (same node ID) vs community loops.
        """
        # Base case: reached the end of metapath
        if metapath_index == len(metapath) - 1:
            return [current_path.copy()]
        
        valid_paths = []
        next_target_community = metapath[metapath_index + 1]
        is_final_step = (metapath_index == len(metapath) - 2)
        is_potential_loop = self._is_valid_loop(metapath)  # Community-level loop check
        
        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            # Check if neighbor is in the correct community
            if self.community_labels[neighbor] == next_target_community:
                
                # For potential loops: check if this creates an actual NODE loop (same node ID)
                if is_potential_loop and is_final_step and len(metapath) >= 4:
                    # Final step of potential loop: allow connection back to start NODE
                    if neighbor == current_path[0]:  # Same node ID, not just same community
                        valid_paths.append(current_path + [neighbor])
                elif neighbor not in visited:
                    # Regular step: avoid revisiting nodes
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    
                    paths_from_neighbor = self._dfs_metapath_search(
                        graph, neighbor, metapath, metapath_index + 1,
                        current_path + [neighbor], new_visited
                    )
                    valid_paths.extend(paths_from_neighbor)
        
        return valid_paths

class UniverseMetapathSelector:
    """
    Selects metapath candidates based purely on GraphUniverse parameters
    (original P matrix and degree_centers) before any graph sampling.
    """
    
    def __init__(
        self,
        universe: 'GraphUniverse',
        degree_weight: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize selector with universe parameters.
        
        Args:
            universe: GraphUniverse object
            degree_weight: Weight for degree center influence
            verbose: Whether to print details
        """
        self.universe = universe
        self.P = universe.P  # Original unscaled probability matrix
        self.degree_centers = universe.degree_centers  # Original degree centers
        self.K = universe.K
        self.degree_weight = degree_weight
        self.verbose = verbose
        
        if verbose:
            print(f"Universe metapath selector initialized:")
            print(f"  Communities: {self.K}")
            print(f"  P matrix shape: {self.P.shape}")
            print(f"  Degree centers: {self.degree_centers}")
    
    def calculate_theoretical_probability(self, metapath: List[int]) -> float:
        """
        Calculate theoretical probability using universe P and degree_centers.
        Uses geometric mean of transition probabilities to avoid extremely small values.
        
        Args:
            metapath: List of universe community IDs
            
        Returns:
            Theoretical probability
        """
        probs = []
        
        for i in range(len(metapath) - 1):
            from_comm = metapath[i]
            to_comm = metapath[i + 1]
            
            # Base probability from universe P matrix
            base_prob = self.P[from_comm, to_comm]
            
            # Degree factor influence (higher degree center = more attractive)
            degree_multiplier = 1.0
            if self.degree_weight > 0:
                # Use actual degree center value
                degree_center = self.degree_centers[to_comm]
                # Map from [-1, 1] to [0.5, 1.5] for multiplier
                attraction_weight = 1.0 + 0.5 * degree_center
                degree_multiplier = (1 - self.degree_weight) + self.degree_weight * attraction_weight
            
            final_prob = base_prob * degree_multiplier
            probs.append(final_prob)
            
            if self.verbose and hasattr(self, '_debug_mode'):
                print(f"    {from_comm} -> {to_comm}: base={base_prob:.4f}, "
                      f"deg_mult={degree_multiplier:.4f}, final={final_prob:.4f}")
        
        # Use geometric mean instead of product
        if not probs:
            return 0.0
        return np.exp(np.mean(np.log(probs)))
    
    def generate_diverse_metapath_candidates(
        self,
        k_values: List[int] = [3, 4, 5],
        require_loop: bool = False,
        n_candidates_per_k: int = 50,
        diversity_factor: float = 0.4,
        min_prob_threshold: float = 0.0001
    ) -> Dict[int, List[Tuple[List[int], float]]]:
        """
        Generate diverse metapath candidates based on universe parameters only.
        
        Args:
            k_values: List of metapath lengths
            require_loop: Whether metapaths must return to start
            n_candidates_per_k: Number of candidates to generate per k
            diversity_factor: Factor to encourage diversity over high probability
            min_prob_threshold: Minimum acceptable probability
            
        Returns:
            Dict mapping k -> list of (metapath, probability) tuples
        """
        all_candidates = {}
        
        for k in k_values:
            if self.verbose:
                print(f"\nGenerating {k}-length metapath candidates from universe...")
            
            candidates = []
            
            # Try from all possible starting communities
            for start_comm in range(self.K):
                comm_candidates = self._generate_candidates_from_start(
                    start_comm, k, require_loop, n_candidates_per_k // self.K,
                    diversity_factor, min_prob_threshold
                )
                candidates.extend(comm_candidates)
            
            # Remove duplicates and sort by probability
            unique_candidates = {}
            for metapath, prob in candidates:
                key = tuple(metapath)
                if key not in unique_candidates or unique_candidates[key] < prob:
                    unique_candidates[key] = prob
            
            final_candidates = [(list(path), prob) for path, prob in unique_candidates.items()]
            final_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Filter for diversity and interest
            filtered_candidates = self._filter_for_diversity(final_candidates, k)
            
            all_candidates[k] = filtered_candidates
            
            if self.verbose:
                print(f"  Generated {len(candidates)} raw candidates")
                print(f"  After filtering: {len(filtered_candidates)} diverse candidates")
                if filtered_candidates:
                    print(f"  Top candidate: {' -> '.join(map(str, filtered_candidates[0][0]))} "
                          f"(prob: {filtered_candidates[0][1]:.6f})")
        
        return all_candidates
    
    def _generate_candidates_from_start(
        self,
        start_comm: int,
        k: int,
        require_loop: bool,
        n_candidates: int,
        diversity_factor: float,
        min_prob_threshold: float
    ) -> List[Tuple[List[int], float]]:
        """Generate candidates starting from a specific community."""
        candidates = []
        
        for _ in range(n_candidates):
            metapath = self._generate_single_diverse_metapath(
                start_comm, k, require_loop, diversity_factor
            )
            
            if metapath is not None:
                prob = self.calculate_theoretical_probability(metapath)
                if prob >= min_prob_threshold:
                    candidates.append((metapath, prob))
        
        return candidates
    
    def _generate_single_diverse_metapath(
        self,
        start_comm: int,
        k: int,
        require_loop: bool,
        diversity_factor: float
    ) -> Optional[List[int]]:
        """Generate a single diverse metapath using stochastic selection."""
        metapath = [start_comm]
        current_comm = start_comm
        
        for step in range(k - 1):
            is_final_step = (step == k - 2)
            
            # Get all possible next communities
            next_options = []
            for next_comm in range(self.K):
                # Check constraints
                if require_loop and is_final_step and next_comm != start_comm:
                    continue
                if not require_loop and is_final_step and next_comm == start_comm:
                    continue
                
                # Calculate probability with degree influence
                base_prob = self.P[current_comm, next_comm]
                
                degree_multiplier = 1.0
                if self.degree_weight > 0:
                    degree_center = self.degree_centers[next_comm]
                    attraction_weight = 1.0 + 0.5 * degree_center
                    degree_multiplier = (1 - self.degree_weight) + self.degree_weight * attraction_weight
                
                final_prob = base_prob * degree_multiplier
                
                if final_prob > 0:
                    next_options.append((next_comm, final_prob))
            
            if not next_options:
                return None
            
            # Apply diversity factor to encourage exploration
            adjusted_options = []
            for next_comm, prob in next_options:
                # Penalize very high probabilities to encourage diversity
                adjusted_prob = prob * (1 - diversity_factor * prob)
                adjusted_options.append((next_comm, adjusted_prob))
            
            # Stochastic selection
            communities, probs = zip(*adjusted_options)
            total_prob = sum(probs)
            
            if total_prob == 0:
                return None
            
            probs = [p / total_prob for p in probs]
            next_comm = np.random.choice(communities, p=probs)
            
            metapath.append(next_comm)
            current_comm = next_comm
        
        return metapath
    
    def _filter_for_diversity(
        self,
        candidates: List[Tuple[List[int], float]],
        k: int,
        max_candidates: int = 20
    ) -> List[Tuple[List[int], float]]:
        """Filter candidates to ensure diversity and randomly select one valid metapath."""
        if not candidates:
            return []
            
        # First ensure we have some minimum diversity by removing very similar paths
        diverse_candidates = []
        remaining = candidates.copy()
        
        while remaining:
            current = remaining.pop(0)
            diverse_candidates.append(current)
            
            # Remove paths that are too similar to current
            remaining = [
                (path, prob) for path, prob in remaining
                if self._calculate_diversity_score(path, [current]) > 0.3
            ]
        
        # Randomly select one of the diverse candidates
        if diverse_candidates:
            selected_idx = np.random.randint(0, len(diverse_candidates))
            return [diverse_candidates[selected_idx]]
        
        return []
    
    def _calculate_diversity_score(
        self,
        candidate_path: List[int],
        selected_paths: List[Tuple[List[int], float]]
    ) -> float:
        """Calculate how diverse a candidate is from already selected paths."""
        if not selected_paths:
            return 1.0
        
        min_similarity = float('inf')
        
        for selected_path, _ in selected_paths:
            # Calculate path similarity (Jaccard index of communities)
            set1 = set(candidate_path)
            set2 = set(selected_path)
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            similarity = intersection / union if union > 0 else 0
            min_similarity = min(min_similarity, similarity)
        
        # Return diversity (1 - similarity)
        return 1.0 - min_similarity
    
    def analyze_candidate_properties(
        self,
        candidates: Dict[int, List[Tuple[List[int], float]]]
    ) -> Dict[str, Any]:
        """Analyze properties of generated candidates."""
        analysis = {
            'total_candidates': sum(len(cands) for cands in candidates.values()),
            'candidates_per_k': {k: len(cands) for k, cands in candidates.items()},
            'probability_ranges': {},
            'community_usage': {},
            'loop_analysis': {}
        }
        
        # Analyze probability ranges
        for k, cands in candidates.items():
            if cands:
                probs = [prob for _, prob in cands]
                analysis['probability_ranges'][k] = {
                    'min': min(probs),
                    'max': max(probs),
                    'mean': np.mean(probs),
                    'std': np.std(probs)
                }
        
        # Analyze community usage
        community_counts = {i: 0 for i in range(self.K)}
        for k, cands in candidates.items():
            for metapath, _ in cands:
                for comm in set(metapath):
                    community_counts[comm] += 1
        
        analysis['community_usage'] = community_counts
        
        # Analyze loop patterns
        for k, cands in candidates.items():
            loop_count = sum(1 for metapath, _ in cands if metapath[0] == metapath[-1])
            analysis['loop_analysis'][k] = {
                'total_candidates': len(cands),
                'loop_candidates': loop_count,
                'loop_ratio': loop_count / len(cands) if cands else 0
            }
        
        return analysis
    
class FamilyMetapathEvaluator:
    """
    Evaluates universe-generated metapath candidates across graph families
    and reports community participation rates by split.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def evaluate_candidates_on_family(
        self,
        candidates: Dict[int, List[Tuple[List[int], float]]],
        family_graphs: List['GraphSample'],
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        max_community_participation: float = 0.95
    ) -> Dict[str, Any]:
        """
        Evaluate metapath candidates across graph family splits.
        Selects one random valid metapath for labeling.
        
        Args:
            candidates: Universe-generated candidates {k: [(metapath, prob), ...]}
            family_graphs: List of GraphSample objects
            train_indices: Indices for training split
            val_indices: Indices for validation split  
            test_indices: Indices for test split
            max_community_participation: Max allowed participation rate per community
            
        Returns:
            Dictionary with evaluation results and one valid metapath
        """
        if self.verbose:
            print(f"\nEvaluating metapath candidates on graph family...")
            print(f"  Total graphs: {len(family_graphs)}")
            print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        evaluation_results = {
            'valid_metapaths': {},
            'rejected_metapaths': {},
            'participation_reports': {},
            'split_analysis': {}
        }
        
        splits = {
            'train': train_indices,
            'val': val_indices, 
            'test': test_indices
        }
        
        # Collect all valid metapaths first
        valid_candidates = []
        
        for k, k_candidates in candidates.items():
            if self.verbose:
                print(f"\nEvaluating {k}-length metapath candidates")
            
            for candidate_idx, (metapath, universe_prob) in enumerate(k_candidates):
                metapath_str = ' -> '.join(map(str, metapath))
                
                # Evaluate this metapath across all graphs and splits
                metapath_evaluation = self._evaluate_single_metapath(
                    metapath, family_graphs, splits, max_community_participation
                )
                
                # Check if metapath is valid (no 100% participation in any split)
                is_valid = metapath_evaluation['is_valid']
                
                task_name = f"metapath_k{k}_{'loop' if metapath[0] == metapath[-1] else 'path'}_{candidate_idx}"
                
                if is_valid:
                    valid_candidates.append((task_name, metapath, universe_prob, metapath_evaluation))
                else:
                    evaluation_results['rejected_metapaths'][task_name] = {
                        'metapath': metapath,
                        'universe_probability': universe_prob,
                        'k': k,
                        'evaluation': metapath_evaluation,
                        'rejection_reason': metapath_evaluation['rejection_reason']
                    }
                
                # Store detailed participation report
                evaluation_results['participation_reports'][task_name] = metapath_evaluation['detailed_report']
        
        # Randomly select one valid metapath
        if valid_candidates:
            selected_idx = np.random.randint(0, len(valid_candidates))
            task_name, metapath, universe_prob, metapath_evaluation = valid_candidates[selected_idx]
            
            evaluation_results['valid_metapaths'][task_name] = {
                'metapath': metapath,
                'universe_probability': universe_prob,
                'k': len(metapath),
                'evaluation': metapath_evaluation
            }
            
            if self.verbose:
                print(f"\nSelected valid metapath: {task_name}")
                print(f"  Metapath: {' -> '.join(map(str, metapath))}")
                print(f"  Max participation: {metapath_evaluation['max_participation_across_splits']:.3f}")
        
        # Generate summary analysis
        evaluation_results['split_analysis'] = self._analyze_split_performance(
            evaluation_results['valid_metapaths'], splits
        )
        
        if self.verbose:
            self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_single_metapath(
        self,
        metapath: List[int],
        family_graphs: List['GraphSample'],
        splits: Dict[str, List[int]],
        max_participation: float
    ) -> Dict[str, Any]:
        """Evaluate a single metapath across all graphs and splits."""
        
        # Map metapath from universe IDs to graph-local IDs for each graph
        participation_by_split = {}
        detailed_participation = {}
        
        for split_name, graph_indices in splits.items():
            participation_by_split[split_name] = {}
            detailed_participation[split_name] = []
            
            for graph_idx in graph_indices:
                graph_sample = family_graphs[graph_idx]
                
                # Map universe metapath to this graph's community indices
                try:
                    local_metapath = self._map_metapath_to_graph(metapath, graph_sample)
                    if local_metapath is None:
                        continue  # Graph doesn't contain all required communities
                    
                    # Find metapath instances in this graph
                    participation_data = self._analyze_metapath_participation(
                        local_metapath, graph_sample
                    )
                    
                    detailed_participation[split_name].append({
                        'graph_idx': graph_idx,
                        'local_metapath': local_metapath,
                        'participation_data': participation_data
                    })
                    
                    # Accumulate community participation across graphs in this split
                    for comm_universe_id, comm_data in participation_data['community_participation'].items():
                        if comm_universe_id not in participation_by_split[split_name]:
                            participation_by_split[split_name][comm_universe_id] = {
                                'total_nodes': 0,
                                'participating_nodes': 0,
                                'graphs_with_community': 0
                            }
                        
                        participation_by_split[split_name][comm_universe_id]['total_nodes'] += comm_data['total_nodes']
                        participation_by_split[split_name][comm_universe_id]['participating_nodes'] += comm_data['participating_nodes']
                        participation_by_split[split_name][comm_universe_id]['graphs_with_community'] += 1
                
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Error processing graph {graph_idx}: {e}")
                    continue
        
        # Calculate participation rates and check validity
        max_participation_overall = 0.0
        has_valid_community = False  # Track if any community is below threshold
        
        for split_name, split_participation in participation_by_split.items():
            for comm_id, comm_data in split_participation.items():
                if comm_data['total_nodes'] > 0:
                    participation_rate = comm_data['participating_nodes'] / comm_data['total_nodes']
                    comm_data['participation_rate'] = participation_rate
                    
                    max_participation_overall = max(max_participation_overall, participation_rate)
                    
                    # Check if this community has acceptable participation
                    if participation_rate < max_participation:
                        has_valid_community = True
        
        # Accept if at least one community has acceptable participation
        is_valid = has_valid_community
        
        return {
            'is_valid': is_valid,
            'participation_by_split': participation_by_split,
            'detailed_participation': detailed_participation,
            'max_participation_across_splits': max_participation_overall,
            'rejection_reason': None if is_valid else "No community has participation below threshold",
            'detailed_report': self._create_participation_report(metapath, participation_by_split)
        }
    
    def _map_metapath_to_graph(
        self,
        universe_metapath: List[int],
        graph_sample: 'GraphSample'
    ) -> Optional[List[int]]:
        """Map universe community IDs to graph-local community indices."""
        local_metapath = []
        
        for universe_comm_id in universe_metapath:
            # Find local index of this universe community in the graph
            try:
                local_idx = graph_sample.communities.index(universe_comm_id)
                local_metapath.append(local_idx)
            except ValueError:
                # This graph doesn't contain this community
                return None
        
        return local_metapath
    
    def _analyze_metapath_participation(
        self,
        local_metapath: List[int],
        graph_sample: 'GraphSample'
    ) -> Dict[str, Any]:
        """Analyze participation for a metapath in a specific graph."""
        # Create analyzer for this graph
        analyzer = MetapathAnalyzer(
            P_sub=graph_sample.P_sub,
            community_labels=graph_sample.community_labels,
            degree_factors=graph_sample.degree_factors,
            verbose=False
        )
        
        # Generate labels
        labels = analyzer.create_node_labels_from_metapath(
            graph_sample.graph, local_metapath, "temp_task"
        )
        
        # Analyze participation by community (convert back to universe IDs)
        community_participation = {}
        unique_local_communities = np.unique(graph_sample.community_labels)
        
        for local_comm_idx in unique_local_communities:
            universe_comm_id = graph_sample.communities[local_comm_idx]
            
            # Get nodes in this community
            community_nodes = np.where(graph_sample.community_labels == local_comm_idx)[0]
            participating_nodes = np.where(labels[community_nodes] > 0)[0]
            
            community_participation[universe_comm_id] = {
                'total_nodes': len(community_nodes),
                'participating_nodes': len(participating_nodes),
                'participation_rate': len(participating_nodes) / len(community_nodes) if len(community_nodes) > 0 else 0.0
            }
        
        return {
            'labels': labels,
            'total_participating': np.sum(labels > 0),
            'total_nodes': len(labels),
            'overall_participation_rate': np.mean(labels > 0),
            'community_participation': community_participation
        }
    
    def _create_participation_report(
        self,
        metapath: List[int],
        participation_by_split: Dict[str, Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Create detailed participation report for a metapath."""
        report = {
            'metapath': metapath,
            'metapath_str': ' -> '.join(map(str, metapath)),
            'communities_in_metapath': list(set(metapath)),
            'split_reports': {}
        }
        
        for split_name, split_data in participation_by_split.items():
            split_report = {
                'communities': {},
                'summary': {
                    'total_communities': len(split_data),
                    'max_participation_rate': 0.0,
                    'min_participation_rate': 1.0,
                    'avg_participation_rate': 0.0
                }
            }
            
            rates = []
            for comm_id, comm_data in split_data.items():
                split_report['communities'][comm_id] = {
                    'total_nodes': comm_data['total_nodes'],
                    'participating_nodes': comm_data['participating_nodes'],
                    'participation_rate': comm_data.get('participation_rate', 0.0),
                    'graphs_with_community': comm_data['graphs_with_community']
                }
                
                rate = comm_data.get('participation_rate', 0.0)
                rates.append(rate)
                split_report['summary']['max_participation_rate'] = max(split_report['summary']['max_participation_rate'], rate)
                split_report['summary']['min_participation_rate'] = min(split_report['summary']['min_participation_rate'], rate)
            
            if rates:
                split_report['summary']['avg_participation_rate'] = np.mean(rates)
            
            report['split_reports'][split_name] = split_report
        
        return report
    
    def _analyze_split_performance(
        self,
        valid_metapaths: Dict[str, Any],
        splits: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Analyze overall performance across splits."""
        analysis = {
            'total_valid_metapaths': len(valid_metapaths),
            'metapaths_by_k': {},
            'participation_statistics': {}
        }
        
        # Group by k-value
        for task_name, metapath_data in valid_metapaths.items():
            k = metapath_data['k']
            if k not in analysis['metapaths_by_k']:
                analysis['metapaths_by_k'][k] = 0
            analysis['metapaths_by_k'][k] += 1
        
        # Participation statistics
        all_participation_rates = []
        for task_name, metapath_data in valid_metapaths.items():
            participation_report = metapath_data['evaluation']['detailed_report']
            
            for split_name, split_report in participation_report['split_reports'].items():
                for comm_id, comm_data in split_report['communities'].items():
                    all_participation_rates.append(comm_data['participation_rate'])
        
        if all_participation_rates:
            analysis['participation_statistics'] = {
                'mean': np.mean(all_participation_rates),
                'std': np.std(all_participation_rates),
                'min': np.min(all_participation_rates),
                'max': np.max(all_participation_rates),
                'median': np.median(all_participation_rates)
            }
        
        return analysis
    
    def _print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print detailed evaluation summary."""
        print(f"\n{'='*70}")
        print("METAPATH EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        valid_count = len(evaluation_results['valid_metapaths'])
        rejected_count = len(evaluation_results['rejected_metapaths'])
        
        print(f"Valid metapaths: {valid_count}")
        print(f"Rejected metapaths: {rejected_count}")
        
        if valid_count > 0:
            print(f"\nVALID METAPATHS:")
            for task_name, metapath_data in evaluation_results['valid_metapaths'].items():
                metapath_str = ' -> '.join(map(str, metapath_data['metapath']))
                max_participation = metapath_data['evaluation']['max_participation_across_splits']
                print(f"  {task_name}: {metapath_str} (max participation: {max_participation:.3f})")
        
        if rejected_count > 0:
            print(f"\nREJECTED METAPATHS:")
            for task_name, metapath_data in evaluation_results['rejected_metapaths'].items():
                metapath_str = ' -> '.join(map(str, metapath_data['metapath']))
                reason = metapath_data['rejection_reason']
                print(f"  {task_name}: {metapath_str}")
                print(f"    Reason: {reason}")
        
        # Print participation statistics
        split_analysis = evaluation_results['split_analysis']
        if 'participation_statistics' in split_analysis:
            stats = split_analysis['participation_statistics']
            print(f"\nPARTICIPATION STATISTICS:")
            print(f"  Mean participation rate: {stats['mean']:.3f}")
            print(f"  Std deviation: {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Median: {stats['median']:.3f}")
    
    def generate_detailed_report(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate a detailed text report of the evaluation."""
        report_lines = [
            "METAPATH FAMILY EVALUATION DETAILED REPORT",
            "=" * 50,
            "",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY:",
            f"  Valid metapaths: {len(evaluation_results['valid_metapaths'])}",
            f"  Rejected metapaths: {len(evaluation_results['rejected_metapaths'])}",
            ""
        ]
        
        # Detailed valid metapath reports
        if evaluation_results['valid_metapaths']:
            report_lines.append("VALID METAPATHS DETAILED ANALYSIS:")
            report_lines.append("-" * 40)
            
            for task_name, metapath_data in evaluation_results['valid_metapaths'].items():
                participation_report = evaluation_results['participation_reports'][task_name]
                
                report_lines.extend([
                    f"\nTask: {task_name}",
                    f"  Metapath: {participation_report['metapath_str']}",
                    f"  Universe probability: {metapath_data['universe_probability']:.6f}",
                    f"  Communities involved: {participation_report['communities_in_metapath']}",
                    ""
                ])
                
                # Split-by-split analysis
                for split_name, split_report in participation_report['split_reports'].items():
                    report_lines.extend([
                        f"  {split_name.upper()} SPLIT:",
                        f"    Communities analyzed: {split_report['summary']['total_communities']}",
                        f"    Max participation rate: {split_report['summary']['max_participation_rate']:.3f}",
                        f"    Min participation rate: {split_report['summary']['min_participation_rate']:.3f}",
                        f"    Avg participation rate: {split_report['summary']['avg_participation_rate']:.3f}",
                        ""
                    ])
                    
                    # Per-community details
                    for comm_id, comm_data in split_report['communities'].items():
                        report_lines.append(
                            f"    Community {comm_id}: {comm_data['participating_nodes']}/{comm_data['total_nodes']} "
                            f"({comm_data['participation_rate']:.3f}) across {comm_data['graphs_with_community']} graphs"
                        )
                    
                    report_lines.append("")
        
        # Rejected metapath analysis
        if evaluation_results['rejected_metapaths']:
            report_lines.extend([
                "REJECTED METAPATHS ANALYSIS:",
                "-" * 30,
                ""
            ])
            
            for task_name, metapath_data in evaluation_results['rejected_metapaths'].items():
                metapath_str = ' -> '.join(map(str, metapath_data['metapath']))
                report_lines.extend([
                    f"Task: {task_name}",
                    f"  Metapath: {metapath_str}",
                    f"  Universe probability: {metapath_data['universe_probability']:.6f}",
                    f"  Rejection reason: {metapath_data['rejection_reason']}",
                    ""
                ])
        
        # Overall statistics
        split_analysis = evaluation_results['split_analysis']
        report_lines.extend([
            "OVERALL STATISTICS:",
            "-" * 20,
            f"Valid metapaths by k-value: {split_analysis['metapaths_by_k']}",
            ""
        ])
        
        if 'participation_statistics' in split_analysis:
            stats = split_analysis['participation_statistics']
            report_lines.extend([
                "Participation rate statistics across all valid metapaths:",
                f"  Mean: {stats['mean']:.3f}",
                f"  Standard deviation: {stats['std']:.3f}",
                f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]",
                f"  Median: {stats['median']:.3f}",
                ""
            ])
        
        report_text = '\n'.join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Detailed report saved to: {save_path}")
        
        return report_text
