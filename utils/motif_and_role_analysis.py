"""
Motif and Role Analysis module for graph structural analysis.

This module provides functionality to:
1. Extract motif participation profiles for nodes
2. Perform structural role decomposition
3. Combine these approaches for meaningful node labeling
4. Analyze and visualize the results

The implementation follows a two-layer approach:
- First layer: Compute motif participation profiles for basic structures
- Second layer: Apply non-negative matrix factorization for role discovery
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from collections import defaultdict, Counter
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy import sparse
import itertools
from matplotlib.colors import ListedColormap
import random


class MotifRoleAnalyzer:
    """
    Analyzes motifs and structural roles in graph structures.
    
    This class implements the combined motif-role framework for structural node labeling,
    providing both interpretable features and principled abstraction.
    """
    
    def __init__(
        self,
        graph_sample,
        max_motif_size: int = 4,
        n_roles: int = 5,
        normalize_features: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the motif and role analyzer.
        
        Args:
            graph_sample: GraphSample object to analyze
            max_motif_size: Maximum size of motifs to extract (3-4 recommended)
            n_roles: Number of structural roles to discover
            normalize_features: Whether to normalize motif profiles
            verbose: Whether to print detailed output
        """
        self.graph_sample = graph_sample
        self.graph = graph_sample.graph
        self.communities = graph_sample.communities
        self.community_labels = graph_sample.community_labels  # Now using community_labels
        self.max_motif_size = max_motif_size
        self.n_roles = n_roles
        self.normalize_features = normalize_features
        self.verbose = verbose
        
        # Dictionary to store community for each node
        self.node_communities = self._get_node_communities()
        
        # Motif set and definitions
        self.motif_set = self._define_motif_set()
        
        # Structures to store results
        self.motif_profiles = None
        self.role_membership = None
        self.role_definitions = None
        self.node_structural_labels = None
        
    def _get_node_communities(self) -> Dict[int, int]:
        """
        Get the community for each node.
        
        Returns:
            Dictionary mapping node index to community ID
        """
        node_communities = {}
        for i in range(self.graph_sample.n_nodes):
            # Get community directly from labels
            community_idx = self.community_labels[i]
            community = self.communities[community_idx]
            node_communities[i] = community
        
        return node_communities
    
    def _define_motif_set(self) -> Dict:
        """
        Define the set of motifs to be analyzed.
        
        Returns:
            Dictionary of motif definitions with names and functions
        """
        motif_set = {}
        
        # 2-node motifs
        motif_set["edge"] = {
            "size": 2,
            "description": "Simple edge between two nodes",
            "count_func": self._count_edges
        }
        
        # 3-node motifs
        motif_set["triangle"] = {
            "size": 3,
            "description": "Complete triangle (3-clique)",
            "count_func": self._count_triangles
        }
        
        motif_set["2_star"] = {
            "size": 3,
            "description": "Center node connected to 2 nodes (2-star)",
            "count_func": self._count_2stars
        }
        
        # 4-node motifs (if requested)
        if self.max_motif_size >= 4:
            motif_set["4_clique"] = {
                "size": 4,
                "description": "Complete 4-clique",
                "count_func": self._count_4cliques
            }
            
            motif_set["4_cycle"] = {
                "size": 4,
                "description": "Cycle of 4 nodes",
                "count_func": self._count_4cycles
            }
            
            motif_set["3_star"] = {
                "size": 4,
                "description": "Center node connected to 3 nodes (3-star)",
                "count_func": self._count_3stars
            }
            
            motif_set["chordal_cycle"] = {
                "size": 4,
                "description": "4-cycle with one chord",
                "count_func": self._count_chordal_cycles
            }
        
        # Add structural metrics
        motif_set["degree"] = {
            "size": 1,
            "description": "Node degree (number of connections)",
            "count_func": self._get_node_degrees
        }
        
        motif_set["clustering"] = {
            "size": 1,
            "description": "Local clustering coefficient",
            "count_func": self._get_clustering_coefficients
        }
        
        if self.max_motif_size >= 4:
            motif_set["betweenness"] = {
                "size": 1,
                "description": "Betweenness centrality",
                "count_func": self._get_betweenness_centrality
            }
        
        return motif_set
    
    # Motif counting functions
    def _count_edges(self) -> np.ndarray:
        """Count edges for each node (same as degree)."""
        return np.array([d for _, d in self.graph.degree()])
    
    def _count_triangles(self) -> np.ndarray:
        """Count triangles for each node."""
        triangles = nx.triangles(self.graph)
        return np.array([triangles[n] for n in sorted(self.graph.nodes())])
    
    def _count_2stars(self) -> np.ndarray:
        """Count 2-stars centered at each node."""
        degrees = np.array([d for _, d in self.graph.degree()])
        triangles = np.array([t for _, t in nx.triangles(self.graph).items()])
        
        # Each node with degree k is the center of (k choose 2) 2-stars
        # We need to subtract triangles as they're counted differently
        two_stars = np.zeros(len(degrees))
        for i, (d, t) in enumerate(zip(degrees, triangles)):
            if d >= 2:
                two_stars[i] = (d * (d - 1)) // 2 - t
        
        return two_stars
    
    def _count_4cliques(self) -> np.ndarray:
        """Count 4-cliques for each node."""
        # This is computationally expensive, so we'll use an approximation
        # based on enumerating all 4-node combinations for smaller graphs
        if self.graph.number_of_nodes() > 500:
            if self.verbose:
                print("Graph too large for exact 4-clique counting, using approximation")
            return self._approximate_4cliques()
        
        # For smaller graphs, enumerate all possible 4-node combinations
        clique_counts = np.zeros(self.graph.number_of_nodes())
        
        for nodes in tqdm(itertools.combinations(self.graph.nodes(), 4),
                          disable=not self.verbose,
                          desc="Counting 4-cliques"):
            subgraph = self.graph.subgraph(nodes)
            if subgraph.number_of_edges() == 6:  # Complete graph K4 has 6 edges
                for node in nodes:
                    clique_counts[node] += 1
        
        return clique_counts
    
    def _approximate_4cliques(self) -> np.ndarray:
        """Approximate 4-clique counts for large graphs."""
        # Use an ego-network based approach
        clique_counts = np.zeros(self.graph.number_of_nodes())
        
        for node in tqdm(self.graph.nodes(), disable=not self.verbose, 
                         desc="Approximating 4-cliques"):
            # Get neighbors
            neighbors = list(self.graph.neighbors(node))
            
            # If node has less than 3 neighbors, it can't be in a 4-clique
            if len(neighbors) < 3:
                continue
            
            # Sample neighbors if there are too many
            if len(neighbors) > 50:
                neighbors = random.sample(neighbors, 50)
            
            # Check for 3-cliques among neighbors
            for combo in itertools.combinations(neighbors, 3):
                # Check if these 3 neighbors form a triangle
                if (self.graph.has_edge(combo[0], combo[1]) and
                    self.graph.has_edge(combo[1], combo[2]) and
                    self.graph.has_edge(combo[0], combo[2])):
                    clique_counts[node] += 1
        
        return clique_counts
    
    def _count_4cycles(self) -> np.ndarray:
        """Count 4-cycles for each node."""
        cycle_counts = np.zeros(self.graph.number_of_nodes())
        
        # This is a simplified approximation that counts potential 4-cycles
        # through 2-hop neighbors
        for node in tqdm(self.graph.nodes(), disable=not self.verbose,
                         desc="Counting 4-cycles"):
            # Get all neighbors
            neighbors = set(self.graph.neighbors(node))
            
            # For each neighbor, get its neighbors (excluding the original node)
            two_hop_neighbors = defaultdict(set)
            for nbr in neighbors:
                for nbr2 in self.graph.neighbors(nbr):
                    if nbr2 != node and nbr2 not in neighbors:
                        two_hop_neighbors[nbr].add(nbr2)
            
            # Count 4-cycles
            for nbr1, nbr2 in itertools.combinations(neighbors, 2):
                common_two_hops = two_hop_neighbors[nbr1].intersection(two_hop_neighbors[nbr2])
                cycle_counts[node] += len(common_two_hops)
        
        # Divide by 2 as each cycle is counted twice from different starting points
        return cycle_counts / 2
    
    def _count_3stars(self) -> np.ndarray:
        """Count 3-stars centered at each node."""
        degrees = np.array([d for _, d in self.graph.degree()])
        
        # A node with degree k is the center of (k choose 3) 3-stars
        three_stars = np.zeros(len(degrees))
        for i, d in enumerate(degrees):
            if d >= 3:
                three_stars[i] = (d * (d - 1) * (d - 2)) // 6
        
        return three_stars
    
    def _count_chordal_cycles(self) -> np.ndarray:
        """Count 4-cycles with a chord for each node."""
        # This is a more complex motif, we'll use an approximation
        chordal_counts = np.zeros(self.graph.number_of_nodes())
        
        for node in tqdm(self.graph.nodes(), disable=not self.verbose,
                         desc="Counting chordal cycles"):
            # Get neighbors
            neighbors = list(self.graph.neighbors(node))
            
            # Need at least 3 neighbors to be part of a chordal cycle
            if len(neighbors) < 3:
                continue
                
            # Check all triples of neighbors
            for n1, n2, n3 in itertools.combinations(neighbors, 3):
                edges = 0
                if self.graph.has_edge(n1, n2): edges += 1
                if self.graph.has_edge(n2, n3): edges += 1
                if self.graph.has_edge(n1, n3): edges += 1
                
                # If exactly 2 edges exist between the neighbors,
                # this forms a chordal cycle
                if edges == 2:
                    chordal_counts[node] += 1
        
        return chordal_counts
    
    # Structural metric functions
    def _get_node_degrees(self) -> np.ndarray:
        """Get the degree of each node."""
        return np.array([d for _, d in self.graph.degree()])
    
    def _get_clustering_coefficients(self) -> np.ndarray:
        """Get the local clustering coefficient of each node."""
        clustering = nx.clustering(self.graph)
        return np.array([clustering[n] for n in sorted(self.graph.nodes())])
    
    def _get_betweenness_centrality(self) -> np.ndarray:
        """
        Get the betweenness centrality of each node.
        
        For large graphs, use an approximation with k-shortest paths.
        """
        # For large graphs, use approximation
        if self.graph.number_of_nodes() > 500:
            betweenness = nx.betweenness_centrality(
                self.graph, 
                k=min(500, self.graph.number_of_nodes()),
                normalized=True
            )
        else:
            betweenness = nx.betweenness_centrality(
                self.graph,
                normalized=True
            )
        
        return np.array([betweenness[n] for n in sorted(self.graph.nodes())])
    
    def compute_motif_profiles(self) -> np.ndarray:
        """
        Compute motif participation profiles for all nodes.
        
        Returns:
            Array of shape (n_nodes, n_motifs) with motif counts
        """
        if self.verbose:
            print("Computing motif participation profiles...")
        
        # Initialize profiles matrix
        n_motifs = len(self.motif_set)
        n_nodes = self.graph.number_of_nodes()
        profiles = np.zeros((n_nodes, n_motifs))
        
        # Compute each motif count
        for i, (motif_name, motif_info) in enumerate(self.motif_set.items()):
            if self.verbose:
                print(f"  Computing {motif_name} counts...")
            
            # Call the count function
            counts = motif_info["count_func"]()
            profiles[:, i] = counts
        
        # Normalize if requested
        if self.normalize_features:
            # Avoid division by zero
            row_sums = profiles.sum(axis=1)
            row_sums[row_sums == 0] = 1
            profiles = profiles / row_sums[:, np.newaxis]
        
        # Store the profiles
        self.motif_profiles = profiles
        self.motif_names = list(self.motif_set.keys())
        
        return profiles
    
    def discover_structural_roles(self, n_roles: Optional[int] = None) -> np.ndarray:
        """
        Discover structural roles using non-negative matrix factorization on motif profiles.
        
        Args:
            n_roles: Number of roles to discover (uses self.n_roles if None)
            
        Returns:
            Array of shape (n_nodes, n_roles) with role memberships
        """
        if self.motif_profiles is None:
            self.compute_motif_profiles()
        
        if n_roles is None:
            n_roles = self.n_roles
        
        if self.verbose:
            print(f"Discovering {n_roles} structural roles...")
        
        # Apply NMF to discover roles
        nmf = NMF(
            n_components=n_roles,
            init='random',
            random_state=42,
            max_iter=500
        )
        
        # Fit the model
        role_membership = nmf.fit_transform(self.motif_profiles)
        role_definitions = nmf.components_
        
        # Normalize role memberships (rows sum to 1)
        row_sums = role_membership.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        role_membership = role_membership / row_sums[:, np.newaxis]
        
        # Store results
        self.role_membership = role_membership
        self.role_definitions = role_definitions
        
        return role_membership
    
    def get_primary_roles(self) -> np.ndarray:
        """
        Get the primary (dominant) role for each node.
        
        Returns:
            Array of primary role indices for each node
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        return np.argmax(self.role_membership, axis=1)
    
    def interpret_roles(self) -> Dict:
        """
        Interpret the discovered roles based on their motif components.
        
        Returns:
            Dictionary with role interpretations
        """
        if self.role_definitions is None:
            self.discover_structural_roles()
        
        role_interpretations = {}
        
        for role_idx in range(self.n_roles):
            # Get the motif weights for this role
            weights = self.role_definitions[role_idx]
            
            # Get the top 3 most important motifs
            top_motifs = [(self.motif_names[i], weights[i]) 
                           for i in np.argsort(-weights)[:3]
                           if weights[i] > 0.1]  # Only include significant weights
            
            # Create interpretation based on top motifs
            if top_motifs:
                description = f"Role {role_idx}: "
                description += " + ".join([f"{w:.2f}×{m}" for m, w in top_motifs])
            else:
                description = f"Role {role_idx}: No significant motifs"
            
            # Attempt role naming based on the dominant motifs
            if top_motifs:
                top_motif_name = top_motifs[0][0]
                if top_motif_name == "triangle" and weights[self.motif_names.index("clustering")] > 0.1:
                    name = "Community Core"
                elif top_motif_name == "betweenness" or top_motif_name == "4_cycle":
                    name = "Bridge/Broker"
                elif top_motif_name == "3_star" or top_motif_name == "2_star":
                    name = "Hub/Authority"
                elif top_motif_name == "degree":
                    if weights[self.motif_names.index("clustering")] > 0.1:
                        name = "Local Connector"
                    else:
                        name = "Peripheral Hub"
                elif top_motif_name == "4_clique":
                    name = "Clique Member"
                else:
                    name = f"Role {role_idx}"
            else:
                name = f"Role {role_idx}"
            
            role_interpretations[role_idx] = {
                "name": name,
                "description": description,
                "top_motifs": top_motifs
            }
        
        return role_interpretations
    
    def create_structural_labels(self, method: str = "primary_role") -> Dict:
        """
        Create structural labels for nodes based on role decomposition.
        
        Args:
            method: Method for creating labels:
                   "primary_role" - Use the primary role
                   "mixed_roles" - Use roles with membership > threshold
                   "role_community" - Combine role and community information
                   
        Returns:
            Dictionary mapping nodes to structural labels
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        labels = {}
        
        if method == "primary_role":
            # Assign the primary role as the label
            primary_roles = self.get_primary_roles()
            role_interpretations = self.interpret_roles()
            
            for node in range(self.graph.number_of_nodes()):
                role_idx = primary_roles[node]
                labels[node] = role_interpretations[role_idx]["name"]
                
        elif method == "mixed_roles":
            # Assign nodes to multiple roles if membership exceeds threshold
            threshold = 0.25
            role_interpretations = self.interpret_roles()
            
            for node in range(self.graph.number_of_nodes()):
                # Get roles with significant membership
                significant_roles = [idx for idx, val in enumerate(self.role_membership[node])
                                     if val >= threshold]
                
                if significant_roles:
                    # Create label from all significant roles
                    label = " + ".join([role_interpretations[r]["name"] for r in significant_roles])
                else:
                    # If no significant roles, use the primary role
                    primary_role = np.argmax(self.role_membership[node])
                    label = role_interpretations[primary_role]["name"]
                
                labels[node] = label
                
        elif method == "role_community":
            # Combine role and community information
            primary_roles = self.get_primary_roles()
            role_interpretations = self.interpret_roles()
            
            for node in range(self.graph.number_of_nodes()):
                role_idx = primary_roles[node]
                community = self.node_communities[node]
                
                labels[node] = f"{role_interpretations[role_idx]['name']} (C{community})"
        
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        self.node_structural_labels = labels
        return labels
    
    def create_conditional_labels(self, condition_rules: List[Dict]) -> Dict:
        """
        Create conditional structural labels based on custom rules.
        
        Args:
            condition_rules: List of rule dictionaries, each with:
                - condition: Function taking node, role membership, motif profile, returns bool
                - label: String or function that returns the label
                
        Returns:
            Dictionary mapping nodes to structural labels
        """
        if self.role_membership is None:
            self.discover_structural_roles()
            
        if self.motif_profiles is None:
            self.compute_motif_profiles()
        
        labels = {}
        
        for node in range(self.graph.number_of_nodes()):
            # Check each rule in order (first match wins)
            label_assigned = False
            
            for rule in condition_rules:
                condition = rule["condition"]
                label = rule["label"]
                
                # Check if node matches this condition
                if condition(node, self.role_membership[node], self.motif_profiles[node]):
                    # Apply the label (could be static or a function)
                    if callable(label):
                        labels[node] = label(node, self.role_membership[node], self.motif_profiles[node])
                    else:
                        labels[node] = label
                    
                    label_assigned = True
                    break
            
            # If no rule matched, use a default label
            if not label_assigned:
                primary_role = np.argmax(self.role_membership[node])
                role_interpretations = self.interpret_roles()
                labels[node] = role_interpretations[primary_role]["name"]
        
        self.node_structural_labels = labels
        return labels
    
    def visualize_motif_profiles(self, n_nodes: int = 20) -> plt.Figure:
        """
        Visualize motif profiles for a subset of nodes.
        
        Args:
            n_nodes: Number of nodes to visualize
            
        Returns:
            Matplotlib figure
        """
        if self.motif_profiles is None:
            self.compute_motif_profiles()
        
        # Sample nodes to visualize
        n_sample = min(n_nodes, self.graph.number_of_nodes())
        sampled_nodes = np.random.choice(self.graph.number_of_nodes(), n_sample, replace=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(
            self.motif_profiles[sampled_nodes],
            cmap="YlGnBu",
            ax=ax,
            yticklabels=[f"Node {n}" for n in sampled_nodes],
            xticklabels=self.motif_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        ax.set_title(f"Motif Participation Profiles for {n_sample} Random Nodes")
        ax.set_ylabel("Node")
        ax.set_xlabel("Motif Type")
        
        plt.tight_layout()
        return fig
    
    def visualize_role_membership(self, n_nodes: int = 20) -> plt.Figure:
        """
        Visualize role membership for a subset of nodes.
        
        Args:
            n_nodes: Number of nodes to visualize
            
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Sample nodes to visualize
        n_sample = min(n_nodes, self.graph.number_of_nodes())
        sampled_nodes = np.random.choice(self.graph.number_of_nodes(), n_sample, replace=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Plot heatmap
        sns.heatmap(
            self.role_membership[sampled_nodes],
            cmap="YlOrRd",
            ax=ax,
            yticklabels=[f"Node {n}" for n in sampled_nodes],
            xticklabels=role_names,
            cbar_kws={'label': 'Role Membership'}
        )
        
        ax.set_title(f"Structural Role Membership for {n_sample} Random Nodes")
        ax.set_ylabel("Node")
        ax.set_xlabel("Structural Role")
        
        plt.tight_layout()
        return fig
    
    def visualize_role_definitions(self) -> plt.Figure:
        """
        Visualize the definition of each role in terms of motifs.
        
        Returns:
            Matplotlib figure
        """
        if self.role_definitions is None:
            self.discover_structural_roles()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Plot heatmap
        sns.heatmap(
            self.role_definitions,
            cmap="viridis",
            ax=ax,
            yticklabels=role_names,
            xticklabels=self.motif_names,
            cbar_kws={'label': 'Motif Weight'}
        )
        
        ax.set_title("Structural Role Definitions")
        ax.set_ylabel("Role")
        ax.set_xlabel("Motif Type")
        
        plt.tight_layout()
        return fig
    
    def visualize_graph_with_roles(self) -> plt.Figure:
        """
        Visualize the graph with nodes colored by their primary role.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get role interpretations for legend
        role_interpretations = self.interpret_roles()
        
        # Create position layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes colored by role
        cmap = plt.cm.get_cmap("tab10", self.n_roles)
        
        # Draw nodes for each role
        for role in range(self.n_roles):
            role_nodes = [n for n, r in enumerate(primary_roles) if r == role]
            
            if role_nodes:
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=role_nodes,
                    node_color=[cmap(role)] * len(role_nodes),
                    node_size=80,
                    alpha=0.8,
                    label=role_interpretations[role]["name"],
                    ax=ax
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            ax=ax
        )
        
        ax.set_title("Graph Visualization with Structural Roles")
        ax.legend()
        ax.set_axis_off()
        
        return fig
    
    def visualize_role_distribution(self) -> plt.Figure:
        """
        Visualize the distribution of primary roles in the network.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Count roles
        role_counts = Counter(primary_roles)
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        roles = sorted(role_counts.keys())
        counts = [role_counts[r] for r in roles]
        labels = [role_interpretations[r]["name"] for r in roles]
        
        # Plot bar chart
        bars = ax.bar(roles, counts, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_xticks(roles)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Structural Role")
        ax.set_ylabel("Number of Nodes")
        ax.set_title("Distribution of Primary Structural Roles")
        
        plt.tight_layout()
        return fig
    
    def visualize_roles_by_community(self) -> plt.Figure:
        """
        Visualize the distribution of roles across different communities.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Group nodes by community
        community_role_counts = {}
        for node, community in self.node_communities.items():
            if community not in community_role_counts:
                community_role_counts[community] = Counter()
            
            role = primary_roles[node]
            community_role_counts[community][role] += 1
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for stacked bar chart
        communities = sorted(community_role_counts.keys())
        roles = list(range(self.n_roles))
        
        bottom = np.zeros(len(communities))
        
        # Plot each role as a stack
        for role in roles:
            counts = [community_role_counts[c][role] for c in communities]
            ax.bar(
                communities, 
                counts, 
                bottom=bottom, 
                label=role_interpretations[role]["name"],
                alpha=0.7
            )
            bottom += counts
        
        ax.set_xlabel("Community")
        ax.set_ylabel("Nodes")
        ax.set_title("Roles by Community")
        ax.set_xticks(communities)
        ax.set_xticklabels([f"C{c}" for c in communities])
        ax.legend(fontsize=8)
    
    def visualize_role_similarity(self) -> plt.Figure:
        """
        Visualize the similarity between different roles based on their definitions.
        
        Returns:
            Matplotlib figure
        """
        if self.role_definitions is None:
            self.discover_structural_roles()
        
        # Compute similarity matrix
        similarity = np.dot(self.role_definitions, self.role_definitions.T)
        
        # Normalize to [0, 1]
        row_norms = np.sqrt(np.sum(self.role_definitions**2, axis=1))
        similarity = similarity / np.outer(row_norms, row_norms)
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        im = ax.imshow(similarity, cmap="YlGnBu", vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        
        # Set labels
        ax.set_xticks(np.arange(len(role_names)))
        ax.set_yticks(np.arange(len(role_names)))
        ax.set_xticklabels(role_names, rotation=45, ha="right")
        ax.set_yticklabels(role_names)
        ax.set_title("Role Similarity Matrix")
        
        # Add values in cells
        for i in range(len(role_names)):
            for j in range(len(role_names)):
                ax.text(j, i, f"{similarity[i, j]:.2f}",
                         ha="center", va="center", color="black" if similarity[i, j] < 0.7 else "white")
        
        plt.tight_layout()
        return fig
    
    def visualize_node_role_transitions(self) -> plt.Figure:
        """
        Visualize how roles transition between neighboring nodes.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Count role transitions along edges
        transitions = np.zeros((self.n_roles, self.n_roles))
        
        for u, v in self.graph.edges():
            role_u = primary_roles[u]
            role_v = primary_roles[v]
            transitions[role_u, role_v] += 1
            transitions[role_v, role_u] += 1  # Undirected graph
        
        # Normalize by row sums
        row_sums = transitions.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_probs = transitions / row_sums
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(transition_probs, cmap="viridis", vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Transition Probability")
        
        # Set labels
        ax.set_xticks(np.arange(len(role_names)))
        ax.set_yticks(np.arange(len(role_names)))
        ax.set_xticklabels(role_names, rotation=45, ha="right")
        ax.set_yticklabels(role_names)
        ax.set_title("Role Transition Probabilities Between Neighbors")
        ax.set_xlabel("To Role")
        ax.set_ylabel("From Role")
        
        # Add values in cells
        for i in range(len(role_names)):
            for j in range(len(role_names)):
                ax.text(j, i, f"{transition_probs[i, j]:.2f}",
                         ha="center", va="center", 
                         color="black" if transition_probs[i, j] < 0.7 else "white")
        
        plt.tight_layout()
        return fig
    
    def visualize_role_community_correlation(self) -> plt.Figure:
        """
        Visualize the correlation between role membership and community membership.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Create community membership matrix (one-hot encoding)
        communities = sorted(set(self.node_communities.values()))
        n_communities = len(communities)
        
        community_matrix = np.zeros((self.graph.number_of_nodes(), n_communities))
        for node, comm in self.node_communities.items():
            idx = communities.index(comm)
            community_matrix[node, idx] = 1
        
        # Compute correlation matrix
        correlation = np.zeros((self.n_roles, n_communities))
        
        for r in range(self.n_roles):
            for c in range(n_communities):
                role_vector = self.role_membership[:, r]
                comm_vector = community_matrix[:, c]
                
                # Compute correlation
                corr = np.corrcoef(role_vector, comm_vector)[0, 1]
                correlation[r, c] = corr if not np.isnan(corr) else 0
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Correlation")
        
        # Set labels
        ax.set_xticks(np.arange(len(communities)))
        ax.set_yticks(np.arange(len(role_names)))
        ax.set_xticklabels([f"C{c}" for c in communities])
        ax.set_yticklabels(role_names)
        ax.set_title("Role-Community Correlation Matrix")
        ax.set_xlabel("Community")
        ax.set_ylabel("Role")
        
        # Add values in cells
        for i in range(len(role_names)):
            for j in range(len(communities)):
                ax.text(j, i, f"{correlation[i, j]:.2f}",
                         ha="center", va="center", 
                         color="black" if abs(correlation[i, j]) < 0.7 else "white")
        
        plt.tight_layout()
        return fig
    
    def create_role_dashboard(self) -> plt.Figure:
        """
        Create a comprehensive dashboard with role analysis visualizations.
        
        Returns:
            Matplotlib figure
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Role definitions (motif weights)
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        self._plot_role_definitions(ax1)
        
        # 2. Role distribution
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        self._plot_role_distribution(ax2)
        
        # 3. Role by community
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        self._plot_roles_by_community(ax3)
        
        # 4. Role-community correlation
        ax4 = plt.subplot2grid((2, 3), (1, 0))
        self._plot_role_community_correlation(ax4)
        
        # 5. Role transitions
        ax5 = plt.subplot2grid((2, 3), (1, 1))
        self._plot_node_role_transitions(ax5)
        
        # 6. Network with roles
        ax6 = plt.subplot2grid((2, 3), (1, 2))
        self._plot_graph_with_roles(ax6)
        
        # Add overall title
        plt.suptitle("Structural Role Analysis Dashboard", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return fig
    
    # Helper methods for dashboard plots
    def _plot_role_definitions(self, ax):
        """Plot role definitions on given axis."""
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Plot heatmap
        sns.heatmap(
            self.role_definitions,
            cmap="viridis",
            ax=ax,
            yticklabels=role_names,
            xticklabels=self.motif_names,
            cbar_kws={'label': 'Weight'}
        )
        
        ax.set_title("Role Definitions")
        ax.set_ylabel("Role")
        ax.set_xlabel("Motif")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    def _plot_role_distribution(self, ax):
        """Plot role distribution on given axis."""
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Count roles
        role_counts = Counter(primary_roles)
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        
        # Prepare data
        roles = sorted(role_counts.keys())
        counts = [role_counts[r] for r in roles]
        labels = [role_interpretations[r]["name"] for r in roles]
        
        # Plot bar chart
        bars = ax.bar(roles, counts, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        ax.set_xticks(roles)
        ax.set_xticklabels([role_interpretations[r]["name"].split()[0] for r in roles], 
                          rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Role")
        ax.set_ylabel("Nodes")
        ax.set_title("Role Distribution")
    
    def _plot_roles_by_community(self, ax):
        """Plot roles by community on given axis."""
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Group nodes by community
        community_role_counts = {}
        for node, community in self.node_communities.items():
            if community not in community_role_counts:
                community_role_counts[community] = Counter()
            
            role = primary_roles[node]
            community_role_counts[community][role] += 1
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        
        # Prepare data for stacked bar chart
        communities = sorted(community_role_counts.keys())
        roles = list(range(self.n_roles))
        
        bottom = np.zeros(len(communities))
        
        # Plot each role as a stack
        for role in roles:
            counts = [community_role_counts[c][role] for c in communities]
            ax.bar(
                communities, 
                counts, 
                bottom=bottom, 
                label=role_interpretations[role]["name"],
                alpha=0.7
            )
            bottom += counts
        
        ax.set_xlabel("Community")
        ax.set_ylabel("Nodes")
        ax.set_title("Roles by Community")
        ax.set_xticks(communities)
        ax.set_xticklabels([f"C{c}" for c in communities])
        ax.legend(fontsize=8)
    
    def _plot_role_community_correlation(self, ax):
        """Plot role-community correlation on given axis."""
        # Create community membership matrix (one-hot encoding)
        communities = sorted(set(self.node_communities.values()))
        n_communities = len(communities)
        
        community_matrix = np.zeros((self.graph.number_of_nodes(), n_communities))
        for node, comm in self.node_communities.items():
            idx = communities.index(comm)
            community_matrix[node, idx] = 1
        
        # Compute correlation matrix
        correlation = np.zeros((self.n_roles, n_communities))
        
        for r in range(self.n_roles):
            for c in range(n_communities):
                role_vector = self.role_membership[:, r]
                comm_vector = community_matrix[:, c]
                
                # Compute correlation
                corr = np.corrcoef(role_vector, comm_vector)[0, 1]
                correlation[r, c] = corr if not np.isnan(corr) else 0
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"] for i in range(self.n_roles)]
        
        # Plot heatmap
        im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Correlation")
        
        # Set labels
        ax.set_xticks(np.arange(len(communities)))
        ax.set_yticks(np.arange(len(role_names)))
        ax.set_xticklabels([f"C{c}" for c in communities])
        ax.set_yticklabels(role_names)
        ax.set_title("Role-Community Correlation")
        ax.set_xlabel("Community")
        ax.set_ylabel("Role")
    
    def _plot_node_role_transitions(self, ax):
        """Plot role transitions on given axis."""
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Count role transitions along edges
        transitions = np.zeros((self.n_roles, self.n_roles))
        
        for u, v in self.graph.edges():
            role_u = primary_roles[u]
            role_v = primary_roles[v]
            transitions[role_u, role_v] += 1
            transitions[role_v, role_u] += 1  # Undirected graph
        
        # Normalize by row sums
        row_sums = transitions.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_probs = transitions / row_sums
        
        # Get role interpretations
        role_interpretations = self.interpret_roles()
        role_names = [role_interpretations[i]["name"].split()[0] for i in range(self.n_roles)]
        
        # Plot heatmap
        im = ax.imshow(transition_probs, cmap="viridis", vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Probability")
        
        # Set labels
        ax.set_xticks(np.arange(len(role_names)))
        ax.set_yticks(np.arange(len(role_names)))
        ax.set_xticklabels(role_names, rotation=45, ha="right")
        ax.set_yticklabels(role_names)
        ax.set_title("Role Transitions")
        ax.set_xlabel("To Role")
        ax.set_ylabel("From Role")
    
    def _plot_graph_with_roles(self, ax):
        """Plot graph with roles on given axis."""
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Get role interpretations for legend
        role_interpretations = self.interpret_roles()
        
        # Create position layout (with smaller graph for speed)
        if self.graph.number_of_nodes() > 500:
            # Take a subgraph for visualization
            sampled_nodes = np.random.choice(
                list(self.graph.nodes()), 
                size=min(500, self.graph.number_of_nodes()),
                replace=False
            )
            subgraph = self.graph.subgraph(sampled_nodes)
            pos = nx.spring_layout(subgraph, seed=42)
            graph_to_draw = subgraph
            roles_to_draw = primary_roles[sampled_nodes]
        else:
            pos = nx.spring_layout(self.graph, seed=42)
            graph_to_draw = self.graph
            roles_to_draw = primary_roles
        
        # Draw nodes colored by role
        cmap = plt.cm.get_cmap("tab10", self.n_roles)
        
        # Draw edges first for better visual
        nx.draw_networkx_edges(
            graph_to_draw, pos,
            alpha=0.2,
            ax=ax
        )
        
        # Draw nodes for each role
        for role in range(self.n_roles):
            role_nodes = [n for n, r in enumerate(roles_to_draw) if r == role]
            
            if role_nodes:
                nx.draw_networkx_nodes(
                    graph_to_draw, pos,
                    nodelist=role_nodes,
                    node_color=[cmap(role)] * len(role_nodes),
                    node_size=30,
                    alpha=0.8,
                    label=role_interpretations[role]["name"].split()[0],
                    ax=ax
                )
        
        ax.set_title("Network with Roles")
        ax.legend(fontsize=8)
        ax.set_axis_off()
    
    def create_example_rule_set(self) -> List[Dict]:
        """
        Create an example set of conditional labeling rules.
        
        Returns:
            List of rule dictionaries
        """
        rules = []
        
        # Rules based on role membership
        rules.append({
            "name": "Pure Hub",
            "condition": lambda node, roles, motifs: 
                roles[1] > 0.7,  # Assuming role 1 is the hub role
            "label": "Pure Hub"
        })
        
        rules.append({
            "name": "Bridge Node",
            "condition": lambda node, roles, motifs: 
                roles[2] > 0.5 and motifs[self.motif_names.index("betweenness")] > 0.5,
            "label": "Bridge Node"
        })
        
        # Rules based on motif counts
        rules.append({
            "name": "Triangle-Rich",
            "condition": lambda node, roles, motifs: 
                motifs[self.motif_names.index("triangle")] > 0.5 and motifs[self.motif_names.index("clustering")] > 0.6,
            "label": "Triangle-Rich"
        })
        
        rules.append({
            "name": "Connector",
            "condition": lambda node, roles, motifs: 
                motifs[self.motif_names.index("degree")] > 0.7 and motifs[self.motif_names.index("triangle")] < 0.3,
            "label": "Connector"
        })
        
        # More complex rules with neighborhood conditions
        rules.append({
            "name": "Community Boundary",
            "condition": lambda node, roles, motifs: 
                self._is_on_community_boundary(node),
            "label": "Community Boundary"
        })
        
        return rules
    
    def _is_on_community_boundary(self, node) -> bool:
        """
        Check if a node is on the boundary between communities.
        
        Args:
            node: Node index
            
        Returns:
            True if node is on community boundary, False otherwise
        """
        # Get node's community
        node_comm = self.node_communities[node]
        
        # Check neighbors' communities
        neighbor_comms = set()
        for nbr in self.graph.neighbors(node):
            nbr_comm = self.node_communities[nbr]
            neighbor_comms.add(nbr_comm)
        
        # Node is on boundary if it has neighbors in different communities
        return len(neighbor_comms) > 1
    
    def get_role_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for structural roles.
        
        Returns:
            DataFrame with role metrics
        """
        if self.role_membership is None:
            self.discover_structural_roles()
        
        # Get primary roles
        primary_roles = self.get_primary_roles()
        
        # Initialize metrics
        metrics = []
        
        # Get interpretations
        role_interpretations = self.interpret_roles()
        
        for role_idx in range(self.n_roles):
            role_name = role_interpretations[role_idx]["name"]
            
            # Get nodes with this primary role
            role_nodes = [n for n, r in enumerate(primary_roles) if r == role_idx]
            n_nodes = len(role_nodes)
            
            if n_nodes == 0:
                continue
            
            # Calculate average degree
            degrees = [self.graph.degree(n) for n in role_nodes]
            avg_degree = np.mean(degrees)
            
            # Calculate average clustering coefficient
            clustering = [nx.clustering(self.graph, n) for n in role_nodes]
            avg_clustering = np.mean(clustering)
            
            # Calculate average betweenness (for smaller graphs)
            if self.graph.number_of_nodes() <= 1000:
                try:
                    betweenness = nx.betweenness_centrality(self.graph)
                    avg_betweenness = np.mean([betweenness[n] for n in role_nodes])
                except:
                    avg_betweenness = np.nan
            else:
                avg_betweenness = np.nan
            
            # Calculate community diversity
            communities = [self.node_communities[n] for n in role_nodes]
            unique_comms = len(set(communities))
            
            # Store metrics
            metrics.append({
                "Role": role_name,
                "Node Count": n_nodes,
                "Avg Degree": avg_degree,
                "Avg Clustering": avg_clustering,
                "Avg Betweenness": avg_betweenness,
                "Community Diversity": unique_comms,
                "% of Graph": (n_nodes / self.graph.number_of_nodes()) * 100
            })
        
        return pd.DataFrame(metrics)
    
    def run_full_analysis(self) -> Dict:
        """
        Run a complete analysis and return comprehensive results.
        
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        # 1. Compute motif profiles
        self.compute_motif_profiles()
        results["motif_profiles"] = self.motif_profiles
        results["motif_names"] = self.motif_names
        
        # 2. Discover structural roles
        self.discover_structural_roles()
        results["role_membership"] = self.role_membership
        results["role_definitions"] = self.role_definitions
        
        # 3. Interpret roles
        results["role_interpretations"] = self.interpret_roles()
        
        # 4. Create structural labels
        results["structural_labels"] = self.create_structural_labels(method="primary_role")
        
        # 5. Calculate role metrics
        results["role_metrics"] = self.get_role_metrics()
        
        # 6. Create visualizations
        results["visualizations"] = {
            "motif_profiles": self.visualize_motif_profiles(),
            "role_membership": self.visualize_role_membership(),
            "role_definitions": self.visualize_role_definitions(),
            "graph_with_roles": self.visualize_graph_with_roles(),
            "role_distribution": self.visualize_role_distribution(),
            "roles_by_community": self.visualize_roles_by_community(),
            "role_community_correlation": self.visualize_role_community_correlation(),
            "role_dashboard": self.create_role_dashboard()
        }
        
        # 7. Example rule set
        results["example_rules"] = self.create_example_rule_set()
        
        return results