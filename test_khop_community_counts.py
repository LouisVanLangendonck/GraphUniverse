"""
Test file to compare two implementations of compute_khop_community_counts_universe_indexed
and visualize the differences.
"""

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import random

def compute_khop_community_counts_new(
    graph: nx.Graph,
    community_labels: np.ndarray,
    universe_communities: Dict[int, int],
    universe_K: int,
    k: int
) -> torch.Tensor:
    """
    NEW implementation: Compute k-hop community counts (only nodes at exactly k-hops) with universe indexing.
    """
    n_nodes = graph.number_of_nodes()
    counts = np.zeros((n_nodes, universe_K))
    
    for node in range(n_nodes):
        # Get nodes at exact distance k using single-source shortest path
        sp_lengths = nx.single_source_shortest_path_length(graph, node, cutoff=k)
        khop_nodes = [n for n, dist in sp_lengths.items() if dist == k]
        
        for neighbor in khop_nodes:
            local_comm = community_labels[neighbor]
            if local_comm in universe_communities:
                universe_comm = universe_communities[local_comm]
                counts[node, universe_comm] += 1
            else:
                raise ValueError(f"Community {local_comm} not in universe_communities")
    
    return torch.tensor(counts, dtype=torch.float)

def compute_khop_community_counts_old(
    graph: nx.Graph,
    community_labels: np.ndarray,
    universe_communities: Dict[int, int],
    universe_K: int,
    k: int
) -> torch.Tensor:
    """
    OLD implementation: Compute k-hop community counts with universe indexing.
    """
    n_nodes = graph.number_of_nodes()
    counts = np.zeros((n_nodes, universe_K))
    
    # For each node
    for node in range(n_nodes):
        # Get k-hop neighborhood
        neighbors = set([node])
        for _ in range(k):
            new_neighbors = set()
            for n in neighbors:
                new_neighbors.update(graph.neighbors(n))
            neighbors.update(new_neighbors)
        
        # Count communities in neighborhood
        for neighbor in neighbors:
            local_comm = community_labels[neighbor]
            if local_comm in universe_communities:
                universe_comm = universe_communities[local_comm]
                counts[node, universe_comm] += 1
            else:
                raise ValueError(f"Community {local_comm} not found in universe communities")
    
    return torch.tensor(counts, dtype=torch.float)

def create_test_graphs() -> List[Tuple[nx.Graph, np.ndarray, Dict[int, int]]]:
    """
    Create several test graphs with different structures to test k-hop counting.
    Returns: List of (graph, community_labels, universe_communities) tuples
    """
    test_cases = []
    
    # Test Case 1: Simple chain graph
    print("Creating Test Case 1: Chain Graph")
    G1 = nx.Graph()
    G1.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)])
    # Communities: [0,0,1,1,2,2,3] - alternating pattern
    community_labels1 = np.array([0, 0, 1, 1, 2, 2, 3])
    universe_communities1 = {0: 0, 1: 1, 2: 2, 3: 3}
    test_cases.append((G1, community_labels1, universe_communities1))
    
    # Test Case 2: Star graph
    print("Creating Test Case 2: Star Graph")
    G2 = nx.Graph()
    G2.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5)])
    # Communities: [0,1,1,2,2,3] - center node different from leaves
    community_labels2 = np.array([0, 1, 1, 2, 2, 3])
    universe_communities2 = {0: 0, 1: 1, 2: 2, 3: 3}
    test_cases.append((G2, community_labels2, universe_communities2))
    
    # Test Case 3: Cycle graph
    print("Creating Test Case 3: Cycle Graph")
    G3 = nx.Graph()
    G3.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)])
    # Communities: [0,0,1,1,2,2] - repeating pattern
    community_labels3 = np.array([0, 0, 1, 1, 2, 2])
    universe_communities3 = {0: 0, 1: 1, 2: 2}
    test_cases.append((G3, community_labels3, universe_communities3))
    
    # Test Case 4: Tree with multiple levels
    print("Creating Test Case 4: Tree Graph")
    G4 = nx.Graph()
    G4.add_edges_from([
        (0,1), (0,2),  # Level 1
        (1,3), (1,4), (2,5), (2,6),  # Level 2
        (3,7), (3,8), (4,9), (5,10), (6,11)  # Level 3
    ])
    # Communities: [0,1,1,2,2,2,2,3,3,3,3,3] - hierarchical
    community_labels4 = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    universe_communities4 = {0: 0, 1: 1, 2: 2, 3: 3}
    test_cases.append((G4, community_labels4, universe_communities4))
    
    # Test Case 5: Random graph
    print("Creating Test Case 5: Random Graph")
    random.seed(42)
    G5 = nx.erdos_renyi_graph(10, 0.4, seed=42)
    # Communities: random assignment
    community_labels5 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1])
    universe_communities5 = {0: 0, 1: 1, 2: 2, 3: 3}
    test_cases.append((G5, community_labels5, universe_communities5))
    
    return test_cases

def get_khop_neighbors_new(graph: nx.Graph, node: int, k: int) -> List[int]:
    """Get nodes at exactly k hops from node using NEW method."""
    sp_lengths = nx.single_source_shortest_path_length(graph, node, cutoff=k)
    return [n for n, dist in sp_lengths.items() if dist == k]

def get_khop_neighbors_old(graph: nx.Graph, node: int, k: int) -> List[int]:
    """Get nodes within k hops from node using OLD method."""
    neighbors = set([node])
    for _ in range(k):
        new_neighbors = set()
        for n in neighbors:
            new_neighbors.update(graph.neighbors(n))
        neighbors.update(new_neighbors)
    return list(neighbors)

def visualize_graph_comparison(graph: nx.Graph, community_labels: np.ndarray, 
                             universe_communities: Dict[int, int], k: int, 
                             test_case_name: str):
    """
    Visualize the graph focusing on ONE center node and show what each method counts for that node.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Color scheme for communities
    community_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Pick a center node (middle node for clarity)
    center_node = graph.number_of_nodes() // 2
    print(f"\nAnalyzing center node {center_node} (Community {community_labels[center_node]})")
    
    # Create layout
    if graph.number_of_nodes() <= 12:
        pos = nx.spring_layout(graph, seed=42)
    else:
        pos = nx.kamada_kawai_layout(graph)
    
    # Plot 1: Original graph with center node highlighted
    ax1 = axes[0]
    node_colors = [community_colors[community_labels[i] % len(community_colors)] for i in range(graph.number_of_nodes())]
    
    # Highlight center node with a border
    nx.draw(graph, pos, ax=ax1, 
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_weight='bold')
    
    # Add special highlighting for center node
    center_pos = pos[center_node]
    circle = plt.Circle(center_pos, 0.15, fill=False, edgecolor='black', linewidth=3)
    ax1.add_patch(circle)
    
    # Add community labels
    for i, (node, (x, y)) in enumerate(pos.items()):
        ax1.text(x, y+0.1, f'C{community_labels[i]}', 
                ha='center', va='bottom', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax1.set_title(f'{test_case_name}\nCenter Node {center_node} (Community {community_labels[center_node]})', fontsize=12)
    
    # Plot 2: NEW method - only exactly k-hop nodes from center
    ax2 = axes[1]
    
    # Get k-hop neighbors for center node using NEW method
    khop_nodes_new = get_khop_neighbors_new(graph, center_node, k)
    print(f"NEW method (exactly k={k} hops from node {center_node}): {khop_nodes_new}")
    
    # Color nodes: center node in bright color, k-hop neighbors highlighted, others gray
    node_colors_new = []
    for i in range(graph.number_of_nodes()):
        if i == center_node:
            node_colors_new.append('yellow')  # Center node
        elif i in khop_nodes_new:
            node_colors_new.append(community_colors[community_labels[i] % len(community_colors)])  # K-hop neighbors
        else:
            node_colors_new.append('lightgray')  # Others
    
    nx.draw(graph, pos, ax=ax2,
            node_color=node_colors_new,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_weight='bold')
    
    # Highlight center node with border
    circle = plt.Circle(center_pos, 0.15, fill=False, edgecolor='black', linewidth=3)
    ax2.add_patch(circle)
    
    ax2.set_title(f'NEW Method: Exactly {k}-hop Nodes\nfrom Center Node {center_node}', fontsize=12)
    
    # Plot 3: OLD method - all nodes within k hops from center
    ax3 = axes[2]
    
    # Get k-hop neighborhood for center node using OLD method
    khop_nodes_old = get_khop_neighbors_old(graph, center_node, k)
    print(f"OLD method (up to k={k} hops from node {center_node}): {khop_nodes_old}")
    
    # Color nodes: center node in bright color, k-hop neighborhood highlighted, others gray
    node_colors_old = []
    for i in range(graph.number_of_nodes()):
        if i == center_node:
            node_colors_old.append('yellow')  # Center node
        elif i in khop_nodes_old:
            node_colors_old.append(community_colors[community_labels[i] % len(community_colors)])  # K-hop neighborhood
        else:
            node_colors_old.append('lightgray')  # Others
    
    nx.draw(graph, pos, ax=ax3,
            node_color=node_colors_old,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_weight='bold')
    
    # Highlight center node with border
    circle = plt.Circle(center_pos, 0.15, fill=False, edgecolor='black', linewidth=3)
    ax3.add_patch(circle)
    
    ax3.set_title(f'OLD Method: All Nodes within {k} Hops\nfrom Center Node {center_node}', fontsize=12)
    
    # Add legend
    legend_elements = []
    unique_communities = np.unique(community_labels)
    for comm in unique_communities:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=community_colors[comm % len(community_colors)],
                                        markersize=10, label=f'Community {comm}'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='yellow', markersize=10, label='Center Node'))
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Calculate and display the actual count vectors for the center node
    universe_K = len(universe_communities)
    counts_new = compute_khop_community_counts_new(graph, community_labels, universe_communities, universe_K, k)
    counts_old = compute_khop_community_counts_old(graph, community_labels, universe_communities, universe_K, k)
    
    new_counts = counts_new[center_node].numpy()
    old_counts = counts_old[center_node].numpy()
    
    # Add text below the plots
    fig.text(0.02, 0.02, f'Center Node {center_node} Count Vectors:\n'
                          f'NEW method (exactly k={k}): {new_counts}\n'
                          f'OLD method (up to k={k}): {old_counts}\n'
                          f'Difference: {new_counts - old_counts}',
              fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'khop_comparison_{test_case_name.lower().replace(" ", "_")}_k{k}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis for this center node
    print(f"\nDetailed analysis for center node {center_node}:")
    print(f"  Community: {community_labels[center_node]}")
    print(f"  NEW method counts: {new_counts}")
    print(f"  OLD method counts: {old_counts}")
    print(f"  Difference: {new_counts - old_counts}")
    print(f"  Total difference: {np.sum(np.abs(new_counts - old_counts)):.2f}")

def analyze_khop_differences(graph: nx.Graph, community_labels: np.ndarray, 
                           universe_communities: Dict[int, int], universe_K: int, k: int) -> Dict:
    """
    Compare the two implementations and analyze differences.
    """
    print(f"\nAnalyzing k={k}-hop community counts...")
    
    # Run both implementations
    counts_new = compute_khop_community_counts_new(graph, community_labels, universe_communities, universe_K, k)
    counts_old = compute_khop_community_counts_old(graph, community_labels, universe_communities, universe_K, k)
    
    # Convert to numpy for analysis
    counts_new_np = counts_new.numpy()
    counts_old_np = counts_old.numpy()
    
    # Calculate differences
    diff = counts_new_np - counts_old_np
    abs_diff = np.abs(diff)
    
    analysis = {
        'counts_new': counts_new_np,
        'counts_old': counts_old_np,
        'difference': diff,
        'abs_difference': abs_diff,
        'total_diff': np.sum(abs_diff),
        'max_diff': np.max(abs_diff),
        'mean_diff': np.mean(abs_diff),
        'nodes_with_diff': np.sum(abs_diff > 0),
        'total_nodes': len(community_labels)
    }
    
    print(f"  Total absolute difference: {analysis['total_diff']:.2f}")
    print(f"  Max difference per node: {analysis['max_diff']:.2f}")
    print(f"  Mean difference per node: {analysis['mean_diff']:.2f}")
    print(f"  Nodes with differences: {analysis['nodes_with_diff']}/{analysis['total_nodes']}")
    
    return analysis

def detailed_node_analysis(graph: nx.Graph, community_labels: np.ndarray, 
                         universe_communities: Dict[int, int], universe_K: int, k: int):
    """
    Detailed analysis for a single graph showing exactly what each method counts.
    """
    print(f"\n=== DETAILED ANALYSIS: k={k}-hop ===")
    
    # Get counts from both methods
    counts_new = compute_khop_community_counts_new(graph, community_labels, universe_communities, universe_K, k)
    counts_old = compute_khop_community_counts_old(graph, community_labels, universe_communities, universe_K, k)
    
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Community labels: {community_labels}")
    print(f"Universe communities: {universe_communities}")
    
    # Analyze each node
    for node in range(graph.number_of_nodes()):
        print(f"\nNode {node} (Community {community_labels[node]}):")
        
        # Get k-hop neighbors using shortest path (NEW method)
        khop_nodes_new = get_khop_neighbors_new(graph, node, k)
        
        # Get k-hop neighborhood (OLD method)
        khop_nodes_old = get_khop_neighbors_old(graph, node, k)
        
        print(f"  NEW method (exactly k={k} hops): {khop_nodes_new}")
        print(f"  OLD method (up to k={k} hops): {khop_nodes_old}")
        
        # Show community counts
        new_counts = counts_new[node].numpy()
        old_counts = counts_old[node].numpy()
        
        print(f"  NEW counts: {new_counts}")
        print(f"  OLD counts: {old_counts}")
        print(f"  Difference: {new_counts - old_counts}")

def create_summary_statistics(test_cases: List[Tuple], k_values: List[int] = [1, 2, 3]):
    """
    Create summary statistics comparing the two methods.
    """
    print("\n=== SUMMARY STATISTICS ===")
    
    summary_data = []
    
    for i, (graph, community_labels, universe_communities) in enumerate(test_cases):
        universe_K = len(universe_communities)
        
        for k in k_values:
            analysis = analyze_khop_differences(graph, community_labels, universe_communities, universe_K, k)
            
            summary_data.append({
                'test_case': i + 1,
                'k': k,
                'n_nodes': graph.number_of_nodes(),
                'n_edges': graph.number_of_edges(),
                'total_diff': analysis['total_diff'],
                'max_diff': analysis['max_diff'],
                'mean_diff': analysis['mean_diff'],
                'nodes_with_diff': analysis['nodes_with_diff'],
                'total_nodes': analysis['total_nodes']
            })
    
    # Create summary table
    print("\nSummary Table:")
    print("Test | k | Nodes | Edges | Total Diff | Max Diff | Mean Diff | Nodes w/ Diff")
    print("-" * 80)
    
    for data in summary_data:
        print(f"{data['test_case']:4d} | {data['k']:1d} | {data['n_nodes']:5d} | {data['n_edges']:5d} | "
              f"{data['total_diff']:9.2f} | {data['max_diff']:7.2f} | {data['mean_diff']:8.2f} | "
              f"{data['nodes_with_diff']:13d}")
    
    return summary_data

def main():
    """
    Main function to run the comparison.
    """
    print("=== K-HOP COMMUNITY COUNT COMPARISON ===")
    print("Comparing NEW (exact k-hop) vs OLD (up to k-hop) implementations")
    
    # Create test graphs
    test_cases = create_test_graphs()
    
    # Test with different k values
    k_values = [1, 2, 3]
    
    # Test case names for visualization
    test_names = ["Chain Graph", "Star Graph", "Cycle Graph", "Tree Graph", "Random Graph"]
    
    # Create graph visualizations for each test case and k value
    print("\nCreating graph visualizations...")
    for i, ((graph, community_labels, universe_communities), test_name) in enumerate(zip(test_cases, test_names)):
        print(f"\nVisualizing {test_name}...")
        for k in k_values:
            print(f"  Creating visualization for k={k}...")
            visualize_graph_comparison(graph, community_labels, universe_communities, k, test_name)
    
    # Create summary statistics
    summary_data = create_summary_statistics(test_cases, k_values)
    
    # Detailed analysis for first test case
    print("\n" + "="*50)
    print("DETAILED ANALYSIS OF FIRST TEST CASE")
    print("="*50)
    detailed_node_analysis(*test_cases[0], len(test_cases[0][2]), 2)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key differences:")
    print("- NEW method: Only counts nodes at exactly k hops from each node")
    print("- OLD method: Counts all nodes within k hops (including closer nodes)")
    print("- This means OLD method can double-count nodes that are closer than k hops")
    print("\nVisualization files created:")
    for test_name in test_names:
        for k in k_values:
            print(f"  - khop_comparison_{test_name.lower().replace(' ', '_')}_k{k}.png")

if __name__ == "__main__":
    main() 