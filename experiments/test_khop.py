"""
Script to test and visualize k-hop community counting implementation.
Generates graphs and shows community/k-hop count visualizations for verification.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
import torch

# Import your modules (adjust paths as needed)
from mmsb.model import GraphUniverse, GraphSample
from experiments.inductive.data import compute_khop_community_counts_batch


def create_test_universe(K: int = 4, feature_dim: int = 8, seed: int = 42) -> GraphUniverse:
    """Create a test universe with specified parameters."""
    return GraphUniverse(
        K=K,
        feature_dim=feature_dim,
        edge_density=0.15,
        homophily=0.8,
        randomness_factor=0.1,
        cluster_count_factor=1.0,
        center_variance=1.0,
        cluster_variance=0.1,
        assignment_skewness=0.0,
        community_exclusivity=1.0,
        degree_center_method="linear",
        seed=seed
    )


def create_test_graph(universe: GraphUniverse, n_nodes: int = 50, seed: int = 42) -> GraphSample:
    """Create a test graph from the universe."""
    np.random.seed(seed)
    
    # Sample a subset of communities
    n_communities = min(3, universe.K)
    communities = universe.sample_connected_community_subset(size=n_communities, seed=seed)
    
    graph_sample = GraphSample(
        universe=universe,
        communities=communities,
        n_nodes=n_nodes,
        min_component_size=5,
        degree_heterogeneity=0.3,
        edge_noise=0.1,
        feature_regime_balance=0.5,
        target_homophily=0.8,
        target_density=0.15,
        use_configuration_model=False,
        degree_distribution="standard",
        power_law_exponent=None,
        target_avg_degree=None,
        triangle_enhancement=0.0,
        max_mean_community_deviation=0.15,
        max_max_community_deviation=0.3,
        max_parameter_search_attempts=20,
        parameter_search_range=0.5,
        min_edge_density=0.005,
        max_retries=5,
        seed=seed
    )
    
    return graph_sample


def get_khop_neighbors_manual(graph: nx.Graph, node: int, k: int) -> List[int]:
    """
    Manually compute k-hop neighbors for verification.
    Returns nodes at exactly k hops from the given node.
    """
    if k == 0:
        return [node]
    
    # Get shortest path lengths from the node
    path_lengths = nx.single_source_shortest_path_length(graph, node, cutoff=k)
    
    # Filter to get nodes at exactly k hops
    khop_neighbors = [n for n, dist in path_lengths.items() if dist == k]
    
    return khop_neighbors


def analyze_khop_counts(graph_sample: GraphSample, k: int, sample_nodes: List[int] = None) -> Dict:
    """
    Analyze k-hop community counts for verification.
    """
    graph = graph_sample.graph
    community_labels = graph_sample.community_labels
    communities = graph_sample.communities
    
    # Compute k-hop counts using the implementation
    khop_counts = compute_khop_community_counts_batch(graph, community_labels, k)
    
    # If no sample nodes provided, pick a few random ones
    if sample_nodes is None:
        sample_nodes = np.random.choice(graph.number_of_nodes(), 
                                      size=min(5, graph.number_of_nodes()), 
                                      replace=False).tolist()
    
    analysis = {
        'k': k,
        'khop_counts_tensor': khop_counts,
        'sample_analyses': []
    }
    
    for node in sample_nodes:
        if node >= graph.number_of_nodes():
            continue
            
        # Get k-hop neighbors manually
        khop_neighbors = get_khop_neighbors_manual(graph, node, k)
        
        # Count communities manually
        manual_counts = {}
        for neighbor in khop_neighbors:
            if neighbor < len(community_labels):
                comm = community_labels[neighbor]
                manual_counts[comm] = manual_counts.get(comm, 0) + 1
        
        # Get tensor counts for this node
        tensor_counts = khop_counts[node].numpy()
        
        # Create community mapping for display
        community_breakdown = {}
        for i, count in enumerate(tensor_counts):
            if count > 0:
                original_community = communities[i] if i < len(communities) else i
                community_breakdown[original_community] = count
        
        node_analysis = {
            'node': node,
            'node_community': communities[community_labels[node]] if node < len(community_labels) else 'Unknown',
            'khop_neighbors': khop_neighbors,
            'neighbor_communities': [community_labels[n] for n in khop_neighbors if n < len(community_labels)],
            'manual_counts': manual_counts,
            'tensor_counts': tensor_counts.tolist(),
            'community_breakdown': community_breakdown,
            'total_khop_neighbors': len(khop_neighbors)
        }
        
        analysis['sample_analyses'].append(node_analysis)
    
    return analysis


def visualize_graph_and_khop(graph_sample: GraphSample, k: int = 1, figsize: Tuple[int, int] = (15, 10)):
    """
    Create comprehensive visualization of graph with communities and k-hop analysis.
    """
    graph = graph_sample.graph
    community_labels = graph_sample.community_labels
    communities = graph_sample.communities
    
    # Get k-hop analysis
    analysis = analyze_khop_counts(graph_sample, k)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main graph visualization
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Create layout
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Create color map for communities
    unique_communities = np.unique(community_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {comm: colors[i] for i, comm in enumerate(unique_communities)}
    
    # Draw nodes colored by community
    node_colors = [community_color_map[community_labels[node]] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=200, alpha=0.8, ax=ax1)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5, ax=ax1)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax1)
    
    # Create legend for communities
    legend_elements = []
    for comm in unique_communities:
        original_comm = communities[comm] if comm < len(communities) else comm
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=community_color_map[comm], 
                                        markersize=10, label=f'Community {original_comm}'))
    ax1.legend(handles=legend_elements, loc='upper right')
    
    ax1.set_title(f'Graph with Communities (n={graph.number_of_nodes()}, m={graph.number_of_edges()})')
    ax1.axis('off')
    
    # K-hop analysis text
    ax2 = plt.subplot(2, 2, 3)
    ax2.axis('off')
    
    # Print detailed analysis for sample nodes
    text_lines = [f"K-HOP COMMUNITY COUNTING ANALYSIS (k={k})", "=" * 50, ""]
    
    for node_analysis in analysis['sample_analyses']:
        node = node_analysis['node']
        node_comm = node_analysis['node_community']
        khop_neighbors = node_analysis['khop_neighbors']
        manual_counts = node_analysis['manual_counts']
        tensor_counts = node_analysis['tensor_counts']
        community_breakdown = node_analysis['community_breakdown']
        
        text_lines.append(f"NODE {node} (Community {node_comm}):")
        text_lines.append(f"  {k}-hop neighbors: {khop_neighbors}")
        text_lines.append(f"  Total {k}-hop neighbors: {len(khop_neighbors)}")
        text_lines.append(f"  Manual community counts: {manual_counts}")
        text_lines.append(f"  Tensor counts: {tensor_counts}")
        text_lines.append(f"  Community breakdown: {community_breakdown}")
        
        # Verify correctness
        manual_total = sum(manual_counts.values())
        tensor_total = sum(tensor_counts)
        match = "✓" if manual_total == tensor_total else "✗"
        text_lines.append(f"  Verification: {match} (manual={manual_total}, tensor={tensor_total})")
        text_lines.append("")
    
    # Display text
    ax2.text(0.05, 0.95, '\n'.join(text_lines), transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # Summary statistics
    ax3 = plt.subplot(2, 2, 4)
    ax3.axis('off')
    
    # Graph statistics
    summary_lines = [
        "GRAPH STATISTICS:",
        "=" * 30,
        f"Nodes: {graph.number_of_nodes()}",
        f"Edges: {graph.number_of_edges()}",
        f"Density: {nx.density(graph):.3f}",
        f"Communities: {len(unique_communities)}",
        f"Community mapping: {dict(zip(unique_communities, [communities[c] for c in unique_communities]))}",
        "",
        "COMMUNITY SIZES:",
    ]
    
    # Community size distribution
    for comm in unique_communities:
        size = np.sum(community_labels == comm)
        original_comm = communities[comm] if comm < len(communities) else comm
        summary_lines.append(f"  Community {original_comm}: {size} nodes")
    
    summary_lines.extend([
        "",
        f"K-HOP ANALYSIS (k={k}):",
        f"  Sample nodes analyzed: {len(analysis['sample_analyses'])}",
        f"  Tensor shape: {analysis['khop_counts_tensor'].shape}",
    ])
    
    ax3.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig, analysis


def run_khop_test(n_graphs: int = 2, k_values: List[int] = [1, 2], seed: int = 42):
    """
    Run comprehensive test of k-hop community counting.
    """
    print("Testing K-hop Community Counting Implementation")
    print("=" * 60)
    
    # Create universe
    universe = create_test_universe(K=4, feature_dim=8, seed=seed)
    print(f"Created universe with {universe.K} communities")
    
    for graph_idx in range(n_graphs):
        print(f"\nTesting Graph {graph_idx + 1}/{n_graphs}")
        print("-" * 40)
        
        # Create graph
        graph_sample = create_test_graph(universe, n_nodes=30, seed=seed + graph_idx)
        print(f"Created graph with {graph_sample.n_nodes} nodes and {graph_sample.graph.number_of_edges()} edges")
        print(f"Communities in graph: {graph_sample.communities}")
        
        for k in k_values:
            print(f"\nAnalyzing {k}-hop community counting...")
            
            # Create visualization
            fig, analysis = visualize_graph_and_khop(graph_sample, k=k, figsize=(16, 12))
            
            # Set title
            fig.suptitle(f'Graph {graph_idx + 1}: {k}-hop Community Counting Test', fontsize=14, fontweight='bold')
            
            plt.show()
            
            # Print verification summary
            print(f"\nVerification Summary for {k}-hop counting:")
            all_correct = True
            for node_analysis in analysis['sample_analyses']:
                manual_total = sum(node_analysis['manual_counts'].values())
                tensor_total = sum(node_analysis['tensor_counts'])
                is_correct = manual_total == tensor_total
                all_correct &= is_correct
                status = "✓" if is_correct else "✗"
                print(f"  Node {node_analysis['node']}: {status} (manual={manual_total}, tensor={tensor_total})")
            
            overall_status = "✓ PASSED" if all_correct else "✗ FAILED"
            print(f"  Overall: {overall_status}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the test
    print("Starting K-hop Community Counting Visualization Test")
    print("This will generate graphs and show detailed k-hop analysis.")
    print("Check the visualizations and verification results carefully!\n")
    
    # Test with different k values
    run_khop_test(n_graphs=2, k_values=[1, 2], seed=42)
    
    print("\nTest completed! Review the visualizations above to verify correctness.")