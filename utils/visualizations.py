"""
Visualization utilities for Stochastic Block Model graphs and communities.

This module provides functions to visualize graphs with community structure,
node embeddings, and other graph properties.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def visualize_graph_generation_process(
    graph: nx.Graph,
    community_labels: np.ndarray,
    P_sub: np.ndarray,
    communities: List[int],
    enforce_connectivity: bool = True,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Visualize the graph generation process with and without connectivity enforcement.
    
    Args:
        graph: The final NetworkX graph
        community_labels: Node community assignments (one-hot encoded)
        P_sub: Community edge probability submatrix
        communities: Community IDs
        enforce_connectivity: Whether connectivity was enforced
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing the generation process
    """
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Community probability matrix
    im = axs[0, 0].imshow(P_sub, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[0, 0], label="Edge Probability")
    
    # Set labels
    axs[0, 0].set_xticks(np.arange(len(communities)))
    axs[0, 0].set_yticks(np.arange(len(communities)))
    axs[0, 0].set_xticklabels([f"C{c}" for c in communities], rotation=90)
    axs[0, 0].set_yticklabels([f"C{c}" for c in communities])
    axs[0, 0].set_title("Community Probability Matrix")
    
    # Plot 2: Community assignments
    im2 = axs[0, 1].imshow(community_labels, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im2, ax=axs[0, 1], label="Community Assignment")
    
    # Set labels
    axs[0, 1].set_xlabel("Community Index")
    axs[0, 1].set_ylabel("Node Index")
    axs[0, 1].set_title("Node-Community Assignments")
    
    # Plot 3: Graph visualization (original SBM model without connectivity enforcement)
    # This requires recomputing the graph without connectivity enforcement
    # Create a copy of the graph data
    n_nodes = graph.number_of_nodes()
    
    # Compute edge probabilities
    edge_probabilities = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            ci = np.argmax(community_labels[i])
            cj = np.argmax(community_labels[j])
            edge_probabilities[i, j] = P_sub[ci, cj]
            edge_probabilities[j, i] = P_sub[ci, cj]
    
    # Threshold at mean probability to create a "typical" graph
    mean_prob = edge_probabilities[np.triu_indices(n_nodes, 1)].mean()
    threshold_graph = nx.Graph()
    threshold_graph.add_nodes_from(range(n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if edge_probabilities[i, j] > mean_prob:
                threshold_graph.add_edge(i, j)
    
    # Position nodes
    pos = nx.spring_layout(graph, seed=42)
    
    # Get node colors by community
    node_colors = []
    cmap = plt.get_cmap("tab20")
    
    for i in range(n_nodes):
        comm = np.argmax(community_labels[i])
        node_colors.append(cmap(comm % cmap.N))
    
    # Draw the original model graph
    nx.draw_networkx_nodes(
        threshold_graph,
        pos,
        ax=axs[1, 0],
        node_color=node_colors,
        node_size=50,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        threshold_graph,
        pos,
        ax=axs[1, 0],
        alpha=0.3,
        width=0.5
    )
    axs[1, 0].axis("off")
    
    # Add statistics
    components = list(nx.connected_components(threshold_graph))
    components.sort(key=len, reverse=True)
    isolated_nodes = sum(1 for _, degree in threshold_graph.degree() if degree == 0)
    
    stats_text = [
        f"Components: {len(components)}",
        f"Largest Component: {len(components[0]) if components else 0} nodes",
        f"Isolated Nodes: {isolated_nodes}"
    ]
    
    axs[1, 0].text(
        0.05, 0.05, "\n".join(stats_text),
        transform=axs[1, 0].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )
    
    axs[1, 0].set_title("Without Connectivity Enforcement")
    
    # Plot 4: Final graph with connectivity enforcement (if applied)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=axs[1, 1],
        node_color=node_colors,
        node_size=50,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=axs[1, 1],
        alpha=0.3,
        width=0.5
    )
    axs[1, 1].axis("off")
    
    # Add statistics
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    isolated_nodes = sum(1 for _, degree in graph.degree() if degree == 0)
    
    stats_text = [
        f"Components: {len(components)}",
        f"Largest Component: {len(components[0]) if components else 0} nodes",
        f"Isolated Nodes: {isolated_nodes}"
    ]
    
    axs[1, 1].text(
        0.05, 0.05, "\n".join(stats_text),
        transform=axs[1, 1].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )
    
    title = "With Connectivity Enforcement" if enforce_connectivity else "Final Graph"
    axs[1, 1].set_title(title)
    
    # Add overall title
    fig.suptitle("SBM Graph Generation Process", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    return fig

def plot_graph_communities(
    graph: Union[nx.Graph, 'GraphSample'],
    community_key: str = "community",
    layout: str = "spring",
    node_size: float = 50,
    edge_width: float = 0.5,
    edge_alpha: float = 0.2,
    figsize: Tuple[int, int] = (10, 8),
    with_labels: bool = False,
    cmap: str = "tab20",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    pos: Optional[Dict] = None,
    min_component_size: int = 0  # New parameter
) -> plt.Figure:
    """
    Plot a graph with nodes colored by community.
    Only shows components that meet the minimum size requirement.
    
    Args:
        graph: NetworkX graph or GraphSample object to plot
        community_key: Node attribute key for community assignment
        layout: Graph layout algorithm ("spring", "kamada_kawai", "spectral", etc.)
        node_size: Size of nodes
        edge_width: Width of edges
        edge_alpha: Edge transparency
        figsize: Figure size
        with_labels: Whether to show node labels
        cmap: Colormap for communities
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        pos: Pre-computed node positions (if None, computes based on layout)
        min_component_size: Minimum size for a component to be kept
        
    Returns:
        Matplotlib figure
    """
    # Handle GraphSample objects
    if hasattr(graph, 'graph'):
        # Get the NetworkX graph and community information
        nx_graph = graph.graph
        community_labels = graph.community_labels
        communities = graph.communities
        
        # Add community information to nodes
        for i, node in enumerate(nx_graph.nodes()):
            nx_graph.nodes[node][community_key] = communities[community_labels[i]]
    else:
        nx_graph = graph
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get components and filter by size
    components = list(nx.connected_components(nx_graph))
    components.sort(key=len, reverse=True)
    kept_components = [c for c in components if len(c) >= min_component_size]
    
    if not kept_components:
        ax.text(0.5, 0.5, "No components above size threshold", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Create subgraph of kept components
    kept_nodes = set().union(*kept_components)
    kept_graph = nx_graph.subgraph(kept_nodes).copy()
    
    # Get communities from kept graph
    communities = {}
    for node, attrs in kept_graph.nodes(data=True):
        if community_key in attrs:
            comm = attrs[community_key]
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
    
    # If community information not found
    if not communities:
        communities = {0: list(kept_graph.nodes())}
    
    # Compute layout if not provided
    if pos is None:
        if layout == "spring":
            pos = nx.spring_layout(kept_graph, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(kept_graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(kept_graph)
        elif layout == "circular":
            pos = nx.circular_layout(kept_graph)
        else:
            # Default to spring
            pos = nx.spring_layout(kept_graph, seed=42)
    else:
        # Filter pre-computed positions to only kept nodes
        pos = {node: pos[node] for node in kept_graph.nodes()}
    
    # Create colormap
    cmap_obj = plt.get_cmap(cmap)
    
    # Draw edges first
    nx.draw_networkx_edges(
        kept_graph,
        pos,
        alpha=edge_alpha,
        width=edge_width,
        ax=ax
    )
    
    # Draw nodes for each community
    for i, (comm, nodes) in enumerate(communities.items()):
        color = cmap_obj(i % cmap_obj.N)
        nx.draw_networkx_nodes(
            kept_graph,
            pos,
            nodelist=nodes,
            node_color=[color] * len(nodes),
            node_size=node_size,
            alpha=0.8,
            ax=ax
        )
    
    # Add node labels if requested
    if with_labels:
        nx.draw_networkx_labels(
            kept_graph,
            pos,
            font_size=8,
            ax=ax
        )
    
    # Set title
    if title:
        if min_component_size > 0:
            title = f"{title}\n(Components ≥ {min_component_size} nodes)"
        ax.set_title(title)
    elif min_component_size > 0:
        ax.set_title(f"Graph Components ≥ {min_component_size} nodes")
    
    # Turn off axis
    ax.axis("off")
    
    # Add legend for communities
    for i, comm in enumerate(sorted(communities.keys())):
        color = cmap_obj(i % cmap_obj.N)
        ax.scatter([], [], c=[color], label=f"Community {comm}")
    
    ax.legend(loc="best", title="Communities")
    
    return fig

def plot_membership_matrix(
    graph: Union[nx.Graph, 'GraphSample'],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    title: str = "Community Distribution",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a visualization of community distribution.
    
    Args:
        graph: GraphSample object or NetworkX graph
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get community information
    if hasattr(graph, 'graph'):
        # GraphSample object
        community_labels = graph.community_labels
        communities = graph.communities
    else:
        # NetworkX graph
        community_labels = []
        communities = set()
        for node in graph.nodes():
            comm = graph.nodes[node].get('community', 0)
            community_labels.append(comm)
            communities.add(comm)
        community_labels = np.array(community_labels)
        communities = sorted(list(communities))

    # Count nodes in each community
    community_counts = {}
    for comm in communities:
        count = np.sum(community_labels == communities.index(comm))
        community_counts[comm] = count

    # Create bar plot
    communities_list = sorted(community_counts.keys())
    counts = [community_counts[comm] for comm in communities_list]
    
    bars = ax.bar(range(len(communities_list)), counts, alpha=0.7)
    
    # Add value labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), 
                horizontalalignment='center',
                verticalalignment='bottom')

    # Customize plot
    ax.set_xticks(range(len(communities_list)))
    ax.set_xticklabels([f'C{c}' for c in communities_list])
    ax.set_xlabel('Community')
    ax.set_ylabel('Number of Nodes')
    
    # Add percentage labels
    total_nodes = sum(counts)
    percentages = [count/total_nodes * 100 for count in counts]
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        ax.text(i, count/2, f'{percentage:.1f}%', 
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if count > max(counts)/3 else 'black')

    # Add total nodes info
    ax.text(0.98, 0.98, f'Total Nodes: {total_nodes}',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    if title:
        ax.set_title(title)

    return fig

def plot_community_matrix(
    P: np.ndarray,
    communities: List[int],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    title: str = "Community Probability Matrix",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the community probability matrix.
    
    Args:
        P: Probability matrix
        communities: List of community IDs
        figsize: Figure size
        cmap: Colormap to use
        title: Figure title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Edge Probability")
    
    ax.set_xticks(np.arange(len(communities)))
    ax.set_yticks(np.arange(len(communities)))
    ax.set_xticklabels([f"C{c}" for c in communities], rotation=90)
    ax.set_yticklabels([f"C{c}" for c in communities])
    
    if title:
        ax.set_title(title)
    
    return fig

def plot_degree_distribution(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (14, 6),
    bins: int = 30,
    color: str = "blue",
    title: str = "Degree Distribution",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the degree distribution of a graph as a two-panel figure:
    - Left: linear scale
    - Right: log-log scale (both axes, with log-spaced bins)
    Args:
        graph: NetworkX graph
        figsize: Figure size
        bins: Number of histogram bins
        color: Color for the histogram
        title: Figure title
        ax: Optional axes to plot on (ignored, always creates new figure)
    Returns:
        Matplotlib figure
    """
    degrees = np.array([d for _, d in graph.degree()])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Linear scale
    ax1.hist(degrees, bins=bins, color=color, alpha=0.7)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Count")
    ax1.set_title("Linear Scale")

    # Log-log scale with log-spaced bins
    min_deg = degrees[degrees > 0].min() if np.any(degrees > 0) else 1
    max_deg = degrees.max() if degrees.max() > min_deg else min_deg + 1
    if min_deg == max_deg:
        log_bins = np.logspace(np.log10(min_deg), np.log10(min_deg + 1), bins)
    else:
        log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bins)
    ax2.hist(degrees, bins=log_bins, color=color, alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Degree (log)")
    ax2.set_ylabel("Count (log)")
    ax2.set_title("Log-Log Scale (Log Bins)")

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_community_size_distribution(
    community_labels: np.ndarray,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 20,
    color: str = "green",
    title: str = "Community Size Distribution",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the distribution of community sizes.
    
    Args:
        community_labels: Node community assignments (one-hot encoded)
        log_scale: Whether to use log scale
        figsize: Figure size
        bins: Number of histogram bins
        color: Color for the histogram
        title: Figure title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get community sizes
    community_sizes = np.sum(community_labels, axis=0)
    
    if log_scale:
        ax.hist(community_sizes, bins=bins, color=color, alpha=0.7, log=True)
        ax.set_xscale("log")
    else:
        ax.hist(community_sizes, bins=bins, color=color, alpha=0.7)
    
    ax.set_xlabel("Community Size")
    ax.set_ylabel("Count")
    
    if title:
        ax.set_title(title)
    
    return fig

def create_dashboard(
    graph: Union[nx.Graph, 'GraphSample'],
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create a comprehensive dashboard for a graph sample.
    
    Args:
        graph: NetworkX graph or GraphSample object
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with a grid of subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # Graph with communities
    ax1 = fig.add_subplot(gs[0, 0])
    plot_graph_communities(graph, ax=ax1, title="Graph with Communities")
    
    # Community distribution
    ax2 = fig.add_subplot(gs[0, 1])
    plot_membership_matrix(graph, ax=ax2)
    
    # Degree distribution
    ax3 = fig.add_subplot(gs[0, 2])
    plot_degree_distribution(graph.graph if hasattr(graph, 'graph') else graph, ax=ax3)
    
    # Community size distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(graph, 'community_labels'):
        plot_community_size_distribution(graph.community_labels, ax=ax4)
    
    # Feature analysis if available
    ax5 = fig.add_subplot(gs[1, 1])
    if hasattr(graph, 'features') and graph.features is not None:
        plot_feature_heatmap(graph.features, graph.community_labels, ax=ax5)
    else:
        ax5.text(0.5, 0.5, "No features available",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax5.transAxes)
    
    fig.tight_layout()
    return fig

def plot_probability_matrix_comparison(
    universe_P: np.ndarray,
    randomized_P: np.ndarray,
    communities: List[int],
    figsize: Tuple[int, int] = (15, 6),
    cmap: str = "viridis",
    title: str = "Edge Probability Matrix Comparison"
) -> plt.Figure:
    """
    Plot a comparison of the base and randomized probability matrices.
    
    Args:
        universe_P: The base probability matrix
        randomized_P: The randomized probability matrix 
        communities: List of community IDs to include
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Extract the submatrices for the selected communities
    P_base = np.zeros((len(communities), len(communities)))
    P_rand = np.zeros((len(communities), len(communities)))
    P_diff = np.zeros((len(communities), len(communities)))
    
    for i, ci in enumerate(communities):
        for j, cj in enumerate(communities):
            P_base[i, j] = universe_P[ci, cj]
            P_rand[i, j] = randomized_P[ci, cj]
            P_diff[i, j] = randomized_P[ci, cj] - universe_P[ci, cj]
    
    # Plot base matrix
    im1 = ax1.imshow(P_base, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title("Base Matrix")
    fig.colorbar(im1, ax=ax1, label="Edge Probability")
    
    # Set labels
    ax1.set_xticks(np.arange(len(communities)))
    ax1.set_yticks(np.arange(len(communities)))
    ax1.set_xticklabels([f"C{c}" for c in communities], rotation=45, ha="right")
    ax1.set_yticklabels([f"C{c}" for c in communities])
    
    # Plot randomized matrix
    im2 = ax2.imshow(P_rand, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title("Randomized Matrix")
    fig.colorbar(im2, ax=ax2, label="Edge Probability")
    
    # Set labels
    ax2.set_xticks(np.arange(len(communities)))
    ax2.set_yticks(np.arange(len(communities)))
    ax2.set_xticklabels([f"C{c}" for c in communities], rotation=45, ha="right")
    ax2.set_yticklabels([f"C{c}" for c in communities])
    
    # Plot difference matrix
    max_diff = max(abs(P_diff.min()), abs(P_diff.max()))
    im3 = ax3.imshow(P_diff, cmap="coolwarm", vmin=-max_diff, vmax=max_diff)
    ax3.set_title("Difference")
    fig.colorbar(im3, ax=ax3, label="Difference")
    
    # Set labels
    ax3.set_xticks(np.arange(len(communities)))
    ax3.set_yticks(np.arange(len(communities)))
    ax3.set_xticklabels([f"C{c}" for c in communities], rotation=45, ha="right")
    ax3.set_yticklabels([f"C{c}" for c in communities])
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def plot_connectivity_analysis(
    graph: nx.Graph,
    min_component_size: int = 0,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Analyze and visualize connectivity properties of a graph.
    Shows which components were removed due to size threshold.
    
    Args:
        graph: NetworkX graph to analyze
        min_component_size: Minimum size for a component to be kept
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Get connected components
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    
    # Separate kept and removed components
    kept_components = [c for c in components if len(c) >= min_component_size]
    removed_components = [c for c in components if len(c) < min_component_size]
    
    # Plot 1: Component size distribution with removed components highlighted
    component_sizes = [len(c) for c in components]
    
    if len(component_sizes) > 1:
        # Create bars for all components
        bars = ax1.bar(range(len(component_sizes)), component_sizes, alpha=0.7)
        
        # Color the bars based on whether component was kept or removed
        for i, bar in enumerate(bars):
            if len(components[i]) < min_component_size:
                bar.set_color('red')
                bar.set_alpha(0.5)
            else:
                bar.set_color('blue')
                bar.set_alpha(0.7)
        
        ax1.axhline(y=min_component_size, color='r', linestyle='--', alpha=0.5, 
                    label=f'Size threshold ({min_component_size})')
        
        # Add legend
        if min_component_size > 0:
            ax1.legend(['Size threshold', 'Kept components', 'Removed components'])
            
        ax1.set_xlabel("Component Rank")
        ax1.set_ylabel("Size")
        ax1.set_title(f"Component Analysis\nKept: {len(kept_components)}, Removed: {len(removed_components)}")
    else:
        ax1.text(0.5, 0.5, "Graph is fully connected\n(1 component)", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title("Connected Components")
        ax1.axis('off')
    
    # Plot 2: Degree distribution (only for kept components)
    if kept_components:
        # Create subgraph of kept components
        kept_nodes = set().union(*kept_components)
        kept_graph = graph.subgraph(kept_nodes)
        degrees = [d for _, d in kept_graph.degree()]
    else:
        degrees = []
        
    if degrees:
        bins = max(10, min(30, max(degrees) - min(degrees) + 1))
        ax2.hist(degrees, bins=bins, alpha=0.7, color='blue')
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count")
        ax2.set_title("Degree Distribution\n(Kept Components)")
        
        # Add statistics
        min_degree = min(degrees)
        max_degree = max(degrees)
        avg_degree = sum(degrees) / len(degrees)
        
        stats_text = f"Min: {min_degree}\nMax: {max_degree}\nAvg: {avg_degree:.2f}"
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, "No components above\nsize threshold", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.axis('off')
    
    # Plot 3: Distance distribution (only for kept components)
    if kept_components:
        kept_graph = graph.subgraph(kept_nodes)
        
        # For large graphs, sample node pairs for efficiency
        if kept_graph.number_of_nodes() > 500:
            sample_size = 500
            nodes = list(kept_graph.nodes())
            sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False)
            
            distances = []
            for i, u in enumerate(sampled_nodes):
                for v in sampled_nodes[i+1:]:
                    try:
                        dist = nx.shortest_path_length(kept_graph, u, v)
                        distances.append(dist)
                    except nx.NetworkXNoPath:
                        pass
        else:
            # Compute all shortest paths in kept components
            try:
                path_lengths = dict(nx.all_pairs_shortest_path_length(kept_graph))
                distances = [path_lengths[u][v] for u in path_lengths for v in path_lengths[u] if u != v]
            except:
                distances = []
        
        if distances:
            bins = max(5, min(20, max(distances) - min(distances) + 1))
            ax3.hist(distances, bins=bins, alpha=0.7, color='blue')
            ax3.set_xlabel("Shortest Path Length")
            ax3.set_ylabel("Count")
            ax3.set_title("Distance Distribution\n(Kept Components)")
            
            # Add statistics
            diameter = max(distances)
            avg_path = sum(distances) / len(distances)
            
            stats_text = f"Diameter: {diameter}\nAvg Path: {avg_path:.2f}"
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, "Cannot compute distances\n(disconnected or too large)", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes, fontsize=12)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, "No components above\nsize threshold", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=14)
        ax3.axis('off')
    
    fig.tight_layout()
    return fig

def plot_community_probability_heatmap(
    P: np.ndarray,
    randomness_factor: float = 0.0,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    sample_size: int = 20
) -> plt.Figure:
    """
    Create a heatmap of community edge probabilities with examples of randomization.
    
    Args:
        P: Original probability matrix
        randomness_factor: Amount of randomness to show in examples
        figsize: Figure size
        cmap: Colormap
        sample_size: Number of communities to sample for visualization
        
    Returns:
        Matplotlib figure
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Sample communities if the matrix is too large
    if P.shape[0] > sample_size:
        indices = np.random.choice(P.shape[0], size=sample_size, replace=False)
        indices = np.sort(indices)
        P_sample = P[np.ix_(indices, indices)]
    else:
        indices = np.arange(P.shape[0])
        P_sample = P
    
    # Plot original matrix
    im0 = axs[0].imshow(P_sample, cmap=cmap, vmin=0, vmax=1)
    axs[0].set_title("Original")
    
    # Create first example of randomization
    if randomness_factor > 0:
        noise1 = np.random.uniform(-randomness_factor/2, randomness_factor/2, size=P_sample.shape)
        P_rand1 = P_sample * (1 + noise1)
        P_rand1 = np.clip(P_rand1, 0, 1)
    else:
        # Create a small amount of noise for demonstration
        noise1 = np.random.uniform(-0.1, 0.1, size=P_sample.shape)
        P_rand1 = P_sample * (1 + noise1)
        P_rand1 = np.clip(P_rand1, 0, 1)
    
    # Create second example with more randomness
    noise2 = np.random.uniform(-max(randomness_factor, 0.2), max(randomness_factor, 0.2), size=P_sample.shape)
    P_rand2 = P_sample * (1 + noise2)
    P_rand2 = np.clip(P_rand2, 0, 1)
    
    # Plot randomized matrices
    im1 = axs[1].imshow(P_rand1, cmap=cmap, vmin=0, vmax=1)
    im2 = axs[2].imshow(P_rand2, cmap=cmap, vmin=0, vmax=1)
    
    rf_value = randomness_factor if randomness_factor > 0 else 0.1
    axs[1].set_title(f"Low Randomness ({rf_value:.1f})")
    axs[2].set_title(f"High Randomness ({max(randomness_factor, 0.2):.1f})")
    
    # Set labels and ticks for all plots
    for i, ax in enumerate(axs):
        ax.set_xticks(np.arange(len(indices)))
        ax.set_yticks(np.arange(len(indices)))
        ax.set_xticklabels([f"C{idx}" for idx in indices], rotation=90)
        ax.set_yticklabels([f"C{idx}" for idx in indices])
    
    # Add colorbar
    fig.colorbar(im0, ax=axs, label="Edge Probability")
    
    fig.suptitle("Community Probability Matrix with Randomization Examples", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def compare_graph_structures(
    graphs: List[nx.Graph],
    titles: List[str],
    community_key: str = "primary_community",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Compare multiple graph structures side by side.
    
    Args:
        graphs: List of NetworkX graphs to compare
        titles: Titles for each graph
        community_key: Node attribute key for community assignment
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_graphs = len(graphs)
    fig, axs = plt.subplots(2, n_graphs, figsize=figsize)
    
    # If there's only one graph, make sure axs is still a 2D array
    if n_graphs == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    
    # First row: Graph visualization
    for i, (graph, title) in enumerate(zip(graphs, titles)):
        # Get positions
        if graph.number_of_nodes() > 1000:
            # For large graphs, use faster layout
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        # Get communities from graph
        communities = {}
        for node, attrs in graph.nodes(data=True):
            if community_key in attrs:
                comm = attrs[community_key]
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)
        
        # If community information not found
        if not communities:
            communities = {0: list(graph.nodes())}
        
        # Create colormap
        cmap = plt.get_cmap("tab20")
        
        # Draw edges first
        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.2,
            width=0.5,
            ax=axs[0, i]
        )
        
        # Draw nodes for each community
        for j, (comm, nodes) in enumerate(communities.items()):
            color = cmap(j % cmap.N)
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=[color] * len(nodes),
                node_size=30,
                alpha=0.8,
                ax=axs[0, i]
            )
        
        # Set title and turn off axis
        axs[0, i].set_title(title)
        axs[0, i].axis("off")
    
    # Second row: Graph statistics
    for i, (graph, title) in enumerate(zip(graphs, titles)):
        # Calculate statistics
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        # Degree statistics
        degrees = [d for _, d in graph.degree()]
        avg_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        # Connected components
        n_components = nx.number_connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_cc_size = len(largest_cc)
        largest_cc_pct = largest_cc_size / n_nodes * 100 if n_nodes > 0 else 0
        
        # Calculate average clustering coefficient
        try:
            avg_clustering = nx.average_clustering(graph)
        except:
            avg_clustering = 0
        
        # Plot degree distribution
        axs[1, i].hist(degrees, bins=20, alpha=0.7)
        axs[1, i].set_xlabel("Degree")
        axs[1, i].set_ylabel("Count")
        axs[1, i].set_title(f"Degree Distribution")
        
        # Add statistics as text
        stats = [
            f"Nodes: {n_nodes}",
            f"Edges: {n_edges}",
            f"Avg Degree: {avg_degree:.2f}",
            f"Components: {n_components}",
            f"Largest Comp: {largest_cc_pct:.1f}%",
            f"Clustering: {avg_clustering:.3f}"
        ]
        
        stats_text = "\n".join(stats)
        axs[1, i].text(
            0.95, 0.95, stats_text,
            transform=axs[1, i].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )
    
    fig.tight_layout()
    return fig



def plot_community_graph(
    universe_P: np.ndarray,
    communities: List[int],
    figsize: Tuple[int, int] = (10, 8),
    edge_threshold: float = 0.01,
    node_size_factor: float = 100,
    edge_width_factor: float = 5,
    layout: str = "spring",
    cmap: str = "tab20",
    title: str = "Community Interaction Graph",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a graph of communities with edge weights based on connectivity.
    
    Args:
        universe_P: The full probability matrix from the universe
        communities: List of community IDs to include
        figsize: Figure size
        edge_threshold: Minimum edge weight to include
        node_size_factor: Scaling factor for node sizes
        edge_width_factor: Scaling factor for edge widths
        layout: Graph layout algorithm
        cmap: Colormap
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract the submatrix for the selected communities
    P_sub = np.zeros((len(communities), len(communities)))
    for i, ci in enumerate(communities):
        for j, cj in enumerate(communities):
            P_sub[i, j] = universe_P[ci, cj]
    
    # Create community graph
    G = nx.Graph()
    
    # Add nodes (communities)
    for i, comm in enumerate(communities):
        # Node size based on self-connection probability
        size = P_sub[i, i] * node_size_factor
        G.add_node(i, community=comm, size=max(50, size))
    
    # Add edges
    for i in range(len(communities)):
        for j in range(i+1, len(communities)):
            weight = P_sub[i, j]
            if weight > edge_threshold:
                G.add_edge(i, j, weight=weight)
    
    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, weight='weight')
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight='weight')
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, weight='weight')
    
    # Get node sizes
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    
    # Get edge weights for width
    edge_weights = [G[u][v]['weight'] * edge_width_factor for u, v in G.edges]
    
    # Create colormap
    cmap_obj = plt.get_cmap(cmap)
    node_colors = [cmap_obj(i % cmap_obj.N) for i in range(len(communities))]
    
    # Draw the network
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={i: f"C{communities[i]}" for i in G.nodes},
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_weights,
        edge_color='gray',
        alpha=0.8,
        ax=ax
    )
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Turn off axis
    ax.axis("off")
    
    return fig


def plot_node_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "tab20",
    marker_size: int = 50,
    title: str = "Node Embeddings",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot node embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Node embedding matrix (n_nodes × dim)
        labels: Node labels for coloring (if None, no coloring)
        method: Dimensionality reduction method ("tsne", "pca")
        figsize: Figure size
        cmap: Colormap for labels
        marker_size: Size of markers
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Apply dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    # For empty or 1-node embeddings, handle specially
    if embeddings.shape[0] <= 1 or embeddings.shape[1] == 0:
        embeddings_2d = np.zeros((embeddings.shape[0], 2))
    else:
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot with or without labels
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap_obj = plt.get_cmap(cmap)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = cmap_obj(i % cmap_obj.N)
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f"Community {label}",
                s=marker_size,
                alpha=0.7
            )
        
        ax.legend(title="Communities")
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            s=marker_size,
            alpha=0.7
        )
    
    # Set title and axis labels
    if title:
        ax.set_title(title)
    
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    
    return fig


def plot_feature_heatmap(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_features_plot: int = 20,
    sort_by_label: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    title: str = "Node Features",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a heatmap of node features.
    
    Args:
        features: Node feature matrix (n_nodes × n_features)
        labels: Node labels for sorting (if None, no sorting)
        n_features_plot: Maximum number of features to plot
        sort_by_label: Whether to sort nodes by label
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Limit number of features if needed
    if features.shape[1] > n_features_plot:
        # Choose features with highest variance
        feature_var = np.var(features, axis=0)
        top_indices = np.argsort(feature_var)[-n_features_plot:]
        plot_features = features[:, top_indices]
    else:
        plot_features = features
    
    # Sort by label if requested
    if labels is not None and sort_by_label:
        sorted_indices = np.argsort(labels)
        plot_features = plot_features[sorted_indices]
        plot_labels = labels[sorted_indices]
    else:
        plot_labels = labels
    
    # Plot heatmap
    im = ax.imshow(
        plot_features,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest"
    )
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label="Feature Value")
    
    # Set labels
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Node Index")
    
    # Add label separators if available
    if plot_labels is not None and sort_by_label:
        unique_labels = np.unique(plot_labels)
        label_boundaries = []
        
        for i in range(1, len(unique_labels)):
            boundary = np.argmax(plot_labels >= unique_labels[i])
            label_boundaries.append(boundary)
            ax.axhline(y=boundary - 0.5, color='red', linestyle='-', linewidth=1)
        
        # Add label annotations
        for i, label in enumerate(unique_labels):
            start = 0 if i == 0 else label_boundaries[i-1]
            end = label_boundaries[i] if i < len(label_boundaries) else len(plot_labels)
            middle = (start + end) / 2
            
            ax.text(
                -0.05 * plot_features.shape[1],
                middle,
                f"C{label}",
                verticalalignment='center',
                horizontalalignment='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
            )
    
    # Set title
    if title:
        ax.set_title(title)
    
    return fig


def plot_community_size_distribution(
    community_labels: np.ndarray,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 20,
    color: str = "green",
    title: str = "Community Size Distribution",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the distribution of community sizes.
    
    Args:
        communities: List of communities, where each community is a list of node indices
        log_scale: Whether to use log scales
        figsize: Figure size
        bins: Number of histogram bins
        color: Bar color
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get community sizes
    sizes = [len(comm) for comm in communities]
    
    # Plot histogram
    ax.hist(sizes, bins=bins, color=color, alpha=0.7)
    
    # Set log scale if requested
    if log_scale and max(sizes) > 10:
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # Set labels and title
    ax.set_xlabel("Community Size")
    ax.set_ylabel("Count")
    
    if title:
        ax.set_title(title)
    
    # Add statistics
    avg_size = np.mean(sizes)
    max_size = np.max(sizes)
    min_size = np.min(sizes)
    
    stats_text = f"Avg: {avg_size:.2f}\nMax: {max_size}\nMin: {min_size}"
    ax.text(
        0.95, 0.95, stats_text,
        verticalalignment='top',
        horizontalalignment='right',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    return fig


def plot_community_overlap_distribution(
    membership_vectors: np.ndarray,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (8, 6),
    color: str = "purple",
    title: str = "Community Overlap Distribution",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the distribution of number of communities per node.
    
    Args:
        membership_vectors: Node-community membership matrix (n_nodes × n_communities)
        threshold: Minimum membership weight to count
        figsize: Figure size
        color: Bar color
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates new)
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Count communities per node
    n_communities = (membership_vectors > threshold).sum(axis=1)
    
    # Get unique counts and their frequencies
    unique_counts, frequencies = np.unique(n_communities, return_counts=True)
    
    # Create bar plot
    ax.bar(unique_counts, frequencies, color=color, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel("Number of Communities per Node")
    ax.set_ylabel("Count")
    
    if title:
        ax.set_title(title)
    
    # Add statistics
    avg_comm = np.mean(n_communities)
    max_comm = np.max(n_communities)
    min_comm = np.min(n_communities)
    
    stats_text = f"Avg: {avg_comm:.2f}\nMax: {max_comm}\nMin: {min_comm}"
    ax.text(
        0.95, 0.95, stats_text,
        verticalalignment='top',
        horizontalalignment='right',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Set x-ticks to integers
    ax.set_xticks(unique_counts)
    
    return fig


def plot_multiple_graphs(
    graphs: List[nx.Graph],
    community_key: str = "primary_community",
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    node_size: float = 30,
    titles: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot multiple graphs in a grid layout.
    
    Args:
        graphs: List of NetworkX graphs to plot
        community_key: Node attribute key for community assignment
        n_cols: Number of columns in the grid
        figsize: Figure size
        node_size: Size of nodes
        titles: List of titles for each graph (if None, uses default titles)
        
    Returns:
        Matplotlib figure
    """
    n_graphs = len(graphs)
    n_rows = (n_graphs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Generate default titles if not provided
    if titles is None:
        titles = [f"Graph {i+1}" for i in range(n_graphs)]
    
    # Plot each graph
    for i, (graph, title) in enumerate(zip(graphs, titles)):
        if i < len(axes):
            plot_graph_communities(
                graph,
                community_key=community_key,
                node_size=node_size,
                title=title,
                ax=axes[i]
            )
    
    # Hide unused axes
    for i in range(n_graphs, len(axes)):
        axes[i].axis('off')
    
    fig.tight_layout()
    return fig


def plot_membership_comparison(
    true_membership: np.ndarray,
    pred_membership: np.ndarray,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "coolwarm",
    title: str = "Community Membership Comparison",
    sample_nodes: Optional[int] = None
) -> plt.Figure:
    """
    Plot a comparison between true and predicted community memberships.
    
    Args:
        true_membership: True membership matrix (n_nodes × n_communities)
        pred_membership: Predicted membership matrix
        threshold: Minimum membership weight to count
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        sample_nodes: Number of nodes to sample (if None, uses all)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Sample nodes if requested
    if sample_nodes is not None and sample_nodes < true_membership.shape[0]:
        indices = np.random.choice(
            true_membership.shape[0],
            size=sample_nodes,
            replace=False
        )
        true_sample = true_membership[indices]
        pred_sample = pred_membership[indices]
    else:
        true_sample = true_membership
        pred_sample = pred_membership
    
    # Plot true memberships
    im1 = ax1.imshow(true_sample, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax1.set_title("True Memberships")
    ax1.set_xlabel("Community")
    ax1.set_ylabel("Node")
    
    # Plot predicted memberships
    im2 = ax2.imshow(pred_sample, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax2.set_title("Predicted Memberships")
    ax2.set_xlabel("Community")
    ax2.set_ylabel("Node")
    
    # Add colorbars
    fig.colorbar(im1, ax=ax1, label="Membership Strength")
    fig.colorbar(im2, ax=ax2, label="Membership Strength")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the overall title
    return fig


def plot_transfer_performance(
    metrics: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot transfer learning performance comparison.
    
    Args:
        metrics: Dictionary of metrics for different training scenarios
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract metrics to plot
    scenarios = list(metrics.keys())
    metric_names = set()
    for scenario in scenarios:
        metric_names.update(metrics[scenario].keys())
    
    metric_names = sorted(list(metric_names))
    
    # Create grouped bar chart
    x = np.arange(len(metric_names))
    width = 0.8 / len(scenarios)
    
    for i, scenario in enumerate(scenarios):
        values = [metrics[scenario].get(metric, 0) for metric in metric_names]
        offset = (i - len(scenarios) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=scenario)
    
    # Set labels and title
    ax.set_ylabel('Value')
    ax.set_title('Transfer Learning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    return fig

"""
Utility functions for visualizing and analyzing the enhanced node features.
These can be added to the visualizations.py file to extend its capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def visualize_feature_subtypes(
    universe,
    communities_to_plot: Optional[List[int]] = None,
    n_communities: int = 5,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize feature subtypes (clusters) and their relationship with communities.
    
    Args:
        universe: GraphUniverse instance
        communities_to_plot: Optional list of community indices to plot
        n_communities: Number of communities to plot if communities_to_plot is None
        figsize: Figure size
        
    Returns:
        plt.Figure: The visualization figure
    """
    if universe.feature_generator is None:
        raise ValueError("Universe has no feature generator")
        
    # Get cluster centers and community-cluster mapping
    cluster_centers = universe.feature_generator._generate_cluster_centers()
    community_cluster_mapping = universe.feature_generator._create_community_cluster_mapping()
    
    # Select communities to plot
    if communities_to_plot is None:
        communities_to_plot = np.random.choice(
            universe.K, 
            min(n_communities, universe.K), 
            replace=False
        )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Cluster Centers
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    centers_2d = pca.fit_transform(cluster_centers)
    
    # Plot cluster centers
    scatter = ax1.scatter(
        centers_2d[:, 0], 
        centers_2d[:, 1],
        c=np.arange(len(centers_2d)),
        cmap='tab20',
        s=100
    )
    
    # Add cluster labels
    for i, (x, y) in enumerate(centers_2d):
        ax1.annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax1.set_title('Cluster Centers (PCA projection)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Plot 2: Community-Cluster Mapping
    # Create a heatmap of the mapping
    mapping_matrix = np.zeros((len(communities_to_plot), len(cluster_centers)))
    for i, comm in enumerate(communities_to_plot):
        mapping_matrix[i] = community_cluster_mapping[comm]
    
    sns.heatmap(
        mapping_matrix,
        ax=ax2,
        cmap='YlOrRd',
        xticklabels=[f'C{i}' for i in range(len(cluster_centers))],
        yticklabels=[f'Comm{i}' for i in communities_to_plot]
    )
    ax2.set_title('Community-Cluster Mapping')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Community')
    
    plt.tight_layout()
    return fig


def visualize_feature_similarity_matrix(
    universe,
    communities_to_plot: Optional[List[int]] = None,
    n_communities: int = 20,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize the feature similarity matrix for selected communities.
    
    Args:
        universe: GraphUniverse object with feature similarity matrix
        communities_to_plot: List of community indices to visualize (if None, selects n_communities at random)
        n_communities: Number of communities to plot if communities_to_plot is None
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not hasattr(universe, 'feature_similarity_matrix') or universe.feature_similarity_matrix is None:
        raise ValueError("Universe does not have feature similarity matrix defined")
    
    # Select communities to plot
    if communities_to_plot is None:
        communities_to_plot = np.random.choice(
            universe.K, size=min(n_communities, universe.K), replace=False
        ).tolist()
        communities_to_plot.sort()  # Sort for better visualization
    
    # Extract the submatrix for the selected communities
    sim_submatrix = np.zeros((len(communities_to_plot), len(communities_to_plot)))
    for i, ci in enumerate(communities_to_plot):
        for j, cj in enumerate(communities_to_plot):
            sim_submatrix[i, j] = universe.feature_similarity_matrix[ci, cj]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(sim_submatrix, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label="Feature Similarity")
    
    # Set labels
    ax.set_xticks(np.arange(len(communities_to_plot)))
    ax.set_yticks(np.arange(len(communities_to_plot)))
    ax.set_xticklabels([f"C{c}" for c in communities_to_plot])
    ax.set_yticklabels([f"C{c}" for c in communities_to_plot])
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add cell values
    for i in range(len(communities_to_plot)):
        for j in range(len(communities_to_plot)):
            text = ax.text(j, i, f"{sim_submatrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if sim_submatrix[i, j] < 0.5 else "black")
    
    ax.set_title("Feature Similarity Matrix")
    fig.tight_layout()
    
    return fig


def visualize_feature_correlations(
    graph,
    n_features_to_plot: int = 10,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Visualize correlations between node features, community memberships, and graph topology.
    
    Args:
        graph: GraphSample object with features
        n_features_to_plot: Number of features to include in visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if graph.features is None:
        raise ValueError("Graph does not have features")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Feature-Membership Correlation Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Select a subset of features for readability
    feature_dim = graph.features.shape[1]
    if feature_dim > n_features_to_plot:
        # Choose features with highest variance
        feature_vars = np.var(graph.features, axis=0)
        top_indices = np.argsort(feature_vars)[-n_features_to_plot:]
        top_indices.sort()  # Sort for better visualization
        features_subset = graph.features[:, top_indices]
        feature_labels = [f"F{i}" for i in top_indices]
    else:
        features_subset = graph.features
        feature_labels = [f"F{i}" for i in range(feature_dim)]
    
    # Calculate correlation matrix between features and memberships
    n_communities = len(graph.communities)
    corr_matrix = np.zeros((n_communities, len(feature_labels)))
    
    for i, comm_idx in enumerate(range(n_communities)):
        for j, feat_idx in enumerate(range(len(feature_labels))):
            if feature_dim > n_features_to_plot:
                actual_feat_idx = top_indices[feat_idx]
            else:
                actual_feat_idx = feat_idx
                
            corr_matrix[i, j] = np.corrcoef(
                graph.membership_vectors[:, comm_idx], 
                graph.features[:, actual_feat_idx]
            )[0, 1]
    
    # Create heatmap
    im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax1, label="Correlation Coefficient")
    
    # Set labels
    ax1.set_xticks(np.arange(len(feature_labels)))
    ax1.set_yticks(np.arange(n_communities))
    ax1.set_xticklabels(feature_labels)
    ax1.set_yticklabels([f"C{c}" for c in graph.communities])
    
    # Rotate x labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax1.set_title("Feature-Community Membership Correlations")
    
    # 2. Node Degree vs Feature Value Scatter Plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get node degrees
    degrees = np.array([d for _, d in graph.graph.degree()])
    
    # Select a representative feature (the one with highest variance)
    if feature_dim > 1:
        feature_vars = np.var(graph.features, axis=0)
        rep_feature_idx = np.argmax(feature_vars)
    else:
        rep_feature_idx = 0
        
    rep_feature = graph.features[:, rep_feature_idx]
    
    # Calculate correlation
    degree_feature_corr = np.corrcoef(degrees, rep_feature)[0, 1]
    
    # Color nodes by primary community
    primary_communities = np.argmax(graph.membership_vectors, axis=1)
    
    scatter = ax2.scatter(
        degrees,
        rep_feature,
        c=primary_communities,
        cmap='tab20',
        alpha=0.7,
        s=50
    )
    
    # Add correlation text
    ax2.text(
        0.05, 0.95, f"Correlation: {degree_feature_corr:.3f}",
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    ax2.set_xlabel("Node Degree")
    ax2.set_ylabel(f"Feature {rep_feature_idx} Value")
    ax2.set_title(f"Node Degree vs Feature {rep_feature_idx}")
    
    # Add colorbar for communities
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Primary Community")
    
    # 3. Feature PCA with Community Coloring
    ax3 = fig.add_subplot(gs[1, 0])
    
    if feature_dim > 2:
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(graph.features)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
    else:
        features_2d = graph.features
        explained_variance = [1.0, 0.0] if feature_dim > 1 else [1.0]
    
    # Color by primary community
    scatter = ax3.scatter(
        features_2d[:, 0],
        features_2d[:, 1] if features_2d.shape[1] > 1 else np.zeros(features_2d.shape[0]),
        c=primary_communities,
        cmap='tab20',
        alpha=0.7,
        s=50
    )
    
    ax3.set_xlabel(f"PCA 1 ({explained_variance[0]:.1%})")
    if features_2d.shape[1] > 1:
        ax3.set_ylabel(f"PCA 2 ({explained_variance[1]:.1%})")
    else:
        ax3.set_ylabel("N/A")
    
    ax3.set_title("Node Features (PCA) Colored by Community")
    
    # Add colorbar for communities
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Primary Community")
    
    # 4. Feature distribution by community
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Select a representative feature (different from previous one for variety)
    if feature_dim > 2:
        remaining_indices = np.argsort(feature_vars)[-3:-1]
        rep_feature_idx2 = remaining_indices[0]
    else:
        rep_feature_idx2 = 0
        
    rep_feature2 = graph.features[:, rep_feature_idx2]
    
    # Prepare data for boxplot
    community_features = []
    community_labels = []
    
    # Limit to top 8 communities for readability
    top_communities = np.bincount(primary_communities).argsort()[-8:]
    
    for comm_idx in top_communities:
        # Get features for nodes with this primary community
        mask = primary_communities == comm_idx
        if np.any(mask):
            community_features.append(rep_feature2[mask])
            comm_label = graph.communities[comm_idx]
            community_labels.append(f"C{comm_label}")
    
    # Create boxplot
    ax4.boxplot(community_features, labels=community_labels)
    
    ax4.set_xlabel("Community")
    ax4.set_ylabel(f"Feature {rep_feature_idx2} Value")
    ax4.set_title(f"Feature {rep_feature_idx2} Distribution by Community")
    
    # Set overall title
    fig.suptitle("Feature Analysis Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the title
    
    return fig


def compare_feature_distributions(
    graphs: List,
    feature_idx: Optional[int] = None,
    n_pca_components: int = 2,
    group_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Compare feature distributions across multiple graphs or graph families.
    
    Args:
        graphs: List of GraphSample objects or lists of GraphSample objects
        feature_idx: Index of specific feature to compare (if None, uses all features with dimensionality reduction)
        n_pca_components: Number of PCA components for dimensionality reduction
        group_labels: Labels for each graph or graph family
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    all_features = []
    all_labels = []
    all_graph_indices = []
    
    for i, item in enumerate(graphs):
        # Handle both individual graphs and lists of graphs
        if isinstance(item, list):
            for j, graph in enumerate(item):
                if graph.features is not None:
                    if feature_idx is not None:
                        # Extract specific feature
                        features = graph.features[:, feature_idx].reshape(-1, 1)
                    else:
                        # Use all features
                        features = graph.features
                        
                    all_features.append(features)
                    # Use provided labels if available, otherwise use index
                    group_label = group_labels[i] if group_labels is not None else f"Group {i+1}"
                    all_labels.extend([group_label] * features.shape[0])
                    all_graph_indices.extend([i] * features.shape[0])
        else:
            graph = item
            if graph.features is not None:
                if feature_idx is not None:
                    # Extract specific feature
                    features = graph.features[:, feature_idx].reshape(-1, 1)
                else:
                    # Use all features
                    features = graph.features
                    
                all_features.append(features)
                # Use provided labels if available, otherwise use index
                group_label = group_labels[i] if group_labels is not None else f"Graph {i+1}"
                all_labels.extend([group_label] * features.shape[0])
                all_graph_indices.extend([i] * features.shape[0])
    
    if not all_features:
        raise ValueError("No features found in provided graphs")
    
    # Combine features
    combined_features = np.vstack(all_features)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. PCA visualization
    ax1 = fig.add_subplot(gs[0, 0])
    
    if combined_features.shape[1] > 2:
        # Apply PCA
        pca = PCA(n_components=min(n_pca_components, combined_features.shape[1]))
        features_reduced = pca.fit_transform(combined_features)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
    else:
        features_reduced = combined_features
        explained_variance = [1.0] * combined_features.shape[1]
    
    # Create a colormap
    unique_labels = list(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each group
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        
        if features_reduced.shape[1] >= 2:
            ax1.scatter(
                features_reduced[mask, 0],
                features_reduced[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=30
            )
            
            # Add confidence ellipse
            if np.sum(mask) > 5:  # Need enough points for an ellipse
                mean_x = np.mean(features_reduced[mask, 0])
                mean_y = np.mean(features_reduced[mask, 1])
                
                cov = np.cov(features_reduced[mask, 0], features_reduced[mask, 1])
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Sort eigenvalues and eigenvectors
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Calculate 95% confidence ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                ellipse_radius_x = np.sqrt(5.991 * eigenvalues[0])  # 95% confidence
                ellipse_radius_y = np.sqrt(5.991 * eigenvalues[1])
                
                ellipse_x = (mean_x + ellipse_radius_x * np.cos(theta) * eigenvectors[0, 0] + 
                             ellipse_radius_y * np.sin(theta) * eigenvectors[0, 1])
                ellipse_y = (mean_y + ellipse_radius_x * np.cos(theta) * eigenvectors[1, 0] + 
                             ellipse_radius_y * np.sin(theta) * eigenvectors[1, 1])
                
                ax1.plot(ellipse_x, ellipse_y, color=colors[i], linestyle='--', alpha=0.5)
        else:
            # For 1D data, create a horizontal scatter
            ax1.scatter(
                features_reduced[mask, 0],
                np.random.normal(i, 0.1, size=np.sum(mask)),  # Jitter for visibility
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=30
            )
    
    if features_reduced.shape[1] >= 2:
        ax1.set_xlabel(f"PCA 1 ({explained_variance[0]:.1%})")
        ax1.set_ylabel(f"PCA 2 ({explained_variance[1]:.1%})")
        ax1.set_title("Feature Distribution Comparison (PCA)")
    else:
        ax1.set_xlabel(f"Feature Value")
        ax1.set_ylabel("Group")
        ax1.set_title("Feature Value Distribution by Group")
    
    ax1.legend()
    
    # 2. t-SNE visualization
    ax2 = fig.add_subplot(gs[0, 1])
    
    if combined_features.shape[1] > 2 and combined_features.shape[0] > 10:
        # Apply t-SNE
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(combined_features)
            
            # Plot each group
            for i, label in enumerate(unique_labels):
                mask = np.array(all_labels) == label
                ax2.scatter(
                    features_tsne[mask, 0],
                    features_tsne[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.7,
                    s=30
                )
                
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")
            ax2.set_title("Feature Distribution Comparison (t-SNE)")
            ax2.legend()
        except Exception as e:
            ax2.text(0.5, 0.5, f"t-SNE failed: {str(e)}", 
                     ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "Not enough data\nfor t-SNE visualization", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Feature distribution histograms
    ax3 = fig.add_subplot(gs[1, 0])
    
    if feature_idx is not None:
        # For a specific feature, show histogram for each group
        for i, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label
            sns.kdeplot(
                combined_features[mask, 0],
                ax=ax3,
                label=label,
                color=colors[i]
            )
            
        ax3.set_xlabel(f"Feature {feature_idx} Value")
        ax3.set_ylabel("Density")
        ax3.set_title(f"Feature {feature_idx} Distribution by Group")
        ax3.legend()
    else:
        # For multiple features, show histogram of feature means
        feature_means = []
        feature_mean_labels = []
        
        for i, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label
            mean_values = np.mean(combined_features[mask], axis=0)
            feature_means.extend(mean_values)
            feature_mean_labels.extend([label] * len(mean_values))
            
        sns.boxplot(
            x=feature_mean_labels, 
            y=feature_means,
            ax=ax3
        )
            
        ax3.set_xlabel("Group")
        ax3.set_ylabel("Feature Mean Value")
        ax3.set_title("Feature Mean Distribution by Group")
        plt.xticks(rotation=45)
    
    # 4. Feature statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Compute statistics for each group
    stats_data = []
    
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        group_features = combined_features[mask]
        
        stats_data.append([
            label,
            np.mean(group_features),
            np.std(group_features),
            np.min(group_features),
            np.max(group_features),
            np.median(group_features)
        ])
    
    table = ax4.table(
        cellText=[[f"{x:.3f}" if isinstance(x, (int, float)) else x for x in row] for row in stats_data],
        colLabels=["Group", "Mean", "Std Dev", "Min", "Max", "Median"],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax4.set_title("Feature Statistics by Group")
    
    # Set overall title
    fig.suptitle("Feature Distribution Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the title
    
    return fig

def plot_community_degree_distributions(graph, ax=None, kde=True, title=None):
    """
    Create a detailed visualization of degree distributions by community.
    
    Args:
        graph: GraphSample instance
        ax: Matplotlib axis (optional)
        kde: Whether to use KDE for smoother distributions
        title: Plot title (optional)
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Get community labels and node degrees
    community_labels = graph.community_labels
    degrees = np.array([d for _, d in graph.graph.degree()])
    
    # Group degrees by community
    community_degrees = {}
    for i, degree in enumerate(degrees):
        if i < len(community_labels):
            community = int(community_labels[i])
            if community not in community_degrees:
                community_degrees[community] = []
            community_degrees[community].append(degree)
    
    # Number of communities
    n_communities = len(community_degrees)
    
    # Generate a color palette
    colors = sns.color_palette("husl", n_communities)
    
    # Create KDE plots for each community
    for i, (community, degrees_list) in enumerate(sorted(community_degrees.items())):
        if kde:
            # KDE plot for smoother visualization
            sns.kdeplot(
                degrees_list, 
                ax=ax, 
                color=colors[i], 
                label=f"Community {community}",
                fill=True, 
                alpha=0.3
            )
        else:
            # Histogram for more direct visualization
            ax.hist(
                degrees_list, 
                bins=20, 
                alpha=0.6, 
                color=colors[i],
                label=f"Community {community}", 
                density=True
            )
    
    # Add legend, labels and title
    ax.legend(title="Communities")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Density")
    
    if title:
        ax.set_title(title)
    else:
        if hasattr(graph, 'degree_distribution_overlap'):
            title = f"Degree Distributions by Community (Overlap={graph.degree_distribution_overlap:.2f})"
        else:
            title = "Degree Distributions by Community"
        ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def create_dccc_sbm_dashboard(graph):
    """
    Create a dashboard specifically for analyzing degree-community coupling.
    
    Args:
        graph: GraphSample instance
        
    Returns:
        Matplotlib figure with multiple subplots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Degree distribution by community (KDE)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_community_degree_distributions(graph, ax=ax1, kde=True)
    
    # 2. Degree distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    plot_community_degree_distributions(graph, ax=ax2, kde=False, title="Degree Histograms by Community")
    
    # 3. Community size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    community_counts = np.bincount(graph.community_labels)
    ax3.bar(
        range(len(community_counts)), 
        community_counts, 
        color=sns.color_palette("husl", len(community_counts))
    )
    ax3.set_xlabel('Community')
    ax3.set_ylabel('Number of Nodes')
    ax3.set_title('Community Size Distribution')
    ax3.set_xticks(range(len(community_counts)))
    
    # 4. Box plot of degrees by community
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare data for box plot
    community_data = []
    community_labels_plot = []
    
    for comm, degrees in sorted({
        comm: [d for i, d in enumerate(np.array([d for _, d in graph.graph.degree()])) 
               if i < len(graph.community_labels) and graph.community_labels[i] == comm]
        for comm in np.unique(graph.community_labels)
    }.items()):
        community_data.append(degrees)
        community_labels_plot.append(f"Comm {comm}")
    
    # Create box plot
    bplot = ax4.boxplot(
        community_data, 
        patch_artist=True, 
        labels=community_labels_plot,
        showfliers=False  # Hide outliers for cleaner visualization
    )
    
    # Color boxes
    colors = sns.color_palette("husl", len(community_data))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_title('Degree Distribution by Community (Box Plot)')
    ax4.set_xlabel('Community')
    ax4.set_ylabel('Node Degree')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add parameter information
    if hasattr(graph, 'generation_params'):
        params = graph.generation_params
        param_text = '\n'.join([
            f"Generation Method: {graph.generation_method}",
            f"Degree Distribution Overlap: {params.get('degree_distribution_overlap', 'N/A')}",
            f"Community Imbalance: {params.get('community_imbalance', 'N/A')}",
            f"Alpha (Structure vs Degree): {params.get('alpha', 'N/A')}",
            f"Aggressive Separation: {params.get('aggressive_separation', 'N/A')}"
        ])
        
        fig.text(0.02, 0.02, param_text, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for parameter text
    plt.suptitle("Community-Degree Distribution Analysis", fontsize=16)
    
    return fig

def plot_degree_community_interaction(graph):
    """
    Create a combined visualization showing how degrees and community structure interact.
    
    Args:
        graph: GraphSample instance
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Graph visualization with node sizes proportional to degrees
    pos = nx.spring_layout(graph.graph, seed=42)
    degrees = np.array([d for _, d in graph.graph.degree()])
    # Scale node sizes for better visualization
    node_sizes = 50 + 100 * (degrees / max(degrees)) ** 2
    
    # Draw nodes colored by community
    nx.draw_networkx_nodes(
        graph.graph,
        pos=pos,
        node_size=node_sizes,
        node_color=[int(c) for c in graph.community_labels],
        cmap='tab20',
        alpha=0.8,
        ax=ax1
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph.graph,
        pos=pos,
        width=0.5,
        alpha=0.4,
        ax=ax1
    )
    
    ax1.set_title("Graph with Node Sizes by Degree, Colors by Community")
    ax1.axis('off')
    
    # 2. Scatter plot: Community (x) vs Degree (y)
    communities = graph.community_labels
    
    # Generate colors for communities
    import matplotlib.cm as cm
    n_communities = len(np.unique(communities))
    colors = cm.tab20(np.linspace(0, 1, n_communities))
    
    # Add jitter to x-coordinates for better visualization
    jitter = np.random.normal(0, 0.05, size=len(communities))
    
    # Scatter plot
    for i, comm in enumerate(np.unique(communities)):
        mask = communities == comm
        ax2.scatter(
            communities[mask] + jitter[mask],
            degrees[mask],
            color=colors[i],
            s=30,
            alpha=0.7,
            label=f"Community {comm}"
        )
    
    # Add labels and legend
    ax2.set_xlabel("Community")
    ax2.set_ylabel("Node Degree")
    ax2.set_xticks(np.unique(communities))
    ax2.legend(title="Communities")
    ax2.set_title("Relationship Between Community and Degree")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title with parameters
    if hasattr(graph, 'generation_params'):
        params = graph.generation_params
        plt.suptitle(
            f"Community-Degree Coupling Analysis\n" +
            f"Overlap={params.get('degree_distribution_overlap', 'N/A')}, " +
            f"Alpha={params.get('alpha', 'N/A')}, " +
            f"Method={graph.generation_method}",
            fontsize=14
        )
    
    plt.tight_layout()
    
    return fig

def add_dccc_visualization_to_app(graph):
    """
    Add enhanced visualizations for the improved DCCC-SBM model.
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    
    # Only show if using DCCC-SBM
    if hasattr(graph, 'generation_method') and graph.generation_method == "dccc_sbm":
        st.markdown("## Enhanced DCCC-SBM Analysis")
        
        st.markdown("""
        This section provides detailed analysis of the DCCC-SBM model with enhanced
        degree distribution control and alpha parameter for balancing community structure
        vs. degree-based edge formation.
        """)
        
        # Create basic degree distribution by community plot
        fig1 = plot_community_degree_distributions(graph)
        st.pyplot(fig1)
        
        # Create comprehensive dashboard
        with st.expander("Detailed Degree Distribution Dashboard", expanded=False):
            fig2 = create_dccc_sbm_dashboard(graph)
            st.pyplot(fig2)
        
        # Create degree-community interaction visualization
        with st.expander("Community-Degree Interaction Analysis", expanded=False):
            fig3 = plot_degree_community_interaction(graph)
            st.pyplot(fig3)
        
        # Add description of the parameters
        st.markdown("### DCCC-SBM Parameters Explanation")
        
        params = graph.generation_params
        
        # Create a DataFrame to display parameters
        params_df = pd.DataFrame([
            ["Degree Distribution Overlap", params.get('degree_distribution_overlap', 'N/A'), 
             "Controls how much degree distributions overlap between communities (0=disjoint, 1=identical)"],
            ["Community Imbalance", params.get('community_imbalance', 'N/A'),
             "Controls how imbalanced community sizes are (0=balanced, 1=highly imbalanced)"],
            ["Alpha (α)", params.get('alpha', 'N/A'),
             "Weight of community structure vs. degree factors (0=only degrees matter, 1=only community structure matters)"],
            ["Aggressive Separation", params.get('aggressive_separation', 'N/A'),
             "Whether to use more aggressive approach to separate degree distributions"],
            ["Degree Distribution", params.get('degree_distribution_type', 'N/A'),
             "Type of global degree distribution to partition among communities"]
        ], columns=["Parameter", "Value", "Description"])
        
        st.table(params_df)
        
        # Add explanation of how the improved model works
        st.markdown("""
        ### Understanding Improved DCCC-SBM
        
        The Distribution-Community-Coupled Corrected SBM (DCCC-SBM) creates graphs where
        community membership and node degrees are correlated in a controllable way.
        
        #### Key Improvements:
        
        1. **Aggressive Degree Separation**: When enabled and overlap is low, communities are
           assigned to different "bins" of the global degree distribution, creating much more
           distinct degree patterns.
        
        2. **Alpha Parameter**: Controls the balance between community structure (P matrix) and
           degree factors when determining edge probabilities:
           - α = 0: Only degree factors matter (ignores community structure)
           - α = 1: Only community structure matters (ignores degree factors)
           - α = 0.5: Equal weight to both factors
        
        3. **Community Imbalance**: Controls how unevenly nodes are distributed across communities,
           allowing for more realistic graph structures.
        """)
        
        # Add metrics about the degree-community correlation
        st.markdown("### Degree-Community Correlation Metrics")
        
        # Calculate metrics
        degrees = np.array([d for _, d in graph.graph.degree()])
        communities = graph.community_labels
        
        # Average degree by community
        community_avg_degrees = {}
        for comm in np.unique(communities):
            mask = communities == comm
            community_avg_degrees[comm] = np.mean(degrees[mask])
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame([
            [comm, avg, np.std(degrees[communities == comm])]
            for comm, avg in community_avg_degrees.items()
        ], columns=["Community", "Average Degree", "Std Deviation"])
        
        st.dataframe(metrics_df)
        
        # Calculate degree range statistics to measure degree separation
        min_degrees = [np.min(degrees[communities == comm]) for comm in np.unique(communities)]
        max_degrees = [np.max(degrees[communities == comm]) for comm in np.unique(communities)]
        
        # Display range statistics
        range_df = pd.DataFrame({
            "Community": np.unique(communities),
            "Min Degree": min_degrees,
            "Max Degree": max_degrees,
            "Range": [max_d - min_d for min_d, max_d in zip(min_degrees, max_degrees)]
        })
        
        st.dataframe(range_df)

def create_dashboard(graph, membership_matrix, communities, universe_P, figsize=(18, 12)):
    """
    Create a comprehensive dashboard for a graph sample.
    
    Args:
        graph: NetworkX graph
        membership_matrix: Node-community membership matrix
        communities: List of community IDs
        universe_P: Full probability matrix from the universe
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with a grid of subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # Graph with communities
    ax1 = fig.add_subplot(gs[0, 0])
    plot_graph_communities(graph, ax=ax1, title="Graph with Communities")
    
    # Community probability matrix
    ax2 = fig.add_subplot(gs[0, 1])
    plot_community_matrix(universe_P, communities, ax=ax2)
    
    # Community network
    ax3 = fig.add_subplot(gs[0, 2])
    plot_community_graph(universe_P, communities, ax=ax3)
    
    # Degree distribution
    ax4 = fig.add_subplot(gs[1, 0])
    plot_degree_distribution(graph, ax=ax4)
    
    # Membership matrix
    ax5 = fig.add_subplot(gs[1, 1])
    plot_membership_matrix(membership_matrix, communities, ax=ax5)
    
    # Community overlap distribution
    ax6 = fig.add_subplot(gs[1, 2])
    plot_community_overlap_distribution(membership_matrix, ax=ax6)
    
    fig.tight_layout()
    return fig

def visualize_community_cluster_assignments(
    feature_generator: 'SimplifiedFeatureGenerator',
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    title: str = "Community-Cluster Assignments"
) -> plt.Figure:
    """
    Visualize how communities are assigned to feature clusters.
    
    Args:
        feature_generator: The SimplifiedFeatureGenerator instance
        figsize: Figure size
        cmap: Colormap for the heatmap
        title: Plot title
        
    Returns:
        Matplotlib figure showing the community-cluster assignments
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Community-Cluster Probability Matrix
    probs = feature_generator.community_cluster_probs
    im1 = ax1.imshow(probs, cmap=cmap, aspect='auto')
    plt.colorbar(im1, ax=ax1, label="Assignment Probability")
    
    # Set labels
    ax1.set_xlabel("Cluster Index")
    ax1.set_ylabel("Community Index")
    ax1.set_title("Community-Cluster Assignment Probabilities")
    
    # Add grid
    ax1.set_xticks(np.arange(probs.shape[1]))
    ax1.set_yticks(np.arange(probs.shape[0]))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster Statistics
    if hasattr(feature_generator, 'cluster_stats') and feature_generator.cluster_stats:
        # Get cluster counts
        cluster_counts = feature_generator.cluster_stats['cluster_counts']
        
        # Create bar plot
        bars = ax2.bar(range(len(cluster_counts)), cluster_counts)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax2.set_xlabel("Cluster Index")
        ax2.set_ylabel("Number of Nodes")
        ax2.set_title("Node Distribution Across Clusters")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No cluster statistics available yet.\nGenerate a graph to see distribution.",
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Cluster Statistics")
    
    # Add overall title
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    return fig

def plot_universe_cooccurrence_matrix(
    universe: 'GraphUniverse',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    title: str = "Universe Community Co-occurrence Matrix",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the universe's community co-occurrence matrix.
    
    Args:
        universe: GraphUniverse instance
        figsize: Figure size
        cmap: Colormap for the heatmap
        title: Figure title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get the co-occurrence matrix
    cooccurrence_matrix = universe.community_cooccurrence_matrix
    K = universe.K
    
    # Create heatmap
    im = ax.imshow(cooccurrence_matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Co-occurrence Probability")
    
    # Set labels
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"C{i}" for i in range(K)], rotation=45)
    ax.set_yticklabels([f"C{i}" for i in range(K)])
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Community ID")
    
    # Add text annotations
    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, f"{cooccurrence_matrix[i, j]:.2f}",
                          ha="center", va="center", color="white" if cooccurrence_matrix[i, j] > 0.5 else "black")
    
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    return fig

def plot_universe_feature_centers(
    universe: 'GraphUniverse',
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Universe Feature Centers",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the universe's feature centers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    feature_centers = universe.feature_generator.cluster_centers
    K = universe.K
    
    # Create a heatmap of the feature centers
    im = ax.imshow(feature_centers, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Feature Center")
    ax.set_title(title)
    ax.set_xticks(np.arange(feature_centers.shape[1]))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"dim{i}" for i in range(feature_centers.shape[1])])
    ax.set_yticklabels([f"C{i}" for i in range(K)])
    
    fig.tight_layout()
    return fig

def plot_universe_community_degree_propensity_vector(
    universe: 'GraphUniverse',
    figsize: Tuple[int, int] = (12, 6),
    color: str = "blue",
    title: str = "Universe Community-Degree Propensity Vector",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the universe's community-degree propensity vector.
    
    Args:
        universe: GraphUniverse instance
        figsize: Figure size
        color: Color for the bars
        title: Figure title
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get the community-degree propensity vector
    community_degree_propensity_vector = universe.community_degree_propensity_vector
    K = universe.K
    
    # Create bar plot
    bars = ax.bar(range(K), community_degree_propensity_vector, color=color, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, community_degree_propensity_vector)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{value:.2f}", ha="center", va="bottom", fontweight='bold')
    
    # Set labels
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Community-Degree Propensity Value")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"C{i}" for i in range(K)])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    return fig

def plot_property_validation(family_generator, figsize=(14, 10)):
    """
    Create a validation plot showing target vs. actual property ranges.
    
    Args:
        family_generator: GraphFamilyGenerator object with generated graphs
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    
    if not family_generator.graphs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No graphs available for validation", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Extract properties from the family generator
    properties = family_generator.analyze_graph_family_properties()
    
    # Define the property configurations with maximum possible ranges from the Streamlit UI
    property_configs = [
        {
            'name': 'homophily',
            'values': properties.get('homophily_levels', []),
            'target_range': family_generator.homophily_range,
            'max_possible_range': (0.0, 1.0),  # Full homophily range in Streamlit
            'title': 'Homophily',
            'format': '{:.2f}',
            'position': 0  # Top to bottom order
        },
        {
            'name': 'avg_degree',
            'values': properties.get('avg_degrees', []),
            'target_range': family_generator.avg_degree_range,
            'max_possible_range': (1.0, 30.0),  # Max degree range in Streamlit
            'title': 'Average Degree',
            'format': '{:.1f}',
            'position': 1
        },
        {
            'name': 'n_nodes',
            'values': properties.get('node_counts', []),
            'target_range': (family_generator.min_n_nodes, family_generator.max_n_nodes),
            'max_possible_range': (10, 500),  # Node count range in Streamlit
            'title': 'Node Count',
            'format': '{:d}',
            'position': 2
        },
        {
            'name': 'n_communities',
            'values': properties.get('community_counts', []),
            'target_range': (family_generator.min_communities, family_generator.max_communities),
            'max_possible_range': (2, family_generator.universe.K),  # Communities range in Streamlit
            'title': 'Community Count',
            'format': '{:d}',
            'position': 3
        }
    ]
    
    # Sort configs by position
    property_configs.sort(key=lambda x: x['position'])
    
    # Check if we have any data
    has_data = any(len(config['values']) > 0 for config in property_configs)
    if not has_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available for validation", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create a figure with subplots stacked vertically
    fig = plt.figure(figsize=figsize)
    
    # Create a gridspec with tight spacing between subplots
    gs = gridspec.GridSpec(len(property_configs), 1, hspace=0.4)
    
    # Create shared legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Within Range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outside Range')
    ]
    
    # Plot each property in its own subplot
    for i, config in enumerate(property_configs):
        ax = plt.subplot(gs[i])
        
        values = config['values']
        target_range = config['target_range']
        
        if not values:
            ax.text(0.5, 0.5, f"No data for {config['title']}", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Calculate coverage
        within_range = sum(1 for v in values if target_range[0] <= v <= target_range[1])
        coverage = within_range / len(values) * 100 if values else 0
        
        # Choose color based on coverage
        if coverage >= 90:
            color = 'green'
        elif coverage >= 70:
            color = 'orange'
        else:
            color = 'red'
        
        # Use the maximum possible range from Streamlit UI as reference
        max_possible_range = config.get('max_possible_range', target_range)
        
        # Calculate axis limits with padding
        min_val = min(min(values), max_possible_range[0])
        max_val = max(max(values), max_possible_range[1])
        padding = (max_val - min_val) * 0.1
        x_min = min_val - padding
        x_max = max_val + padding
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, 0.5)
        
        # Draw maximum possible range as a light gray box
        max_possible_range = config.get('max_possible_range', target_range)
        box_height = 0.4
        max_box = patches.Rectangle(
            (max_possible_range[0], -box_height/2), 
            max_possible_range[1] - max_possible_range[0], 
            box_height, 
            alpha=0.15, 
            facecolor='gray',
            edgecolor='gray', 
            linewidth=1
        )
        ax.add_patch(max_box)
        
        # Draw target range as a colored box on top
        target_box = patches.Rectangle(
            (target_range[0], -box_height/2), 
            target_range[1] - target_range[0], 
            box_height, 
            alpha=0.4, 
            facecolor=color,
            edgecolor='black', 
            linewidth=1
        )
        ax.add_patch(target_box)
        
        # Add target range text
        if config['name'] in ['homophily', 'avg_degree']:
            range_text = f"Target: {config['format'].format(target_range[0])}-{config['format'].format(target_range[1])}"
        else:
            range_text = f"Target: {int(target_range[0])}-{int(target_range[1])}"
            
        # Add max possible range text (smaller and lighter)
        max_possible_range = config.get('max_possible_range', target_range)
        # if config['name'] in ['homophily', 'avg_degree']:
        #     max_range_text = f"Max Range: {config['format'].format(max_possible_range[0])}-{config['format'].format(max_possible_range[1])}"
        # else:
        #     max_range_text = f"Max Range: {int(max_possible_range[0])}-{int(max_possible_range[1])}"
        
        # Position target range text in the middle of the box
        text_x = target_range[0] + (target_range[1] - target_range[0]) / 2
        ax.text(text_x, 0, range_text, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='black')
               
        # # Position max range text at the top of the subplot
        # ax.text(0.02, 0.95, max_range_text, ha='left', va='top', 
        #        fontsize=8, color='gray', transform=ax.transAxes)
        
        # Add scatter points for actual values (with jitter on y-axis)
        y_jitter = np.random.normal(0, 0.1, size=len(values))
        scatter = ax.scatter(values, y_jitter, 
                  alpha=0.7, s=30, 
                  c=['red' if (v < target_range[0] or v > target_range[1]) else 'blue' for v in values],
                  marker='o', zorder=5)
        
        # Add coverage annotation
        ax.text(0.99, 0.5, f'Coverage: {coverage:.1f}%', 
               ha='right', va='center', fontsize=10, fontweight='bold', color=color,
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Set title and labels
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        
        # Only show x-label on the bottom subplot
        if i == len(property_configs) - 1:
            ax.set_xlabel('Value')
        
        # Remove y-ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Add grid lines
        ax.grid(True, alpha=0.3, axis='x')
    
    # Add a common legend at the top of the figure
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.95),
              ncol=2, frameon=True, fontsize=10)
    
    # Add a title to the figure
    # fig.suptitle("Graph Property Validation", fontsize=16, fontweight='bold', y=0.99)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    
    return fig


# Backward compatibility alias
plot_universe_degree_centers = plot_universe_community_degree_propensity_vector

def plot_universe_summary(
    universe: 'GraphUniverse',
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Create a comprehensive visualization of the universe properties.
    
    Args:
        universe: GraphUniverse instance
        figsize: Figure size
        cmap: Colormap for the heatmap
        
    Returns:
        Matplotlib figure with multiple subplots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Inter-community probability matrix
    im1 = ax1.imshow(universe.P, cmap=cmap, aspect='auto')
    plt.colorbar(im1, ax=ax1, label="Edge Probability")
    ax1.set_title("Inter-Community Probability Matrix")
    ax1.set_xticks(np.arange(universe.K))
    ax1.set_yticks(np.arange(universe.K))
    ax1.set_xticklabels([f"C{i}" for i in range(universe.K)], rotation=45)
    ax1.set_yticklabels([f"C{i}" for i in range(universe.K)])
    
    # 2. Co-occurrence matrix
    im2 = ax2.imshow(universe.community_cooccurrence_matrix, cmap=cmap, aspect='auto')
    plt.colorbar(im2, ax=ax2, label="Co-occurrence Probability")
    ax2.set_title("Community Co-occurrence Matrix")
    ax2.set_xticks(np.arange(universe.K))
    ax2.set_yticks(np.arange(universe.K))
    ax2.set_xticklabels([f"C{i}" for i in range(universe.K)], rotation=45)
    ax2.set_yticklabels([f"C{i}" for i in range(universe.K)])
    
    # 3. Community-degree propensity vector
    bars = ax3.bar(range(universe.K), universe.community_degree_propensity_vector, color='blue', alpha=0.7, edgecolor='black')
    for i, (bar, value) in enumerate(zip(bars, universe.community_degree_propensity_vector)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{value:.2f}", ha="center", va="bottom", fontweight='bold')
    ax3.set_title("Community-Degree Propensity Vector")
    ax3.set_xlabel("Community ID")
    ax3.set_ylabel("Community-Degree Propensity Value")
    ax3.set_xticks(range(universe.K))
    ax3.set_xticklabels([f"C{i}" for i in range(universe.K)])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Universe parameters summary
    ax4.axis('off')
    summary_text = f"""
    Universe Parameters:
    • Number of Communities (K): {universe.K}
    • Feature Dimension: {universe.feature_dim}
    • Inter-Community Variance: {universe.inter_community_variance:.3f}
    • Community Co-occurrence Homogeneity: {universe.community_cooccurrence_homogeneity:.3f}
    • Degree Center Method: {getattr(universe, 'degree_center_method', 'N/A')}
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle("Universe Summary", fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig