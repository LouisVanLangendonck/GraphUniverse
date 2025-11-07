"""
Utilities to visualize graphs with community structure,
node embeddings, and other graph properties.
"""

from typing import Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_graph_communities(
    graph: Union[nx.Graph, "GraphSample"],
    community_key: str = "community",
    layout: str = "spring",
    node_size: float = 50,
    edge_width: float = 0.5,
    edge_alpha: float = 0.2,
    figsize: tuple[int, int] = (10, 8),
    with_labels: bool = False,
    cmap: str = "tab20",
    title: str | None = None,
    ax: plt.Axes | None = None,
    pos: dict | None = None,
    min_component_size: int = 0,
) -> plt.Figure:
    """
    Plot a graph with nodes colored by community.
    Only shows components that meet the minimum size requirement.
    
    Community colors are automatically mapped based on community ID, ensuring
    consistent coloring across multiple graphs (e.g., community 0 is always
    the same color, community 1 is always the same color, etc.).

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
    if hasattr(graph, "graph"):
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
        ax.text(
            0.5,
            0.5,
            "No components above size threshold",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
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
    nx.draw_networkx_edges(kept_graph, pos, alpha=edge_alpha, width=edge_width, ax=ax)

    # Draw nodes for each community
    for comm, nodes in communities.items():
        # Use community ID directly for consistent coloring across graphs
        # This ensures community 0 is always the same color, community 1 is always the same color, etc.
        color = cmap_obj(comm % cmap_obj.N)
        
        nx.draw_networkx_nodes(
            kept_graph,
            pos,
            nodelist=nodes,
            node_color=[color] * len(nodes),
            node_size=node_size,
            alpha=0.8,
            ax=ax,
        )

    # Add node labels if requested
    if with_labels:
        nx.draw_networkx_labels(kept_graph, pos, font_size=8, ax=ax)

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
    for comm in sorted(communities.keys()):
        # Use same color logic as for drawing nodes
        color = cmap_obj(comm % cmap_obj.N)
        ax.scatter([], [], c=[color], label=f"Community {comm}")

    ax.legend(loc="best", title="Communities")

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
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    if not family_generator.graphs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5, 0.5, "No graphs available for validation", ha="center", va="center", fontsize=14
        )
        ax.axis("off")
        return fig

    # Extract properties from the family generator
    properties = family_generator.analyze_graph_family_properties()

    # Define the property configurations with maximum possible ranges from the Streamlit UI
    property_configs = [
        {
            "name": "homophily",
            "values": properties.get("homophily_levels", []),
            "target_range": family_generator.homophily_range,
            "max_possible_range": (0.0, 1.0),  # Full homophily range in Streamlit
            "title": "Homophily",
            "format": "{:.2f}",
            "position": 0,  # Top to bottom order
        },
        {
            "name": "avg_degree",
            "values": properties.get("avg_degrees", []),
            "target_range": family_generator.avg_degree_range,
            "max_possible_range": (1.0, 30.0),  # Max degree range in Streamlit
            "title": "Average Degree",
            "format": "{:.1f}",
            "position": 1,
        },
        {
            "name": "n_nodes",
            "values": properties.get("node_counts", []),
            "target_range": (family_generator.min_n_nodes, family_generator.max_n_nodes),
            "max_possible_range": (10, 500),  # Node count range in Streamlit
            "title": "Node Count",
            "format": "{:d}",
            "position": 2,
        },
        {
            "name": "n_communities",
            "values": properties.get("community_counts", []),
            "target_range": (family_generator.min_communities, family_generator.max_communities),
            "max_possible_range": (
                2,
                family_generator.universe.K,
            ),  # Communities range in Streamlit
            "title": "Community Count",
            "format": "{:d}",
            "position": 3,
        },
    ]

    # Sort configs by position
    property_configs.sort(key=lambda x: x["position"])

    # Check if we have any data
    has_data = any(len(config["values"]) > 0 for config in property_configs)
    if not has_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available for validation", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create a figure with subplots stacked vertically
    fig = plt.figure(figsize=figsize)

    # Create a gridspec with tight spacing between subplots
    gs = gridspec.GridSpec(len(property_configs), 1, hspace=0.4)

    # Create shared legend elements
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Within Range",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Outside Range",
        ),
    ]

    # Plot each property in its own subplot
    for i, config in enumerate(property_configs):
        ax = plt.subplot(gs[i])

        values = config["values"]
        target_range = config["target_range"]

        if not values:
            ax.text(
                0.5, 0.5, f"No data for {config['title']}", ha="center", va="center", fontsize=12
            )
            ax.axis("off")
            continue

        # Calculate coverage
        within_range = sum(1 for v in values if target_range[0] <= v <= target_range[1])
        coverage = within_range / len(values) * 100 if values else 0

        # Choose color based on coverage
        if coverage >= 90:
            color = "green"
        elif coverage >= 70:
            color = "orange"
        else:
            color = "red"

        # Use the maximum possible range from Streamlit UI as reference
        max_possible_range = config.get("max_possible_range", target_range)

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
        max_possible_range = config.get("max_possible_range", target_range)
        box_height = 0.4
        max_box = patches.Rectangle(
            (max_possible_range[0], -box_height / 2),
            max_possible_range[1] - max_possible_range[0],
            box_height,
            alpha=0.15,
            facecolor="gray",
            edgecolor="gray",
            linewidth=1,
        )
        ax.add_patch(max_box)

        # Draw target range as a colored box on top
        target_box = patches.Rectangle(
            (target_range[0], -box_height / 2),
            target_range[1] - target_range[0],
            box_height,
            alpha=0.4,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(target_box)

        # Add target range text
        if config["name"] in ["homophily", "avg_degree"]:
            range_text = f"Target: {config['format'].format(target_range[0])}-{config['format'].format(target_range[1])}"
        else:
            range_text = f"Target: {int(target_range[0])}-{int(target_range[1])}"

        # Add max possible range text (smaller and lighter)
        max_possible_range = config.get("max_possible_range", target_range)
        # if config['name'] in ['homophily', 'avg_degree']:
        #     max_range_text = f"Max Range: {config['format'].format(max_possible_range[0])}-{config['format'].format(max_possible_range[1])}"
        # else:
        #     max_range_text = f"Max Range: {int(max_possible_range[0])}-{int(max_possible_range[1])}"

        # Position target range text in the middle of the box
        text_x = target_range[0] + (target_range[1] - target_range[0]) / 2
        ax.text(
            text_x,
            0,
            range_text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

        # # Position max range text at the top of the subplot
        # ax.text(0.02, 0.95, max_range_text, ha='left', va='top',
        #        fontsize=8, color='gray', transform=ax.transAxes)

        # Add scatter points for actual values (with jitter on y-axis)
        y_jitter = np.random.normal(0, 0.1, size=len(values))
        scatter = ax.scatter(
            values,
            y_jitter,
            alpha=0.7,
            s=30,
            c=["red" if (v < target_range[0] or v > target_range[1]) else "blue" for v in values],
            marker="o",
            zorder=5,
        )

        # Add coverage annotation
        ax.text(
            0.99,
            0.5,
            f"Coverage: {coverage:.1f}%",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
        )

        # Set title and labels
        ax.set_title(config["title"], fontsize=12, fontweight="bold")

        # Only show x-label on the bottom subplot
        if i == len(property_configs) - 1:
            ax.set_xlabel("Value")

        # Remove y-ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Add grid lines
        ax.grid(True, alpha=0.3, axis="x")

    # Add a common legend at the top of the figure
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=True,
        fontsize=10,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title

    return fig


def plot_universe_community_degree_propensity_vector(
    universe: "GraphUniverse",
    figsize: tuple[int, int] = (12, 6),
    color: str = "blue",
    title: str = "Universe Community-Degree Propensity Vector",
    ax: plt.Axes | None = None,
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
    bars = ax.bar(
        range(K), community_degree_propensity_vector, color=color, alpha=0.7, edgecolor="black"
    )

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, community_degree_propensity_vector, strict=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Set labels
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Community-Degree Propensity Value")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"C{i}" for i in range(K)])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_universe_feature_centers(
    universe: "GraphUniverse",
    figsize: tuple[int, int] = (12, 6),
    title: str = "Universe Feature Centers",
    ax: plt.Axes | None = None,
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
    im = ax.imshow(feature_centers, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Feature Center")
    ax.set_title(title)
    ax.set_xticks(np.arange(feature_centers.shape[1]))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"dim{i}" for i in range(feature_centers.shape[1])])
    ax.set_yticklabels([f"C{i}" for i in range(K)])

    fig.tight_layout()
    return fig

