"""
Graph parameter analysis module for evaluating statistical properties of graph families.

This module provides functions to analyze graph families along key parameter dimensions
and visualize their distribution in parameter space.

The key parameters analyzed are:
1. Homophily level
2. Power law exponent of degree distribution
3. Clustering coefficient
4. Triangle count
5. Node count
6. Average node degree
7. Graph density
8. Connected components
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
import matplotlib

def analyze_graph_parameters(
    graph: nx.Graph,
    community_labels: np.ndarray,
    communities: List[int]
) -> Dict[str, float]:
    """
    Analyze graph parameters.
    
    Args:
        graph: NetworkX graph
        community_labels: Node community assignments (one-hot encoded)
        communities: List of community IDs
        
    Returns:
        Dictionary of parameter values
    """
    result = {}
    
    # Basic graph properties
    n_nodes = len(graph)
    n_edges = graph.number_of_edges()
    
    # Handle empty or invalid graphs
    if n_nodes == 0 or n_edges == 0:
        return {
            "homophily": 0.0,
            "power_law_exponent": None,
            "clustering_coefficient": 0.0,
            "node_count": 0,
            "avg_degree": 0.0,
            "density": 0.0,
            "connected_components": 0,
            "largest_component_size": 0,
            "intra_community_density": 0.0,
            "inter_community_density": 0.0
        }
    
    # Calculate density
    result["density"] = nx.density(graph)
    result["node_count"] = n_nodes
    
    # Average degree
    degrees = [d for _, d in graph.degree()]
    result["avg_degree"] = sum(degrees) / n_nodes
    
    # Calculate homophily
    try:
        # Get community for each node
        node_communities = np.argmax(community_labels, axis=1)
        
        # Create mapping from graph node labels to indices
        node_to_idx = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        # Count same-community and different-community edges
        same_community = 0
        diff_community = 0
        
        for u, v in graph.edges():
            # Map node labels to indices in community labels
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Compare communities
            if node_communities[u_idx] == node_communities[v_idx]:
                same_community += 1
            else:
                diff_community += 1
        
        # Calculate homophily
        total_edges = same_community + diff_community
        result["homophily"] = same_community / total_edges if total_edges > 0 else 0.0
    except (AttributeError, TypeError, KeyError):
        result["homophily"] = 0.0
    
    # Clustering coefficient (handle case with no triangles)
    try:
        result["clustering_coefficient"] = nx.average_clustering(graph)
    except ZeroDivisionError:
        result["clustering_coefficient"] = 0.0
    
    # Power law exponent
    if degrees:
        # Fit power law to degree distribution
        degrees = np.array(degrees) + 1  # Add 1 to avoid log(0)
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        if len(unique_degrees) > 1:
            log_degrees = np.log(unique_degrees)
            log_counts = np.log(counts)
            slope, _, _, _, _ = stats.linregress(log_degrees, log_counts)
            result["power_law_exponent"] = -slope
        else:
            result["power_law_exponent"] = None
    else:
        result["power_law_exponent"] = None
    
    # Connected components
    try:
        components = list(nx.connected_components(graph))
        result["connected_components"] = len(components)
        result["largest_component_size"] = len(max(components, key=len)) if components else 0
    except nx.NetworkXError:
        result["connected_components"] = 0
        result["largest_component_size"] = 0
    
    # Edge density within and between communities
    try:
        total_possible_edges = n_nodes * (n_nodes - 1) / 2
        total_possible_intra = 0
        total_possible_inter = 0
        actual_intra = 0
        actual_inter = 0
        
        # Get communities for all nodes
        node_communities = np.argmax(community_labels, axis=1)
        
        # Count possible and actual edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if node_communities[i] == node_communities[j]:
                    total_possible_intra += 1
                    if graph.has_edge(i, j):
                        actual_intra += 1
                else:
                    total_possible_inter += 1
                    if graph.has_edge(i, j):
                        actual_inter += 1
        
        # Calculate densities
        result["intra_community_density"] = actual_intra / total_possible_intra if total_possible_intra > 0 else 0.0
        result["inter_community_density"] = actual_inter / total_possible_inter if total_possible_inter > 0 else 0.0
    except (AttributeError, TypeError, ZeroDivisionError):
        result["intra_community_density"] = 0.0
        result["inter_community_density"] = 0.0
    
    return result

def analyze_community_connectivity(
    universe_P: np.ndarray,
    communities: List[int],
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze connectivity characteristics of a sampled subset of communities.
    
    Args:
        universe_P: The full probability matrix from the universe
        communities: List of community indices to analyze
        threshold: Threshold for considering communities connected
        
    Returns:
        Dictionary with connectivity metrics
    """
    # Extract the submatrix for the selected communities
    n_communities = len(communities)
    P_sub = np.zeros((n_communities, n_communities))
    
    for i, ci in enumerate(communities):
        for j, cj in enumerate(communities):
            P_sub[i, j] = universe_P[ci, cj]
    
    # Create a thresholded adjacency matrix for connectivity analysis
    adjacency = (P_sub >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)  # Ignore self-connections
    
    # Analyze connectivity
    results = {}
    
    # Count connections per community
    connection_counts = np.sum(adjacency, axis=1)
    results["connection_counts"] = connection_counts
    results["avg_connections"] = float(np.mean(connection_counts))
    results["min_connections"] = int(np.min(connection_counts))
    results["isolated_communities"] = int(np.sum(connection_counts == 0))
    
    # Check if the community graph is connected
    # Convert to NetworkX for easy analysis
    G = nx.Graph()
    G.add_nodes_from(range(n_communities))
    for i in range(n_communities):
        for j in range(i+1, n_communities):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)
    
    results["is_connected"] = nx.is_connected(G)
    
    # Get number of connected components
    components = list(nx.connected_components(G))
    results["n_components"] = len(components)
    
    # Calculate average component size
    component_sizes = [len(comp) for comp in components]
    results["component_sizes"] = component_sizes
    if component_sizes:
        results["avg_component_size"] = float(np.mean(component_sizes))
    else:
        results["avg_component_size"] = 0.0
        
    # Calculate overall community connectivity
    if n_communities > 1:
        # Density of connections
        max_possible_connections = n_communities * (n_communities - 1) / 2
        actual_connections = np.sum(adjacency) / 2  # Divide by 2 for undirected
        results["connectivity_density"] = float(actual_connections / max_possible_connections)
    else:
        results["connectivity_density"] = 0.0
    
    # Calculate average, min, and max edge probabilities (excluding self-connections)
    off_diag_probs = P_sub[~np.eye(n_communities, dtype=bool)]
    results["avg_edge_probability"] = float(np.mean(off_diag_probs))
    results["min_edge_probability"] = float(np.min(off_diag_probs))
    results["max_edge_probability"] = float(np.max(off_diag_probs))
    
    return results

def visualize_community_connectivity(
    universe_P: np.ndarray,
    communities: List[int],
    threshold: float = 0.05,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize connectivity characteristics of sampled communities.
    
    Args:
        universe_P: The full probability matrix from the universe
        communities: List of community indices to analyze
        threshold: Threshold for considering communities connected
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract the submatrix and analyze connectivity
    n_communities = len(communities)
    P_sub = np.zeros((n_communities, n_communities))
    
    for i, ci in enumerate(communities):
        for j, cj in enumerate(communities):
            P_sub[i, j] = universe_P[ci, cj]
    
    # Create thresholded adjacency
    adjacency = (P_sub >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)  # Ignore self-connections
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Probability matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(P_sub, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, label="Edge Probability")
    
    # Add threshold contour
    if threshold > 0:
        ax1.contour(P_sub >= threshold, colors='red', levels=[0.5])
    
    ax1.set_xticks(np.arange(n_communities))
    ax1.set_yticks(np.arange(n_communities))
    ax1.set_xticklabels([f"C{c}" for c in communities])
    ax1.set_yticklabels([f"C{c}" for c in communities])
    ax1.set_title("Community Probability Matrix")
    
    # 2. Community graph
    ax2 = fig.add_subplot(gs[0, 1])
    G = nx.Graph()
    G.add_nodes_from(range(n_communities))
    
    # Add edges with weights from probability matrix
    for i in range(n_communities):
        for j in range(i+1, n_communities):
            if P_sub[i, j] >= threshold:
                G.add_edge(i, j, weight=P_sub[i, j])
    
    # Position nodes in a circular layout
    pos = nx.spring_layout(G)
    
    # Draw nodes with community labels
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=500, 
        node_color='lightblue',
        ax=ax2
    )
    
    # Draw edges with width proportional to probability
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_weights, 
        alpha=0.7, 
        edge_color='navy',
        ax=ax2
    )
    
    # Draw node labels
    labels = {i: f"C{communities[i]}" for i in range(n_communities)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax2)
    
    # Draw edge weights
    if n_communities < 10:  # Only for small graphs to avoid clutter
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax2)
    
    ax2.set_title(f"Community Graph (p ≥ {threshold})")
    ax2.axis('off')
    
    # 3. Connection count distribution
    ax3 = fig.add_subplot(gs[1, 0])
    connection_counts = np.sum(adjacency, axis=1)
    
    # Create bar chart
    bars = ax3.bar(
        range(n_communities), 
        connection_counts,
        color='lightblue'
    )
    
    # Highlight isolated communities
    for i, count in enumerate(connection_counts):
        if count == 0:
            bars[i].set_color('red')
    
    ax3.set_xlabel("Community Index")
    ax3.set_ylabel("Number of Connections")
    ax3.set_title("Connections per Community")
    ax3.set_xticks(range(n_communities))
    ax3.set_xticklabels([f"C{communities[i]}" for i in range(n_communities)])
    
    # 4. Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    connectivity_analysis = analyze_community_connectivity(universe_P, communities, threshold)
    
    # Create table data
    table_data = [
        ["Metric", "Value"],
        ["Connected", str(connectivity_analysis["is_connected"])],
        ["Components", str(connectivity_analysis["n_components"])],
        ["Isolated Communities", str(connectivity_analysis["isolated_communities"])],
        ["Avg. Connections", f"{connectivity_analysis['avg_connections']:.2f}"],
        ["Min. Connections", str(connectivity_analysis["min_connections"])],
        ["Connectivity Density", f"{connectivity_analysis['connectivity_density']:.3f}"],
        ["Avg. Edge Probability", f"{connectivity_analysis['avg_edge_probability']:.3f}"]
    ]
    
    # Create table
    table = ax4.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax4.set_title("Connectivity Statistics")
    
    fig.tight_layout()
    return fig

def calculate_homophily(
    graph: nx.Graph,
    community_labels: np.ndarray,
    communities: List[int]
) -> float:
    """
    Calculate homophily level of the graph.
    
    Homophily is measured as the ratio of edges between nodes in the same
    community to total edges.
    
    Args:
        graph: NetworkX graph
        community_labels: Node community assignments (one-hot encoded)
        communities: List of community IDs
        
    Returns:
        Homophily score in [0, 1]
    """
    # Get community for each node
    node_communities = np.argmax(community_labels, axis=1)
    
    # Create mapping from graph node labels to indices
    node_to_idx = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    
    # Count same-community and different-community edges
    same_community = 0
    diff_community = 0
    
    for u, v in graph.edges():
        # Map node labels to indices in community labels
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Compare communities
        if node_communities[u_idx] == node_communities[v_idx]:
            same_community += 1
        else:
            diff_community += 1
    
    # Calculate homophily
    total_edges = same_community + diff_community
    homophily = same_community / total_edges if total_edges > 0 else 0
    
    return homophily


def fit_power_law(degrees: List[int]) -> float:
    """
    Fit a power law distribution to the degree distribution.
    
    Args:
        degrees: List of node degrees
        
    Returns:
        Power law exponent
    """
    # Filter out zeros and get unique degrees with counts
    degrees = [d for d in degrees if d > 0]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    
    # Log transform for power law fitting
    log_degrees = np.log10(unique_degrees)
    log_counts = np.log10(counts)
    
    # Define power law function
    def power_law(x, alpha, c):
        return c - alpha * x
    
    # Fit power law
    params, _ = curve_fit(power_law, log_degrees, log_counts)
    alpha = params[0]
    
    return alpha


def analyze_graph_family(graphs: List[Any], attribute_name: str = "graph") -> pd.DataFrame:
    """
    Analyze parameters for a family of graphs.
    
    Args:
        graphs: List of graph objects (can be GraphSample objects or NetworkX graphs)
        attribute_name: Name of the attribute containing the NetworkX graph (if not graph objects)
        
    Returns:
        DataFrame with parameter values for each graph
    """
    results = []
    
    for i, graph_obj in enumerate(graphs):
        # Get the NetworkX graph
        if attribute_name == "graph":
            if isinstance(graph_obj, nx.Graph):
                graph = graph_obj
                community_labels = None
                communities = None
            else:
                graph = graph_obj.graph
                community_labels = getattr(graph_obj, "community_labels", None)
                communities = getattr(graph_obj, "communities", None)
        else:
            graph = getattr(graph_obj, attribute_name)
            community_labels = getattr(graph_obj, "community_labels", None)
            communities = getattr(graph_obj, "communities", None)
        
        # Analyze parameters
        params = analyze_graph_parameters(graph, community_labels, communities)
        params["graph_id"] = i
        
        results.append(params)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def compute_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for graph family parameters.
    
    Args:
        df: DataFrame with graph parameters
        
    Returns:
        Dictionary with statistics for each parameter
    """
    stats = {}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "graph_id"]
    
    for col in numeric_cols:
        # Filter out NaN values
        values = df[col].dropna().values
        
        if len(values) > 0:
            stats[col] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75)
            }
        else:
            stats[col] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "q25": None,
                "q75": None
            }
    
    return stats


def compare_graph_families(
    family_dfs: Dict[str, pd.DataFrame],
    params: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare statistics for multiple graph families.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        params: List of parameters to compare (if None, uses all common parameters)
        
    Returns:
        DataFrame with statistics for each family
    """
    if params is None:
        # Use intersection of available parameters
        param_sets = [set(df.columns) - {"graph_id"} for df in family_dfs.values()]
        params = list(set.intersection(*param_sets))
    
    # Compute statistics for each family
    family_stats = {}
    for name, df in family_dfs.items():
        family_stats[name] = compute_statistics(df)
    
    # Combine into a single DataFrame for comparison
    results = []
    
    for param in params:
        for family_name, stats in family_stats.items():
            if param in stats:
                param_stats = stats[param]
                row = {
                    "family": family_name,
                    "parameter": param,
                    "mean": param_stats["mean"],
                    "std": param_stats["std"],
                    "min": param_stats["min"],
                    "max": param_stats["max"],
                    "median": param_stats["median"]
                }
                results.append(row)
    
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def plot_parameter_distribution(
    df: pd.DataFrame,
    parameter: str,
    family_name: str = "Graph Family",
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of a specific parameter.
    
    Args:
        df: DataFrame with graph parameters
        parameter: Parameter to plot
        family_name: Name of the graph family
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out NaN values
    values = df[parameter].dropna()
    
    # Plot histogram
    sns.histplot(values, bins=bins, kde=True, ax=ax)
    
    # Add vertical line for mean
    mean_val = values.mean()
    ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    
    # Add vertical lines for standard deviation
    std_val = values.std()
    ax.axvline(mean_val + std_val, color='g', linestyle=':', label=f'Mean ± Std: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='g', linestyle=':')
    
    ax.set_xlabel(parameter.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{parameter.replace("_", " ").title()} Distribution for {family_name}')
    ax.legend()
    
    return fig


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    facecolor: str = 'none',
    **kwargs
) -> matplotlib.patches.Ellipse:
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.
    
    Args:
        x, y: Input data
        ax: Matplotlib axes
        n_std: Number of standard deviations for ellipse
        facecolor: Color of ellipse interior
        **kwargs: Additional arguments for ellipse patch
        
    Returns:
        Matplotlib ellipse patch
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the sqrt of the variance and
    # multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_parameter_scatter(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    family_name: str = "Graph Family",
    add_ellipse: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a scatter plot of two parameters.
    
    Args:
        df: DataFrame with graph parameters
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        family_name: Name of the graph family
        add_ellipse: Whether to add confidence ellipse
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out rows with NaN in either parameter
    valid_data = df.dropna(subset=[x_param, y_param])
    
    # Scatter plot
    scatter = ax.scatter(
        valid_data[x_param],
        valid_data[y_param],
        alpha=0.7,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Add confidence ellipse
    if add_ellipse and len(valid_data) > 2:
        confidence_ellipse(
            valid_data[x_param],
            valid_data[y_param],
            ax, n_std=2.0,
            edgecolor='red', linestyle='--',
            label='95% Confidence Region'
        )
    
    # Compute correlation
    correlation = valid_data[x_param].corr(valid_data[y_param])
    
    # Add correlation text
    ax.text(
        0.05, 0.95, f'Correlation: {correlation:.2f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_title(f'{x_param.replace("_", " ").title()} vs {y_param.replace("_", " ").title()} for {family_name}')
    
    if add_ellipse:
        ax.legend()
    
    return fig


def plot_parameter_space(
    family_dfs: Dict[str, pd.DataFrame],
    x_param: str,
    y_param: str,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot multiple graph families in parameter space.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color cycle for different families
    colors = plt.cm.tab10.colors
    
    for i, (name, df) in enumerate(family_dfs.items()):
        # Filter out rows with NaN in either parameter
        valid_data = df.dropna(subset=[x_param, y_param])
        
        if len(valid_data) > 0:
            # Scatter plot for this family
            scatter = ax.scatter(
                valid_data[x_param],
                valid_data[y_param],
                alpha=0.7,
                s=50,
                edgecolor='k',
                linewidth=0.5,
                color=colors[i % len(colors)],
                label=name
            )
            
            # Add confidence ellipse
            if len(valid_data) > 2:
                confidence_ellipse(
                    valid_data[x_param],
                    valid_data[y_param],
                    ax, n_std=2.0,
                    edgecolor=colors[i % len(colors)],
                    linestyle='--',
                    alpha=0.3,
                    facecolor=colors[i % len(colors)]
                )
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_title(f'Graph Families in Parameter Space: {x_param.replace("_", " ").title()} vs {y_param.replace("_", " ").title()}')
    ax.legend()
    
    return fig


def create_parameter_dashboard(
    df: pd.DataFrame,
    family_name: str = "Graph Family",
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Create a comprehensive parameter dashboard for a graph family.
    
    Args:
        df: DataFrame with graph parameters
        family_name: Name of the graph family
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Check if we have any data
    if df.empty:
        # Create an empty figure with a message
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, 'No data available for visualization',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        return fig
    
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = plt.GridSpec(3, 4, figure=fig)
    
    # Key parameters to plot
    key_params = [
        'homophily',
        'power_law_exponent',
        'clustering_coefficient',
        'node_count',
        'avg_degree',
        'density',
        'connected_components'
    ]
    
    # Filter to available parameters
    available_params = [p for p in key_params if p in df.columns]
    
    # If no parameters are available, return empty figure
    if not available_params:
        plt.text(0.5, 0.5, 'No parameters available for visualization',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        return fig
    
    # Calculate statistics
    stats = compute_statistics(df)
    
    # Create statistical summary table
    ax_stats = fig.add_subplot(gs[0, :2])
    ax_stats.axis('tight')
    ax_stats.axis('off')
    
    table_data = []
    for param in available_params:
        if param in stats:
            s = stats[param]
            if s['mean'] is not None:  # Only add if we have valid statistics
                table_data.append([
                    param.replace('_', ' ').title(),
                    f"{s['mean']:.3f}",
                    f"{s['std']:.3f}",
                    f"{s['min']:.3f}",
                    f"{s['max']:.3f}"
                ])
    
    if table_data:  # Only create table if we have data
        table = ax_stats.table(
            cellText=table_data,
            colLabels=['Parameter', 'Mean', 'Std', 'Min', 'Max'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    ax_stats.set_title(f'Parameter Statistics for {family_name}', fontsize=14)
    
    # Key histograms
    histograms = [
        (0, 2, 'homophily'),
        (0, 3, 'power_law_exponent'),
        (1, 0, 'clustering_coefficient'),
        (1, 1, 'node_count'),
        (1, 2, 'avg_degree'),
        (1, 3, 'density')
    ]
    
    for row, col, param in histograms:
        if param in df.columns and not df[param].isna().all():
            ax = fig.add_subplot(gs[row, col])
            
            # Filter out NaN values
            values = df[param].dropna()
            
            if len(values) > 0:  # Only plot if we have valid values
                # Plot histogram
                sns.histplot(values, bins=15, kde=True, ax=ax)
                
                # Add vertical line for mean
                mean_val = values.mean()
                ax.axvline(mean_val, color='r', linestyle='--')
                
                ax.set_xlabel(param.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.set_title(f'{param.replace("_", " ").title()}')
    
    # Scatter plots for interesting parameter relationships
    scatter_plots = [
        (2, 0, 'homophily', 'clustering_coefficient'),
        (2, 1, 'node_count', 'homophily'),
        (2, 2, 'avg_degree', 'power_law_exponent'),
        (2, 3, 'clustering_coefficient', 'node_count')
    ]
    
    for row, col, x_param, y_param in scatter_plots:
        if x_param in df.columns and y_param in df.columns:
            # Check if we have enough non-NaN values
            valid_data = df.dropna(subset=[x_param, y_param])
            
            if len(valid_data) > 2:
                ax = fig.add_subplot(gs[row, col])
                
                # Scatter plot
                scatter = ax.scatter(
                    valid_data[x_param],
                    valid_data[y_param],
                    alpha=0.7,
                    s=30,
                    edgecolor='k',
                    linewidth=0.5
                )
                
                # Add confidence ellipse
                if len(valid_data) > 2:  # Need at least 3 points for ellipse
                    confidence_ellipse(
                        valid_data[x_param].values,
                        valid_data[y_param].values,
                        ax, n_std=2.0,
                        edgecolor='red', linestyle='--',
                        alpha=0.2,
                        facecolor='red'
                    )
                
                # Compute correlation
                correlation = valid_data[x_param].corr(valid_data[y_param])
                
                # Add correlation text
                ax.text(
                    0.05, 0.95, f'Corr: {correlation:.2f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                
                ax.set_xlabel(x_param.replace('_', ' ').title())
                ax.set_ylabel(y_param.replace('_', ' ').title())
                ax.set_title(f'{x_param.title()} vs {y_param.title()}')
    
    try:
        fig.tight_layout()
    except ValueError:
        # If tight_layout fails, we'll just continue without it
        pass
    
    fig.suptitle(f'Parameter Dashboard for {family_name}', fontsize=16, y=1.02)
    
    return fig


def compare_parameter_distributions(
    family_dfs: Dict[str, pd.DataFrame],
    param: str,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Compare distributions of a parameter across different graph families.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        param: Parameter to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Combined data for violin plot
    combined_data = []
    
    for name, df in family_dfs.items():
        if param in df.columns:
            # Filter out NaN values
            values = df[param].dropna()
            
            # Add to combined data
            for val in values:
                combined_data.append({'Family': name, 'Value': val})
    
    # Create DataFrame for seaborn
    combined_df = pd.DataFrame(combined_data)
    
    if not combined_df.empty:
        # Violin plot
        sns.violinplot(x='Family', y='Value', data=combined_df, ax=axes[0])
        axes[0].set_title(f'{param.replace("_", " ").title()} Distribution')
        axes[0].set_ylabel(param.replace('_', ' ').title())
        
        # Box plot
        sns.boxplot(x='Family', y='Value', data=combined_df, ax=axes[1])
        axes[1].set_title(f'{param.replace("_", " ").title()} Box Plot')
        axes[1].set_ylabel(param.replace('_', ' ').title())
    
    fig.suptitle(f'Comparing {param.replace("_", " ").title()} Across Graph Families', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig