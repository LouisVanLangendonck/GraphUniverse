"""
Specialized visualization functions for graph family analysis.

This module provides functions to visualize and analyze graph families,
including parameter distributions, graph statistics, and comparisons
between different graph samples.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from .visualizations import plot_graph_communities, plot_membership_matrix, plot_community_matrix

def plot_parameter_distributions(
    parameter_samples: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 30,
    title: str = "Parameter Distributions"
) -> plt.Figure:
    """
    Plot distributions of graph generation parameters.
    
    Args:
        parameter_samples: Dictionary of parameter names to arrays of samples
        figsize: Figure size
        bins: Number of histogram bins
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_params = len(parameter_samples)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, (param_name, samples) in enumerate(parameter_samples.items()):
        ax = axes[i]
        
        # Plot histogram
        sns.histplot(samples, bins=bins, ax=ax, kde=True)
        
        # Add statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        
        # Set labels and title
        ax.set_xlabel(param_name)
        ax.set_ylabel('Count')
        ax.set_title(f'{param_name} Distribution')
        
        # Add legend
        ax.legend()
        
        # Add statistics text
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_graph_statistics(
    graph_samples: List[nx.Graph],
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Graph Statistics Comparison"
) -> plt.Figure:
    """
    Plot comparison of key graph statistics across multiple samples.
    
    Args:
        graph_samples: List of NetworkX graphs
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate statistics for each graph
    stats = {
        'Number of Nodes': [g.number_of_nodes() for g in graph_samples],
        'Number of Edges': [g.number_of_edges() for g in graph_samples],
        'Average Degree': [np.mean([d for _, d in g.degree()]) for g in graph_samples],
        'Density': [nx.density(g) for g in graph_samples],
        'Average Clustering': [nx.average_clustering(g) for g in graph_samples],
        'Number of Components': [len(list(nx.connected_components(g))) for g in graph_samples]
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        ax = axes[i]
        
        # Create box plot
        sns.boxplot(y=values, ax=ax)
        
        # Add individual points
        sns.stripplot(y=values, ax=ax, color='red', alpha=0.5)
        
        # Set labels
        ax.set_ylabel(stat_name)
        ax.set_title(f'{stat_name} Distribution')
        
        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_community_statistics(
    graph_samples: List[nx.Graph],
    community_key: str = "community",
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Community Statistics Comparison"
) -> plt.Figure:
    """
    Plot comparison of community-related statistics across multiple samples.
    
    Args:
        graph_samples: List of NetworkX graphs
        community_key: Node attribute key for community assignment
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate community statistics for each graph
    stats = {
        'Number of Communities': [],
        'Average Community Size': [],
        'Community Size Std': [],
        'Modularity': [],
        'Conductance': [],
        'Coverage': []
    }
    
    for graph in graph_samples:
        # Get community assignments
        communities = {}
        for node, attrs in graph.nodes(data=True):
            comm = attrs.get(community_key, 0)
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        
        # Calculate statistics
        comm_sizes = [len(nodes) for nodes in communities.values()]
        stats['Number of Communities'].append(len(communities))
        stats['Average Community Size'].append(np.mean(comm_sizes))
        stats['Community Size Std'].append(np.std(comm_sizes))
        
        # Calculate modularity
        modularity = nx.community.modularity(graph, communities.values())
        stats['Modularity'].append(modularity)
        
        # Calculate conductance (average)
        conductances = []
        for comm in communities.values():
            if len(comm) > 0 and len(comm) < graph.number_of_nodes():
                conductance = nx.community.conductance(graph, comm)
                conductances.append(conductance)
        stats['Conductance'].append(np.mean(conductances) if conductances else 0)
        
        # Calculate coverage
        coverage = sum(len(comm) for comm in communities.values()) / graph.number_of_nodes()
        stats['Coverage'].append(coverage)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        ax = axes[i]
        
        # Create box plot
        sns.boxplot(y=values, ax=ax)
        
        # Add individual points
        sns.stripplot(y=values, ax=ax, color='red', alpha=0.5)
        
        # Set labels
        ax.set_ylabel(stat_name)
        ax.set_title(f'{stat_name} Distribution')
        
        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def create_graph_family_dashboard(
    graph_samples: List[nx.Graph],
    parameter_samples: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (20, 15),
    title: str = "Graph Family Analysis Dashboard"
) -> plt.Figure:
    """
    Create a comprehensive dashboard for analyzing a graph family.
    
    Args:
        graph_samples: List of NetworkX graphs
        parameter_samples: Dictionary of parameter names to arrays of samples
        figsize: Figure size
        title: Dashboard title
        
    Returns:
        Matplotlib figure
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # 1. Parameter distributions (top row)
    ax1 = fig.add_subplot(gs[0, :])
    plot_parameter_distributions(parameter_samples, ax=ax1)
    
    # 2. Graph statistics (middle row)
    ax2 = fig.add_subplot(gs[1, :])
    plot_graph_statistics(graph_samples, ax=ax2)
    
    # 3. Community statistics (bottom row)
    ax3 = fig.add_subplot(gs[2, :])
    plot_community_statistics(graph_samples, ax=ax3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_graph_family_comparison(
    graph_families: Dict[str, List[nx.Graph]],
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Graph Family Comparison"
) -> plt.Figure:
    """
    Compare statistics across different graph families.
    
    Args:
        graph_families: Dictionary mapping family names to lists of graphs
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate statistics for each family
    stats = {
        'Number of Nodes': [],
        'Number of Edges': [],
        'Average Degree': [],
        'Density': [],
        'Average Clustering': [],
        'Number of Components': []
    }
    
    family_names = []
    for family_name, graphs in graph_families.items():
        family_names.append(family_name)
        for graph in graphs:
            stats['Number of Nodes'].append(graph.number_of_nodes())
            stats['Number of Edges'].append(graph.number_of_edges())
            stats['Average Degree'].append(np.mean([d for _, d in graph.degree()]))
            stats['Density'].append(nx.density(graph))
            stats['Average Clustering'].append(nx.average_clustering(graph))
            stats['Number of Components'].append(len(list(nx.connected_components(graph))))
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        ax = axes[i]
        
        # Create box plot
        sns.boxplot(x=family_names, y=values, ax=ax)
        
        # Add individual points
        sns.stripplot(x=family_names, y=values, ax=ax, color='red', alpha=0.5)
        
        # Set labels
        ax.set_xlabel('Graph Family')
        ax.set_ylabel(stat_name)
        ax.set_title(f'{stat_name} by Family')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig 