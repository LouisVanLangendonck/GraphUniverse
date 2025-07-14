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
from graph_universe.model import GraphSample

def plot_parameter_distributions(parameter_samples: List[Dict[str, List[float]]], family_names: List[str], ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot distributions of parameters across multiple graph families.
    
    Args:
        parameter_samples: List of parameter dictionaries for each family
        family_names: Names of the families
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Get all unique parameters
    all_params = set()
    for params in parameter_samples:
        all_params.update(params.keys())
    
    # Create figure if no axes provided
    if ax is None:
        n_params = len(all_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
    else:
        fig = ax.figure
        axes = [ax]
    
    # Plot each parameter
    for i, param in enumerate(sorted(all_params)):
        if i < len(axes):  # Only plot if we have an axis available
            ax = axes[i]
            
            # Plot distribution for each family
            for family_idx, (params, name) in enumerate(zip(parameter_samples, family_names)):
                if param in params:
                    values = params[param]
                    sns.kdeplot(values, ax=ax, label=name)
            
            ax.set_title(f'{param} Distribution')
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.legend()
    
    # Hide unused axes
    for i in range(len(all_params), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_graph_statistics(
    graph_samples: List[GraphSample],
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Graph Statistics Comparison",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot comparison of key graph statistics across multiple samples.
    
    Args:
        graph_samples: List of GraphSample objects
        figsize: Figure size
        title: Plot title
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Calculate statistics for each graph
    stats = {
        'Number of Nodes': [g.n_nodes for g in graph_samples],
        'Number of Edges': [g.graph.number_of_edges() for g in graph_samples],
        'Average Degree': [np.mean([d for _, d in g.graph.degree()]) for g in graph_samples],
        'Density': [nx.density(g.graph) for g in graph_samples],
        'Average Clustering': [nx.average_clustering(g.graph) for g in graph_samples],
        'Number of Components': [len(list(nx.connected_components(g.graph))) for g in graph_samples]
    }
    
    if ax is None:
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
    else:
        # Use single provided axis
        fig = ax.figure
        axes = [ax]
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        if i < len(axes):  # Only plot if we have an axis available
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
    
    if ax is None:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Hide unused axes
        for i in range(len(stats), len(axes)):
            axes[i].axis('off')
    
    return fig

def plot_community_statistics(
    graph_samples: List[GraphSample],
    community_key: str = "community",
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Community Statistics Comparison",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot comparison of community-related statistics across multiple samples.
    
    Args:
        graph_samples: List of GraphSample objects
        community_key: Node attribute key for community assignment
        figsize: Figure size
        title: Plot title
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Calculate community statistics for each graph
    stats = {
        'Number of Communities': [],
        'Average Community Size': [],
        'Community Size Std': [],
        'Modularity': [],
        'Internal Density': [],  # Replace conductance with internal density
        'Coverage': []
    }
    
    for graph_sample in graph_samples:
        graph = graph_sample.graph
        # Get community assignments using community_labels from GraphSample
        communities = {}
        for node, comm in enumerate(graph_sample.community_labels):
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
        
        # Calculate internal density for each community
        internal_densities = []
        for comm in communities.values():
            if len(comm) > 1:  # Need at least 2 nodes for density
                subgraph = graph.subgraph(comm)
                n = subgraph.number_of_nodes()
                m = subgraph.number_of_edges()
                max_edges = (n * (n - 1)) / 2
                density = m / max_edges if max_edges > 0 else 0
                internal_densities.append(density)
        
        stats['Internal Density'].append(np.mean(internal_densities) if internal_densities else 0)
        
        # Calculate coverage
        coverage = sum(len(comm) for comm in communities.values()) / graph.number_of_nodes()
        stats['Coverage'].append(coverage)
    
    if ax is None:
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
    else:
        # Use single provided axis
        fig = ax.figure
        axes = [ax]
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        if i < len(axes):  # Only plot if we have an axis available
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
    
    if ax is None:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Hide unused axes
        for i in range(len(stats), len(axes)):
            axes[i].axis('off')
    
    return fig

def create_graph_family_dashboard(graphs: List[GraphSample], parameters: Dict[str, List[float]]) -> plt.Figure:
    """
    Create a comprehensive dashboard for a graph family.
    
    Args:
        graphs: List of graphs in the family
        parameters: Dictionary of parameter values
        
    Returns:
        Matplotlib figure with dashboard
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Parameter distributions
    ax1 = fig.add_subplot(gs[0, 0])
    plot_parameter_distributions([parameters], ["Family"], ax=ax1)
    
    # Graph statistics
    ax2 = fig.add_subplot(gs[0, 1])
    plot_graph_statistics(graphs, ax=ax2)
    
    # Community statistics
    ax3 = fig.add_subplot(gs[1, 0])
    plot_community_statistics(graphs, ax=ax3)
    
    # Graph comparison
    ax4 = fig.add_subplot(gs[1, 1])
    plot_graph_family_comparison(graphs, ax=ax4)
    
    plt.tight_layout()
    return fig

def plot_graph_family_comparison(
    graph_samples: List[GraphSample],
    figsize: Tuple[int, int] = (15, 10),
    title: str = "Graph Statistics",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Compare statistics for a set of graph samples.
    
    Args:
        graph_samples: List of GraphSample objects
        figsize: Figure size
        title: Plot title
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Calculate statistics for each graph
    stats = {
        'Number of Nodes': [],
        'Number of Edges': [],
        'Average Degree': [],
        'Density': [],
        'Average Clustering': [],
        'Number of Components': []
    }
    
    for graph_sample in graph_samples:
        graph = graph_sample.graph
        stats['Number of Nodes'].append(graph.number_of_nodes())
        stats['Number of Edges'].append(graph.number_of_edges())
        stats['Average Degree'].append(np.mean([d for _, d in graph.degree()]))
        stats['Density'].append(nx.density(graph))
        stats['Average Clustering'].append(nx.average_clustering(graph))
        stats['Number of Components'].append(len(list(nx.connected_components(graph))))
    
    if ax is None:
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
    else:
        # Use single provided axis
        fig = ax.figure
        axes = [ax]
    
    # Plot each statistic
    for i, (stat_name, values) in enumerate(stats.items()):
        if i < len(axes):  # Only plot if we have an axis available
            ax = axes[i]
            
            # Create violin plot with individual points
            sns.violinplot(y=values, ax=ax)
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
    
    if ax is None:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Hide unused axes
        for i in range(len(stats), len(axes)):
            axes[i].axis('off')
    
    return fig 