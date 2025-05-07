"""
Custom Streamlit widgets and UI components for MMSB Explorer.

This module provides specialized widgets and components for the MMSB
Streamlit application, including interactive graph visualization,
parameter controls, and dashboard components.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MMSB modules
from mmsb.model import GraphUniverse, GraphSample
from utils.visualizations import (
    plot_graph_communities,
    plot_membership_matrix,
    plot_community_matrix
)


def universe_parameters_widget() -> Dict[str, Any]:
    """
    Widget for collecting universe parameters.
    
    Returns:
        Dictionary of universe parameters
    """
    st.markdown("### Universe Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.slider("Number of communities (K)", min_value=5, max_value=100, value=20, 
                     help="Total number of community types in the universe")
        block_structure = st.selectbox(
            "Block structure",
            ["assortative", "disassortative", "core-periphery", "hierarchical"],
            help="Structure of the edge probability matrix"
        )
    
    with col2:
        feature_dim = st.slider("Feature dimension", min_value=0, max_value=128, value=32, 
                               help="Dimension of node features (0 for no features)")
        overlap_structure = st.selectbox(
            "Overlap structure",
            ["modular", "hierarchical", "hub-spoke"],
            help="Structure of community overlaps"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        edge_density = st.slider("Edge density within communities", min_value=0.01, max_value=0.5, value=0.1, 
                                step=0.01, help="Probability of edges within communities")
        
    with col4:
        inter_community_density = st.slider("Edge density between communities", min_value=0.001, max_value=0.1, 
                                          value=0.01, step=0.001, 
                                          help="Probability of edges between different communities")
    
    overlap_density = st.slider("Community overlap density", min_value=0.0, max_value=0.5, value=0.2, 
                              step=0.01, help="Density of community overlaps")
    
    return {
        "K": K,
        "feature_dim": feature_dim,
        "block_structure": block_structure,
        "overlap_structure": overlap_structure,
        "edge_density": edge_density,
        "inter_community_density": inter_community_density,
        "overlap_density": overlap_density
    }


def universe_stats_widget(universe: GraphUniverse):
    """
    Display statistics for a graph universe.
    
    Args:
        universe: The graph universe to display stats for
    """
    st.markdown("### Universe Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Communities", universe.K)
    
    with col2:
        st.metric("Feature Dimension", universe.feature_dim)
    
    with col3:
        avg_density = universe.P.mean()
        st.metric("Average Edge Density", f"{avg_density:.4f}")
    
    # Show additional stats in an expandable section
    with st.expander("Additional Universe Statistics"):
        # Edge probability matrix stats
        p_min = universe.P.min()
        p_max = universe.P.max()
        p_median = np.median(universe.P)
        
        # Co-membership matrix stats
        co_min = universe.community_co_membership.min()
        co_max = universe.community_co_membership.max()
        co_median = np.median(universe.community_co_membership)
        
        # Create dataframe for display
        stats_df = pd.DataFrame({
            "Statistic": [
                "Min Edge Probability",
                "Max Edge Probability",
                "Median Edge Probability",
                "Min Co-membership Probability",
                "Max Co-membership Probability",
                "Median Co-membership Probability"
            ],
            "Value": [
                f"{p_min:.4f}",
                f"{p_max:.4f}",
                f"{p_median:.4f}",
                f"{co_min:.4f}",
                f"{co_max:.4f}",
                f"{co_median:.4f}"
            ]
        })
        
        st.table(stats_df)


def graph_parameters_widget(universe: GraphUniverse) -> Dict[str, Any]:
    """
    Widget for collecting graph sampling parameters.
    
    Args:
        universe: The graph universe to sample from
        
    Returns:
        Dictionary of graph parameters
    """
    st.markdown("### Sampling Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_communities = st.slider("Number of communities", min_value=2, 
                                 max_value=min(20, universe.K), value=min(5, universe.K))
        n_nodes = st.slider("Number of nodes", min_value=50, max_value=500, value=200)
    
    with col2:
        sampling_method = st.selectbox(
            "Community sampling method",
            ["random", "similar", "diverse", "correlated"],
            help="Method for selecting community subsets"
        )
        avg_memberships = st.slider("Average communities per node", min_value=1.0, max_value=5.0, value=1.5,
                                  step=0.1, help="Controls community overlap")
    
    col3, col4 = st.columns(2)
    
    with col3:
        membership_concentration = st.slider("Membership concentration", min_value=1.0, max_value=20.0, value=5.0,
                                           step=0.5, help="Controls how evenly distributed memberships are")
        
    with col4:
        degree_heterogeneity = st.slider("Degree heterogeneity", min_value=0.0, max_value=1.0, value=0.5)
    
    edge_noise = st.slider("Edge noise", min_value=0.0, max_value=1.0, value=0.0)
    
    return {
        "n_communities": n_communities,
        "n_nodes": n_nodes,
        "sampling_method": sampling_method,
        "avg_memberships": avg_memberships,
        "membership_concentration": membership_concentration,
        "degree_heterogeneity": degree_heterogeneity,
        "edge_noise": edge_noise
    }


def graph_stats_widget(graph: GraphSample):
    """
    Display statistics for a graph sample.
    
    Args:
        graph: The graph sample to display stats for
    """
    st.markdown("### Graph Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", graph.n_nodes)
    
    with col2:
        st.metric("Edges", graph.graph.number_of_edges())
    
    with col3:
        avg_degree = 2 * graph.graph.number_of_edges() / graph.n_nodes
        st.metric("Average Degree", f"{avg_degree:.2f}")
    
    with col4:
        n_communities = len(graph.communities)
        st.metric("Communities", n_communities)
    
    # Show additional stats in an expandable section
    with st.expander("Additional Graph Statistics"):
        G = graph.graph
        
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = "N/A"
            
        try:
            density = nx.density(G)
        except:
            density = "N/A"
            
        try:
            n_components = nx.number_connected_components(G)
        except:
            n_components = "N/A"
            
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_size = len(largest_cc)
        except:
            largest_cc_size = "N/A"
        
        # Create dataframe for display
        stats_df = pd.DataFrame({
            "Statistic": [
                "Graph Density",
                "Clustering Coefficient",
                "Connected Components",
                "Largest Component Size",
                "Min Degree",
                "Max Degree"
            ],
            "Value": [
                f"{density}",
                f"{clustering}",
                f"{n_components}",
                f"{largest_cc_size}",
                min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else "N/A",
                max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else "N/A"
            ]
        })
        
        st.table(stats_df)


def benchmark_parameters_widget(universe: GraphUniverse) -> Dict[str, Any]:
    """
    Widget for collecting benchmark generation parameters.
    
    Args:
        universe: The graph universe to sample from
        
    Returns:
        Dictionary of benchmark parameters
    """
    st.markdown("### Benchmark Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_pretrain = st.slider("Number of pretraining graphs", min_value=5, max_value=100, value=20)
        min_communities = st.slider("Min communities per graph", min_value=2, 
                                  max_value=min(10, universe.K - 5), value=min(3, universe.K - 5))
        
    with col2:
        n_transfer = st.slider("Number of transfer graphs", min_value=5, max_value=50, value=10)
        max_communities = st.slider("Max communities per graph", min_value=min_communities + 1, 
                                  max_value=min(15, universe.K), value=min(8, universe.K))
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_nodes = st.slider("Min nodes per graph", min_value=50, max_value=500, value=100)
    
    with col4:
        max_nodes = st.slider("Max nodes per graph", min_value=min_nodes + 50, max_value=1000, value=300)
    
    transfer_modes = st.multiselect(
        "Transfer modes",
        ["new_combinations", "rare_communities", "novel_communities"],
        default=["new_combinations"],
        help="Types of distributional shift in transfer graphs"
    )
    
    if not transfer_modes:
        transfer_modes = ["new_combinations"]  # Default if none selected
    
    transfer_difficulty = st.slider("Transfer difficulty", min_value=0.1, max_value=0.9, value=0.5,
                                  step=0.1, help="Controls how challenging the transfer task is")
    
    return {
        "n_pretrain": n_pretrain,
        "n_transfer": n_transfer,
        "min_communities": min_communities,
        "max_communities": max_communities,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "transfer_modes": transfer_modes,
        "transfer_difficulty": transfer_difficulty
    }


def benchmark_stats_widget(pretrain_graphs: List[GraphSample], transfer_graphs: List[GraphSample]):
    """
    Display statistics for a benchmark dataset.
    
    Args:
        pretrain_graphs: List of pretraining graphs
        transfer_graphs: List of transfer graphs
    """
    st.markdown("### Benchmark Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pretraining Graphs", len(pretrain_graphs))
    
    with col2:
        st.metric("Transfer Graphs", len(transfer_graphs))
    
    with col3:
        total_nodes = sum(g.n_nodes for g in pretrain_graphs) + sum(g.n_nodes for g in transfer_graphs)
        st.metric("Total Nodes", total_nodes)
    
    # Show additional stats in an expandable section
    with st.expander("Additional Benchmark Statistics"):
        # Calculate statistics for each set
        pretrain_nodes = [g.n_nodes for g in pretrain_graphs]
        pretrain_edges = [g.graph.number_of_edges() for g in pretrain_graphs]
        pretrain_communities = [len(g.communities) for g in pretrain_graphs]
        
        transfer_nodes = [g.n_nodes for g in transfer_graphs]
        transfer_edges = [g.graph.number_of_edges() for g in transfer_graphs]
        transfer_communities = [len(g.communities) for g in transfer_graphs]
        
        # Create dataframe for display
        stats_df = pd.DataFrame({
            "Statistic": [
                "Avg Nodes (Pretrain)",
                "Avg Edges (Pretrain)",
                "Avg Communities (Pretrain)",
                "Avg Nodes (Transfer)",
                "Avg Edges (Transfer)",
                "Avg Communities (Transfer)"
            ],
            "Value": [
                f"{np.mean(pretrain_nodes):.1f}",
                f"{np.mean(pretrain_edges):.1f}",
                f"{np.mean(pretrain_communities):.1f}",
                f"{np.mean(transfer_nodes):.1f}",
                f"{np.mean(transfer_edges):.1f}",
                f"{np.mean(transfer_communities):.1f}"
            ]
        })
        
        st.table(stats_df)


def community_analysis_widget(graph: GraphSample, universe: GraphUniverse):
    """
    Widget for analyzing community structure.
    
    Args:
        graph: GraphSample object to analyze
        universe: GraphUniverse object containing graph parameters
    """
    st.markdown("## Community Analysis")
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs([
        "Community Distribution",
        "Community Structure",
        "Community Metrics"
    ])
    
    with analysis_tabs[0]:
        # Plot community distribution
        fig = plot_membership_matrix(graph)
        st.pyplot(fig)
        
        st.markdown("""
        This visualization shows the distribution of nodes across communities.
        The height of each bar represents the number of nodes in that community,
        and the percentage labels show the relative size of each community.
        """)
    
    with analysis_tabs[1]:
        # Plot graph with communities
        fig = plot_graph_communities(graph)
        st.pyplot(fig)
        
        st.markdown("""
        This visualization shows the network structure with nodes colored by community.
        The layout is computed using a force-directed algorithm to show the community structure.
        """)
    
    with analysis_tabs[2]:
        # Calculate and display community metrics
        metrics = {}
        
        # Number of communities
        metrics["Number of Communities"] = len(graph.communities)
        
        # Community sizes
        community_sizes = []
        for comm in graph.communities:
            size = np.sum(graph.community_labels == graph.communities.index(comm))
            community_sizes.append(size)
        
        metrics["Average Community Size"] = np.mean(community_sizes)
        metrics["Min Community Size"] = np.min(community_sizes)
        metrics["Max Community Size"] = np.max(community_sizes)
        
        # Modularity
        try:
            modularity = nx.community.modularity(
                graph.graph,
                [np.where(graph.community_labels == i)[0].tolist() for i in range(len(graph.communities))]
            )
            metrics["Modularity"] = modularity
        except:
            metrics["Modularity"] = "N/A"
        
        # Intra-community density
        try:
            intra_density = graph.extract_parameters()["intra_community_density"]
            metrics["Intra-community Density"] = intra_density
        except:
            metrics["Intra-community Density"] = "N/A"
        
        # Display metrics
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })
        st.dataframe(metrics_df)
        
        st.markdown("""
        These metrics provide insights into the community structure:
        - **Number of Communities**: Total number of distinct communities
        - **Community Sizes**: Statistics about the size distribution of communities
        - **Modularity**: Measure of community structure strength (-1 to 1)
        - **Intra-community Density**: Average edge density within communities
        """)


def interactive_graph_widget(graph: GraphSample):
    """
    Interactive graph visualization widget.
    
    Args:
        graph: The graph sample to visualize
    """
    st.markdown("### Interactive Graph Visualization")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layout = st.selectbox("Layout", ["spring", "kamada_kawai", "spectral", "circular"])
    
    with col2:
        node_size = st.slider("Node size", min_value=10, max_value=200, value=50)
    
    with col3:
        edge_width = st.slider("Edge width", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    
    # Show labels for smaller graphs
    show_labels = st.checkbox("Show labels", value=False)
    
    # Plot the graph
    fig = plot_graph_communities(
        graph.graph,
        layout=layout,
        node_size=node_size,
        edge_width=edge_width,
        with_labels=show_labels
    )
    
    st.pyplot(fig)


def feature_analysis_widget(graph: GraphSample):
    """
    Widget for analyzing node features.
    
    Args:
        graph: The graph sample to analyze
    """
    if graph.features is None:
        st.warning("This graph doesn't have node features.")
        return
    
    st.markdown("### Feature Analysis")
    
    # Feature statistics
    feature_mean = np.mean(graph.features, axis=0)
    feature_std = np.std(graph.features, axis=0)
    feature_min = np.min(graph.features, axis=0)
    feature_max = np.max(graph.features, axis=0)
    
    # Show statistics for first few features
    n_show = min(5, graph.features.shape[1])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Feature Dimension", graph.features.shape[1])
    
    with col2:
        avg_norm = np.linalg.norm(graph.features, axis=1).mean()
        st.metric("Average Feature Norm", f"{avg_norm:.2f}")
    
    with col3:
        feature_sparsity = (graph.features == 0).mean() * 100
        st.metric("Feature Sparsity", f"{feature_sparsity:.1f}%")
    
    # Feature visualization
    feature_tabs = st.tabs([
        "Feature Statistics",
        "Feature Visualization",
        "Feature-Community Correlation"
    ])
    
    with feature_tabs[0]:
        # Show statistics for first few features
        feature_stats_df = pd.DataFrame({
            "Feature": [f"Feature {i}" for i in range(n_show)],
            "Mean": feature_mean[:n_show],
            "Std Dev": feature_std[:n_show],
            "Min": feature_min[:n_show],
            "Max": feature_max[:n_show]
        })
        
        st.table(feature_stats_df)
        
        if graph.features.shape[1] > n_show:
            st.info(f"Showing statistics for first {n_show} features out of {graph.features.shape[1]}.")
    
    with feature_tabs[1]:
        # Visualize features with dimensionality reduction
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(graph.features)
        
        # Get primary community for each node for coloring
        primary_communities = np.argmax(graph.membership_vectors, axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cmap = plt.get_cmap("tab20")
        
        for c_idx, c in enumerate(graph.communities):
            # Get nodes with this as primary community
            mask = primary_communities == c_idx
            
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    label=f"C{c}",
                    color=cmap(c_idx % 20),
                    alpha=0.7
                )
        
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Node Features Visualization (PCA)")
        ax.legend(title="Primary Community")
        
        st.pyplot(fig)
        
        # Show explained variance
        explained_variance = pca.explained_variance_ratio_
        st.markdown(f"Explained variance by two principal components: {sum(explained_variance)*100:.2f}%")
    
    with feature_tabs[2]:
        # Calculate correlation between features and community memberships
        n_plot_features = min(20, graph.features.shape[1])
        correlation_matrix = np.zeros((len(graph.communities), n_plot_features))
        
        for c_idx, c in enumerate(graph.communities):
            for f_idx in range(n_plot_features):
                correlation = np.corrcoef(graph.membership_vectors[:, c_idx], graph.features[:, f_idx])[0, 1]
                correlation_matrix[c_idx, f_idx] = correlation
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Correlation Coefficient")
        
        # Set labels
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Community")
        ax.set_yticks(range(len(graph.communities)))
        ax.set_yticklabels([f"C{c}" for c in graph.communities])
        
        ax.set_title("Correlation between Features and Community Memberships")
        
        st.pyplot(fig)


def community_transfer_analysis_widget(pretrain_graphs: List[GraphSample], transfer_graphs: List[GraphSample]):
    """
    Widget for analyzing community transfer patterns.
    
    Args:
        pretrain_graphs: List of pretraining graphs
        transfer_graphs: List of transfer graphs
    """
    st.markdown("### Community Transfer Analysis")
    
    # Count community occurrences
    community_counts = {}
    
    for graph in pretrain_graphs:
        for comm in graph.communities:
            if comm not in community_counts:
                community_counts[comm] = {"pretrain": 0, "transfer": 0}
            community_counts[comm]["pretrain"] += 1
    
    for graph in transfer_graphs:
        for comm in graph.communities:
            if comm not in community_counts:
                community_counts[comm] = {"pretrain": 0, "transfer": 0}
            community_counts[comm]["transfer"] += 1
    
    # Create dataframe for visualization
    comm_df = pd.DataFrame.from_dict(community_counts, orient='index')
    comm_df.index.name = "Community"
    comm_df.reset_index(inplace=True)
    comm_df = comm_df.sort_values("Community")
    
    # Plot community distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    
    x = np.arange(len(comm_df))
    ax.bar(x - width/2, comm_df["pretrain"], width, label="Pretraining")
    ax.bar(x + width/2, comm_df["transfer"], width, label="Transfer")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in comm_df["Community"]], rotation=45, ha="right")
    ax.set_xlabel("Community")
    ax.set_ylabel("Frequency")
    ax.set_title("Community Distribution in Pretraining vs. Transfer Graphs")
    ax.legend()
    
    st.pyplot(fig)
    
    # Calculate transfer statistics
    pretrain_communities = set()
    for graph in pretrain_graphs:
        pretrain_communities.update(graph.communities)
    
    transfer_communities = set()
    for graph in transfer_graphs:
        transfer_communities.update(graph.communities)
    
    novel_communities = transfer_communities - pretrain_communities
    shared_communities = transfer_communities.intersection(pretrain_communities)
    
    # Show transfer statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Communities", len(pretrain_communities.union(transfer_communities)))
    
    with col2:
        st.metric("Novel Communities", len(novel_communities))
    
    with col3:
        st.metric("Shared Communities", len(shared_communities))
    
    # Show community details
    with st.expander("Community Details"):
        # Create dataframe for all communities
        all_communities = list(pretrain_communities.union(transfer_communities))
        all_communities.sort()
        
        details_data = []
        for comm in all_communities:
            pretrain_count = sum(1 for g in pretrain_graphs if comm in g.communities)
            transfer_count = sum(1 for g in transfer_graphs if comm in g.communities)
            
            details_data.append({
                "Community": f"C{comm}",
                "In Pretraining": pretrain_count,
                "In Transfer": transfer_count,
                "Pretrain %": pretrain_count / len(pretrain_graphs) * 100 if pretrain_graphs else 0,
                "Transfer %": transfer_count / len(transfer_graphs) * 100 if transfer_graphs else 0,
                "Status": "Novel" if comm in novel_communities else "Shared"
            })
        
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df)


def save_benchmark_widget(benchmark, pretrain_graphs, transfer_graphs):
    """
    Widget for saving a benchmark dataset.
    
    Args:
        benchmark: The benchmark object
        pretrain_graphs: List of pretraining graphs
        transfer_graphs: List of transfer graphs
    """
    st.markdown("### Save Benchmark")
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_dir = st.text_input("Save directory", "benchmark_data")
    
    with col2:
        save_format = st.selectbox("Save format", ["networkx", "pyg", "dgl"])
    
    if st.button("Save Benchmark"):
        with st.spinner("Saving benchmark..."):
            try:
                # Create directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                
                # Save benchmark
                benchmark.save_benchmark(
                    directory=save_dir,
                    pretraining_graphs=pretrain_graphs,
                    transfer_graphs=transfer_graphs,
                    format=save_format
                )
                
                st.success(f"Benchmark saved to {save_dir}")
            except Exception as e:
                st.error(f"Error saving benchmark: {str(e)}")


def graph_comparison_widget(graphs: List[GraphSample], max_graphs: int = 4):
    """
    Widget for comparing multiple graphs side by side.
    
    Args:
        graphs: List of GraphSample objects to compare
        max_graphs: Maximum number of graphs to display
    """
    if not graphs:
        st.warning("No graphs to compare")
        return
    
    # Select graphs to compare
    st.markdown("### Select Graphs to Compare")
    selected_indices = st.multiselect(
        "Choose graphs to compare (up to 4)",
        options=range(len(graphs)),
        default=list(range(min(max_graphs, len(graphs)))),
        format_func=lambda x: f"Graph {x}"
    )
    
    if not selected_indices:
        st.warning("Please select at least one graph to compare")
        return
    
    # Limit to max_graphs
    selected_indices = selected_indices[:max_graphs]
    selected_graphs = [graphs[i] for i in selected_indices]
    
    # Create figure with subplots
    n_graphs = len(selected_graphs)
    n_cols = min(2, n_graphs)
    n_rows = (n_graphs + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    # Plot each graph
    for i, (graph, ax) in enumerate(zip(selected_graphs, axes)):
        # Get community information
        community_labels = graph.community_labels
        communities = graph.communities
        
        # Create position layout
        pos = nx.spring_layout(graph.graph, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(
            graph.graph,
            pos,
            alpha=0.2,
            width=0.5,
            ax=ax
        )
        
        # Draw nodes for each community
        for comm in sorted(set(communities)):
            # Get nodes in this community
            comm_nodes = [n for n, c in enumerate(community_labels) if communities[c] == comm]
            
            if comm_nodes:
                nx.draw_networkx_nodes(
                    graph.graph,
                    pos,
                    nodelist=comm_nodes,
                    node_color=[f"C{comm}"] * len(comm_nodes),
                    node_size=30,
                    alpha=0.8,
                    ax=ax
                )
        
        ax.set_title(f"Graph {selected_indices[i]}")
        ax.axis("off")
    
    # Hide unused axes
    for i in range(len(selected_graphs), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Community distribution comparison
    st.markdown("#### Community Distribution Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, graph in enumerate(selected_graphs):
        # Count nodes in each community
        community_counts = {}
        for comm in graph.communities:
            count = np.sum(graph.community_labels == graph.communities.index(comm))
            community_counts[comm] = count
        
        # Get distribution
        communities_list = sorted(community_counts.keys())
        counts = [community_counts[comm] for comm in communities_list]
        frequencies_norm = np.array(counts) / graph.n_nodes
        
        # Plot
        ax.plot(communities_list, frequencies_norm, marker='o', label=f"Graph {selected_indices[i]}")
    
    ax.set_xlabel("Community")
    ax.set_ylabel("Fraction of Nodes")
    ax.set_title("Community Distribution Comparison")
    ax.legend()
    
    # Set integer x-ticks
    max_comm = max(max(g.communities) for g in selected_graphs)
    ax.set_xticks(range(max_comm + 1))
    
    st.pyplot(fig)


def plot_graph_analysis(graph):
    """Plot graph analysis visualizations."""
    # Graph visualization
    st.subheader("Graph Visualization")
    st.pyplot(plot_graph_communities(graph))
    
    # Community distribution
    st.subheader("Community Distribution")
    st.pyplot(plot_membership_matrix(graph))  # Pass the graph object directly
    
    # Parameter analysis
    st.subheader("Parameter Analysis")
    params = graph.extract_parameters()
    st.write(params)
    
    # Feature analysis if available
    if hasattr(graph, 'features') and graph.features is not None:
        st.subheader("Feature Analysis")
        st.pyplot(plot_feature_heatmap(graph.features, graph.community_labels))