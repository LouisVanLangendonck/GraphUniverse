"""
Streamlit application for Mixed-Membership Stochastic Block Model visualization.

This application allows users to:
1. Generate and visualize graph universes with overlapping community structure
2. Sample and explore individual graphs from the universe
3. Generate pretraining and transfer learning benchmarks
4. Visualize community structures and network properties
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MMSB modules
from mmsb.model import GraphUniverse, GraphSample, MMSBBenchmark
from mmsb.feature_regimes import FeatureRegimeGenerator, NeighborhoodFeatureAnalyzer, FeatureRegimeLabelGenerator, GenerativeRuleBasedLabeler
from utils.visualizations import (
    plot_graph_communities, 
    plot_membership_matrix,
    plot_community_matrix,
    plot_community_graph,
    plot_degree_distribution,
    plot_community_overlap_distribution,
    create_dashboard,
)
from utils.sampler import GraphSampler
from utils.parameter_analysis import (
    analyze_graph_family,
    compute_statistics,
    compare_graph_families,
    plot_parameter_distribution,
    plot_parameter_scatter,
    plot_parameter_space,
    create_parameter_dashboard,
    compare_parameter_distributions,
    analyze_community_connectivity,
    visualize_community_connectivity,
)
from utils.motif_and_role_analysis import MotifRoleAnalyzer
from motif_and_role_analysis_integration import add_motif_role_analysis_tab, add_motif_role_analysis_page

# Set page config
st.set_page_config(
    page_title="MMSB Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B89DC;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3D5A80;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #4B89DC;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #F0F4F8;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #FFF3CD;
        padding: 0.5rem;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for persistence across reruns
if 'universe' not in st.session_state:
    st.session_state.universe = None
if 'current_graph' not in st.session_state:
    st.session_state.current_graph = None
if 'benchmark' not in st.session_state:
    st.session_state.benchmark = None
if 'pretrain_graphs' not in st.session_state:
    st.session_state.pretrain_graphs = []
if 'transfer_graphs' not in st.session_state:
    st.session_state.transfer_graphs = []


# Main header
st.markdown('<div class="main-header">Mixed-Membership Stochastic Block Model Explorer</div>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates the Mixed-Membership Stochastic Block Model (MMSB) for generating
graph datasets with overlapping community structure. The model supports transfer learning
by sampling graphs from subsets of a larger universe.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Universe Creation", "Graph Sampling", "Benchmark Generation", "Community Analysis", "Feature Analysis", "Parameter Space Analysis", "Motif and Role Analysis", "Neighborhood Analysis"]
)

# Universe Creation Page
if page == "Universe Creation":
    st.markdown('<div class="section-header">Universe Creation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    The <b>Graph Universe</b> defines the core generative structure from which individual graphs are sampled.
    It consists of:
    <ul>
        <li>A set of communities with edge probability patterns</li>
        <li>Optional feature prototypes for each community</li>
        <li>Community co-membership probabilities that model how communities overlap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Universe parameters
    st.markdown('<div class="subsection-header">Universe Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.slider("Number of communities (K)", min_value=5, max_value=100, value=20, 
                     help="Total number of community types in the universe")
        block_structure = st.selectbox(
            "Block structure",
            ["assortative", "disassortative", "core-periphery", "hierarchical", "random_blocks"],
            help="Structure of the edge probability matrix"
        )
        mixed_membership = st.checkbox(
            "Enable Mixed Membership",
            value=True,
            help="If enabled, nodes can belong to multiple communities. If disabled, each node belongs to exactly one community (standard SBM)."
        )
    
    with col2:
        # Group feature parameters in an expander for cleaner UI
        with st.expander("Feature Generation Parameters", expanded=True):
            feature_dim = st.slider("Feature dimension", min_value=0, max_value=128, value=32, 
                                help="Dimension of node features (0 for no features)")
            
            # Only show feature parameters if features are enabled
            if feature_dim > 0:
                # Feature Regime Parameters
                st.markdown("### Feature Regime Parameters")
                intra_community_regime_similarity = st.slider(
                    "Intra-community Regime Similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="How similar regimes within the same community should be (0=very different, 1=identical)"
                )
                
                inter_community_regime_similarity = st.slider(
                    "Inter-community Regime Similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.1,
                    help="How similar regimes between different communities should be (0=very different, 1=identical)"
                )
                
                regimes_per_community = st.slider(
                    "Regimes per Community",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Number of feature regimes per community"
                )
            else:
                intra_community_regime_similarity = 0.8  # Default value
                inter_community_regime_similarity = 0.2  # Default value
                regimes_per_community = 2  # Default value

        overlap_structure = st.selectbox(
            "Overlap structure",
            ["modular", "hierarchical", "hub-spoke"],
            help="Structure of community overlaps"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        edge_density = st.slider("Overall edge density", min_value=0.01, max_value=0.5, value=0.1, 
                                step=0.01, help="Controls the overall density of edges in the graph")
        
    with col4:
        homophily = st.slider("Homophily", min_value=0.0, max_value=1.0, value=0.8, 
                             step=0.01, help="Controls the ratio between intra and inter-community probabilities. " +
                                           "0 = equal probabilities, 1 = maximum homophily (all edges within communities)")
        
    randomness_factor = st.slider(
            "Edge probability randomness", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.05, 
            help="Amount of random variation in edge probabilities (0=deterministic, 1=highly random)"
        )
    
    overlap_density = st.slider("Community overlap density", min_value=0.0, max_value=0.5, value=0.2, 
                              step=0.01, help="Density of community overlaps")
    
    # Add seed parameter
    seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")

    # Generate universe button
    if st.button("Generate Universe"):
        with st.spinner("Generating universe..."):
            # Create universe
            universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                block_structure=block_structure,
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
                mixed_membership=mixed_membership,
                regimes_per_community=regimes_per_community,
                seed=seed
            )
            
            # Generate community co-membership matrix
            co_membership = universe.generate_community_co_membership_matrix(
                overlap_density=overlap_density,
                structure=overlap_structure
            )
            universe.community_co_membership = co_membership
            
            # Store in session state
            st.session_state.universe = universe
            
            st.success(f"Universe with {K} communities successfully generated!")

    # Show universe properties if it exists
    if st.session_state.universe is not None:
        universe = st.session_state.universe
        
        st.markdown('<div class="subsection-header">Universe Properties</div>', unsafe_allow_html=True)
        
        # Display universe statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Communities", universe.K)
        
        with col2:
            st.metric("Feature Dimension", universe.feature_dim)
        
        with col3:
            avg_density = universe.P.mean()
            st.metric("Average Edge Density", f"{avg_density:.4f}")
        
        # Show visualizations
        st.markdown('<div class="subsection-header">Universe Visualizations</div>', unsafe_allow_html=True)
        
        viz_tab1, viz_tab2 = st.tabs(["Edge Probabilities", "Community Co-membership"])
        
        with viz_tab1:
            # Sample subset of communities for visualization
            if universe.K > 10:
                sample_size = 10
                sample_communities = sorted(np.random.choice(universe.K, size=sample_size, replace=False).tolist())
                st.info(f"Showing a sample of {sample_size} communities for visualization clarity.")
            else:
                sample_communities = list(range(universe.K))
            
            fig = plot_community_matrix(universe.P, sample_communities)
            st.pyplot(fig)
        
        with viz_tab2:
            if universe.K > 10:
                # Use same sample as above for consistency
                fig = plot_community_matrix(universe.community_co_membership, sample_communities, 
                                           title="Community Co-membership Matrix")
            else:
                sample_communities = list(range(universe.K))
                fig = plot_community_matrix(universe.community_co_membership, sample_communities, 
                                           title="Community Co-membership Matrix")
            st.pyplot(fig)
            
            # Plot community graph
            st.markdown('<div class="subsection-header">Community Interaction Graph</div>', unsafe_allow_html=True)
            st.markdown("This graph shows how communities relate to each other. Node size represents self-connection strength, edge width represents inter-community connection strength.")
            
            fig = plot_community_graph(universe.P, sample_communities)
            st.pyplot(fig)

    # If universe has features, show feature structure visualization
    if st.session_state.universe is not None and st.session_state.universe.feature_dim > 0:
        universe = st.session_state.universe
        
        st.markdown('<div class="subsection-header">Feature Structure</div>', unsafe_allow_html=True)
        
        viz_tab3, viz_tab4 = st.tabs(["Feature Similarity Matrix", "Feature Subtypes"])
        
        with viz_tab3:
            if hasattr(universe, 'feature_similarity_matrix') and universe.feature_similarity_matrix is not None:
                # Import visualization function
                from utils.visualizations import visualize_feature_similarity_matrix
                
                # Sample subset of communities for visualization
                if universe.K > 20:
                    sample_size = 20
                    sample_communities = sorted(np.random.choice(universe.K, size=sample_size, replace=False).tolist())
                    st.info(f"Showing a sample of {sample_size} communities for visualization clarity.")
                else:
                    sample_communities = list(range(universe.K))
                
                # Create visualization
                fig = visualize_feature_similarity_matrix(
                    universe,
                    communities_to_plot=sample_communities
                )
                st.pyplot(fig)
                
                st.markdown("""
                This matrix shows the feature similarity between communities. 
                Higher values (brighter colors) indicate more similar features between communities.
                """)
        
        with viz_tab4:
            if hasattr(universe, 'feature_subtypes') and universe.feature_subtypes is not None and universe.feature_subtypes_per_community > 1:
                # Import visualization function
                from utils.visualizations import visualize_feature_subtypes
                
                # Sample subset of communities for visualization
                if universe.K > 10:
                    sample_size = 6
                    sample_communities = sorted(np.random.choice(universe.K, size=sample_size, replace=False).tolist())
                    st.info(f"Showing a sample of {sample_size} communities for visualization clarity.")
                else:
                    sample_communities = list(range(universe.K))
                
                # Create visualization
                fig = visualize_feature_subtypes(
                    universe,
                    communities_to_plot=sample_communities
                )
                st.pyplot(fig)
                
                st.markdown("""
                This visualization shows feature subtypes for each community. 
                Subtypes represent clusters within the community's feature space.
                The star represents the base prototype, and points represent subtypes.
                """)
            else:
                st.info("No feature subtypes defined (feature_subtypes_per_community = 1).")

## Graph Sampling Page
elif page == "Graph Sampling":
    st.markdown('<div class="section-header">Graph Sampling</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    else:
        universe = st.session_state.universe
        
        st.markdown("""
        <div class="info-box">
        Each graph is sampled from a <b>subset</b> of the universe communities. This simulates the real-world
        scenario where different graphs capture different aspects of an underlying generative process.
        </div>
        """, unsafe_allow_html=True)
        
        # Graph sampling parameters
        st.markdown('<div class="subsection-header">Sampling Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_communities = st.slider("Number of communities", min_value=2, 
                                     max_value=min(20, universe.K), value=min(5, universe.K))
            n_nodes = st.slider("Number of nodes", min_value=50, max_value=1000, value=50)
            
            # Add new parameter for community connectivity
            min_connection_strength = st.slider(
                "Minimum community connection strength", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.05,
                step=0.01,
                help="Minimum edge probability between communities to consider them connected"
            )
        
        with col2:
            sampling_method = st.selectbox(
                "Community sampling method",
                ["random", "similar", "diverse", "correlated"],
                help="Method for selecting community subsets"
            )
            
            min_component_size = st.slider(
                "Minimum component size",
                min_value=0,
                max_value=50,
                value=10,
                help="Components smaller than this will be removed from the final graph"
            )
            
            # Add checkbox for ensuring connected communities
            connected_communities = st.checkbox(
                "Ensure connected communities", 
                value=True,
                help="Ensure selected communities have meaningful connections to each other"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            degree_heterogeneity = st.slider(
                "Degree heterogeneity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Controls variability in node degrees"
            )
            
        with col4:
            edge_noise = st.slider(
                "Edge noise",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                help="Random noise added to edge probabilities"
            )
            
            indirect_influence = st.slider(
                "Co-membership influence",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="How strongly co-memberships influence edge formation (0=no effect, 0.5=strong effect)"
            )
        
        avg_memberships = st.slider("Average communities per node", min_value=1.0, max_value=5.0, value=1.5,
                                      step=0.1, help="Controls community overlap")
        
        feature_regime_balance = st.slider(
            "Feature Regime Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="How evenly regimes are distributed within communities (0=one regime dominates, 1=equal distribution)"
        )
        
        # Add seed parameter
        seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")

        # Generate graph button
        if st.button("Sample Graph"):
            with st.spinner("Sampling graph..."):
                # Sample community subset
                if connected_communities:     
                    st.write("Using enhanced connected community sampling method...")
                    # Use the enhanced sampling method
                    communities = universe.sample_connected_community_subset(
                        size=n_communities,
                        method=sampling_method,
                        min_connection_strength=min_connection_strength
                    )
                    st.write(f"Sampled {len(communities)} communities using enhanced method")
                else:
                    st.write("Using standard community sampling method...")
                    # Use standard sampling
                    communities = universe.sample_community_subset(
                        size=n_communities,
                        method=sampling_method
                    )
                    st.write(f"Sampled {len(communities)} communities using standard method")
                
                # Generate graph sample with the new parameters
                graph = GraphSample(
                    universe=universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    min_component_size=min_component_size,
                    degree_heterogeneity=degree_heterogeneity,
                    edge_noise=edge_noise,
                    indirect_influence=indirect_influence,
                    feature_regime_balance=feature_regime_balance,
                    seed=seed
                )
                
                # Store in session state
                st.session_state.current_graph = graph
                
                # Display component filtering results if any components were removed
                if hasattr(graph, 'deleted_components') and graph.deleted_components:
                    st.warning(f"Removed {len(graph.deleted_components)} small components " +
                             f"({sum(len(c) for c in graph.deleted_components)} nodes total)")
                    
                    # Show distribution of deleted nodes by community
                    if graph.deleted_node_types:
                        st.write("Distribution of removed nodes by community:")
                        for comm, count in graph.deleted_node_types.items():
                            st.write(f"- Community {comm}: {count} nodes")
                
                # Get number of connected components for status message
                n_components = nx.number_connected_components(graph.graph)
                
                if n_components > 1:
                    st.success(f"Graph with {n_nodes} nodes and {graph.graph.number_of_edges()} edges successfully sampled! "
                              f"({n_components} connected components)")
                else:
                    st.success(f"Graph with {n_nodes} nodes and {graph.graph.number_of_edges()} edges successfully sampled!")
        
        # Show graph properties if it exists
        if st.session_state.current_graph is not None:
            graph = st.session_state.current_graph
            
            st.markdown('<div class="subsection-header">Graph Properties</div>', unsafe_allow_html=True)
            
            # Display graph statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes", graph.n_nodes)
            
            with col2:
                st.metric("Edges", graph.graph.number_of_edges())
            
            with col3:
                avg_degree = 2 * graph.graph.number_of_edges() / graph.n_nodes
                st.metric("Average Degree", f"{avg_degree:.2f}")
            
            with col4:
                n_components = nx.number_connected_components(graph.graph)
                st.metric("Connected Components", n_components)
            
            # Show visualizations
            st.markdown('<div class="subsection-header">Graph Visualizations</div>', unsafe_allow_html=True)
            
            viz_tabs = st.tabs([
                "Graph Structure", 
                "Community Memberships", 
                "Degree Distribution", 
                "Community Overlap",
                "Component Analysis",  # New tab for component analysis
                "Community Connectivity",  # New tab for community connectivity
                "Feature Visualization"  # This tab will only show if features are enabled
            ])
            
            with viz_tabs[0]:
                # Community visualization tab
                st.markdown("Nodes are colored by their primary community membership.")
                fig = plot_graph_communities(graph.graph)
                st.pyplot(fig)
            
            with viz_tabs[1]:
                # Membership matrix visualization
                st.markdown("This heatmap shows the strength of each node's membership in each community.")
                fig = plot_membership_matrix(
                    graph.membership_vectors, 
                    graph.communities
                )
                st.pyplot(fig)
            
            with viz_tabs[2]:
                # Degree distribution visualization
                fig = plot_degree_distribution(graph.graph)
                st.pyplot(fig)
            
            with viz_tabs[3]:
                # Community overlap visualization
                fig = plot_community_overlap_distribution(graph.membership_vectors)
                st.pyplot(fig)
                
            with viz_tabs[4]:
                # Component analysis visualization
                if n_components > 1:
                    st.markdown(f"This graph has {n_components} connected components.")
                    
                    # Get connected components and sort by size
                    components = [c for c in nx.connected_components(graph.graph)]
                    components.sort(key=len, reverse=True)
                    
                    # Display component sizes
                    component_sizes = [len(c) for c in components]
                    
                    # Create bar chart of component sizes
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(component_sizes)), component_sizes)
                    ax.set_xlabel("Component Index")
                    ax.set_ylabel("Component Size")
                    ax.set_title("Connected Component Sizes")
                    st.pyplot(fig)
                    
                    # Option to visualize specific components
                    selected_component = st.selectbox(
                        "Select component to visualize",
                        options=range(min(10, n_components)),
                        format_func=lambda x: f"Component {x+1} (size: {component_sizes[x]})"
                    )
                    
                    # Visualize selected component
                    component_nodes = list(components[selected_component])
                    component_subgraph = graph.graph.subgraph(component_nodes)
                    
                    # Use the same color scheme as the main graph
                    fig = plot_graph_communities(component_subgraph, title=f"Component {selected_component+1}")
                    st.pyplot(fig)
                    
                    # Add component analysis metrics
                    component_stats = pd.DataFrame({
                        "Component": range(1, min(10, n_components) + 1),
                        "Size": component_sizes[:min(10, n_components)],
                        "Percentage": [size/graph.n_nodes * 100 for size in component_sizes[:min(10, n_components)]]
                    })
                    
                    st.table(component_stats)
                else:
                    st.markdown("This graph has a single connected component.")
                    # Create empty component stats for consistency
                    component_stats = pd.DataFrame({
                        "Component": [1],
                        "Size": [graph.n_nodes],
                        "Percentage": [100.0]
                    })
                    st.table(component_stats)
                
            with viz_tabs[5]:
                # Community connectivity analysis
                st.markdown("This analysis shows how the selected communities are connected to each other.")
                
                # Show community connectivity parameters
                conn_threshold = st.slider(
                    "Connection threshold", 
                    min_value=0.01, 
                    max_value=0.5, 
                    value=0.05,
                    step=0.01,
                    help="Minimum probability for considering communities connected"
                )
                
                # Analyze community connectivity
                community_connectivity = analyze_community_connectivity(
                    universe.P, 
                    graph.communities, 
                    threshold=conn_threshold
                )
                
                # Display key connectivity metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Connected Components", community_connectivity["n_components"])
                
                with metric_cols[1]:
                    st.metric("Isolated Communities", community_connectivity["isolated_communities"])
                
                with metric_cols[2]:
                    st.metric("Avg. Connections", f"{community_connectivity['avg_connections']:.2f}")
                
                with metric_cols[3]:
                    connected_label = "Yes" if community_connectivity["is_connected"] else "No"
                    st.metric("Fully Connected", connected_label)
                
                # Visualize community connectivity
                fig = visualize_community_connectivity(
                    universe.P,
                    graph.communities,
                    threshold=conn_threshold
                )
                st.pyplot(fig)
                
                st.markdown("""
                This analysis shows how communities in this graph are connected:
                - **Probability Matrix**: Edge probabilities between communities
                - **Community Graph**: Visual representation of community connections above threshold
                - **Connections per Community**: Number of connections for each community
                - **Connectivity Statistics**: Overall metrics for community connectivity
                
                Isolated communities (shown in red) may form disconnected components in the graph.
                """)

            # Feature visualization tab
            if st.session_state.universe.feature_dim > 0 and st.session_state.current_graph.features is not None:
                with viz_tabs[6]:  # Index should be 6 now since we added two tabs
                    # Import visualization function
                    from utils.visualizations import visualize_feature_correlations
                    
                    # Create feature dashboard
                    fig = visualize_feature_correlations(st.session_state.current_graph)
                    st.pyplot(fig)
                    
                    st.markdown("""
                    This dashboard shows the relationships between node features, community memberships, and graph topology.
                    - Top left: Correlation between features and community memberships
                    - Top right: Correlation between node degree and feature values
                    - Bottom left: PCA visualization of features colored by community
                    - Bottom right: Feature distribution by community
                    """)
            
            # Full dashboard
            st.markdown('<div class="subsection-header">Graph Dashboard</div>', unsafe_allow_html=True)
            
            fig = create_dashboard(
                graph.graph, 
                graph.membership_vectors, 
                graph.communities, 
                universe.P,
                figsize=(15, 12)
            )
            st.pyplot(fig)

# Add motif analysis to the page options
elif page == "Motif and Role Analysis":
    add_motif_role_analysis_page()

# Benchmark Generation Page
elif page == "Benchmark Generation":
    st.markdown('<div class="section-header">Benchmark Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    else:
        universe = st.session_state.universe
        
        st.markdown("""
        <div class="info-box">
        The benchmark consists of two sets of graphs:
        <ul>
            <li><b>Pretraining graphs:</b> Used for self-supervised learning</li>
            <li><b>Transfer graphs:</b> Test graphs with controlled distributional shift</li>
        </ul>
        This simulates the transfer learning scenario where models must generalize to new community combinations.
        </div>
        """, unsafe_allow_html=True)
        
        # Benchmark parameters
        st.markdown('<div class="subsection-header">Benchmark Parameters</div>', unsafe_allow_html=True)
        
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
        
        # Add new parameters for community connectivity and component filtering
        st.markdown('<div class="subsection-header">Graph Structure Parameters</div>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            min_connection_strength = st.slider(
                "Minimum community connection strength", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.05,
                step=0.01,
                help="Minimum edge probability between communities to consider them connected"
            )
            
            min_component_size = st.slider(
                "Minimum component size",
                min_value=0,
                max_value=100,
                value=20,
                help="Components smaller than this will be removed from all generated graphs"
            )
        
        with col6:
            indirect_influence = st.slider(
                "Co-membership influence",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="How strongly co-memberships influence edge formation (0=no effect, 0.5=strong effect)"
            )
            
            connected_communities = st.checkbox(
                "Ensure connected communities", 
                value=True,
                help="Ensure selected communities have meaningful connections to each other"
            )

        # Define transfer modes before using them
        transfer_modes = ["new_combinations", "rare_communities", "novel_communities"]
        transfer_difficulty = st.slider(
            "Transfer difficulty",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Controls how challenging the transfer learning task is (0=easy, 1=hard)"
        )

        # Add seed parameter
        seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")

        if st.button("Generate Benchmark"):
            with st.spinner("Generating benchmark..."):
                # Create a minimal benchmark instance
                benchmark = MMSBBenchmark.__new__(MMSBBenchmark)
                benchmark.universe = universe
                benchmark.graphs = []
                
                # Ensure the benchmark class has sample_connected_community_subset
                if connected_communities and not hasattr(benchmark.universe, 'sample_connected_community_subset'):
                    from utils.sampler import GraphSampler
                    sampler = GraphSampler(benchmark.universe)
                    sampler.add_connected_community_sampling_to_universe()
                
                # Generate pretraining graphs
                pretrain_graphs = benchmark.generate_pretraining_graphs(
                    n_graphs=n_pretrain,
                    min_communities=min_communities,
                    max_communities=max_communities,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    degree_heterogeneity=0.5,
                    edge_noise=0.0,
                    sampling_method="random",
                    min_connection_strength=min_connection_strength if connected_communities else 0.0,
                    min_component_size=min_component_size,
                    indirect_influence=indirect_influence,
                    seed=seed
                )
                
                # Generate transfer graphs for each mode
                transfer_graphs = []
                
                # Calculate how many graphs per mode
                n_per_mode = n_transfer // len(transfer_modes)
                remainder = n_transfer % len(transfer_modes)
                
                for i, mode in enumerate(transfer_modes):
                    # Add extras to first few modes
                    mode_count = n_per_mode + (1 if i < remainder else 0)
                    
                    mode_graphs = benchmark.generate_transfer_graphs(
                        n_graphs=mode_count,
                        reference_graphs=pretrain_graphs,
                        transfer_mode=mode,
                        transfer_difficulty=transfer_difficulty,
                        min_nodes=min_nodes,
                        max_nodes=max_nodes,
                        degree_heterogeneity=0.5,
                        min_connection_strength=min_connection_strength if connected_communities else 0.0,
                        min_component_size=min_component_size,
                        indirect_influence=indirect_influence,
                        seed=seed + i if seed is not None else None  # Increment seed for each mode
                    )
                    
                    transfer_graphs.extend(mode_graphs)
                
                # Store in session state
                st.session_state.benchmark = benchmark
                st.session_state.pretrain_graphs = pretrain_graphs
                st.session_state.transfer_graphs = transfer_graphs
                
                # Count total number of components across all graphs
                total_components_pretrain = sum(nx.number_connected_components(g.graph) for g in pretrain_graphs)
                total_components_transfer = sum(nx.number_connected_components(g.graph) for g in transfer_graphs)
                
                # Count graphs with multiple components
                multi_comp_pretrain = sum(1 for g in pretrain_graphs if nx.number_connected_components(g.graph) > 1)
                multi_comp_transfer = sum(1 for g in transfer_graphs if nx.number_connected_components(g.graph) > 1)
                
                # Track deleted nodes information
                total_deleted_pretrain = sum(sum(len(c) for c in g.deleted_components) for g in pretrain_graphs if hasattr(g, 'deleted_components'))
                total_deleted_transfer = sum(sum(len(c) for c in g.deleted_components) for g in transfer_graphs if hasattr(g, 'deleted_components'))
                
                # Create summary of deleted nodes by community
                deleted_by_community_pretrain = {}
                deleted_by_community_transfer = {}
                
                for g in pretrain_graphs:
                    if hasattr(g, 'deleted_node_types'):
                        for comm, count in g.deleted_node_types.items():
                            if comm not in deleted_by_community_pretrain:
                                deleted_by_community_pretrain[comm] = 0
                            deleted_by_community_pretrain[comm] += count
                
                for g in transfer_graphs:
                    if hasattr(g, 'deleted_node_types'):
                        for comm, count in g.deleted_node_types.items():
                            if comm not in deleted_by_community_transfer:
                                deleted_by_community_transfer[comm] = 0
                            deleted_by_community_transfer[comm] += count
                
                # Display results
                st.success(
                    f"Generated benchmark with {len(pretrain_graphs)} pretraining graphs and {len(transfer_graphs)} transfer graphs"
                )
                
                # Show component filtering results
                if total_deleted_pretrain > 0 or total_deleted_transfer > 0:
                    st.info("Component Filtering Results:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Pretraining Graphs:")
                        st.write(f"- Total nodes removed: {total_deleted_pretrain}")
                        st.write("- Removed nodes by community:")
                        for comm, count in sorted(deleted_by_community_pretrain.items()):
                            st.write(f"  • Community {comm}: {count} nodes")
                    
                    with col2:
                        st.write("Transfer Graphs:")
                        st.write(f"- Total nodes removed: {total_deleted_transfer}")
                        st.write("- Removed nodes by community:")
                        for comm, count in sorted(deleted_by_community_transfer.items()):
                            st.write(f"  • Community {comm}: {count} nodes")

        # Show benchmark properties if it exists
        if st.session_state.benchmark is not None and st.session_state.pretrain_graphs and st.session_state.transfer_graphs:
            benchmark = st.session_state.benchmark
            pretrain_graphs = st.session_state.pretrain_graphs
            transfer_graphs = st.session_state.transfer_graphs
            
            st.markdown('<div class="subsection-header">Benchmark Properties</div>', unsafe_allow_html=True)
            
            # Display benchmark statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pretraining Graphs", len(pretrain_graphs))
            
            with col2:
                st.metric("Transfer Graphs", len(transfer_graphs))
            
            with col3:
                total_nodes = sum(g.n_nodes for g in pretrain_graphs) + sum(g.n_nodes for g in transfer_graphs)
                st.metric("Total Nodes", total_nodes)
            
            # Show community distribution
            st.markdown('<div class="subsection-header">Community Distribution</div>', unsafe_allow_html=True)
            
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
            ax.set_xticklabels([f"C{c}" for c in comm_df["Community"]], rotation=45)
            ax.set_xlabel("Community")
            ax.set_ylabel("Frequency")
            ax.set_title("Community Distribution in Pretraining vs. Transfer Graphs")
            ax.legend()
            
            st.pyplot(fig)
            
            # Show graph examples
            st.markdown('<div class="subsection-header">Graph Examples</div>', unsafe_allow_html=True)
            
            example_tabs = st.tabs(["Pretraining Examples", "Transfer Examples"])
            
            with example_tabs[0]:
                if pretrain_graphs:
                    n_examples = min(3, len(pretrain_graphs))
                    example_indices = np.random.choice(len(pretrain_graphs), size=n_examples, replace=False)
                    
                    for i, idx in enumerate(example_indices):
                        st.markdown(f"**Pretraining Graph {idx+1}**")
                        st.markdown(f"Communities: {pretrain_graphs[idx].communities}")
                        st.markdown(f"Nodes: {pretrain_graphs[idx].n_nodes}, Edges: {pretrain_graphs[idx].graph.number_of_edges()}")
                        
                        fig = plot_graph_communities(pretrain_graphs[idx].graph)
                        st.pyplot(fig)
            
            with example_tabs[1]:
                if transfer_graphs:
                    n_examples = min(3, len(transfer_graphs))
                    example_indices = np.random.choice(len(transfer_graphs), size=n_examples, replace=False)
                    
                    for i, idx in enumerate(example_indices):
                        st.markdown(f"**Transfer Graph {idx+1}**")
                        st.markdown(f"Communities: {transfer_graphs[idx].communities}")
                        st.markdown(f"Nodes: {transfer_graphs[idx].n_nodes}, Edges: {transfer_graphs[idx].graph.number_of_edges()}")
                        
                        fig = plot_graph_communities(transfer_graphs[idx].graph)
                        st.pyplot(fig)
            
            # Compare feature distributions between pretraining and transfer
            if st.session_state.universe.feature_dim > 0:
                # Import visualization function
                from utils.visualizations import compare_feature_distributions
                
                # Sample graphs to keep visualization manageable
                max_graphs = 5
                pretrain_sample = st.session_state.pretrain_graphs[:max_graphs]
                transfer_sample = st.session_state.transfer_graphs[:max_graphs]
                
                # Create visualization
                fig = compare_feature_distributions(
                    [pretrain_sample, transfer_sample],
                    group_labels=["Pretraining", "Transfer"]
                )
                st.pyplot(fig)
                
                st.markdown("""
                This visualization compares the feature distributions between pretraining and transfer graphs.
                - Top left: PCA projection of features with confidence ellipses
                - Top right: t-SNE visualization for non-linear relationships
                - Bottom left: Feature distribution densities
                - Bottom right: Feature statistics table
                """)

            # Show transfer learning scenarios
            st.markdown('<div class="subsection-header">Transfer Learning Scenarios</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            The benchmark supports multiple transfer learning scenarios:
            <ul>
                <li><b>New combinations:</b> New combinations of communities seen during pretraining</li>
                <li><b>Rare communities:</b> Communities that were rare in pretraining graphs</li>
                <li><b>Novel communities:</b> Communities that were never or rarely seen during pretraining</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a widget for saving benchmark
            st.markdown('<div class="subsection-header">Save Benchmark</div>', unsafe_allow_html=True)
            
            save_dir = st.text_input("Save directory", "benchmark_data")
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

# Feature Analysis
elif page == "Feature Analysis":
    st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    elif st.session_state.universe.feature_dim == 0:
        st.warning("The current universe does not have features enabled. Please generate a universe with features > 0.")
    else:
        universe = st.session_state.universe
        
        st.markdown("""
        <div class="info-box">
        This page provides in-depth analysis of node features, their structure, and 
        their relationship with community membership and graph topology. Use these 
        visualizations to understand feature patterns and distribution shifts in 
        transfer learning scenarios.
        </div>
        """, unsafe_allow_html=True)
        
        # Select the graph or graphs to analyze
        st.markdown('<div class="subsection-header">Select Data to Analyze</div>', unsafe_allow_html=True)
        
        analysis_mode = st.radio(
            "Analysis mode",
            ["Single Graph", "Graph Comparison", "Benchmark Comparison"]
        )
        
        if analysis_mode == "Single Graph":
            # Select a single graph to analyze
            graph_sources = []
            if st.session_state.current_graph is not None:
                graph_sources.append("Current Graph")
            if st.session_state.pretrain_graphs:
                graph_sources.append("Pretraining Graphs")
            if st.session_state.transfer_graphs:
                graph_sources.append("Transfer Graphs")
            
            if not graph_sources:
                st.warning("No graphs available for analysis. Please generate a graph in the 'Graph Sampling' page or a benchmark in the 'Benchmark Generation' page.")
            else:
                graph_source = st.selectbox("Graph source", graph_sources)
                
                if graph_source == "Current Graph":
                    graph = st.session_state.current_graph
                elif graph_source == "Pretraining Graphs":
                    graph_idx = st.slider("Select graph index", min_value=0, max_value=len(st.session_state.pretrain_graphs)-1, value=0)
                    graph = st.session_state.pretrain_graphs[graph_idx]
                elif graph_source == "Transfer Graphs":
                    graph_idx = st.slider("Select graph index", min_value=0, max_value=len(st.session_state.transfer_graphs)-1, value=0)
                    graph = st.session_state.transfer_graphs[graph_idx]
                
                # Display various feature analyses
                st.markdown('<div class="subsection-header">Feature Analysis Dashboard</div>', unsafe_allow_html=True)
                
                # Import visualization function
                from utils.visualizations import visualize_feature_correlations
                
                # Create feature dashboard
                fig = visualize_feature_correlations(graph, n_features_to_plot=20)
                st.pyplot(fig)
                
                # Add explanatory text
                st.markdown("""
                This dashboard shows various aspects of node features in the selected graph:
                - **Top left:** Correlation between features and community memberships (red=positive, blue=negative)
                - **Top right:** Relationship between node degree and a representative feature
                - **Bottom left:** Feature projection using PCA, colored by primary community
                - **Bottom right:** Distribution of a feature across different communities
                """)
                
                # PCA vs t-SNE comparison
                st.markdown('<div class="subsection-header">Dimensionality Reduction Comparison</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    n_components = st.slider("Number of components", min_value=2, max_value=min(5, universe.feature_dim), value=2)
                
                with col2:
                    perplexity = st.slider("t-SNE perplexity", min_value=5, max_value=50, value=30)
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Get primary community for coloring
                primary_communities = np.argmax(graph.membership_vectors, axis=1)
                
                # PCA plot
                if universe.feature_dim > 2:
                    pca = PCA(n_components=n_components)
                    features_pca = pca.fit_transform(graph.features)
                    
                    # Plot first two components
                    scatter = axes[0].scatter(
                        features_pca[:, 0],
                        features_pca[:, 1],
                        c=primary_communities,
                        cmap='tab20',
                        alpha=0.7,
                        s=50
                    )
                    
                    axes[0].set_title("PCA Projection")
                    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
                else:
                    axes[0].text(0.5, 0.5, "Need more than 2 feature dimensions for PCA",
                                ha='center', va='center')
                    axes[0].set_title("PCA Projection")
                
                # t-SNE plot
                if universe.feature_dim > 2:
                    try:
                        from sklearn.manifold import TSNE
                        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        features_tsne = tsne.fit_transform(graph.features)
                        
                        scatter = axes[1].scatter(
                            features_tsne[:, 0],
                            features_tsne[:, 1],
                            c=primary_communities,
                            cmap='tab20',
                            alpha=0.7,
                            s=50
                        )
                        
                        axes[1].set_title("t-SNE Projection")
                        axes[1].set_xlabel("t-SNE 1")
                        axes[1].set_ylabel("t-SNE 2")
                    except Exception as e:
                        axes[1].text(0.5, 0.5, f"t-SNE failed: {str(e)}",
                                    ha='center', va='center')
                        axes[1].set_title("t-SNE Projection")
                else:
                    axes[1].text(0.5, 0.5, "Need more than 2 feature dimensions for t-SNE",
                                ha='center', va='center')
                    axes[1].set_title("t-SNE Projection")
                
                # Add colorbar for communities
                if universe.feature_dim > 2:
                    cbar = plt.colorbar(scatter, ax=axes)
                    cbar.set_label("Primary Community")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                This comparison shows two different dimensionality reduction techniques:
                - **PCA**: Linear projection that preserves global structure
                - **t-SNE**: Non-linear embedding that preserves local structure
                
                The effectiveness of clustering by community indicates how well features align with community structure.
                """)
                
                # Feature-topology correlation analysis
                st.markdown('<div class="subsection-header">Feature-Topology Correlation</div>', unsafe_allow_html=True)
                
                # Select features to analyze
                n_features = min(5, universe.feature_dim)
                feature_indices = st.multiselect(
                    "Select features to analyze",
                    options=list(range(universe.feature_dim)),
                    default=list(range(n_features))
                )
                
                if feature_indices:
                    # Create correlation plot
                    fig, axes = plt.subplots(1, len(feature_indices), figsize=(4*len(feature_indices), 4))
                    if len(feature_indices) == 1:
                        axes = [axes]  # Ensure axes is always a list
                    
                    # Get node degrees
                    degrees = np.array([d for _, d in graph.graph.degree()])
                    
                    # Plot each selected feature
                    for i, feat_idx in enumerate(feature_indices):
                        # Calculate correlation
                        correlation = np.corrcoef(degrees, graph.features[:, feat_idx])[0, 1]
                        
                        # Create scatter plot
                        axes[i].scatter(
                            degrees,
                            graph.features[:, feat_idx],
                            alpha=0.7,
                            s=30,
                            c=primary_communities,
                            cmap='tab20'
                        )
                        
                        axes[i].set_title(f"Feature {feat_idx} vs. Degree (r={correlation:.3f})")
                        axes[i].set_xlabel("Node Degree")
                        axes[i].set_ylabel(f"Feature {feat_idx} Value")
                        
                        # Add regression line
                        z = np.polyfit(degrees, graph.features[:, feat_idx], 1)
                        p = np.poly1d(z)
                        axes[i].plot(degrees, p(degrees), "r--", alpha=0.7)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    These plots show the relationship between node degree and feature values:
                    - Strong correlations indicate features are influenced by graph topology
                    - The regression line shows the trend direction
                    - Points are colored by primary community to show potential community-specific patterns
                    """)
        
        elif analysis_mode == "Graph Comparison":
            # Compare features across multiple graphs
            if not st.session_state.pretrain_graphs and not st.session_state.transfer_graphs:
                st.warning("No graph families available for comparison. Please generate a benchmark in the 'Benchmark Generation' page.")
            else:
                # Select graphs to compare
                compare_options = []
                
                if st.session_state.current_graph is not None:
                    compare_options.append("Current Graph")
                
                if st.session_state.pretrain_graphs:
                    compare_options.append("Pretraining Graphs (Sample)")
                    
                    # Add options for individual pretraining graphs
                    for i in range(min(5, len(st.session_state.pretrain_graphs))):
                        compare_options.append(f"Pretrain Graph {i}")
                
                if st.session_state.transfer_graphs:
                    compare_options.append("Transfer Graphs (Sample)")
                    
                    # Add options for individual transfer graphs
                    for i in range(min(5, len(st.session_state.transfer_graphs))):
                        compare_options.append(f"Transfer Graph {i}")
                
                selected_graphs = st.multiselect(
                    "Select graphs to compare",
                    options=compare_options,
                    default=["Pretraining Graphs (Sample)", "Transfer Graphs (Sample)"] if "Transfer Graphs (Sample)" in compare_options else [compare_options[0]]
                )
                
                if selected_graphs:
                    # Prepare graphs for comparison
                    graphs_to_compare = []
                    graph_labels = []
                    
                    for selection in selected_graphs:
                        if selection == "Current Graph":
                            graphs_to_compare.append(st.session_state.current_graph)
                            graph_labels.append("Current Graph")
                        elif selection == "Pretraining Graphs (Sample)":
                            # Sample up to 5 pretraining graphs
                            sample_size = min(5, len(st.session_state.pretrain_graphs))
                            sample_indices = range(sample_size)  # Use first few for consistency
                            graphs_to_compare.append([st.session_state.pretrain_graphs[i] for i in sample_indices])
                            graph_labels.append("Pretraining")
                        elif selection == "Transfer Graphs (Sample)":
                            # Sample up to 5 transfer graphs
                            sample_size = min(5, len(st.session_state.transfer_graphs))
                            sample_indices = range(sample_size)  # Use first few for consistency
                            graphs_to_compare.append([st.session_state.transfer_graphs[i] for i in sample_indices])
                            graph_labels.append("Transfer")
                        elif selection.startswith("Pretrain Graph "):
                            idx = int(selection.split()[-1])
                            graphs_to_compare.append(st.session_state.pretrain_graphs[idx])
                            graph_labels.append(f"Pretrain {idx}")
                        elif selection.startswith("Transfer Graph "):
                            idx = int(selection.split()[-1])
                            graphs_to_compare.append(st.session_state.transfer_graphs[idx])
                            graph_labels.append(f"Transfer {idx}")
                    
                    # Select analysis type
                    analysis_type = st.radio(
                        "Analysis type",
                        ["Full Distribution Comparison", "Specific Feature Analysis"]
                    )
                    
                    # Import visualization functions
                    from utils.visualizations import compare_feature_distributions
                    
                    if analysis_type == "Full Distribution Comparison":
                        # Compare overall feature distributions
                        fig = compare_feature_distributions(
                            graphs_to_compare,
                            group_labels=graph_labels
                        )
                        st.pyplot(fig)
                        
                        st.markdown("""
                        This visualization compares feature distributions across selected graphs or graph families:
                        - **Top left:** PCA projection showing feature clusters with confidence ellipses
                        - **Top right:** t-SNE visualization for non-linear patterns
                        - **Bottom left:** Feature distribution density curves
                        - **Bottom right:** Feature statistics table for each graph/family
                        """)
                    
                    else:  # Specific Feature Analysis
                        # Select specific feature to analyze
                        feature_idx = st.slider(
                            "Select feature index",
                            min_value=0,
                            max_value=universe.feature_dim - 1,
                            value=0
                        )
                        
                        # Compare specific feature
                        fig = compare_feature_distributions(
                            graphs_to_compare,
                            feature_idx=feature_idx,
                            group_labels=graph_labels
                        )
                        st.pyplot(fig)
                        
                        st.markdown(f"""
                        This visualization compares Feature {feature_idx} across selected graphs or graph families:
                        - **Top left:** Feature distribution in 2D space
                        - **Top right:** Alternative visualization (t-SNE if applicable)
                        - **Bottom left:** Density plot showing distribution shapes
                        - **Bottom right:** Feature statistics table
                        """)
        
        elif analysis_mode == "Benchmark Comparison":
            # Compare feature distributions in benchmark scenarios
            if not st.session_state.pretrain_graphs or not st.session_state.transfer_graphs:
                st.warning("No benchmark data available. Please generate a benchmark in the 'Benchmark Generation' page.")
            else:
                st.markdown('<div class="subsection-header">Benchmark Feature Distribution Analysis</div>', unsafe_allow_html=True)
                
                # Select feature analysis approach
                analysis_approach = st.radio(
                    "Analysis approach",
                    ["Overall Distribution", "Community-Specific", "Feature Shift Trajectory"]
                )
                
                if analysis_approach == "Overall Distribution":
                    # Import visualization function
                    from utils.visualizations import compare_feature_distributions
                    
                    # Sample graphs to keep visualization manageable
                    max_graphs = 5
                    pretrain_sample = st.session_state.pretrain_graphs[:max_graphs]
                    transfer_sample = st.session_state.transfer_graphs[:max_graphs]
                    
                    # Create visualization
                    fig = compare_feature_distributions(
                        [pretrain_sample, transfer_sample],
                        group_labels=["Pretraining", "Transfer"]
                    )
                    st.pyplot(fig)
                    
                    st.markdown("""
                    This visualization compares the overall feature distributions between pretraining and transfer graphs.
                    Significant separation between the distributions indicates a feature distribution shift that
                    may challenge transfer learning algorithms.
                    """)
                    
                elif analysis_approach == "Community-Specific":
                    # Analyze feature shifts per community
                    
                    # Get all communities present in both pretraining and transfer
                    pretrain_communities = set()
                    for graph in st.session_state.pretrain_graphs:
                        pretrain_communities.update(graph.communities)
                    
                    transfer_communities = set()
                    for graph in st.session_state.transfer_graphs:
                        transfer_communities.update(graph.communities)
                    
                    common_communities = pretrain_communities.intersection(transfer_communities)
                    
                    if not common_communities:
                        st.warning("No communities common to both pretraining and transfer graphs found.")
                    else:
                        # Select communities to analyze
                        selected_communities = st.multiselect(
                            "Select communities to analyze",
                            options=sorted(list(common_communities)),
                            default=sorted(list(common_communities))[:min(5, len(common_communities))]
                        )
                        
                        if selected_communities:
                            # Calculate feature shifts per community
                            community_shifts = {}
                            community_features_pretrain = {}
                            community_features_transfer = {}
                            
                            # Extract features for selected communities
                            for comm in selected_communities:
                                # Collect features from pretraining graphs
                                pretrain_comm_features = []
                                
                                for graph in st.session_state.pretrain_graphs:
                                    if comm in graph.communities:
                                        comm_idx = graph.communities.index(comm)
                                        # Get nodes with this as primary community
                                        primary_nodes = [i for i in range(graph.n_nodes) 
                                                        if np.argmax(graph.membership_vectors[i]) == comm_idx]
                                        
                                        if primary_nodes:
                                            # Sample up to 10 nodes per graph
                                            sample_nodes = np.random.choice(
                                                primary_nodes, 
                                                size=min(10, len(primary_nodes)), 
                                                replace=False
                                            )
                                            pretrain_comm_features.extend([graph.features[i] for i in sample_nodes])
                                
                                # Collect features from transfer graphs
                                transfer_comm_features = []
                                
                                for graph in st.session_state.transfer_graphs:
                                    if comm in graph.communities:
                                        comm_idx = graph.communities.index(comm)
                                        # Get nodes with this as primary community
                                        primary_nodes = [i for i in range(graph.n_nodes) 
                                                        if np.argmax(graph.membership_vectors[i]) == comm_idx]
                                        
                                        if primary_nodes:
                                            # Sample up to 10 nodes per graph
                                            sample_nodes = np.random.choice(
                                                primary_nodes, 
                                                size=min(10, len(primary_nodes)), 
                                                replace=False
                                            )
                                            transfer_comm_features.extend([graph.features[i] for i in sample_nodes])
                                
                                if pretrain_comm_features and transfer_comm_features:
                                    # Store features
                                    community_features_pretrain[comm] = np.array(pretrain_comm_features)
                                    community_features_transfer[comm] = np.array(transfer_comm_features)
                                    
                                    # Calculate shift (Euclidean distance between means)
                                    pretrain_mean = np.mean(pretrain_comm_features, axis=0)
                                    transfer_mean = np.mean(transfer_comm_features, axis=0)
                                    shift = np.linalg.norm(pretrain_mean - transfer_mean)
                                    
                                    community_shifts[comm] = shift
                            
                            # Visualize community-specific shifts
                            if community_shifts:
                                # Bar chart of feature shifts by community
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                communities_list = list(community_shifts.keys())
                                shifts_list = [community_shifts[c] for c in communities_list]
                                
                                ax.bar(
                                    range(len(communities_list)), 
                                    shifts_list, 
                                    color='purple', 
                                    alpha=0.7
                                )
                                ax.set_xticks(range(len(communities_list)))
                                ax.set_xticklabels([f"C{c}" for c in communities_list])
                                ax.set_xlabel('Community')
                                ax.set_ylabel('Feature Shift Magnitude')
                                ax.set_title('Feature Shifts by Community Between Pretraining and Transfer')
                                
                                st.pyplot(fig)
                                
                                # Select a community for detailed visualization
                                selected_comm = st.selectbox(
                                    "Select a community for detailed analysis",
                                    options=communities_list
                                )
                                
                                if selected_comm in community_features_pretrain and selected_comm in community_features_transfer:
                                    # Compare feature distributions for this community
                                    st.markdown(f"#### Feature Distribution for Community {selected_comm}")
                                    
                                    # Create visualization
                                    fig = compare_feature_distributions(
                                        [community_features_pretrain[selected_comm], community_features_transfer[selected_comm]],
                                        group_labels=["Pretraining", "Transfer"]
                                    )
                                    st.pyplot(fig)
                                    
                                    st.markdown(f"""
                                    This visualization shows how features for Community {selected_comm} differ between 
                                    pretraining and transfer graphs. This can help identify community-specific 
                                    distribution shifts that affect transfer learning performance.
                                    """)
                            else:
                                st.warning("Could not find enough data for community-specific analysis.")
                
                elif analysis_approach == "Feature Shift Trajectory":
                    # Visualize trajectory of feature shifts across transfer graphs
                    
                    # Only relevant for gradual shifts
                    st.info("This analysis is most useful for benchmarks with gradual feature distribution shifts.")
                    
                    # Apply t-SNE to visualize feature distributions
                    from sklearn.manifold import TSNE
                    
                    # Sample nodes from each graph family
                    pretrain_features = []
                    for graph in st.session_state.pretrain_graphs[:5]:  # Sample from first 5 graphs
                        indices = np.random.choice(graph.n_nodes, size=20, replace=False)  # 20 nodes per graph
                        pretrain_features.append(graph.features[indices])
                    
                    transfer_features = []
                    for i, graph in enumerate(st.session_state.transfer_graphs):
                        indices = np.random.choice(graph.n_nodes, size=20, replace=False)
                        # Add graph index to track progression
                        features = graph.features[indices]
                        transfer_features.append((i, features))
                    
                    # Combine for t-SNE
                    all_features = np.vstack([np.vstack(pretrain_features)] + [f for _, f in transfer_features])
                    
                    # Apply t-SNE
                    tsne = TSNE(n_components=2, random_state=42)
                    embedded = tsne.fit_transform(all_features)
                    
                    # Split back for plotting
                    n_pretrain = len(pretrain_features) * 20
                    pretrain_embedded = embedded[:n_pretrain]
                    transfer_embedded = embedded[n_pretrain:]
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot pretraining points
                    ax.scatter(
                        pretrain_embedded[:, 0], 
                        pretrain_embedded[:, 1],
                        c='blue',
                        alpha=0.5,
                        label='Pretraining'
                    )
                    
                    # Plot transfer points with color gradient to show progression
                    cmap = plt.cm.autumn
                    for i, (graph_idx, _) in enumerate(transfer_features):
                        start_idx = i * 20
                        end_idx = start_idx + 20
                        color = cmap(graph_idx / (len(st.session_state.transfer_graphs) - 1))
                        
                        ax.scatter(
                            transfer_embedded[start_idx:end_idx, 0],
                            transfer_embedded[start_idx:end_idx, 1],
                            c=[color],
                            alpha=0.7
                        )
                    
                    # Add colorbar for transfer graphs
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(st.session_state.transfer_graphs) - 1))
                    sm.set_array([])
                    cbar = plt.colorbar(sm)
                    cbar.set_label('Transfer Graph Index (Progression of Shift)')
                    
                    ax.set_xlabel('t-SNE Dimension 1')
                    ax.set_ylabel('t-SNE Dimension 2')
                    ax.set_title('Feature Distribution Shift Trajectory (t-SNE)')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    st.markdown("""
                    This visualization shows how the feature distribution evolves across transfer graphs:
                    - **Blue points**: Pretraining feature distribution
                    - **Color gradient**: Progression of feature shift across transfer graphs
                    
                    A clear trajectory from pretraining to transfer indicates a structured shift that
                    may be easier for transfer learning algorithms to adapt to, compared to random or
                    abrupt shifts.
                    """)

# Community Analysis Page
elif page == "Community Analysis":
    st.markdown('<div class="section-header">Community Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    elif st.session_state.current_graph is None and not st.session_state.pretrain_graphs:
        st.warning("Please generate a graph in the 'Graph Sampling' page or a benchmark in the 'Benchmark Generation' page.")
    else:
        universe = st.session_state.universe
        
        # Select graph to analyze
        st.markdown('<div class="subsection-header">Select Graph to Analyze</div>', unsafe_allow_html=True)
        
        graph_sources = []
        if st.session_state.current_graph is not None:
            graph_sources.append("Sampled Graph")
        if st.session_state.pretrain_graphs:
            graph_sources.append("Pretraining Graphs")
        if st.session_state.transfer_graphs:
            graph_sources.append("Transfer Graphs")
        
        graph_source = st.selectbox("Graph source", graph_sources)
        
        if graph_source == "Sampled Graph":
            graph = st.session_state.current_graph
            
        elif graph_source == "Pretraining Graphs":
            pretrain_graphs = st.session_state.pretrain_graphs
            graph_idx = st.slider("Select graph index", min_value=0, max_value=len(pretrain_graphs)-1, value=0)
            graph = pretrain_graphs[graph_idx]
            
        elif graph_source == "Transfer Graphs":
            transfer_graphs = st.session_state.transfer_graphs
            graph_idx = st.slider("Select graph index", min_value=0, max_value=len(transfer_graphs)-1, value=0)
            graph = transfer_graphs[graph_idx]
        
        # Display community analysis
        st.markdown('<div class="subsection-header">Community Structure</div>', unsafe_allow_html=True)
        
        # Community statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_communities = len(graph.communities)
            st.metric("Number of Communities", n_communities)
        
        with col2:
            avg_memberships = (graph.membership_vectors > 0.1).sum(axis=1).mean()
            st.metric("Avg Communities per Node", f"{avg_memberships:.2f}")
        
        with col3:
            # Calculate modularity
            # Convert to community lists for modularity calculation
            community_lists = []
            for c_idx, c in enumerate(graph.communities):
                # Get nodes with this community as primary
                primary_nodes = [i for i in range(graph.n_nodes) 
                              if np.argmax(graph.membership_vectors[i]) == c_idx]
                community_lists.append(primary_nodes)
            
            try:
                modularity = nx.algorithms.community.modularity(graph.graph, community_lists)
                st.metric("Modularity", f"{modularity:.4f}")
            except:
                st.metric("Modularity", "N/A")
        
        # Community membership visualization
        st.markdown('<div class="subsection-header">Community Membership Analysis</div>', unsafe_allow_html=True)
        
        analysis_tabs = st.tabs([
            "Membership Matrix", 
            "Community Overlap", 
            "Community Probability Matrix",
            "Node-Community Network"
        ])
        
        with analysis_tabs[0]:
            # Membership matrix visualization
            fig = plot_membership_matrix(graph.membership_vectors, graph.communities)
            st.pyplot(fig)
            
            # Add explanation
            st.markdown("""
            The heatmap shows the strength of each node's membership in each community.
            Brighter colors indicate stronger membership.
            """)
        
        with analysis_tabs[1]:
            # Community overlap visualization
            fig = plot_community_overlap_distribution(graph.membership_vectors)
            st.pyplot(fig)
            
            # Add explanation
            st.markdown("""
            This bar chart shows how many nodes belong to different numbers of communities.
            Nodes with memberships above 0.1 are counted as belonging to a community.
            """)
            
            # Add detailed overlap statistics
            membership_threshold = 0.1
            binary_memberships = graph.membership_vectors > membership_threshold
            
            overlap_counts = binary_memberships.sum(axis=1)
            unique_counts, count_frequencies = np.unique(overlap_counts, return_counts=True)
            
            # Create dataframe for display
            overlap_df = pd.DataFrame({
                "Number of Communities": unique_counts,
                "Count of Nodes": count_frequencies,
                "Percentage": count_frequencies / graph.n_nodes * 100
            })
            
            st.table(overlap_df)
        
        if universe.feature_dim > 0 and graph.features is not None:
            st.markdown('<div class="subsection-header">Feature-Community Relationship</div>', unsafe_allow_html=True)
            
            # Import visualization function
            from utils.visualizations import visualize_feature_correlations
            
            # Create feature-community correlation plot
            fig = visualize_feature_correlations(graph)
            st.pyplot(fig)
            
            st.markdown("""
            This dashboard shows the relationships between node features and community memberships:
            - **Top left**: Correlation between features and community memberships
            - **Top right**: Relationship between node degree and feature values
            - **Bottom left**: Feature structure visualization (PCA)
            - **Bottom right**: Feature distribution by community
            """)
            
            # Show community-specific feature distributions
            st.markdown('<div class="subsection-header">Community-Specific Feature Distributions</div>', unsafe_allow_html=True)
            
            # Select a community to analyze
            selected_community_idx = st.selectbox(
                "Select community to analyze",
                options=range(len(graph.communities)),
                format_func=lambda i: f"Community {graph.communities[i]}"
            )
            
            # Get nodes with this community as primary
            primary_nodes = [i for i in range(graph.n_nodes) 
                            if np.argmax(graph.membership_vectors[i]) == selected_community_idx]
            
            if primary_nodes:
                # Get features for these nodes
                community_features = graph.features[primary_nodes]
                
                # PCA for visualization
                if universe.feature_dim > 2:
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(community_features)
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Color by membership strength
                    membership_strength = graph.membership_vectors[primary_nodes, selected_community_idx]
                    
                    scatter = ax.scatter(
                        features_2d[:, 0],
                        features_2d[:, 1],
                        c=membership_strength,
                        cmap='viridis',
                        alpha=0.7,
                        s=50
                    )
                    
                    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%})")
                    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%})")
                    ax.set_title(f"Feature Distribution for Community {graph.communities[selected_community_idx]}")
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter)
                    cbar.set_label("Membership Strength")
                    
                    st.pyplot(fig)
                    
                    st.markdown(f"""
                    This visualization shows the feature distribution for nodes primarily belonging to 
                    Community {graph.communities[selected_community_idx]}. Points are colored by membership strength.
                    
                    Clusters within the distribution may indicate subtypes or subgroups within the community.
                    """)
                    
                    # Show feature statistics for this community
                    st.markdown("#### Feature Statistics")
                    
                    # Calculate statistics
                    feature_mean = np.mean(community_features, axis=0)
                    feature_std = np.std(community_features, axis=0)
                    feature_min = np.min(community_features, axis=0)
                    feature_max = np.max(community_features, axis=0)
                    
                    # Show statistics for first few features
                    n_show = min(10, universe.feature_dim)
                    
                    stats_df = pd.DataFrame({
                        "Feature": [f"Feature {i}" for i in range(n_show)],
                        "Mean": feature_mean[:n_show],
                        "Std Dev": feature_std[:n_show],
                        "Min": feature_min[:n_show],
                        "Max": feature_max[:n_show]
                    })
                    
                    st.table(stats_df)
                else:
                    st.info("Feature dimension too small for meaningful visualization.")
            else:
                st.info(f"No nodes with Community {graph.communities[selected_community_idx]} as primary community.")

        with analysis_tabs[2]:
            # Community probability matrix
            fig = plot_community_matrix(universe.P, graph.communities)
            st.pyplot(fig)
            
            # Add explanation
            st.markdown("""
            This matrix shows the edge probabilities between communities.
            Higher values (brighter colors) indicate higher probability of connections.
            """)
        
        with analysis_tabs[3]:
            # Node-community bipartite network visualization
            st.markdown("This visualization shows the relationship between nodes and communities as a bipartite network.")
            
            # Create bipartite graph
            B = nx.Graph()
            
            # Add node and community nodes
            for i in range(graph.n_nodes):
                B.add_node(f"n{i}", bipartite=0, type="node")
            
            for c_idx, c in enumerate(graph.communities):
                B.add_node(f"c{c}", bipartite=1, type="community")
            
            # Add edges based on significant memberships
            threshold = 0.2
            for i in range(graph.n_nodes):
                for c_idx, c in enumerate(graph.communities):
                    if graph.membership_vectors[i, c_idx] > threshold:
                        B.add_edge(f"n{i}", f"c{c}", weight=graph.membership_vectors[i, c_idx])
            
            # Draw bipartite layout
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Partition nodes
            nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
            communities = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
            
            # Position nodes
            pos = {}
            pos.update({node: (1, i) for i, node in enumerate(nodes)})
            pos.update({node: (2, i) for i, node in enumerate(communities)})
            
            # Draw nodes
            nx.draw_networkx_nodes(B, pos, nodelist=nodes, node_color="lightblue", node_size=50, ax=ax)
            nx.draw_networkx_nodes(B, pos, nodelist=communities, node_color="lightgreen", node_size=100, ax=ax)
            
            # Draw edges with weights as width
            weights = [B[u][v]["weight"] * 2 for u, v in B.edges()]
            nx.draw_networkx_edges(B, pos, width=weights, edge_color="gray", alpha=0.6, ax=ax)
            
            # Draw labels for communities only to avoid clutter
            comm_labels = {node: node for node in communities}
            nx.draw_networkx_labels(B, pos, labels=comm_labels, font_size=8, ax=ax)
            
            ax.set_axis_off()
            ax.set_title("Node-Community Bipartite Network")
            
            st.pyplot(fig)
            
            st.markdown("""
            In this visualization:
            - Blue nodes represent graph nodes
            - Green nodes represent communities
            - Edge thickness shows the strength of community membership
            - Only memberships above 0.2 are shown to reduce visual clutter
            """)
        
        # Add a new section for community connectivity analysis
        st.markdown('<div class="subsection-header">Community Connectivity Analysis</div>', unsafe_allow_html=True)
        
        # Show community connectivity parameters
        st.markdown("#### Community Connectivity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conn_threshold = st.slider(
                "Connection threshold", 
                min_value=0.01, 
                max_value=0.5, 
                value=0.05,
                step=0.01,
                help="Minimum probability for considering communities connected"
            )
        
        # Analyze community connectivity
        community_connectivity = analyze_community_connectivity(
            universe.P, 
            graph.communities, 
            threshold=conn_threshold
        )
        
        # Display key connectivity metrics
        with col2:
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Connected Components", community_connectivity["n_components"])
                st.metric("Isolated Communities", community_connectivity["isolated_communities"])
            
            with metric_cols[1]:
                st.metric("Avg. Connections", f"{community_connectivity['avg_connections']:.2f}")
                connected_label = "Yes" if community_connectivity["is_connected"] else "No"
                st.metric("Fully Connected", connected_label)
        
        # Visualize community connectivity
        fig = visualize_community_connectivity(
            universe.P,
            graph.communities,
            threshold=conn_threshold
        )
        st.pyplot(fig)
        
        st.markdown("""
        This analysis shows how communities in this graph are connected:
        - **Probability Matrix**: Edge probabilities between communities
        - **Community Graph**: Visual representation of community connections above threshold
        - **Connections per Community**: Number of connections for each community
        - **Connectivity Statistics**: Overall metrics for community connectivity
        
        Isolated communities (shown in red) may form disconnected components in the graph.
        """)
        
        # Add connectivity analysis to the benchmark generation page as well
        if len(graph.communities) > 1:
            st.markdown('<div class="subsection-header">Component Structure Analysis</div>', unsafe_allow_html=True)
            
            # Get connected components from the graph
            components = list(nx.connected_components(graph.graph))
            components.sort(key=len, reverse=True)
            
            # Create a plot showing community distribution across components
            if len(components) > 1:
                st.markdown("#### Community Distribution in Components")
                
                # Analyze which communities appear in each component
                component_communities = []
                for comp in components[:min(5, len(components))]:  # Analyze up to 5 components
                    comp_nodes = list(comp)
                    
                    # Get primary communities of nodes in this component
                    community_counts = {}
                    for node in comp_nodes:
                        primary_comm_idx = np.argmax(graph.membership_vectors[node])
                        primary_comm = graph.communities[primary_comm_idx]
                        
                        if primary_comm not in community_counts:
                            community_counts[primary_comm] = 0
                        community_counts[primary_comm] += 1
                    
                    component_communities.append(community_counts)
                
                # Create a plot showing community distribution
                fig, ax = plt.subplots(figsize=(12, 6))
                
                all_communities = sorted(list(set().union(*[set(cc.keys()) for cc in component_communities])))
                x = np.arange(len(all_communities))
                width = 0.8 / len(component_communities)
                
                for i, comp_comms in enumerate(component_communities):
                    counts = [comp_comms.get(comm, 0) for comm in all_communities]
                    offset = (i - len(component_communities) / 2 + 0.5) * width
                    ax.bar(x + offset, counts, width, label=f"Component {i+1}")
                
                ax.set_xticks(x)
                ax.set_xticklabels([f"C{c}" for c in all_communities])
                ax.set_xlabel("Community")
                ax.set_ylabel("Number of Nodes")
                ax.set_title("Community Distribution Across Components")
                ax.legend()
                
                st.pyplot(fig)
                
                st.markdown("""
                This plot shows how communities are distributed across different connected components.
                Components with diverse communities indicate that the component structure is not
                strictly determined by community membership.
                """)
                
                # Add component statistics
                component_stats = []
                for i, comp in enumerate(components[:min(5, len(components))]):
                    comp_nodes = list(comp)
                    comp_size = len(comp_nodes)
                    comp_edges = graph.graph.subgraph(comp_nodes).number_of_edges()
                    
                    stats = {
                        "Component": i + 1,
                        "Nodes": comp_size,
                        "Edges": comp_edges,
                        "Density": 2 * comp_edges / (len(comp_nodes) * (len(comp_nodes) - 1)) if len(comp_nodes) > 1 else 0,
                        "Communities": len(set(graph.communities[np.argmax(graph.membership_vectors[node])] for node in comp_nodes))
                    }
                    component_stats.append(stats)
                
                st.table(pd.DataFrame(component_stats))


        # Graph structure analysis
        st.markdown('<div class="subsection-header">Graph Structure Analysis</div>', unsafe_allow_html=True)
        
        structure_tabs = st.tabs([
            "Community Visualization", 
            "Degree Distribution", 
            "Graph Statistics"
        ])
        
        with structure_tabs[0]:
            # Community visualization
            fig = plot_graph_communities(graph.graph)
            st.pyplot(fig)
        
        with structure_tabs[1]:
            # Degree distribution
            fig = plot_degree_distribution(graph.graph)
            st.pyplot(fig)
            
            # Add detailed degree statistics
            degrees = [d for _, d in graph.graph.degree()]
            degree_df = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Value": [
                    np.mean(degrees),
                    np.median(degrees),
                    np.min(degrees),
                    np.max(degrees),
                    np.std(degrees)
                ]
            })
            
            st.table(degree_df)
            
            # Degree distribution as table
            unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
            
            degree_distr_df = pd.DataFrame({
                "Degree": unique_degrees,
                "Count": degree_counts,
                "Percentage": degree_counts / graph.n_nodes * 100
            })
            
            degree_distr_df = degree_distr_df.sort_values("Degree")
            
            st.dataframe(degree_distr_df)
        
        with structure_tabs[2]:
            # Graph statistics
            G = graph.graph
            
            stats_df = pd.DataFrame({
                "Metric": [
                    "Nodes",
                    "Edges",
                    "Density",
                    "Average Clustering Coefficient",
                    "Number of Connected Components",
                    "Largest Component Size",
                    "Average Path Length (sample)",
                    "Diameter (sample)"
                ],
                "Value": [
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    nx.density(G),
                    nx.average_clustering(G),
                    nx.number_connected_components(G),
                    len(max(nx.connected_components(G), key=len)),
                    "Computing...",
                    "Computing..."
                ]
            })
            
            # For larger graphs, estimate path length and diameter
            try:
                if G.number_of_nodes() <= 1000:
                    # For small graphs, compute exactly
                    largest_cc = max(nx.connected_components(G), key=len)
                    largest_cc_graph = G.subgraph(largest_cc)
                    
                    avg_path = nx.average_shortest_path_length(largest_cc_graph)
                    stats_df.loc[6, "Value"] = f"{avg_path:.4f}"
                    
                    diameter = nx.diameter(largest_cc_graph)
                    stats_df.loc[7, "Value"] = str(diameter)
                else:
                    # For large graphs, sample
                    largest_cc = max(nx.connected_components(G), key=len)
                    largest_cc_graph = G.subgraph(largest_cc)
                    
                    # Sample nodes for path length calculation
                    sample_size = min(100, len(largest_cc))
                    sample_nodes = np.random.choice(list(largest_cc), size=sample_size, replace=False)
                    
                    # Calculate average path length on sample
                    path_lengths = []
                    for i, u in enumerate(sample_nodes):
                        for v in sample_nodes[i+1:]:
                            try:
                                path_lengths.append(nx.shortest_path_length(largest_cc_graph, u, v))
                            except:
                                pass
                    
                    if path_lengths:
                        avg_path = np.mean(path_lengths)
                        stats_df.loc[6, "Value"] = f"{avg_path:.4f} (estimated)"
                        
                        diameter = max(path_lengths)
                        stats_df.loc[7, "Value"] = f"{diameter} (estimated)"
                    else:
                        stats_df.loc[6, "Value"] = "N/A"
                        stats_df.loc[7, "Value"] = "N/A"
            except:
                stats_df.loc[6, "Value"] = "Error in computation"
                stats_df.loc[7, "Value"] = "Error in computation"
            
            st.table(stats_df)
            
        # Community-based node features
        if universe.feature_dim > 0 and graph.features is not None:
            st.markdown('<div class="subsection-header">Node Features Analysis</div>', unsafe_allow_html=True)
            
            feature_tabs = st.tabs([
                "Feature Correlation with Communities",
                "Feature Statistics",
                "Feature Visualization"
            ])
            
            with feature_tabs[0]:
                # Calculate correlation between features and community memberships
                correlation_matrix = np.zeros((len(graph.communities), universe.feature_dim))
                
                for c_idx, c in enumerate(graph.communities):
                    for f_idx in range(universe.feature_dim):
                        correlation = np.corrcoef(graph.membership_vectors[:, c_idx], graph.features[:, f_idx])[0, 1]
                        correlation_matrix[c_idx, f_idx] = correlation
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(12, 6))
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
                
                st.markdown("""
                This heatmap shows the correlation between node features and community memberships.
                Positive correlation (red) indicates that the feature tends to increase with community membership.
                Negative correlation (blue) indicates that the feature tends to decrease with community membership.
                """)
            
            with feature_tabs[1]:
                # Feature statistics
                feature_mean = np.mean(graph.features, axis=0)
                feature_std = np.std(graph.features, axis=0)
                feature_min = np.min(graph.features, axis=0)
                feature_max = np.max(graph.features, axis=0)
                
                # Show statistics for first 10 features
                n_show = min(10, universe.feature_dim)
                
                feature_stats_df = pd.DataFrame({
                    "Feature": [f"Feature {i}" for i in range(n_show)],
                    "Mean": feature_mean[:n_show],
                    "Std Dev": feature_std[:n_show],
                    "Min": feature_min[:n_show],
                    "Max": feature_max[:n_show]
                })
                
                st.table(feature_stats_df)
                
                if universe.feature_dim > n_show:
                    st.info(f"Showing statistics for first {n_show} features out of {universe.feature_dim}.")
            
            with feature_tabs[2]:
                # Visualize features with dimensionality reduction
                from sklearn.decomposition import PCA
                
                # Apply PCA to reduce to 2D
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(graph.features)
                
                # Get primary community for each node for coloring
                primary_communities = np.argmax(graph.membership_vectors, axis=1)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
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
                
                st.markdown("""
                This plot shows a 2D projection of the node features using PCA.
                Nodes are colored by their primary community membership.
                Clustering of same-colored points indicates that features capture community structure.
                """)

            # Feature visualization tab
            if st.session_state.universe.feature_dim > 0 and st.session_state.current_graph.features is not None:
                with viz_tabs[5]:  # Index should be 5 since we added a new tab
                    # Import visualization function
                    from utils.visualizations import visualize_feature_correlations
                    
                    # Create feature dashboard
                    fig = visualize_feature_correlations(st.session_state.current_graph)
                    st.pyplot(fig)
                    
                    st.markdown("""
                    This dashboard shows the relationships between node features, community memberships, and graph topology.
                    - Top left: Correlation between features and community memberships
                    - Top right: Correlation between node degree and feature values
                    - Bottom left: PCA visualization of features colored by community
                    - Bottom right: Feature distribution by community
                    """)
            
            # Full dashboard
            st.markdown('<div class="subsection-header">Graph Dashboard</div>', unsafe_allow_html=True)
            
            fig = create_dashboard(
                graph.graph, 
                graph.membership_vectors, 
                graph.communities, 
                universe.P,
                figsize=(15, 12)
            )
            st.pyplot(fig)

# Add new Parameter Space Analysis page
elif page == "Parameter Space Analysis":
    st.markdown('<div class="section-header">Parameter Space Analysis</div>', unsafe_allow_html=True)
    
    # Check if we have graphs to analyze
    has_pretrain = len(st.session_state.pretrain_graphs) > 0
    has_transfer = len(st.session_state.transfer_graphs) > 0
    has_current = st.session_state.current_graph is not None
    
    if not (has_pretrain or has_transfer or has_current):
        st.warning("Please generate graphs or benchmarks first in the previous pages.")
    else:
        st.markdown("""
        <div class="info-box">
        Parameter space analysis helps visualize where your graph families fall in the space of key graph properties:
        <ul>
            <li><strong>Homophily level</strong>: Extent to which edges connect similar nodes (within same community)</li>
            <li><strong>Power law exponent</strong>: Characterizes the degree distribution's tail behavior</li>
            <li><strong>Clustering coefficient</strong>: Measures how nodes tend to cluster together</li>
            <li><strong>Triangle count/density</strong>: Number and density of triangles in the graph</li>
            <li><strong>Node/edge statistics</strong>: Basic graph properties like average degree</li>
            <li><strong>Community overlap</strong>: How many communities nodes belong to on average</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Options for analysis
        st.markdown('<div class="subsection-header">Analysis Options</div>', unsafe_allow_html=True)
        
        # Select which graph families to analyze
        st.markdown("### Select Graph Families to Analyze")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analyze_current = st.checkbox("Current Graph", value=has_current)
        
        with col2:
            analyze_pretrain = st.checkbox("Pretraining Graphs", value=has_pretrain)
        
        with col3:
            analyze_transfer = st.checkbox("Transfer Graphs", value=has_transfer)
        
        # Parameters to analyze
        st.markdown("### Select Parameters")
        
        param_options = [
            "homophily",
            "power_law_exponent",
            "clustering_coefficient",
            "triangle_density",
            "node_count",
            "avg_degree",
            "avg_communities_per_node",
            "density",
            "connected_components",
            "largest_component_size"
        ]
        
        selected_params = st.multiselect(
            "Parameters to analyze",
            param_options,
            default=["homophily", "clustering_coefficient", "avg_degree", "avg_communities_per_node"]
        )
        
        # Initialize session state for analysis results if not exists
        if 'parameter_analysis_results' not in st.session_state:
            st.session_state.parameter_analysis_results = None
        
        # Analyze button
        if st.button("Run Analysis"):
            with st.spinner("Analyzing graph parameters..."):
                # Container for family DataFrames
                family_dfs = {}
                
                # Analyze current graph if selected
                if analyze_current and st.session_state.current_graph is not None:
                    current_graph = st.session_state.current_graph
                    current_df = analyze_graph_family([current_graph])
                    family_dfs["Current Graph"] = current_df
                
                # Analyze pretraining graphs if selected
                if analyze_pretrain and st.session_state.pretrain_graphs:
                    pretrain_df = analyze_graph_family(st.session_state.pretrain_graphs)
                    family_dfs["Pretraining Graphs"] = pretrain_df
                
                # Analyze transfer graphs if selected
                if analyze_transfer and st.session_state.transfer_graphs:
                    transfer_df = analyze_graph_family(st.session_state.transfer_graphs)
                    family_dfs["Transfer Graphs"] = transfer_df
                
                # Store results in session state
                st.session_state.parameter_analysis_results = {
                    'family_dfs': family_dfs,
                    'selected_params': selected_params
                }
        
        # Display results if available
        if st.session_state.parameter_analysis_results is not None:
            family_dfs = st.session_state.parameter_analysis_results['family_dfs']
            selected_params = st.session_state.parameter_analysis_results['selected_params']
            
            # Compute statistics for each family
            family_stats = {}
            for name, df in family_dfs.items():
                family_stats[name] = compute_statistics(df)
            
            # Display table of statistics
            st.markdown("### Parameter Statistics")
            
            # Create comparison DataFrame
            comparison_df = compare_graph_families(family_dfs, selected_params)
            
            # Display as table
            pivoted = comparison_df.pivot(index='parameter', columns='family', values=['mean', 'std'])
            st.dataframe(pivoted)
            
            # Create parameter visualizations
            st.markdown("### Parameter Distributions")
            
            # For each selected parameter, create distribution plot
            for param in selected_params:
                param_available = any(param in df.columns and not df[param].isna().all() for df in family_dfs.values())
                
                if param_available:
                    st.markdown(f"#### {param.replace('_', ' ').title()}")
                    
                    # Create distribution comparison
                    fig = compare_parameter_distributions(family_dfs, param)
                    st.pyplot(fig)
            
            # Parameter space visualizations (scatter plots)
            if len(selected_params) >= 2:
                st.markdown("### Parameter Space Visualization")
                
                # Let user select which parameters to plot
                col1, col2 = st.columns(2)
                
                with col1:
                    x_param = st.selectbox(
                        "X-axis parameter",
                        options=selected_params,
                        index=0
                    )
                
                with col2:
                    y_param = st.selectbox(
                        "Y-axis parameter",
                        options=selected_params,
                        index=min(1, len(selected_params)-1)
                    )
                
                # Create parameter space plot
                if x_param != y_param:
                    fig = plot_parameter_space(family_dfs, x_param, y_param)
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    This plot shows where different graph families lie in parameter space. 
                    Each point represents a graph, and the ellipses show the 95% confidence regions for each family.
                    
                    - **Separated ellipses** indicate distinct graph families with different characteristics
                    - **Overlapping ellipses** indicate similar graph families
                    - **Large ellipses** indicate high variability within a family
                    - **Small ellipses** indicate consistent graph properties within a family
                    """)
            
            # Comprehensive dashboard
            st.markdown("### Parameter Dashboards")
            
            # For each family, create a dashboard
            for name, df in family_dfs.items():
                st.markdown(f"#### {name} Dashboard")
                
                fig = create_parameter_dashboard(df, name)
                st.pyplot(fig)

# Add motif analysis to the page options
elif page == "Motif and Role Analysis":
    add_motif_role_analysis_page()

# Add new page section:
elif page == "Neighborhood Analysis":
    st.markdown('<div class="section-header">Neighborhood Feature Analysis & Label Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    elif st.session_state.current_graph is None and not st.session_state.pretrain_graphs:
        st.warning("Please generate a graph in the 'Graph Sampling' page or a benchmark in the 'Benchmark Generation' page.")
    else:
        st.markdown("""
        <div class="info-box">
        This page analyzes how feature regimes distribute in node neighborhoods and generates balanced labels based on these distributions.
        The analysis helps understand:
        - How feature regimes cluster in different parts of the graph
        - What patterns emerge in k-hop neighborhoods
        - How to create meaningful node classification tasks
        </div>
        """, unsafe_allow_html=True)
        
        # Select graph to analyze
        graph_sources = []
        if st.session_state.current_graph is not None:
            graph_sources.append("Current Graph")
        if st.session_state.pretrain_graphs:
            graph_sources.append("Pretraining Graphs")
        if st.session_state.transfer_graphs:
            graph_sources.append("Transfer Graphs")
            
        graph_source = st.selectbox("Graph source", graph_sources)
        
        if graph_source == "Current Graph":
            graph = st.session_state.current_graph
        elif graph_source == "Pretraining Graphs":
            graph_idx = st.slider("Select graph index", min_value=0, max_value=len(st.session_state.pretrain_graphs)-1, value=0)
            graph = st.session_state.pretrain_graphs[graph_idx]
        elif graph_source == "Transfer Graphs":
            graph_idx = st.slider("Select graph index", min_value=0, max_value=len(st.session_state.transfer_graphs)-1, value=0)
            graph = st.session_state.transfer_graphs[graph_idx]
            
        # Neighborhood Analysis Parameters
        st.markdown('<div class="subsection-header">Neighborhood Analysis Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_labels = st.slider("Number of labels", min_value=2, max_value=10, value=4,
                               help="Number of distinct classes to generate")
        
        # Add labeling method selection
        labeling_method = st.radio(
            "Labeling Method",
            ["Clustering-based", "Rule-based"],
            help="""
            Clustering-based: Groups nodes based on similarity of neighborhood features
            Rule-based: Generates explicit, transferable rules based on feature regime frequencies
            """
        )
        
        if labeling_method == "Clustering-based":
            balance_tolerance = st.slider("Balance tolerance", min_value=0.0, max_value=0.5, value=0.1,
                                        help="How much imbalance to allow between classes (0=perfectly balanced)")
        else:
            # Rule-based parameters
            st.markdown("##### Rule Generation Parameters")
            
            # Let user define max hop range with reasonable limits
            max_allowed_hops = 5  # Maximum allowed for computational reasons
            
            # Get graph diameter if available
            if hasattr(graph, 'graph'):
                try:
                    # Sample a few nodes for efficiency in large graphs
                    sample_size = min(100, graph.graph.number_of_nodes())
                    sample_nodes = np.random.choice(list(graph.graph.nodes()), size=sample_size, replace=False)
                    max_path = 0
                    for u in sample_nodes:
                        for v in sample_nodes:
                            try:
                                path_len = nx.shortest_path_length(graph.graph, u, v)
                                max_path = max(max_path, path_len)
                            except nx.NetworkXNoPath:
                                continue
                    suggested_max = min(max_path, max_allowed_hops)
                except:
                    suggested_max = 3  # Default if calculation fails
            else:
                suggested_max = 3
            
            # Let user choose max_hops with guidance
            max_hops = st.slider(
                "Maximum hop distance",
                min_value=1,
                max_value=max_allowed_hops,
                value=min(suggested_max, 3),
                help=f"Maximum number of hops to analyze. Suggested maximum based on graph structure: {suggested_max}"
            )
            
            # Hop range selection
            col1, col2 = st.columns(2)
            with col1:
                min_hop = st.slider("Minimum hop distance", min_value=1, max_value=max_hops, value=1,
                                  help="Minimum hop distance to consider for rules")
            with col2:
                max_hop = st.slider("Maximum hop distance", min_value=min_hop, max_value=max_hops, value=max_hops,
                                  help="Maximum hop distance to consider for rules")
        
        # Other rule parameters
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum rule support", min_value=0.05, max_value=0.3, value=0.1,
                                      help="Minimum fraction of nodes a rule should apply to")
        with col2:
            max_rules_per_label = st.slider("Max rules per label", min_value=1, max_value=5, value=3,
                                              help="Maximum number of rules to generate per label")
        
        st.info(f"Note: Label {n_labels-1} will be reserved as the 'rest' class for nodes that don't match any rules.")

        # Run Analysis button
        if st.button("Run Neighborhood Analysis"):
            with st.spinner("Analyzing neighborhoods and generating labels..."):
                # Initialize neighborhood analyzer if not already done or if max_hops changed
                if (graph.neighborhood_analyzer is None or 
                    graph.neighborhood_analyzer.max_hops < max_hops):
                    graph.neighborhood_analyzer = NeighborhoodFeatureAnalyzer(
                        graph=graph.graph,
                        node_regimes=graph.node_regimes,
                        total_regimes=len(graph.communities) * graph.universe.regimes_per_community,
                        max_hops=max_hops
                    )
                
                # Get frequency vectors for visualization
                freq_vectors = {}
                for k in range(1, max_hops + 1):
                    freq_vectors[k] = graph.neighborhood_analyzer.get_all_frequency_vectors(k)
                
                # Generate labels based on selected method
                if labeling_method == "Clustering-based":
                    label_generator = FeatureRegimeLabelGenerator(
                        frequency_vectors=freq_vectors[1],  # Use 1-hop for initial labels
                        n_labels=n_labels,
                        balance_tolerance=balance_tolerance,
                        seed=42
                    )
                    labels = label_generator.get_node_labels()
                    rules = label_generator._extract_label_rules()
                    applied_rules = None
                else:
                    rule_generator = GenerativeRuleBasedLabeler(
                        n_labels=n_labels,
                        min_support=min_support,
                        max_rules_per_label=max_rules_per_label,
                        min_hop=min_hop,
                        max_hop=max_hop,
                        seed=42
                    )
                    rules = rule_generator.generate_rules(freq_vectors)
                    labels, applied_rules = rule_generator.apply_rules(freq_vectors)
                
                # Visualizations
                st.markdown('<div class="subsection-header">Analysis Results</div>', unsafe_allow_html=True)
                
                # 1. Neighborhood Feature Distribution
                st.markdown("#### Feature Regime Distribution by Hop Distance")
                
                # Only create tabs for the selected hop range
                hop_range = list(range(min_hop, max_hop + 1))
                tabs = st.tabs([f"{k}-hop" for k in hop_range])
                
                for k_idx, k in enumerate(hop_range):
                    with tabs[k_idx]:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot average frequency for each regime
                        avg_freq = np.mean(freq_vectors[k], axis=0)
                        ax.bar(range(len(avg_freq)), avg_freq)
                        
                        ax.set_xlabel("Feature Regime")
                        ax.set_ylabel("Average Frequency")
                        ax.set_title(f"Average Feature Regime Distribution in {k}-hop Neighborhoods")
                        
                        st.pyplot(fig)
                
                # 2. Label Distribution and Graph Visualization
                st.markdown("#### Generated Label Distribution")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Label counts
                unique_labels, counts = np.unique(labels, return_counts=True)
                ax1.bar(unique_labels, counts)
                ax1.set_xlabel("Label")
                ax1.set_ylabel("Count")
                ax1.set_title("Label Distribution")
                
                # Graph visualization with labels
                pos = nx.spring_layout(graph.graph)
                nx.draw(graph.graph, pos, node_color=labels, cmap='tab20', 
                       node_size=50, ax=ax2)
                ax2.set_title("Graph Colored by Generated Labels")
                
                st.pyplot(fig)
                
                # 3. Rules
                st.markdown("#### Generated Rules")
                if labeling_method == "Clustering-based":
                    for rule in rules:
                        st.markdown(f"- {rule}")
                else:
                    # Unpack all 4 values from the rules
                    for rule_fn, label, rule_str, hop in rules:
                        st.markdown(f"- Label {label}: {rule_str}")
                    
                    # Show rule coverage statistics
                    rule_coverage = Counter(r for r in applied_rules if r is not None)
                    st.markdown("#### Rule Coverage Statistics")
                    for rule, count in rule_coverage.items():
                        st.markdown(f"- {rule}: Applied to {count} nodes ({count/len(labels):.1%})")
                
                # 4. Feature Analysis
                st.markdown("#### Feature Analysis by Label")
                
                # Create PCA visualization of frequency vectors colored by label
                pca = PCA(n_components=2)
                freq_2d = pca.fit_transform(freq_vectors[1])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(freq_2d[:, 0], freq_2d[:, 1], c=labels, 
                                   cmap='tab20', alpha=0.6)
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                ax.set_title("Neighborhood Feature Vectors by Label")
                plt.colorbar(scatter, label="Label")
                
                st.pyplot(fig)
                
                # Store results in graph object
                graph.node_labels = labels
                if labeling_method == "Clustering-based":
                    graph.label_rules = rules
                else:
                    # Store rule strings with their associated labels and hops
                    graph.label_rules = [f"Label {label} ({hop}-hop): {rule_str}" for rule_fn, label, rule_str, hop in rules]
                    graph.applied_rules = applied_rules
                
                st.success("Analysis complete! Labels have been generated and stored in the graph object.")

# Add footer
st.markdown("""
---
**MMSB Explorer** | Mixed-Membership Stochastic Block Model for Graph Transfer Learning
""")