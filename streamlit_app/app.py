"""
Streamlit application for Mixed-Membership Stochastic Block Model visualization.

This application allows users to:
1. Generate and visualize graph universes with community structure
2. Sample and explore individual graphs from the universe
3. Generate graph families with controlled properties
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
from mmsb.model import GraphUniverse, GraphSample
from mmsb.graph_family import GraphFamilyGenerator
from mmsb.feature_regimes import (
    FeatureRegimeGenerator, 
    NeighborhoodFeatureAnalyzer, 
    FeatureRegimeLabelGenerator,
    GenerativeRuleBasedLabeler
)
from utils.visualizations import (
    plot_graph_communities, 
    plot_membership_matrix,
    plot_community_matrix,
    plot_community_graph,
    plot_degree_distribution,
    plot_community_overlap_distribution,
    create_dashboard,
)
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
from motif_and_role_analysis_integration import add_motif_role_analysis_page

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
if 'graph_family' not in st.session_state:
    st.session_state.graph_family = []

# Main header
st.markdown('<div class="main-header">Mixed-Membership Stochastic Block Model Explorer</div>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates the Mixed-Membership Stochastic Block Model (MMSB) for generating
graph datasets with community structure. The model supports generating families
of graphs with similar characteristics from a common universe.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Universe Creation", "Graph Sampling", "Graph Family Generation", "Parameter Space Analysis", "Motif and Role Analysis", "Neighborhood Analysis"]
)

# Universe Creation Page
if page == "Universe Creation":
    st.markdown('<div class="section-header">Universe Creation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    The <b>Graph Universe</b> defines the core structure from which all graph instances are sampled.
    It specifies:
    <ul>
        <li>Total number of possible communities</li>
        <li>Community connection patterns</li>
        <li>Feature generation rules (if enabled)</li>
        <li>Overlap structure between communities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Universe parameters
    st.markdown('<div class="subsection-header">Universe Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.slider("Number of communities", min_value=10, max_value=100, value=50)
        feature_dim = st.slider("Feature dimension", min_value=0, max_value=128, value=64)
        mixed_membership = st.checkbox(
            "Enable Mixed Membership",
            value=False,
            help="If enabled, nodes can belong to multiple communities. If disabled, each node belongs to exactly one community."
        )
        
    with col2:
        edge_density = st.slider(
            "Edge density",
            min_value=0.01,
            max_value=0.5,
            value=0.07,
            step=0.01,
            help="Overall probability of edges"
        )
        homophily = st.slider(
            "Homophily",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Strength of within-community connections"
        )
        randomness_factor = st.slider(
            "Randomness factor",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Amount of random noise in edge probabilities"
        )
    
    # Feature parameters if features are enabled
    if feature_dim > 0:
        st.markdown('<div class="subsection-header">Feature Parameters</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            intra_community_regime_similarity = st.slider(
                "Intra-community regime similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="How similar regimes within same community should be"
            )
            
        with col4:
            inter_community_regime_similarity = st.slider(
                "Inter-community regime similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="How similar regimes between communities should be"
            )
            
        regimes_per_community = st.slider(
            "Regimes per community",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of feature regimes per community"
        )
    else:
        intra_community_regime_similarity = 0.2
        inter_community_regime_similarity = 0.8
        regimes_per_community = 2
    
    # Add seed parameter
    seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")
    
    if st.button("Generate Universe"):
        with st.spinner("Generating universe..."):
            # Create universe
            universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                block_structure="assortative",  # Only assortative structure supported
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
                mixed_membership=mixed_membership,
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                regimes_per_community=regimes_per_community,
                seed=seed
            )
            
            # Store in session state
            st.session_state.universe = universe
            
            st.success("Universe generated successfully!")
            
            # Show universe properties
            st.markdown('<div class="subsection-header">Universe Properties</div>', unsafe_allow_html=True)
            
            # Plot community matrix
            fig = plot_community_matrix(universe.P, communities=range(K))
            st.pyplot(fig)
            
            # Show feature regimes if enabled
            if feature_dim > 0:
                st.markdown('<div class="subsection-header">Feature Regimes</div>', unsafe_allow_html=True)
                
                # Plot regime prototypes
                fig = plt.figure(figsize=(10, 6))
                plt.imshow(universe.regime_prototypes, aspect='auto', cmap='viridis')
                plt.colorbar(label='Feature Value')
                plt.title('Feature Regime Prototypes')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Regime')
                st.pyplot(fig)

# Graph Sampling Page
elif page == "Graph Sampling":
    st.markdown('<div class="section-header">Graph Sampling</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    else:
        st.markdown("""
        <div class="info-box">
        Sample individual graph instances from the universe with specific properties.
        Each graph will have:
        <ul>
            <li>Selected number of nodes</li>
            <li>Specific community structure</li>
            <li>Feature vectors (if enabled)</li>
            <li>Edge structure based on community memberships</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sampling parameters
        st.markdown('<div class="subsection-header">Sampling Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_nodes = st.slider("Number of nodes", min_value=30, max_value=300, value=80)
            num_communities = st.slider(
                "Number of communities",
                min_value=2,
                max_value=min(10, st.session_state.universe.K),
                value=5,
                help="Number of communities to include in the graph"
            )
            
        with col2:
            degree_heterogeneity = st.slider(
                "Degree heterogeneity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="How much node degrees should vary"
            )
            edge_noise = st.slider(
                "Edge noise",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                help="Amount of random noise in edge generation"
            )
        
        # Only show mixed membership related parameters if mixed membership is enabled
        if st.session_state.universe.mixed_membership:
            indirect_influence = st.slider(
                "Co-membership influence",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                help="How strongly co-memberships influence edge formation (0=no effect, 0.5=strong effect)"
            )
            feature_regime_balance = st.slider(
                "Feature Regime Balance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="How evenly regimes are distributed within communities (0=one regime dominates, 1=equal distribution)"
            )
        else:
            indirect_influence = 0.0
            feature_regime_balance = 0.5
        
        # Add seed parameter
        seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")
        
        if st.button("Generate Graph"):
            with st.spinner("Sampling graph..."):
                # Sample communities
                communities = st.session_state.universe.sample_community_subset(
                    size=num_communities,
                    method="random"
                )
                
                # Sample graph
                graph = GraphSample(
                    universe=st.session_state.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    degree_heterogeneity=degree_heterogeneity,
                    edge_noise=edge_noise,
                    indirect_influence=indirect_influence,
                    feature_regime_balance=feature_regime_balance,
                    seed=seed
                )
                
                # Store in session state
                st.session_state.current_graph = graph
                
                st.success("Graph generated successfully!")
                
                # Show graph properties
                st.markdown('<div class="subsection-header">Graph Properties</div>', unsafe_allow_html=True)
                
                # Plot community structure
                fig = plot_graph_communities(graph)
                st.pyplot(fig)
                
                # Plot membership matrix
                fig = plot_membership_matrix(graph.membership_vectors)
                st.pyplot(fig)
                
                # Plot degree distribution
                fig = plot_degree_distribution(graph.graph)
                st.pyplot(fig)
                
                # Show community overlap if mixed membership is enabled
                if st.session_state.universe.mixed_membership:
                    fig = plot_community_overlap_distribution(graph)
                    st.pyplot(fig)
                
                # Show feature distributions if enabled
                if graph.universe.feature_dim > 0:
                    st.markdown('<div class="subsection-header">Feature Distributions</div>', unsafe_allow_html=True)
                    
                    # Plot feature distributions
                    fig = plt.figure(figsize=(12, 6))
                    plt.imshow(graph.features, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Feature Value')
                    plt.title('Node Features')
                    plt.xlabel('Feature Dimension')
                    plt.ylabel('Node')
                    st.pyplot(fig)

# Parameter Space Analysis Page
elif page == "Parameter Space Analysis":
    st.markdown('<div class="section-header">Parameter Space Analysis</div>', unsafe_allow_html=True)
    
    # Check if we have graphs to analyze
    has_current = st.session_state.current_graph is not None
    has_family = len(st.session_state.graph_family) > 0
    
    if not (has_current or has_family):
        st.warning("Please generate a graph or graph family first in the previous pages.")
    else:
        st.markdown("""
        <div class="info-box">
        Parameter space analysis helps visualize where your graphs fall in the space of key graph properties:
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
        
        # Select which graphs to analyze
        st.markdown("### Select Graphs to Analyze")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_current = st.checkbox("Current Graph", value=has_current)
        
        with col2:
            analyze_family = st.checkbox("Graph Family", value=has_family)
        
        # Parameters to analyze
        st.markdown("### Select Parameters")
        
        param_options = [
            "homophily",
            "power_law_exponent",
            "clustering_coefficient",
            "triangle_density",
            "node_count",
            "avg_degree",
            "density",
            "connected_components",
            "largest_component_size"
        ]
        
        # Add community overlap only if mixed membership is enabled
        if st.session_state.universe is not None and st.session_state.universe.mixed_membership:
            param_options.append("avg_communities_per_node")
        
        selected_params = st.multiselect(
            "Parameters to analyze",
            param_options,
            default=["homophily", "clustering_coefficient", "avg_degree"]
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
                
                # Analyze graph family if selected
                if analyze_family and st.session_state.graph_family:
                    family_df = analyze_graph_family(st.session_state.graph_family)
                    family_dfs["Graph Family"] = family_df
                
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
                    This plot shows where different graphs lie in parameter space. 
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

# Motif and Role Analysis Page
elif page == "Motif and Role Analysis":
    add_motif_role_analysis_page()

# Neighborhood Analysis Page
elif page == "Neighborhood Analysis":
    st.markdown('<div class="section-header">Neighborhood Feature Analysis & Label Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    elif st.session_state.current_graph is None and not st.session_state.graph_family:
        st.warning("Please generate a graph in the 'Graph Sampling' page or a graph family in the 'Graph Family Generation' page.")
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
        if st.session_state.graph_family:
            graph_idx = st.selectbox(
                "Select graph from family",
                range(len(st.session_state.graph_family)),
                format_func=lambda x: f"Graph {x+1}"
            )
            graph = st.session_state.graph_family[graph_idx]
        else:
            graph = st.session_state.current_graph
            
        if graph.universe.feature_dim > 0:
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
                    if labeling_method == "Rule-based":
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
                        
                        # Generate labels based on rules
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
                    else:
                        # Clustering-based labeling
                        graph.compute_neighborhood_features(max_hops=1)
                        freq_vectors = {1: graph.neighborhood_analyzer.get_all_frequency_vectors(1)}
                        
                        label_generator = FeatureRegimeLabelGenerator(
                            frequency_vectors=freq_vectors[1],  # Use 1-hop for initial labels
                            n_labels=n_labels,
                            balance_tolerance=balance_tolerance,
                            seed=42
                        )
                        labels = label_generator.get_node_labels()
                        rules = label_generator._extract_label_rules()
                        applied_rules = None
                    
                    # Visualizations
                    st.markdown('<div class="subsection-header">Analysis Results</div>', unsafe_allow_html=True)
                    
                    # 1. Neighborhood Feature Distribution
                    st.markdown("#### Feature Regime Distribution by Hop Distance")
                    
                    # Only create tabs for the selected hop range
                    hop_range = list(range(min_hop, max_hop + 1)) if labeling_method == "Rule-based" else [1]
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
        else:
            st.warning("Neighborhood analysis is not available for graphs without features.")

# Graph Family Generation Page
elif page == "Graph Family Generation":
    st.markdown('<div class="section-header">Graph Family Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    else:
        st.markdown("""
        <div class="info-box">
        Generate a family of graphs that share similar properties but vary in their specific characteristics.
        Each graph in the family will have:
        <ul>
            <li>Different number of nodes</li>
            <li>Varying community structure</li>
            <li>Different feature distributions</li>
            <li>Varying edge patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Family generation parameters
        st.markdown('<div class="subsection-header">Family Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_graphs = st.slider("Number of graphs", min_value=1, max_value=100, value=20)
            min_nodes = st.slider("Minimum nodes", min_value=20, max_value=300, value=50)
            max_nodes = st.slider("Maximum nodes", min_value=min_nodes, max_value=500, value=100)
            min_communities = st.slider("Minimum communities", min_value=1, max_value=15, value=2)
            max_communities = st.slider("Maximum communities", min_value=min_communities, max_value=20, value=8)
        
        with col2:
            degree_heterogeneity = st.slider("Degree heterogeneity", min_value=0.0, max_value=1.0, value=0.5)
            edge_noise = st.slider("Edge noise", min_value=0.0, max_value=0.5, value=0.1)
            sampling_method = st.selectbox("Community sampling method", ["random", "similar", "diverse", "correlated"])
            min_component_size = st.slider("Minimum component size", min_value=2, max_value=50, value=5)
            feature_regime_balance = st.slider("Feature regime balance", min_value=0.0, max_value=1.0, value=0.5)
        
        # Generate button
        if st.button("Generate Graph Family"):
            with st.spinner("Generating graph family..."):
                # Initialize generator with existing universe
                generator = GraphFamilyGenerator(universe=st.session_state.universe)
                
                # Generate graph family
                graph_family = generator.generate(
                    n_graphs=n_graphs,
                    min_communities=min_communities,
                    max_communities=max_communities,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    degree_heterogeneity=degree_heterogeneity,
                    edge_noise=edge_noise,
                    sampling_method=sampling_method,
                    min_component_size=min_component_size,
                    feature_regime_balance=feature_regime_balance
                )
                
                # Store in session state
                st.session_state.graph_family = graph_family
                
                # Show success message
                st.success(f"Generated {n_graphs} graphs!")
                
                # Show up to 3 example graphs
                n_examples = min(3, len(graph_family))
                example_indices = np.random.choice(len(graph_family), size=n_examples, replace=False)
                
                for idx in example_indices:
                    graph = graph_family[idx]
                    st.markdown(f"**Graph {idx+1}**")
                    st.markdown(f"Nodes: {graph.n_nodes}, Edges: {graph.graph.number_of_edges()}, Communities: {len(graph.communities)}")
                    
                    fig = plot_graph_communities(graph)
                    st.pyplot(fig)

# Add footer
st.markdown("""
---
**MMSB Explorer** | Mixed-Membership Stochastic Block Model for Graph Transfer Learning
""") 