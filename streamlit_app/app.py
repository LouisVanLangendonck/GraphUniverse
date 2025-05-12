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
import plotly.graph_objects as go

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
from utils.graph_family_visualizations import (
    plot_parameter_distributions,
    plot_graph_statistics,
    plot_community_statistics,
    create_graph_family_dashboard,
    plot_graph_family_comparison
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
if 'graph_families' not in st.session_state:
    st.session_state.graph_families = {}

# Initialize session state for current family if not exists
if 'current_family_graphs' not in st.session_state:
    st.session_state.current_family_graphs = None
if 'current_family_params' not in st.session_state:
    st.session_state.current_family_params = None

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
    ["Universe Creation", "Graph Sampling", "Graph Family Generation", "Graph Family Analysis", "Parameter Space Analysis", "Motif and Role Analysis", "Neighborhood Analysis"]
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
            help="Controls the amount of Gaussian noise added to the universe's strength matrix. Noise has standard deviation = randomness_factor × min(p, q), where p and q are the within- and between-community strengths. At 1.0, noise can set some strengths to zero and can be large compared to the original values. Only negative values are clipped to zero; values above 1 are allowed and will be scaled/clipped later in graph instances."
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
                intra_community_regime_similarity=intra_community_regime_similarity,
                inter_community_regime_similarity=inter_community_regime_similarity,
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
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
        
        # Universal parameters
        st.markdown('<div class="subsection-header">Universal Parameters</div>', unsafe_allow_html=True)
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
            min_component_size = st.slider(
                "Minimum component size",
                min_value=1,
                max_value=50,
                value=3,
                help="Minimum size for connected components (all smaller components will be filtered out)"
            )
            sampling_method = st.selectbox(
                "Community sampling method",
                ["random", "similar", "diverse", "correlated"],
                help="Method for selecting community subsets"
            )
        with col2:
            max_mean_community_deviation = st.slider(
                "Max mean community deviation",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                help="Maximum allowed mean deviation from community structure"
            )
            max_max_community_deviation = st.slider(
                "Max max community deviation",
                min_value=0.01,
                max_value=0.5,
                value=0.2,
                help="Maximum allowed maximum deviation from community structure"
            )
            parameter_search_range = st.slider(
                "Parameter search range",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                help="How aggressively to search parameter space"
            )
            max_parameter_search_attempts = st.slider(
                "Max parameter search attempts",
                min_value=5,
                max_value=50,
                value=10,
                help="Maximum number of parameter combinations to try"
            )
            min_edge_density = st.slider(
                "Min edge density",
                min_value=0.001,
                max_value=0.1,
                value=0.005,
                help="Minimum acceptable edge density"
            )
            max_retries = st.slider(
                "Max retries",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of retries for edge generation"
            )
        
        # Edge generation method
        st.markdown('<div class="subsection-header">Edge Generation Method</div>', unsafe_allow_html=True)
        method = st.selectbox(
            "Edge generation method",
            ["Standard", "Power Law", "Exponential", "Uniform"],
            help="Choose the method for edge generation."
        )
        
        # Method-specific parameters
        method_params = {}
        if method == "Standard":
            st.markdown('<div class="subsection-header">Standard Method Parameters</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                degree_heterogeneity = st.slider(
                    "Degree heterogeneity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    help="How much node degrees should vary"
                )
            with col2:
                edge_noise = st.slider(
                    "Edge noise",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.0,
                    help="Amount of random noise in edge generation"
                )
            method_params = {
                'degree_heterogeneity': degree_heterogeneity,
                'edge_noise': edge_noise
            }
        elif method == "Power Law":
            st.markdown('<div class="subsection-header">Power Law Parameters</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                power_law_exponent = st.slider(
                    "Power Law Exponent",
                    min_value=1.5,
                    max_value=3.0,
                    value=2.1,
                    step=0.1,
                    help="Lower values create more skewed distributions with stronger hubs"
                )
                target_avg_degree = st.number_input(
                    "Target Average Degree",
                    min_value=1,
                    max_value=50,
                    value=4,
                    help="Target average degree for the graph. If None, calculated from density."
                )
            with col2:
                triangle_enhancement = st.slider(
                    "Triangle Enhancement",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    help="How much to enhance triangle formation (higher values increase clustering)"
                )
            method_params = {
                'degree_heterogeneity': 0.0,  # Not used
                'edge_noise': 0.0,  # Not used
                'power_law_exponent': power_law_exponent,
                'target_avg_degree': target_avg_degree,
                'triangle_enhancement': triangle_enhancement
            }
        elif method == "Exponential":
            st.markdown('<div class="subsection-header">Exponential Parameters</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                rate = st.slider(
                    "Rate",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    help="Rate parameter for the exponential degree distribution"
                )
                target_avg_degree = st.number_input(
                    "Target Average Degree",
                    min_value=1,
                    max_value=50,
                    value=4,
                    help="Target average degree for the graph. If None, calculated from density."
                )
            with col2:
                triangle_enhancement = st.slider(
                    "Triangle Enhancement",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    help="How much to enhance triangle formation (higher values increase clustering)"
                )
            method_params = {
                'degree_heterogeneity': 0.0,  # Not used
                'edge_noise': 0.0,  # Not used
                'rate': rate,
                'target_avg_degree': target_avg_degree,
                'triangle_enhancement': triangle_enhancement
            }
        elif method == "Uniform":
            st.markdown('<div class="subsection-header">Uniform Parameters</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                min_factor = st.slider(
                    "Min factor",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    help="Minimum factor for uniform degree distribution"
                )
                max_factor = st.slider(
                    "Max factor",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.5,
                    help="Maximum factor for uniform degree distribution"
                )
                target_avg_degree = st.number_input(
                    "Target Average Degree",
                    min_value=1,
                    max_value=50,
                    value=4,
                    help="Target average degree for the graph. If None, calculated from density."
                )
            with col2:
                triangle_enhancement = st.slider(
                    "Triangle Enhancement",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    help="How much to enhance triangle formation (higher values increase clustering)"
                )
            method_params = {
                'degree_heterogeneity': 0.0,  # Not used
                'edge_noise': 0.0,  # Not used
                'min_factor': min_factor,
                'max_factor': max_factor,
                'target_avg_degree': target_avg_degree,
                'triangle_enhancement': triangle_enhancement
            }
        
        # Add seed parameter
        seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")
        
        if st.button("Generate Graph"):
            with st.spinner("Sampling graph..."):
                # Sample communities
                communities = st.session_state.universe.sample_connected_community_subset(
                    size=num_communities,
                    method=sampling_method
                )
                # Build config_model_params for method-specific parameters
                config_model_params = {}
                if method == "Power Law":
                    config_model_params['power_law_exponent'] = method_params.get('power_law_exponent')
                if method == "Exponential":
                    config_model_params['rate'] = method_params.get('rate')
                if method == "Uniform":
                    config_model_params['min_factor'] = method_params.get('min_factor')
                    config_model_params['max_factor'] = method_params.get('max_factor')
                # Set up universal parameters for GraphSample
                universal_params = dict(
                    universe=st.session_state.universe,
                    communities=communities,
                    n_nodes=n_nodes,
                    min_component_size=min_component_size,
                    degree_heterogeneity=method_params.get('degree_heterogeneity', 0.0),
                    edge_noise=method_params.get('edge_noise', 0.0),
                    feature_regime_balance=0.5,  # Default, can be exposed if needed
                    target_homophily=None,  # Use universe default
                    target_density=None,    # Use universe default
                    use_configuration_model=(method != "Standard"),
                    degree_distribution=(
                        "power_law" if method == "Power Law"
                        else "exponential" if method == "Exponential"
                        else "uniform" if method == "Uniform"
                        else None
                    ),
                    power_law_exponent=method_params.get('power_law_exponent'),
                    target_avg_degree=method_params.get('target_avg_degree'),
                    triangle_enhancement=method_params.get('triangle_enhancement', 0.0),
                    max_mean_community_deviation=max_mean_community_deviation,
                    max_max_community_deviation=max_max_community_deviation,
                    max_parameter_search_attempts=max_parameter_search_attempts,
                    parameter_search_range=parameter_search_range,
                    min_edge_density=min_edge_density,
                    max_retries=max_retries,
                    seed=seed,
                    config_model_params=config_model_params
                )
                # Create the graph sample
                graph = GraphSample(**universal_params)
                st.session_state.current_graph = graph
                st.success("Graph generated successfully!")
                # Show graph properties
                st.markdown('<div class="subsection-header">Graph Properties</div>', unsafe_allow_html=True)
                
                # Plot community structure
                fig = plot_graph_communities(graph)
                st.pyplot(fig)
                
                # Plot membership matrix
                fig = plot_membership_matrix(graph)
                st.pyplot(fig)
                
                # Plot degree distribution
                fig = plot_degree_distribution(graph.graph)
                st.pyplot(fig)
                
                # Add community connection analysis
                st.markdown('<div class="subsection-header">Community Connection Analysis</div>', unsafe_allow_html=True)
                
                # Analyze community connections
                connection_analysis = graph.analyze_community_connections()
                
                # Verify that the deviations are within constraints
                if method != "Standard":
                    if connection_analysis['mean_deviation'] > max_mean_community_deviation:
                        st.error(f"Mean deviation ({connection_analysis['mean_deviation']:.4f}) exceeds constraint ({max_mean_community_deviation:.4f})")
                    if connection_analysis['max_deviation'] > max_max_community_deviation:
                        st.error(f"Maximum deviation ({connection_analysis['max_deviation']:.4f}) exceeds constraint ({max_max_community_deviation:.4f})")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs([
                    "Connection Matrices",
                    "Deviation Analysis",
                    "Community Statistics"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Expected Probabilities (Universe)")
                        fig = plt.figure(figsize=(6, 5))
                        plt.imshow(connection_analysis["expected_matrix"], cmap='viridis')
                        plt.colorbar(label='Probability')
                        plt.title('Expected Community Connections')
                        plt.xlabel('Community')
                        plt.ylabel('Community')
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("#### Actual Probabilities (Graph)")
                        fig = plt.figure(figsize=(6, 5))
                        plt.imshow(connection_analysis["actual_matrix"], cmap='viridis')
                        plt.colorbar(label='Probability')
                        plt.title('Actual Community Connections')
                        plt.xlabel('Community')
                        plt.ylabel('Community')
                        st.pyplot(fig)
                
                with tab2:
                    st.markdown("#### Deviation Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot deviation matrix
                        fig = plt.figure(figsize=(6, 5))
                        plt.imshow(connection_analysis["deviation_matrix"], cmap='Reds')
                        plt.colorbar(label='Absolute Deviation')
                        plt.title('Deviation from Expected Probabilities')
                        plt.xlabel('Community')
                        plt.ylabel('Community')
                        st.pyplot(fig)
                    
                    with col2:
                        # Show deviation statistics
                        st.markdown("##### Deviation Statistics")
                        st.metric("Mean Absolute Deviation", f"{connection_analysis['mean_deviation']:.4f}")
                        st.metric("Maximum Absolute Deviation", f"{connection_analysis['max_deviation']:.4f}")
                        
                        # Show deviation distribution
                        fig = plt.figure(figsize=(6, 4))
                        plt.hist(connection_analysis["deviation_matrix"].flatten(), bins=20)
                        plt.title('Distribution of Deviations')
                        plt.xlabel('Absolute Deviation')
                        plt.ylabel('Count')
                        st.pyplot(fig)
                    
                    # Add degree distribution analysis
                    st.markdown("#### Degree Distribution Analysis")
                    
                    # Get degree analysis results
                    degree_analysis = connection_analysis["degree_analysis"]
                    used_params = degree_analysis["used_parameters"]
                    
                    # Display the actual parameters used
                    st.write("### Distribution Parameters Used")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "power_law_exponent" in used_params and used_params["power_law_exponent"] is not None:
                            st.metric("Power Law Exponent", f"{used_params['power_law_exponent']:.2f}")
                        elif "rate" in used_params and used_params["rate"] is not None:
                            st.metric("Rate", f"{used_params['rate']:.2f}")
                        elif "min_factor" in used_params and used_params["min_factor"] is not None:
                            st.metric("Min Factor", f"{used_params['min_factor']:.2f}")
                    with col2:
                        if "target_avg_degree" in used_params and used_params["target_avg_degree"] is not None:
                            st.metric("Target Average Degree", f"{used_params['target_avg_degree']:.2f}")
                    with col3:
                        if "scale_factor" in used_params and used_params["scale_factor"] is not None and used_params["scale_factor"] != 1.0:
                            st.metric("Scale Factor", f"{used_params['scale_factor']:.2f}")
                    
                    # Display degree statistics with null checks
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("##### Actual Degree Statistics")
                        mean_actual = degree_analysis.get('mean_actual_degree')
                        std_actual = degree_analysis.get('std_actual_degree')
                        if mean_actual is not None:
                            st.write(f"Mean: {mean_actual:.2f}")
                        if std_actual is not None:
                            st.write(f"Std Dev: {std_actual:.2f}")
                    with col2:
                        st.write("##### Target Degree Statistics")
                        mean_target = degree_analysis.get('mean_target_degree')
                        std_target = degree_analysis.get('std_target_degree')
                        if mean_target is not None:
                            st.write(f"Mean: {mean_target:.2f}")
                        if std_target is not None:
                            st.write(f"Std Dev: {std_target:.2f}")
                    
                    # Display degree deviation and correlation with null checks
                    col1, col2 = st.columns(2)
                    with col1:
                        deviation = degree_analysis.get('degree_deviation')
                        if deviation is not None:
                            st.metric("Degree Deviation", f"{deviation:.4f}")
                    with col2:
                        correlation = degree_analysis.get('degree_correlation')
                        if correlation is not None:
                            st.metric("Degree Correlation", f"{correlation:.4f}")
                
                with tab3:
                    st.markdown("#### Community Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Show community sizes
                        st.markdown("##### Community Sizes")
                        fig = plt.figure(figsize=(6, 4))
                        plt.bar(range(len(connection_analysis["community_sizes"])), 
                               connection_analysis["community_sizes"])
                        plt.title('Number of Nodes per Community')
                        plt.xlabel('Community')
                        plt.ylabel('Number of Nodes')
                        st.pyplot(fig)
                    
                    with col2:
                        # Show connection counts
                        fig = plt.figure(figsize=(6, 5))
                        plt.imshow(connection_analysis["connection_counts"], cmap='Blues')
                        plt.colorbar(label='Number of Edges')
                        plt.title('Raw Edge Counts Between Communities')
                        plt.xlabel('Community')
                        plt.ylabel('Community')
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
                
                fig = create_parameter_dashboard(df)
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
    
    st.markdown("""
    <div class="info-box">
    Generate families of graphs with controlled properties. Each family will have:
    <ul>
        <li>Multiple graphs with similar characteristics</li>
        <li>Controlled community structure</li>
        <li>Consistent parameter distributions</li>
        <li>Feature vectors (if enabled)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Family name input
    family_name = st.text_input("Family Name", value="Family 1", help="Name for this graph family")
    
    # Basic parameters
    st.markdown('<div class="subsection-header">Basic Parameters</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        n_graphs = st.slider("Number of graphs", 1, 100, 10)
        min_nodes = st.slider("Minimum nodes", 10, 1000, 30)
        max_nodes = st.slider("Maximum nodes", 10, 1000, 150)
    with col2:
        min_communities = st.slider("Minimum communities", 2, 20, 3)
        max_communities = st.slider("Maximum communities", 2, 20, 7)
        sampling_method = st.selectbox(
            "Community sampling method",
            ["random", "similar", "diverse", "correlated"],
            help="Method for selecting community subsets"
        )
    
    # Global generation parameters
    st.markdown('<div class="subsection-header">Global Generation Parameters</div>', unsafe_allow_html=True)
    st.markdown("These parameters apply to all generation methods:")
    
    col1, col2 = st.columns(2)
    with col1:
        max_mean_community_deviation = st.slider(
            "Max mean community deviation",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            help="Maximum allowed mean deviation from community structure"
        )
        max_max_community_deviation = st.slider(
            "Max max community deviation",
            min_value=0.01,
            max_value=0.5,
            value=0.10,
            help="Maximum allowed maximum deviation from community structure"
        )
        parameter_search_range = st.slider(
            "Parameter search range",
            min_value=0.05,
            max_value=1.0,
            value=0.4,
            help="How aggressively to search parameter space"
        )
    
    with col2:
        max_parameter_search_attempts = st.slider(
            "Max parameter search attempts",
            min_value=5,
            max_value=50,
            value=40,
            help="Maximum number of parameter combinations to try"
        )
        min_edge_density = st.slider(
            "Min edge density",
            min_value=0.001,
            max_value=0.1,
            value=0.005,
            help="Minimum acceptable edge density"
        )
        max_retries = st.slider(
            "Max retries",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum number of retries for edge generation"
        )
    
    min_component_size = st.slider(
        "Minimum component size",
        min_value=1,
        max_value=50,
        value=3,
        help="Minimum size for connected components (all smaller components will be filtered out)"
    )
    
    # Method distribution
    st.markdown('<div class="subsection-header">Generation Method Distribution</div>', unsafe_allow_html=True)
    st.markdown("Configure the probability of each generation method:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        standard_prob = st.slider("Standard method", min_value=0.0, max_value=1.0, value=0.5)
    with col2:
        power_law_prob = st.slider("Power law", min_value=0.0, max_value=1.0, value=0.3)
    with col3:
        exponential_prob = st.slider("Exponential", min_value=0.0, max_value=1.0, value=0.1)
    with col4:
        uniform_prob = st.slider("Uniform", min_value=0.0, max_value=1.0, value=0.1)
    
    # Normalize probabilities
    total_prob = standard_prob + power_law_prob + exponential_prob + uniform_prob
    if total_prob > 0:
        method_distribution = {
            "standard": standard_prob / total_prob,
            "power_law": power_law_prob / total_prob,
            "exponential": exponential_prob / total_prob,
            "uniform": uniform_prob / total_prob
        }
    else:
        method_distribution = {"standard": 1.0}  # Default to standard if all probabilities are 0
    
    # Method-specific parameters
    st.markdown('<div class="subsection-header">Method-Specific Parameters</div>', unsafe_allow_html=True)
    
    # Standard method parameters
    st.markdown("#### Standard Method Parameters")
    col1, col2 = st.columns(2)
    with col1:
        degree_heterogeneity = st.slider(
            "Degree heterogeneity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="How much node degrees should vary"
        )
    with col2:
        edge_noise = st.slider(
            "Edge noise",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            help="Amount of random noise in edge generation"
        )
    
    # Power law parameters
    st.markdown("#### Power Law Parameters")
    col1, col2 = st.columns(2)
    with col1:
        power_law_exponent_min = st.slider("Min power law exponent", min_value=1.0, max_value=4.0, value=1.5)
    with col2:
        power_law_exponent_max = st.slider("Max power law exponent", min_value=1.0, max_value=4.0, value=3.0)
    
    # Exponential parameters
    st.markdown("#### Exponential Parameters")
    col1, col2 = st.columns(2)
    with col1:
        rate_min = st.slider("Min rate", min_value=0.1, max_value=2.0, value=0.1)
    with col2:
        rate_max = st.slider("Max rate", min_value=0.1, max_value=2.0, value=1.0)
    
    # Uniform parameters
    st.markdown("#### Uniform Parameters")
    col1, col2 = st.columns(2)
    with col1:
        min_factor = st.slider("Min factor", min_value=0.1, max_value=1.0, value=0.5)
    with col2:
        max_factor = st.slider("Max factor", min_value=1.0, max_value=2.0, value=1.5)
    
    # Common parameters for configuration models
    st.markdown("#### Common Configuration Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        target_avg_degree_min = st.slider("Min target average degree", min_value=1.0, max_value=50.0, value=2.0)
    with col2:
        target_avg_degree_max = st.slider("Max target average degree", min_value=1.0, max_value=50.0, value=5.0)
    
    # Generate button
    if st.button("Generate Graph Family"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize generator
        generator = GraphFamilyGenerator(
            K=st.session_state.universe.K,
            feature_dim=st.session_state.universe.feature_dim,
            block_structure="assortative",
            edge_density=st.session_state.universe.edge_density,
            homophily=st.session_state.universe.homophily,
            randomness_factor=st.session_state.universe.randomness_factor,
            intra_community_regime_similarity=st.session_state.universe.intra_community_regime_similarity,
            inter_community_regime_similarity=st.session_state.universe.inter_community_regime_similarity,
            regimes_per_community=st.session_state.universe.regimes_per_community,
            method_distribution=method_distribution,
            standard_method_params={
                "degree_heterogeneity": degree_heterogeneity,
                "edge_noise": edge_noise
            },
            config_model_params={
                "power_law": {
                    "exponent_min": power_law_exponent_min,
                    "exponent_max": power_law_exponent_max,
                    "target_avg_degree_min": target_avg_degree_min,
                    "target_avg_degree_max": target_avg_degree_max
                },
                "exponential": {
                    "rate_min": rate_min,
                    "rate_max": rate_max,
                    "target_avg_degree_min": target_avg_degree_min,
                    "target_avg_degree_max": target_avg_degree_max
                },
                "uniform": {
                    "min_factor": min_factor,
                    "max_factor": max_factor,
                    "target_avg_degree_min": target_avg_degree_min,
                    "target_avg_degree_max": target_avg_degree_max
                }
            },
            max_mean_community_deviation=max_mean_community_deviation,
            max_max_community_deviation=max_max_community_deviation,
            max_parameter_search_attempts=max_parameter_search_attempts,
            parameter_search_range=parameter_search_range,
            min_edge_density=min_edge_density,
            max_retries=max_retries
        )
        
        # Generate graphs with progress updates
        status_text.text("Generating graphs...")
        graph_family = []
        parameter_samples = defaultdict(list)
        attempts = 0
        max_attempts = n_graphs * 3
        
        while len(graph_family) < n_graphs and attempts < max_attempts:
            try:
                # Generate a single graph
                new_graphs = generator.generate(
                    n_graphs=1,
                    min_communities=min_communities,
                    max_communities=max_communities,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    sampling_method=sampling_method,
                    min_component_size=min_component_size
                )
                
                if new_graphs:
                    graph = new_graphs[0]
                    graph_family.append(graph)
                    
                    # Store parameter samples
                    params = graph.extract_parameters()
                    for param_name, value in params.items():
                        parameter_samples[param_name].append(value)
                    
                    progress = len(graph_family) / n_graphs
                    progress_bar.progress(progress)
                    status_text.text(f"Generated {len(graph_family)} out of {n_graphs} graphs")
                
            except Exception as e:
                st.warning(f"Failed to generate graph {len(graph_family) + 1}: {str(e)}")
            
            attempts += 1
        
        if len(graph_family) < n_graphs:
            st.warning(f"Could only generate {len(graph_family)} out of {n_graphs} graphs after {attempts} attempts")
        
        # Store in session state
        if graph_family:
            # Store in both places
            st.session_state.graph_families[family_name] = {
                'graphs': graph_family,
                'parameters': dict(parameter_samples)
            }
            st.session_state.graph_family = graph_family  # For Parameter Space Analysis
            
            # Store current family for visualization
            st.session_state.current_family_graphs = graph_family
            st.session_state.current_family_params = dict(parameter_samples)
            
            st.success(f"Successfully generated {len(graph_family)} graphs for family '{family_name}'!")
    
    # Display visualization if we have a current family
    if st.session_state.current_family_graphs is not None:
        # Display basic statistics
        st.markdown('<div class="subsection-header">Graph Family Statistics</div>', unsafe_allow_html=True)
        stats = {
            "Number of graphs": len(st.session_state.current_family_graphs),
            "Average nodes": np.mean([g.n_nodes for g in st.session_state.current_family_graphs]),
            "Average edges": np.mean([g.graph.number_of_edges() for g in st.session_state.current_family_graphs]),
            "Average density": np.mean([nx.density(g.graph) for g in st.session_state.current_family_graphs]),
            "Average clustering": np.mean([nx.average_clustering(g.graph) for g in st.session_state.current_family_graphs])
        }
        
        # Display statistics in a table
        st.table(pd.DataFrame([stats]))
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Graph Visualization",
            "Degree Distribution",
            "Community Analysis",
            "Community Deviation Analysis"
        ])
        
        with tab1:
            # Select a graph to visualize with a unique key
            graph_idx = st.selectbox(
                "Select graph to visualize",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_viz_{family_name}"
            )
            graph = st.session_state.current_family_graphs[graph_idx]
            
            # Get generation method and parameters for title
            generation_info = "Method: Unknown"
            if hasattr(graph, 'generation_method'):
                method = graph.generation_method
                params = graph.generation_params
                if method == "standard":
                    generation_info = f"Method: Standard (heterogeneity={params.get('degree_heterogeneity', 'N/A'):.2f}, noise={params.get('edge_noise', 'N/A'):.2f})"
                elif method == "power_law":
                    generation_info = f"Method: Power Law (exponent={params.get('power_law_exponent', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "exponential":
                    generation_info = f"Method: Exponential (rate={params.get('rate', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "uniform":
                    generation_info = f"Method: Uniform (min={params.get('min_factor', 'N/A'):.2f}, max={params.get('max_factor', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
            
            st.markdown(f"#### Graph {graph_idx + 1}")
            st.markdown(f"*{generation_info}*")
            
            # Create visualization
            fig = plot_graph_communities(graph)
            st.pyplot(fig)
        
        with tab2:
            # Select a graph for degree distribution with a unique key
            graph_idx = st.selectbox(
                "Select graph for degree distribution",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_degree_{family_name}"
            )
            graph = st.session_state.current_family_graphs[graph_idx]
            
            # Get generation method and parameters for title
            generation_info = "Method: Unknown"
            if hasattr(graph, 'generation_method'):
                method = graph.generation_method
                params = graph.generation_params
                if method == "standard":
                    generation_info = f"Method: Standard (heterogeneity={params.get('degree_heterogeneity', 'N/A'):.2f}, noise={params.get('edge_noise', 'N/A'):.2f})"
                elif method == "power_law":
                    generation_info = f"Method: Power Law (exponent={params.get('power_law_exponent', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "exponential":
                    generation_info = f"Method: Exponential (rate={params.get('rate', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "uniform":
                    generation_info = f"Method: Uniform (min={params.get('min_factor', 'N/A'):.2f}, max={params.get('max_factor', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
            
            st.markdown(f"#### Graph {graph_idx + 1}")
            st.markdown(f"*{generation_info}*")
            
            # Create degree distribution plots
            fig = plot_degree_distribution(graph.graph)
            st.pyplot(fig)
        
        with tab3:
            # Select a graph for community analysis with a unique key
            graph_idx = st.selectbox(
                "Select graph for community analysis",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_community_{family_name}"
            )
            graph = st.session_state.current_family_graphs[graph_idx]
            
            # Get generation method and parameters for title
            generation_info = "Method: Unknown"
            if hasattr(graph, 'generation_method'):
                method = graph.generation_method
                params = graph.generation_params
                if method == "standard":
                    generation_info = f"Method: Standard (heterogeneity={params.get('degree_heterogeneity', 'N/A'):.2f}, noise={params.get('edge_noise', 'N/A'):.2f})"
                elif method == "power_law":
                    generation_info = f"Method: Power Law (exponent={params.get('power_law_exponent', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "exponential":
                    generation_info = f"Method: Exponential (rate={params.get('rate', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "uniform":
                    generation_info = f"Method: Uniform (min={params.get('min_factor', 'N/A'):.2f}, max={params.get('max_factor', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
            
            st.markdown(f"#### Graph {graph_idx + 1}")
            st.markdown(f"*{generation_info}*")
            
            # Create community analysis plots
            fig = plot_membership_matrix(graph)
            st.pyplot(fig)
        
        with tab4:
            # Select a graph for deviation analysis with a unique key
            graph_idx = st.selectbox(
                "Select graph for deviation analysis",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_deviation_{family_name}"
            )
            graph = st.session_state.current_family_graphs[graph_idx]
            
            # Get generation method and parameters for title
            generation_info = "Method: Unknown"
            if hasattr(graph, 'generation_method'):
                method = graph.generation_method
                params = graph.generation_params
                if method == "standard":
                    generation_info = f"Method: Standard (heterogeneity={params.get('degree_heterogeneity', 'N/A'):.2f}, noise={params.get('edge_noise', 'N/A'):.2f})"
                elif method == "power_law":
                    generation_info = f"Method: Power Law (exponent={params.get('power_law_exponent', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "exponential":
                    generation_info = f"Method: Exponential (rate={params.get('rate', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
                elif method == "uniform":
                    generation_info = f"Method: Uniform (min={params.get('min_factor', 'N/A'):.2f}, max={params.get('max_factor', 'N/A'):.2f}, avg_degree={params.get('target_avg_degree', 'N/A'):.2f})"
            
            st.markdown(f"#### Graph {graph_idx + 1}")
            st.markdown(f"*{generation_info}*")
            
            # Analyze community connections
            connection_analysis = graph.analyze_community_connections()
            
            # Create subtabs for different views
            subtab1, subtab2, subtab3 = st.tabs([
                "Connection Matrices",
                "Deviation Analysis",
                "Community Statistics"
            ])
            
            with subtab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Expected Probabilities (Universe)")
                    fig = plt.figure(figsize=(6, 5))
                    plt.imshow(connection_analysis["expected_matrix"], cmap='viridis')
                    plt.colorbar(label='Probability')
                    plt.title('Expected Community Connections')
                    plt.xlabel('Community')
                    plt.ylabel('Community')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("##### Actual Probabilities (Graph)")
                    fig = plt.figure(figsize=(6, 5))
                    plt.imshow(connection_analysis["actual_matrix"], cmap='viridis')
                    plt.colorbar(label='Probability')
                    plt.title('Actual Community Connections')
                    plt.xlabel('Community')
                    plt.ylabel('Community')
                    st.pyplot(fig)
            
            with subtab2:
                st.markdown("##### Deviation Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot deviation matrix
                    fig = plt.figure(figsize=(6, 5))
                    plt.imshow(connection_analysis["deviation_matrix"], cmap='Reds')
                    plt.colorbar(label='Absolute Deviation')
                    plt.title('Deviation from Expected Probabilities')
                    plt.xlabel('Community')
                    plt.ylabel('Community')
                    st.pyplot(fig)
                
                with col2:
                    # Show deviation statistics
                    st.markdown("##### Deviation Statistics")
                    st.metric("Mean Absolute Deviation", f"{connection_analysis['mean_deviation']:.4f}")
                    st.metric("Maximum Absolute Deviation", f"{connection_analysis['max_deviation']:.4f}")
                    
                    # Show deviation distribution
                    fig = plt.figure(figsize=(6, 4))
                    plt.hist(connection_analysis["deviation_matrix"].flatten(), bins=20)
                    plt.title('Distribution of Deviations')
                    plt.xlabel('Absolute Deviation')
                    plt.ylabel('Count')
                    st.pyplot(fig)
                
                # Add degree distribution analysis if available
                if "degree_analysis" in connection_analysis:
                    st.markdown("##### Degree Distribution Analysis")
                    degree_analysis = connection_analysis["degree_analysis"]
                    used_params = degree_analysis["used_parameters"]
                    
                    # Display the actual parameters used
                    st.write("##### Distribution Parameters Used")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "power_law_exponent" in used_params and used_params["power_law_exponent"] is not None:
                            st.metric("Power Law Exponent", f"{used_params['power_law_exponent']:.2f}")
                        elif "rate" in used_params and used_params["rate"] is not None:
                            st.metric("Rate", f"{used_params['rate']:.2f}")
                        elif "min_factor" in used_params and used_params["min_factor"] is not None:
                            st.metric("Min Factor", f"{used_params['min_factor']:.2f}")
                    with col2:
                        if "target_avg_degree" in used_params and used_params["target_avg_degree"] is not None:
                            st.metric("Target Average Degree", f"{used_params['target_avg_degree']:.2f}")
                    with col3:
                        if "scale_factor" in used_params and used_params["scale_factor"] is not None and used_params["scale_factor"] != 1.0:
                            st.metric("Scale Factor", f"{used_params['scale_factor']:.2f}")
                    
                    # Display degree statistics with null checks
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("##### Actual Degree Statistics")
                        mean_actual = degree_analysis.get('mean_actual_degree')
                        std_actual = degree_analysis.get('std_actual_degree')
                        if mean_actual is not None:
                            st.write(f"Mean: {mean_actual:.2f}")
                        if std_actual is not None:
                            st.write(f"Std Dev: {std_actual:.2f}")
                    with col2:
                        st.write("##### Target Degree Statistics")
                        mean_target = degree_analysis.get('mean_target_degree')
                        std_target = degree_analysis.get('std_target_degree')
                        if mean_target is not None:
                            st.write(f"Mean: {mean_target:.2f}")
                        if std_target is not None:
                            st.write(f"Std Dev: {std_target:.2f}")
                    
                    # Display degree deviation and correlation with null checks
                    col1, col2 = st.columns(2)
                    with col1:
                        deviation = degree_analysis.get('degree_deviation')
                        if deviation is not None:
                            st.metric("Degree Deviation", f"{deviation:.4f}")
                    with col2:
                        correlation = degree_analysis.get('degree_correlation')
                        if correlation is not None:
                            st.metric("Degree Correlation", f"{correlation:.4f}")
            
            with subtab3:
                st.markdown("##### Community Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show community sizes
                    st.markdown("##### Community Sizes")
                    fig = plt.figure(figsize=(6, 4))
                    plt.bar(range(len(connection_analysis["community_sizes"])), 
                           connection_analysis["community_sizes"])
                    plt.title('Number of Nodes per Community')
                    plt.xlabel('Community')
                    plt.ylabel('Number of Nodes')
                    st.pyplot(fig)
                
                with col2:
                    # Show connection counts
                    fig = plt.figure(figsize=(6, 5))
                    plt.imshow(connection_analysis["connection_counts"], cmap='Blues')
                    plt.colorbar(label='Number of Edges')
                    plt.title('Raw Edge Counts Between Communities')
                    plt.xlabel('Community')
                    plt.ylabel('Community')
                    st.pyplot(fig)

# Graph Family Analysis Page
elif page == "Graph Family Analysis":
    st.markdown('<div class="section-header">Graph Family Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Analyze and compare graph families to understand their properties and relationships.
    Features include:
    <ul>
        <li>Comprehensive family dashboards</li>
        <li>Parameter distribution analysis</li>
        <li>Graph statistics comparison</li>
        <li>Community structure analysis</li>
        <li>Cross-family comparisons</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if any families are available
    if not st.session_state.graph_families:
        st.warning("No graph families available for analysis. Please generate some families first.")
    else:
        # Family selection
        selected_families = st.multiselect(
            "Select families to analyze",
            options=list(st.session_state.graph_families.keys()),
            default=list(st.session_state.graph_families.keys())[:1]
        )
        
        if selected_families:
            # Analysis type selection
            analysis_type = st.selectbox(
                "Select analysis type",
                [
                    "Comprehensive Dashboard",
                    "Parameter Distributions",
                    "Graph Statistics",
                    "Community Statistics",
                    "Family Comparison"
                ]
            )
            
            if analysis_type == "Comprehensive Dashboard":
                # Create dashboard for each selected family
                for family_name in selected_families:
                    st.markdown(f'<div class="subsection-header">{family_name} Dashboard</div>', unsafe_allow_html=True)
                    
                    family_data = st.session_state.graph_families[family_name]
                    graphs = family_data['graphs']
                    parameters = family_data['parameters']
                    
                    # Create dashboard
                    fig = create_graph_family_dashboard(graphs, parameters)
                    st.pyplot(fig)
            
            elif analysis_type == "Parameter Distributions":
                # Plot parameter distributions for selected families
                fig = plot_parameter_distributions(
                    [st.session_state.graph_families[f]['parameters'] for f in selected_families],
                    selected_families
                )
                st.pyplot(fig)
            
            elif analysis_type == "Graph Statistics":
                # Plot graph statistics for selected families
                fig = plot_graph_statistics(
                    [st.session_state.graph_families[f]['graphs'] for f in selected_families],
                    selected_families
                )
                st.pyplot(fig)
            
            elif analysis_type == "Community Statistics":
                # Plot community statistics for selected families
                fig = plot_community_statistics(
                    [st.session_state.graph_families[f]['graphs'] for f in selected_families],
                    selected_families
                )
                st.pyplot(fig)
            
            elif analysis_type == "Family Comparison":
                # Compare statistics across families
                fig = plot_graph_family_comparison(
                    [st.session_state.graph_families[f]['graphs'] for f in selected_families],
                    selected_families
                )
                st.pyplot(fig)

# Add footer
st.markdown("""
---
**MMSB Explorer** | Mixed-Membership Stochastic Block Model for Graph Transfer Learning
""") 

def create_graph_dashboard(graph):
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Graph Visualization",
        "Community Distribution",
        "Parameter Analysis",
        "Feature Analysis"
    ])

    with tab1:
        st.subheader("Graph with Communities")
        fig = plot_graph_communities(graph)
        st.pyplot(fig)

    with tab2:
        st.subheader("Community Distribution")
        # Pass the graph object directly
        fig = plot_membership_matrix(graph)
        st.pyplot(fig) 