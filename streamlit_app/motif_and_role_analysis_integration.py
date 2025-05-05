"""
Integration code to add motif and role analysis functionality to the MMSB app.

This module provides functions to extend app.py with motif participation profiles
and structural role decomposition capabilities.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from utils.motif_and_role_analysis import MotifRoleAnalyzer

def add_motif_role_analysis_tab(graph_tabs):
    """
    Add a motif and role analysis tab to the graph visualization tabs.
    
    Args:
        graph_tabs: Streamlit tabs object
    """
    # Create a new tab for motif and role analysis
    with graph_tabs[6]:  # Assuming this is the 7th tab (index 6)
        st.markdown("### Structural Analysis: Motifs & Roles")
        
        st.markdown("""
        This analysis combines motif participation profiles with structural role decomposition
        to identify meaningful structural patterns and node roles in the network.
        
        **Motif participation** identifies how often nodes participate in specific structural patterns,
        while **role decomposition** groups nodes with similar structural behaviors.
        """)
        
        # Analysis parameters
        st.markdown("#### Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_motif_size = st.slider(
                "Maximum motif size", 
                min_value=3, 
                max_value=4, 
                value=4,
                help="Largest motif pattern to analyze (higher values increase computation time)"
            )
            
            normalize_features = st.checkbox(
                "Normalize features",
                value=True,
                help="Normalize motif counts to emphasize structural patterns over size effects"
            )
        
        with col2:
            n_roles = st.slider(
                "Number of roles",
                min_value=2,
                max_value=8,
                value=5,
                help="Number of structural roles to discover in the network"
            )
        
        # Run analysis button
        if st.button("Run Structural Analysis"):
            # Check if the graph is available
            if st.session_state.current_graph is None:
                st.warning("Please generate a graph first in the 'Graph Sampling' page.")
                return
                
            # Check if graph is too large for interactive analysis
            graph = st.session_state.current_graph.graph
            if graph.number_of_nodes() > 1000:
                st.warning(f"Graph has {graph.number_of_nodes()} nodes, which may be too large for interactive analysis. Consider using a smaller graph or running offline.")
                proceed = st.checkbox("Proceed anyway (may be slow)", value=False)
                if not proceed:
                    return
            
            # Show progress and run analysis
            with st.spinner("Analyzing structural patterns and roles... This may take a while for larger graphs."):
                try:
                    # Initialize analyzer
                    analyzer = MotifRoleAnalyzer(
                        graph_sample=st.session_state.current_graph,
                        max_motif_size=max_motif_size,
                        n_roles=n_roles,
                        normalize_features=normalize_features,
                        verbose=True
                    )
                    
                    # Run motif profile extraction
                    st.text("Computing motif participation profiles...")
                    analyzer.compute_motif_profiles()
                    
                    # Run role discovery
                    st.text("Discovering structural roles...")
                    analyzer.discover_structural_roles()
                    
                    # Store analyzer in session state
                    st.session_state.role_analyzer = analyzer
                    
                    st.success("Structural analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return
            
            # Display results
            if hasattr(st.session_state, 'role_analyzer'):
                display_motif_role_analysis_results(st.session_state.role_analyzer)

def display_motif_role_analysis_results(analyzer):
    """
    Display the results of motif and role analysis.
    
    Args:
        analyzer: MotifRoleAnalyzer object with completed analysis
    """
    st.markdown("## Structural Analysis Results")
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs([
        "Role Dashboard", 
        "Motif Profiles", 
        "Role Definitions",
        "Role Distribution",
        "Community Analysis",
        "Labeling Rules"
    ])
    
    # 1. Main Dashboard tab
    with analysis_tabs[0]:
        st.markdown("### Structural Role Dashboard")
        
        # Display dashboard visualization
        dashboard_fig = analyzer.create_role_dashboard()
        st.pyplot(dashboard_fig)
        
        st.markdown("""
        This dashboard provides a comprehensive view of the structural roles identified in the network:
        - **Role Definitions**: How each role is defined in terms of motif participation
        - **Role Distribution**: How many nodes belong to each structural role
        - **Roles by Community**: How structural roles are distributed across communities
        - **Role-Community Correlation**: Correlation between roles and community membership
        - **Role Transitions**: How roles connect to each other in the network
        - **Network with Roles**: Visualization of the network with nodes colored by role
        """)
        
        # Display role metrics table
        st.markdown("### Role Metrics")
        metrics_df = analyzer.get_role_metrics()
        st.dataframe(metrics_df)
    
    # 2. Motif Profiles tab
    with analysis_tabs[1]:
        st.markdown("### Motif Participation Profiles")
        
        # Display motif profile visualization
        motif_fig = analyzer.visualize_motif_profiles()
        st.pyplot(motif_fig)
        
        st.markdown("""
        This heatmap shows motif participation profiles for a sample of nodes:
        - Each row represents a node
        - Each column represents a motif or structural feature
        - Colors indicate how frequently the node participates in each motif (normalized)
        
        These profiles form the foundation for structural role discovery.
        """)
        
        # Explanation of motifs
        st.markdown("### Motif Definitions")
        
        # Create a table explaining each motif
        motif_desc = []
        for i, motif_name in enumerate(analyzer.motif_names):
            desc = analyzer.motif_set[motif_name]["description"]
            motif_desc.append({"Motif": motif_name, "Description": desc})
        
        motif_df = pd.DataFrame(motif_desc)
        st.table(motif_df)
    
    # 3. Role Definitions tab
    with analysis_tabs[2]:
        st.markdown("### Structural Role Definitions")
        
        # Display role definitions visualization
        role_def_fig = analyzer.visualize_role_definitions()
        st.pyplot(role_def_fig)
        
        # Display role interpretations
        role_interpretations = analyzer.interpret_roles()
        
        st.markdown("### Role Interpretations")
        for role_idx, role_info in role_interpretations.items():
            st.markdown(f"**{role_info['name']}**: {role_info['description']}")
        
        # Display role similarity
        st.markdown("### Role Similarity Matrix")
        role_sim_fig = analyzer.visualize_role_similarity()
        st.pyplot(role_sim_fig)
        
        st.markdown("""
        The similarity matrix shows how similar roles are to each other based on their motif compositions.
        Higher values indicate more similar roles.
        """)
    
    # 4. Role Distribution tab
    with analysis_tabs[3]:
        st.markdown("### Role Distribution")
        
        # Display role distribution visualization
        role_dist_fig = analyzer.visualize_role_distribution()
        st.pyplot(role_dist_fig)
        
        # Display role membership for sample nodes
        st.markdown("### Role Membership for Sample Nodes")
        role_mem_fig = analyzer.visualize_role_membership()
        st.pyplot(role_mem_fig)
        
        st.markdown("""
        This heatmap shows the degree of membership of each node in the discovered roles.
        Brighter colors indicate stronger membership in a role.
        
        Nodes typically have mixed membership across multiple roles, with one or two dominant roles.
        """)
        
        # Display role transitions
        st.markdown("### Role Transitions")
        role_trans_fig = analyzer.visualize_node_role_transitions()
        st.pyplot(role_trans_fig)
        
        st.markdown("""
        This matrix shows the probability of transitions between roles along network edges.
        Higher values indicate that nodes with the row role frequently connect to nodes with the column role.
        """)
    
    # 5. Community Analysis tab
    with analysis_tabs[4]:
        st.markdown("### Role Distribution by Community")
        
        # Display roles by community visualization
        comm_role_fig = analyzer.visualize_roles_by_community()
        st.pyplot(comm_role_fig)
        
        # Display role-community correlation
        st.markdown("### Role-Community Correlation")
        role_comm_fig = analyzer.visualize_role_community_correlation()
        st.pyplot(role_comm_fig)
        
        st.markdown("""
        This correlation matrix shows how structural roles relate to community membership:
        - **Positive correlation (red)**: The role is overrepresented in the community
        - **Negative correlation (blue)**: The role is underrepresented in the community
        - **Near-zero correlation**: The role is independent of community membership
        
        Strong correlations indicate that certain communities have distinctive structural signatures.
        """)
        
        # Display network with roles
        st.markdown("### Network Visualization with Roles")
        graph_role_fig = analyzer.visualize_graph_with_roles()
        st.pyplot(graph_role_fig)
    
    # 6. Labeling Rules tab
    with analysis_tabs[5]:
        st.markdown("### Structural Labeling Rules")
        
        st.markdown("""
        Structural roles can be used to create node labels for machine learning tasks.
        These labels capture the structural position of nodes in the network.
        """)
        
        # Display labeling methods
        st.markdown("#### Labeling Methods")
        
        labeling_methods = [
            {
                "Method": "Primary Role",
                "Description": "Assign each node to its dominant role",
                "Example": "Node 42 → 'Bridge Node'"
            },
            {
                "Method": "Mixed Roles",
                "Description": "Assign nodes to multiple roles with significant membership",
                "Example": "Node 15 → 'Hub + Community Core'"
            },
            {
                "Method": "Role-Community",
                "Description": "Combine structural role with community information",
                "Example": "Node 23 → 'Bridge Node (C5)'"
            },
            {
                "Method": "Conditional Rules",
                "Description": "Use custom rules based on motifs and roles",
                "Example": "If triangle count > 10 and role 1 membership > 0.7 → 'Dense Hub'"
            }
        ]
        
        st.table(pd.DataFrame(labeling_methods))
        
        # Example of conditional rules
        st.markdown("#### Example Conditional Rules")
        
        # Create rules
        example_rules = analyzer.create_example_rule_set()
        
        # Display rules
        for i, rule in enumerate(example_rules):
            st.markdown(f"**Rule {i+1}: {rule['name']}**")
            st.markdown(f"*Condition*: {rule['condition'].__doc__ if rule['condition'].__doc__ else 'Custom condition'}")
            st.markdown(f"*Label*: {rule['label']}")
        
        # Create labels with different methods
        st.markdown("#### Generated Labels")
        
        # Select labeling method
        label_method = st.selectbox(
            "Select labeling method",
            ["primary_role", "mixed_roles", "role_community"]
        )
        
        # Generate labels
        labels = analyzer.create_structural_labels(method=label_method)
        
        # Display sample of labels
        labels_sample = {k: labels[k] for k in sorted(list(labels.keys()))[:20]}
        
        # Create dataframe for display
        labels_df = pd.DataFrame({"Node": list(labels_sample.keys()), 
                                 "Structural Label": list(labels_sample.values())})
        
        st.dataframe(labels_df)
        
        # Offer to download all labels
        all_labels_df = pd.DataFrame({"Node": list(labels.keys()), 
                                     "Structural Label": list(labels.values())})
        
        csv = all_labels_df.to_csv(index=False)
        st.download_button(
            label="Download All Structural Labels",
            data=csv,
            file_name="structural_labels.csv",
            mime="text/csv"
        )

def add_motif_role_analysis_page():
    """
    Add a dedicated page for more comprehensive motif and role analysis.
    """
    st.markdown('<div class="section-header">Motif and Role Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
        return
    
    if not st.session_state.current_graph and not st.session_state.graph_family:
        st.warning("Please generate a graph in the 'Graph Sampling' page or a graph family in the 'Graph Family Generation' page.")
        return
    
    st.markdown("""
    <div class="info-box">
    This page provides comprehensive analysis of network motifs and structural roles:
    <ul>
        <li>Motif participation profiles for nodes</li>
        <li>Structural role discovery and analysis</li>
        <li>Role-community relationships</li>
        <li>Role-based graph visualization</li>
    </ul>
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
    
    # Analysis parameters
    st.markdown('<div class="subsection-header">Analysis Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_motif_size = st.slider(
            "Maximum motif size",
            min_value=3,
            max_value=5,
            value=4,
            help="Largest motif pattern to analyze (higher values increase computation time)"
        )
        
        normalize_features = st.checkbox(
            "Normalize features",
            value=True,
            help="Normalize motif counts to emphasize structural patterns over size effects"
        )
    
    with col2:
        n_roles = st.slider(
            "Number of roles",
            min_value=2,
            max_value=8,
            value=5,
            help="Number of structural roles to discover"
        )
    
    # Run analysis button
    if st.button("Run Structural Analysis"):
        # Check if graph is too large for interactive analysis
        if graph.graph.number_of_nodes() > 1000:
            st.warning(f"Graph has {graph.graph.number_of_nodes()} nodes, which may be too large for interactive analysis. Consider using a smaller graph or running offline.")
            proceed = st.checkbox("Proceed anyway (may be slow)", value=False)
            if not proceed:
                return
        
        # Show progress and run analysis
        with st.spinner("Analyzing structural patterns and roles... This may take a while for larger graphs."):
            try:
                # Initialize analyzer
                analyzer = MotifRoleAnalyzer(
                    graph_sample=graph,
                    max_motif_size=max_motif_size,
                    n_roles=n_roles,
                    normalize_features=normalize_features,
                    verbose=True
                )
                
                # Run motif profile extraction
                st.text("Computing motif participation profiles...")
                analyzer.compute_motif_profiles()
                
                # Run role discovery
                st.text("Discovering structural roles...")
                analyzer.discover_structural_roles()
                
                # Store analyzer in session state
                st.session_state.role_analyzer = analyzer
                
                st.success("Structural analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
        
        # Display results
        if hasattr(st.session_state, 'role_analyzer'):
            display_motif_role_analysis_results(st.session_state.role_analyzer)