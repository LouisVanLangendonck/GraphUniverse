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
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.decomposition import PCA
from collections import Counter, defaultdict
import plotly.graph_objects as go
import io
import seaborn as sns
import scipy.stats as stats

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MMSB modules
from graph_universe.model import GraphUniverse, GraphSample
from graph_universe.graph_family import GraphFamilyGenerator, FamilyConsistencyAnalyzer
from graph_universe.feature_regimes import (
    SimplifiedFeatureGenerator,
    NeighborhoodFeatureAnalyzer, 
    FeatureClusterLabelGenerator
)
from utils.visualizations import (
    plot_graph_communities, 
    plot_membership_matrix,
    plot_community_matrix,
    plot_community_graph,
    plot_degree_distribution,
    plot_community_overlap_distribution,
    create_dashboard,
    create_dccc_sbm_dashboard,
    add_dccc_visualization_to_app,
    plot_community_size_distribution,
    visualize_community_cluster_assignments
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


def create_graph_dashboard(graph):
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Graph with Communities",
        "Community Distribution",
        "Parameter Analysis",
        "Feature Analysis",
        "Triangle Analysis"
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

    with tab3:
        st.subheader("Parameter Analysis")
        # Convert parameters dictionary to DataFrame
        params_dict = graph.extract_parameters()
        params_df = pd.DataFrame([params_dict])
        fig = create_parameter_dashboard(params_df)
        st.pyplot(fig)

    with tab4:
        st.subheader("Feature Analysis")
        if hasattr(graph, 'features') and graph.features is not None:
            # Pass a list containing the single graph sample
            fig = plot_community_statistics([graph])
            st.pyplot(fig)
            
            # Add feature regime analysis if available
            if hasattr(graph, 'node_regimes') and graph.node_regimes is not None:
                st.subheader("Feature Regime Analysis")
                regime_analysis = graph.analyze_neighborhood_features()
                st.write("Regime Distribution:", regime_analysis)
                
                # Add community-cluster visualization if feature generator is available
                if hasattr(graph, 'universe') and hasattr(graph.universe, 'feature_generator'):
                    st.subheader("Community-Cluster Assignments")
                    fig = visualize_community_cluster_assignments(graph.universe.feature_generator)
                    st.pyplot(fig)

    with tab5:
        st.subheader("Triangle Analysis")
        # Get triangle statistics
        triangle_stats = graph.analyze_triangles()
        
        # Display total triangles
        st.metric("Total Triangles", triangle_stats['total_triangles'])
        
        # Display additional triangles if any were added
        if triangle_stats['total_additional_triangles'] > 0:
            st.metric("Additional Triangles", triangle_stats['total_additional_triangles'])
        
        # Create a DataFrame for triangles per community
        triangles_df = pd.DataFrame({
            'Community': list(triangle_stats['triangles_per_community'].keys()),
            'Total Triangles': list(triangle_stats['triangles_per_community'].values()),
            'Additional Triangles': list(triangle_stats['additional_triangles_per_community'].values())
        })
        
        # Display triangles per community
        st.markdown("##### Triangles per Community")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create stacked bar plot
        x = np.arange(len(triangles_df))
        width = 0.35
        
        # Plot natural triangles (total - additional)
        natural_triangles = triangles_df['Total Triangles'] - triangles_df['Additional Triangles']
        ax.bar(x, natural_triangles, width, label='Natural Triangles', color='lightgreen')
        
        # Plot additional triangles on top
        ax.bar(x, triangles_df['Additional Triangles'], width, 
               bottom=natural_triangles, label='Additional Triangles', color='lightcoral')
        
        ax.set_title("Number of Triangles per Community")
        ax.set_xlabel("Community ID")
        ax.set_ylabel("Number of Triangles")
        ax.set_xticks(x)
        ax.set_xticklabels(triangles_df['Community'])
        ax.legend()
        st.pyplot(fig)
        
        # Display correlation with propensities
        st.markdown("##### Correlation with Triangle Propensities")
        st.metric("Correlation Coefficient", f"{triangle_stats['triangle_propensity_correlation']:.3f}")
        
        # Display triangle propensity comparison plot
        st.markdown("##### Triangle Propensities vs Actual Counts")
        st.pyplot(triangle_stats['triangle_propensity_plot'])
        
        # Add explanation
        st.markdown("""
        This analysis shows:
        1. The total number of triangles in the graph
        2. The distribution of triangles across communities
        3. The correlation between the number of triangles per community and their triangle propensities
        4. A direct comparison of triangle propensities vs actual triangle counts per community
        
        A high correlation indicates that communities with higher triangle propensities tend to form more triangles.
        The comparison plot helps visualize how well the actual triangle counts match the expected propensities.
        """)

# def run_metapath_analysis(graph, theta, max_length, allow_loops, allow_backtracking, top_k, min_length):
#     """Callback function to run metapath analysis"""
#     try:
#         # Update session state with new values
#         st.session_state.metapath_analysis_state['params'].update({
#             'theta': theta,
#             'max_length': max_length,
#             'allow_loops': allow_loops,
#             'allow_backtracking': allow_backtracking,
#             'top_k': top_k,
#             'min_length': min_length
#         })
        
#         # Run the metapath analysis
#         metapath_results = analyze_metapaths(
#             graph,
#             theta=theta,
#             max_length=max_length,
#             top_k=top_k,
#             allow_loops=allow_loops,
#             allow_backtracking=allow_backtracking,
#             min_length=min_length
#         )
        
#         # Store results in session state
#         st.session_state.metapath_analysis_state['results'] = metapath_results
#         st.session_state.metapath_analysis_state['analysis_run'] = True
        
#         return True
#     except Exception as e:
#         return False

# def render_metapath_analysis(graph):
#     """Render the metapath analysis section in an isolated container."""
#     container = st.container()
    
#     with container:
#         st.markdown("#### Metapath Analysis")
        
#         # Parameters outside form
#         col1, col2 = st.columns(2)
        
#         with col1:
#             theta = st.slider(
#                 "Probability threshold (θ)",
#                 min_value=0.01,
#                 max_value=0.5,
#                 value=st.session_state.metapath_analysis_state['params']['theta'],
#                 help="Threshold for considering an edge likely in the community graph"
#             )
#             max_length = st.slider(
#                 "Maximum metapath length",
#                 min_value=2,
#                 max_value=5,
#                 value=st.session_state.metapath_analysis_state['params']['max_length'],
#                 help="Maximum number of communities in a metapath"
#             )
        
#         with col2:
#             allow_loops = st.checkbox(
#                 "Allow loops",
#                 value=st.session_state.metapath_analysis_state['params']['allow_loops'],
#                 help="Allow metapaths to visit the same community multiple times"
#             )
#             allow_backtracking = st.checkbox(
#                 "Allow backtracking",
#                 value=st.session_state.metapath_analysis_state['params']['allow_backtracking'],
#                 help="Allow metapaths to return to the previous community"
#             )
#             top_k = st.slider(
#                 "Number of top metapaths",
#                 min_value=1,
#                 max_value=10,
#                 value=st.session_state.metapath_analysis_state['params']['top_k'],
#                 help="Number of top metapaths to analyze in detail"
#             )
        
#         # Run analysis button
#         if st.button("Run Metapath Analysis", key="run_metapath"):
#             with st.spinner("Analyzing metapaths..."):
#                 success = run_metapath_analysis(graph, theta, max_length, allow_loops, allow_backtracking, top_k, st.session_state.metapath_analysis_state['params']['min_length'])
#                 if success:
#                     st.success("Metapath analysis completed successfully!")
#                 else:
#                     st.error("Error during metapath analysis. Check console for details.")
        
#         # Display results if analysis has been run
#         if st.session_state.metapath_analysis_state['analysis_run'] and st.session_state.metapath_analysis_state['results'] is not None:
#             results = st.session_state.metapath_analysis_state['results']
            
#             # Create tabs for different visualizations
#             tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#                 "Community Structure",
#                 "Metapath Statistics",
#                 "Node Classification",
#                 "Multi-Metapath Node Classification",
#                 "K-hop Metapath Detection",  # New tab
#                 "Help"
#             ])

#             with tab1:
#                 # Show P_sub matrix visualization
#                 st.markdown("##### Community Edge Probability Matrix (P_sub)")
#                 st.markdown("""
#                 This heatmap shows the probability of edges between communities. 
#                 - Brighter colors indicate higher probabilities
#                 - The matrix shows how likely nodes from one community are to connect to nodes in another community
#                 - This helps understand the underlying community structure that gives rise to metapaths
#                 """)
#                 st.pyplot(results['P_sub_figure'])
                
#                 # Show community-level graph
#                 st.markdown("##### Community-Level Metapath Graph")
#                 st.markdown("""
#                 This graph shows the community-level structure with edges representing likely metapaths.
#                 - Nodes represent communities
#                 - Edge weights show the probability of connections between communities
#                 - This helps visualize the paths that are most likely to occur
#                 """)
#                 fig = visualize_community_metapath_graph(
#                     results['P_matrix'],
#                     theta=st.session_state.metapath_analysis_state['params']['theta'],
#                     community_mapping=results['community_mapping']
#                 )
#                 st.pyplot(fig)
            
#             with tab2:
#                 st.markdown("##### Metapath Statistics")
#                 stats = results['statistics']
                
#                 # Convert to DataFrame for display
#                 stats_df = pd.DataFrame({
#                     'Metapath': stats['metapaths'],
#                     'Instances': stats['instances_count'],
#                     'Avg Path Length': [f"{x:.2f}" for x in stats['avg_path_length']],
#                     'Node Participation (%)': [f"{x*100:.1f}%" for x in stats['participation']]
#                 })
                
#                 st.markdown("""
#                 **How metapaths are ranked:**
#                 Metapaths are ranked based on their probability in the community graph, not just the number of instances.
#                 The probability is calculated as the product of edge probabilities along the path.
                
#                 For example, a path "0 → 1 → 2" has probability = P(0,1) × P(1,2)
#                 where P(i,j) is the probability of an edge between communities i and j.
                
#                 This means that even if a metapath has fewer instances, it might be ranked higher if:
#                 - The edges between its communities have high probabilities
#                 - The path is more likely to occur in the underlying community structure
#                 """)
                
#                 st.dataframe(stats_df)
                
#                 # Select a metapath to visualize
#                 selected_metapath_idx = st.selectbox(
#                     "Select a metapath to visualize",
#                     range(min(st.session_state.metapath_analysis_state['params']['top_k'], len(results['metapaths']))),
#                     format_func=lambda i: stats['metapaths'][i],
#                     key="metapath_selection"
#                 )
                
#                 # Show selected metapath
#                 selected_metapath = results['metapaths'][selected_metapath_idx]
#                 selected_instances = results['instances'][selected_metapath_idx]
                
#                 if selected_instances:
#                     st.markdown("##### Full Graph with Metapath Instances")
#                     st.markdown("""
#                     This visualization shows the full graph with metapath instances highlighted.
#                     - Nodes are colored by their community
#                     - Regular edges are shown in light gray
#                     - Metapath instances are highlighted with bold, colored edges
#                     """)
#                     fig = visualize_metapaths(
#                         st.session_state.current_graph.graph,
#                         st.session_state.current_graph.community_labels,
#                         selected_metapath,
#                         selected_instances,
#                         title=f"Metapath: {stats['metapaths'][selected_metapath_idx]}"
#                     )
#                     st.pyplot(fig)
#                 else:
#                     st.info("No instances found for this metapath in the graph.")
            
#             with tab3:
#                 st.markdown("##### Node Classification")
#                 st.markdown("""
#                 This tab performs binary classification to predict whether a node participates in the selected metapath.
#                 We use multiple models to compare their performance:
#                 - Random Forest (RF): Traditional ML model using node features
#                 - Multi-Layer Perceptron (MLP): Neural network using node features
#                 - Graph Convolutional Network (GCN): Graph neural network using node features and graph structure
#                 - GraphSAGE: Graph neural network using node features and graph structure with sampling
#                 """)
#                 if selected_metapath:
#                     st.markdown("#### Hyperparameter Optimization (Optuna)")
#                     st.markdown("**Baseline Feature Selection (RF/MLP):**")
#                     col_feat1, col_feat2, col_feat3 = st.columns(3)
#                     with col_feat1:
#                         use_degree = st.checkbox("Use Degree", value=True, key="optuna_baseline_use_degree")
#                     with col_feat2:
#                         use_clustering = st.checkbox("Use Clustering Coefficient", value=True, key="optuna_baseline_use_clustering")
#                     with col_feat3:
#                         use_node_features = st.checkbox("Use Node Features", value=True, key="optuna_baseline_use_node_features")
#                     optuna_baseline_feature_opts = {
#                         'use_degree': use_degree,
#                         'use_clustering': use_clustering,
#                         'use_node_features': use_node_features
#                     }
#                     model_type = st.selectbox("Model type for optimization", ["rf", "mlp", "gcn", "sage"])
#                     n_trials = st.slider("Number of Optuna trials", min_value=5, max_value=100, value=20)
#                     timeout = st.slider("Timeout (seconds)", min_value=30, max_value=1800, value=300)
#                     if st.button("Run Hyperparameter Optimization"):
#                         with st.spinner("Optimizing hyperparameters with Optuna..."):
#                             try:
#                                 X = prepare_node_features(
#                                     st.session_state.current_graph.graph,
#                                     st.session_state.current_graph.community_labels,
#                                     use_degree=optuna_baseline_feature_opts['use_degree'],
#                                     use_clustering=optuna_baseline_feature_opts['use_clustering'],
#                                     use_node_features=optuna_baseline_feature_opts['use_node_features']
#                                 )
#                             except ValueError as e:
#                                 st.error(str(e))
#                                 st.stop()
            
#             with tab4:
#                 st.markdown("#### Multi-Metapath Node Classification")
#                 st.markdown("""
#                 This tab performs multi-label classification to predict whether a node participates in multiple metapaths.
#                 For each node, a binary vector label is created: 1 if the node participates in the metapath, 0 otherwise.
#                 The classifier predicts this vector for each node (multi-label classification).
#                 """)
#                 metapath_options = results['metapaths'][:st.session_state.metapath_analysis_state['params']['top_k']]
#                 metapath_labels = [f"{i}: {stats['metapaths'][i]}" for i in range(len(metapath_options))]
#                 selected_indices = st.multiselect(
#                     "Select metapaths for multi-label classification",
#                     options=list(range(len(metapath_options))),
#                     format_func=lambda i: metapath_labels[i],
#                     default=list(range(min(2, len(metapath_options))))
#                 )
                
#                 if not selected_indices:
#                     st.warning("Please select at least one metapath")
#                 else:
#                     selected_metapaths = [metapath_options[i] for i in selected_indices]
                    
#                     # Add a button to create splits first
#                     st.markdown("### Create Train/Val/Test Splits")
#                     st.markdown("""
#                     First, create consistent train/validation/test splits that will be used for all models.
#                     This ensures fair comparison between different model types.
#                     """)
                    
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         train_size = st.slider("Train size", 0.5, 0.9, 0.7, 0.05, key="ml_train_size")
#                     with col2:
#                         val_size = st.slider("Validation size", 0.05, 0.25, 0.15, 0.05, key="ml_val_size")
#                     with col3:
#                         test_size = st.slider("Test size", 0.05, 0.25, 0.15, 0.05, key="ml_test_size")
                    
#                     total = train_size + val_size + test_size
#                     if abs(total - 1.0) > 1e-6:
#                         st.warning(f"Split sizes sum to {total:.2f}, should be 1.0. Please adjust.")
                    
#                     if st.button("Create Splits", key="create_splits"):
#                         with st.spinner("Creating train/val/test splits..."):
#                             try:
#                                 # Prepare features
#                                 feature_opts = {
#                                     'use_degree': True,
#                                     'use_clustering': True,
#                                     'use_node_features': True
#                                 }
                                
#                                 X = prepare_node_features(
#                                     st.session_state.current_graph,
#                                     st.session_state.current_graph.community_labels,
#                                     use_degree=feature_opts['use_degree'],
#                                     use_clustering=feature_opts['use_clustering'],
#                                     use_node_features=feature_opts['use_node_features']
#                                 )
                                
#                                 # Prepare multi-label targets
#                                 n_nodes = X.shape[0]
#                                 Y = np.zeros((n_nodes, len(selected_metapaths)), dtype=int)
#                                 graph_nx = st.session_state.current_graph.graph
                                
#                                 for j, metapath in enumerate(selected_metapaths):
#                                     instances = find_metapath_instances(graph_nx, st.session_state.current_graph.community_labels, metapath)
#                                     participating_nodes = set()
#                                     for instance in instances:
#                                         participating_nodes.update(instance)
#                                     for i in range(n_nodes):
#                                         if i in participating_nodes:
#                                             Y[i, j] = 1
                                
#                                 # Create splits
#                                 splits = create_consistent_train_val_test_split(
#                                     X, Y, 
#                                     train_size=train_size, 
#                                     val_size=val_size, 
#                                     test_size=test_size,
#                                     stratify=False,  # Can't easily stratify multi-label
#                                     seed=42
#                                 )
                                
#                                 # Store in session state
#                                 if 'metapath_multi_label_state' not in st.session_state.metapath_analysis_state:
#                                     st.session_state.metapath_analysis_state['metapath_multi_label_state'] = {}
                                    
#                                 st.session_state.metapath_analysis_state['metapath_multi_label_state'] = {
#                                     'splits_created': True,
#                                     'splits': splits,
#                                     'X': X,
#                                     'Y': Y,
#                                     'feature_opts': feature_opts,
#                                     'selected_metapaths': selected_metapaths
#                                 }
                                
#                                 st.success("Splits created successfully!")
                                
#                                 # Display split statistics
#                                 st.markdown("#### Split Statistics")
#                                 train_size = len(splits['train_indices']) / n_nodes
#                                 val_size = len(splits['val_indices']) / n_nodes
#                                 test_size = len(splits['test_indices']) / n_nodes
                                
#                                 col1, col2, col3 = st.columns(3)
#                                 col1.metric("Train Split", f"{train_size:.1%}")
#                                 col2.metric("Validation Split", f"{val_size:.1%}")
#                                 col3.metric("Test Split", f"{test_size:.1%}")
                                
#                                 # Display label distribution in splits
#                                 st.markdown("#### Label Distribution in Splits")
#                                 train_label_dist = Y[splits['train_indices']].mean(axis=0)
#                                 val_label_dist = Y[splits['val_indices']].mean(axis=0)
#                                 test_label_dist = Y[splits['test_indices']].mean(axis=0)
                                
#                                 dist_df = pd.DataFrame({
#                                     'Metapath': [metapath_labels[i] for i in selected_indices],
#                                     'Train': [f"{x:.1%}" for x in train_label_dist],
#                                     'Validation': [f"{x:.1%}" for x in val_label_dist],
#                                     'Test': [f"{x:.1%}" for x in test_label_dist]
#                                 })
                                
#                                 st.dataframe(dist_df)
                                
#                             except Exception as e:
#                                 st.error(f"Error creating splits: {str(e)}")
#                                 import traceback
#                                 st.code(traceback.format_exc())
                    
#                     # Rest of multi-metapath classification code remains the same
#                     # (Code omitted for brevity)
            
#             # NEW TAB: K-hop Metapath Detection
#             with tab5:
#                 st.markdown("#### K-hop Metapath Detection")
#                 st.markdown("""
#                 This analysis detects k-hop relationships along a selected metapath and
#                 labels starting nodes based on the feature regime of their k-hop neighbors.
                
#                 **How it works:**
#                 1. Select a metapath and k (hop distance)
#                 2. The analysis identifies instances of the metapath in the graph
#                 3. For each instance, it finds nodes that are k-hops away from starting nodes
#                 4. Starting nodes are labeled based on the feature regime of their k-hop neighbors
#                 """)
                
#                 # Select a metapath to analyze
#                 metapath_options = results['metapaths'][:st.session_state.metapath_analysis_state['params']['top_k']]
#                 stats = results['statistics']
#                 metapath_labels = [f"{i}: {stats['metapaths'][i]}" for i in range(len(metapath_options))]
                
#                 selected_metapath_idx = st.selectbox(
#                     "Select a metapath for k-hop analysis",
#                     range(min(st.session_state.metapath_analysis_state['params']['top_k'], len(results['metapaths']))),
#                     format_func=lambda i: metapath_labels[i],
#                     key="khop_metapath_selection"
#                 )
                
#                 selected_metapath = results['metapaths'][selected_metapath_idx]
                
#                 # Select k (hop distance)
#                 max_k = len(selected_metapath) - 1
#                 k = st.slider("Select k (hop distance)", 1, max(1, max_k), 1, key="khop_k_selection")
                
#                 # Run analysis button
#                 if st.button("Run K-hop Metapath Detection", key="run_khop_analysis"):
#                     with st.spinner("Analyzing k-hop metapath relationships..."):
#                         # Check if graph has node_regimes
#                         if not hasattr(st.session_state.current_graph, 'node_regimes') or st.session_state.current_graph.node_regimes is None:
#                             st.error("Graph does not have feature regime information. Please generate a graph with features.")
#                         else:
#                             try:
#                                 # Run k-hop metapath detection
#                                 khop_result = khop_metapath_detection(
#                                     st.session_state.current_graph.graph,
#                                     st.session_state.current_graph.community_labels,
#                                     st.session_state.current_graph.node_regimes,
#                                     selected_metapath,
#                                     k
#                                 )
#                                 # Store results in session state
#                                 st.session_state.metapath_analysis_state['khop_detection_result'] = {
#                                     'result': khop_result,
#                                     'metapath': selected_metapath,
#                                     'k': k,
#                                     'analysis_run': True
#                                 }
#                                 st.success("K-hop metapath detection completed successfully!")
#                             except Exception as e:
#                                 st.error(f"Error during k-hop metapath detection: {str(e)}")
#                 # Display results if analysis has been run
#                 if ('khop_detection_result' in st.session_state.metapath_analysis_state and 
#                     st.session_state.metapath_analysis_state['khop_detection_result'].get('analysis_run', False)):
#                     khop_data = st.session_state.metapath_analysis_state['khop_detection_result']
#                     khop_result = khop_data['result']
#                     # Create columns for better layout
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         # Display basic statistics
#                         st.markdown("##### K-hop Detection Results")
#                         st.markdown(f"Path: {' → '.join([str(c) for c in khop_result['path_community_sequence']])}")
#                         st.markdown(f"Starting Community: {khop_result['starting_community']}")
#                         st.markdown(f"Target Community (k-hop): {khop_result['target_community']}")
#                         st.markdown(f"Total Relationships: {khop_result['total_relationships']}")
#                         # Add interactive filters
#                         st.markdown("##### Filter Results")
#                         min_relationships = st.slider(
#                             "Minimum relationships per regime",
#                             min_value=1,
#                             max_value=max(khop_result['regime_counts'].values()) if khop_result['regime_counts'] else 1,
#                             value=1,
#                             key="min_relationships"
#                         )
#                         # Filter regimes based on minimum relationships
#                         filtered_regimes = {
#                             regime: count for regime, count in khop_result['regime_counts'].items()
#                             if count >= min_relationships
#                         }
#                     with col2:
#                         # Display regime distribution
#                         st.markdown("##### Feature Regime Distribution")
#                         if filtered_regimes:
#                             # Create bar chart of regime distribution
#                             regimes = list(filtered_regimes.keys())
#                             counts = [filtered_regimes[r] for r in regimes]
#                             fig, ax = plt.subplots(figsize=(8, 4))
#                             bars = ax.bar(regimes, counts)
#                             # Label each bar with its value
#                             for bar in bars:
#                                 height = bar.get_height()
#                                 ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                                         f'{int(height)}', ha='center', va='bottom')
#                             ax.set_xlabel('Feature Regime')
#                             ax.set_ylabel('Count')
#                             ax.set_title('Distribution of Feature Regimes in K-hop Neighbors')
#                             ax.set_xticks(regimes)
#                             st.pyplot(fig)
#                             # Add download button for regime distribution data
#                             regime_df = pd.DataFrame({
#                                 'Regime': regimes,
#                                 'Count': counts
#                             })
#                             st.download_button(
#                                 "Download Regime Distribution",
#                                 regime_df.to_csv(index=False),
#                                 "regime_distribution.csv",
#                                 "text/csv",
#                                 key="download_regime_dist"
#                             )
#                         else:
#                             st.info("No regimes meet the minimum relationship threshold.")
#                     # Visualize relationships in the graph
#                     st.markdown("##### K-hop Relationships Visualization")
#                     if khop_result['total_relationships'] > 0:
#                         # Add visualization options
#                         viz_col1, viz_col2 = st.columns(2)
#                         with viz_col1:
#                             show_all_nodes = st.checkbox("Show all nodes", value=True)
#                             highlight_starting = st.checkbox("Highlight starting nodes", value=True)
#                         with viz_col2:
#                             node_size = st.slider("Node size", 30, 200, 80)
#                             edge_width = st.slider("Edge width", 1, 5, 2)
#                         fig = visualize_khop_metapath_detection(
#                             st.session_state.current_graph.graph,
#                             st.session_state.current_graph.community_labels,
#                             st.session_state.current_graph.node_regimes,
#                             khop_data['metapath'],
#                             khop_data['k'],
#                             khop_result,
#                             title=f"K-hop Metapath Detection (k={khop_data['k']})",
#                             show_all_nodes=show_all_nodes,
#                             highlight_starting=highlight_starting,
#                             node_size=node_size,
#                             edge_width=edge_width
#                         )
#                         st.pyplot(fig)
#                         # Add download button for visualization
#                         buf = io.BytesIO()
#                         fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
#                         st.download_button(
#                             "Download Visualization",
#                             buf.getvalue(),
#                             "khop_visualization.png",
#                             "image/png",
#                             key="download_viz"
#                         )
            
#             with tab6:
#                 st.markdown("""
#                 ### Understanding Metapath Analysis and K-hop Detection
                
#                 **What are Metapaths?**
#                 Metapaths are sequences of communities that frequently appear in the graph. For example, a metapath "0 → 1 → 2" means that nodes from community 0 often connect to nodes in community 1, which then connect to nodes in community 2.
                
#                 **The Analysis Process:**
#                 1. **Community Structure** tab shows:
#                    - P_sub matrix: Probability of edges between communities
#                    - Community graph: Visual representation of likely connections
                
#                 2. **Metapath Statistics** tab shows:
#                    - List of most common metapaths
#                    - Number of instances of each metapath
#                    - Average path length and node participation
                
#                 3. **Node Classification** tab:
#                    - Predicts which nodes participate in a selected metapath
#                    - Uses node features to make predictions
#                    - Shows how well the prediction works
                   
#                 4. **Multi-Metapath Node Classification** tab:
#                    - Predicts which nodes participate in multiple metapaths
#                    - Creates multi-label classifiers
#                    - Compares performance of different models
                
#                 5. **K-hop Metapath Detection** tab:
#                    - **Core Feature**: Labels nodes in a starting community based on their k-hop neighbors' feature regimes
#                    - **How it works**:
#                      1. Select a metapath (e.g., "0 → 1 → 2")
#                      2. Choose k (hop distance) to analyze
#                      3. For each node in the starting community:
#                         - Find all k-hop neighbors along the metapath
#                         - Analyze the feature regimes of these neighbors
#                         - Label the starting node based on the distribution of regimes
#                    - **Visualization**:
#                      - Shows the full graph with highlighted k-hop relationships
#                      - Nodes are colored by their community
#                      - K-hop relationships are highlighted with bold edges
#                      - Feature regime distribution is shown in a bar chart
#                    - **Use Cases**:
#                      - Understanding feature propagation along metapaths
#                      - Identifying nodes with similar k-hop neighborhood patterns
#                      - Analyzing how features spread through the network
                
#                 **How to Use K-hop Detection:**
#                 1. First, examine the metapath statistics to find interesting patterns
#                 2. Select a metapath that you want to analyze
#                 3. Choose an appropriate k value (hop distance)
#                 4. Run the analysis to see:
#                    - Distribution of feature regimes in k-hop neighbors
#                    - Visual representation of k-hop relationships
#                    - Statistics about the relationships found
#                 5. Use the results to understand how features propagate along the metapath
                
#                 **Tips for Effective Analysis:**
#                 - Start with smaller k values (1-2) to understand direct relationships
#                 - Look for patterns in the feature regime distribution
#                 - Compare results across different metapaths
#                 - Use the visualization to identify clusters of similar nodes
#                 """)

# Initialization for session state
if 'metapath_analysis_state' not in st.session_state:
    st.session_state.metapath_analysis_state = {
        'params': {
            'theta': 0.1,
            'max_length': 3,
            'allow_loops': False,
            'allow_backtracking': False,
            'top_k': 5,
            'min_length': 2
        },
        'results': None,
        'analysis_run': False,
        'classification_state': {
            'run': False,
            'results': None
        },
        'metapath_multi_label_state': {
            'splits_created': False,
            'splits': None,
            'X': None,
            'Y': None,
            'feature_opts': None,
            'selected_metapaths': None
        },
        'khop_detection_result': {  # New state for k-hop detection
            'result': None,
            'metapath': None,
            'k': 1,
            'analysis_run': False
        },
        'khop_classification_result': {
            'results': None,
            'metapath': None,
            'k': 1,
            'model_type': 'rf',
            'analysis_run': False
        },
        'khop_optuna_results': None,
    }

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
    ["Universe Creation", "Graph Sampling", "Graph Family Generation", "Graph Family Analysis", "Metapath Analysis"]
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
        <li>Degree center ordering for communities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Universe parameters
    st.markdown('<div class="subsection-header">Universe Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.slider("Number of communities", min_value=3, max_value=50, value=10)
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
            help="Controls the amount of Gaussian noise added to the universe's strength matrix"
        )
        
        # Add new parameters
        community_density_variation = st.slider(
            "Community density variation",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Amount of density variation between communities"
        )
        
        community_cooccurrence_homogeneity = st.slider(
            "Community co-occurrence homogeneity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="How homogeneous the co-occurrence of communities is"
        )
    
    # Add triangle parameters
    st.markdown('<div class="subsection-header">Triangle Parameters</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        triangle_density = st.slider(
            "Triangle density",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Target density of triangles in the graph"
        )
    
    with col4:
        triangle_community_relation_homogeneity = st.slider(
            "Triangle community relation homogeneity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="How homogeneous the triangle formation is across communities"
        )
    
    # Add degree center method selection
    st.markdown('<div class="subsection-header">Degree Center Configuration</div>', unsafe_allow_html=True)
    degree_center_method = st.selectbox(
        "Degree center method",
        options=["linear", "random", "shuffled"],
        index=0,
        help="How to generate degree centers for communities: linear (ordered), random, or shuffled linear"
    )
    
    # Feature parameters if features are enabled
    if feature_dim > 0:
        st.markdown('<div class="subsection-header">Feature Generation Parameters</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            cluster_count_factor = st.slider(
                "Cluster count factor",
                min_value=0.1,
                max_value=4.0,
                value=1.0,
                step=0.1,
                help="Number of clusters relative to communities"
            )
            center_variance = st.slider(
                "Center variance",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Separation between cluster centers"
            )
            
        with col4:
            cluster_variance = st.slider(
                "Cluster variance",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Spread within each cluster"
            )
            assignment_skewness = st.slider(
                "Assignment skewness",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="If some clusters are used more frequently"
            )
            
        community_exclusivity = st.slider(
            "Community exclusivity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="How exclusively clusters map to communities"
        )
    
    # Add seed parameter
    seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")
    
    if st.button("Generate Universe"):
        with st.spinner("Generating universe..."):
            # Create universe with all parameters
            universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                edge_density=edge_density,
                homophily=homophily,
                randomness_factor=randomness_factor,
                # Feature generation parameters
                cluster_count_factor=cluster_count_factor if feature_dim > 0 else 1.0,
                center_variance=center_variance if feature_dim > 0 else 1.0,
                cluster_variance=cluster_variance if feature_dim > 0 else 0.1,
                assignment_skewness=assignment_skewness if feature_dim > 0 else 0.0,
                community_exclusivity=community_exclusivity if feature_dim > 0 else 1.0,
                degree_center_method=degree_center_method,  # Add degree center method
                community_density_variation=community_density_variation,
                community_cooccurrence_homogeneity=community_cooccurrence_homogeneity,
                triangle_density=triangle_density,
                triangle_community_relation_homogeneity=triangle_community_relation_homogeneity,
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
            
            # Plot community co-occurrence matrix
            st.markdown('<div class="subsection-header">Community Co-occurrence Matrix</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(universe.community_cooccurrence_matrix, 
                       cmap='viridis', 
                       annot=True, 
                       fmt='.2f',
                       ax=ax)
            ax.set_title('Community Co-occurrence Matrix')
            ax.set_xlabel('Community')
            ax.set_ylabel('Community')
            st.pyplot(fig)
            
            # Plot triangle propensities
            st.markdown('<div class="subsection-header">Triangle Propensities</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            propensities = universe.community_triangle_propensities
            communities = range(len(propensities))
            
            # Sort by propensity for better visualization
            sorted_indices = np.argsort(propensities)
            sorted_propensities = propensities[sorted_indices]
            sorted_communities = np.array(communities)[sorted_indices]
            
            ax.bar(range(len(sorted_communities)), sorted_propensities, color='skyblue')
            ax.set_xlabel('Community ID')
            ax.set_ylabel('Triangle Propensity')
            ax.set_title('Community Triangle Propensities')
            ax.set_xticks(range(len(sorted_communities)))
            ax.set_xticklabels(sorted_communities)
            
            # Add mean propensity line
            mean_propensity = np.mean(propensities)
            ax.axhline(y=mean_propensity, color='red', linestyle='--', 
                      label=f'Mean: {mean_propensity:.3f}')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add explanation
            st.markdown("""
            The triangle propensities show how likely each community is to form triangles:
            - Higher values indicate communities that are more likely to form triangles
            - The red dashed line shows the mean propensity across all communities
            - Communities are sorted by propensity for better visualization
            """)
            
            # Plot degree centers
            st.markdown('<div class="subsection-header">Degree Centers</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(K), universe.degree_centers, 'o-', label='Degree Centers')
            ax.set_xlabel('Community Index')
            ax.set_ylabel('Degree Center Value')
            ax.set_title(f'Degree Centers ({degree_center_method})')
            ax.grid(True)
            st.pyplot(fig)
            
            # Show feature analysis if enabled
            if feature_dim > 0:
                st.markdown('<div class="subsection-header">Feature Analysis</div>', unsafe_allow_html=True)
                
                # Plot feature clusters
                fig = plt.figure(figsize=(10, 6))
                plt.imshow(universe.feature_generator.cluster_centers, aspect='auto', cmap='viridis')
                plt.colorbar(label='Feature Value')
                plt.title('Feature Cluster Centers')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Cluster')
                st.pyplot(fig)
                
                # Show community-cluster assignments
                st.subheader("Community-Cluster Assignments")
                fig = visualize_community_cluster_assignments(universe.feature_generator)
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
        
        # Initialize session state for graph parameters if not exists
        if 'graph_params' not in st.session_state:
            st.session_state.graph_params = {
                'n_nodes': 80,
                'num_communities': 5,
                'min_component_size': 3,
                'sampling_method': "random",
                'max_mean_community_deviation': 0.1,
                'max_max_community_deviation': 0.2,
                'parameter_search_range': 0.2,
                'max_parameter_search_attempts': 10,
                'min_edge_density': 0.005,
                'max_retries': 5,
                'method': "Standard",
                'method_params': {
                    'degree_heterogeneity': 0.5,
                    'edge_noise': 0.0
                },
                'seed': 42
            }
        
        # Universal parameters
        st.markdown('<div class="subsection-header">Universal Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.graph_params['n_nodes'] = st.number_input(
                "Number of nodes",
                min_value=10,
                max_value=1000,
                value=st.session_state.graph_params['n_nodes'],
                help="Total number of nodes in the graph"
            )
            st.session_state.graph_params['num_communities'] = st.number_input(
                "Number of communities",
                min_value=2,
                max_value=10,
                value=st.session_state.graph_params['num_communities'],
                help="Number of communities to sample from the universe"
            )
            st.session_state.graph_params['min_component_size'] = st.number_input(
                "Minimum component size",
                min_value=1,
                max_value=10,
                value=st.session_state.graph_params['min_component_size'],
                help="Minimum size for a connected component to be kept"
            )
        
        with col2:
            st.session_state.graph_params['sampling_method'] = st.selectbox(
                "Community sampling method",
                options=["connected", "random"],
                index=["connected", "random"].index(st.session_state.graph_params['sampling_method']),
                help="Method for sampling communities from the universe"
            )
            st.session_state.graph_params['max_mean_community_deviation'] = st.slider(
                "Max mean community deviation",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.graph_params['max_mean_community_deviation'],
                step=0.01,
                help="Maximum allowed deviation in mean community properties"
            )
            st.session_state.graph_params['max_max_community_deviation'] = st.slider(
                "Max max community deviation",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.graph_params['max_max_community_deviation'],
                step=0.01,
                help="Maximum allowed deviation in any community property"
            )
            st.session_state.graph_params['disable_deviation_limiting'] = st.checkbox(
                "Disable deviation limiting",
                value=st.session_state.graph_params.get('disable_deviation_limiting', False),
                help="If checked, community deviation limits will not be enforced during graph generation"
            )
        
        # Method-specific parameters
        st.markdown('<div class="subsection-header">Method Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.graph_params['method'] = st.selectbox(
                "Graph generation method",
                options=["Standard", "DCCC-SBM", "Power Law", "Exponential", "Uniform"],
                index=["Standard", "DCCC-SBM", "Power Law", "Exponential", "Uniform"].index(st.session_state.graph_params.get('method', "Standard")),
                help="Method for generating the graph structure"
            )
            
            # DCCC-SBM specific parameters
            if st.session_state.graph_params['method'] == "DCCC-SBM":
                st.session_state.graph_params['method_params']['community_imbalance'] = st.slider(
                    "Community imbalance",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.graph_params['method_params'].get('community_imbalance', 0.3),
                    step=0.1,
                    help="Controls how imbalanced community sizes are (0=balanced, 1=maximally imbalanced)"
                )
                
                st.session_state.graph_params['method_params']['degree_separation'] = st.slider(
                    "Degree separation",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.graph_params['method_params'].get('degree_separation', 0.5),
                    step=0.1,
                    help="Controls how much degree distributions are separated between communities (0=overlapping, 1=well separated)"
                )
      
                degree_dist_type = st.selectbox(
                    "Degree distribution type",
                    options=["power_law", "exponential", "uniform"],
                    index=["power_law", "exponential", "uniform"].index(
                        st.session_state.graph_params['method_params'].get('degree_distribution_type', "power_law")
                    ),
                    help="Type of global degree distribution to partition among communities"
                )
                st.session_state.graph_params['method_params']['degree_distribution_type'] = degree_dist_type
                
                # Additional parameters based on distribution type
                if degree_dist_type == "power_law":
                    st.session_state.graph_params['method_params']['power_law_exponent'] = st.slider(
                        "Power law exponent",
                        min_value=2.0,
                        max_value=3.0,
                        value=st.session_state.graph_params['method_params'].get('power_law_exponent', 2.5),
                        step=0.1,
                        help="Exponent for power law degree distribution (lower values = more heavy-tailed)"
                    )
                elif degree_dist_type == "exponential":
                    st.session_state.graph_params['method_params']['rate'] = st.slider(
                        "Exponential rate",
                        min_value=0.1,
                        max_value=2.0,
                        value=st.session_state.graph_params['method_params'].get('rate', 0.5),
                        step=0.1,
                        help="Rate parameter for exponential degree distribution"
                    )
                elif degree_dist_type == "uniform":
                    st.session_state.graph_params['method_params']['min_factor'] = st.slider(
                        "Minimum degree factor",
                        min_value=0.1,
                        max_value=1.0,
                        value=st.session_state.graph_params['method_params'].get('min_factor', 0.5),
                        step=0.1,
                        help="Minimum factor for uniform degree distribution"
                    )
                    st.session_state.graph_params['method_params']['max_factor'] = st.slider(
                        "Maximum degree factor",
                        min_value=1.0,
                        max_value=2.0,
                        value=st.session_state.graph_params['method_params'].get('max_factor', 1.5),
                        step=0.1,
                        help="Maximum factor for uniform degree distribution"
                    )
                    
            # Original parameters for other methods
            elif st.session_state.graph_params['method'] == "Power Law":
                st.session_state.graph_params['method_params']['power_law_exponent'] = st.slider(
                    "Power law exponent",
                    min_value=2.0,
                    max_value=4.0,
                    value=st.session_state.graph_params['method_params'].get('power_law_exponent', 2.5),
                    step=0.1,
                    help="Exponent for power law degree distribution"
                )
            elif st.session_state.graph_params['method'] == "Exponential":
                st.session_state.graph_params['method_params']['rate'] = st.slider(
                    "Exponential rate",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.graph_params['method_params'].get('rate', 0.5),
                    step=0.1,
                    help="Rate parameter for exponential degree distribution"
                )
            elif st.session_state.graph_params['method'] == "Uniform":
                st.session_state.graph_params['method_params']['min_factor'] = st.slider(
                    "Minimum degree factor",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.graph_params['method_params'].get('min_factor', 0.5),
                    step=0.1,
                    help="Minimum factor for uniform degree distribution"
                )
                st.session_state.graph_params['method_params']['max_factor'] = st.slider(
                    "Maximum degree factor",
                    min_value=1.0,
                    max_value=2.0,
                    value=st.session_state.graph_params['method_params'].get('max_factor', 1.5),
                    step=0.1,
                    help="Maximum factor for uniform degree distribution"
                )

        with col2:
            # Standard paramaters (degree heterogeneity and edge noise)
            if st.session_state.graph_params['method'] == "Standard":
                st.session_state.graph_params['method_params']['degree_heterogeneity'] = st.slider(
                    "Degree heterogeneity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.graph_params['method_params'].get('degree_heterogeneity', 0.5),
                    step=0.1,
                    help="Amount of degree heterogeneity to introduce"
                )
            # Edge noise is available for all methods
            st.session_state.graph_params['method_params']['edge_noise'] = st.slider(
                "Edge noise",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.graph_params['method_params'].get('edge_noise', 0.0),
                step=0.01,
                help="Amount of random noise to add to edge probabilities"
            )
            
        st.session_state.graph_params['seed'] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=1000,
                value=st.session_state.graph_params['seed'],
                help="Random seed for reproducibility"
        )
        
        # Generate button
        if st.button("Generate Graph", key="generate_graph"):
            with st.spinner("Sampling graph..."):
                # # Sample communities
                # if st.session_state.graph_params['sampling_method'] == "connected":
                #     communities = st.session_state.universe.sample_connected_community_subset(
                #         size=st.session_state.graph_params['num_communities']
                #     )
                # else:
                #     communities = st.session_state.universe.sample_community_subset(
                #         size=st.session_state.graph_params['num_communities']
                #     )

                # Initialize config_model_params
                config_model_params = {}
                
                # Build configuration parameters based on the selected method
                if st.session_state.graph_params['method'] == "DCCC-SBM":
                    # Setup for DCCC-SBM
                    use_dccc_sbm = True
                    community_imbalance = st.session_state.graph_params['method_params'].get('community_imbalance', 0.3)
                    degree_separation = st.session_state.graph_params['method_params'].get('degree_separation', 0.5)
                    degree_distribution = st.session_state.graph_params['method_params'].get('degree_distribution_type', 'power_law')

                    # Distribution-specific parameters
                    dccc_global_degree_params = {}
                    if degree_distribution == "power_law":
                        dccc_global_degree_params = {
                            "exponent": st.session_state.graph_params['method_params'].get('power_law_exponent', 2.5),
                            "x_min": 1.0
                        }
                    elif degree_distribution == "exponential":
                        dccc_global_degree_params = {
                            "rate": st.session_state.graph_params['method_params'].get('rate', 0.5)
                        }
                    elif degree_distribution == "uniform":
                        dccc_global_degree_params = {
                            "min_degree": st.session_state.graph_params['method_params'].get('min_factor', 0.5),
                            "max_degree": st.session_state.graph_params['method_params'].get('max_factor', 1.5)
                        }
                    
                    # Other parameters
                    use_configuration_model = False
                    power_law_exponent = None if degree_distribution != "power_law" else dccc_global_degree_params["exponent"]
                    
                else:
                    # Setup for other methods
                    use_dccc_sbm = False
                    community_imbalance = 0.0
                    degree_separation = 0.5
                    dccc_global_degree_params = None
                    
                    # Configuration model parameters 
                    use_configuration_model = st.session_state.graph_params['method'] != "Standard"
                    degree_distribution = (
                        "power_law" if st.session_state.graph_params['method'] == "Power Law"
                        else "exponential" if st.session_state.graph_params['method'] == "Exponential"
                        else "uniform" if st.session_state.graph_params['method'] == "Uniform"
                        else None
                    )
                    power_law_exponent = st.session_state.graph_params['method_params'].get('power_law_exponent')
                    
                    # Set config_model_params based on method
                    if st.session_state.graph_params['method'] == "Power Law":
                        config_model_params['power_law_exponent'] = st.session_state.graph_params['method_params'].get('power_law_exponent')
                    elif st.session_state.graph_params['method'] == "Exponential":
                        config_model_params['rate'] = st.session_state.graph_params['method_params'].get('rate')
                    elif st.session_state.graph_params['method'] == "Uniform":
                        config_model_params['min_factor'] = st.session_state.graph_params['method_params'].get('min_factor')
                        config_model_params['max_factor'] = st.session_state.graph_params['method_params'].get('max_factor')

                # Set up universal parameters for GraphSample
                universal_params = dict(
                    universe=st.session_state.universe,
                    num_communities=st.session_state.graph_params['num_communities'],
                    n_nodes=st.session_state.graph_params['n_nodes'],
                    min_component_size=st.session_state.graph_params['min_component_size'],
                    degree_heterogeneity=st.session_state.graph_params['method_params'].get('degree_heterogeneity', 0.0),
                    edge_noise=st.session_state.graph_params['method_params'].get('edge_noise', 0.0),
                    feature_regime_balance=0.5,
                    target_homophily=None, # Set to None to use universe homophily
                    target_density=None, # Set to None to use universe density
                    use_configuration_model=use_configuration_model,
                    degree_distribution=degree_distribution,
                    power_law_exponent=power_law_exponent,
                    target_avg_degree=st.session_state.graph_params['method_params'].get('target_avg_degree', None),  # Add placeholder for target_avg_degree
                    max_mean_community_deviation=st.session_state.graph_params['max_mean_community_deviation'],
                    max_max_community_deviation=st.session_state.graph_params['max_max_community_deviation'],
                    max_parameter_search_attempts=st.session_state.graph_params['max_parameter_search_attempts'],
                    parameter_search_range=st.session_state.graph_params['parameter_search_range'],
                    min_edge_density=st.session_state.graph_params['min_edge_density'],
                    max_retries=st.session_state.graph_params['max_retries'],
                    seed=st.session_state.graph_params['seed'],
                    config_model_params=config_model_params,
                    # New DCCC-SBM parameters
                    use_dccc_sbm=use_dccc_sbm,
                    community_imbalance=community_imbalance if use_dccc_sbm else 0.0,
                    degree_separation=degree_separation if use_dccc_sbm else 0.5,
                    dccc_global_degree_params=dccc_global_degree_params if use_dccc_sbm else None,
                    disable_deviation_limiting=st.session_state.graph_params.get('disable_deviation_limiting', False)
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
                fig = plot_degree_distribution(st.session_state.current_graph.graph)
                st.pyplot(fig)
                
                # Add DCCC-SBM specific visualizations if applicable
                if st.session_state.graph_params['method'] == "DCCC-SBM":
                    add_dccc_visualization_to_app(graph)
                
                # Add triangle analysis
                st.markdown('<div class="subsection-header">Triangle Analysis</div>', unsafe_allow_html=True)
                
                # Get triangle statistics
                triangle_stats = graph.analyze_triangles()
                
                # Display total triangles
                st.metric("Total Triangles", triangle_stats['total_triangles'])
                
                # Display additional triangles if any were added
                if triangle_stats['total_additional_triangles'] > 0:
                    st.metric("Additional Triangles", triangle_stats['total_additional_triangles'])
                
                # Create a DataFrame for triangles per community
                triangles_df = pd.DataFrame({
                    'Community': list(triangle_stats['triangles_per_community'].keys()),
                    'Total Triangles': list(triangle_stats['triangles_per_community'].values()),
                    'Additional Triangles': list(triangle_stats['additional_triangles_per_community'].values())
                })
                
                # Display triangles per community
                st.markdown("##### Triangles per Community")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create stacked bar plot
                x = np.arange(len(triangles_df))
                width = 0.35
                
                # Plot natural triangles (total - additional)
                natural_triangles = triangles_df['Total Triangles'] - triangles_df['Additional Triangles']
                ax.bar(x, natural_triangles, width, label='Natural Triangles', color='lightgreen')
                
                # Plot additional triangles on top
                ax.bar(x, triangles_df['Additional Triangles'], width, 
                       bottom=natural_triangles, label='Additional Triangles', color='lightcoral')
                
                ax.set_title("Number of Triangles per Community")
                ax.set_xlabel("Community ID")
                ax.set_ylabel("Number of Triangles")
                ax.set_xticks(x)
                ax.set_xticklabels(triangles_df['Community'])
                ax.legend()
                st.pyplot(fig)
                
                # Display correlation with propensities
                st.markdown("##### Correlation with Triangle Propensities")
                st.metric("Correlation Coefficient", f"{triangle_stats['triangle_propensity_correlation']:.3f}")
                
                # Display triangle propensity comparison plot
                st.markdown("##### Triangle Propensities vs Actual Counts")
                st.pyplot(triangle_stats['triangle_propensity_plot'])
                
                # Add explanation
                st.markdown("""
                This analysis shows:
                1. The total number of triangles in the graph
                2. The distribution of triangles across communities
                3. The correlation between the number of triangles per community and their triangle propensities
                4. A direct comparison of triangle propensities vs actual triangle counts per community
                
                A high correlation indicates that communities with higher triangle propensities tend to form more triangles.
                The comparison plot helps visualize how well the actual triangle counts match the expected propensities.
                """)
                
                # Add community connection analysis
                st.markdown('<div class="subsection-header">Community Connection Analysis</div>', unsafe_allow_html=True)
                
                # Analyze community connections
                connection_analysis = graph.analyze_community_connections()
                
                # Display connection analysis results
                st.markdown("##### Community Connection Matrix")
                st.markdown("""
                This matrix shows the probability of connections between communities.
                - Brighter colors indicate higher probabilities
                - The diagonal shows intra-community connection probabilities
                - Off-diagonal elements show inter-community connection probabilities
                """)
                st.pyplot(connection_analysis['figure'])
                
                # Display community statistics
                st.markdown("##### Community Statistics")
                stats_df = pd.DataFrame({
                    'Community': range(len(connection_analysis['community_sizes'])),
                    'Size': connection_analysis['community_sizes'],
                    'Avg Degree': connection_analysis['avg_degrees'],
                    'Density': connection_analysis['densities']
                })
                st.dataframe(stats_df)
                
                # Store in session state
                st.session_state.current_graph = graph
                
                st.success("Graph sampled successfully!")
                
                # Calculate and display signals
                st.markdown("### Graph Signals")
                
                # Calculate all signals using the unified approach
                signals = graph.calculate_community_signals(
                    structure_metric='kl',
                    degree_method="naive_bayes",
                    degree_metric="accuracy",
                    cv_folds=5
                )
                
                # Display signals in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Structure Signal", f"{signals['mean_structure_signal']:.3f}")
                    st.caption(f"Min: {signals['min_structure_signal']:.3f}, Max: {signals['max_structure_signal']:.3f}")
                    st.caption("Using KL divergence")
                
                with col2:
                    if signals['feature_signal'] is not None:
                        st.metric("Feature Signal", f"{signals['feature_signal']:.3f}")
                        st.caption("Using Random Forest + macro F1")
                    else:
                        st.metric("Feature Signal", "N/A")
                        st.caption("No features available")
                
                with col3:
                    st.metric("Degree Signal", f"{signals['degree_signal']:.3f}")
                    st.caption("Using Naive Bayes classification")
                
                with col4:
                    st.metric("Triangle Signal", f"{signals['triangle_signal']:.3f}")
                    st.caption("Using triangle community signal")
                
                # Display summary statistics
                st.markdown("#### Signal Summary")
                summary_df = pd.DataFrame({
                    'Metric': ['Mean', 'Min', 'Max', 'Std'],
                    'Value': [
                        signals['mean_signal'],
                        signals['min_signal'],
                        signals['max_signal'],
                        signals['std_signal']
                    ]
                })
                st.dataframe(summary_df)
                
                # Display method information
                st.markdown("#### Method Information")
                st.json(signals['method_info'])

                # After generating the graph in the Graph Sampling page, add this visualization:
                # After graph generation, show degree analysis
                st.markdown('<div class="subsection-header">Degree Analysis</div>', unsafe_allow_html=True)
                
                # Calculate degrees from adjacency matrix
                degrees = np.array([d for _, d in graph.graph.degree()])
                
                # Calculate community-wise degree statistics
                community_degrees = {}
                for comm in np.unique(graph.community_labels):
                    # Get the original universe community index
                    universe_comm = graph.communities[comm]
                    comm_nodes = graph.community_labels == comm
                    community_degrees[universe_comm] = {
                        'mean': np.mean(degrees[comm_nodes]),
                        'std': np.std(degrees[comm_nodes]),
                        'min': np.min(degrees[comm_nodes]),
                        'max': np.max(degrees[comm_nodes])
                    }
                
                # Plot 1: Degree centers vs actual mean degrees
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                comm_ids = sorted(community_degrees.keys())  # These are now universe community indices
                centers = [st.session_state.universe.degree_centers[comm] for comm in comm_ids]
                means = [community_degrees[comm]['mean'] for comm in comm_ids]
                stds = [community_degrees[comm]['std'] for comm in comm_ids]
                
                ax1.errorbar(centers, means, yerr=stds, fmt='o', capsize=5)
                ax1.set_xlabel('Degree Center')
                ax1.set_ylabel('Actual Mean Degree')
                ax1.set_title('Degree Centers vs Actual Degrees')
                st.pyplot(fig1)
                
                # Plot 2: Degree distributions per community
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for comm in comm_ids:
                    # Convert back to local community index for node selection
                    local_comm_indices = np.where(graph.communities == comm)[0]
                    if len(local_comm_indices) > 0:
                        local_comm = local_comm_indices[0]
                        comm_nodes = graph.community_labels == local_comm
                        ax2.hist(degrees[comm_nodes], alpha=0.5, label=f'Community {comm}')
                ax2.set_xlabel('Degree')
                ax2.set_ylabel('Count')
                ax2.set_title('Degree Distributions by Community')
                ax2.legend()
                st.pyplot(fig2)
                
                # Add degree assignment debugging visualization
                if hasattr(graph, '_debug_degree_assignment'):
                    st.subheader("Degree Assignment Process")
                    
                    # Plot 3: Degree assignment process
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    
                    # Plot the raw degrees
                    raw_degrees = graph._debug_degree_assignment['raw_degrees']
                    ax3.plot(range(len(raw_degrees)), raw_degrees, 'k-', alpha=0.3, label='Raw Degrees')
                    
                    # Plot the community distributions
                    for comm in comm_ids:
                        # Convert back to local community index for node selection
                        local_comm_indices = np.where(graph.communities == comm)[0]
                        if len(local_comm_indices) > 0:
                            local_comm = local_comm_indices[0]
                            center = graph._debug_degree_assignment['community_centers'][local_comm]
                            mean = graph._debug_degree_assignment['means'][local_comm]
                            std = graph._debug_degree_assignment['stds'][local_comm]
                            
                            # Plot the normal distribution
                            x = np.linspace(mean - 3*std, mean + 3*std, 100)
                            y = stats.norm.pdf(x, mean, std)
                            ax3.plot(x, y * len(raw_degrees) * 0.1, label=f'Community {comm} Distribution')
                            
                            # Plot the actual assignments
                            assignments = graph._debug_degree_assignment['final_assignments'][local_comm]
                            if assignments:
                                nodes, indices, degrees = zip(*assignments)
                                ax3.scatter(indices, degrees, alpha=0.5, label=f'Community {comm} Assignments')
                    
                    ax3.set_xlabel('Node Index')
                    ax3.set_ylabel('Degree')
                    ax3.set_title('Degree Assignment Process')
                    ax3.legend()
                    st.pyplot(fig3)
                    
                    # Display assignment statistics
                    st.write("Assignment Statistics:")
                    stats_data = []
                    for comm in comm_ids:
                        # Convert back to local community index for node selection
                        local_comm_indices = np.where(graph.communities == comm)[0]
                        if len(local_comm_indices) > 0:
                            local_comm = local_comm_indices[0]
                            assignments = graph._debug_degree_assignment['final_assignments'][local_comm]
                            if assignments:
                                nodes, indices, degrees = zip(*assignments)
                                stats_data.append({
                                    'Community': comm,
                                    'Center': graph._debug_degree_assignment['community_centers'][local_comm],
                                    'Mean Index': np.mean(indices),
                                    'Std Index': np.std(indices),
                                    'Mean Degree': np.mean(degrees),
                                    'Std Degree': np.std(degrees)
                                })
                    st.dataframe(pd.DataFrame(stats_data))
                
                # Display community degree statistics
                st.write("Community Degree Statistics:")
                stats_data = []
                for comm in comm_ids:
                    stats_data.append({
                        'Community': comm,
                        'Degree Center': st.session_state.universe.degree_centers[comm],
                        'Mean Degree': community_degrees[comm]['mean'],
                        'Std Degree': community_degrees[comm]['std'],
                        'Min Degree': community_degrees[comm]['min'],
                        'Max Degree': community_degrees[comm]['max']
                    })
                st.dataframe(pd.DataFrame(stats_data))

# Graph Family Generation Page
elif page == "Graph Family Generation":
    st.markdown('<div class="section-header">Graph Family Generation</div>', unsafe_allow_html=True)
    
    if st.session_state.universe is None:
        st.warning("Please generate a universe first in the 'Universe Creation' page.")
    else:
        st.markdown("""
        <div class="info-box">
        Generate families of graphs from the same universe with varying parameters.
        This allows you to explore how different parameter ranges affect graph properties
        while maintaining the same underlying community structure.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for family parameters if not exists
        if 'family_params' not in st.session_state:
            st.session_state.family_params = {
                'n_graphs': 20,
                'min_n_nodes': 50,
                'max_n_nodes': 150,
                'min_communities': 3,
                'max_communities': 6,
                'min_component_size': 5,
                'homophily_range': (-0.1, 0.1),
                'density_range': (-0.02, 0.02),
                'use_dccc_sbm': False,
                'community_imbalance_range': (0.0, 0.5),
                'degree_separation_range': (0.3, 0.8),
                'degree_distribution': 'power_law',
                'power_law_exponent_range': (2.1, 2.9),
                'exponential_rate_range': (0.3, 1.0),
                'uniform_min_factor_range': (0.3, 0.7),
                'uniform_max_factor_range': (1.3, 2.0),
                'disable_deviation_limiting': False,
                'seed': 42
            }
        
        # Family Generation Parameters
        st.markdown('<div class="subsection-header">Family Generation Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.family_params['n_graphs'] = st.number_input(
                "Number of graphs in family",
                min_value=5,
                max_value=100,
                value=st.session_state.family_params['n_graphs'],
                help="Total number of graphs to generate in the family"
            )
            
            st.session_state.family_params['min_n_nodes'] = st.number_input(
                "Minimum nodes per graph",
                min_value=20,
                max_value=500,
                value=st.session_state.family_params['min_n_nodes'],
                help="Minimum number of nodes in any graph"
            )
            
            st.session_state.family_params['max_n_nodes'] = st.number_input(
                "Maximum nodes per graph",
                min_value=st.session_state.family_params['min_n_nodes'],
                max_value=1000,
                value=st.session_state.family_params['max_n_nodes'],
                help="Maximum number of nodes in any graph"
            )
            
        with col2:
            st.session_state.family_params['min_communities'] = st.number_input(
                "Minimum communities per graph",
                min_value=2,
                max_value=st.session_state.universe.K,
                value=st.session_state.family_params['min_communities'],
                help="Minimum number of communities to sample"
            )
            
            st.session_state.family_params['max_communities'] = st.number_input(
                "Maximum communities per graph",
                min_value=st.session_state.family_params['min_communities'],
                max_value=st.session_state.universe.K,
                value=min(st.session_state.family_params['max_communities'], st.session_state.universe.K),
                help="Maximum number of communities to sample"
            )
            
            st.session_state.family_params['min_component_size'] = st.number_input(
                "Minimum component size",
                min_value=1,
                max_value=20,
                value=st.session_state.family_params['min_component_size'],
                help="Minimum size for connected components"
            )
        
        # Parameter Range Configuration
        st.markdown('<div class="subsection-header">Parameter Ranges</div>', unsafe_allow_html=True)
        st.markdown("Configure the ranges for sampling different graph parameters:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Structural Parameters**")
            
            # Homophily range
            homophily_min, homophily_max = st.slider(
                "Homophily range (offset from universe)",
                min_value=-0.3,
                max_value=0.3,
                value=st.session_state.family_params['homophily_range'],
                step=0.05,
                help="Range of homophily offsets relative to universe homophily"
            )
            st.session_state.family_params['homophily_range'] = (homophily_min, homophily_max)
            
            # Density range
            density_min, density_max = st.slider(
                "Density range (offset from universe)",
                min_value=-0.1,
                max_value=0.1,
                value=st.session_state.family_params['density_range'],
                step=0.01,
                help="Range of density offsets relative to universe density"
            )
            st.session_state.family_params['density_range'] = (density_min, density_max)
            
        with col2:
            st.markdown("**Generation Options**")
            
            st.session_state.family_params['use_dccc_sbm'] = st.checkbox(
                "Use DCCC-SBM",
                value=st.session_state.family_params['use_dccc_sbm'],
                help="Use Distribution-Community-Coupled Corrected SBM"
            )
            
            st.session_state.family_params['disable_deviation_limiting'] = st.checkbox(
                "Disable deviation limiting",
                value=st.session_state.family_params['disable_deviation_limiting'],
                help="Allow graphs that exceed community deviation limits"
            )
        
        # DCCC-SBM specific parameters
        if st.session_state.family_params['use_dccc_sbm']:
            st.markdown('<div class="subsection-header">DCCC-SBM Parameters</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Community imbalance range
                imbalance_min, imbalance_max = st.slider(
                    "Community imbalance range",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.family_params['community_imbalance_range'],
                    step=0.1,
                    help="Range for community size imbalance (0=balanced, 1=maximally imbalanced)"
                )
                st.session_state.family_params['community_imbalance_range'] = (imbalance_min, imbalance_max)
                
                # Degree separation range
                separation_min, separation_max = st.slider(
                    "Degree separation range",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.family_params['degree_separation_range'],
                    step=0.1,
                    help="Range for degree distribution separation (0=overlapping, 1=well separated)"
                )
                st.session_state.family_params['degree_separation_range'] = (separation_min, separation_max)
                
            with col2:
                # Degree distribution type
                st.session_state.family_params['degree_distribution'] = st.selectbox(
                    "Degree distribution type",
                    options=['power_law', 'exponential', 'uniform'],
                    index=['power_law', 'exponential', 'uniform'].index(st.session_state.family_params['degree_distribution']),
                    help="Type of degree distribution to use"
                )
                
                # Distribution-specific parameter ranges
                if st.session_state.family_params['degree_distribution'] == 'power_law':
                    exp_min, exp_max = st.slider(
                        "Power law exponent range",
                        min_value=2.0,
                        max_value=4.0,
                        value=st.session_state.family_params['power_law_exponent_range'],
                        step=0.1,
                        help="Range for power law exponents"
                    )
                    st.session_state.family_params['power_law_exponent_range'] = (exp_min, exp_max)
                    
                elif st.session_state.family_params['degree_distribution'] == 'exponential':
                    rate_min, rate_max = st.slider(
                        "Exponential rate range",
                        min_value=0.1,
                        max_value=2.0,
                        value=st.session_state.family_params['exponential_rate_range'],
                        step=0.1,
                        help="Range for exponential distribution rates"
                    )
                    st.session_state.family_params['exponential_rate_range'] = (rate_min, rate_max)
                    
                elif st.session_state.family_params['degree_distribution'] == 'uniform':
                    min_factor_min, min_factor_max = st.slider(
                        "Uniform min factor range",
                        min_value=0.1,
                        max_value=1.0,
                        value=st.session_state.family_params['uniform_min_factor_range'],
                        step=0.1,
                        help="Range for uniform distribution minimum factors"
                    )
                    st.session_state.family_params['uniform_min_factor_range'] = (min_factor_min, min_factor_max)
                    
                    max_factor_min, max_factor_max = st.slider(
                        "Uniform max factor range",
                        min_value=1.0,
                        max_value=3.0,
                        value=st.session_state.family_params['uniform_max_factor_range'],
                        step=0.1,
                        help="Range for uniform distribution maximum factors"
                    )
                    st.session_state.family_params['uniform_max_factor_range'] = (max_factor_min, max_factor_max)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.family_params['max_mean_community_deviation'] = st.slider(
                    "Max mean community deviation",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.15,
                    step=0.01,
                    help="Maximum allowed mean deviation from expected community properties"
                )
                
                st.session_state.family_params['max_max_community_deviation'] = st.slider(
                    "Max maximum community deviation",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.3,
                    step=0.01,
                    help="Maximum allowed deviation for any single community"
                )
                
            with col2:
                st.session_state.family_params['min_edge_density'] = st.slider(
                    "Minimum edge density",
                    min_value=0.001,
                    max_value=0.05,
                    value=0.005,
                    step=0.001,
                    help="Minimum edge density to ensure connected graphs"
                )
                
                st.session_state.family_params['max_retries'] = st.number_input(
                    "Max retries per graph",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum attempts to generate each graph"
                )
        
        # Seed
        st.session_state.family_params['seed'] = st.number_input(
            "Random seed",
            min_value=0,
            max_value=10000,
            value=st.session_state.family_params['seed'],
            help="Random seed for reproducibility"
        )
        
        # Generate Family Button
        if st.button("Generate Graph Family", key="generate_family"):
            with st.spinner("Generating graph family..."):
                try:
                    # Create family generator
                    family_generator = GraphFamilyGenerator(
                        universe=st.session_state.universe,
                        n_graphs=st.session_state.family_params['n_graphs'],
                        min_n_nodes=st.session_state.family_params['min_n_nodes'],
                        max_n_nodes=st.session_state.family_params['max_n_nodes'],
                        min_communities=st.session_state.family_params['min_communities'],
                        max_communities=st.session_state.family_params['max_communities'],
                        min_component_size=st.session_state.family_params['min_component_size'],
                        homophily_range=st.session_state.family_params['homophily_range'],
                        density_range=st.session_state.family_params['density_range'],
                        use_dccc_sbm=st.session_state.family_params['use_dccc_sbm'],
                        community_imbalance_range=st.session_state.family_params['community_imbalance_range'],
                        degree_separation_range=st.session_state.family_params['degree_separation_range'],
                        degree_distribution=st.session_state.family_params['degree_distribution'],
                        power_law_exponent_range=st.session_state.family_params['power_law_exponent_range'],
                        exponential_rate_range=st.session_state.family_params['exponential_rate_range'],
                        uniform_min_factor_range=st.session_state.family_params['uniform_min_factor_range'],
                        uniform_max_factor_range=st.session_state.family_params['uniform_max_factor_range'],
                        disable_deviation_limiting=st.session_state.family_params['disable_deviation_limiting'],
                        max_mean_community_deviation=st.session_state.family_params['max_mean_community_deviation'],
                        max_max_community_deviation=st.session_state.family_params['max_max_community_deviation'],
                        min_edge_density=st.session_state.family_params['min_edge_density'],
                        max_retries=st.session_state.family_params['max_retries'],
                        seed=st.session_state.family_params['seed']
                    )
                    
                    # Generate the family
                    graphs = family_generator.generate_family(
                        show_progress=True,
                        collect_stats=True,
                        max_attempts_per_graph=5
                    )
                    
                    # Store in session state
                    st.session_state.current_family_graphs = graphs
                    st.session_state.current_family_generator = family_generator
                    
                    st.success(f"Successfully generated {len(graphs)} graphs!")
                    
                    # Display generation statistics
                    stats = family_generator.generation_stats
                    
                    st.markdown('<div class="subsection-header">Generation Statistics</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Success Rate", f"{stats['success_rate']:.1%}")
                    with col2:
                        st.metric("Total Time", f"{stats['total_time']:.1f}s")
                    with col3:
                        st.metric("Avg Time/Graph", f"{stats['avg_time_per_graph']:.2f}s")
                    with col4:
                        st.metric("Failed Graphs", stats['failed_graphs'])
                    
                    # Display family summary
                    st.markdown('<div class="subsection-header">Family Summary</div>', unsafe_allow_html=True)
                    
                    summary_df = family_generator.get_family_summary()
                    st.dataframe(summary_df)
                    
                    # Display diversity analysis
                    st.markdown('<div class="subsection-header">Family Diversity Analysis</div>', unsafe_allow_html=True)
                    
                    diversity = family_generator.analyze_family_diversity()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Parameter Ranges**")
                        st.write(f"Node count range: {diversity['node_count_range']}")
                        st.write(f"Edge count range: {diversity['edge_count_range']}")
                        st.write(f"Community count range: {diversity['community_count_range']}")
                        
                    with col2:
                        st.markdown("**Diversity Metrics (CV)**")
                        st.write(f"Node count CV: {diversity['node_count_cv']:.3f}")
                        st.write(f"Edge count CV: {diversity['edge_count_cv']:.3f}")
                        st.write(f"Community count CV: {diversity['community_count_cv']:.3f}")
                    
                    st.markdown("**Community Usage**")
                    st.write(f"Total unique communities: {diversity['total_unique_communities']}")
                    st.write(f"Average communities per graph: {diversity['avg_communities_per_graph']:.1f}")
                    st.write(f"Community reuse rate: {diversity['community_reuse_rate']:.1%}")
                    
                except Exception as e:
                    st.error(f"Error generating graph family: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display results if family exists
        if st.session_state.current_family_graphs is not None and len(st.session_state.current_family_graphs) > 0:
            st.markdown('<div class="section-header">Family Analysis</div>', unsafe_allow_html=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Family Overview",
                "Parameter Distributions", 
                "Graph Statistics",
                "Individual Graphs",
                "Co-occurrence Analysis"
            ])
            
            with tab1:
                st.markdown("### Family Overview")
                
                # Use existing graph family visualization functions
                try:
                    # Create a family dictionary for the visualization functions
                    family_dict = {"Current Family": st.session_state.current_family_graphs}
                    
                    # Try to use the imported visualization functions
                    fig = plot_parameter_distributions(family_dict, ["Current Family"])
                    st.pyplot(fig)
                except Exception as e:
                    st.info("Using built-in family overview visualization")
                    
                    # Fallback: Create our own summary visualization
                    summary_df = st.session_state.current_family_generator.get_family_summary()
                    
                    if not summary_df.empty:
                        # Create a simple overview plot
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # Node count vs edge count
                        axes[0, 0].scatter(summary_df['n_nodes'], summary_df['n_edges'], alpha=0.6)
                        axes[0, 0].set_xlabel('Number of Nodes')
                        axes[0, 0].set_ylabel('Number of Edges')
                        axes[0, 0].set_title('Nodes vs Edges')
                        
                        # Community count distribution
                        axes[0, 1].hist(summary_df['n_communities'], bins=10, alpha=0.7)
                        axes[0, 1].set_xlabel('Number of Communities')
                        axes[0, 1].set_ylabel('Frequency')
                        axes[0, 1].set_title('Community Count Distribution')
                        
                        # Density distribution
                        axes[1, 0].hist(summary_df['density'], bins=15, alpha=0.7)
                        axes[1, 0].set_xlabel('Edge Density')
                        axes[1, 0].set_ylabel('Frequency')
                        axes[1, 0].set_title('Density Distribution')
                        
                        # Generation attempts
                        axes[1, 1].hist(summary_df['attempts'], bins=10, alpha=0.7)
                        axes[1, 1].set_xlabel('Generation Attempts')
                        axes[1, 1].set_ylabel('Frequency')
                        axes[1, 1].set_title('Generation Attempts Distribution')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                try:
                    fig = plot_graph_statistics(st.session_state.current_family_graphs)
                    st.pyplot(fig)
                except Exception as e:
                    st.info("Graph statistics visualization not available")
            
            with tab2:
                st.markdown("### Parameter Distributions")
                
                # Create parameter distribution plots
                summary_df = st.session_state.current_family_generator.get_family_summary()
                
                if not summary_df.empty:
                    # Plot parameter distributions
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Node count distribution
                    axes[0, 0].hist(summary_df['n_nodes'], bins=20, alpha=0.7)
                    axes[0, 0].set_title('Node Count Distribution')
                    axes[0, 0].set_xlabel('Number of Nodes')
                    
                    # Edge count distribution
                    axes[0, 1].hist(summary_df['n_edges'], bins=20, alpha=0.7)
                    axes[0, 1].set_title('Edge Count Distribution')
                    axes[0, 1].set_xlabel('Number of Edges')
                    
                    # Community count distribution
                    axes[0, 2].hist(summary_df['n_communities'], bins=20, alpha=0.7)
                    axes[0, 2].set_title('Community Count Distribution')
                    axes[0, 2].set_xlabel('Number of Communities')
                    
                    # Density distribution
                    axes[1, 0].hist(summary_df['density'], bins=20, alpha=0.7)
                    axes[1, 0].set_title('Density Distribution')
                    axes[1, 0].set_xlabel('Edge Density')
                    
                    # Homophily distribution (if available)
                    if 'target_homophily' in summary_df.columns:
                        axes[1, 1].hist(summary_df['target_homophily'].dropna(), bins=20, alpha=0.7)
                        axes[1, 1].set_title('Target Homophily Distribution')
                        axes[1, 1].set_xlabel('Homophily')
                    
                    # Generation time distribution
                    if 'total_generation_time' in summary_df.columns:
                        axes[1, 2].hist(summary_df['total_generation_time'].dropna(), bins=20, alpha=0.7)
                        axes[1, 2].set_title('Generation Time Distribution')
                        axes[1, 2].set_xlabel('Time (seconds)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with tab3:
                st.markdown("### Graph Statistics")
                
                # Calculate and display aggregate statistics
                summary_df = st.session_state.current_family_generator.get_family_summary()
                
                if not summary_df.empty:
                    # Summary statistics table
                    stats_summary = summary_df[['n_nodes', 'n_edges', 'n_communities', 'density']].describe()
                    st.dataframe(stats_summary)
                    
                    # Correlation matrix
                    st.markdown("#### Parameter Correlations")
                    numeric_cols = ['n_nodes', 'n_edges', 'n_communities', 'density']
                    if 'target_homophily' in summary_df.columns:
                        numeric_cols.append('target_homophily')
                    
                    corr_matrix = summary_df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(numeric_cols)))
                    ax.set_yticks(range(len(numeric_cols)))
                    ax.set_xticklabels(numeric_cols, rotation=45)
                    ax.set_yticklabels(numeric_cols)
                    
                    # Add correlation values
                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                         ha="center", va="center", color="black")
                    
                    plt.colorbar(im)
                    plt.title('Parameter Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with tab4:
                st.markdown("### Individual Graphs")
                
                # Graph selector
                graph_idx = st.selectbox(
                    "Select a graph to examine",
                    range(len(st.session_state.current_family_graphs)),
                    format_func=lambda i: f"Graph {i} ({st.session_state.current_family_graphs[i].n_nodes} nodes, {st.session_state.current_family_graphs[i].graph.number_of_edges()} edges)"
                )
                
                selected_graph = st.session_state.current_family_graphs[graph_idx]
                
                # Set as current graph for other analyses
                if st.button("Set as Current Graph"):
                    st.session_state.current_graph = selected_graph
                    st.success("Graph set as current graph for analysis!")
                
                # Display graph properties
                st.markdown(f"#### Graph {graph_idx} Properties")
                
                # Basic properties
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nodes", selected_graph.n_nodes)
                with col2:
                    st.metric("Edges", selected_graph.graph.number_of_edges())
                with col3:
                    st.metric("Communities", len(selected_graph.communities))
                with col4:
                    density = selected_graph.graph.number_of_edges() / (selected_graph.n_nodes * (selected_graph.n_nodes - 1) / 2) if selected_graph.n_nodes > 1 else 0
                    st.metric("Density", f"{density:.4f}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Community Structure")
                    fig = plot_graph_communities(selected_graph)
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("##### Degree Distribution")
                    fig = plot_degree_distribution(selected_graph.graph)
                    st.pyplot(fig)
                
                # Community analysis
                st.markdown("##### Community Analysis")
                connection_analysis = selected_graph.analyze_community_connections()
                st.pyplot(connection_analysis['figure'])
            
            with tab5:
                st.markdown("### Co-occurrence Analysis")
                
                # Create consistency analyzer if not exists
                if 'consistency_analyzer' not in st.session_state:
                    st.session_state.consistency_analyzer = FamilyConsistencyAnalyzer(
                        st.session_state.current_family_graphs,
                        st.session_state.universe
                    )
                
                # Run co-occurrence analysis
                cooccurrence_results = st.session_state.consistency_analyzer.analyze_cooccurrence()
                
                # Display correlation
                st.markdown("#### Correlation with Expected Co-occurrence")
                st.metric("Correlation", f"{cooccurrence_results['correlation']:.3f}")
                
                # Display difference matrix
                st.markdown("#### Difference Matrix (Actual - Expected)")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cooccurrence_results['difference_matrix'], 
                           cmap='RdBu_r', 
                           center=0,
                           annot=True,
                           fmt='.2f',
                           ax=ax)
                ax.set_title('Co-occurrence Difference Matrix')
                st.pyplot(fig)
            
            # Download family data
            st.markdown('<div class="subsection-header">Export Family Data</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download summary CSV
                summary_df = st.session_state.current_family_generator.get_family_summary()
                if not summary_df.empty:
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "Download Family Summary (CSV)",
                        csv,
                        "graph_family_summary.csv",
                        "text/csv"
                    )
            
            with col2:
                # Save family with graphs
                if st.button("Save Family with Graphs"):
                    st.session_state.current_family_generator.save_family(
                        "graph_family.pkl", 
                        include_graphs=True
                    )
                    st.success("Family saved to graph_family.pkl!")

# Graph Family Analysis Page
elif page == "Graph Family Analysis":
    st.markdown('<div class="section-header">Graph Family Consistency Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.current_family_graphs is None or len(st.session_state.current_family_graphs) == 0:
        st.warning("Please generate a graph family first in the 'Graph Family Generation' page.")
    else:
        st.markdown("""
        <div class="info-box">
        Analyze the topological consistency of your graph family. This analysis measures how well 
        the family preserves structural patterns from the universe and how similar graphs are to each other.
        <br><br>
        <b>Key Metrics:</b>
        <ul>
            <li><b>Pattern Preservation</b>: How well rank ordering of community connections is maintained vs universe</li>
            <li><b>Cross-Graph Similarity</b>: How similar community patterns are across graphs in the family</li>
            <li><b>Generation Fidelity</b>: How well graphs match their intended generation targets</li>
            <li><b>Degree Consistency</b>: How well node degrees correlate with universe degree centers</li>
            <li><b>Triangle Consistency</b>: How well triangle patterns are preserved across the family</li>
            <li><b>Co-occurrence Consistency</b>: How well community co-occurrence patterns are preserved</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize consistency analyzer in session state if not exists
        if 'consistency_analyzer' not in st.session_state:
            st.session_state.consistency_analyzer = None
        if 'consistency_results' not in st.session_state:
            st.session_state.consistency_results = None
        
        # Analysis Controls
        st.markdown('<div class="subsection-header">Analysis Controls</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Current Family Info**")
            st.write(f"Number of graphs: {len(st.session_state.current_family_graphs)}")
            if hasattr(st.session_state, 'current_family_generator'):
                stats = st.session_state.current_family_generator.generation_stats
                st.write(f"Success rate: {stats.get('success_rate', 0):.1%}")
                st.write(f"Generation time: {stats.get('total_time', 0):.1f}s")
        
        with col2:
            st.markdown("**Universe Info**")
            if st.session_state.universe:
                st.write(f"Total communities: {st.session_state.universe.K}")
                st.write(f"Universe homophily: {st.session_state.universe.homophily:.3f}")
                st.write(f"Universe density: {st.session_state.universe.edge_density:.3f}")
        
        with col3:
            st.markdown("**Analysis Options**")
            show_warnings = st.checkbox(
                "Show calculation warnings", 
                value=False,
                help="Display warnings about calculation issues"
            )
            
            detailed_plots = st.checkbox(
                "Show detailed plots",
                value=True,
                help="Include individual score plots in dashboard"
            )
        
        # Run Analysis Button
        if st.button("Run Consistency Analysis", key="run_consistency"):
            with st.spinner("Analyzing family consistency..."):
                try:
                    # Create consistency analyzer
                    analyzer = FamilyConsistencyAnalyzer(
                        st.session_state.current_family_graphs,
                        st.session_state.universe
                    )
                    
                    # Run analysis
                    if not show_warnings:
                        import warnings
                        warnings.filterwarnings('ignore')
                    
                    results = analyzer.analyze_consistency()
                    
                    # Store results
                    st.session_state.consistency_analyzer = analyzer
                    st.session_state.consistency_results = results
                    
                    st.success("Consistency analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error during consistency analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display Results if Analysis has been run
        if st.session_state.consistency_results is not None:
            results = st.session_state.consistency_results
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Overview Dashboard",
                "Detailed Metrics", 
                "Community Coverage",
                "Text Report",
                "Export & Advanced"
            ])
            
            with tab1:
                st.markdown("### Consistency Dashboard")
                
                # Display overall consistency first
                if 'overall' in results and 'score' in results['overall']:
                    overall_score = results['overall']['score']
                    interpretation = results['overall']['interpretation']
                    
                    # Large metric display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            "Overall Consistency Score", 
                            f"{overall_score:.3f}",
                            help="Weighted average of all consistency metrics"
                        )
                        
                        # Color-coded interpretation
                        if overall_score >= 0.8:
                            st.success(f"✅ {interpretation}")
                        elif overall_score >= 0.6:
                            st.warning(f"⚠️ {interpretation}")
                        else:
                            st.error(f"❌ {interpretation}")
                
                # Quick metrics overview
                st.markdown("#### Quick Metrics Overview")
                
                metrics_data = []
                metric_names = ['Pattern Preservation', 'Generation Fidelity', 'Degree Consistency', 
                                'Triangle Consistency', 'Co-occurrence Consistency']
                metric_keys = ['pattern_preservation', 'generation_fidelity', 'degree_consistency',
                               'triangle_consistency', 'cooccurrence_consistency']
                
                for name, key in zip(metric_names, metric_keys):
                    if key in results and 'score' in results[key]:
                        score = results[key]['score']
                        std = results[key].get('std', 0)
                        metrics_data.append({
                            'Metric': name,
                            'Score': f"{score:.3f}",
                            'Std Dev': f"{std:.3f}",
                            'Status': '🟢' if score >= 0.8 else '🟡' if score >= 0.6 else '🔴'
                        })
                
                if metrics_data:
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                # Generate and display dashboard
                try:
                    if detailed_plots:
                        st.markdown("#### Detailed Dashboard")
                        fig = st.session_state.consistency_analyzer.create_consistency_dashboard()
                        st.pyplot(fig)
                    else:
                        st.info("Enable 'Show detailed plots' to see the full dashboard")
                except Exception as e:
                    st.error(f"Error creating dashboard: {str(e)}")
            
            with tab2:
                st.markdown("### Detailed Consistency Metrics")
                
                # Pattern Preservation
                if 'pattern_preservation' in results:
                    st.markdown("#### 🔄 Pattern Preservation")
                    pattern_result = results['pattern_preservation']
                    
                    if 'score' in pattern_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Score", f"{pattern_result['score']:.3f}")
                            st.caption(pattern_result['description'])
                        
                        with col2:
                            if 'individual_correlations' in pattern_result:
                                correlations = pattern_result['individual_correlations']
                                if correlations:
                                    st.metric("Std Dev", f"{pattern_result['std']:.3f}")
                                    st.caption(f"Based on {len(correlations)} graphs")
                        
                        st.markdown(f"**Interpretation:** {pattern_result['interpretation']}")
                        
                        # Plot individual correlations
                        if 'individual_correlations' in pattern_result and pattern_result['individual_correlations']:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            correlations = pattern_result['individual_correlations']
                            ax.plot(range(len(correlations)), correlations, 'o-', alpha=0.7)
                            ax.axhline(y=np.mean(correlations), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(correlations):.3f}')
                            ax.set_xlabel('Graph Index')
                            ax.set_ylabel('Rank Correlation')
                            ax.set_title('Pattern Preservation: Individual Graph Correlations')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    else:
                        st.error(f"Error: {pattern_result.get('error', 'Unknown error')}")
                
                st.divider()
                
                # Generation Fidelity
                if 'generation_fidelity' in results:
                    st.markdown("#### 🎯 Generation Fidelity")
                    fidelity_result = results['generation_fidelity']
                    
                    if 'score' in fidelity_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Score", f"{fidelity_result['score']:.3f}")
                            st.caption(fidelity_result['description'])
                        
                        with col2:
                            if 'individual_scores' in fidelity_result:
                                scores = fidelity_result['individual_scores']
                                if scores:
                                    st.metric("Std Dev", f"{fidelity_result['std']:.3f}")
                                    st.caption(f"Based on {len(scores)} graphs")
                        
                        st.markdown(f"**Interpretation:** {fidelity_result['interpretation']}")
                    else:
                        st.error(f"Error: {fidelity_result.get('error', 'Unknown error')}")
                
                st.divider()
                
                # Degree Consistency
                if 'degree_consistency' in results:
                    st.markdown("#### 📊 Degree Consistency")
                    degree_result = results['degree_consistency']
                    
                    if 'score' in degree_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Score", f"{degree_result['score']:.3f}")
                            st.caption(degree_result['description'])
                        
                        with col2:
                            if 'individual_scores' in degree_result:
                                scores = degree_result['individual_scores']
                                if scores:
                                    st.metric("Std Dev", f"{degree_result['std']:.3f}")
                                    st.caption(f"Based on {len(scores)} graphs")
                        
                        st.markdown(f"**Interpretation:** {degree_result['interpretation']}")
                        
                        # Plot degree consistency scores
                        if 'individual_scores' in degree_result and degree_result['individual_scores']:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            scores = degree_result['individual_scores']
                            ax.plot(range(len(scores)), scores, 'o-', alpha=0.7)
                            ax.axhline(y=np.mean(scores), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(scores):.3f}')
                            ax.set_xlabel('Graph Index')
                            ax.set_ylabel('Degree Correlation')
                            ax.set_title('Degree Consistency: Individual Graph Correlations')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    else:
                        st.error(f"Error: {degree_result.get('error', 'Unknown error')}")
            
            with tab3:
                st.markdown("### Community Coverage Analysis")
                
                if 'community_coverage' in results:
                    coverage = results['community_coverage']
                    
                    # Overall coverage metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Universe Coverage", 
                            f"{coverage.get('coverage_fraction', 0):.1%}",
                            help="Fraction of universe communities used"
                        )
                    
                    with col2:
                        st.metric(
                            "Unique Communities", 
                            coverage.get('total_unique_communities', 0),
                            help="Number of distinct communities used"
                        )
                    
                    with col3:
                        st.metric(
                            "Avg Usage", 
                            f"{coverage.get('avg_usage_per_community', 0):.1f}",
                            help="Average times each community was used"
                        )
                    
                    with col4:
                        st.metric(
                            "Common Communities", 
                            len(coverage.get('communities_in_all_graphs', [])),
                            help="Communities that appear in ALL graphs"
                        )
                    
                    # Community usage visualization
                    st.markdown("#### Community Usage Distribution")
                    
                    if 'community_usage' in coverage and coverage['community_usage']:
                        usage_data = coverage['community_usage']
                        
                        # Create usage histogram
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Usage frequency histogram
                        usage_counts = list(usage_data.values())
                        ax1.hist(usage_counts, bins=max(1, len(set(usage_counts))), alpha=0.7, edgecolor='black')
                        ax1.set_xlabel('Times Used')
                        ax1.set_ylabel('Number of Communities')
                        ax1.set_title('Distribution of Community Usage Frequency')
                        ax1.grid(True, alpha=0.3)
                        
                        # Community usage bar plot (top 20 communities)
                        sorted_usage = sorted(usage_data.items(), key=lambda x: x[1], reverse=True)
                        top_communities = sorted_usage[:20]
                        
                        if top_communities:
                            communities, usage_counts = zip(*top_communities)
                            ax2.bar(range(len(communities)), usage_counts, alpha=0.7)
                            ax2.set_xlabel('Community Index')
                            ax2.set_ylabel('Usage Count')
                            ax2.set_title(f'Top {len(communities)} Most Used Communities')
                            ax2.set_xticks(range(len(communities)))
                            ax2.set_xticklabels([str(c) for c in communities], rotation=45)
                            ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Detailed usage table
                        st.markdown("#### Detailed Community Usage")
                        
                        usage_df = pd.DataFrame([
                            {'Community': comm, 'Usage Count': count, 'Usage %': f"{count/len(st.session_state.current_family_graphs)*100:.1f}%"}
                            for comm, count in sorted_usage
                        ])
                        
                        st.dataframe(usage_df, use_container_width=True)
                        
                        # Special communities
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Communities in ALL graphs:**")
                            common_comms = coverage.get('communities_in_all_graphs', [])
                            if common_comms:
                                st.write(", ".join(map(str, common_comms)))
                            else:
                                st.write("None")
                        
                        with col2:
                            st.markdown("**Rarely used communities (only 1 graph):**")
                            rare_comms = coverage.get('rarely_used_communities', [])
                            if rare_comms:
                                st.write(", ".join(map(str, rare_comms)))
                            else:
                                st.write("None")
                else:
                    st.error("No community coverage data available")
            
            with tab4:
                st.markdown("### Text Report")
                
                try:
                    report = st.session_state.consistency_analyzer.get_summary_report()
                    
                    # Display report in expandable sections
                    st.markdown("#### Summary Report")
                    st.text_area(
                        "Full Analysis Report",
                        value=report,
                        height=400,
                        help="Complete text summary of the consistency analysis"
                    )
                    
                    # Download report button
                    st.download_button(
                        "Download Report (TXT)",
                        report,
                        "family_consistency_report.txt",
                        "text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
            
            with tab5:
                st.markdown("### Export & Advanced Options")
                
                # Export options
                st.markdown("#### Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export results as JSON
                    if st.button("Export Results (JSON)"):
                        import json
                        
                        # Convert results to JSON-serializable format
                        export_results = {}
                        for key, value in results.items():
                            if isinstance(value, dict):
                                export_results[key] = {}
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, (list, float, int, str)):
                                        export_results[key][subkey] = subvalue
                                    else:
                                        export_results[key][subkey] = str(subvalue)
                            else:
                                export_results[key] = str(value)
                        
                        json_str = json.dumps(export_results, indent=2)
                        st.download_button(
                            "Download Results JSON",
                            json_str,
                            "consistency_results.json",
                            "application/json"
                        )
                
                with col2:
                    # Export dashboard as image
                    if st.button("Export Dashboard (PNG)"):
                        try:
                            fig = st.session_state.consistency_analyzer.create_consistency_dashboard()
                            
                            import io
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                            
                            st.download_button(
                                "Download Dashboard PNG",
                                buf.getvalue(),
                                "consistency_dashboard.png",
                                "image/png"
                            )
                        except Exception as e:
                            st.error(f"Error exporting dashboard: {str(e)}")
                
                # Advanced options
                st.markdown("#### Advanced Analysis Options")
                
                if st.button("Recalculate with Warnings Enabled"):
                    with st.spinner("Recalculating with detailed warnings..."):
                        try:
                            import warnings
                            warnings.resetwarnings()
                            
                            analyzer = FamilyConsistencyAnalyzer(
                                st.session_state.current_family_graphs,
                                st.session_state.universe
                            )
                            
                            results = analyzer.analyze_consistency()
                            
                            st.session_state.consistency_analyzer = analyzer
                            st.session_state.consistency_results = results
                            
                            st.success("Analysis recalculated with warnings enabled!")
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"Error in recalculation: {str(e)}")
                
                # Debug information
                with st.expander("Debug Information"):
                    st.markdown("**Analysis Results Structure:**")
                    st.json(results)
        
        else:
            st.info("Click 'Run Consistency Analysis' to analyze your graph family.")

# Metapath Analysis Page
elif page == "Metapath Analysis":
    st.markdown('<div class="section-header">Metapath Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.current_graph is None:
        st.warning("Please generate a graph first in the 'Graph Sampling' page.")
    else:
        st.markdown("""
        <div class="info-box">
        Analyze metapaths in the graph to understand community-level patterns and relationships.
        Features include:
        <ul>
            <li>Community-level metapath visualization</li>
            <li>Metapath statistics and analysis</li>
            <li>Node classification based on metapaths</li>
            <li>Feature importance analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameters
        st.markdown('<div class="subsection-header">Metapath Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            theta = st.slider(
                "Probability threshold (θ)",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.metapath_analysis_state['params']['theta'],
                help="Threshold for considering an edge likely in the community graph"
            )
            min_length = st.slider(
                "Minimum metapath length",
                min_value=2,
                max_value=5,
                value=2,
                help="Minimum number of communities in a metapath"
            )
            max_length = st.slider(
                "Maximum metapath length",
                min_value=min_length,
                max_value=5,
                value=max(min_length, st.session_state.metapath_analysis_state['params']['max_length']),
                help="Maximum number of communities in a metapath"
            )
        
        with col2:
            allow_loops = st.checkbox(
                "Allow loops",
                value=st.session_state.metapath_analysis_state['params']['allow_loops'],
                help="Allow metapaths to visit the same community multiple times"
            )
            allow_backtracking = st.checkbox(
                "Allow backtracking",
                value=st.session_state.metapath_analysis_state['params']['allow_backtracking'],
                help="Allow metapaths to return to the previous community"
            )
            top_k = st.slider(
                "Number of top metapaths",
                min_value=1,
                max_value=10,
                value=st.session_state.metapath_analysis_state['params']['top_k'],
                help="Number of top metapaths to analyze in detail"
            )
        
        # Run analysis button
        if st.button("Run Metapath Analysis", key="run_metapath"):
            print("\n=== DEBUG: Run Metapath button clicked ===")
            print(f"Current graph state: {st.session_state.current_graph is not None}")
            print(f"Graph type: {type(st.session_state.current_graph)}")
            
            with st.spinner("Analyzing metapaths..."):
                success = run_metapath_analysis(
                    st.session_state.current_graph,
                    theta,
                    max_length,
                    allow_loops,
                    allow_backtracking,
                    top_k,
                    min_length
                )
                if success:
                    st.success("Metapath analysis completed successfully!")
                else:
                    st.error("Error during metapath analysis. Check console for details.")
        
        # Display results if analysis has been run
        if st.session_state.metapath_analysis_state['analysis_run'] and st.session_state.metapath_analysis_state['results'] is not None:
            results = st.session_state.metapath_analysis_state['results']
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Community Structure",
                "Metapath Statistics",
                "Node Classification",
                "Multi-Metapath Node Classification",
                "K-hop Metapath Detection",
            ])
            
            with tab1:
                # Show P_sub matrix visualization
                st.markdown("##### Community Edge Probability Matrix (P_sub)")
                st.markdown("""
                This heatmap shows the probability of edges between communities. 
                - Brighter colors indicate higher probabilities
                - The matrix shows how likely nodes from one community are to connect to nodes in another community
                - This helps understand the underlying community structure that gives rise to metapaths
                """)
                st.pyplot(results['P_sub_figure'])
                
                # Show community-level graph
                st.markdown("##### Community-Level Metapath Graph")
                st.markdown("""
                This graph shows the community-level structure with edges representing likely metapaths.
                - Nodes represent communities
                - Edge weights show the probability of connections between communities
                - This helps visualize the paths that are most likely to occur
                """)
                fig = visualize_community_metapath_graph(
                    results['P_matrix'],
                    theta=st.session_state.metapath_analysis_state['params']['theta'],
                    community_mapping=results['community_mapping']
                )
                st.pyplot(fig)
            
            with tab2:
                st.markdown("##### Metapath Statistics")
                stats = results['statistics']
                
                # Convert to DataFrame for display
                stats_df = pd.DataFrame({
                    'Metapath': stats['metapaths'],
                    'Instances': stats['instances_count'],
                    'Avg Path Length': [f"{x:.2f}" for x in stats['avg_path_length']],
                    'Node Participation (%)': [f"{x*100:.1f}%" for x in stats['participation']]
                })
                
                st.markdown("""
                **How metapaths are ranked:**
                Metapaths are ranked based on their probability in the community graph, not just the number of instances.
                The probability is calculated as the product of edge probabilities along the path.
                
                For example, a path "0 → 1 → 2" has probability = P(0,1) × P(1,2)
                where P(i,j) is the probability of an edge between communities i and j.
                
                This means that even if a metapath has fewer instances, it might be ranked higher if:
                - The edges between its communities have high probabilities
                - The path is more likely to occur in the underlying community structure
                """)
                
                st.dataframe(stats_df)
                
                # Select a metapath to visualize
                selected_metapath_idx = st.selectbox(
                    "Select a metapath to visualize",
                    range(min(st.session_state.metapath_analysis_state['params']['top_k'], len(results['metapaths']))),
                    format_func=lambda i: stats['metapaths'][i],
                    key="metapath_selection"
                )
                
                # Show selected metapath
                selected_metapath = results['metapaths'][selected_metapath_idx]
                selected_instances = results['instances'][selected_metapath_idx]
                
                if selected_instances:
                    st.markdown("##### Full Graph with Metapath Instances")
                    st.markdown("""
                    This visualization shows the full graph with metapath instances highlighted.
                    - Nodes are colored by their community
                    - Regular edges are shown in light gray
                    - Metapath instances are highlighted with bold, colored edges
                    """)
                    fig = visualize_metapaths(
                        st.session_state.current_graph.graph,
                        st.session_state.current_graph.community_labels,
                        selected_metapath,
                        selected_instances,
                        title=f"Metapath: {stats['metapaths'][selected_metapath_idx]}"
                    )
                    st.pyplot(fig)
                else:
                    st.info("No instances found for this metapath in the graph.")
            
            with tab3:
                st.markdown("#### Multi-Metapath Node Classification")
                st.markdown("""
                This tab performs multi-label classification to predict whether a node participates in multiple metapaths.
                For each node, a binary vector label is created: 1 if the node participates in the metapath, 0 otherwise.
                The classifier predicts this vector for each node (multi-label classification).
                """)
                metapath_options = results['metapaths'][:st.session_state.metapath_analysis_state['params']['top_k']]
                metapath_labels = [f"{i}: {stats['metapaths'][i]}" for i in range(len(metapath_options))]
                selected_indices = st.multiselect(
                    "Select metapaths for multi-label classification",
                    options=list(range(len(metapath_options))),
                    format_func=lambda i: metapath_labels[i],
                    default=list(range(min(2, len(metapath_options))))
                )
                
                if not selected_indices:
                    st.warning("Please select at least one metapath")
                else:
                    selected_metapaths = [metapath_options[i] for i in selected_indices]
                    
                    # Add a button to create splits first
                    st.markdown("### Create Train/Val/Test Splits")
                    st.markdown("""
                    First, create consistent train/validation/test splits that will be used for all models.
                    This ensures fair comparison between different model types.
                    """)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        train_size = st.slider("Train size", 0.5, 0.9, 0.7, 0.05, key="ml_train_size")
                    with col2:
                        val_size = st.slider("Validation size", 0.05, 0.25, 0.15, 0.05, key="ml_val_size")
                    with col3:
                        test_size = st.slider("Test size", 0.05, 0.25, 0.15, 0.05, key="ml_test_size")
                    
                    total = train_size + val_size + test_size
                    if abs(total - 1.0) > 1e-6:
                        st.warning(f"Split sizes sum to {total:.2f}, should be 1.0. Please adjust.")
                    
                    if st.button("Create Splits", key="create_splits"):
                        with st.spinner("Creating train/val/test splits..."):
                            try:
                                # Prepare features
                                feature_opts = {
                                    'use_degree': True,
                                    'use_clustering': True,
                                    'use_node_features': True
                                }
                                
                                X = prepare_node_features(
                                    st.session_state.current_graph,
                                    st.session_state.current_graph.community_labels,
                                    use_degree=feature_opts['use_degree'],
                                    use_clustering=feature_opts['use_clustering'],
                                    use_node_features=feature_opts['use_node_features']
                                )
                                
                                # Prepare multi-label targets
                                n_nodes = X.shape[0]
                                Y = np.zeros((n_nodes, len(selected_metapaths)), dtype=int)
                                graph_nx = st.session_state.current_graph.graph
                                
                                for j, metapath in enumerate(selected_metapaths):
                                    instances = find_metapath_instances(graph_nx, st.session_state.current_graph.community_labels, metapath)
                                    participating_nodes = set()
                                    for instance in instances:
                                        participating_nodes.update(instance)
                                    for i in range(n_nodes):
                                        if i in participating_nodes:
                                            Y[i, j] = 1
                                
                                # Create splits
                                splits = create_consistent_train_val_test_split(
                                    X, Y, 
                                    train_size=train_size, 
                                    val_size=val_size, 
                                    test_size=test_size,
                                    stratify=False,  # Can't easily stratify multi-label
                                    seed=42
                                )
                                
                                # Store in session state
                                if 'metapath_multi_label_state' not in st.session_state.metapath_analysis_state:
                                    st.session_state.metapath_analysis_state['metapath_multi_label_state'] = {}
                                    
                                st.session_state.metapath_analysis_state['metapath_multi_label_state'] = {
                                    'splits_created': True,
                                    'splits': splits,
                                    'X': X,
                                    'Y': Y,
                                    'feature_opts': feature_opts,
                                    'selected_metapaths': selected_metapaths
                                }
                                
                                st.success("Splits created successfully!")
                                
                                # Display split statistics
                                st.markdown("#### Split Statistics")
                                train_size = len(splits['train_indices']) / n_nodes
                                val_size = len(splits['val_indices']) / n_nodes
                                test_size = len(splits['test_indices']) / n_nodes
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Train Split", f"{train_size:.1%}")
                                col2.metric("Validation Split", f"{val_size:.1%}")
                                col3.metric("Test Split", f"{test_size:.1%}")
                                
                                # Display label distribution in splits
                                st.markdown("#### Label Distribution in Splits")
                                train_label_dist = Y[splits['train_indices']].mean(axis=0)
                                val_label_dist = Y[splits['val_indices']].mean(axis=0)
                                test_label_dist = Y[splits['test_indices']].mean(axis=0)
                                
                                dist_df = pd.DataFrame({
                                    'Metapath': [metapath_labels[i] for i in selected_indices],
                                    'Train': [f"{x:.1%}" for x in train_label_dist],
                                    'Validation': [f"{x:.1%}" for x in val_label_dist],
                                    'Test': [f"{x:.1%}" for x in test_label_dist]
                                })
                                
                                st.dataframe(dist_df)
                                
                            except Exception as e:
                                st.error(f"Error creating splits: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    # Feature selection 
                    if 'metapath_multi_label_state' in st.session_state.metapath_analysis_state and \
                       st.session_state.metapath_analysis_state['metapath_multi_label_state'].get('splits_created', False):
                        
                        st.markdown("### Feature Selection")
                        col_feat1, col_feat2, col_feat3 = st.columns(3)
                        with col_feat1:
                            use_degree = st.checkbox("Use Degree", value=True, key="ml_use_degree")
                        with col_feat2:
                            use_clustering = st.checkbox("Use Clustering Coefficient", value=True, key="ml_use_clustering")
                        with col_feat3:
                            use_node_features = st.checkbox("Use Node Features", value=True, key="ml_use_node_features")
                        
                        feature_opts = {
                            'use_degree': use_degree,
                            'use_clustering': use_clustering,
                            'use_node_features': use_node_features
                        }
                        
                        st.markdown("### Model Selection")
                        model_type = st.selectbox(
                            "Select model type",
                            ["Random Forest", "MLP", "GCN", "GraphSAGE"],
                            key="ml_model_type"
                        )
                        
                        model_type_map = {
                            "Random Forest": "rf",
                            "MLP": "mlp",
                            "GCN": "gcn",
                            "GraphSAGE": "sage"
                        }
                        
                        selected_model_type = model_type_map[model_type]
                        
                        if st.button("Run Multi-Metapath Classification", key="run_multilabel"):
                            with st.spinner("Running multi-label classification..."):
                                try:
                                    # Get the splits from session state
                                    multi_label_state = st.session_state.metapath_analysis_state['metapath_multi_label_state']
                                    splits = multi_label_state['splits']
                                    
                                    # Run the improved classification
                                    results_ml = multi_metapath_node_classification_improved(
                                        st.session_state.current_graph,
                                        st.session_state.current_graph.community_labels,
                                        multi_label_state['selected_metapaths'],
                                        model_type=selected_model_type,
                                        feature_opts=feature_opts,
                                        splits=splits,
                                        seed=42
                                    )
                                    
                                    # Store results in session state
                                    st.session_state.metapath_analysis_state['classification_state'] = {
                                        'run': True,
                                        'results': results_ml,
                                        'model_type': selected_model_type
                                    }
                                    
                                    st.success("Classification completed successfully!")
                                    
                                except Exception as e:
                                    st.error(f"Error running classification: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        
                        if ('classification_state' in st.session_state.metapath_analysis_state and 
                            st.session_state.metapath_analysis_state['classification_state']['run']):
                            classification_state = st.session_state.metapath_analysis_state['classification_state']
                            results_ml = classification_state['results']
                            model_type = classification_state['model_type']
                            
                            st.markdown("### Classification Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### F1 Score")
                                f1_df = pd.DataFrame({
                                    'Set': ['Train', 'Validation', 'Test'],
                                    'F1 Score': [
                                        results_ml['f1_score']['train'],
                                        results_ml['f1_score']['val'],
                                        results_ml['f1_score']['test']
                                    ]
                                })
                                st.dataframe(f1_df)
                            with col2:
                                st.markdown("#### Accuracy")
                                acc_df = pd.DataFrame({
                                    'Set': ['Train', 'Validation', 'Test'],
                                    'Accuracy': [
                                        results_ml['accuracy']['train'],
                                        results_ml['accuracy']['val'],
                                        results_ml['accuracy']['test']
                                    ]
                                })
                                st.dataframe(acc_df)
                            
                            st.markdown("#### Metapath Participation Rates")
                            participation_df = pd.DataFrame({
                                'Metapath': st.session_state.metapath_analysis_state['metapath_multi_label_state']['selected_metapaths'],
                                'Participation Rate': results_ml['participation_rates']
                            })
                            st.dataframe(participation_df)
                            
                            st.markdown("#### Model Performance Visualization")
                            vis_results = {
                                'train_f1': results_ml['f1_score']['train'],
                                'val_f1': results_ml['f1_score']['val'],
                                'test_f1': results_ml['f1_score']['test'],
                                'feature_importance': results_ml['feature_importance'],
                                'history': results_ml.get('history')
                            }
                            fig = visualize_model_performance(vis_results, model_type, multi_label=True)
                            st.pyplot(fig)
                            
                            # Display GNN embeddings if applicable
                            if model_type in ['gcn', 'sage'] and 'model' in results_ml:
                                st.markdown("#### Node Embedding Visualization")
                                visualize_embeddings = st.checkbox("Visualize embeddings", value=False)
                                if visualize_embeddings:
                                    with st.spinner("Computing node embeddings..."):
                                        # [Visualization code remains the same]
                                        pass
                            
                            st.markdown("### Hyperparameter Optimization")
                            run_optuna = st.checkbox("Run hyperparameter optimization", value=False)
                            if run_optuna:
                                n_trials = st.slider("Number of Optuna trials", min_value=5, max_value=100, value=20)
                                timeout = st.slider("Timeout (seconds)", min_value=30, max_value=1800, value=300)
                                
                                if st.button("Run Optimization", key="run_optuna_multi"):
                                    with st.spinner("Running hyperparameter optimization..."):
                                        try:
                                            # Get the splits from session state
                                            multi_label_state = st.session_state.metapath_analysis_state['metapath_multi_label_state']
                                            splits = multi_label_state['splits']
                                            
                                            optim_results = optimize_hyperparameters_for_metapath_improved(
                                                st.session_state.current_graph,
                                                st.session_state.current_graph.community_labels,
                                                multi_label_state['selected_metapaths'],
                                                model_type=selected_model_type,
                                                multi_label=True,
                                                feature_opts=feature_opts,
                                                splits=splits,
                                                n_trials=n_trials,
                                                timeout=timeout,
                                                seed=42
                                            )
                                            
                                            st.session_state.metapath_analysis_state['optuna_results'] = optim_results
                                            st.success("Optimization completed successfully!")
                                            
                                            st.markdown("#### Best Parameters")
                                            st.json(optim_results['best_params'])
                                            
                                            st.markdown("#### Performance with Best Parameters")
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric("Train F1", f"{optim_results['train_f1']:.4f}")
                                            col2.metric("Validation F1", f"{optim_results['val_f1']:.4f}")
                                            col3.metric("Test F1", f"{optim_results['test_f1']:.4f}")
                                            
                                        except Exception as e:
                                            st.error(f"Error running optimization: {str(e)}")
                                            import traceback
                                            st.code(traceback.format_exc())
                    else:
                        st.warning("Please create train/val/test splits first before running classification.")

            with tab4:
                # K-hop Metapath Detection logic (moved here)
                st.markdown("#### K-hop Metapath Detection")
                st.markdown("""
                This analysis detects k-hop relationships along a selected metapath and
                labels starting nodes based on the feature regime of their k-hop neighbors.
                
                **How it works:**
                1. Select a metapath and k (hop distance)
                2. The analysis identifies instances of the metapath in the graph
                3. For each instance, it finds nodes that are k-hops away from starting nodes
                4. Starting nodes are labeled based on the feature regime of their k-hop neighbors
                """)
                
                # Select a metapath to analyze
                metapath_options = results['metapaths'][:st.session_state.metapath_analysis_state['params']['top_k']]
                stats = results['statistics']
                metapath_labels = [f"{i}: {stats['metapaths'][i]}" for i in range(len(metapath_options))]
                
                selected_metapath_idx = st.selectbox(
                    "Select a metapath for k-hop analysis",
                    range(min(st.session_state.metapath_analysis_state['params']['top_k'], len(results['metapaths']))),
                    format_func=lambda i: metapath_labels[i],
                    key="khop_metapath_selection"
                )
                
                selected_metapath = results['metapaths'][selected_metapath_idx]
                
                # Select k (hop distance)
                max_k = len(selected_metapath) - 1
                k = st.slider("Select k (hop distance)", 1, max(1, max_k), 1, key="khop_k_selection")
                
                # Run analysis button
                if st.button("Run K-hop Metapath Detection", key="run_khop_analysis"):
                    with st.spinner("Analyzing k-hop metapath relationships..."):
                        # Check if graph has node_regimes
                        if not hasattr(st.session_state.current_graph, 'node_regimes') or st.session_state.current_graph.node_regimes is None:
                            st.error("Graph does not have feature regime information. Please generate a graph with features.")
                        else:
                            try:
                                # Run k-hop metapath detection
                                khop_result = khop_metapath_detection(
                                    st.session_state.current_graph.graph,
                                    st.session_state.current_graph.community_labels,
                                    st.session_state.current_graph.node_regimes,
                                    selected_metapath,
                                    k
                                )
                                # Store results in session state
                                st.session_state.metapath_analysis_state['khop_detection_result'] = {
                                    'result': khop_result,
                                    'metapath': selected_metapath,
                                    'k': k,
                                    'analysis_run': True
                                }
                                st.success("K-hop metapath detection completed successfully!")
                            except Exception as e:
                                st.error(f"Error during k-hop metapath detection: {str(e)}")
                # Display results if analysis has been run
                if ('khop_detection_result' in st.session_state.metapath_analysis_state and 
                    st.session_state.metapath_analysis_state['khop_detection_result'].get('analysis_run', False)):
                    khop_data = st.session_state.metapath_analysis_state['khop_detection_result']
                    khop_result = khop_data['result']
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    with col1:
                        # Display basic statistics
                        st.markdown("##### K-hop Detection Results")
                        st.markdown(f"Path: {' → '.join([str(c) for c in khop_result['path_community_sequence']])}")
                        st.markdown(f"Starting Community: {khop_result['starting_community']}")
                        st.markdown(f"Target Community (k-hop): {khop_result['target_community']}")
                        st.markdown(f"Total Relationships: {khop_result['total_relationships']}")
                        # Add interactive filters
                        st.markdown("##### Filter Results")
                        min_relationships = st.slider(
                            "Minimum relationships per regime",
                            min_value=1,
                            max_value=max(khop_result['regime_counts'].values()) if khop_result['regime_counts'] else 1,
                            value=1,
                            key="min_relationships"
                        )
                        # Filter regimes based on minimum relationships
                        filtered_regimes = {
                            regime: count for regime, count in khop_result['regime_counts'].items()
                            if count >= min_relationships
                        }
                    with col2:
                        # Display regime distribution
                        st.markdown("##### Feature Regime Distribution")
                        if filtered_regimes:
                            # Create bar chart of regime distribution
                            regimes = list(filtered_regimes.keys())
                            counts = [filtered_regimes[r] for r in regimes]
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.bar(regimes, counts)
                            # Label each bar with its value
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{int(height)}', ha='center', va='bottom')
                            ax.set_xlabel('Feature Regime')
                            ax.set_ylabel('Count')
                            ax.set_title('Distribution of Feature Regimes in K-hop Neighbors')
                            ax.set_xticks(regimes)
                            st.pyplot(fig)
                            # Add download button for regime distribution data
                            regime_df = pd.DataFrame({
                                'Regime': regimes,
                                'Count': counts
                            })
                            st.download_button(
                                "Download Regime Distribution",
                                regime_df.to_csv(index=False),
                                "regime_distribution.csv",
                                "text/csv",
                                key="download_regime_dist"
                            )
                        else:
                            st.info("No regimes meet the minimum relationship threshold.")
                    # Visualize relationships in the graph
                    st.markdown("##### K-hop Relationships Visualization")
                    if khop_result['total_relationships'] > 0:
                        # Add visualization options
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            show_all_nodes = st.checkbox("Show all nodes", value=True)
                            highlight_starting = st.checkbox("Highlight starting nodes", value=True)
                        with viz_col2:
                            node_size = st.slider("Node size", 30, 200, 80)
                            edge_width = st.slider("Edge width", 1, 5, 2)
                        fig = visualize_khop_metapath_detection(
                            st.session_state.current_graph.graph,
                            st.session_state.current_graph.community_labels,
                            st.session_state.current_graph.node_regimes,
                            khop_data['metapath'],
                            khop_data['k'],
                            khop_result,
                            title=f"K-hop Metapath Detection (k={khop_data['k']})",
                            show_all_nodes=show_all_nodes,
                            highlight_starting=highlight_starting,
                            node_size=node_size,
                            edge_width=edge_width
                        )
                        st.pyplot(fig)
                        # Add download button for visualization
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                        st.download_button(
                            "Download Visualization",
                            buf.getvalue(),
                            "khop_visualization.png",
                            "image/png",
                            key="download_viz"
                        )

                    # Inside the K-hop Metapath Detection tab, add a new section after visualization
                    # In the K-hop Feature Regime Distribution Prediction section
                    st.markdown("#### K-hop Feature Regime Distribution Prediction")
                    st.markdown("""
                    This analysis predicts the distribution or counts of feature regimes in k-hop neighbors.
                    Rather than binary classification (whether a node participates in a metapath),
                    this task predicts the numerical pattern of regimes in neighbors.

                    **How it works:**
                    1. For each starting node, we calculate the counts/distribution of regimes in its k-hop neighbors
                    2. We train a regression model to predict these values based on node features
                    3. The model learns to estimate how many nodes of each regime type will be found in the neighborhood
                    """)

                    # Add option to predict counts or normalized distributions
                    predict_counts = st.checkbox(
                        "Predict raw counts (instead of normalized distribution)",
                        value=True,
                        help="If checked, predicts actual regime counts. If unchecked, predicts normalized regime distributions (probabilities)."
                    )

                    # Add model selection
                    prediction_model = st.selectbox(
                        "Select model for regime prediction",
                        ["Random Forest", "MLP", "GCN", "GraphSAGE"],
                        key="regime_pred_model"
                    )

                    model_type_map = {
                        "Random Forest": "rf",
                        "MLP": "mlp",
                        "GCN": "gcn",
                        "GraphSAGE": "sage"
                    }

                    selected_prediction_model = model_type_map[prediction_model]

                    # Feature selection 
                    st.markdown("### Feature Selection")
                    col_feat1, col_feat2, col_feat3 = st.columns(3)
                    with col_feat1:
                        pred_use_degree = st.checkbox("Use Degree", value=True, key="regime_pred_use_degree")
                    with col_feat2:
                        pred_use_clustering = st.checkbox("Use Clustering Coefficient", value=True, key="regime_pred_use_clustering")
                    with col_feat3:
                        pred_use_node_features = st.checkbox("Use Node Features", value=True, key="regime_pred_use_node_features")

                    pred_feature_opts = {
                        'use_degree': pred_use_degree,
                        'use_clustering': pred_use_clustering,
                        'use_node_features': pred_use_node_features
                    }

                    # Run prediction button
                    if st.button("Run Regime Prediction", key="run_regime_pred"):
                        # Check if we already have splits in session state
                        if 'regime_prediction_splits' in st.session_state:
                            # Try to get existing splits for this configuration
                            try:
                                metapath_str = '_'.join(map(str, selected_metapath))
                                feature_opts_str = '_'.join([f"{k}_{v}" for k, v in pred_feature_opts.items()])
                                split_key = f"regime_splits_{metapath_str}_{k}_{not predict_counts}_{feature_opts_str}_42"
                                
                                existing_splits = None
                                if split_key in st.session_state.regime_prediction_splits:
                                    existing_splits = st.session_state.regime_prediction_splits[split_key]['splits']
                            except:
                                existing_splits = None
                        else:
                            existing_splits = None
                        with st.spinner("Running regime prediction..."):
                            try:
                                # Run the prediction
                                prediction_results = khop_regime_prediction(
                                    st.session_state.current_graph,
                                    st.session_state.current_graph.community_labels,
                                    st.session_state.current_graph.node_regimes,
                                    selected_metapath,
                                    k,
                                    model_type=selected_prediction_model,
                                    task_type='regression',
                                    normalize_counts=not predict_counts,
                                    feature_opts=pred_feature_opts,
                                    splits=existing_splits,  # Use existing splits if available
                                    seed=42,
                                    use_cached_splits=True  # Enable using cached splits
                                )
                                
                                # Store results in session state
                                st.session_state.metapath_analysis_state['regime_prediction_result'] = {
                                    'results': prediction_results,
                                    'metapath': selected_metapath,
                                    'k': k,
                                    'model_type': selected_prediction_model,
                                    'predict_counts': predict_counts,
                                    'analysis_run': True
                                }
                                
                                prediction_type = "count" if predict_counts else "distribution"
                                st.success(f"Regime {prediction_type} prediction completed successfully!")
                                
                            except Exception as e:
                                st.error(f"Error during regime prediction: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                    # Display results if prediction has been run
                    if ('regime_prediction_result' in st.session_state.metapath_analysis_state and 
                        st.session_state.metapath_analysis_state['regime_prediction_result'].get('analysis_run', False)):
                        
                        pred_results = st.session_state.metapath_analysis_state['regime_prediction_result']
                        model_type = pred_results['model_type']
                        results = pred_results['results']
                        predict_counts = pred_results.get('predict_counts', True)
                        
                        st.markdown("### Prediction Results")
                        prediction_type = "Counts" if predict_counts else "Distribution"
                        st.markdown(f"#### Regime {prediction_type} Prediction Performance")
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Mean Squared Error")
                            mse_df = pd.DataFrame({
                                'Set': ['Train', 'Validation', 'Test'],
                                'MSE': [
                                    results['metrics']['mse']['train'],
                                    results['metrics']['mse']['val'],
                                    results['metrics']['mse']['test']
                                ]
                            })
                            st.dataframe(mse_df)
                        
                        with col2:
                            st.markdown("#### R² Score")
                            r2_df = pd.DataFrame({
                                'Set': ['Train', 'Validation', 'Test'],
                                'R²': [
                                    results['metrics']['r2']['train'],
                                    results['metrics']['r2']['val'],
                                    results['metrics']['r2']['test']
                                ]
                            })
                            st.dataframe(r2_df)
                        
                        # Visualization
                        st.markdown("#### Prediction Visualization")
                        fig = visualize_regime_prediction_performance(results, model_type)
                        st.pyplot(fig)
                        
                        # Regime counts/distribution examples
                        st.markdown(f"#### Example Regime {prediction_type} Predictions")
                        # Select a subset of test samples to display
                        test_indices = results['splits']['test_indices']
                        n_examples = min(5, len(test_indices))
                        
                        if n_examples > 0:
                            example_indices = np.random.choice(len(test_indices), size=n_examples, replace=False)
                            
                            for i, idx in enumerate(example_indices):
                                test_idx = test_indices[idx]
                                true_vals = results['true_values']['test'][idx]
                                pred_vals = results['predictions']['test'][idx]
                                
                                st.markdown(f"##### Example {i+1}")
                                fig, ax = plt.subplots(figsize=(10, 4))
                                
                                x = np.arange(len(true_vals))
                                width = 0.35
                                
                                rects1 = ax.bar(x - width/2, true_vals, width, label='True')
                                rects2 = ax.bar(x + width/2, pred_vals, width, label='Predicted')
                                
                                ax.set_xticks(x)
                                ax.set_xticklabels([f'Regime {j}' for j in range(len(true_vals))])
                                ax.set_ylabel('Count' if predict_counts else 'Probability')
                                ax.set_title(f'Regime {prediction_type} Prediction - Example {i+1}')
                                ax.legend()
                                
                                # Add value labels on top of bars
                                for rect in rects1:
                                    height = rect.get_height()
                                    if height > 0:
                                        ax.text(rect.get_x() + rect.get_width()/2., height,
                                            f'{height:.1f}' if not predict_counts else f'{int(height)}',
                                            ha='center', va='bottom')
                                            
                                for rect in rects2:
                                    height = rect.get_height()
                                    if height > 0:
                                        ax.text(rect.get_x() + rect.get_width()/2., height,
                                            f'{height:.1f}' if not predict_counts else f'{int(height)}',
                                            ha='center', va='bottom')
                                
                                st.pyplot(fig)
                        
                        # Add aggregate statistics across all test samples
                        st.markdown("#### Aggregate Regime Importance")
                        
                        true_vals_test = results['true_values']['test']
                        pred_vals_test = results['predictions']['test']
                        
                        # Calculate mean values for each regime across test set
                        mean_true = np.mean(true_vals_test, axis=0)
                        mean_pred = np.mean(pred_vals_test, axis=0)
                        
                        # Create a bar chart comparing mean true vs predicted values
                        fig, ax = plt.subplots(figsize=(10, 5))
                        x = np.arange(len(mean_true))
                        width = 0.35
                        
                        rects1 = ax.bar(x - width/2, mean_true, width, label='True')
                        rects2 = ax.bar(x + width/2, mean_pred, width, label='Predicted')
                        
                        ax.set_xlabel('Regime')
                        ax.set_ylabel('Average ' + ('Count' if predict_counts else 'Probability'))
                        ax.set_title('Average Regime ' + prediction_type + ' Across Test Set')
                        ax.set_xticks(x)
                        ax.set_xticklabels([f'Regime {i}' for i in range(len(mean_true))])
                        ax.legend()
                        
                        # Add value labels on top of bars
                        for rect in rects1:
                            height = rect.get_height()
                            if height > 0:
                                ax.text(rect.get_x() + rect.get_width()/2., height,
                                    f'{height:.2f}',
                                    ha='center', va='bottom')
                                    
                        for rect in rects2:
                            height = rect.get_height()
                            if height > 0:
                                ax.text(rect.get_x() + rect.get_width()/2., height,
                                    f'{height:.2f}',
                                    ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                        # Offer hyperparameter optimization
                        st.markdown("### Hyperparameter Optimization")
                        run_optuna = st.checkbox("Run hyperparameter optimization", value=False, key="run_optuna_regime")
                        if run_optuna:
                            n_trials = st.slider("Number of Optuna trials", min_value=5, max_value=100, value=20, key="regime_optuna_trials")
                            timeout = st.slider("Timeout (seconds)", min_value=30, max_value=1800, value=300, key="regime_optuna_timeout")
                            
                            if st.button("Run Optimization", key="run_optuna_regime_button"):
                                with st.spinner("Running hyperparameter optimization..."):
                                    try:
                                        optim_results = optimize_hyperparameters_for_regime_prediction(
                                            st.session_state.current_graph,
                                            st.session_state.current_graph.community_labels,
                                            st.session_state.current_graph.node_regimes,
                                            selected_metapath,
                                            k,
                                            model_type=selected_prediction_model,
                                            feature_opts=pred_feature_opts,
                                            normalize_counts=not predict_counts,
                                            splits=results['splits'] if 'splits' in results else None,
                                            n_trials=n_trials,
                                            timeout=timeout,
                                            seed=42
                                        )
                                        
                                        st.session_state.metapath_analysis_state['regime_optuna_results'] = optim_results
                                        st.success("Optimization completed successfully!")
                                        
                                        st.markdown("#### Best Parameters")
                                        st.json(optim_results['best_params'])
                                        
                                        st.markdown("#### Performance with Best Parameters")
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Train MSE", f"{optim_results['metrics']['mse']['train']:.4f}")
                                        col2.metric("Validation MSE", f"{optim_results['metrics']['mse']['val']:.4f}")
                                        col3.metric("Test MSE", f"{optim_results['metrics']['mse']['test']:.4f}")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Train R²", f"{optim_results['metrics']['r2']['train']:.4f}")
                                        col2.metric("Validation R²", f"{optim_results['metrics']['r2']['val']:.4f}")
                                        col3.metric("Test R²", f"{optim_results['metrics']['r2']['test']:.4f}")
                                        
                                        # Visualization
                                        st.markdown("#### Optimized Model Performance")
                                        fig = visualize_regime_prediction_performance(optim_results, model_type)
                                        st.pyplot(fig)
                                        
                                    except Exception as e:
                                        st.error(f"Error during optimization: {str(e)}")
                                        import traceback
                                        st.code(traceback.format_exc())
            
            with tab5:
                st.markdown("""
                ### Understanding Metapath Analysis and K-hop Detection
                
                **What are Metapaths?**
                Metapaths are sequences of communities that frequently appear in the graph. For example, a metapath "0 → 1 → 2" means that nodes from community 0 often connect to nodes in community 1, which then connect to nodes in community 2.
                
                **The Analysis Process:**
                1. **Community Structure** tab shows:
                   - P_sub matrix: Probability of edges between communities
                   - Community graph: Visual representation of likely connections
                
                2. **Metapath Statistics** tab shows:
                   - List of most common metapaths
                   - Number of instances of each metapath
                   - Average path length and node participation
                
                3. **Node Classification** tab:
                   - Predicts which nodes participate in a selected metapath
                   - Uses node features to make predictions
                   - Shows how well the prediction works
                   
                4. **Multi-Metapath Node Classification** tab:
                   - Predicts which nodes participate in multiple metapaths
                   - Creates multi-label classifiers
                   - Compares performance of different models
                
                5. **K-hop Metapath Detection** tab:
                   - **Core Feature**: Labels nodes in a starting community based on their k-hop neighbors' feature regimes
                   - **How it works**:
                     1. Select a metapath (e.g., "0 → 1 → 2")
                     2. Choose k (hop distance) to analyze
                     3. For each node in the starting community:
                        - Find all k-hop neighbors along the metapath
                        - Analyze the feature regimes of these neighbors
                        - Label the starting node based on the distribution of regimes
                   - **Visualization**:
                     - Shows the full graph with highlighted k-hop relationships
                     - Nodes are colored by their community
                     - K-hop relationships are highlighted with bold edges
                     - Feature regime distribution is shown in a bar chart
                   - **Use Cases**:
                     - Understanding feature propagation along metapaths
                     - Identifying nodes with similar k-hop neighborhood patterns
                     - Analyzing how features spread through the network
                
                **How to Use K-hop Detection:**
                1. First, examine the metapath statistics to find interesting patterns
                2. Select a metapath that you want to analyze
                3. Choose an appropriate k value (hop distance)
                4. Run the analysis to see:
                   - Distribution of feature regimes in k-hop neighbors
                   - Visual representation of k-hop relationships
                   - Statistics about the relationships found
                5. Use the results to understand how features propagate along the metapath
                
                **Tips for Effective Analysis:**
                - Start with smaller k values (1-2) to understand direct relationships
                - Look for patterns in the feature regime distribution
                - Compare results across different metapaths
                - Use the visualization to identify clusters of similar nodes
                """)

# Add footer
st.markdown("""
---
**MMSB Explorer** | Mixed-Membership Stochastic Block Model for Graph Transfer Learning
""") 