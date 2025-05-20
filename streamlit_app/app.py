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
import io

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
from utils.metapath_analysis import (
    analyze_metapaths,
    visualize_community_metapath_graph,
    visualize_metapaths,
    metapath_node_classification,
    run_all_classifications,
    run_all_metapath_tasks,
    optimize_hyperparameters,
    multi_metapath_node_classification,
    feature_regime_metapath_classification,
    visualize_model_performance,
    GCNMultiLabel,
    GraphSAGEMultiLabel,
    prepare_node_features,
    create_consistent_train_val_test_split,
    find_metapath_instances,
    multi_metapath_node_classification_improved,
    optimize_hyperparameters_for_metapath_improved,
    khop_metapath_detection,
    visualize_khop_metapath_detection,
    visualize_regime_prediction_performance,
    optimize_hyperparameters_for_regime_prediction,
    khop_regime_prediction
)

def run_metapath_analysis(graph, theta, max_length, allow_loops, allow_backtracking, top_k, min_length):
    """Callback function to run metapath analysis"""
    try:
        # Update session state with new values
        st.session_state.metapath_analysis_state['params'].update({
            'theta': theta,
            'max_length': max_length,
            'allow_loops': allow_loops,
            'allow_backtracking': allow_backtracking,
            'top_k': top_k,
            'min_length': min_length
        })
        
        # Run the metapath analysis
        metapath_results = analyze_metapaths(
            graph,
            theta=theta,
            max_length=max_length,
            top_k=top_k,
            allow_loops=allow_loops,
            allow_backtracking=allow_backtracking,
            min_length=min_length
        )
        
        # Store results in session state
        st.session_state.metapath_analysis_state['results'] = metapath_results
        st.session_state.metapath_analysis_state['analysis_run'] = True
        
        return True
    except Exception as e:
        return False

def render_metapath_analysis(graph):
    """Render the metapath analysis section in an isolated container."""
    container = st.container()
    
    with container:
        st.markdown("#### Metapath Analysis")
        
        # Parameters outside form
        col1, col2 = st.columns(2)
        
        with col1:
            theta = st.slider(
                "Probability threshold (θ)",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.metapath_analysis_state['params']['theta'],
                help="Threshold for considering an edge likely in the community graph"
            )
            max_length = st.slider(
                "Maximum metapath length",
                min_value=2,
                max_value=5,
                value=st.session_state.metapath_analysis_state['params']['max_length'],
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
            with st.spinner("Analyzing metapaths..."):
                success = run_metapath_analysis(graph, theta, max_length, allow_loops, allow_backtracking, top_k, st.session_state.metapath_analysis_state['params']['min_length'])
                if success:
                    st.success("Metapath analysis completed successfully!")
                else:
                    st.error("Error during metapath analysis. Check console for details.")
        
        # Display results if analysis has been run
        if st.session_state.metapath_analysis_state['analysis_run'] and st.session_state.metapath_analysis_state['results'] is not None:
            results = st.session_state.metapath_analysis_state['results']
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Community Structure",
                "Metapath Statistics",
                "Node Classification",
                "Multi-Metapath Node Classification",
                "K-hop Metapath Detection",  # New tab
                "Help"
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
                st.markdown("##### Node Classification")
                st.markdown("""
                This tab performs binary classification to predict whether a node participates in the selected metapath.
                We use multiple models to compare their performance:
                - Random Forest (RF): Traditional ML model using node features
                - Multi-Layer Perceptron (MLP): Neural network using node features
                - Graph Convolutional Network (GCN): Graph neural network using node features and graph structure
                - GraphSAGE: Graph neural network using node features and graph structure with sampling
                """)
                if selected_metapath:
                    st.markdown("#### Hyperparameter Optimization (Optuna)")
                    st.markdown("**Baseline Feature Selection (RF/MLP):**")
                    col_feat1, col_feat2, col_feat3 = st.columns(3)
                    with col_feat1:
                        use_degree = st.checkbox("Use Degree", value=True, key="optuna_baseline_use_degree")
                    with col_feat2:
                        use_clustering = st.checkbox("Use Clustering Coefficient", value=True, key="optuna_baseline_use_clustering")
                    with col_feat3:
                        use_node_features = st.checkbox("Use Node Features", value=True, key="optuna_baseline_use_node_features")
                    optuna_baseline_feature_opts = {
                        'use_degree': use_degree,
                        'use_clustering': use_clustering,
                        'use_node_features': use_node_features
                    }
                    model_type = st.selectbox("Model type for optimization", ["rf", "mlp", "gcn", "sage"])
                    n_trials = st.slider("Number of Optuna trials", min_value=5, max_value=100, value=20)
                    timeout = st.slider("Timeout (seconds)", min_value=30, max_value=1800, value=300)
                    if st.button("Run Hyperparameter Optimization"):
                        with st.spinner("Optimizing hyperparameters with Optuna..."):
                            try:
                                X = prepare_node_features(
                                    st.session_state.current_graph.graph,
                                    st.session_state.current_graph.community_labels,
                                    use_degree=optuna_baseline_feature_opts['use_degree'],
                                    use_clustering=optuna_baseline_feature_opts['use_clustering'],
                                    use_node_features=optuna_baseline_feature_opts['use_node_features']
                                )
                            except ValueError as e:
                                st.error(str(e))
                                st.stop()
            
            with tab4:
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
                    
                    # Rest of multi-metapath classification code remains the same
                    # (Code omitted for brevity)
            
            # NEW TAB: K-hop Metapath Detection
            with tab5:
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
            
            with tab6:
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
    ["Universe Creation", "Graph Sampling", "Graph Family Generation", "Graph Family Analysis", "Parameter Space Analysis", "Motif and Role Analysis", "Neighborhood Analysis", "Metapath Analysis"]
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
        
        # Method-specific parameters
        st.markdown('<div class="subsection-header">Method Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.graph_params['method'] = st.selectbox(
                "Graph generation method",
                options=["Standard", "Power Law", "Exponential", "Uniform"],
                index=["Standard", "Power Law", "Exponential", "Uniform"].index(st.session_state.graph_params['method']),
                help="Method for generating the graph structure"
            )
            
            if st.session_state.graph_params['method'] == "Power Law":
                st.session_state.graph_params['method_params']['power_law_exponent'] = st.slider(
                    "Power law exponent",
                    min_value=2.0,
                    max_value=3.0,
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
                st.session_state.graph_params['method_params']['degree_heterogeneity'] = st.slider(
                    "Degree heterogeneity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.graph_params['method_params']['degree_heterogeneity'],
                step=0.1,
                help="Amount of degree heterogeneity to introduce"
                )
                st.session_state.graph_params['method_params']['edge_noise'] = st.slider(
                    "Edge noise",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.graph_params['method_params']['edge_noise'],
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
                # Sample communities
                if st.session_state.graph_params['sampling_method'] == "connected":
                    communities = st.session_state.universe.sample_connected_community_subset(
                        size=st.session_state.graph_params['num_communities']
                    )
                else:
                    communities = st.session_state.universe.sample_random_community_subset(
                        size=st.session_state.graph_params['num_communities'],
                        method=st.session_state.graph_params['sampling_method']
                    )
                
                # Build config_model_params for method-specific parameters
                config_model_params = {}
                if st.session_state.graph_params['method'] == "Power Law":
                    config_model_params['power_law_exponent'] = st.session_state.graph_params['method_params'].get('power_law_exponent')
                if st.session_state.graph_params['method'] == "Exponential":
                    config_model_params['rate'] = st.session_state.graph_params['method_params'].get('rate')
                if st.session_state.graph_params['method'] == "Uniform":
                    config_model_params['min_factor'] = st.session_state.graph_params['method_params'].get('min_factor')
                    config_model_params['max_factor'] = st.session_state.graph_params['method_params'].get('max_factor')
                
                # Set up universal parameters for GraphSample
                universal_params = dict(
                    universe=st.session_state.universe,
                    communities=communities,
                    n_nodes=st.session_state.graph_params['n_nodes'],
                    min_component_size=st.session_state.graph_params['min_component_size'],
                    degree_heterogeneity=st.session_state.graph_params['method_params'].get('degree_heterogeneity', 0.0),
                    edge_noise=st.session_state.graph_params['method_params'].get('edge_noise', 0.0),
                    feature_regime_balance=0.5,
                    target_homophily=None,
                    target_density=None,
                    use_configuration_model=(st.session_state.graph_params['method'] != "Standard"),
                    degree_distribution=(
                        "power_law" if st.session_state.graph_params['method'] == "Power Law"
                        else "exponential" if st.session_state.graph_params['method'] == "Exponential"
                        else "uniform" if st.session_state.graph_params['method'] == "Uniform"
                        else None
                    ),
                    power_law_exponent=st.session_state.graph_params['method_params'].get('power_law_exponent'),
                    target_avg_degree=st.session_state.graph_params['method_params'].get('target_avg_degree'),
                    triangle_enhancement=st.session_state.graph_params['method_params'].get('triangle_enhancement', 0.0),
                    max_mean_community_deviation=st.session_state.graph_params['max_mean_community_deviation'],
                    max_max_community_deviation=st.session_state.graph_params['max_max_community_deviation'],
                    max_parameter_search_attempts=st.session_state.graph_params['max_parameter_search_attempts'],
                    parameter_search_range=st.session_state.graph_params['parameter_search_range'],
                    min_edge_density=st.session_state.graph_params['min_edge_density'],
                    max_retries=st.session_state.graph_params['max_retries'],
                    seed=st.session_state.graph_params['seed'],
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
                fig = plot_degree_distribution(st.session_state.current_graph.graph)
                st.pyplot(fig)
                
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
    Generate families of graphs with controlled properties. <b>All graphs in a family will use the same distribution type.</b> Each family will have:
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
    
    # --- NEW: Single distribution type selection ---
    st.markdown('<div class="subsection-header">Distribution Type</div>', unsafe_allow_html=True)
    dist_type = st.selectbox(
        "Select distribution type for this family",
        ["Standard", "Power Law", "Exponential", "Uniform"],
        help="All graphs in this family will use the selected distribution type."
    )
    
    # --- Only show relevant method-specific parameters ---
    if dist_type == "Standard":
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
        method_params = {
            'degree_heterogeneity': degree_heterogeneity,
            'edge_noise': edge_noise
        }
        config_model_params = {}
    elif dist_type == "Power Law":
        st.markdown("#### Power Law Parameters")
        col1, col2 = st.columns(2)
        with col1:
            power_law_exponent_min = st.slider("Min power law exponent", min_value=1.0, max_value=4.0, value=1.5)
            power_law_exponent_max = st.slider("Max power law exponent", min_value=1.0, max_value=4.0, value=3.0)
        with col2:
            target_avg_degree_min = st.slider("Min target average degree", min_value=1.0, max_value=50.0, value=2.0)
            target_avg_degree_max = st.slider("Max target average degree", min_value=1.0, max_value=50.0, value=5.0)
        method_params = {
            'degree_heterogeneity': 0.0,
            'edge_noise': 0.0
        }
        config_model_params = {
            "power_law": {
                "exponent_min": power_law_exponent_min,
                "exponent_max": power_law_exponent_max,
                "target_avg_degree_min": target_avg_degree_min,
                "target_avg_degree_max": target_avg_degree_max
            }
        }
    elif dist_type == "Exponential":
        st.markdown("#### Exponential Parameters")
        col1, col2 = st.columns(2)
        with col1:
            rate_min = st.slider("Min rate", min_value=0.1, max_value=2.0, value=0.1)
            rate_max = st.slider("Max rate", min_value=0.1, max_value=2.0, value=1.0)
        with col2:
            target_avg_degree_min = st.slider("Min target average degree", min_value=1.0, max_value=50.0, value=2.0)
            target_avg_degree_max = st.slider("Max target average degree", min_value=1.0, max_value=50.0, value=5.0)
        method_params = {
            'degree_heterogeneity': 0.0,
            'edge_noise': 0.0
        }
        config_model_params = {
            "exponential": {
                "rate_min": rate_min,
                "rate_max": rate_max,
                "target_avg_degree_min": target_avg_degree_min,
                "target_avg_degree_max": target_avg_degree_max
            }
        }
    elif dist_type == "Uniform":
        st.markdown("#### Uniform Parameters")
        col1, col2 = st.columns(2)
        with col1:
            min_factor = st.slider("Min factor", min_value=0.1, max_value=1.0, value=0.5)
            max_factor = st.slider("Max factor", min_value=1.0, max_value=2.0, value=1.5)
        with col2:
            target_avg_degree_min = st.slider("Min target average degree", min_value=1.0, max_value=50.0, value=2.0)
            target_avg_degree_max = st.slider("Max target average degree", min_value=1.0, max_value=50.0, value=5.0)
        method_params = {
            'degree_heterogeneity': 0.0,
            'edge_noise': 0.0
        }
        config_model_params = {
            "uniform": {
                "min_factor": min_factor,
                "max_factor": max_factor,
                "target_avg_degree_min": target_avg_degree_min,
                "target_avg_degree_max": target_avg_degree_max
            }
        }
    # --- Set method_distribution for single type ---
    method_distribution = {
        dist_type.lower().replace(" ", "_"): 1.0
    }
    
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
            standard_method_params=method_params if dist_type == "Standard" else {},
            config_model_params=config_model_params,
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
                    graph_obj = new_graphs[0]
                    # If it's a tuple, extract the first element
                    if isinstance(graph_obj, tuple):
                        graph_obj = graph_obj[0]
                    graph_family.append(graph_obj)
                    
                    # Store parameter samples
                    params = graph_obj.extract_parameters()
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
            "Average nodes": np.mean([g[0].n_nodes if isinstance(g, tuple) else g.n_nodes for g in st.session_state.current_family_graphs]),
            "Average edges": np.mean([g[0].graph.number_of_edges() if isinstance(g, tuple) else g.graph.number_of_edges() for g in st.session_state.current_family_graphs]),
            "Average density": np.mean([nx.density(g[0].graph) if isinstance(g, tuple) else nx.density(g.graph) for g in st.session_state.current_family_graphs]),
            "Average clustering": np.mean([nx.average_clustering(g[0].graph) if isinstance(g, tuple) else nx.average_clustering(g.graph) for g in st.session_state.current_family_graphs])
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
            g = st.session_state.current_family_graphs[graph_idx]
            if isinstance(g, tuple):
                g = g[0]
            # Get generation method and parameters for title
            generation_info = "Method: Unknown"
            if hasattr(g, 'generation_method'):
                method = g.generation_method
                params = g.generation_params
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
            fig = plot_graph_communities(g)
            st.pyplot(fig)
        
        with tab2:
            graph_idx = st.selectbox(
                "Select graph for degree distribution",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_degree_{family_name}"
            )
            g = st.session_state.current_family_graphs[graph_idx]
            if isinstance(g, tuple):
                g = g[0]
            generation_info = "Method: Unknown"
            if hasattr(g, 'generation_method'):
                method = g.generation_method
                params = g.generation_params
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
            fig = plot_degree_distribution(g.graph)
            st.pyplot(fig)
        
        with tab3:
            graph_idx = st.selectbox(
                "Select graph for community analysis",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_community_{family_name}"
            )
            g = st.session_state.current_family_graphs[graph_idx]
            if isinstance(g, tuple):
                g = g[0]
            generation_info = "Method: Unknown"
            if hasattr(g, 'generation_method'):
                method = g.generation_method
                params = g.generation_params
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
            fig = plot_membership_matrix(g)
            st.pyplot(fig)
        
        with tab4:
            graph_idx = st.selectbox(
                "Select graph for deviation analysis",
                range(len(st.session_state.current_family_graphs)),
                key=f"family_deviation_{family_name}"
            )
            g = st.session_state.current_family_graphs[graph_idx]
            if isinstance(g, tuple):
                g = g[0]
            generation_info = "Method: Unknown"
            if hasattr(g, 'generation_method'):
                method = g.generation_method
                params = g.generation_params
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
            connection_analysis = g.analyze_community_connections()
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
                    fig = plt.figure(figsize=(6, 5))
                    plt.imshow(connection_analysis["deviation_matrix"], cmap='Reds')
                    plt.colorbar(label='Absolute Deviation')
                    plt.title('Deviation from Expected Probabilities')
                    plt.xlabel('Community')
                    plt.ylabel('Community')
                    st.pyplot(fig)
                with col2:
                    st.markdown("##### Deviation Statistics")
                    st.metric("Mean Absolute Deviation", f"{connection_analysis['mean_deviation']:.4f}")
                    st.metric("Maximum Absolute Deviation", f"{connection_analysis['max_deviation']:.4f}")
                    fig = plt.figure(figsize=(6, 4))
                    plt.hist(connection_analysis["deviation_matrix"].flatten(), bins=20)
                    plt.title('Distribution of Deviations')
                    plt.xlabel('Absolute Deviation')
                    plt.ylabel('Count')
                    st.pyplot(fig)
                if "degree_analysis" in connection_analysis:
                    st.markdown("##### Degree Distribution Analysis")
                    degree_analysis = connection_analysis["degree_analysis"]
                    used_params = degree_analysis["used_parameters"]
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
                    st.markdown("##### Community Sizes")
                    fig = plt.figure(figsize=(6, 4))
                    plt.bar(range(len(connection_analysis["community_sizes"])), 
                           connection_analysis["community_sizes"])
                    plt.title('Number of Nodes per Community')
                    plt.xlabel('Community')
                    plt.ylabel('Number of Nodes')
                    st.pyplot(fig)
                with col2:
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

    with tab3:
        st.subheader("Parameter Analysis")
        fig = create_parameter_dashboard(graph.extract_parameters())
        st.pyplot(fig)

    with tab4:
        st.subheader("Feature Analysis")
        fig = plot_community_statistics(graph)
        st.pyplot(fig)