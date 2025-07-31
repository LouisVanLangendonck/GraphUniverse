import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphSample
from graph_universe.graph_family import GraphFamilyGenerator, FamilyConsistencyAnalyzer
from utils.visualizations import (
    plot_graph_communities, 
    create_dashboard,
    plot_universe_cooccurrence_matrix,
    plot_universe_degree_centers,
    plot_universe_summary
)

# Set page config
st.set_page_config(
    page_title="Graph Family Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .parameter-group {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">Graph Family Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    with st.sidebar:
        st.markdown('<h2 class="section-header">Parameters</h2>', unsafe_allow_html=True)
        
        # Universe Parameters
        st.markdown("### Graph Universe Parameters")
        
        K = st.slider("Number of Communities (K)", 2, 20, 8, help="Number of communities in the universe")
        
        feature_dim = st.slider("Feature Dimension", 0, 50, 10, help="Dimension of node features (0 for no features)")
        
        inter_community_variance = st.slider(
            "Inter-Community Variance", 
            0.0, 1.0, 0.1, 0.01,
            help="Amount of variance in inter-community probabilities"
        )
        
        # Feature generation parameters
        if feature_dim > 0:
            st.markdown("#### Feature Generation Parameters")
            
            cluster_count_factor = st.slider(
                "Cluster Count Factor", 
                0.1, 4.0, 1.0, 0.1,
                help="Number of clusters relative to communities"
            )
            
            center_variance = st.slider(
                "Center Variance", 
                0.1, 5.0, 1.0, 0.1,
                help="Separation between cluster centers"
            )
            
            cluster_variance = st.slider(
                "Cluster Variance", 
                0.01, 1.0, 0.1, 0.01,
                help="Spread within each cluster"
            )
            
            assignment_skewness = st.slider(
                "Assignment Skewness", 
                0.0, 1.0, 0.0, 0.01,
                help="If some clusters are used more frequently"
            )
            
            community_exclusivity = st.slider(
                "Community Exclusivity", 
                0.0, 1.0, 1.0, 0.01,
                help="How exclusively clusters map to communities"
            )
        else:
            cluster_count_factor = 1.0
            center_variance = 1.0
            cluster_variance = 0.1
            assignment_skewness = 0.0
            community_exclusivity = 1.0
        
        # Degree center parameters
        degree_center_method = st.selectbox(
            "Degree Center Method",
            ["linear", "random", "shuffled"],
            help="How to generate degree centers"
        )
        
        community_cooccurrence_homogeneity = st.slider(
            "Community Co-occurrence Homogeneity",
            0.0, 1.0, 1.0, 0.01,
            help="Controls community co-occurrence patterns"
        )
        
        # Graph Family Parameters
        st.markdown("### Graph Family Parameters")
        
        min_n_nodes = st.slider("Min Nodes", 50, 500, 100, help="Minimum number of nodes per graph")
        max_n_nodes = st.slider("Max Nodes", 100, 1000, 300, help="Maximum number of nodes per graph")
        
        min_communities = st.slider("Min Communities", 2, 10, 3, help="Minimum number of communities per graph")
        max_communities = st.slider("Max Communities", 3, 15, 8, help="Maximum number of communities per graph")
        
        min_component_size = st.slider("Min Component Size", 0, 50, 10, help="Minimum size of connected components")
        
        homophily_min, homophily_max = st.slider(
            "Homophily Range",
            0.0, 1.0, (0.1, 0.4),
            help="Range for target homophily"
        )
        
        avg_degree_min, avg_degree_max = st.slider(
            "Average Degree Range",
            1.0, 100.0, (1.0, 5.0),
            help="Range for target average degree"
        )
        
        # DCCC-SBM parameters
        use_dccc_sbm = st.checkbox("Use DCCC-SBM", value=True, help="Use Degree-Community-Coupled Corrected SBM")
        
        if use_dccc_sbm:
            st.markdown("#### DCCC-SBM Parameters")
            
            degree_distribution = st.selectbox(
                "Degree Distribution",
                ["power_law", "exponential", "uniform"],
                help="Type of degree distribution"
            )
            
            if degree_distribution == "power_law":
                power_law_min, power_law_max = st.slider(
                    "Power Law Exponent Range",
                    1.5, 5.0, (2.0, 3.5),
                    help="Range for power law exponent"
                )
            else:
                power_law_min, power_law_max = 2.5, 2.5
                
            if degree_distribution == "exponential":
                exp_min, exp_max = st.slider(
                    "Exponential Rate Range",
                    0.1, 2.0, (0.3, 1.0),
                    help="Range for exponential rate"
                )
            else:
                exp_min, exp_max = 0.5, 0.5
                
            if degree_distribution == "uniform":
                uniform_min_min, uniform_min_max = st.slider(
                    "Uniform Min Factor Range",
                    0.1, 1.0, (0.3, 0.7),
                    help="Range for uniform min factor"
                )
                uniform_max_min, uniform_max_max = st.slider(
                    "Uniform Max Factor Range",
                    1.0, 3.0, (1.3, 2.0),
                    help="Range for uniform max factor"
                )
            else:
                uniform_min_min, uniform_min_max = 0.5, 0.5
                uniform_max_min, uniform_max_max = 1.5, 1.5
                
            degree_separation_min, degree_separation_max = st.slider(
                "Degree Separation Range",
                0.0, 1.0, (0.5, 1.0),
                help="Range for degree separation"
            )
        else:
            degree_distribution = "standard"
            power_law_min, power_law_max = 2.5, 2.5
            exp_min, exp_max = 0.5, 0.5
            uniform_min_min, uniform_min_max = 0.5, 0.5
            uniform_max_min, uniform_max_max = 1.5, 1.5
            degree_separation_min, degree_separation_max = 0.5, 0.5
        
        # Standard DC-SBM parameters
        degree_heterogeneity = st.slider(
            "Degree Heterogeneity",
            0.0, 1.0, 0.5,
            help="Controls degree variability for standard DC-SBM"
        )
        
        # Deviation limiting parameters
        disable_deviation_limiting = st.checkbox("Disable Deviation Limiting", value=False)
        
        max_mean_community_deviation = st.slider(
            "Max Mean Community Deviation",
            0.01, 0.5, 0.10,
            help="Maximum allowed mean deviation from expected community patterns"
        )
        
        max_max_community_deviation = st.slider(
            "Max Max Community Deviation",
            0.01, 0.5, 0.15,
            help="Maximum allowed maximum deviation from expected community patterns"
        )
        
        min_edge_density = st.slider(
            "Min Edge Density",
            0.001, 0.1, 0.001,
            help="Minimum acceptable edge density"
        )
        
        max_retries = st.slider("Max Retries", 1, 20, 5, help="Maximum retries for graph generation")
        
        # Generation parameters
        st.markdown("### Generation Parameters")
        
        n_graphs = st.slider("Number of Graphs", 1, 100, 5, help="Number of graphs to generate")
        
        seed = st.number_input("Random Seed", value=42, help="Random seed for reproducibility")
        
        timeout_minutes = st.slider("Timeout (minutes)", 1.0, 30.0, 5.0, help="Timeout for generation")
        
        max_attempts_per_graph = st.slider("Max Attempts per Graph", 1, 50, 10, help="Maximum attempts per graph")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Graph Family Generation</h2>', unsafe_allow_html=True)
        
        if st.button("Generate Graph Family", type="primary"):
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            stats_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with status_container:
                status_text = st.empty()
            
            with stats_container:
                stats_text = st.empty()
        
            # Step 1: Create universe
            status_text.text("Step 1/4: Creating Graph Universe...")
            progress_bar.progress(0)
            progress_text.text("0% - Initializing...")
            
            universe = GraphUniverse(
                K=K,
                feature_dim=feature_dim,
                inter_community_variance=inter_community_variance,
                cluster_count_factor=cluster_count_factor,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
                assignment_skewness=assignment_skewness,
                community_exclusivity=community_exclusivity,
                degree_center_method=degree_center_method,
                community_cooccurrence_homogeneity=community_cooccurrence_homogeneity,
                seed=seed
            )
            
            progress_bar.progress(25)
            progress_text.text("25% - Universe created")
            
            # Step 2: Create family generator
            status_text.text("Step 2/4: Setting up Graph Family Generator...")
            family_generator = GraphFamilyGenerator(
                universe=universe,
                min_n_nodes=min_n_nodes,
                max_n_nodes=max_n_nodes,
                min_communities=min_communities,
                max_communities=max_communities,
                min_component_size=min_component_size,
                homophily_range=(homophily_min, homophily_max),
                avg_degree_range=(avg_degree_min, avg_degree_max),
                use_dccc_sbm=use_dccc_sbm,
                community_cooccurrence_homogeneity=community_cooccurrence_homogeneity,
                disable_deviation_limiting=disable_deviation_limiting,
                max_mean_community_deviation=max_mean_community_deviation,
                max_max_community_deviation=max_max_community_deviation,
                min_edge_density=min_edge_density,
                degree_distribution=degree_distribution,
                power_law_exponent_range=(power_law_min, power_law_max),
                exponential_rate_range=(exp_min, exp_max),
                uniform_min_factor_range=(uniform_min_min, uniform_min_max),
                uniform_max_factor_range=(uniform_max_min, uniform_max_max),
                degree_separation_range=(degree_separation_min, degree_separation_max),
                degree_heterogeneity=degree_heterogeneity,
                max_retries=max_retries,
                seed=seed
            )
            
            progress_bar.progress(50)
            progress_text.text("50% - Family generator ready")
            
            # Step 3: Generate family with simple progress tracking
            status_text.text("Step 3/4: Generating Graph Family...")
            
            start_time = time.time()
            
            # Use the family generator's generate_family method
            graphs = family_generator.generate_family(
                n_graphs=n_graphs,
                show_progress=False,  # We'll handle progress manually
                collect_stats=True,
                max_attempts_per_graph=max_attempts_per_graph,
                timeout_minutes=timeout_minutes
            )
            
            # Update progress
            progress_bar.progress(90)
            progress_text.text("90% - Family generation complete")
            status_text.text("Family generation completed successfully!")
            
            # except Exception as e:
            #     st.error(f"Family generation failed: {str(e)}")
            #     st.exception(e)
            #     return
            
            # Step 4: Finalize
            status_text.text("Step 4/4: Finalizing...")
            progress_bar.progress(100)
            progress_text.text("100% - Complete!")
            
            # Final stats
            elapsed_total = time.time() - start_time
            success_rate = len(graphs) / n_graphs * 100
            
            st.success(f"✅ Generation complete! Generated {len(graphs)}/{n_graphs} graphs "
                        f"({success_rate:.1f}% success rate) in {elapsed_total:.1f}s")
            
            # Display generation stats if available
            if hasattr(family_generator, 'generation_stats') and family_generator.generation_stats:
                stats = family_generator.generation_stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Time", f"{stats.get('total_time', 0):.2f}s")
                with col2:
                    st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
                with col3:
                    st.metric("Avg Time per Graph", f"{stats.get('avg_time_per_graph', 0):.2f}s")
            
            # Store in session state
            st.session_state.graphs = graphs
            st.session_state.universe = universe
            st.session_state.family_generator = family_generator
            
            # Debug information
            st.info(f"📊 Debug Info: Family generator has {len(family_generator.graphs)} graphs stored")
            if len(family_generator.graphs) == 0:
                st.warning("⚠️ Warning: No graphs were stored in the family generator. This may cause analysis issues.")
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            stats_container.empty()
    
    with col2:
        st.markdown('<h2 class="section-header">Quick Stats</h2>', unsafe_allow_html=True)
        
        if 'graphs' in st.session_state and st.session_state.graphs:
            graphs = st.session_state.graphs
            
            # Basic stats
            total_nodes = sum(g.n_nodes for g in graphs)
            avg_nodes = total_nodes / len(graphs)
            total_edges = sum(g.graph.number_of_edges() for g in graphs)
            avg_edges = total_edges / len(graphs)
            
            st.metric("Total Graphs", len(graphs))
            st.metric("Avg Nodes per Graph", f"{avg_nodes:.1f}")
            st.metric("Avg Edges per Graph", f"{avg_edges:.1f}")
            st.metric("Total Nodes", total_nodes)
            st.metric("Total Edges", total_edges)
            
            # Community stats
            all_communities = set()
            for g in graphs:
                all_communities.update(g.communities)
            
            st.metric("Unique Communities Used", len(all_communities))
    
    # Visualization section
    if 'graphs' in st.session_state and st.session_state.graphs:
        st.markdown('<h2 class="section-header">Visualization</h2>', unsafe_allow_html=True)
        
        graphs = st.session_state.graphs
        
        # Graph selection
        graph_idx = st.selectbox(
            "Select Graph to Visualize",
            range(len(graphs)),
            format_func=lambda x: f"Graph {x+1} ({graphs[x].n_nodes} nodes, {graphs[x].graph.number_of_edges()} edges)"
        )
        
        selected_graph = graphs[graph_idx]
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout = st.selectbox("Layout", ["spring", "kamada_kawai", "spectral", "circular"])
        
        with col2:
            node_size = st.slider("Node Size", 10, 200, 50)
        
        with col3:
            show_labels = st.checkbox("Show Node Labels", value=False)
        
        # Create visualization
        fig = plot_graph_communities(
            selected_graph,
            layout=layout,
            node_size=node_size,
            with_labels=show_labels,
            figsize=(12, 10),
            title=f"Graph {graph_idx+1} - {selected_graph.n_nodes} nodes, {selected_graph.graph.number_of_edges()} edges"
        )
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Graph statistics
        st.markdown('<h3 class="section-header">Graph Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", selected_graph.n_nodes)
            st.metric("Edges", selected_graph.graph.number_of_edges())
            st.metric("Density", f"{selected_graph.graph.number_of_edges() / (selected_graph.n_nodes * (selected_graph.n_nodes - 1) / 2):.4f}")
        
        with col2:
            st.metric("Communities", len(selected_graph.communities))
            st.metric("Components", len(list(nx.connected_components(selected_graph.graph))))
            st.metric("Avg Degree", f"{2 * selected_graph.graph.number_of_edges() / selected_graph.n_nodes:.2f}")
        
        with col3:
            if selected_graph.features is not None:
                st.metric("Feature Dim", selected_graph.features.shape[1])
                st.metric("Has Features", "Yes")
            else:
                st.metric("Feature Dim", 0)
                st.metric("Has Features", "No")
            
            st.metric("Generation Method", selected_graph.generation_method)
        
        with col4:
            # Calculate some additional metrics
            degrees = [d for _, d in selected_graph.graph.degree()]
            st.metric("Min Degree", min(degrees))
            st.metric("Max Degree", max(degrees))
            st.metric("Std Degree", f"{np.std(degrees):.2f}")
        
        # Community analysis
        if st.checkbox("Show Community Analysis"):
            st.markdown('<h3 class="section-header">Community Analysis</h3>', unsafe_allow_html=True)
            
            # Community sizes
            community_sizes = {}
            for label in selected_graph.community_labels:
                comm_id = selected_graph.community_id_mapping[label]
                if comm_id not in community_sizes:
                    community_sizes[comm_id] = 0
                community_sizes[comm_id] += 1
            
            # Create community size plot
            fig, ax = plt.subplots(figsize=(10, 6))
            communities = list(community_sizes.keys())
            sizes = list(community_sizes.values())
            
            bars = ax.bar(range(len(communities)), sizes, color='skyblue', alpha=0.7)
            ax.set_xlabel("Community ID")
            ax.set_ylabel("Number of Nodes")
            ax.set_title("Community Size Distribution")
            ax.set_xticks(range(len(communities)))
            ax.set_xticklabels([f"C{c}" for c in communities])
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(size), ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close(fig)
        
        # Family summary
        if st.checkbox("Show Family Summary"):
            st.markdown('<h3 class="section-header">Family Summary</h3>', unsafe_allow_html=True)
            
            # Check if family generator exists in session state
            if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                # Analyze basic properties
                family_properties = st.session_state.family_generator.analyze_graph_family_properties()
                st.write(family_properties)
            else:
                st.warning("No graph family has been generated yet. Please generate a graph family first.")

        # Consistency
        if st.checkbox("Show Consistency"):
            st.markdown('<h3 class="section-header">Consistency</h3>', unsafe_allow_html=True)
            
            # Check if family generator exists in session state
            if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                # Analyze consistency
                consistency = st.session_state.family_generator.analyze_graph_family_consistency()
                # Go over each consistency metric and if a list calc mean and std.
                for metric, value in consistency.items():
                    if isinstance(value, list):
                        st.metric(metric, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
                    else:
                        st.metric(metric, value)
            else:
                st.warning("No graph family has been generated yet. Please generate a graph family first.")

        # Signals
        if st.checkbox("Show Signals"):
            st.markdown('<h3 class="section-header">Signals</h3>', unsafe_allow_html=True)
            
            # Check if family generator exists in session state
            if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                signals = st.session_state.family_generator.analyze_graph_family_signals()
                
                # Now go over each signal and if a list calc mean and std.
                for signal, value in signals.items():
                    if isinstance(value, list):
                        st.metric(signal, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
                    else:
                        st.metric(signal, value)
            else:
                st.warning("No graph family has been generated yet. Please generate a graph family first.")

        # Universe visualization
        if st.checkbox("Show Universe Properties"):
            st.markdown('<h3 class="section-header">Universe Properties</h3>', unsafe_allow_html=True)
            
            # Check if universe exists in session state
            if 'universe' in st.session_state and st.session_state.universe is not None:
                universe = st.session_state.universe
                
                # Universe visualization options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    show_cooccurrence = st.checkbox("Show Co-occurrence Matrix", value=True)
                
                with col2:
                    show_degree_centers = st.checkbox("Show Degree Centers", value=True)
                
                with col3:
                    show_summary = st.checkbox("Show Universe Summary", value=False)
                
                # Create visualizations
                if show_cooccurrence:
                    st.markdown("#### Community Co-occurrence Matrix")
                    fig_cooccurrence = plot_universe_cooccurrence_matrix(universe)
                    st.pyplot(fig_cooccurrence)
                    plt.close(fig_cooccurrence)
                
                if show_degree_centers:
                    st.markdown("#### Degree Centers")
                    fig_degree_centers = plot_universe_degree_centers(universe)
                    st.pyplot(fig_degree_centers)
                    plt.close(fig_degree_centers)
                
                if show_summary:
                    st.markdown("#### Universe Summary")
                    fig_summary = plot_universe_summary(universe)
                    st.pyplot(fig_summary)
                    plt.close(fig_summary)
                
                # Universe statistics
                st.markdown("#### Universe Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Number of Communities", universe.K)
                    st.metric("Feature Dimension", universe.feature_dim)
                
                with col2:
                    st.metric("Inter-Community Variance", f"{universe.inter_community_variance:.3f}")
                    st.metric("Co-occurrence Homogeneity", f"{universe.community_cooccurrence_homogeneity:.3f}")
                
                with col3:
                    # Calculate degree center statistics
                    degree_centers = universe.degree_centers
                    st.metric("Min Degree Center", f"{np.min(degree_centers):.3f}")
                    st.metric("Max Degree Center", f"{np.max(degree_centers):.3f}")
                
                with col4:
                    st.metric("Mean Degree Center", f"{np.mean(degree_centers):.3f}")
                    st.metric("Std Degree Center", f"{np.std(degree_centers):.3f}")
                
                # Co-occurrence matrix statistics
                st.markdown("#### Co-occurrence Matrix Statistics")
                cooccurrence_matrix = universe.community_cooccurrence_matrix
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Min Co-occurrence", f"{np.min(cooccurrence_matrix):.3f}")
                    st.metric("Max Co-occurrence", f"{np.max(cooccurrence_matrix):.3f}")
                
                with col2:
                    st.metric("Mean Co-occurrence", f"{np.mean(cooccurrence_matrix):.3f}")
                    st.metric("Std Co-occurrence", f"{np.std(cooccurrence_matrix):.3f}")
                
                with col3:
                    # Diagonal vs off-diagonal statistics
                    diagonal_values = np.diag(cooccurrence_matrix)
                    off_diagonal_values = cooccurrence_matrix[np.triu_indices_from(cooccurrence_matrix, k=1)]
                    st.metric("Mean Diagonal", f"{np.mean(diagonal_values):.3f}")
                    st.metric("Mean Off-Diagonal", f"{np.mean(off_diagonal_values):.3f}")
                
                with col4:
                    st.metric("Diagonal Std", f"{np.std(diagonal_values):.3f}")
                    st.metric("Off-Diagonal Std", f"{np.std(off_diagonal_values):.3f}")
                
            else:
                st.warning("No universe has been created yet. Please generate a graph family first.")

if __name__ == "__main__":
    main() 