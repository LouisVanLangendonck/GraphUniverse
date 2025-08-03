import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
from scipy.stats import ks_2samp
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphSample, GraphFamilyGenerator
from utils.visualizations import (
    plot_graph_communities, 
    create_dashboard,
    plot_universe_cooccurrence_matrix,
    plot_universe_degree_centers,
    plot_universe_summary
)

# Add function to generate theoretical power law distribution
def generate_theoretical_power_law(n_nodes: int, exponent: float, x_min: float = 1.0) -> np.ndarray:
    """
    Generate theoretical power law distribution.
    
    Args:
        n_nodes: Number of nodes
        exponent: Power law exponent (alpha)
        x_min: Minimum value for power law
        
    Returns:
        Array of theoretical degrees
    """
    # Generate theoretical power law distribution
    theoretical_degrees = np.random.pareto(exponent, size=n_nodes) + x_min
    return theoretical_degrees

def plot_probability_matrices_comparison(graph_sample, figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Plot target (P_sub) and actual (P_real) probability matrices side by side with same color scale.
    
    Args:
        graph_sample: GraphSample object
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Get the target probability matrix (P_sub)
    P_sub = graph_sample.P_sub
    
    # Get the actual probability matrix
    P_real, community_sizes, connection_counts = graph_sample.calculate_actual_probability_matrix()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Find the global min and max for consistent color scaling
    vmin = min(np.min(P_sub), np.min(P_real))
    vmax = max(np.max(P_sub), np.max(P_real))
    
    # Plot target probability matrix (P_sub)
    im1 = ax1.imshow(P_sub, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('Target Probability Matrix (P_sub)')
    ax1.set_xlabel('Community')
    ax1.set_ylabel('Community')
    
    # Add colorbar for the first plot
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Probability')
    
    # Add text annotations to the first plot
    for i in range(P_sub.shape[0]):
        for j in range(P_sub.shape[1]):
            # Adaptive text color based on background
            value = P_sub[i, j]
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            text = ax1.text(j, i, f'{value:.3f}',
                           ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
    
    # Plot actual probability matrix (P_real)
    im2 = ax2.imshow(P_real, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('Actual Probability Matrix (P_real)')
    ax2.set_xlabel('Community')
    ax2.set_ylabel('Community')
    
    # Add colorbar for the second plot
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Probability')
    
    # Add text annotations to the second plot
    for i in range(P_real.shape[0]):
        for j in range(P_real.shape[1]):
            # Adaptive text color based on background
            value = P_real[i, j]
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            text = ax2.text(j, i, f'{value:.3f}',
                           ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
    
    # Set tick labels
    n_communities = len(graph_sample.communities)
    tick_labels = [f'C{i}' for i in range(n_communities)]
    ax1.set_xticks(range(n_communities))
    ax1.set_yticks(range(n_communities))
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticklabels(tick_labels)
    
    ax2.set_xticks(range(n_communities))
    ax2.set_yticks(range(n_communities))
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticklabels(tick_labels)
    
    # Calculate and display statistics
    deviation = np.abs(P_real - P_sub)
    mean_deviation = np.mean(deviation)
    max_deviation = np.max(deviation)
    
    # Add statistics as text
    stats_text = f'Mean Deviation: {mean_deviation:.4f}\nMax Deviation: {max_deviation:.4f}'
    fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_degree_distribution_comparison(actual_degrees: np.ndarray, theoretical_degrees: np.ndarray, 
                                      exponent: float, title: str = "Degree Distribution Comparison"):
    """
    Plot comparison between actual and theoretical degree distributions.
    
    Args:
        actual_degrees: Actual degrees from the graph
        theoretical_degrees: Theoretical degrees from power law
        exponent: Power law exponent used
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram comparison
    ax1.hist(actual_degrees, bins=30, alpha=0.7, label='Actual', density=True, color='blue')
    ax1.hist(theoretical_degrees, bins=30, alpha=0.7, label='Theoretical', density=True, color='red')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Degree Distribution Histogram (α={exponent:.2f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log plot for power law verification
    # Sort degrees for CCDF
    actual_sorted = np.sort(actual_degrees)[::-1]  # Descending
    theoretical_sorted = np.sort(theoretical_degrees)[::-1]  # Descending
    
    # Calculate CCDF (Complementary Cumulative Distribution Function)
    n_actual = len(actual_sorted)
    n_theoretical = len(theoretical_sorted)
    
    actual_ccdf = np.arange(1, n_actual + 1) / n_actual
    theoretical_ccdf = np.arange(1, n_theoretical + 1) / n_theoretical
    
    ax2.loglog(actual_sorted, actual_ccdf, 'o-', label='Actual', alpha=0.7, markersize=4)
    ax2.loglog(theoretical_sorted, theoretical_ccdf, 's-', label='Theoretical', alpha=0.7, markersize=4)
    
    # Add theoretical power law line
    x_range = np.logspace(np.log10(min(actual_sorted.min(), theoretical_sorted.min())), 
                          np.log10(max(actual_sorted.max(), theoretical_sorted.max())), 100)
    ccdf_theoretical = (x_range / x_range.min()) ** (-exponent + 1)
    ax2.loglog(x_range, ccdf_theoretical, '--', label=f'Theoretical α={exponent:.2f}', alpha=0.8)
    
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('CCDF (log scale)')
    ax2.set_title('Power Law Verification (Log-Log Plot)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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
        
        edge_probability_variance = st.slider(
            "Edge Probability Variance", 
            0.0, 1.0, 0.5, 0.01,
            help="Amount of variance in edge probabilities"
        )
        
        # Feature generation parameters
        if feature_dim > 0:
            st.markdown("#### Feature Generation Parameters")
            
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
            
        else:
            center_variance = 1.0
            cluster_variance = 0.1
        
        # Degree center parameters
        degree_center_method = st.selectbox(
            "Degree Center Method",
            ["linear", "random", "constant"],
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
                    1.5, 5.0, (2.0, 2.5),
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
                0.0, 1.0, (0.7, 1.0),
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
            0.01, 0.5, 0.05,
            help="Maximum allowed mean deviation from expected community patterns"
        )
        
        min_edge_density = st.slider(
            "Min Edge Density",
            0.001, 0.1, 0.0005,
            help="Minimum acceptable edge density"
        )
        
        max_retries = st.slider("Max Retries", 1, 20, 5, help="Maximum retries for graph generation")
        
        # Generation parameters
        st.markdown("### Generation Parameters")
        
        n_graphs = st.slider("Number of Graphs", 1, 100, 20, help="Number of graphs to generate")
        
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
                edge_probability_variance=edge_probability_variance,
                center_variance=center_variance,
                cluster_variance=cluster_variance,
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
                homophily_range=(homophily_min, homophily_max),
                avg_degree_range=(avg_degree_min, avg_degree_max),
                use_dccc_sbm=use_dccc_sbm,
                community_cooccurrence_homogeneity=community_cooccurrence_homogeneity,
                disable_deviation_limiting=disable_deviation_limiting,
                max_mean_community_deviation=max_mean_community_deviation,
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
                show_progress=True,  # We'll handle progress manually
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
        
        # Theoretical vs Actual Power Law Distribution
        if st.checkbox("Show Theoretical vs Actual Power Law Distribution"):
            st.markdown('<h3 class="section-header">Theoretical vs Actual Power Law Distribution</h3>', unsafe_allow_html=True)
            
            # Get actual degrees from the graph
            actual_degrees = np.array([d for _, d in selected_graph.graph.degree()])
            
            # Check if graph has nodes and edges
            if len(actual_degrees) == 0:
                st.warning("Graph has no nodes. Cannot analyze degree distribution.")
                return
            
            # Check if all degrees are zero
            if np.all(actual_degrees == 0):
                st.warning("Graph has no edges. Cannot analyze degree distribution.")
                return
            
            # Determine the power law exponent based on generation method and parameters
            if selected_graph.generation_method == "dccc_sbm" and hasattr(selected_graph, 'generation_params'):
                # Try to get the exponent from generation parameters
                if 'power_law_exponent' in selected_graph.generation_params:
                    exponent = selected_graph.generation_params['power_law_exponent']
                elif 'dccc_global_degree_params' in selected_graph.generation_params:
                    dccc_params = selected_graph.generation_params['dccc_global_degree_params']
                    exponent = dccc_params.get('exponent', 2.5)
                else:
                    exponent = 2.5  # Default
            else:
                # For standard DC-SBM, estimate the exponent from actual degrees
                 try:
                     def negative_log_likelihood(alpha):
                         if alpha <= 1.0:
                             return np.inf
                         try:
                             degrees_array = actual_degrees[actual_degrees > 0]
                             if len(degrees_array) < 2:
                                 return np.inf
                             
                             k_min = np.min(degrees_array)
                             n = len(degrees_array)
                             
                             # Approximation: zeta(alpha, k_min) ≈ sum_{k=k_min}^{k_max} k^(-alpha)
                             k_max = max(100, np.max(degrees_array) * 2)
                             k_range = np.arange(k_min, k_max + 1)
                             zeta_approx = np.sum(k_range**(-alpha))
                             
                             if zeta_approx <= 0:
                                 return np.inf
                                 
                             log_likelihood = -alpha * np.sum(np.log(degrees_array)) - n * np.log(zeta_approx)
                             return -log_likelihood
                             
                         except (OverflowError, ZeroDivisionError, ValueError):
                             return np.inf
                     
                     result = minimize_scalar(negative_log_likelihood, bounds=(1.01, 10.0), method='bounded')
                     if result.success and 1.0 < result.x < 20.0:
                         exponent = result.x
                     else:
                         exponent = 2.5
                 except:
                     exponent = 2.5
            
            # Generate theoretical power law distribution
            theoretical_degrees = generate_theoretical_power_law(selected_graph.n_nodes, exponent)
            
            # Display exponent information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Power Law Exponent (α)", f"{exponent:.3f}")
            with col2:
                st.metric("Actual Min Degree", f"{np.min(actual_degrees):.1f}")
            with col3:
                st.metric("Actual Max Degree", f"{np.max(actual_degrees):.1f}")
            
            # Plot comparison
            fig = plot_degree_distribution_comparison(
                actual_degrees,
                theoretical_degrees,
                exponent,
                title=f"Theoretical vs Actual Power Law Distribution (N={selected_graph.n_nodes})"
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Additional statistics
            st.markdown("#### Distribution Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Actual Mean", f"{np.mean(actual_degrees):.2f}")
                st.metric("Theoretical Mean", f"{np.mean(theoretical_degrees):.2f}")
            
            with col2:
                st.metric("Actual Std", f"{np.std(actual_degrees):.2f}")
                st.metric("Theoretical Std", f"{np.std(theoretical_degrees):.2f}")
            
            with col3:
                st.metric("Actual Median", f"{np.median(actual_degrees):.2f}")
                st.metric("Theoretical Median", f"{np.median(theoretical_degrees):.2f}")
            
            with col4:
                # Calculate Kolmogorov-Smirnov statistic
                from scipy.stats import ks_2samp
                ks_stat, ks_pvalue = ks_2samp(actual_degrees, theoretical_degrees)
                st.metric("KS Statistic", f"{ks_stat:.3f}")
                st.metric("KS p-value", f"{ks_pvalue:.3f}")

        # Probability Matrices Comparison
        if st.checkbox("Show Probability Matrices Comparison"):
            st.markdown('<h3 class="section-header">Probability Matrices Comparison</h3>', unsafe_allow_html=True)
            
            # Check if selected_graph is a GraphSample
            if isinstance(selected_graph, GraphSample):
                # Get the matrices for detailed analysis
                P_sub = selected_graph.P_sub
                P_real, community_sizes, connection_counts = selected_graph.calculate_actual_probability_matrix()
                
                # Calculate deviation statistics
                deviation = np.abs(P_real - P_sub)
                mean_deviation = np.mean(deviation)
                max_deviation = np.max(deviation)
                std_deviation = np.std(deviation)
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Deviation", f"{mean_deviation:.4f}")
                with col2:
                    st.metric("Max Deviation", f"{max_deviation:.4f}")
                with col3:
                    st.metric("Std Deviation", f"{std_deviation:.4f}")
                with col4:
                    st.metric("Relative Error", f"{mean_deviation/np.mean(P_sub)*100:.2f}%")
                
                # Plot the matrices
                fig = plot_probability_matrices_comparison(selected_graph)
                st.pyplot(fig)
                plt.close(fig)
                
                # Show detailed deviation matrix
                if st.checkbox("Show Deviation Matrix"):
                    st.markdown("#### Deviation Matrix (|P_real - P_sub|)")
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(deviation, cmap='Reds', aspect='auto')
                    ax.set_title('Deviation Matrix')
                    ax.set_xlabel('Community')
                    ax.set_ylabel('Community')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Absolute Deviation')
                    
                    # Add text annotations
                    for i in range(deviation.shape[0]):
                        for j in range(deviation.shape[1]):
                            value = deviation[i, j]
                            text_color = "white" if value > np.max(deviation) / 2 else "black"
                            ax.text(j, i, f'{value:.4f}',
                                   ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
                    
                    # Set tick labels
                    n_communities = len(selected_graph.communities)
                    tick_labels = [f'C{i}' for i in range(n_communities)]
                    ax.set_xticks(range(n_communities))
                    ax.set_yticks(range(n_communities))
                    ax.set_xticklabels(tick_labels)
                    ax.set_yticklabels(tick_labels)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("Selected graph is not a GraphSample object, so probability matrices cannot be displayed.")

        # Family summary
        if st.checkbox("Show Family Summary"):
            st.markdown('<h3 class="section-header">Family Summary</h3>', unsafe_allow_html=True)
            
            # Check if family generator exists in session state
            if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                # Analyze basic properties
                family_properties = st.session_state.family_generator.analyze_graph_family_properties()
                for key, value in family_properties.items():
                    if isinstance(value, list):
                        # Check if the values are in the list are ints of floats else skip
                        if all(isinstance(v, (int, float)) for v in value):
                            st.metric(key, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
                        else:
                            # Skip the metric
                            continue
                    else:
                        # Skip the metric
                        continue
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
                    st.metric("Edge Probability Variance", f"{universe.edge_probability_variance:.3f}")
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