import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import the graph generation classes
from graph_universe.graph_universe import GraphUniverse
from graph_universe.graph_sample import GraphSample
from graph_universe.graph_family import GraphFamilyGenerator

from utils.visualizations import (
    plot_graph_communities, 
    plot_universe_community_degree_propensity_vector,
    plot_universe_feature_centers,
    plot_property_validation
)
import time

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
    ax1.set_title('Target Propensity Matrix (P_sub)')
    ax1.set_xlabel('Community')
    ax1.set_ylabel('Community')
    
    # Add colorbar for the first plot
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Propensity')
    
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
    st.markdown('<h1 class="main-header">GraphUniverse</h1>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["🌌 Universe Generation", "📊 Graph Family Generation"])
    
    # TAB 1: UNIVERSE GENERATION
    with tab1:
        st.markdown('<h2 class="section-header">Graph Universe Generation</h2>', unsafe_allow_html=True)
        st.markdown("**Step 1:** Create a new universe or load an existing one. The universe holds all community-related latent properties that can reappear throughout all graphs in the family.")
        
        # Universe creation/loading options
        universe_option = st.radio(
            "Choose Universe Option:",
            ["Create New Universe", "Load Existing Universe"],
            horizontal=True,
            help="Create a new universe from scratch or load a previously saved universe from a pickle file"
        )
        
        if universe_option == "Load Existing Universe":
            st.markdown("### Load Universe from File")
            
            # File uploader for universe pickle files
            uploaded_universe_file = st.file_uploader(
                "Choose a universe pickle file",
                type=['pkl'],
                help="Select a graph_universe.pkl file from a previously saved graph family"
            )
            
            if uploaded_universe_file is not None:
                try:
                    # Load the universe from the uploaded file
                    import pickle
                    universe = pickle.load(uploaded_universe_file)
                    
                    # Validate that it's actually a GraphUniverse object
                    if not isinstance(universe, GraphUniverse):
                        st.error("❌ The uploaded file does not contain a valid GraphUniverse object.")
                    else:
                        # Store in session state
                        st.session_state.universe = universe
                        st.session_state.universe_params = {
                            'K': universe.K,
                            'feature_dim': universe.feature_dim,
                            'edge_propensity_variance': universe.edge_propensity_variance,
                            'seed': 'unknown'  # We don't know the original seed
                        }
                        
                        # Set active tab to Graph Family Generation
                        st.session_state.active_tab = 1
                        
                        st.success("✅ Universe loaded successfully!")
                        
                        # Show a prominent call-to-action to go to next tab
                        st.info("🎯 **Next Step:** Go to the '📊 Graph Family Generation' tab to create your graph family!")
                        st.rerun()
                        
                        # Display loaded universe info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Communities (K)", universe.K)
                            st.metric("Feature Dimension", universe.feature_dim)
                        with col_info2:
                            st.metric("Edge Propensity Variance", f"{universe.edge_propensity_variance:.3f}")
                            st.metric("Community-Degree Propensity Vector Shape", f"{universe.community_degree_propensity_vector.shape}")
                        
                        # Show universe properties
                        if st.checkbox("Show Loaded Universe Properties", key="loaded_universe_props"):
                            st.markdown("#### Community-Degree Propensity Vector")
                            fig_degree_centers = plot_universe_community_degree_propensity_vector(universe)
                            st.pyplot(fig_degree_centers)
                            plt.close(fig_degree_centers)
                            
                            st.markdown("#### Propensity Matrix")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(universe.P, cmap='viridis', aspect='auto')
                            plt.colorbar(im, ax=ax, label="Edge Propensity")
                            ax.set_title("Inter-Community Propensity Matrix")
                            ax.set_xticks(np.arange(universe.K))
                            ax.set_yticks(np.arange(universe.K))
                            ax.set_xticklabels([f"C{i}" for i in range(universe.K)])
                            ax.set_yticklabels([f"C{i}" for i in range(universe.K)])
                            
                            # Add text annotations
                            for i in range(universe.K):
                                for j in range(universe.K):
                                    value = universe.P[i, j]
                                    text_color = "white" if value > np.mean(universe.P) else "black"
                                    ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                                           color=text_color, fontsize=8, fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                except Exception as e:
                    st.error(f"❌ Error loading universe file: {str(e)}")
            else:
                st.info("📁 Please upload a universe pickle file to proceed.")
        
        else:  # Create New Universe
            st.markdown("### Create New Universe")
            
            # Universe parameters in main area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Universe Parameters")
                
                # Basic universe parameters
                col_a, col_b = st.columns(2)
                with col_a:
                    K = st.slider(r"Number of Communities $\boldsymbol{K}$", 2, 20, 10)
                    edge_propensity_variance = st.slider(
                        r"Edge Propensity Variance $\boldsymbol{\epsilon}$", 
                        0.0, 1.0, 0.5, 0.01,
                        help="Amount of variance in edge propensities"
                    )
                
                with col_b:
                    feature_dim = st.slider("Feature Dimension", 1, 50, 15)
            
                # Feature generation parameters
                if feature_dim > 0:
                    st.markdown("#### Feature Generation Parameters")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        center_variance = st.slider(
                            r"Center Variance $\boldsymbol{\sigma_{\text{center}}}$", 
                            0.01, 1.0, 0.1, 0.01,
                            help="Separation between cluster centers"
                        )
                    with col_d:
                        cluster_variance = st.slider(
                            r"Cluster Variance $\boldsymbol{\sigma_{\text{cluster}}}$", 
                            0.01, 1.0, 0.5, 0.01,
                            help="Spread within each cluster"
                        )
                else:
                    center_variance = 1.0
                    cluster_variance = 0.1
                
                # Generation parameters
                universe_seed = st.number_input("Universe Seed", value=42, help="Random seed for universe generation")
            
                # Generate Universe Button
                if st.button("🌌 Generate Universe", type="primary", use_container_width=True):
                    with st.spinner("Creating Graph Universe..."):
                        universe = GraphUniverse(
                            K=K,
                            feature_dim=feature_dim,
                            edge_propensity_variance=edge_propensity_variance,
                            center_variance=center_variance,
                            cluster_variance=cluster_variance,
                            seed=universe_seed
                        )
                        
                        # Store in session state
                        st.session_state.universe = universe
                        st.session_state.universe_params = {
                            'K': K,
                            'feature_dim': feature_dim,
                            'edge_propensity_variance': edge_propensity_variance,
                            'center_variance': center_variance,
                            'cluster_variance': cluster_variance,
                            'seed': universe_seed
                        }
                        
                        # Reset analysis state
                        if 'graphs' in st.session_state:
                            del st.session_state.graphs
                        if 'family_generator' in st.session_state:
                            del st.session_state.family_generator
                        if 'show_analysis' in st.session_state:
                            del st.session_state.show_analysis
                        if 'show_download' in st.session_state:
                            del st.session_state.show_download
                        if 'show_properties' in st.session_state:
                            del st.session_state.show_properties
                        if 'show_consistency' in st.session_state:
                            del st.session_state.show_consistency
                        if 'show_signals' in st.session_state:
                            del st.session_state.show_signals
                        if 'cached_signals' in st.session_state:
                            del st.session_state.cached_signals
                        if 'validation_plot' in st.session_state:
                            del st.session_state.validation_plot
                        
                        # Set active tab to Graph Family Generation
                        st.session_state.active_tab = 1
                        
                        st.success("✅ Universe generated successfully!")
                        
                    # Show a prominent call-to-action to go to next tab
                    st.success("🎯 **Next Step:** Go to the '📊 Graph Family Generation' tab to create your graph family!")
                    st.rerun()
        
        # Universe Status section (visible for both options)
        st.markdown("---")
        col_status1, col_status2 = st.columns([1, 1])
        
        with col_status1:
            st.markdown("### Universe Status")
            
            if 'universe' in st.session_state and st.session_state.universe is not None:
                universe = st.session_state.universe
                params = st.session_state.get('universe_params', {})
                
                st.success("🌌 Universe Ready")
                st.metric("Communities", universe.K)
                st.metric("Feature Dim", universe.feature_dim)
                st.metric("Edge Propensity Variance", f"{universe.edge_propensity_variance:.3f}")
                st.metric("Fixed Within-cluster Variance " + r"$\boldsymbol{\sigma_{\text{cluster}}}$", f"{universe.feature_generator.cluster_variance:.3f}")

                # Here we'll plot the values of the feature centers
                st.markdown("#### Feature Centers " + r"$\boldsymbol{\mu}$")
                fig_feature_centers = plot_universe_feature_centers(universe)
                st.pyplot(fig_feature_centers)
                plt.close(fig_feature_centers)
            else:
                st.warning("⚠️ No Universe Available")
                st.info("Create a new universe or load an existing one to proceed with graph family generation.")
        
        with col_status2:
            if 'universe' in st.session_state and st.session_state.universe is not None:
                universe = st.session_state.universe

                st.markdown("#### Universe Raw Edge Propensity Matrix " + r"$\tilde{\mathbf{P}}$")
                # Create a simple probability matrix visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(universe.P, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, label="Edge Propensity")
                ax.set_title(r"$\tilde{\mathbf{P}}$")
                ax.set_xticks(np.arange(universe.K))
                ax.set_yticks(np.arange(universe.K))
                ax.set_xticklabels([f"C{i}" for i in range(universe.K)])
                ax.set_yticklabels([f"C{i}" for i in range(universe.K)])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("#### Community-Degree Propensity Vector " + r"$\boldsymbol{\delta}$")
                fig_degree_centers = plot_universe_community_degree_propensity_vector(universe)
                st.pyplot(fig_degree_centers)
                plt.close(fig_degree_centers)
                
                # Add text annotations
                for i in range(universe.K):
                    for j in range(universe.K):
                        value = universe.P[i, j]
                        text_color = "white" if value > np.mean(universe.P) else "black"
                        ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                                color=text_color, fontsize=8, fontweight='bold')
                    
    # TAB 2: GRAPH FAMILY GENERATION  
    with tab2:
        st.markdown('<h2 class="section-header">Graph Family Generation</h2>', unsafe_allow_html=True)
        
        # Check if user just created/loaded a universe
        if 'active_tab' in st.session_state and st.session_state.active_tab == 1:
            st.success("🎉 Welcome to Graph Family Generation! Your universe is ready.")
            # Reset the active tab flag so balloons don't show every time
            if 'active_tab' in st.session_state:
                del st.session_state.active_tab
        
        st.markdown("**Step 2:** Generate a family of graphs using the created universe with user-definable properties.")
        
        # Check if universe exists
        if 'universe' not in st.session_state or st.session_state.universe is None:
            st.error("❌ No universe available. Please generate a universe first in the 'Universe Generation' tab.")
            return
        
        # Sidebar for graph family parameters
        with st.sidebar:
            st.markdown('<h2 class="section-header">Graph Family Parameters</h2>', unsafe_allow_html=True)
            
            n_graphs = st.slider("Number of Graphs " + r"$\boldsymbol{g}$", 1, 100, 20, help="Number of graphs to generate. In this demo version, for limited app hosting purposes, the number of graphs is limited to 100.")
            
            homophily_min, homophily_max = st.slider(
                r"Homophily Range $\boldsymbol{h_{\text{min}}}$ - " + r"$\boldsymbol{h_{\text{max}}}$",
                0.0, 1.0, (0.1, 0.4),
            )
            
            avg_degree_min, avg_degree_max = st.slider(
                "Average Degree Range " + r"$\boldsymbol{d_{\text{min}}}$ - " + r"$\boldsymbol{d_{\text{max}}}$",
                1.0, 30.0, (2.0, 5.0),
                help="In this demo version, for limited app hosting purposes, the average degree is limited to 30."
            )

            min_n_nodes, max_n_nodes = st.slider("Node Count " + r"$\boldsymbol{n_{\text{min}}}$ - " + r"$\boldsymbol{n_{\text{max}}}$", 10, 500, (100, 300), help="Minimum and maximum number of nodes per graph. In this demo version, for limited app hosting purposes, the number of nodes is limited to 500.")

            min_communities, max_communities = st.slider("Number of participating communities per graph " + r"$\boldsymbol{k_{\text{min}}}$ - " + r"$\boldsymbol{k_{\text{max}}}$", 2, universe.K, (3, 8))

            degree_separation_min, degree_separation_max = st.slider(
                r"Degree Separation Range $\boldsymbol{\rho _{\text{min}}}$ - " + r"$\boldsymbol{\rho_{\text{max}}}$",
                0.0, 1.0, (0.7, 1.0),
            )
            
            degree_distribution = st.selectbox(
                "Degree Distribution",
                ["power_law", "exponential", "uniform"],
                help="Type of degree distribution. More options included here than in the original paper to play around with. Power law generally seen as most realistic.",
                key="degree_distribution_selector"
            )
            
            if degree_distribution == "power_law":
                power_law_min, power_law_max = st.slider(
                    "Power Law Exponent Range",
                    1.5, 5.0, (2.0, 2.5),
                )
            else:
                power_law_min, power_law_max = 2.5, 2.5
                
            if degree_distribution == "exponential":
                exp_min, exp_max = st.slider(
                    "Exponential Rate Range",
                    0.1, 2.0, (0.3, 1.0),
                )
            else:
                exp_min, exp_max = 0.5, 0.5
                
            if degree_distribution == "uniform":
                uniform_min_min, uniform_min_max = st.slider(
                    "Uniform Min Factor Range",
                    0.1, 1.0, (0.3, 0.7),
                )
                uniform_max_min, uniform_max_max = st.slider(
                    "Uniform Max Factor Range",
                    1.0, 3.0, (1.3, 2.0),
                )
            else:
                uniform_min_min, uniform_min_max = 0.5, 0.5
                uniform_max_min, uniform_max_max = 1.5, 1.5
                                         
            
            family_seed = st.number_input("Family Seed", value=123)
            
            timeout_minutes = st.slider("Timeout (minutes)", 1.0, 30.0, 5.0, help="Timeout for generation. If the generation takes too long, you can increase this value.")
            
        # Main generation area
        universe = st.session_state.universe
        
        # Display current universe info
        st.info(f"🌌 Using universe with {universe.K} communities, {universe.feature_dim}D features")
        
        # Initialize session state for UI control if not already set
        if 'show_download' not in st.session_state:
            st.session_state.show_download = False
        if 'show_analysis' not in st.session_state:
            st.session_state.show_analysis = False
            
        # Generate Graph Family Button
        if st.button("📊 Generate Graph Family", type="primary", use_container_width=True):
            # Reset analysis state
            if 'show_analysis' in st.session_state:
                del st.session_state.show_analysis
            if 'show_download' in st.session_state:
                del st.session_state.show_download
            if 'show_properties' in st.session_state:
                del st.session_state.show_properties
            if 'show_consistency' in st.session_state:
                del st.session_state.show_consistency
            if 'show_signals' in st.session_state:
                del st.session_state.show_signals
            if 'cached_signals' in st.session_state:
                del st.session_state.cached_signals
            if 'validation_plot' in st.session_state:
                del st.session_state.validation_plot
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with status_container:
                status_text = st.empty()
        
            # Step 1: Create family generator
            status_text.text("Step 1/3: Setting up Graph Family Generator...")
            progress_bar.progress(0)
            progress_text.text("0% - Initializing...")
            
            family_generator = GraphFamilyGenerator(
                universe=universe,
                min_n_nodes=min_n_nodes,
                max_n_nodes=max_n_nodes,
                min_communities=min_communities,
                max_communities=max_communities,
                homophily_range=(homophily_min, homophily_max),
                avg_degree_range=(avg_degree_min, avg_degree_max),
                use_dccc_sbm=True,
                degree_distribution=degree_distribution,
                power_law_exponent_range=(power_law_min, power_law_max),
                exponential_rate_range=(exp_min, exp_max),
                uniform_min_factor_range=(uniform_min_min, uniform_min_max),
                uniform_max_factor_range=(uniform_max_min, uniform_max_max),
                degree_separation_range=(degree_separation_min, degree_separation_max),
                degree_heterogeneity=0.5,
                seed=family_seed
            )
            
            progress_bar.progress(33)
            progress_text.text("33% - Family generator ready")
            
            # Step 2: Generate family
            status_text.text("Step 2/3: Generating Graph Family...")
            
            start_time = time.time()
            
            # Use the family generator's generate_family method
            family_generator.generate_family(
                n_graphs=n_graphs,
                show_progress=True,
                collect_stats=True,
                timeout_minutes=timeout_minutes
            )
            graphs = family_generator.graphs
            
            progress_bar.progress(90)
            progress_text.text("90% - Family generation complete")
            
            # Step 3: Finalize
            status_text.text("Step 3/3: Finalizing...")
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
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Time", f"{stats.get('total_time', 0):.2f}s")
                with col_b:
                    st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
                with col_c:
                    st.metric("Avg Time per Graph", f"{stats.get('avg_time_per_graph', 0):.2f}s")
            
            # Store in session state
            st.session_state.graphs = graphs
            st.session_state.family_generator = family_generator
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            
            st.rerun()
        
        # Show existing graphs if available
        if 'graphs' in st.session_state and st.session_state.graphs:
            graphs = st.session_state.graphs
            family_generator = st.session_state.family_generator
            
            # Add an announcement banner for downloading or moving to the next page
            st.success("🎉 Graph family successfully generated! You can now download the family or analyze it in detail.")
            
            # We'll show the property validation plot when the user clicks 'Analyze Graph Family'
            
            st.markdown("### Quick Stats")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            # Basic stats
            total_nodes = sum(g.n_nodes for g in graphs)
            avg_nodes = total_nodes / len(graphs)
            total_edges = sum(g.graph.number_of_edges() for g in graphs)
            avg_edges = total_edges / len(graphs)
            
            with col_a:
                st.metric("Total Graphs", len(graphs))
            with col_b:
                st.metric("Avg Nodes", f"{avg_nodes:.1f}")
            with col_c:
                st.metric("Avg Edges", f"{avg_edges:.1f}")
            with col_d:
                # Community stats
                all_communities = set()
                for g in graphs:
                    all_communities.update(g.communities)
                st.metric("Unique Communities", len(all_communities))
            
            # Graph visualization section
            st.markdown("### Graph Visualizations")
            
            # Graph selection
            graph_idx = st.selectbox(
                "Select Graph to Visualize",
                range(len(graphs)),
                format_func=lambda x: f"Graph {x+1} ({graphs[x].n_nodes} nodes, {graphs[x].graph.number_of_edges()} edges)",
                key="graph_selector_tab2"
            )
            
            selected_graph = graphs[graph_idx]
            
            # Visualization options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                layout = st.selectbox("Layout", ["spring", "kamada_kawai", "spectral", "circular"], key="layout_selector_tab2")
            
            with col2:
                node_size = st.slider("Node Size", 10, 200, 50, key="node_size_slider_tab2")
            
            with col3:
                show_labels = st.checkbox("Show Node Labels", value=False, key="show_labels_checkbox_tab2")
            
            # Create visualization
            fig = plot_graph_communities(
                selected_graph,
                layout=layout,
                node_size=node_size,
                with_labels=show_labels,
                figsize=(12, 8),
                title=f"Graph {graph_idx+1} - {selected_graph.n_nodes} nodes, {selected_graph.graph.number_of_edges()} edges"
            )
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Add buttons for download and analysis
            st.markdown("### Next Steps")
            col_nav1, col_nav2 = st.columns(2)
            
            with col_nav1:
                download_option = st.button("💾 Download Graph Family", use_container_width=True)
            with col_nav2:
                analyze_option = st.button("📊 Analyze Graph Family", use_container_width=True)
                
            # Show download options if download button is clicked
            if download_option:
                st.session_state.show_download = True
                st.session_state.show_analysis = False
                st.rerun()
                
            # Show analysis if analyze button is clicked
            if analyze_option:
                st.session_state.show_analysis = True
                st.session_state.show_download = False
                st.rerun()
                
            # Show download section if selected
            if st.session_state.get('show_download', False):
                st.markdown("### 💾 Download PyTorch Geometric Graphs")
                
                # Check if family generator exists in session state
                if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                    family_generator = st.session_state.family_generator
                    
                    if len(family_generator.graphs) > 0:
                        # Task selection
                        st.markdown("#### Select Tasks and Configure Download")
                        available_tasks = [
                            "community_detection",
                            "triangle_counting",
                            "k_hop_community_counts_k1",
                            "k_hop_community_counts_k2", 
                            "k_hop_community_counts_k3",
                            "k_hop_community_counts_k4",
                            "k_hop_community_counts_k5",
                        ]
                        
                        selected_tasks = st.multiselect(
                            "Choose tasks to generate PyG graphs for:",
                            available_tasks,
                            default=["community_detection", "triangle_counting", "k_hop_community_counts_k1"],
                            help="Select one or more tasks. Each task will create different target labels for the graphs."
                        )
                        
                        # Directory for saving
                        family_dir = st.text_input(
                            "Save Directory",
                            value="datasets",
                            help="Directory where the PyG graphs will be saved"
                        )
                    
                        # Validation and download button
                        if not selected_tasks:
                            st.warning("Please select at least one task to proceed.")
                        
                        # Prominent download button
                        if selected_tasks:
                            if st.button("💾 Download PyG Graphs", type="primary", use_container_width=True):
                                try:
                                    # Create progress containers
                                    progress_container = st.container()
                                    status_container = st.container()
                                    
                                    with progress_container:
                                        progress_bar = st.progress(0)
                                        progress_text = st.empty()
                                    
                                    with status_container:
                                        status_text = st.empty()
                                    
                                    # Step 1: Convert to PyG graphs
                                    status_text.text("Step 1/2: Converting graphs to PyTorch Geometric format...")
                                    progress_bar.progress(25)
                                    progress_text.text("25% - Starting conversion...")
                                    
                                    # Check if graphs have features (required for PyG conversion)
                                    graphs_with_features = all(hasattr(g, 'features') and g.features is not None for g in family_generator.graphs)
                                    if not graphs_with_features:
                                        st.error("❌ Cannot convert to PyG: Some graphs don't have features. Please generate graphs with feature_dim > 0.")
                                        progress_container.empty()
                                        status_container.empty()
                                        return
                                    
                                    progress_bar.progress(50)
                                    progress_text.text("50% - Converting graphs...")
                                    
                                    # Step 2: Save PyG graphs and universe
                                    status_text.text("Step 2/2: Saving PyG graphs and universe...")
                                    progress_bar.progress(75)
                                    progress_text.text("75% - Saving to disk...")
                                    
                                    # Use the family generator's save method
                                    family_generator.save_pyg_graphs_and_universe(
                                        tasks=selected_tasks,
                                        root_dir=family_dir
                                    )
                                    
                                    progress_bar.progress(100)
                                    progress_text.text("100% - Complete!")
                                    
                                    st.success(f"✅ Successfully saved PyG graphs for {len(selected_tasks)} task(s)!")
                                    
                                    # Clear progress indicators after a short delay
                                    time.sleep(1)
                                    progress_container.empty()
                                    status_container.empty()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error saving PyG graphs: {str(e)}")
                                    st.exception(e)
                                    progress_container.empty()
                                    status_container.empty()
                    else:
                        st.warning("No graphs available in family generator. Please generate a graph family first.")
                else:
                    st.warning("No graph family has been generated yet. Please generate a graph family first.")
            
            # Show analysis section if selected
            if st.session_state.get('show_analysis', False):
                st.markdown("### 📊 Family-level Analysis")
                
                # Check if family generator exists in session state
                if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                    family_generator = st.session_state.family_generator
                    
                    # Property Validation Visualization
                    st.markdown("#### Property Target vs. Actual Ranges")
                    st.markdown("This visualization shows how well the generated graphs match the target property ranges:")
                    
                    # Display property validation plot
                    with st.spinner("Generating property validation plot..."):
                        try:
                            # Check if we have cached the validation plot
                            if 'validation_plot' not in st.session_state:
                                validation_fig = plot_property_validation(family_generator)
                                st.session_state.validation_plot = validation_fig
                            else:
                                validation_fig = st.session_state.validation_plot
                            
                            st.pyplot(validation_fig)
                            plt.close(validation_fig)
                        except Exception as e:
                            st.error(f"Error generating property validation plot: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("#### Detailed Analysis")
                    
                    # Create three buttons for different analysis types
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        analyze_properties = st.button("General Family Properties", use_container_width=True, key="analyze_properties_btn")
                    
                    with col_btn2:
                        analyze_consistency = st.button("Consistency Metrics", use_container_width=True, key="analyze_consistency_btn")
                    
                    with col_btn3:
                        analyze_signals = st.button("Signal Properties", use_container_width=True, key="analyze_signals_btn")
                    
                    # Store button states in session state if clicked
                    if analyze_properties:
                        st.session_state.show_properties = True
                        st.session_state.show_consistency = False
                        st.session_state.show_signals = False
                        st.rerun()
                    
                    if analyze_consistency:
                        st.session_state.show_consistency = True
                        st.session_state.show_properties = False
                        st.session_state.show_signals = False
                        st.rerun()
                    
                    if analyze_signals:
                        st.session_state.show_signals = True
                        st.session_state.show_properties = False
                        st.session_state.show_consistency = False
                        st.rerun()
                    
                    # Initialize session state variables if they don't exist
                    if 'show_properties' not in st.session_state:
                        st.session_state.show_properties = True
                    if 'show_consistency' not in st.session_state:
                        st.session_state.show_consistency = False
                    if 'show_signals' not in st.session_state:
                        st.session_state.show_signals = False
                        
                    # Add a cancel button to stop analysis
                    if st.session_state.show_signals:
                        if st.button("Cancel Signal Analysis", type="secondary"):
                            del st.session_state.show_signals
                            st.session_state.show_properties = True
                            st.rerun()
                    
                    # Show analysis based on selection
                    try:
                        # General Family Properties
                        if st.session_state.show_properties:
                            st.markdown("#### General Family Properties")
                            family_properties = family_generator.analyze_graph_family_properties()
                            
                            # Convert to DataFrame for table display
                            data = []
                            for key, value in family_properties.items():
                                if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                                    if len(value) > 0:
                                        mean_val = np.mean(value)
                                        std_val = np.std(value)
                                        min_val = np.min(value)
                                        max_val = np.max(value)
                                        data.append({
                                            "Metric": key,
                                            "Mean": f"{mean_val:.3f}",
                                            "Std Dev": f"{std_val:.3f}",
                                            "Min": f"{min_val:.3f}",
                                            "Max": f"{max_val:.3f}"
                                        })
                            
                            # Display as a table
                            if data:
                                df = pd.DataFrame(data)
                                st.table(df.set_index("Metric"))
                            else:
                                st.info("No property data available.")
                        
                        # Consistency Metrics
                        elif st.session_state.show_consistency:
                            st.markdown("#### Consistency Metrics")
                            consistency = family_generator.analyze_graph_family_consistency()
                            
                            # Convert to DataFrame for table display
                            data = []
                            for metric, value in consistency.items():
                                if isinstance(value, list):
                                    if len(value) > 0:
                                        mean_val = np.mean(value)
                                        std_val = np.std(value)
                                        min_val = np.min(value)
                                        max_val = np.max(value)
                                        data.append({
                                            "Metric": metric,
                                            "Mean": f"{mean_val:.3f}",
                                            "Std Dev": f"{std_val:.3f}",
                                            "Min": f"{min_val:.3f}",
                                            "Max": f"{max_val:.3f}"
                                        })
                                else:
                                    data.append({
                                        "Metric": metric,
                                        "Value": str(value),
                                        "Std Dev": "-",
                                        "Min": "-",
                                        "Max": "-"
                                    })
                            
                            # Display as a table
                            if data:
                                df = pd.DataFrame(data)
                                if "Value" in df.columns:
                                    st.table(df.set_index("Metric"))
                                else:
                                    st.table(df.set_index("Metric"))
                            else:
                                st.info("No consistency data available.")
                        
                        # Signal Properties
                        elif st.session_state.show_signals:
                            st.markdown("#### Signal Properties")
                            
                            # Check if signals have already been calculated
                            if 'cached_signals' not in st.session_state:
                                with st.spinner("Calculating signal properties... This may take a while."):
                                    signals = family_generator.analyze_graph_family_signals()
                                    # Cache the results
                                    st.session_state.cached_signals = signals
                            else:
                                signals = st.session_state.cached_signals
                            
                            if signals:
                                signals_structured_dict = {signal_name: {'mean': np.mean(metric_data), 'std': np.std(metric_data), 'min': np.min(metric_data), 'max': np.max(metric_data)} for signal_name, metric_data in signals.items()}
                                df = pd.DataFrame.from_dict(signals_structured_dict)

                                # Display as a table
                                st.table(df)
                                
                            else:
                                st.info("No signal data available.")
                    
                    except Exception as e:
                        st.error(f"❌ Error analyzing family properties: {str(e)}")
                        st.exception(e)
    
    # Set active tab if needed
    if 'active_tab' in st.session_state:
        if st.session_state.active_tab == 2:
            # Activate the Download & Analysis tab
            tab3.selectbox = True
            # Reset the active tab flag
            del st.session_state.active_tab
    
    # # TAB 3: DOWNLOAD & ANALYSIS
    # with tab3:
    #     st.markdown('<h2 class="section-header">Download & Analysis</h2>', unsafe_allow_html=True)
    #     st.markdown("**Step 3:** Download your generated graph families and analyze their properties.")
        
    #     # Check if graphs exist
    #     if 'graphs' not in st.session_state or not st.session_state.graphs:
    #         st.error("❌ No graphs available. Please generate a graph family first in the 'Graph Family Generation' tab.")
    #         return
        
    #     graphs = st.session_state.graphs
        
    #     # ANALYSIS SECTION
    #     st.markdown("### 📊 Graph Analysis & Visualization")
        
    #     # Graph selection and visualization
    #     graphs = st.session_state.graphs
        
    #     # Graph selection
    #     graph_idx = st.selectbox(
    #         "Select Graph to Visualize",
    #         range(len(graphs)),
    #         format_func=lambda x: f"Graph {x+1} ({graphs[x].n_nodes} nodes, {graphs[x].graph.number_of_edges()} edges)",
    #         key="graph_selector_tab3"
    #     )
        
    #     selected_graph = graphs[graph_idx]
        
    #     # Visualization options
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         layout = st.selectbox("Layout", ["spring", "kamada_kawai", "spectral", "circular"], key="layout_selector_tab3")
        
    #     with col2:
    #         node_size = st.slider("Node Size", 10, 200, 50, key="node_size_slider_tab3")
        
    #     with col3:
    #         show_labels = st.checkbox("Show Node Labels", value=False, key="show_labels_checkbox_tab3")
        
    #     # Create visualization
    #     fig = plot_graph_communities(
    #         selected_graph,
    #         layout=layout,
    #         node_size=node_size,
    #         with_labels=show_labels,
    #         figsize=(12, 8),
    #         title=f"Graph {graph_idx+1} - {selected_graph.n_nodes} nodes, {selected_graph.graph.number_of_edges()} edges"
    #     )
        
    #     st.pyplot(fig)
    #     plt.close(fig)
        
    #     # Graph statistics
    #     st.markdown("#### Graph Statistics")
        
    #     col1, col2, col3, col4 = st.columns(4)
        
    #     with col1:
    #         st.metric("Nodes", selected_graph.n_nodes)
    #         st.metric("Edges", selected_graph.graph.number_of_edges())
        
    #     with col2:
    #         st.metric("Communities", len(selected_graph.communities))
    #         st.metric("Components", len(list(nx.connected_components(selected_graph.graph))))
        
    #     with col3:
    #         if selected_graph.features is not None:
    #             st.metric("Feature Dim", selected_graph.features.shape[1])
    #         else:
    #             st.metric("Feature Dim", 0)
    #         st.metric("Generation Method", selected_graph.generation_method)
        
    #     with col4:
    #         # Calculate some additional metrics
    #         degrees = [d for _, d in selected_graph.graph.degree()]
    #         if degrees:
    #             st.metric("Avg Degree", f"{2 * selected_graph.graph.number_of_edges() / selected_graph.n_nodes:.2f}")
    #             st.metric("Degree Std", f"{np.std(degrees):.2f}")
    #         else:
    #             st.metric("Avg Degree", "0")
    #             st.metric("Degree Std", "0")
        
    #     # Additional analysis options
    #     st.markdown("---")
    #     st.markdown("#### Additional Analysis Options")
        
    #     # Community analysis
    #     if st.checkbox("Show Community Analysis"):
    #         st.markdown("##### Community Analysis")
            
    #         # Community sizes
    #         community_sizes = {}
    #         for label in selected_graph.community_labels:
    #             comm_id = selected_graph.community_id_mapping[label]
    #             if comm_id not in community_sizes:
    #                 community_sizes[comm_id] = 0
    #             community_sizes[comm_id] += 1
            
    #         # Create community size plot
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         communities = list(community_sizes.keys())
    #         sizes = list(community_sizes.values())
            
    #         bars = ax.bar(range(len(communities)), sizes, color='skyblue', alpha=0.7)
    #         ax.set_xlabel("Community ID")
    #         ax.set_ylabel("Number of Nodes")
    #         ax.set_title("Community Size Distribution")
    #         ax.set_xticks(range(len(communities)))
    #         ax.set_xticklabels([f"C{c}" for c in communities])
            
    #         # Add value labels on bars
    #         for bar, size in zip(bars, sizes):
    #             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
    #                    str(size), ha='center', va='bottom')
            
    #         st.pyplot(fig)
    #         plt.close(fig)
        
    #     # Probability Matrices Comparison
    #     if st.checkbox("Show Probability Matrices Comparison"):
    #         st.markdown("##### Probability Matrices Comparison")
            
    #         # Check if selected_graph is a GraphSample
    #         if isinstance(selected_graph, GraphSample):
    #             # Get the matrices for detailed analysis
    #             P_sub = selected_graph.P_sub
    #             P_real, community_sizes, connection_counts = selected_graph.calculate_actual_probability_matrix()
                
    #             # Calculate deviation statistics
    #             deviation = np.abs(P_real - P_sub)
    #             mean_deviation = np.mean(deviation)
    #             max_deviation = np.max(deviation)
    #             std_deviation = np.std(deviation)
                
    #             # Display statistics
    #             col1, col2, col3, col4 = st.columns(4)
    #             with col1:
    #                 st.metric("Mean Deviation", f"{mean_deviation:.4f}")
    #             with col2:
    #                 st.metric("Max Deviation", f"{max_deviation:.4f}")
    #             with col3:
    #                 st.metric("Std Deviation", f"{std_deviation:.4f}")
    #             with col4:
    #                 st.metric("Relative Error", f"{mean_deviation/np.mean(P_sub)*100:.2f}%")
                
    #             # Plot the matrices
    #             fig = plot_probability_matrices_comparison(selected_graph)
    #             st.pyplot(fig)
    #             plt.close(fig)
    #         else:
    #             st.warning("Selected graph is not a GraphSample object, so probability matrices cannot be displayed.")
        
    #     # Family-level analysis
    #     if st.checkbox("Show Family-level Analysis"):
    #         st.markdown("##### Family-level Analysis")
            
    #         # Check if family generator exists in session state
    #         if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
    #             family_generator = st.session_state.family_generator
                
    #             # Family summary
    #             try:
    #                 family_properties = family_generator.analyze_graph_family_properties()
    #                 col1, col2 = st.columns(2)
                    
    #                 with col1:
    #                     st.markdown("**Family Properties**")
    #                     for key, value in family_properties.items():
    #                         if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
    #                             if len(value) > 0:
    #                                 st.metric(key, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
    #                             else:
    #                                 st.metric(key, "No data")
                    
    #                 with col2:
    #                     st.markdown("**Family Consistency**")
    #                     consistency = family_generator.analyze_graph_family_consistency()
                        
    #                     # Display consistency metrics with better formatting
    #                     for metric, value in consistency.items():
    #                         if isinstance(value, list):
    #                             if len(value) > 0:
    #                                 mean_val = np.mean(value)
    #                                 std_val = np.std(value)
                                    
    #                                 # Color code based on consistency level
    #                                 if metric == 'feature_consistency':
    #                                     if family_generator.universe.feature_dim == 0:
    #                                         st.metric(metric, "N/A (no features)")
    #                                     elif mean_val >= 0.7:
    #                                         st.metric(metric, f"🟢 {mean_val:.3f} ± {std_val:.3f}")
    #                                     elif mean_val >= 0.5:
    #                                         st.metric(metric, f"🟡 {mean_val:.3f} ± {std_val:.3f}")
    #                                     else:
    #                                         st.metric(metric, f"🔴 {mean_val:.3f} ± {std_val:.3f}")
    #                                 else:
    #                                     if mean_val >= 0.7:
    #                                         st.metric(metric, f"🟢 {mean_val:.3f} ± {std_val:.3f}")
    #                                     elif mean_val >= 0.5:
    #                                         st.metric(metric, f"🟡 {mean_val:.3f} ± {std_val:.3f}")
    #                                     else:
    #                                         st.metric(metric, f"🔴 {mean_val:.3f} ± {std_val:.3f}")
    #                             else:
    #                                 st.metric(metric, "No data")
    #                         else:
    #                             st.metric(metric, str(value))
                        
    #                     # Additional feature consistency details
    #                     if 'feature_consistency' in consistency and len(consistency['feature_consistency']) > 0:
    #                         st.markdown("---")
    #                         st.markdown("**Feature Consistency Details**")
    #                         feature_scores = consistency['feature_consistency']
                            
    #                         col_a, col_b = st.columns(2)
    #                         with col_a:
    #                             st.metric("Min Score", f"{np.min(feature_scores):.3f}")
    #                             st.metric("Max Score", f"{np.max(feature_scores):.3f}")
    #                         with col_b:
    #                             st.metric("Median Score", f"{np.median(feature_scores):.3f}")
    #                             good_consistency = np.sum(np.array(feature_scores) >= 0.7)
    #                             st.metric("High Consistency Graphs", f"{good_consistency}/{len(feature_scores)}")
                            
    #                         # Show distribution of feature consistency scores
    #                         if len(feature_scores) > 1:
    #                             fig, ax = plt.subplots(figsize=(10, 4))
    #                             ax.hist(feature_scores, bins=min(20, len(feature_scores)), alpha=0.7, color='skyblue', edgecolor='black')
    #                             ax.set_xlabel('Feature Consistency Score')
    #                             ax.set_ylabel('Number of Graphs')
    #                             ax.set_title('Distribution of Feature Consistency Scores')
    #                             ax.grid(True, alpha=0.3)
                                
    #                             # Add vertical lines for mean and median
    #                             ax.axvline(np.mean(feature_scores), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(feature_scores):.3f}')
    #                             ax.axvline(np.median(feature_scores), color='green', linestyle='--', alpha=0.8, label=f'Median: {np.median(feature_scores):.3f}')
    #                             ax.legend()
                                
    #                             st.pyplot(fig)
    #                             plt.close(fig)
                        
    #                     elif family_generator.universe.feature_dim == 0:
    #                         st.info("💡 Feature consistency cannot be calculated because the universe has no node features (feature_dim = 0). Generate a universe with feature_dim > 0 to enable feature consistency analysis.")
                        
    #             except Exception as e:
    #                 st.error(f"❌ Error analyzing family properties: {str(e)}")
    #                 st.exception(e)
    #         else:
    #             st.warning("No graph family has been generated yet.")

if __name__ == "__main__":
    main() 