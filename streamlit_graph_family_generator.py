import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphSample, GraphFamilyGenerator
from utils.visualizations import (
    plot_graph_communities, 
    plot_universe_degree_centers
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
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🌌 Universe Generation", "📊 Graph Family Generation", "💾 Download & Analysis"])
    
    # TAB 1: UNIVERSE GENERATION
    with tab1:
        st.markdown('<h2 class="section-header">Graph Universe Generation</h2>', unsafe_allow_html=True)
        st.markdown("**Step 1:** Create a new universe or load an existing one. The universe defines the communities, their features, and degree centers.")
        
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
                            'edge_probability_variance': universe.edge_probability_variance,
                            'seed': 'unknown'  # We don't know the original seed
                        }
                        
                        st.success("✅ Universe loaded successfully!")
                        
                        # Display loaded universe info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Communities (K)", universe.K)
                            st.metric("Feature Dimension", universe.feature_dim)
                        with col_info2:
                            st.metric("Edge Probability Variance", f"{universe.edge_probability_variance:.3f}")
                            st.metric("Degree Centers Shape", f"{universe.degree_centers.shape}")
                        
                        # Show universe properties
                        if st.checkbox("Show Loaded Universe Properties", key="loaded_universe_props"):
                            st.markdown("#### Degree Centers")
                            fig_degree_centers = plot_universe_degree_centers(universe)
                            st.pyplot(fig_degree_centers)
                            plt.close(fig_degree_centers)
                            
                            st.markdown("#### Probability Matrix")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(universe.P, cmap='viridis', aspect='auto')
                            plt.colorbar(im, ax=ax, label="Edge Probability")
                            ax.set_title("Inter-Community Probability Matrix")
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
                    K = st.slider("Number of Communities (K)", 2, 20, 10, help="Number of communities in the universe")
                    edge_probability_variance = st.slider(
                        "Edge Probability Variance", 
                        0.0, 1.0, 0.5, 0.01,
                        help="Amount of variance in edge probabilities"
                    )
                
                with col_b:
                    feature_dim = st.slider("Feature Dimension", 0, 50, 15, help="Dimension of node features (0 for no features)")
            
                # Feature generation parameters
                if feature_dim > 0:
                    st.markdown("#### Feature Generation Parameters")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        center_variance = st.slider(
                            "Center Variance", 
                            0.1, 5.0, 0.1, 0.01,
                            help="Separation between cluster centers"
                        )
                    with col_d:
                        cluster_variance = st.slider(
                            "Cluster Variance", 
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
                            edge_probability_variance=edge_probability_variance,
                            center_variance=center_variance,
                            cluster_variance=cluster_variance,
                            seed=universe_seed
                        )
                        
                        # Store in session state
                        st.session_state.universe = universe
                        st.session_state.universe_params = {
                            'K': K,
                            'feature_dim': feature_dim,
                            'edge_probability_variance': edge_probability_variance,
                            'center_variance': center_variance,
                            'cluster_variance': cluster_variance,
                            'seed': universe_seed
                        }
                        
                    st.success("✅ Universe generated successfully!")
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
                st.metric("Edge Prob Variance", f"{universe.edge_probability_variance:.3f}")
            else:
                st.warning("⚠️ No Universe Available")
                st.info("Create a new universe or load an existing one to proceed with graph family generation.")
        
        with col_status2:
            if 'universe' in st.session_state and st.session_state.universe is not None:
                universe = st.session_state.universe
                
                # Universe properties visualization
                if st.checkbox("Show Universe Properties", key="universe_props_tab1"):
                    st.markdown("#### Degree Centers")
                    fig_degree_centers = plot_universe_degree_centers(universe)
                    st.pyplot(fig_degree_centers)
                    plt.close(fig_degree_centers)
                    
                    st.markdown("#### Probability Matrix")
                    # Create a simple probability matrix visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(universe.P, cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, label="Edge Probability")
                    ax.set_title("Inter-Community Probability Matrix")
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
    
    # TAB 2: GRAPH FAMILY GENERATION  
    with tab2:
        st.markdown('<h2 class="section-header">Graph Family Generation</h2>', unsafe_allow_html=True)
        st.markdown("**Step 2:** Generate a family of graphs using the created universe with different structural properties.")
        
        # Check if universe exists
        if 'universe' not in st.session_state or st.session_state.universe is None:
            st.error("❌ No universe available. Please generate a universe first in the 'Universe Generation' tab.")
            return
        
        # Sidebar for graph family parameters
        with st.sidebar:
            st.markdown('<h2 class="section-header">Graph Family Parameters</h2>', unsafe_allow_html=True)
            
            col_nodes_a, col_nodes_b = st.columns(2)
            with col_nodes_a:
                min_n_nodes = st.number_input("Min Nodes", min_value=10, max_value=10000, value=100, step=10, help="Minimum number of nodes per graph")
            with col_nodes_b:
                max_n_nodes = st.number_input("Max Nodes", min_value=20, max_value=10000, value=300, step=10, help="Maximum number of nodes per graph")
            
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
             
            # Generation parameters
            st.markdown("### Generation Parameters")
            
            n_graphs = st.slider("Number of Graphs", 1, 2000, 100, help="Number of graphs to generate")
            
            family_seed = st.number_input("Family Seed", value=123, help="Random seed for graph family generation")
            
            timeout_minutes = st.slider("Timeout (minutes)", 1.0, 30.0, 5.0, help="Timeout for generation")
            
        # Main generation area
        universe = st.session_state.universe
        
        # Display current universe info
        st.info(f"🌌 Using universe with {universe.K} communities, {universe.feature_dim}D features")
        
        # Generate Graph Family Button
        if st.button("📊 Generate Graph Family", type="primary", use_container_width=True):
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
                use_dccc_sbm=use_dccc_sbm,
                degree_distribution=degree_distribution,
                power_law_exponent_range=(power_law_min, power_law_max),
                exponential_rate_range=(exp_min, exp_max),
                uniform_min_factor_range=(uniform_min_min, uniform_min_max),
                uniform_max_factor_range=(uniform_max_min, uniform_max_max),
                degree_separation_range=(degree_separation_min, degree_separation_max),
                degree_heterogeneity=degree_heterogeneity,
                seed=family_seed
            )
            
            progress_bar.progress(33)
            progress_text.text("33% - Family generator ready")
            
            # Step 2: Generate family
            status_text.text("Step 2/3: Generating Graph Family...")
            
            start_time = time.time()
            
            # Use the family generator's generate_family method
            graphs = family_generator.generate_family(
                n_graphs=n_graphs,
                show_progress=True,
                collect_stats=True,
                timeout_minutes=timeout_minutes
            )
            
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
    
    # TAB 3: DOWNLOAD & ANALYSIS
    with tab3:
        st.markdown('<h2 class="section-header">Download & Analysis</h2>', unsafe_allow_html=True)
        st.markdown("**Step 3:** Download your generated graph families and analyze their properties.")
        
        # Check if graphs exist
        if 'graphs' not in st.session_state or not st.session_state.graphs:
            st.error("❌ No graphs available. Please generate a graph family first in the 'Graph Family Generation' tab.")
            return
        
        graphs = st.session_state.graphs
        
        # DOWNLOAD SECTION - Always visible and prominent
        st.markdown("### 💾 Download PyTorch Geometric Graphs")
        
        # Check if family generator exists in session state
        if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
            family_generator = st.session_state.family_generator
            
            if len(family_generator.graphs) > 0:
                # Task selection
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Select Tasks and Configure Download")
                    available_tasks = [
                        "community_detection",
                        "triangle_counting",
                        "k_hop_community_counts_k1",
                        "k_hop_community_counts_k2", 
                        "k_hop_community_counts_k3"
                    ]
                    
                    selected_tasks = st.multiselect(
                        "Choose tasks to generate PyG graphs for:",
                        available_tasks,
                        default=["community_detection"],
                        help="Select one or more tasks. Each task will create different target labels for the graphs."
                    )
                    
                    # Family naming and directory
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        family_id = st.text_input(
                            "Family ID",
                            value="family_001",
                            help="Alphanumeric identifier for this graph family"
                        )
                    
                    with col_b:
                        family_dir = st.text_input(
                            "Save Directory",
                            value="graph_family",
                            help="Directory where the PyG graphs will be saved"
                        )
                
                with col2:
                    st.markdown("#### Download Status")
                    st.info(f"Ready to convert and save {len(family_generator.graphs)} graphs to PyG format.")
                    
                    # Show what files will be created
                    if selected_tasks and family_id:
                        st.markdown("**Files to be created:**")
                        for task in selected_tasks:
                            st.code(f"{family_dir}/pyg_graph_list_{family_id}_{task}.pkl")
                        st.code(f"{family_dir}/graph_universe_{family_id}.pkl")
                        st.code(f"{family_dir}/metadata_{family_id}.json")
                
                # Validation and download button
                valid_family_id = family_id.replace("_", "").isalnum() if family_id else False
                if not valid_family_id:
                    st.error("Family ID must be alphanumeric (underscores allowed)")
                
                if not selected_tasks:
                    st.warning("Please select at least one task to proceed.")
                
                # Prominent download button
                if valid_family_id and selected_tasks:
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

                            # Get uniquely identifying metadata
                            uniquely_identifying_metadata = family_generator.get_uniquely_identifying_metadata(n_graphs=len(family_generator.graphs))

                            
                            # Use the family generator's save method
                            family_generator.save_pyg_graphs_and_universe(
                                n_graphs=n_graphs,
                                tasks=selected_tasks,
                                uniquely_identifying_metadata=uniquely_identifying_metadata,
                                family_dir=family_dir
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
        
        # ANALYSIS SECTION
        st.markdown("---")
        st.markdown("### 📊 Graph Analysis & Visualization")
        
        # Graph selection and visualization
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
            figsize=(12, 8),
            title=f"Graph {graph_idx+1} - {selected_graph.n_nodes} nodes, {selected_graph.graph.number_of_edges()} edges"
        )
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Graph statistics
        st.markdown("#### Graph Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", selected_graph.n_nodes)
            st.metric("Edges", selected_graph.graph.number_of_edges())
        
        with col2:
            st.metric("Communities", len(selected_graph.communities))
            st.metric("Components", len(list(nx.connected_components(selected_graph.graph))))
        
        with col3:
            if selected_graph.features is not None:
                st.metric("Feature Dim", selected_graph.features.shape[1])
            else:
                st.metric("Feature Dim", 0)
            st.metric("Generation Method", selected_graph.generation_method)
        
        with col4:
            # Calculate some additional metrics
            degrees = [d for _, d in selected_graph.graph.degree()]
            if degrees:
                st.metric("Avg Degree", f"{2 * selected_graph.graph.number_of_edges() / selected_graph.n_nodes:.2f}")
                st.metric("Degree Std", f"{np.std(degrees):.2f}")
            else:
                st.metric("Avg Degree", "0")
                st.metric("Degree Std", "0")
        
        # Additional analysis options
        st.markdown("---")
        st.markdown("#### Additional Analysis Options")
        
        # Community analysis
        if st.checkbox("Show Community Analysis"):
            st.markdown("##### Community Analysis")
            
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
        
        # Probability Matrices Comparison
        if st.checkbox("Show Probability Matrices Comparison"):
            st.markdown("##### Probability Matrices Comparison")
            
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
            else:
                st.warning("Selected graph is not a GraphSample object, so probability matrices cannot be displayed.")
        
        # Family-level analysis
        if st.checkbox("Show Family-level Analysis"):
            st.markdown("##### Family-level Analysis")
            
            # Check if family generator exists in session state
            if 'family_generator' in st.session_state and st.session_state.family_generator is not None:
                family_generator = st.session_state.family_generator
                
                # Family summary
                family_properties = family_generator.analyze_graph_family_properties()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Family Properties**")
                    for key, value in family_properties.items():
                        if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                            st.metric(key, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
                
                with col2:
                    st.markdown("**Family Consistency**")
                    consistency = family_generator.analyze_graph_family_consistency()
                    for metric, value in consistency.items():
                        if isinstance(value, list):
                            st.metric(metric, f"{np.mean(value):.3f} ± {np.std(value):.3f}")
                        else:
                            st.metric(metric, value)
            else:
                st.warning("No graph family has been generated yet.")

if __name__ == "__main__":
    main() 