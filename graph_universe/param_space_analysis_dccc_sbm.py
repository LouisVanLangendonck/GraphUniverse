# param_space_exploration.py

"""
Parameter space exploration for DC-SBM and DCCC-SBM graph models.

This script generates diverse individual graphs using both DC-SBM and DCCC-SBM methods,
analyzes their parameters, and visualizes how they cover parameter space.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import random
from collections import defaultdict
import sys
import warnings

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MMSB modules
from graph_universe.model import GraphUniverse, GraphSample

# Parameter ranges to explore
DEFAULT_PARAM_RANGES = {
    # Universe parameters
    "K": {"type": "int", "min": 3, "max": 15},  # Number of communities
    "feature_dim": {"type": "int", "min": 1, "max": 64},  # Feature dimension
    "edge_density": {"type": "float", "min": 0.01, "max": 0.2},  # Base edge density
    "homophily": {"type": "float", "min": 0.0, "max": 1.0},  # Community homophily strength
    "randomness_factor": {"type": "float", "min": 0.0, "max": 1.0},  # Edge randomness
    
    # Sample parameters
    "n_nodes": {"type": "int", "min": 50, "max": 300},  # Number of nodes
    "min_component_size": {"type": "int", "min": 0, "max": 5},  # Min component size
    "degree_heterogeneity": {"type": "float", "min": 0.0, "max": 1.0},  # Degree variability
    "edge_noise": {"type": "float", "min": 0.0, "max": 0.1},  # Edge noise level
    
    # DCCC-SBM specific parameters
    "community_imbalance": {"type": "float", "min": 0.0, "max": 0.6},  # Community size imbalance
    "degree_separation": {"type": "float", "min": 0.4, "max": 1.0},  # Degree distribution separation
    "degree_distribution": {"type": "choice", "options": ["power_law", "exponential", "uniform"]},  # Degree distribution type
    "power_law_exponent": {"type": "float", "min": 2.0, "max": 3.5},  # Power law exponent (if power law)
    "exponential_rate": {"type": "float", "min": 0.01, "max": 0.2},  # Exponential rate (if exponential)
    "uniform_min": {"type": "float", "min": 0.5, "max": 1.0},  # Uniform min factor
    "uniform_max": {"type": "float", "min": 1.0, "max": 2.0},  # Uniform max factor

    # Feature cluster parameters
    "cluster_count_factor": {"type": "float", "min": 0.1, "max": 4.0},  # Number of clusters relative to communities
    "center_variance": {"type": "float", "min": 0.1, "max": 2.0},  # Separation between cluster centers
    "cluster_variance": {"type": "float", "min": 0.01, "max": 0.5},  # Spread within each cluster
    "assignment_skewness": {"type": "float", "min": 0.0, "max": 1.0},  # If some clusters are used more frequently
    "community_exclusivity": {"type": "float", "min": 0.0, "max": 1.0},  # How exclusively clusters map to communities
}

def sample_parameter(param_config):
    """Sample a parameter value according to its configuration."""
    if param_config["type"] == "int":
        return np.random.randint(param_config["min"], param_config["max"] + 1)
    elif param_config["type"] == "float":
        return np.random.uniform(param_config["min"], param_config["max"])
    elif param_config["type"] == "bool":
        return np.random.random() < param_config["prob"]
    elif param_config["type"] == "choice":
        return np.random.choice(param_config["options"])
    else:
        raise ValueError(f"Unknown parameter type: {param_config['type']}")

def generate_random_parameters(param_ranges):
    """Generate random parameters according to the specified ranges."""
    params = {}
    for param_name, param_config in param_ranges.items():
        params[param_name] = sample_parameter(param_config)
    return params

def create_graph(params, use_dccc_sbm=False):
    """
    Create a graph using either DC-SBM or DCCC-SBM method.
    
    Args:
        params: Dictionary of parameters
        use_dccc_sbm: Whether to use DCCC-SBM (True) or DC-SBM (False)
        
    Returns:
        GraphSample object
    """
    # Ensure feature dimension is at least 1
    feature_dim = max(1, params["feature_dim"])
    
    # Create universe
    universe = GraphUniverse(
        K=params["K"],
        feature_dim=feature_dim,  # Ensure non-zero feature dimension
        block_structure="assortative",  # Only using assortative structure
        edge_density=params["edge_density"],
        homophily=params["homophily"],
        randomness_factor=params["randomness_factor"],
        # Add feature cluster parameters
        cluster_count_factor=params["cluster_count_factor"],
        center_variance=params["center_variance"],
        cluster_variance=params["cluster_variance"],
        assignment_skewness=params["assignment_skewness"],
        community_exclusivity=params["community_exclusivity"]
    )
    
    # Sample communities (all of them for simplicity)
    communities = list(range(params["K"]))
    
    # Set degree distribution parameters based on the chosen distribution
    degree_distribution = params.get("degree_distribution", "power_law")
    degree_params = {}
    
    if degree_distribution == "power_law":
        degree_params["power_law_exponent"] = params.get("power_law_exponent", 2.5)
    elif degree_distribution == "exponential":
        degree_params["rate"] = params.get("exponential_rate", 0.01)
    elif degree_distribution == "uniform":
        degree_params["min_factor"] = params.get("uniform_min", 0.5)
        degree_params["max_factor"] = params.get("uniform_max", 1.5)
    
    # Create graph sample
    try:
        graph = GraphSample(
            universe=universe,
            communities=communities,
            n_nodes=params["n_nodes"],
            min_component_size=params["min_component_size"],
            degree_heterogeneity=params["degree_heterogeneity"],
            edge_noise=params["edge_noise"],
            feature_regime_balance=0.5,  # Fixed for simplicity
            target_homophily=None,  # Use universe value
            target_density=None,    # Use universe value
            use_configuration_model=False,  # Not using configuration model
            degree_distribution=degree_distribution,  # Use selected distribution
            power_law_exponent=degree_params.get("power_law_exponent", 2.5),
            target_avg_degree=None,  # Let the model determine this
            triangle_enhancement=0.0,  # No enhancement
            max_mean_community_deviation=0.5,  # Allow significant deviation
            max_max_community_deviation=0.8,  # Allow significant deviation
            max_parameter_search_attempts=5,
            parameter_search_range=0.5,
            min_edge_density=0.005,
            max_retries=3,
            # DCCC-SBM specific parameters
            use_dccc_sbm=use_dccc_sbm,
            community_imbalance=params.get("community_imbalance", 0.0),
            degree_separation=params.get("degree_separation", 0.5),
            degree_method=params.get("degree_method", "standard"),
            dccc_global_degree_params=degree_params,
            disable_deviation_limiting=True  # Allow more graphs to be generated
        )
        
        # Ensure features are generated
        if graph.features is None or graph.features.size == 0:
            graph.regenerate_features()
            
        return graph
    except Exception as e:
        print(f"Error creating graph: {e}")
        return None

def extract_graph_parameters(graph):
    """Extract parameters from a graph sample"""
    if graph is None or graph.graph.number_of_nodes() == 0:
        return None
        
    # Get community analysis
    community_analysis = graph.analyze_community_connections()
    
    # Calculate degree heterogeneity
    degrees = np.array([d for _, d in graph.graph.degree()])
    degree_heterogeneity = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0
    
    # Calculate clustering coefficient
    try:
        clustering_coefficient = float(nx.average_clustering(graph.graph))
    except:
        clustering_coefficient = 0.0
    
    # Calculate actual density
    edge_density = float(nx.density(graph.graph))
    
    # Calculate community signals using the new unified method
    signals = graph.calculate_community_signals(
        structure_metric='kl',
        degree_method="naive_bayes",
        degree_metric="accuracy",
        cv_folds=5,
        random_state=42
    )
    
    # Extract parameters
    params = {
        # Basic graph properties
        "n_nodes": graph.graph.number_of_nodes(),
        "n_edges": graph.graph.number_of_edges(),
        "edge_density": edge_density,
        "clustering_coefficient": clustering_coefficient,
        "degree_heterogeneity": degree_heterogeneity,
        
        # Community structure
        "n_communities": len(graph.communities),
        "mean_community_size": np.mean([len(c) for c in nx.community.label_propagation_communities(graph.graph)]),
        "community_imbalance": graph.community_imbalance if hasattr(graph, 'community_imbalance') else 0.0,
        
        # Signal metrics
        "structure_signal": signals.get('mean_structure_signal', 0.0),
        "feature_signal": signals.get('feature_signal', 0.0),
        "degree_signal": signals.get('degree_signal', 0.0),
        "mean_signal": signals.get('mean_signal', 0.0),
        
        # Community deviations
        "mean_community_deviation": community_analysis.get('mean_deviation', 0.0),
        "max_community_deviation": community_analysis.get('max_deviation', 0.0),
        
        # Degree distribution parameters
        "degree_distribution": graph.degree_distribution if hasattr(graph, 'degree_distribution') else None,
        "degree_separation": graph.degree_separation if hasattr(graph, 'degree_separation') else 0.0,
        "degree_method": graph.degree_method if hasattr(graph, 'degree_method') else "standard",
        
        # Feature parameters
        "feature_dim": graph.features.shape[1] if graph.features is not None else 0,
    }
    
    # Add feature generator parameters if available
    if hasattr(graph.universe, 'feature_generator') and graph.universe.feature_generator is not None:
        feature_gen = graph.universe.feature_generator
        params.update({
            "cluster_count_factor": getattr(feature_gen, '_cluster_count_factor', 0.0),
            "center_variance": getattr(feature_gen, '_center_variance', 0.0),
            "cluster_variance": getattr(feature_gen, '_cluster_variance', 0.0),
            "assignment_skewness": getattr(feature_gen, '_assignment_skewness', 0.0),
            "community_exclusivity": getattr(feature_gen, '_community_exclusivity', 0.0)
        })
    else:
        # Add default values if feature generator is not available
        params.update({
            "cluster_count_factor": 0.0,
            "center_variance": 0.0,
            "cluster_variance": 0.0,
            "assignment_skewness": 0.0,
            "community_exclusivity": 0.0
        })
    
    return params

def generate_diverse_graphs(
    n_graphs_per_method: int,
    param_ranges: Dict[str, Dict[str, Any]] = DEFAULT_PARAM_RANGES,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate diverse graphs using both DC-SBM and DCCC-SBM methods.
    
    Args:
        n_graphs_per_method: Number of graphs to generate per method
        param_ranges: Dictionary defining parameter ranges
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with graph parameters
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    all_params = []
    
    # Generate graphs using DC-SBM
    print(f"Generating {n_graphs_per_method} graphs using DC-SBM...")
    for i in tqdm(range(n_graphs_per_method)):
        # Generate random parameters
        params = generate_random_parameters(param_ranges)
        
        # Create graph
        graph = create_graph(params, use_dccc_sbm=False)
        
        # Extract parameters
        graph_params = extract_graph_parameters(graph)
        if graph_params is not None:
            graph_params["graph_id"] = i
            graph_params["method"] = "DC-SBM"  # Add method information
            all_params.append(graph_params)
    
    # Generate graphs using DCCC-SBM
    print(f"Generating {n_graphs_per_method} graphs using DCCC-SBM...")
    for i in tqdm(range(n_graphs_per_method)):
        # Generate random parameters
        params = generate_random_parameters(param_ranges)
        
        # Create graph
        graph = create_graph(params, use_dccc_sbm=True)
        
        # Extract parameters
        graph_params = extract_graph_parameters(graph)
        if graph_params is not None:
            graph_params["graph_id"] = i + n_graphs_per_method
            graph_params["method"] = "DCCC-SBM"  # Add method information
            all_params.append(graph_params)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_params)
    
    return df

def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    facecolor: str = 'none',
    **kwargs
) -> matplotlib.patches.Ellipse:
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.
    
    Args:
        x, y: Input data
        ax: Matplotlib axes
        n_std: Number of standard deviations for ellipse
        facecolor: Color of ellipse interior
        **kwargs: Additional arguments for ellipse patch
        
    Returns:
        Matplotlib ellipse patch
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the sqrt of the variance and
    # multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_parameter_distributions(
    df: pd.DataFrame,
    params: List[str],
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot distributions of parameters for each method.
    
    Args:
        df: DataFrame with graph parameters
        params: List of parameters to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Filter out parameters that don't exist in the DataFrame
    available_params = [param for param in params if param in df.columns]
    
    if not available_params:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid parameters to plot", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Determine number of rows and columns
    n_params = len(available_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot distributions for each parameter
    for i, param in enumerate(available_params):
        if i < len(axes):
            # Filter out invalid values
            valid_mask = ~df[param].isna()
            valid_df = df[valid_mask]
            
            if valid_df.empty:
                axes[i].text(0.5, 0.5, f"No valid data for {param}", 
                            ha='center', va='center', fontsize=10)
                continue
            
            # Split by method
            for method in ["DC-SBM", "DCCC-SBM"]:
                method_df = valid_df[valid_df["method"] == method]
                if method_df.empty:
                    continue
                    
                color = "blue" if method == "DC-SBM" else "red"
                
                # Check data type and plot accordingly
                sample_value = method_df[param].iloc[0]
                
                if isinstance(sample_value, (int, float)):
                    # Numerical data - use KDE or histogram
                    if len(method_df) > 1:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                sns.kdeplot(
                                    data=method_df,
                                    x=param,
                                    ax=axes[i],
                                    label=method,
                                    color=color,
                                    fill=True,
                                    alpha=0.3
                                )
                        except Exception:
                            # Fall back to histogram if KDE fails
                            axes[i].hist(
                                method_df[param],
                                alpha=0.3,
                                label=method,
                                color=color,
                                bins=10
                            )
                    else:
                        # Single point - plot as vertical line
                        axes[i].axvline(
                            method_df[param].iloc[0],
                            color=color,
                            label=f"{method} (single point)",
                            alpha=0.3
                        )
                
                elif isinstance(sample_value, dict):
                    # Dictionary data - plot as bar chart of keys
                    keys = list(sample_value.keys())
                    values = [method_df[param].apply(lambda x: x.get(k, 0)).mean() for k in keys]
                    x_pos = np.arange(len(keys))
                    axes[i].bar(
                        x_pos + (0.2 if method == "DC-SBM" else -0.2),
                        values,
                        width=0.4,
                        label=method,
                        color=color,
                        alpha=0.3
                    )
                    axes[i].set_xticks(x_pos)
                    axes[i].set_xticklabels(keys, rotation=45, ha='right')
                
                else:
                    # Categorical or other data - use countplot
                    try:
                        sns.countplot(
                            data=method_df,
                            x=param,
                            ax=axes[i],
                            label=method,
                            color=color,
                            alpha=0.3
                        )
                    except Exception:
                        # If countplot fails, show text message
                        axes[i].text(
                            0.5, 0.5,
                            f"Cannot plot {param} for {method}",
                            ha='center', va='center',
                            fontsize=10
                        )
            
            # Set labels and title
            axes[i].set_xlabel(param.replace('_', ' ').title())
            axes[i].set_ylabel("Density" if isinstance(sample_value, (int, float)) else "Count")
            axes[i].set_title(f"Distribution of {param.title()}")
            axes[i].legend()
            
            # Rotate x-axis labels if needed
            if isinstance(sample_value, dict) or (not isinstance(sample_value, (int, float))):
                plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    return fig

def plot_parameter_space(
    df: pd.DataFrame,
    param_x: str,
    param_y: str,
    figsize: Tuple[int, int] = (12, 10),
    real_world_df: Optional[pd.DataFrame] = None
) -> Optional[plt.Figure]:
    """
    Plot parameter space for two parameters.
    
    Args:
        df: DataFrame with graph parameters
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        figsize: Figure size
        real_world_df: DataFrame with real-world graph parameters
        
    Returns:
        Matplotlib figure or None if parameters are not available
    """
    # Check if parameters exist in DataFrame
    if param_x not in df.columns or param_y not in df.columns:
        return None
        
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out invalid values
    valid_mask = ~df[param_x].isna() & ~df[param_y].isna()
    valid_df = df[valid_mask]
    
    if valid_df.empty:
        ax.text(0.5, 0.5, "No valid data points to plot", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Check data types
    sample_x = valid_df[param_x].iloc[0]
    sample_y = valid_df[param_y].iloc[0]
    
    # If either parameter is a dictionary or non-numeric, show error message
    if not (isinstance(sample_x, (int, float)) and isinstance(sample_y, (int, float))):
        ax.text(0.5, 0.5, 
                f"Cannot plot {param_x} vs {param_y}:\nParameters must be numeric", 
                ha='center', va='center', fontsize=12)
        return fig
    
    # Split by method
    dc_sbm_df = valid_df[valid_df["method"] == "DC-SBM"]
    dccc_sbm_df = valid_df[valid_df["method"] == "DCCC-SBM"]
    
    # Plot points for each method
    if not dc_sbm_df.empty:
        ax.scatter(
            dc_sbm_df[param_x], dc_sbm_df[param_y],
            alpha=0.7,
            s=50,
            label="DC-SBM",
            color="blue"
        )
        
        # Plot confidence ellipses for DC-SBM
        if len(dc_sbm_df) > 2:
            try:
                confidence_ellipse(
                    dc_sbm_df[param_x].values, dc_sbm_df[param_y].values,
                    ax, n_std=2.0,
                    edgecolor="blue",
                    linestyle='--',
                    alpha=0.3,
                    facecolor="blue"
                )
            except Exception as e:
                print(f"Warning: Could not plot confidence ellipse for DC-SBM: {e}")
    
    if not dccc_sbm_df.empty:
        ax.scatter(
            dccc_sbm_df[param_x], dccc_sbm_df[param_y],
            alpha=0.7,
            s=50,
            label="DCCC-SBM",
            color="red"
        )
        
        # Plot confidence ellipses for DCCC-SBM
        if len(dccc_sbm_df) > 2:
            try:
                confidence_ellipse(
                    dccc_sbm_df[param_x].values, dccc_sbm_df[param_y].values,
                    ax, n_std=2.0,
                    edgecolor="red",
                    linestyle='--',
                    alpha=0.3,
                    facecolor="red"
                )
            except Exception as e:
                print(f"Warning: Could not plot confidence ellipse for DCCC-SBM: {e}")
    
    # Plot real-world data if provided
    if real_world_df is not None and param_x in real_world_df.columns and param_y in real_world_df.columns:
        valid_mask = ~real_world_df[param_x].isna() & ~real_world_df[param_y].isna()
        valid_real_df = real_world_df[valid_mask]
        
        if not valid_real_df.empty:
            try:
                ax.scatter(
                    valid_real_df[param_x], valid_real_df[param_y],
                    alpha=0.7,
                    s=80,
                    label="Real-World",
                    color="green",
                    marker="^",
                    edgecolor="black"
                )
            except Exception as e:
                print(f"Warning: Could not plot real-world data: {e}")
    
    # Set labels and title
    ax.set_xlabel(param_x.replace('_', ' ').title())
    ax.set_ylabel(param_y.replace('_', ' ').title())
    ax.set_title(f"Parameter Space: {param_x.title()} vs {param_y.title()}")
    
    # Add legend
    ax.legend(loc='best')
    
    return fig

def plot_pca_visualization(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create PCA visualization of parameter space.
    
    Args:
        df: DataFrame with graph parameters
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Select numerical columns (excluding graph_id)
    exclude_cols = ["graph_id", "method"]
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.drop(columns=[col for col in exclude_cols if col in num_df.columns])
    
    # Skip PCA if not enough columns
    if num_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Not enough numerical columns for PCA visualization", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Handle missing values (use mean imputation)
    num_df = num_df.fillna(num_df.mean())
    
    # Skip PCA if still have NaN values
    if num_df.isna().any().any():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Cannot perform PCA due to missing values", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(num_df)
    
    # Apply PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points for each method
    for method, color in zip(["DC-SBM", "DCCC-SBM"], ["blue", "red"]):
        method_mask = df["method"] == method
        method_pca = data_pca[method_mask]
        
        ax.scatter(
            method_pca[:, 0], method_pca[:, 1],
            alpha=0.7,
            s=50,
            label=method,
            color=color
        )
        
        # Plot confidence ellipses for each method
        if len(method_pca) > 2:
            confidence_ellipse(
                method_pca[:, 0], method_pca[:, 1],
                ax, n_std=2.0,
                edgecolor=color,
                linestyle='--',
                alpha=0.3,
                facecolor=color
            )
    
    # Set labels and title
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA Visualization of Graph Parameters")
    
    # Add legend
    ax.legend(loc='best')
    
    # Add loading vectors
    if num_df.shape[1] < 15:  # Only show loadings if not too many parameters
        param_names = num_df.columns
        loading_scale = 5
        for i, param in enumerate(param_names):
            ax.arrow(0, 0, 
                     pca.components_[0, i] * loading_scale, 
                     pca.components_[1, i] * loading_scale,
                     head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.5)
            ax.text(pca.components_[0, i] * loading_scale * 1.1, 
                    pca.components_[1, i] * loading_scale * 1.1, 
                    param.replace('_', ' ').title(), color='black', fontsize=8)
    
    return fig

def load_real_world_datasets(
    output_dir: str = "parameter_coverage_results"
) -> Optional[pd.DataFrame]:
    """
    Load real-world dataset analysis results.
    
    Args:
        output_dir: Directory containing cached results
        
    Returns:
        DataFrame with real-world graph parameters or None if not available
    """
    # Check if summary file exists
    summary_file = os.path.join(output_dir, "real_world_datasets_summary.csv")
    if not os.path.exists(summary_file):
        return None
    
    # Load datasets
    real_world_dfs = []
    
    dataset_files = [f for f in os.listdir(output_dir) if f.startswith("real_world_") and f.endswith("_analysis.csv")]
    for file in dataset_files:
        df = pd.read_csv(os.path.join(output_dir, file))
        dataset_name = file.replace("real_world_", "").replace("_analysis.csv", "")
        df["dataset"] = dataset_name
        real_world_dfs.append(df)
    
    # Combine all datasets
    if real_world_dfs:
        return pd.concat(real_world_dfs, ignore_index=True)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Parameter space exploration for DC-SBM and DCCC-SBM graph models")
    parser.add_argument("--n_graphs", type=int, default=100, help="Number of graphs to generate per method")
    parser.add_argument("--output_dir", type=str, default="param_space_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_real_datasets", action="store_true", help="Include real-world datasets in comparison")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate graphs
    df = generate_diverse_graphs(
        n_graphs_per_method=args.n_graphs,
        seed=args.seed
    )
    
    # Save generated data
    df.to_csv(os.path.join(args.output_dir, "graph_parameters.csv"), index=False)
    
    # Load real-world datasets if requested
    real_world_df = None
    if args.include_real_datasets:
        real_world_df = load_real_world_datasets()
    
    # Define parameters to analyze (only include those that exist in the DataFrame)
    basic_params = [
        "homophily",
        "clustering_coefficient",
        "avg_degree",
        "edge_density",
        "n_nodes"
    ]
    
    # Only add power_law_exponent if it exists in the DataFrame
    if "power_law_exponent" in df.columns:
        basic_params.append("power_law_exponent")
    
    signal_params = [
        "feature_signal",
        "structure_signal",
        "degree_signal"
    ]
    
    targeted_params = [
        "target_homophily",
        "target_density",
        "homophily_ratio",
        "density_ratio"
    ]
    
    # Filter out parameters that don't exist in the DataFrame
    all_params = [param for param in basic_params + signal_params + targeted_params if param in df.columns]
    
    # Create parameter distribution plots
    dist_fig = plot_parameter_distributions(df, all_params)
    dist_fig.savefig(os.path.join(args.output_dir, "parameter_distributions.png"), dpi=300, bbox_inches='tight')
    
    # Create PCA visualization
    pca_fig = plot_pca_visualization(df)
    pca_fig.savefig(os.path.join(args.output_dir, "pca_visualization.png"), dpi=300, bbox_inches='tight')
    
    # Create parameter space plots for various pairs
    param_pairs = [
        # Basic parameter pairs
        ("homophily", "edge_density"),
        ("homophily", "clustering_coefficient"),
        ("avg_degree", "edge_density"),
        ("edge_density", "clustering_coefficient"),
        ("n_nodes", "avg_degree"),
        
        # Signal parameter pairs
        ("feature_signal", "structure_signal"),
        ("feature_signal", "degree_signal"),
        ("structure_signal", "degree_signal"),
        
        # Cross-category pairs
        ("homophily", "feature_signal"),
        ("edge_density", "structure_signal"),
        
        # Target vs. actual pairs
        ("target_homophily", "homophily"),
        ("target_density", "edge_density"),
    ]
    
    # Only add power law related pairs if the parameter exists
    if "power_law_exponent" in df.columns:
        param_pairs.extend([
            ("avg_degree", "power_law_exponent"),
            ("power_law_exponent", "degree_signal")
        ])
    
    # Generate parameter space plots
    for param_x, param_y in param_pairs:
        if param_x in df.columns and param_y in df.columns:
            fig = plot_parameter_space(df, param_x, param_y, real_world_df=real_world_df)
            if fig is not None:
                fig.savefig(os.path.join(args.output_dir, f"param_space_{param_x}_vs_{param_y}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()