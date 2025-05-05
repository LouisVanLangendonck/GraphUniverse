"""
Parameter space coverage analysis for MMSB graph families.

This script analyzes how different families of graphs generated with the 
Mixed-Membership Stochastic Block Model (MMSB) cover the parameter space
of key graph properties. It helps demonstrate that these synthetic graphs
can be used to test GNN models on properties not found in typical benchmarks.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import glob
import pickle
import random
from collections import defaultdict
import matplotlib

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MMSB modules
from mmsb.model import GraphUniverse, GraphSample
from mmsb.graph_family import GraphFamilyGenerator
from utils.parameter_analysis import (
    analyze_graph_parameters,
    compute_statistics,
    plot_parameter_space,
    create_parameter_dashboard
)
from utils.sampler import GraphSampler


def generate_diverse_graph_families(
    n_families: int = 10,
    n_graphs_per_family: int = 20,
    base_params: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, List[GraphSample]]:
    """
    Generate diverse families of graphs by varying key parameters.
    
    Args:
        n_families: Number of graph families to generate
        n_graphs_per_family: Number of graphs per family
        base_params: Base parameters for all families
        seed: Random seed
        
    Returns:
        Dictionary mapping family names to lists of graph samples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Parameter variations to create diverse families
    family_variations = []
    
    print(f"\nGenerating {n_families} diverse graph families...")
    
    # Ensure we have at least n_families parameter sets
    while len(family_variations) < n_families:
        # Create random variations if needed
        K = random.choice([10, 15, 20, 30])  # Reduced max K to avoid sampling issues
        
        # Generate edge densities first so we can use them to constrain connection strength
        edge_density = random.uniform(0.03, 0.3)
        inter_community_density = random.uniform(0.03, 0.3)
        
        # Connection strength should never exceed the minimum edge probability
        max_connection_strength = min(edge_density, inter_community_density)
        min_connection_strength = random.uniform(0.01, max_connection_strength)
        
        new_variation = {
            # Universe parameters
            "K": K,
            "feature_dim": 32,
            "feature_signal": 1.0,
            "block_structure": random.choice(["assortative"]), #, "disassortative", "hierarchical", "core-periphery"]),
            "overlap_structure": random.choice(["modular"]), #, "hierarchical", "hub-spoke"]),
            "edge_density": edge_density,
            "inter_community_density": inter_community_density,
            "overlap_density": random.uniform(0.05, 0.4),
            "randomness_factor": random.uniform(0.0, 0.7),
            
            # Graph generation parameters
            "min_communities_ratio": random.uniform(0.1, 0.3),  # Increased minimum ratio
            "max_communities_ratio": random.uniform(0.2, 0.6),  # Adjusted maximum ratio
            "min_nodes": random.randint(50, 100),
            "max_nodes": random.randint(100, 200),
            "degree_heterogeneity": random.uniform(0.1, 0.9),
            "edge_noise": random.uniform(0.0, 0.1),
            "sampling_method": random.choice(["random"]), #, "similar", "diverse"]),
            "similarity_bias": random.uniform(-0.5, 0.5),
            "min_connection_strength": min_connection_strength,
            "min_component_size": random.randint(3,7),
            "indirect_influence": random.uniform(0, 0.2),
            
            "name": f"random_variation_{len(family_variations)}"
        }
        family_variations.append(new_variation)
    
    # Select the first n_families variations
    family_variations = family_variations[:n_families]
    
    # Generate graph families
    families = {}
    
    for variation in tqdm(family_variations):
        family_name = variation.pop("name")
        
        # Create parameters for this family
        family_params = base_params.copy() if base_params else {}
        family_params.update(variation)
        
        # Create benchmark instance
        try:
            benchmark = MMSBBenchmark(
                K=family_params["K"],
                feature_dim=family_params.get("feature_dim", 32),
                feature_signal=family_params.get("feature_signal", 1.0),
                block_structure=family_params["block_structure"],
                overlap_structure=family_params["overlap_structure"],
                edge_density=family_params["edge_density"],
                inter_community_density=family_params["inter_community_density"],
                overlap_density=family_params["overlap_density"],
                randomness_factor=family_params["randomness_factor"]
            )
            
            # Calculate community ranges based on K
            min_communities = max(2, int(family_params["K"] * family_params["min_communities_ratio"]))
            max_communities = max(min_communities + 1, int(family_params["K"] * family_params["max_communities_ratio"]))
            
            # Only print parameters if generation fails
            failure_occurred = False
            
            # Generate graphs using pretraining method
            try:
                graphs = benchmark.generate_pretraining_graphs(
                    n_graphs=n_graphs_per_family,
                    min_communities=min_communities,
                    max_communities=max_communities,
                    min_nodes=family_params["min_nodes"],
                    max_nodes=family_params["max_nodes"],
                    degree_heterogeneity=family_params["degree_heterogeneity"],
                    edge_noise=family_params["edge_noise"],
                    sampling_method=family_params["sampling_method"],
                    similarity_bias=family_params.get("similarity_bias", 0.0),
                    min_connection_strength=family_params.get("min_connection_strength", 0.05),
                    min_component_size=family_params.get("min_component_size", 0),
                    indirect_influence=family_params.get("indirect_influence", 0.1),
                )
                
                if graphs is None or len(graphs) == 0:
                    failure_occurred = True
                else:
                    families[family_name] = graphs
                    
            except Exception as e:
                failure_occurred = True
                
            # If failure occurred, print the parameters and error details
            if failure_occurred:
                print(f"\nFAILED family: {family_name}")
                print("Parameters:")
                for key, value in family_params.items():
                    print(f"  {key}: {value}")
                print(f"  min_communities: {min_communities}")
                print(f"  max_communities: {max_communities}")
                if graphs is None:
                    print("Error: generate_pretraining_graphs returned None")
                elif len(graphs) == 0:
                    print("Error: generate_pretraining_graphs returned empty list")
                
        except Exception as e:
            print(f"\nFAILED family: {family_name}")
            print("Parameters:")
            for key, value in family_params.items():
                print(f"  {key}: {value}")
            print(f"Error: {str(e.__class__.__name__)}: {str(e)}")
    
    print(f"\nSuccessfully generated {len(families)} out of {n_families} families")
    return families


def analyze_families(
    families: Dict[str, List[GraphSample]],
    param_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Analyze graph families to extract parameter distributions.
    
    Args:
        families: Dictionary mapping family names to lists of graph samples
        param_names: List of parameter names to analyze
        
    Returns:
        Dictionary mapping family names to parameter DataFrames
    """
    if param_names is None:
        param_names = [
            "homophily",
            "power_law_exponent",
            "clustering_coefficient",
            "triangle_density",
            "node_count",
            "avg_degree",
            "avg_communities_per_node",
            "density",
            "connected_components",
            "largest_component_size"
        ]
    
    family_dfs = {}
    print(f"Analyzing parameter distributions for {len(families)} graph families...")
    
    for family_name, graphs in tqdm(families.items()):
        # Analyze graphs in this family
        df = analyze_graph_family(graphs)
        
        # Store DataFrame
        family_dfs[family_name] = df
    
    return family_dfs


def analyze_graph_family(graphs: List[GraphSample]) -> pd.DataFrame:
    """
    Analyze parameters for a family of graphs.
    
    Args:
        graphs: List of graph samples
        
    Returns:
        DataFrame with parameter values for each graph
    """
    results = []
    
    for i, graph in enumerate(graphs):
        # Extract parameters from graph
        params = graph.extract_parameters()
        params["graph_id"] = i
        
        results.append(params)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def compute_family_statistics(
    family_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute statistics for each parameter in each family.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        
    Returns:
        Nested dictionary with statistics for each parameter in each family
    """
    family_stats = {}
    
    for family_name, df in family_dfs.items():
        family_stats[family_name] = compute_statistics(df)
    
    return family_stats


def plot_parameter_distribution_by_family(
    family_dfs: Dict[str, pd.DataFrame],
    param: str,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot distribution of a parameter across different families.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        param: Parameter to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Prepare data for plotting
    data = []
    
    for family_name, df in family_dfs.items():
        if param in df.columns:
            for value in df[param].dropna():
                data.append({"Family": family_name, "Value": value})
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(data)
    
    if plot_df.empty:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for parameter: {param}", 
                ha='center', va='center', fontsize=14)
        ax.set_title(f"Parameter Distribution: {param}")
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(x="Family", y="Value", data=plot_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add summary statistics as text
    family_stats = {}
    for family_name, df in family_dfs.items():
        if param in df.columns:
            values = df[param].dropna()
            if not values.empty:
                family_stats[family_name] = {
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std()
                }
    
    # Add text annotations
    y_pos = 0.02
    for family_name, stats in family_stats.items():
        ax.text(
            0.02, y_pos, 
            f"{family_name}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}",
            transform=ax.transAxes, fontsize=8
        )
        y_pos += 0.03
    
    ax.set_title(f"Distribution of {param} Across Graph Families")
    
    return fig


def compute_parameter_space_coverage(
    family_dfs: Dict[str, pd.DataFrame],
    real_world_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    param_pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Compute the coverage of parameter space by different graph families.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        real_world_dfs: Dictionary mapping real-world dataset names to DataFrames (optional)
        param_pairs: List of parameter pairs to analyze
        
    Returns:
        Dictionary with coverage metrics for each parameter pair
    """
    if param_pairs is None:
        param_pairs = [
            ("homophily", "avg_degree"),
            ("homophily", "clustering_coefficient"),
            ("avg_degree", "power_law_exponent"),
            ("avg_degree", "density"),
            ("clustering_coefficient", "triangle_density"),
            ("power_law_exponent", "avg_communities_per_node")
        ]
    
    coverage_metrics = {}
    
    for param_x, param_y in param_pairs:
        # Check that both parameters are present in at least some families
        param_present = False
        for df in family_dfs.values():
            if param_x in df.columns and param_y in df.columns:
                param_present = True
                break
        
        if not param_present:
            continue
        
        # Collect all values for this parameter pair
        all_x_values = []
        all_y_values = []
        family_points = defaultdict(list)
        
        for family_name, df in family_dfs.items():
            if param_x in df.columns and param_y in df.columns:
                # Get valid pairs of values (where both x and y are not NaN)
                valid_mask = ~df[param_x].isna() & ~df[param_y].isna()
                valid_x = df.loc[valid_mask, param_x].values
                valid_y = df.loc[valid_mask, param_y].values
                
                # Only proceed if we have matching lengths
                if len(valid_x) == len(valid_y) and len(valid_x) > 0:
                    all_x_values.extend(valid_x)
                    all_y_values.extend(valid_y)
                    # Store points for this family
                    points = list(zip(valid_x, valid_y))
                    family_points[family_name] = points
        
        # Real-world data points (if available)
        real_world_points = defaultdict(list)
        if real_world_dfs is not None:
            for dataset_name, df in real_world_dfs.items():
                if param_x in df.columns and param_y in df.columns:
                    valid_mask = ~df[param_x].isna() & ~df[param_y].isna()
                    valid_x = df.loc[valid_mask, param_x].values
                    valid_y = df.loc[valid_mask, param_y].values
                    
                    if len(valid_x) == len(valid_y) and len(valid_x) > 0:
                        points = list(zip(valid_x, valid_y))
                        real_world_points[dataset_name] = points
        
        # Compute coverage metrics
        coverage = {}
        
        # Range coverage
        if all_x_values and all_y_values:
            x_min, x_max = min(all_x_values), max(all_x_values)
            y_min, y_max = min(all_y_values), max(all_y_values)
            
            coverage["x_range"] = (x_min, x_max)
            coverage["y_range"] = (y_min, y_max)
            coverage["area"] = (x_max - x_min) * (y_max - y_min)
        
            # Standard deviation coverage
            coverage["x_std"] = np.std(all_x_values)
            coverage["y_std"] = np.std(all_y_values)
            
            # Compute 2D histogram for density coverage
            try:
                hist, x_edges, y_edges = np.histogram2d(
                    all_x_values, all_y_values, bins=10
                )
                
                # Calculate percentage of bins that are non-empty
                total_bins = hist.size
                non_empty_bins = np.count_nonzero(hist)
                coverage["bin_coverage"] = non_empty_bins / total_bins
                
                # Calculate entropy of the distribution
                hist_flat = hist.flatten()
                hist_norm = hist_flat / hist_flat.sum() if hist_flat.sum() > 0 else hist_flat
                hist_norm = hist_norm[hist_norm > 0]  # Keep only non-zero probabilities
                entropy = -np.sum(hist_norm * np.log2(hist_norm)) if len(hist_norm) > 0 else 0
                max_entropy = np.log2(total_bins) if total_bins > 0 else 1  # Maximum possible entropy
                coverage["entropy_ratio"] = entropy / max_entropy if max_entropy > 0 else 0
            except Exception as e:
                print(f"Error computing histogram for {param_x} vs {param_y}: {e}")
                coverage["bin_coverage"] = 0.0
                coverage["entropy_ratio"] = 0.0
                
            # Store points for plotting
            coverage["family_points"] = family_points
            coverage["real_world_points"] = real_world_points
        
        coverage_metrics[(param_x, param_y)] = coverage
    
    return coverage_metrics



def plot_parameter_space_coverage(
    coverage_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 12)
) -> Dict[Tuple[str, str], plt.Figure]:
    """
    Plot parameter space coverage for different parameter pairs.
    
    Args:
        coverage_metrics: Dictionary with coverage metrics for each parameter pair
        figsize: Figure size
        
    Returns:
        Dictionary mapping parameter pairs to figures
    """
    figures = {}
    
    for (param_x, param_y), coverage in coverage_metrics.items():
        if "family_points" not in coverage:
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points for each family
        family_points = coverage["family_points"]
        real_world_points = coverage.get("real_world_points", {})
        
        # Define colors for families
        family_colors = {}
        cmap = plt.cm.tab20
        for i, family_name in enumerate(family_points.keys()):
            family_colors[family_name] = cmap(i % cmap.N)
        
        # Plot synthetic family points
        for family_name, points in family_points.items():
            if points:
                x_values, y_values = zip(*points)
                ax.scatter(
                    x_values, y_values, 
                    alpha=0.5,
                    s=30,
                    label=family_name,
                    color=family_colors[family_name]
                )
                
                # Plot confidence ellipses for each family
                if len(points) > 2:
                    try:
                        confidence_ellipse(
                            np.array(x_values), np.array(y_values),
                            ax, n_std=2.0,
                            edgecolor=family_colors[family_name],
                            linestyle='--',
                            alpha=0.3,
                            facecolor=family_colors[family_name]
                        )
                    except:
                        pass  # Skip if ellipse cannot be computed
        
        # Plot real-world points (if available)
        for dataset_name, points in real_world_points.items():
            if points:
                x_values, y_values = zip(*points)
                ax.scatter(
                    x_values, y_values,
                    marker='X',
                    s=100,
                    label=f"Real: {dataset_name}",
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=100
                )
        
        # Set labels and title
        ax.set_xlabel(param_x.replace('_', ' ').title())
        ax.set_ylabel(param_y.replace('_', ' ').title())
        ax.set_title(f"Parameter Space Coverage: {param_x.title()} vs {param_y.title()}")
        
        # Add coverage metrics as text
        metrics_text = [
            f"Range X: [{coverage['x_range'][0]:.2f}, {coverage['x_range'][1]:.2f}]",
            f"Range Y: [{coverage['y_range'][0]:.2f}, {coverage['y_range'][1]:.2f}]",
            f"Area: {coverage['area']:.2f}",
            f"Bin Coverage: {coverage['bin_coverage']:.2f}",
            f"Entropy Ratio: {coverage['entropy_ratio']:.2f}"
        ]
        
        ax.text(
            0.02, 0.98, 
            "\n".join(metrics_text),
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Add legend
        if len(family_points) + len(real_world_points) < 15:
            ax.legend(loc='best')
        else:
            ax.legend(loc='best', fontsize=8)
        
        figures[(param_x, param_y)] = fig
    
    return figures


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


def load_real_world_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load and analyze real-world datasets for comparison.
    This is a placeholder function - implement with actual datasets.
    
    Returns:
        Dictionary mapping dataset names to parameter DataFrames
    """
    # Placeholder - replace with actual loading and analysis of real-world datasets
    real_world_dfs = {}
    
    # Example: manually create some datapoints representing real-world datasets
    # based on values from literature like GraphWorld paper
    
    # Cora
    cora_df = pd.DataFrame({
        "homophily": [0.83],
        "avg_degree": [4.0],
        "clustering_coefficient": [0.24],
        "power_law_exponent": [2.1],
        "density": [0.002],
        "triangle_density": [0.09],
        "avg_communities_per_node": [1.0],
    })
    real_world_dfs["Cora"] = cora_df
    
    # Citeseer
    citeseer_df = pd.DataFrame({
        "homophily": [0.74],
        "avg_degree": [2.8],
        "clustering_coefficient": [0.18],
        "power_law_exponent": [2.2],
        "density": [0.001],
        "triangle_density": [0.05],
        "avg_communities_per_node": [1.0],
    })
    real_world_dfs["Citeseer"] = citeseer_df
    
    # PubMed
    pubmed_df = pd.DataFrame({
        "homophily": [0.79],
        "avg_degree": [5.5],
        "clustering_coefficient": [0.04],
        "power_law_exponent": [2.3],
        "density": [0.0003],
        "triangle_density": [0.02],
        "avg_communities_per_node": [1.0],
    })
    real_world_dfs["PubMed"] = pubmed_df
    
    # OGB-ArXiv
    ogb_arxiv_df = pd.DataFrame({
        "homophily": [0.65],
        "avg_degree": [13.2],
        "clustering_coefficient": [0.21],
        "power_law_exponent": [2.4],
        "density": [0.0001],
        "triangle_density": [0.12],
        "avg_communities_per_node": [1.0],
    })
    real_world_dfs["OGB-ArXiv"] = ogb_arxiv_df
    
    # OGB-Products
    ogb_products_df = pd.DataFrame({
        "homophily": [0.51],
        "avg_degree": [20.9],
        "clustering_coefficient": [0.37],
        "power_law_exponent": [2.1],
        "density": [0.00001],
        "triangle_density": [0.18],
        "avg_communities_per_node": [1.0],
    })
    real_world_dfs["OGB-Products"] = ogb_products_df
    
    return real_world_dfs


def create_summary_figures(
    family_dfs: Dict[str, pd.DataFrame],
    coverage_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: str = "results"
) -> None:
    """
    Create and save summary figures.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        coverage_metrics: Dictionary with coverage metrics for each parameter pair
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save parameter distribution plots
    all_params = set()
    for df in family_dfs.values():
        all_params.update(df.columns)
    
    all_params = [p for p in all_params if p != "graph_id"]
    
    for param in all_params:
        try:
            fig = plot_parameter_distribution_by_family(family_dfs, param)
            fig.savefig(os.path.join(output_dir, f"distribution_{param}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting distribution for {param}: {e}")
    
    # Create and save parameter space coverage plots
    param_space_figs = plot_parameter_space_coverage(coverage_metrics)
    for (param_x, param_y), fig in param_space_figs.items():
        try:
            filename = f"coverage_{param_x}_vs_{param_y}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving coverage plot for {param_x} vs {param_y}: {e}")
    
    # Create a PCA visualization of the parameter space
    try:
        fig = plot_parameter_space_pca(family_dfs)
        fig.savefig(os.path.join(output_dir, "parameter_space_pca.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error creating PCA visualization: {e}")


def create_coverage_tables(
    coverage_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: str = "results"
) -> None:
    """
    Create and save tables of coverage metrics.
    
    Args:
        coverage_metrics: Dictionary with coverage metrics for each parameter pair
        output_dir: Directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame of coverage metrics
    rows = []
    for (param_x, param_y), metrics in coverage_metrics.items():
        if "area" in metrics:
            row = {
                "Parameter X": param_x.replace('_', ' ').title(),
                "Parameter Y": param_y.replace('_', ' ').title(),
                "X Range": f"[{metrics['x_range'][0]:.2f}, {metrics['x_range'][1]:.2f}]",
                "Y Range": f"[{metrics['y_range'][0]:.2f}, {metrics['y_range'][1]:.2f}]",
                "Area": f"{metrics['area']:.2f}",
                "Bin Coverage": f"{metrics['bin_coverage']:.2f}",
                "Entropy Ratio": f"{metrics['entropy_ratio']:.2f}"
            }
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "coverage_metrics.csv"), index=False)
        
        # Create LaTeX table for publications
        with open(os.path.join(output_dir, "coverage_metrics.tex"), "w") as f:
            f.write(df.to_latex(index=False))


def plot_parameter_space_pca(
    family_dfs: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create PCA visualization of parameter space.
    
    Args:
        family_dfs: Dictionary mapping family names to parameter DataFrames
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Collect all parameters across all families
    all_params = set()
    for df in family_dfs.values():
        all_params.update(df.columns)
    
    # Filter out non-numeric and irrelevant columns
    exclude_cols = ["graph_id"]
    params = [p for p in all_params if p not in exclude_cols]
    
    # Collect data for PCA
    data_points = []
    labels = []
    
    for family_name, df in family_dfs.items():
        for _, row in df.iterrows():
            # Extract parameter values
            point = []
            for param in params:
                if param in df.columns:
                    point.append(row.get(param, np.nan))
                else:
                    point.append(np.nan)
            
            # Skip if too many missing values
            if sum(np.isnan(point)) > len(point) // 2:
                continue
                
            data_points.append(point)
            labels.append(family_name)
    
    # Skip PCA if not enough data points
    if len(data_points) < 3:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Not enough data for PCA visualization", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Convert to numpy array and handle missing values
    data = np.array(data_points)
    
    # Find columns with too many NaN values
    nan_ratio = np.isnan(data).mean(axis=0)
    valid_cols = nan_ratio < 0.3  # Keep columns with less than 30% NaN values
    
    if not any(valid_cols):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Too many missing values for PCA visualization", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Filter columns and impute missing values with column mean
    data = data[:, valid_cols]
    param_names = [p for i, p in enumerate(params) if valid_cols[i]]
    
    # Impute missing values with column means
    col_means = np.nanmean(data, axis=0)
    for i in range(data.shape[1]):
        mask = np.isnan(data[:, i])
        data[mask, i] = col_means[i]
    
    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points for each family
    unique_families = sorted(set(labels))
    cmap = plt.cm.tab20
    
    for i, family_name in enumerate(unique_families):
        family_mask = [l == family_name for l in labels]
        family_points = data_pca[family_mask]
        
        ax.scatter(
            family_points[:, 0], family_points[:, 1],
            alpha=0.7,
            s=50,
            label=family_name,
            color=cmap(i % cmap.N)
        )
        
        # Plot confidence ellipses for each family
        if sum(family_mask) > 2:
            try:
                confidence_ellipse(
                    family_points[:, 0], family_points[:, 1],
                    ax, n_std=2.0,
                    edgecolor=cmap(i % cmap.N),
                    linestyle='--',
                    alpha=0.3,
                    facecolor=cmap(i % cmap.N)
                )
            except:
                pass  # Skip if ellipse cannot be computed
    
    # Set labels and title
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA Visualization of Graph Families in Parameter Space")
    
    # Add legend
    if len(unique_families) < 15:
        ax.legend(loc='best')
    else:
        ax.legend(loc='best', fontsize=8)
    
    # Add loading vectors
    if len(param_names) < 10:  # Only show loadings if not too many parameters
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter space coverage of MMSB graph families")
    parser.add_argument("--n_families", type=int, default=30, help="Number of graph families to generate")
    parser.add_argument("--n_graphs_per_family", type=int, default=20, help="Number of graphs per family")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = "parameter_coverage_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate diverse families using the main function
    print(f"Generating {args.n_families} graph families...")
    families = generate_diverse_graph_families(
        n_families=args.n_families,
        n_graphs_per_family=args.n_graphs_per_family
    )
    
    # Analyze graph families
    print("Analyzing parameter distributions...")
    family_dfs = analyze_families(families)
    
    # Define parameter pairs
    param_pairs = [
        ("homophily", "avg_degree"),
        ("clustering_coefficient", "avg_degree"),
        ("power_law_exponent", "density"),
        ("avg_communities_per_node", "homophily")
    ]
    
    # Compute and visualize parameter space coverage
    print("Computing coverage metrics...")
    coverage_metrics = compute_parameter_space_coverage(family_dfs, None, param_pairs)
    
    # Create summary figures
    print("Creating summary figures...")
    create_summary_figures(family_dfs, coverage_metrics, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")