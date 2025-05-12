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


def generate_diverse_graph_families(
    n_families: int = 10,
    n_graphs_per_family: int = 20,
    base_params: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Generate diverse families of graphs by varying key parameters.
    Returns a dict mapping family names to dicts with 'graphs' and 'parameters'.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    def random_method_distribution():
        vals = np.random.dirichlet(np.ones(4))
        return {
            "standard": vals[0],
            "power_law": vals[1],
            "exponential": vals[2],
            "uniform": vals[3]
        }

    def random_config_model_params():
        # Always use the full global range for each method
        return {
            "power_law": {
                "exponent_min": 0.5,
                "exponent_max": 4.0,
                "target_avg_degree_min": 1.0,
                "target_avg_degree_max": 20.0
            },
            "exponential": {
                "rate_min": 0.05,
                "rate_max": 2.0,
                "target_avg_degree_min": 1.0,
                "target_avg_degree_max": 20.0
            },
            "uniform": {
                "min_factor": 0.1,
                "max_factor": 2.0,
                "target_avg_degree_min": 1.0,
                "target_avg_degree_max": 20.0
            }
        }

    def random_standard_method_params():
        return {
            "degree_heterogeneity": np.random.uniform(0.0, 1.0),
            "edge_noise": np.random.uniform(0.0, 0.4)
        }

    family_variations = []
    print(f"\nGenerating {n_families} diverse graph families...")
    while len(family_variations) < n_families:
        K = np.random.randint(10, 50)
        feature_dim = np.random.choice([0, 16, 32, 64, 128])
        block_structure = "assortative"  # Only supported for now
        edge_density = np.random.uniform(0.01, 0.25)
        homophily = np.random.uniform(0.0, 1.0)
        randomness_factor = np.random.uniform(0.0, 1.0)
        intra_community_regime_similarity = np.random.uniform(0.0, 1.0)
        inter_community_regime_similarity = np.random.uniform(0.0, 1.0)
        regimes_per_community = np.random.randint(1, 6)
        homophily_range = np.random.uniform(0.0, 0.35)
        density_range = np.random.uniform(0.0, 0.2)
        method_distribution = random_method_distribution()
        standard_method_params = random_standard_method_params()
        config_model_params = random_config_model_params()
        max_mean_community_deviation = np.random.uniform(0.06, 0.10)
        max_max_community_deviation = np.random.uniform(0.11, 0.17)
        max_parameter_search_attempts = np.random.randint(20, 51)
        parameter_search_range = np.random.uniform(0.9, 1.0)
        min_edge_density = np.random.uniform(0.005, 0.05)
        max_retries = np.random.randint(3, 5)
        triangle_enhancement = np.random.uniform(0.0, 0.0)
        seed_family = np.random.randint(0, 100000)
        # Instance-level parameters
        min_communities = np.random.randint(2, max(3, K // 5))
        max_communities = np.random.randint(min_communities + 1, max(8, K // 2))
        min_nodes = np.random.randint(40, 100)
        max_nodes = np.random.randint(min_nodes + 1, 200)
        degree_heterogeneity = np.random.uniform(0.0, 1.0)
        edge_noise = np.random.uniform(0.0, 0.5)
        sampling_method = np.random.choice(["random", "similar", "diverse", "correlated"])
        min_component_size = np.random.randint(3, 10)
        feature_regime_balance = np.random.uniform(0.0, 1.0)
        # Only restriction: K > min_nodes/2
        if K > min_nodes / 2:
            continue
        family_variations.append({
            # Universe/family-level
            "K": K,
            "feature_dim": feature_dim,
            "block_structure": block_structure,
            "edge_density": edge_density,
            "homophily": homophily,
            "randomness_factor": randomness_factor,
            "intra_community_regime_similarity": intra_community_regime_similarity,
            "inter_community_regime_similarity": inter_community_regime_similarity,
            "regimes_per_community": regimes_per_community,
            "homophily_range": homophily_range,
            "density_range": density_range,
            "method_distribution": method_distribution,
            "standard_method_params": standard_method_params,
            "config_model_params": config_model_params,
            "max_mean_community_deviation": max_mean_community_deviation,
            "max_max_community_deviation": max_max_community_deviation,
            "max_parameter_search_attempts": max_parameter_search_attempts,
            "parameter_search_range": parameter_search_range,
            "min_edge_density": min_edge_density,
            "max_retries": max_retries,
            "triangle_enhancement": triangle_enhancement,
            "seed": seed_family,
            # Instance-level
            "min_communities": min_communities,
            "max_communities": max_communities,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "degree_heterogeneity": degree_heterogeneity,
            "edge_noise": edge_noise,
            "sampling_method": sampling_method,
            "min_component_size": min_component_size,
            "feature_regime_balance": feature_regime_balance,
            "name": f"random_variation_{len(family_variations)}"
        })
    family_variations = family_variations[:n_families]
    families = {}
    max_failed_graphs_in_first_n = 3  # User-definable threshold
    n_check = 10  # Number of first graphs to check
    for variation in tqdm(family_variations):
        family_name = variation.pop("name")
        graphs = None
        family_params = base_params.copy() if base_params else {}
        family_params.update(variation)
        try:
            generator = GraphFamilyGenerator(
                K=family_params["K"],
                feature_dim=family_params["feature_dim"],
                block_structure=family_params["block_structure"],
                edge_density=family_params["edge_density"],
                homophily=family_params["homophily"],
                randomness_factor=family_params["randomness_factor"],
                intra_community_regime_similarity=family_params["intra_community_regime_similarity"],
                inter_community_regime_similarity=family_params["inter_community_regime_similarity"],
                regimes_per_community=family_params["regimes_per_community"],
                homophily_range=family_params["homophily_range"],
                density_range=family_params["density_range"],
                method_distribution=family_params["method_distribution"],
                standard_method_params=family_params["standard_method_params"],
                config_model_params=family_params["config_model_params"],
                max_mean_community_deviation=family_params["max_mean_community_deviation"],
                max_max_community_deviation=family_params["max_max_community_deviation"],
                max_parameter_search_attempts=family_params["max_parameter_search_attempts"],
                parameter_search_range=family_params["parameter_search_range"],
                min_edge_density=family_params["min_edge_density"],
                max_retries=family_params["max_retries"],
                triangle_enhancement=family_params["triangle_enhancement"],
                seed=family_params["seed"]
            )
            print(f"Generating {n_graphs_per_family} graphs for family {family_name}...")
            generated_graphs = []
            attempts = 0
            max_attempts = n_graphs_per_family * 3
            failed_graphs = 0
            failed_graphs_first_n = 0
            skip_family = False
            for graph_idx in range(n_graphs_per_family):
                print(f"  Attempting graph {graph_idx+1}/{n_graphs_per_family} (attempt {attempts+1})...")
                try:
                    single_graphs = generator.generate(
                        n_graphs=1,
                        min_communities=family_params["min_communities"],
                        max_communities=family_params["max_communities"],
                        min_nodes=family_params["min_nodes"],
                        max_nodes=family_params["max_nodes"],
                        degree_heterogeneity=family_params["degree_heterogeneity"],
                        edge_noise=family_params["edge_noise"],
                        sampling_method=family_params["sampling_method"],
                        min_component_size=family_params["min_component_size"],
                        feature_regime_balance=family_params["feature_regime_balance"],
                        seed=family_params["seed"]
                    )
                    if single_graphs and len(single_graphs) > 0:
                        graph, n_attempts = single_graphs[0]
                        # If the very first graph takes more than 1 attempt, skip the family immediately
                        if graph_idx == 0 and n_attempts > 1:
                            print(f"    Very first graph took {n_attempts} attempts (>1) for family {family_name}. Skipping this family.")
                            generated_graphs = []
                            skip_family = True
                            break
                        generated_graphs.append(graph)
                        print(f"    Success: generated graph {len(generated_graphs)}/{n_graphs_per_family} (attempts: {n_attempts})")
                        if n_attempts > 1:
                            failed_graphs += 1
                            if graph_idx < n_check:
                                failed_graphs_first_n += 1
                                if n_attempts > max_failed_graphs_in_first_n:
                                    print(f"    Single graph took {n_attempts} attempts (>{max_failed_graphs_in_first_n}) in first {n_check} for family {family_name}. Skipping this family.")
                                    generated_graphs = []
                                    skip_family = True
                                    break
                    else:
                        print(f"    Failed to generate graph {len(generated_graphs)+1}")
                        failed_graphs += 1
                        if graph_idx < n_check:
                            failed_graphs_first_n += 1
                            print(f"    Failed to generate a graph at all in first {n_check} for family {family_name}. Skipping this family.")
                            generated_graphs = []
                            skip_family = True
                            break
                except Exception as e:
                    print(f"    Exception: {e}")
                    failed_graphs += 1
                    if graph_idx < n_check:
                        failed_graphs_first_n += 1
                attempts += 1
                # Early stopping: as soon as threshold is reached, stop immediately
                if failed_graphs_first_n >= max_failed_graphs_in_first_n:
                    print(f"    Too many failed graphs ({failed_graphs_first_n}) in first {n_check} for family {family_name}. Skipping this family.")
                    generated_graphs = []
                    skip_family = True
                    break
            if skip_family or not generated_graphs:
                continue
            graphs = generated_graphs
            if graphs is not None and len(graphs) > 0:
                families[family_name] = {
                    'graphs': graphs,
                    'parameters': family_params.copy()
                }
            else:
                print(f"\nFAILED family: {family_name}")
                print("Parameters:")
                for key, value in family_params.items():
                    print(f"  {key}: {value}")
                if graphs is None:
                    print("Error: generate_pretraining_graphs returned None")
                else:
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
    families: Dict[str, dict],
    param_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Analyze graph families to extract parameter distributions.
    Args:
        families: Dictionary mapping family names to dicts with 'graphs' and 'parameters'
        param_names: List of parameter names to analyze
    Returns:
        Dictionary mapping family names to parameter DataFrames
    """
    if param_names is None:
        param_names = [
            "homophily",
            "power_law_exponent",
            "clustering_coefficient",
            "node_count",
            "avg_degree",
            "density",
            "connected_components",
            "largest_component_size",
            "intra_community_density",
            "inter_community_density"
        ]
    family_dfs = {}
    print(f"Analyzing parameter distributions for {len(families)} graph families...")
    for family_name, family_data in tqdm(families.items()):
        df = analyze_graph_family(family_data['graphs'])
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
        
        # Add target parameters
        params["target_homophily"] = graph.target_homophily
        params["target_density"] = graph.target_density
        
        # Calculate actual vs target ratios
        if params["homophily"] is not None:
            params["homophily_ratio"] = params["homophily"] / params["target_homophily"]
        if params["density"] is not None:
            params["density_ratio"] = params["density"] / params["target_density"]
        
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
            ("power_law_exponent", "avg_communities_per_node"),
            # Add new parameter pairs for target parameters
            ("target_homophily", "target_density"),
            ("homophily_ratio", "density_ratio"),
            ("target_homophily", "homophily"),
            ("target_density", "density")
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
        
        # Plot synthetic family points with increased visibility
        for family_name, points in family_points.items():
            if points:
                x_values, y_values = zip(*points)
                ax.scatter(
                    x_values, y_values, 
                    alpha=0.4,  # Increased alpha for better visibility
                    s=30,  # Increased size for better visibility
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
                            alpha=0.3,  # Increased alpha for better visibility
                            facecolor=family_colors[family_name]
                        )
                    except:
                        pass  # Skip if ellipse cannot be computed
        
        # Plot real-world points with softer appearance
        markers = ['o', 's', '^', 'D', 'v']  # Different markers for real datasets
        for i, (dataset_name, points) in enumerate(real_world_points.items()):
            if points:
                x_values, y_values = zip(*points)
                ax.scatter(
                    x_values, y_values,
                    marker=markers[i % len(markers)],
                    s=80,  # Slightly smaller size
                    label=f"Real: {dataset_name}",
                    edgecolor='black',
                    linewidth=1.0,  # Thinner edge
                    zorder=100,  # Keep on top but less dominant
                    alpha=0.6  # More transparent
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
        
        # Only add legend for real-world datasets
        if real_world_points:
            ax.legend(loc='best', framealpha=0.8)
        
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


def save_real_world_analysis_results(
    real_world_dfs: Dict[str, pd.DataFrame],
    output_dir: str = "parameter_coverage_results"
) -> None:
    """
    Save real-world dataset analysis results to CSV files.
    
    Args:
        real_world_dfs: Dictionary mapping dataset names to parameter DataFrames
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each dataset's results
    for dataset_name, df in real_world_dfs.items():
        filename = os.path.join(output_dir, f"real_world_{dataset_name}_analysis.csv")
        df.to_csv(filename, index=False)
        
    # Save a summary file with dataset metadata
    summary = {
        "dataset_name": [],
        "num_graphs": [],
        "parameters": [],
        "date_analyzed": []
    }
    
    for dataset_name, df in real_world_dfs.items():
        summary["dataset_name"].append(dataset_name)
        summary["num_graphs"].append(len(df))
        summary["parameters"].append(",".join(df.columns))
        summary["date_analyzed"].append(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "real_world_datasets_summary.csv"), index=False)


def load_cached_real_world_analysis(
    dataset_names: List[str],
    output_dir: str = "parameter_coverage_results"
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load cached real-world dataset analysis results if available.
    
    Args:
        dataset_names: List of dataset names to load
        output_dir: Directory containing cached results
        
    Returns:
        Dictionary mapping dataset names to parameter DataFrames if all requested datasets are cached,
        None otherwise
    """
    # Check if summary file exists
    summary_file = os.path.join(output_dir, "real_world_datasets_summary.csv")
    if not os.path.exists(summary_file):
        return None
    
    # Load summary
    summary_df = pd.read_csv(summary_file)
    cached_datasets = set(summary_df["dataset_name"])
    
    # Check if all requested datasets are cached
    if not all(name in cached_datasets for name in dataset_names):
        return None
    
    # Load cached results
    real_world_dfs = {}
    for dataset_name in dataset_names:
        filename = os.path.join(output_dir, f"real_world_{dataset_name}_analysis.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            real_world_dfs[dataset_name] = df
    
    return real_world_dfs if len(real_world_dfs) == len(dataset_names) else None


def load_real_world_datasets(
    output_dir: str = "parameter_coverage_results"
) -> Dict[str, pd.DataFrame]:
    """
    Load and analyze real-world datasets from TUDataset.
    First checks for cached results, only downloads and analyzes if necessary.
    
    Returns:
        Dictionary mapping dataset names to parameter DataFrames
    """
    from torch_geometric.datasets import PPI, Twitch, TUDataset
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    # Configure which datasets to load
    # Comment out datasets you don't want to include
    node_classification_datasets = {
        'PPI': {'class': PPI, 'args': {'root': './data/PPI', 'split': 'train'}},  # Will load all splits
        'Twitch': {'class': Twitch, 'args': {'root': './data/Twitch', 'name': 'EN'}},
    }
    
    graph_classification_datasets = {
       'MUTAG': {'class': TUDataset, 'args': {'root': './data/MUTAG', 'name': 'MUTAG', 'use_node_attr': True, 'cleaned': True}},
       'ENZYMES': {'class': TUDataset, 'args': {'root': './data/ENZYMES', 'name': 'ENZYMES', 'use_node_attr': True, 'cleaned': True}},
       # 'PROTEINS': {'class': TUDataset, 'args': {'root': './data/PROTEINS', 'name': 'PROTEINS', 'use_node_attr': True, 'cleaned': True}},
       # 'NCI1': {'class': TUDataset, 'args': {'root': './data/NCI1', 'name': 'NCI1', 'use_node_attr': True, 'cleaned': True}},
       # 'DD': {'class': TUDataset, 'args': {'root': './data/DD', 'name': 'DD', 'use_node_attr': True, 'cleaned': True}},
    }
    
    # Combine all dataset names for caching check
    all_dataset_names = list(node_classification_datasets.keys()) + list(graph_classification_datasets.keys())
    
    # Try to load cached results first
    cached_results = load_cached_real_world_analysis(all_dataset_names, output_dir)
    if cached_results is not None:
        print("\nLoaded cached real-world dataset analysis results")
        return cached_results
    
    print("\nNo cached results found. Loading and analyzing datasets...")
    real_world_dfs = {}
    
    # Load node classification datasets
    for name, config in node_classification_datasets.items():
        try:
            print(f"\nLoading {name} dataset...")
            
            if name == 'PPI':
                # Special handling for PPI which has train/val/test splits
                train_dataset = PPI(root='./data/PPI', split='train')
                val_dataset = PPI(root='./data/PPI', split='val')
                test_dataset = PPI(root='./data/PPI', split='test')
                all_graphs = train_dataset + val_dataset + test_dataset
            else:
                # Load other node classification datasets
                dataset = config['class'](**config['args'])
                all_graphs = [dataset[0]]  # Most node classification datasets have one large graph
            
            print(f'{name} dataset loaded: {len(all_graphs)} graphs')
            
            # Collect parameters for each graph
            params_list = []
            for i, data in enumerate(all_graphs):
                try:
                    # Convert to NetworkX graph
                    G = to_networkx(data, to_undirected=True)
                    
                    # Calculate homophily based on node labels
                    if hasattr(data, 'y') and data.y is not None:
                        # Handle multi-label case (PPI) and single-label case
                        node_labels = data.y.argmax(dim=1) if data.y.dim() > 1 else data.y
                        
                        # Calculate homophily
                        same_type_edges = 0
                        total_edges = 0
                        
                        for u, v in G.edges():
                            total_edges += 1
                            if node_labels[u] == node_labels[v]:
                                same_type_edges += 1
                        
                        homophily = same_type_edges / total_edges if total_edges > 0 else 0
                    else:
                        homophily = None
                    
                    # Compute graph parameters
                    params = {
                        "homophily": homophily,
                        "avg_degree": sum(dict(G.degree()).values()) / len(G),
                        "clustering_coefficient": nx.average_clustering(G),
                        "power_law_exponent": None,  # Would need degree distribution analysis
                        "density": nx.density(G),
                        "triangle_density": sum(nx.triangles(G).values()) / (3 * len(G)) if len(G) > 0 else 0,
                        "node_count": len(G),
                        "avg_communities_per_node": 1.0,  # Not applicable for these datasets
                    }
                    params_list.append(params)
                except Exception as e:
                    print(f"  ERROR processing {name} graph {i}: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(params_list)
            real_world_dfs[name] = df
            
        except Exception as e:
            print(f"ERROR loading {name}: {str(e)}")
    
    # Load graph classification datasets
    for name, config in graph_classification_datasets.items():
        try:
            print(f"\nLoading {name} dataset...")
            dataset = config['class'](**config['args'])
            print(f'{name} dataset loaded: {len(dataset)} graphs')
            
            # Collect parameters for each graph
            params_list = []
            for i, data in enumerate(dataset):
                try:
                    # Convert to NetworkX graph
                    G = to_networkx(data, to_undirected=True)
                    
                    # Compute graph parameters (no homophily for graph classification datasets)
                    params = {
                        "homophily": None,  # No node labels for homophily calculation
                        "avg_degree": sum(dict(G.degree()).values()) / len(G),
                        "clustering_coefficient": nx.average_clustering(G),
                        "power_law_exponent": None,  # Would need degree distribution analysis
                        "density": nx.density(G),
                        "triangle_density": sum(nx.triangles(G).values()) / (3 * len(G)) if len(G) > 0 else 0,
                        "node_count": len(G),
                        "avg_communities_per_node": 1.0,  # Not applicable for these datasets
                    }
                    params_list.append(params)
                except Exception as e:
                    print(f"  ERROR processing {name} graph {i}: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(params_list)
            real_world_dfs[name] = df
            
        except Exception as e:
            print(f"ERROR loading {name}: {str(e)}")
    
    # Print summary statistics for each dataset
    for name, df in real_world_dfs.items():
        print(f"\n{name} parameter ranges:")
        for col in df.columns:
            if df[col].notna().any():
                print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # Save results for future use
    if real_world_dfs:
        save_real_world_analysis_results(real_world_dfs, output_dir)
    
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
    parser.add_argument("--n_families", type=int, default=80, help="Number of graph families to generate")
    parser.add_argument("--n_graphs_per_family", type=int, default=50, help="Number of graphs per family")
    parser.add_argument("--exclude_real_datasets", action="store_true", help="Exclude real-world datasets from TUDataset")
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
    
    # Load real-world datasets by default unless explicitly excluded
    real_world_dfs = None
    if not args.exclude_real_datasets:
        real_world_dfs = load_real_world_datasets()
    
    # Define parameters to analyze
    params_to_analyze = [
        "homophily",
        "clustering_coefficient",
        "avg_degree",
        "density",
        "triangle_density",
        "node_count",
    ]
    
    # Generate all possible pairs of parameters
    param_pairs = []
    for i in range(len(params_to_analyze)):
        for j in range(i + 1, len(params_to_analyze)):
            param_pairs.append((params_to_analyze[i], params_to_analyze[j]))
    
    # Compute and visualize parameter space coverage
    print("Computing coverage metrics...")
    coverage_metrics = compute_parameter_space_coverage(family_dfs, real_world_dfs, param_pairs)
    
    # Create summary figures
    print("Creating summary figures...")
    create_summary_figures(family_dfs, coverage_metrics, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")