"""
Script to analyze how individual parameters affect signal and consistency metrics.
Tests each parameter with narrow ranges/values while randomizing all others.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Import the graph generation classes
from graph_universe.graph_universe import GraphUniverse
from graph_universe.graph_family import GraphFamilyGenerator

# Set global Matplotlib style for publication-quality figures
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "axes.linewidth": 1.5,
    "xtick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.size": 6,
    "ytick.major.width": 1.2,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Apply Seaborn styling and a colorblind-friendly palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="colorblind")

def is_invalid_value(value):
    """
    Safely check if a value is invalid (NaN, None, or non-numeric for numeric operations).
    """
    if value is None:
        return True
    try:
        # Try to convert to float and check if it's NaN
        float_val = float(value)
        return np.isnan(float_val)
    except (ValueError, TypeError):
        # If we can't convert to float, it's invalid for numeric operations
        return True

# Helper to apply consistent, publication-quality axis styling
def apply_publication_style(axis):
    axis.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.2)
    axis.tick_params(axis='both', which='minor', labelsize=9, length=3, width=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.set_facecolor('white')
    try:
        axis.margins(x=0.05)
    except Exception:
        pass


def calculate_grid_dimensions(n_items, max_cols=4):
    """
    Calculate optimal grid dimensions for a given number of items.
    
    Args:
        n_items: Number of items to arrange
        max_cols: Maximum number of columns (default 4)
    
    Returns:
        tuple: (n_rows, n_cols)
    """
    if n_items <= 0:
        return 1, 1
    
    # For small numbers, use simple layouts
    if n_items <= 3:
        return 1, n_items
    elif n_items <= 6:
        return 2, 3
    elif n_items <= 9:
        return 3, 3
    else:
        # For larger numbers, try to keep it roughly square but favor more columns
        n_cols = min(max_cols, n_items)
        n_rows = (n_items + n_cols - 1) // n_cols  # Ceiling division
        return n_rows, n_cols


def create_wide_boxplot(axis, box_data, positions, widths=0.65, show_fliers=False):
    """Create a wider, cleaner boxplot with consistent styling."""
    bp = axis.boxplot(
        box_data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=show_fliers,
        boxprops=dict(linewidth=1.4),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        medianprops=dict(linewidth=2.0),
        flierprops=dict(marker='o', markersize=3, alpha=0.4)
    )
    # Tighten x-limits to make boxplots fill more of the axis
    if positions:
        axis.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    return bp


def create_categorical_boxplot_with_tests(ax, box_data, box_positions, x_labels, colors, annotation_color='lightgray'):
    """
    Create categorical boxplot with statistical testing using classical legend approach.
    
    Args:
        ax: matplotlib axis
        box_data: list of data arrays for each group
        box_positions: list of x positions for boxes
        x_labels: list of x-axis labels
        colors: dict with keys 'face', 'median', 'whiskers' for styling
        annotation_color: color for annotation boxes (default: 'lightgray' for baseline, 'lightcoral' for randomized)
    """
    if not box_data:
        return
        
    # Create boxplot
    bp = create_wide_boxplot(ax, box_data, box_positions, widths=0.7, show_fliers=False)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor(colors['face'])
        patch.set_alpha(0.7)
    
    # Style the median line
    for median in bp['medians']:
        median.set_color(colors['median'])
        median.set_linewidth(2)
    
    # Style the whiskers and caps
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color(colors['whiskers'])
            line.set_linewidth(1)
    
    # Style the fliers (outliers)
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor(colors['median'])
        flier.set_markersize(4)
        flier.set_alpha(0.6)
    
    # Set x-axis labels
    ax.set_xticks(box_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='right')
    
    # Calculate non-parametric test significance for categorical parameters
    from scipy.stats import kruskal
    from matplotlib.patches import Patch
    
    legend_elements = []
    
    try:
        if len(box_data) == 2:
            # Use Mann-Whitney U for 2 groups
            from scipy.stats import mannwhitneyu
            u_stat, p_value = mannwhitneyu(box_data[0], box_data[1], alternative='two-sided')
            test_name = 'Mann-Whitney U'
        else:
            # Use Kruskal-Wallis for multiple groups
            h_stat, p_value = kruskal(*box_data)
            test_name = 'Kruskal-Wallis'
        
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'NS'
        
        # Add text-only legend with automatic best position (avoiding center)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='none', 
                                 label=f'{test_name}: p={p_value:.3f} ({significance})')]
        
        # Create legend with 'best' location first to get the automatic position
        legend = ax.legend(handles=legend_elements, loc='best', framealpha=0.9, 
                          fontsize='small', fancybox=True, shadow=True)
        
        # Check if the legend is positioned in the center and relocate if needed
        bbox = legend.get_window_extent()
        ax_bbox = ax.get_window_extent()
        
        # Calculate relative position of legend center
        legend_center_x = (bbox.x0 + bbox.x1) / 2
        legend_center_y = (bbox.y0 + bbox.y1) / 2
        ax_center_x = (ax_bbox.x0 + ax_bbox.x1) / 2
        ax_center_y = (ax_bbox.y0 + ax_bbox.y1) / 2
        
        # If legend is too close to the center, move it to upper right
        center_threshold = 0.3  # Within 30% of center is considered "middle"
        x_distance = abs(legend_center_x - ax_center_x) / (ax_bbox.x1 - ax_bbox.x0)
        y_distance = abs(legend_center_y - ax_center_y) / (ax_bbox.y1 - ax_bbox.y0)
        
        if x_distance < center_threshold and y_distance < center_threshold:
            # Remove the current legend and create a new one at upper right
            legend.remove()
            ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, 
                     fontsize='small', fancybox=True, shadow=True)
            
    except:
        pass


def maybe_add_legend(axis):
    handles, labels = axis.get_legend_handles_labels()
    # Only add a legend if there are non-private labels
    labels = [lab for lab in labels if lab and not lab.startswith("_")]
    if labels:
        # Define preferred legend locations, explicitly avoiding upper left
        # Order matters - we'll try these in sequence
        preferred_locations = [
            'upper right',    # Usually the best choice
            'lower right',    # Good for most cases
            'lower left',     # Often clear
            'center right',   # Safe side placement
            'upper center',   # Top center (avoids corners)
            'lower center',   # Bottom center
            'center',         # Neutral middle position
            'center left'     # Left side but not top
        ]
        
        # First, try matplotlib's 'best' and see where it places the legend
        temp_legend = axis.legend(frameon=False, loc='best')
        
        # Check if matplotlib chose a location that's equivalent to 'upper left'
        # by checking the actual location code
        best_location_is_upper_left = False
        try:
            # Get the legend's location code (matplotlib internal)
            if hasattr(temp_legend, '_loc'):
                # Location codes: 1='upper right', 2='upper left', 3='lower left', 4='lower right', etc.
                if temp_legend._loc == 2:  # 'upper left'
                    best_location_is_upper_left = True
            
            # Alternative check using bounding box position
            if not best_location_is_upper_left:
                legend_bbox = temp_legend.get_window_extent(axis.figure.canvas.get_renderer())
                axes_bbox = axis.get_window_extent(axis.figure.canvas.get_renderer())
                
                # Normalize coordinates relative to axes
                rel_x = (legend_bbox.x0 - axes_bbox.x0) / (axes_bbox.x1 - axes_bbox.x0)
                rel_y = (legend_bbox.y0 - axes_bbox.y0) / (axes_bbox.y1 - axes_bbox.y0)
                
                # Consider it upper left if it's in the top-left quadrant
                if rel_x < 0.5 and rel_y > 0.5:
                    best_location_is_upper_left = True
                    
        except Exception:
            # If we can't determine the position, assume it might be upper left to be safe
            best_location_is_upper_left = True
        
        # If matplotlib's best choice is upper left, use our preferred locations instead
        if best_location_is_upper_left:
            temp_legend.remove()
            
            # Try each of our preferred locations
            legend_placed = False
            for loc in preferred_locations:
                try:
                    axis.legend(frameon=False, loc=loc)
                    legend_placed = True
                    break
                except Exception:
                    continue
            
            # If none of our preferred locations work, use 'upper right' as final fallback
            if not legend_placed:
                try:
                    axis.legend(frameon=False, loc='upper right')
                except Exception:
                    # Last resort - use matplotlib's best even if it's upper left
                    axis.legend(frameon=False, loc='best')
        # If matplotlib's best choice is not upper left, keep it


def set_fig_legend_right(fig, labels_and_handles):
    """Place a unified legend on the right of the figure for multi-panel layouts.
    labels_and_handles: sequence of (handle, label)
    """
    if not labels_and_handles:
        return
    handles, labels = zip(*labels_and_handles)
    # Unique labels while preserving order
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if not uniq:
        return
    handles_u, labels_u = zip(*uniq)
    fig.legend(handles_u, labels_u, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)

# Parameters for baseline analysis (includes use_dccc_sbm)
BASELINE_PARAMS_OF_INTEREST = ['edge_propensity_variance', 'cluster_variance', 'homophily_range', 'avg_degree_range', 'min_n_nodes', 'min_communities', 
'degree_separation_range', 'power_law_exponent_range']

# Parameters for random analysis (excludes use_dccc_sbm)
RANDOM_PARAMS_OF_INTEREST = ['edge_propensity_variance', 'cluster_variance', 'homophily_range', 'avg_degree_range', 'min_n_nodes', 'min_communities', 
'degree_separation_range', 'power_law_exponent_range']

# Keep backward compatibility
PARAMS_OF_INTEREST = BASELINE_PARAMS_OF_INTEREST

# Fixed settings
REPEATS_PER_VALUE = 1
GRAPHS_PER_FAMILY = 30
UNIVERSE_K = 15

# Correlation method settings - always use family means for simplicity and reliability

# All parameters that can be varied
ALL_VARIABLE_PARAMS = {
    # Universe parameters
    'edge_propensity_variance': {
        'type': 'continuous',
        'test_values': [0.0, 0.10, 0.20, 0.30, 0.4, .5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'random_range': (0.0, 1.0),
        'level': 'universe'
    },
    'feature_dim': {
        'type': 'discrete',
        'test_values': [10, 50, 100],
        'random_range': (10, 100),
        'level': 'universe'
    },
    'center_variance': {
        'type': 'continuous',
        'test_values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'random_range': (0.1, 1.0),
        'level': 'universe'
    },
    'cluster_variance': {
        'type': 'continuous',
        'test_values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'random_range': (0.1, 1.0),
        'level': 'universe'
    },
    
    # Family generator parameters
    'min_n_nodes': {
        'type': 'discrete',
        'test_values': [50, 100, 250, 500, 750],
        'random_range': (50, 400),
        'level': 'family',
        'paired_with': 'max_n_nodes'
    },
    'max_n_nodes': {
        'type': 'discrete',
        'test_values': [100, 200, 400, 800],
        'random_range': (100, 1000),
        'level': 'family',
        'paired_with': 'min_n_nodes',
    },
    'min_communities': {
        'type': 'discrete',
        'test_values': [2, 4, 6, 10, 15],
        'random_range': (2, 15),
        'level': 'family',
        'paired_with': 'max_communities'
    },
    'max_communities': {
        'type': 'discrete',
        'test_values': [4, 6, 8],
        'random_range': (4, 8),
        'level': 'family',
        'paired_with': 'min_communities',
        'max_value': UNIVERSE_K
    },
    'homophily_range': {
        'type': 'range',
        'test_values': [(0.0, 0.05), (0.2, 0.25), (0.4, 0.45), (0.6, 0.65), (0.8, 0.85)],
        'random_range': (0.0, 1.0),
        'level': 'family'
    },
    'avg_degree_range': {
        'type': 'range',
        'test_values': [(2.0, 2.5), (4.0, 4.5), (6.0, 6.5), (8.0, 8.5), (10.0, 10.5)],
        'random_range': (2.0, 20.0),
        'level': 'family'
    },
    'degree_heterogeneity': {
        'type': 'continuous',
        'test_values': [0.0, 0.25, 0.5, 0.75, 1.0],
        'random_range': (0.0, 1.0),
        'level': 'family'
    },
    'degree_separation_range': {
        'type': 'range',
        'test_values': [(0.0, 0.05), (0.1, 0.15), (0.2, 0.25), (0.4, 0.45), (0.6, 0.65), (0.8, 0.85), (0.9, 0.95)],
        'random_range': (0.0, 1.0),
        'level': 'family'
    },
    'use_dccc_sbm': {
        'type': 'boolean',
        'test_values': [True, False],
        'random_range': [True],
        'level': 'family'
    },
    'degree_distribution': {
        'type': 'categorical',
        'test_values': ['power_law'],
        'random_range': ['power_law'],
        'level': 'family'
    },
    'power_law_exponent_range': {
        'type': 'range',
        'test_values': [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (4.0, 4.5), (4.5, 5.0)],
        'random_range': (1.5, 4.5),
        'level': 'family'
    },
    'max_mean_community_deviation': {
        'type': 'continuous',
        'test_values': [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20],
        'random_range': (0.025, 0.20),
        'level': 'family'
    },
}

# Baseline configurations for fixed parameter analysis
BASELINE_UNIVERSE_PARAMS = {
    'edge_propensity_variance': 0.5,
    'feature_dim': 15,
    'center_variance': 0.2,
    'cluster_variance': 0.5,
    # 'community_cooccurrence_homogeneity': 1.0
}

BASELINE_FAMILY_PARAMS = {
    'min_n_nodes': 150,
    'max_n_nodes': 150,
    'min_communities': 6,
    'max_communities': 6,
    'homophily_range': (0.50, 0.50),
    'avg_degree_range': (2.5, 2.5),
    'degree_heterogeneity': 0.5,
    'degree_separation_range': (0.8, 0.8),
    'use_dccc_sbm': True,
    'degree_distribution': 'power_law',
    'power_law_exponent_range': (2.2, 2.2),
    'max_mean_community_deviation': 0.05,
}

# Metrics to calculate
SIGNAL_METRICS = ['feature_signal', 'degree_signal', 'structure_signal'] # 'triangle_signal'
CONSISTENCY_METRICS = ['feature_consistency', 'degree_consistency', 'structure_consistency'] # 'generation_fidelity', 
MAIN_PROPERTY_METRICS = ['homophily_levels', 'avg_degrees', 'tail_ratio_99']
TECHNICAL_METRICS = ['mean_edge_probability_deviation', 'graph_generation_times']
PROPERTY_METRICS = MAIN_PROPERTY_METRICS + TECHNICAL_METRICS  # Keep for backward compatibility

def generate_baseline_params(fixed_param, fixed_value):
    """
    Generate parameters with baseline values except for the fixed parameter.
    """
    params = {
        'universe': BASELINE_UNIVERSE_PARAMS.copy(),
        'family': BASELINE_FAMILY_PARAMS.copy()
    }
    
    # Set the fixed parameter
    param_config = ALL_VARIABLE_PARAMS[fixed_param]
    if param_config['level'] == 'universe':
        params['universe'][fixed_param] = fixed_value
    else:
        params['family'][fixed_param] = fixed_value
    
    # Handle paired parameters
    if 'paired_with' in param_config:
        paired_param = param_config['paired_with']
        
        if fixed_param == 'min_n_nodes':
            # Ensure max > min
            if param_config['level'] == 'universe':
                params['universe'][paired_param] = fixed_value + 100
            else:
                params['family'][paired_param] = fixed_value + 100

        elif fixed_param == 'min_communities':
            # Ensure max > min, and use UNIVERSE_K as the maximum limit
            if param_config['level'] == 'universe':
                params['universe'][paired_param] = min(fixed_value + 3, UNIVERSE_K)
            else:
                params['family'][paired_param] = min(fixed_value + 3, UNIVERSE_K)
    
    return params


def run_baseline_analysis(params_to_test=None, output_dir='parameter_analysis_results', 
                          n_baseline_repeats=3):
    """
    Run baseline-only parameter sensitivity analysis.
    Computes results with fixed baseline settings while varying one parameter at a time.
    """
    os.makedirs(output_dir, exist_ok=True)

    if params_to_test is None:
        params_to_test = list(ALL_VARIABLE_PARAMS.keys())

    baseline_results = {}

    for param_name in params_to_test:
        if param_name not in ALL_VARIABLE_PARAMS:
            print(f"Warning: {param_name} not in parameter configuration. Skipping.")
            continue

        print(f"\nAnalyzing parameter (baseline): {param_name}")
        param_config = ALL_VARIABLE_PARAMS[param_name]

        # Determine number of repeats based on parameter type
        if param_config['type'] in ['categorical', 'boolean']:
            n_baseline_repeats_adj = max(n_baseline_repeats, REPEATS_PER_CATEGORICAL_VALUE)
            print(f"  Using {n_baseline_repeats_adj} baseline repeats (categorical parameter)")
        else:
            n_baseline_repeats_adj = max(n_baseline_repeats, REPEATS_PER_CONTINUOUS_VALUE)
            print(f"  Using {n_baseline_repeats_adj} baseline repeats (continuous parameter)")

        # Initialize result structures
        baseline_param_results = {
            'test_values': param_config['test_values'],
            'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
            'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
            'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
            'metadata': []
        }

        for value_idx, test_value in enumerate(param_config['test_values']):
            print(f"  Testing value: {test_value}")

            # BASELINE ANALYSIS
            baseline_signals = {metric: [] for metric in SIGNAL_METRICS}
            baseline_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
            baseline_properties = {metric: [] for metric in PROPERTY_METRICS}
            baseline_metadata = []

            for repeat in range(n_baseline_repeats_adj):
                seed = abs(hash(param_name + "_baseline") + value_idx * 1000 + repeat) % (2**32)
                baseline_params = generate_baseline_params(param_name, test_value)

                try:
                    # Create universe with baseline params
                    universe_params = baseline_params['universe']
                    
                    universe = GraphUniverse(
                        K=UNIVERSE_K,
                        edge_propensity_variance=universe_params['edge_propensity_variance'],
                        feature_dim=universe_params['feature_dim'],
                        center_variance=universe_params['center_variance'],
                        cluster_variance=universe_params['cluster_variance'],
                        # community_cooccurrence_homogeneity=universe_params.get('community_cooccurrence_homogeneity', 1.0),
                        seed=seed
                    )

                    # Create family generator with baseline params
                    family_params = baseline_params['family']
                    generator = GraphFamilyGenerator(
                        universe=universe,
                        min_n_nodes=family_params['min_n_nodes'],
                        max_n_nodes=family_params['max_n_nodes'],
                        min_communities=family_params['min_communities'],
                        max_communities=family_params['max_communities'],
                        homophily_range=family_params['homophily_range'],
                        avg_degree_range=family_params['avg_degree_range'],
                        use_dccc_sbm=family_params['use_dccc_sbm'],
                        degree_heterogeneity=family_params['degree_heterogeneity'],
                        degree_separation_range=family_params['degree_separation_range'],
                        degree_distribution=family_params['degree_distribution'],
                        power_law_exponent_range=family_params['power_law_exponent_range'],
                        max_mean_community_deviation=family_params['max_mean_community_deviation'],
                        seed=seed
                    )

                    # Generate family
                    generator.generate_family(
                        n_graphs=GRAPHS_PER_FAMILY,
                        show_progress=False
                    )

                    # Calculate signals
                    signals = generator.analyze_graph_family_signals()
                    for metric in SIGNAL_METRICS:
                        if metric in signals:
                            metric_values = [v for v in signals[metric] if v is not None]
                            if metric_values:
                                baseline_signals[metric].append(metric_values)

                    # Calculate consistency
                    consistency = generator.analyze_graph_family_consistency()
                    for metric in CONSISTENCY_METRICS:
                        if metric in consistency:
                            if isinstance(consistency[metric], list):
                                if consistency[metric]:
                                    baseline_consistency[metric].append(consistency[metric])
                            else:
                                baseline_consistency[metric].append([consistency[metric]])

                    # Calculate properties
                    properties = generator.analyze_graph_family_properties()
                    for metric in PROPERTY_METRICS:
                        if metric in properties:
                            baseline_properties[metric].append(properties[metric])

                    mean_degree_consistency = np.mean(consistency['degree_consistency'])
                    std_degree_consistency = np.std(consistency['degree_consistency'])
                    
                    baseline_metadata.append({
                        'repeat': repeat,
                        'all_params': baseline_params,
                        'n_graphs': len(generator.graphs)
                    })

                except Exception as e:
                    print(f"      Error in baseline repeat {repeat}: {e}")
                    baseline_metadata.append({
                        'repeat': repeat,
                        'error': str(e)
                    })

            # Store baseline results
            for metric in SIGNAL_METRICS:
                if baseline_signals[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in baseline_signals[metric]:
                        all_family_values.extend(family_values)

                    baseline_param_results['signal_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    baseline_param_results['signal_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })

            for metric in CONSISTENCY_METRICS:
                if baseline_consistency[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in baseline_consistency[metric]:
                        all_family_values.extend(family_values)

                    baseline_param_results['consistency_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    baseline_param_results['consistency_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })

            # Store baseline property results
            for metric in PROPERTY_METRICS:
                if baseline_properties[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in baseline_properties[metric]:
                        all_family_values.extend(family_values)

                    baseline_param_results['property_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    baseline_param_results['property_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })

            baseline_param_results['metadata'].append(baseline_metadata)

        baseline_results[param_name] = baseline_param_results

        # Save intermediate results
        with open(os.path.join(output_dir, f'{param_name}_baseline_analysis.pkl'), 'wb') as f:
            pickle.dump(baseline_param_results, f)

    # Save all results
    with open(os.path.join(output_dir, 'all_baseline_analysis.pkl'), 'wb') as f:
        pickle.dump(baseline_results, f)

    return baseline_results


def calculate_confidence_intervals(data_points, confidence_level=0.95):
    """
    Calculate confidence intervals for the spread of data points.
    
    Args:
        data_points: List of numerical values
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (mean, ci_lower, ci_upper)
    """
    if not data_points or len(data_points) == 0:
        return np.nan, np.nan, np.nan
    
    # Remove NaN values
    valid_data = [x for x in data_points if not np.isnan(x)]
    if len(valid_data) == 0:
        return np.nan, np.nan, np.nan
    
    if len(valid_data) == 1:
        return valid_data[0], valid_data[0], valid_data[0]
    
    # Calculate mean
    mean_val = np.mean(valid_data)
    
    # Calculate percentiles to capture the spread of the data
    # For 95% CI, we want the 2.5th and 97.5th percentiles
    percentile_lower = (1 - confidence_level) / 2 * 100
    percentile_upper = (1 + confidence_level) / 2 * 100
    
    ci_lower = np.percentile(valid_data, percentile_lower)
    ci_upper = np.percentile(valid_data, percentile_upper)
    
    return mean_val, ci_lower, ci_upper


def plot_parameter_effects(results_dict, save_dir='parameter_analysis_plots'):
    """
    Create comprehensive plots showing parameter effects on all metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for param_name, param_results in results_dict.items():
        # Calculate total number of metrics with organization:
        # Row 1: 3 signal metrics
        # Row 2: 3 consistency metrics  
        # Row 3: 3 main property metrics
        # Row 4: 2 technical metrics + 1 empty
        total_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(MAIN_PROPERTY_METRICS) + len(TECHNICAL_METRICS)
        
        # Use simple 4 rows, 3 columns grid layout
        n_rows, n_cols = 4, 3
        
        # Create figure with simple grid layout
        fig_width = 5 * n_cols
        fig_height = 4.5 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
        
        # Get test values and prepare x-axis
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Use helper function to get x values and labels
        x_values, x_labels = get_plot_x_values(param_name, test_values, param_config)
        
        # Create new ordered list of metrics with proper organization:
        # Row 1: Signal metrics (3)
        # Row 2: Consistency metrics (3) 
        # Row 3: Main property metrics (3)
        # Row 4: Technical metrics (2, centered)
        ordered_metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + MAIN_PROPERTY_METRICS + TECHNICAL_METRICS
        metric_types = (['signal'] * len(SIGNAL_METRICS) + 
                       ['consistency'] * len(CONSISTENCY_METRICS) +
                       ['main_property'] * len(MAIN_PROPERTY_METRICS) + 
                       ['technical'] * len(TECHNICAL_METRICS))
        
        # Plot all metrics in the simple grid organization
        for metric_idx, (metric, metric_type) in enumerate(zip(ordered_metrics, metric_types)):
            # Calculate position in simple 3x4 grid
            row = metric_idx // 3
            col = metric_idx % 3
            
            # Skip if we're beyond our grid
            if row >= n_rows or col >= n_cols:
                continue
            
            ax = axes[row, col]
            
            # Get metric data from the appropriate dictionary
            if metric_type == 'signal':
                metric_data = param_results['signal_metrics'][metric]
            elif metric_type == 'consistency':
                metric_data = param_results['consistency_metrics'][metric]
            elif metric_type in ['main_property', 'technical']:
                metric_data = param_results['property_metrics'][metric]
            
            # Check if this is a categorical parameter
            is_categorical = param_config['type'] in ['categorical', 'boolean']
            
            if is_categorical:
                # For categorical parameters, use boxplots with statistical testing (EXACT same logic as randomized method)
                box_data = []
                box_positions = []
                x_labels = []
                
                # Get parameter values for proper grouping
                test_values = param_results['test_values']
                x_values, x_labels_temp = get_plot_x_values(param_name, test_values, param_config)
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            box_data.append(all_graph_values)
                            # Use the correct x_values position, not the iteration index
                            box_positions.append(x_values[i] if i < len(x_values) else i)
                            # Use proper labels
                            if i < len(x_labels_temp):
                                x_labels.append(x_labels_temp[i])
                            else:
                                x_labels.append(str(test_values[i]) if i < len(test_values) else str(i))
                
                # Use appropriate colors based on metric type - colorblind friendly
                if metric_type == 'signal':
                    colors = {'face': 'lightblue', 'median': 'blue', 'whiskers': 'blue'}
                elif metric_type == 'consistency':
                    colors = {'face': 'lightgreen', 'median': 'darkgreen', 'whiskers': 'darkgreen'}
                elif metric_type == 'main_property':
                    colors = {'face': 'mistyrose', 'median': 'firebrick', 'whiskers': 'firebrick'}
                else:  # technical
                    colors = {'face': 'lightgray', 'median': 'black', 'whiskers': 'black'}
                create_categorical_boxplot_with_tests(ax, box_data, box_positions, x_labels, colors, 'lightgray')
            else:
                # For continuous parameters, use the original line plot with confidence intervals
                # Calculate confidence intervals for each data point
                means = []
                stds = []
                ci_lowers = []
                ci_uppers = []
                
                for data_point in metric_data:
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            mean_val, ci_lower, ci_upper = calculate_confidence_intervals(all_graph_values)
                            std_val = np.std(all_graph_values)
                            means.append(mean_val)
                            stds.append(std_val)
                            ci_lowers.append(ci_lower)
                            ci_uppers.append(ci_upper)
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                            ci_lowers.append(np.nan)
                            ci_uppers.append(np.nan)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                        ci_lowers.append(np.nan)
                        ci_uppers.append(np.nan)
                
                # Plot confidence intervals as shaded areas
                valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                if valid_indices:
                    valid_x = [x_values[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_stds = [stds[i] for i in valid_indices]
                    
                    # Calculate standard deviation bounds for shading
                    valid_std_lower = [valid_means[i] - valid_stds[i] for i in range(len(valid_means))]
                    valid_std_upper = [valid_means[i] + valid_stds[i] for i in range(len(valid_means))]
                    
                    # Get color based on metric type - colorblind friendly
                    if metric_type == 'signal':
                        plot_color = 'blue'
                        marker_style = 'o-'
                    elif metric_type == 'consistency':
                        plot_color = 'darkgreen'
                        marker_style = 's-'
                    elif metric_type == 'main_property':
                        plot_color = 'firebrick'
                        marker_style = '^-'
                    else:  # technical
                        plot_color = 'black'
                        marker_style = 'D-'
                    
                    # Plot shaded standard deviation areas (no label)
                    ax.fill_between(valid_x, valid_std_lower, valid_std_upper, 
                                  alpha=0.3, color=plot_color)
                    
                    # Plot mean line with error bars
                    ax.errorbar(valid_x, valid_means, yerr=valid_stds, 
                               fmt=marker_style, markersize=8, linewidth=2, capsize=4, capthick=1.5,
                               color=plot_color, alpha=0.8)
                
                # Add individual points
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                            # Use lighter version of the main color for individual points
                            if metric_type == 'signal':
                                point_color = 'lightblue'
                            elif metric_type == 'consistency':
                                point_color = 'lightgreen'
                            elif metric_type == 'main_property':
                                point_color = 'lightcoral'
                            else:  # technical
                                point_color = 'lightgray'
                            ax.scatter([x_values[i]] * len(all_graph_values), 
                                     np.array(all_graph_values) + y_jitter, 
                                     alpha=0.4, s=35, color=point_color)
            
            # Calculate and display correlation with significance and direction
            # Skip for categorical parameters since they're handled in the helper function
            if not is_categorical:
                # Use all individual data points, not just means
                all_x_values = []
                all_y_values = []
                all_family_ids = []
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            # Add x value for each individual measurement
                            all_x_values.extend([x_values[i]] * len(all_graph_values))
                            all_y_values.extend(all_graph_values)
                            # Create family IDs - each test value represents a different family
                            # For baseline analysis: each parameter value is a separate family
                            all_family_ids.extend([f"family_{i}_{param_name}"] * len(all_graph_values))
                
                if len(all_x_values) > 2:
                    result_tuple = calculate_correlation_with_significance(
                        np.array(all_x_values), np.array(all_y_values), param_config['type'], 
                        family_ids=np.array(all_family_ids)
                    )
                    results = extract_correlation_results(result_tuple)
                    
                    # For continuous parameters, show correlation
                    if results['significance'] == 'ns':
                        display_text = 'NS'
                    else:
                        if abs(results['correlation_value']) < 0.01:
                            formatted_value = '<0.01'
                        else:
                            formatted_value = f'{results["correlation_value"]:.3f}'
                        
                        display_text = f'{formatted_value} ({results["significance"]})'
                    
                    # Add significance with larger, more readable text
                    ax.text(0.05, 0.95, display_text, 
                           transform=ax.transAxes, verticalalignment='top', fontsize=13, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.95))
            
            # Use display names from HEATMAP_DISPLAY_NAMES if available
            metric_title = HEATMAP_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
            ax.set_title(metric_title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(get_plot_param_name(param_name), fontsize=14)
            
            # Set appropriate y-axis label and limits based on metric type
            if metric_type == 'signal':
                ax.set_ylabel('Signal Value', fontsize=14)
                ax.set_ylim(0, 1)
            elif metric_type == 'consistency':
                ax.set_ylabel('Consistency Value', fontsize=14)
                ax.set_ylim(0, 1)
            elif metric_type == 'main_property':
                ax.set_ylabel('Property Value', fontsize=14)
                # Set specific limits for main property metrics
                if metric == 'homophily_levels':
                    ax.set_ylim(0, 1)
                # avg_degrees and tail_ratio_99 will use automatic scaling
            else:  # technical
                ax.set_ylabel('Technical Value', fontsize=14)
                # Don't fix limits for technical metrics as they have different scales
            
            # Enhanced publication styling
            ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5)
            ax.tick_params(axis='both', which='minor', labelsize=10, length=4, width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.set_facecolor('white')
            
            maybe_add_legend(ax)
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
        
        # Hide the unused subplot (last position in 3x4 grid)
        if len(ordered_metrics) < n_rows * n_cols:
            axes[3, 2].set_visible(False)
        
        # Add overall title with publication styling
        fig.suptitle(f'Parameter Effects: {get_plot_param_name(param_name)}', 
                    fontsize=20, fontweight='bold', y=1.03)
        
        # Save figure with high quality
        plt.savefig(os.path.join(save_dir, f'{param_name}_effects.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()


# Dictionary for renaming parameters and metrics in the heatmap
HEATMAP_DISPLAY_NAMES = {
    # Parameters
    'edge_propensity_variance': r"Edge Propensity Variance $\epsilon$",
    'cluster_variance': r"Cluster Variance $\sigma^2$",
    'min_n_nodes': r"Mean Node Count $\bar{n}$",
    'min_communities': r"Mean Communities Participating $\bar{k}$",
    'homophily_range': r"Mean Homophily $\bar{h}$",
    'avg_degree_range': r"Mean Average Degree $\bar{d}$",
    'degree_separation_range': r"Mean Degree Separation $\bar{\rho}$",
    'power_law_exponent_range': r"Mean Power Law Exponent $\bar{\alpha}$",
    
    # Signal Metrics
    'feature_signal': 'Feature Signal',
    'degree_signal': 'Degree Signal',
    'structure_signal': 'Structure Signal',
    
    # Consistency Metrics
    'feature_consistency': 'Feature Consistency',
    'degree_consistency': 'Degree Consistency',
    'structure_consistency': 'Structure Consistency',
    
    # Main Property Metrics
    'homophily_levels': 'Mean Homophily Levels',
    'avg_degrees': 'Mean Average Degree',
    'tail_ratio_99': 'Mean Tail Ratio 99',
    
    # Technical Metrics
    'mean_edge_probability_deviation': 'Mean Edge Probability Deviation',
    'graph_generation_times': 'Mean Graph Generation Time'
}

def create_summary_heatmap(results_dict, save_path='parameter_sensitivity_heatmap.png', display_names=HEATMAP_DISPLAY_NAMES):
    """
    Create split heatmaps showing correlation values for continuous parameters and significance levels for categorical parameters.
    Shows statistical significance levels: * (p < 0.05), ** (p < 0.01), and *** (p < 0.001).
    
    Args:
        results_dict: Dictionary containing results
        save_path: Path to save the heatmap image
        display_names: Dictionary mapping parameter and metric names to display names (e.g., LaTeX formatted)
    """
    params = list(results_dict.keys())
    print(params)
    # Reorder metrics: Property metrics first, then signal metrics, then consistency metrics
    metrics = MAIN_PROPERTY_METRICS + TECHNICAL_METRICS + SIGNAL_METRICS + CONSISTENCY_METRICS
    
    # Separate continuous and categorical parameters
    continuous_params = params
    categorical_params = []
    
    # for param_name in params:
    #     param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
    #     if param_config.get('type') in ['categorical', 'boolean']:
    #         categorical_params.append(param_name)
    #     else:
    #         continuous_params.append(param_name)
    
    # Create matrices for both parameter types
    continuous_correlation_matrix = np.zeros((len(continuous_params), len(metrics)))
    continuous_significance_matrix = np.empty((len(continuous_params), len(metrics)), dtype=object)
    
    categorical_significance_matrix = np.zeros((len(categorical_params), len(metrics)))  # Numeric for colormap
    categorical_display_matrix = np.empty((len(categorical_params), len(metrics)), dtype=object)
    
    # Process continuous parameters
    for i, param_name in enumerate(continuous_params):
        param_results = results_dict[param_name]
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Get x values using helper function
        x_values, _ = get_plot_x_values(param_name, test_values, param_config)
        
        # Calculate correlation for each metric
        for j, metric in enumerate(metrics):
            if metric in SIGNAL_METRICS:
                metric_data = param_results['signal_metrics'][metric]
            elif metric in CONSISTENCY_METRICS:
                metric_data = param_results['consistency_metrics'][metric]
            else:  # PROPERTY_METRICS
                metric_data = param_results['property_metrics'][metric]
            
            # Use all individual data points, not just means
            all_x_values = []
            all_y_values = []
            all_family_ids = []
            
            for k, data_point in enumerate(metric_data):
                if data_point['values']:
                    all_graph_values = data_point['values']
                    if all_graph_values:
                        all_x_values.extend([x_values[k]] * len(all_graph_values))
                        all_y_values.extend(all_graph_values)
                        # Create family IDs - each test value represents a different family
                        all_family_ids.extend([f"family_{k}_{param_name}"] * len(all_graph_values))
            
            if len(all_x_values) > 2 and np.std(all_x_values) > 0:
                result_tuple = calculate_correlation_with_significance(
                    np.array(all_x_values), np.array(all_y_values), param_config['type'],
                    family_ids=np.array(all_family_ids)
                )
                results = extract_correlation_results(result_tuple)
                correlation_value, direction, significance = results['correlation_value'], results['direction'], results['significance']
                
                continuous_correlation_matrix[i, j] = correlation_value
                continuous_significance_matrix[i, j] = significance
            else:
                continuous_correlation_matrix[i, j] = 0.0
                continuous_significance_matrix[i, j] = 'ns'
    
    # Process categorical parameters
    for i, param_name in enumerate(categorical_params):
        param_results = results_dict[param_name]
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Get x values using helper function
        x_values, _ = get_plot_x_values(param_name, test_values, param_config)
        
        # Calculate significance for each metric
        for j, metric in enumerate(metrics):
            if metric in SIGNAL_METRICS:
                metric_data = param_results['signal_metrics'][metric]
            elif metric in CONSISTENCY_METRICS:
                metric_data = param_results['consistency_metrics'][metric]
            else:  # PROPERTY_METRICS
                metric_data = param_results['property_metrics'][metric]
            
            # Use all individual data points, not just means
            all_x_values = []
            all_y_values = []
            all_family_ids = []
            
            for k, data_point in enumerate(metric_data):
                if data_point['values']:
                    all_graph_values = data_point['values']
                    if all_graph_values:
                        all_x_values.extend([x_values[k]] * len(all_graph_values))
                        all_y_values.extend(all_graph_values)
                        # Create family IDs - each test value represents a different family
                        all_family_ids.extend([f"family_{k}_{param_name}"] * len(all_graph_values))
            
            if len(all_x_values) > 2 and np.std(all_x_values) > 0:
                result_tuple = calculate_correlation_with_significance(
                    np.array(all_x_values), np.array(all_y_values), param_config['type'],
                    family_ids=np.array(all_family_ids)
                )
                results = extract_correlation_results(result_tuple)
                correlation_value, direction, significance = results['correlation_value'], results['direction'], results['significance']
                
                # Convert significance to numeric for colormap
                if significance == '***':
                    categorical_significance_matrix[i, j] = 3
                    if direction != 'none':
                        short_direction = '+' if direction == 'positive' else '-'
                        categorical_display_matrix[i, j] = f'{significance} {short_direction}'
                    else:
                        categorical_display_matrix[i, j] = significance
                elif significance == '**':
                    categorical_significance_matrix[i, j] = 2
                    if direction != 'none':
                        short_direction = '+' if direction == 'positive' else '-'
                        categorical_display_matrix[i, j] = f'{significance} {short_direction}'
                    else:
                        categorical_display_matrix[i, j] = significance
                elif significance == '*':
                    categorical_significance_matrix[i, j] = 1
                    if direction != 'none':
                        short_direction = '+' if direction == 'positive' else '-'
                        categorical_display_matrix[i, j] = f'{significance} {short_direction}'
                    else:
                        categorical_display_matrix[i, j] = significance
                else:
                    categorical_significance_matrix[i, j] = 0
                    categorical_display_matrix[i, j] = 'NS'
            else:
                categorical_significance_matrix[i, j] = 0
                categorical_display_matrix[i, j] = 'NS'
    
    # Define metric group boundaries for visual separation
    property_end = len(MAIN_PROPERTY_METRICS + TECHNICAL_METRICS)
    signal_end = property_end + len(SIGNAL_METRICS)
    
    # Create the split heatmap figure
    if len(continuous_params) > 0 and len(categorical_params) > 0:
        # Both types exist - create split heatmap
        height_ratio = [len(continuous_params), len(categorical_params)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, (len(continuous_params) + len(categorical_params)) * 0.6 + 5), 
                                       gridspec_kw={'height_ratios': height_ratio, 'hspace': 0.85})
        
        # Top heatmap: Continuous parameters
        ax1.grid(False)
        im1 = ax1.imshow(continuous_correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # FIX: Set tick positions AND labels for continuous parameters
        ax1.set_xticks(np.arange(len(metrics)))
        ax1.set_yticks(np.arange(len(continuous_params)))
        ax1.set_xticklabels([display_names.get(m, m.replace('_', ' ').title()) for m in metrics], rotation=25, ha='right', fontsize=14)
        ax1.set_yticklabels([display_names.get(p, get_plot_param_name(p)) for p in continuous_params], fontsize=14)
        ax1.set_title('Continuous Parameters: Pearson Correlation Coefficients', fontsize=16, pad=20)
        
        # Add vertical lines to separate metric groups
        # ax1.axvline(x=property_end-0.5, color='black', linestyle='-', linewidth=3, alpha=0.3)
        # ax1.axvline(x=signal_end-0.5, color='black', linestyle='-', linewidth=3, alpha=0.3)
        
        # Add x-axis subtitle
        ax1.set_xlabel('Validation Metrics', fontsize=18, labelpad=15, fontweight='bold')
        
        # Add y-axis title above the labels (horizontal)
        ax1.text(-0.15, 1.08, 'Graph Generation Parameters', fontsize=18, 
                 rotation=0, va='center', ha='center', fontweight='bold', transform=ax1.transAxes)
        
        # Colorbar for continuous
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=14)
        cbar1.ax.tick_params(labelsize=12)
        
        # Add text annotations for continuous
        for i in range(len(continuous_params)):
            for j in range(len(metrics)):
                correlation_val = continuous_correlation_matrix[i, j]
                significance = continuous_significance_matrix[i, j]
                
                if abs(correlation_val) < 0.4:
                    text_color = "black"
                else:
                    text_color = "white"
                
                if significance in ['*', '**', '***']:
                    if abs(correlation_val) < 0.01:
                        display_text = '<0.01'
                    else:
                        display_text = f'{correlation_val:.2f} {significance}'
                    ax1.text(j, i, display_text, ha="center", va="center", color=text_color,
                            fontweight='bold', fontsize=14)
                else:
                    # Set NS entries to light gray background
                    ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              facecolor='lightgray', alpha=0.7, zorder=0))
                    ax1.text(j, i, 'NS', ha="center", va="center", color="black", 
                            fontweight='normal', fontsize=14)
        
        # Bottom heatmap: Categorical parameters
        ax2.grid(False)
        
        # Create custom colormap for significance levels
        colors = ['white', 'lightcoral', 'red', 'darkred']  # 0=NS, 1=*, 2=**, 3=***
        from matplotlib.colors import ListedColormap
        sig_cmap = ListedColormap(colors)
        
        im2 = ax2.imshow(categorical_significance_matrix, cmap=sig_cmap, aspect='auto', vmin=0, vmax=3)
        
        # FIX: Set tick positions AND labels for categorical parameters
        ax2.set_xticks(np.arange(len(metrics)))
        ax2.set_yticks(np.arange(len(categorical_params)))
        ax2.set_xticklabels([display_names.get(m, m.replace('_', ' ').title()) for m in metrics], rotation=25, ha='right', fontsize=14)
        ax2.set_yticklabels([display_names.get(p, get_plot_param_name(p)) for p in categorical_params], fontsize=14)
        ax2.set_title('Categorical Parameters: Mann-Whitney U Test of Identical Distribution', fontsize=16, pad=20)
        
        # Add vertical lines to separate metric groups
        # ax2.axvline(x=property_end-0.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
        # ax2.axvline(x=signal_end-0.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        # Add x-axis subtitle
        ax2.set_xlabel('Validation Metrics', fontsize=18, labelpad=15, fontweight='bold')
        
        # Add y-axis title above the labels (horizontal)
        ax2.text(-0.15, 1.08, 'Graph Generation Parameters', fontsize=18, 
                 rotation=0, va='center', ha='center', fontweight='bold', transform=ax2.transAxes)
        
        # Colorbar for categorical
        cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
        cbar2.set_label('Significance Level', rotation=270, labelpad=20, fontsize=14)
        cbar2.ax.set_yticklabels(['NS', '*', '**', '***'], fontsize=12)
        
        # Add text annotations for categorical
        for i in range(len(categorical_params)):
            for j in range(len(metrics)):
                sig_val = categorical_significance_matrix[i, j]
                display_text = categorical_display_matrix[i, j]
                
                if sig_val >= 2:
                    text_color = "white"
                else:
                    text_color = "black"
                
                if display_text == 'NS':
                    # Set NS entries to light gray background
                    ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              facecolor='lightgray', alpha=0.7, zorder=0))
                    ax2.text(j, i, display_text, ha="center", va="center", color="black",
                            fontweight='normal', fontsize=14)
                else:
                    ax2.text(j, i, display_text, ha="center", va="center", color=text_color,
                            fontweight='bold', fontsize=14)
        
        # plt.suptitle('Parameter Sensitivity Analysis\n(*, **, and *** significance levels shown)', 
        #              fontsize=18, y=1.02)
        
    elif len(continuous_params) > 0:
        # Only continuous parameters
        fig, ax = plt.subplots(figsize=(18, len(continuous_params) * 0.6 + 2))
        ax.grid(False)
        im = ax.imshow(continuous_correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # FIX: Set tick positions AND labels for continuous parameters only case
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(continuous_params)))  # This is the key fix!
        ax.set_xticklabels([display_names.get(m, m.replace('_', ' ').title()) for m in metrics], rotation=25, ha='right', fontsize=16)
        ax.set_yticklabels([display_names.get(p, get_plot_param_name(p)) for p in continuous_params], fontsize=16)
        
        # Add vertical lines to separate metric groups
        # ax.axvline(x=property_end-0.5, color='black', linestyle='-', linewidth=3, alpha=0.3)
        # ax.axvline(x=signal_end-0.5, color='black', linestyle='-', linewidth=3, alpha=0.3)
        
        # Add x-axis subtitle
        ax.set_xlabel('Validation Metrics', fontsize=18, labelpad=15, fontweight='bold')
        
        # Add y-axis title above the labels
        ax.text(-0.15, 1.08, 'Graph Generation Parameters', fontsize=18, 
                rotation=0, va='center', ha='center', fontweight='bold', transform=ax.transAxes)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=17)
        cbar.ax.tick_params(labelsize=16)
        
        # Add text annotations
        for i in range(len(continuous_params)):
            for j in range(len(metrics)):
                correlation_val = continuous_correlation_matrix[i, j]
                significance = continuous_significance_matrix[i, j]
                
                if abs(correlation_val) < 0.4:
                    text_color = "black"
                else:
                    text_color = "white"
                
                if significance in ['*', '**', '***']:
                    if abs(correlation_val) < 0.01:
                        display_text = '<0.01'
                    else:
                        display_text = f'{correlation_val:.2f} {significance}'
                    ax.text(j, i, display_text, ha="center", va="center", color=text_color,
                           fontweight='bold', fontsize=16)
                else:
                    # Set NS entries to light gray background
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                            facecolor='lightgray', alpha=0.7, zorder=0))
                    ax.text(j, i, 'NS', ha="center", va="center", color="black", 
                           fontweight='normal', fontsize=16)
        
        plt.title('*, **, and *** significance levels shown, NS = Not Significant', 
                  fontsize=16, pad=15)
        
    elif len(categorical_params) > 0:
        # Only categorical parameters
        fig, ax = plt.subplots(figsize=(18, len(categorical_params) * 0.6 + 2))
        ax.grid(False)
        
        colors = ['white', 'lightcoral', 'red', 'darkred']
        from matplotlib.colors import ListedColormap
        sig_cmap = ListedColormap(colors)
        
        im = ax.imshow(categorical_significance_matrix, cmap=sig_cmap, aspect='auto', vmin=0, vmax=3)
        
        # FIX: Set tick positions AND labels for categorical parameters only case
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(categorical_params)))  # This is the key fix!
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels([get_plot_param_name(p) for p in categorical_params], fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_label('Significance Level', rotation=270, labelpad=20, fontsize=16)
        cbar.ax.set_yticklabels(['NS', '*', '**', '***'], fontsize=14)
        
        # Add text annotations
        for i in range(len(categorical_params)):
            for j in range(len(metrics)):
                sig_val = categorical_significance_matrix[i, j]
                display_text = categorical_display_matrix[i, j]
                
                if sig_val >= 2:
                    text_color = "white"
                else:
                    text_color = "black"
                
                if display_text == 'NS':
                    # Set NS entries to light gray background
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                            facecolor='lightgray', alpha=0.7, zorder=0))
                    ax.text(j, i, display_text, ha="center", va="center", color="black",
                           fontweight='normal', fontsize=16)
                else:
                    ax.text(j, i, display_text, ha="center", va="center", color=text_color,
                           fontweight='bold', fontsize=16)
        
        plt.title('Categorical Parameters: Mann-Whitney U Test\n(*, **, and *** significance levels shown)', 
                  fontsize=18, pad=25, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_variance_equality_heatmap(results_dict, save_path='categorical_variance_heatmap.png'):
    """
    Create a heatmap showing variance equality test results for categorical parameters only.
    """
    # Filter for categorical parameters only
    categorical_params = []
    for param_name in results_dict.keys():
        param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
        if param_config.get('type') in ['categorical', 'boolean']:
            categorical_params.append(param_name)
    
    if not categorical_params:
        print("No categorical parameters found for variance equality heatmap.")
        return
    
    metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS
    
    # Create variance significance matrix
    variance_matrix = np.zeros((len(categorical_params), len(metrics)))
    display_matrix = np.empty((len(categorical_params), len(metrics)), dtype=object)
    
    for i, param_name in enumerate(categorical_params):
        param_results = results_dict[param_name]
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Get x values using helper function
        x_values, _ = get_plot_x_values(param_name, test_values, param_config)
        
        # Calculate variance equality for each metric
        for j, metric in enumerate(metrics):
            if metric in SIGNAL_METRICS:
                metric_data = param_results['signal_metrics'][metric]
            elif metric in CONSISTENCY_METRICS:
                metric_data = param_results['consistency_metrics'][metric]
            else:  # PROPERTY_METRICS
                metric_data = param_results['property_metrics'][metric]
            
            # Use all individual data points
            all_x_values = []
            all_y_values = []
            
            for k, data_point in enumerate(metric_data):
                if data_point['values']:
                    all_graph_values = data_point['values']
                    if all_graph_values:
                        all_x_values.extend([x_values[k]] * len(all_graph_values))
                        all_y_values.extend(all_graph_values)
            
            if len(all_x_values) > 2 and np.std(all_x_values) > 0:
                result_tuple = calculate_correlation_with_significance(
                    np.array(all_x_values), np.array(all_y_values), param_config['type']
                )
                results = extract_correlation_results(result_tuple)
                
                # Extract variance test results
                if results['variance_test_name'] and not np.isnan(results['variance_p_value']):
                    # Convert significance to numeric for heatmap coloring
                    if results['variance_significance'] == '***':
                        variance_matrix[i, j] = 3
                        display_matrix[i, j] = '***'
                    elif results['variance_significance'] == '**':
                        variance_matrix[i, j] = 2
                        display_matrix[i, j] = '**'
                    elif results['variance_significance'] == '*':
                        variance_matrix[i, j] = 1
                        display_matrix[i, j] = '*'
                    else:
                        variance_matrix[i, j] = 0
                        display_matrix[i, j] = 'NS'
                else:
                    variance_matrix[i, j] = 0
                    display_matrix[i, j] = 'No Data'
            else:
                variance_matrix[i, j] = 0
                display_matrix[i, j] = 'No Data'
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, len(categorical_params) * 0.8 + 2), constrained_layout=True)
    
    # Remove grid lines
    ax.grid(False)
    
    # Create colormap: 0=NS (white), 1=* (light red), 2=** (medium red), 3=*** (dark red)
    colors = ['white', 'lightcoral', 'red', 'darkred']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Create the heatmap
    im = ax.imshow(variance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(categorical_params)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels([get_plot_param_name(p) for p in categorical_params], fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_label('Levene\'s Test Significance\n(H₀: Equal Variances)', rotation=270, labelpad=20, fontsize=16)
    cbar.ax.set_yticklabels(['NS', '*', '**', '***'], fontsize=14)
    
    # Add text annotations
    for i in range(len(categorical_params)):
        for j in range(len(metrics)):
            text = display_matrix[i, j]
            # Choose text color based on background
            if variance_matrix[i, j] >= 2:
                text_color = "white"
            else:
                text_color = "black"
            
            ax.text(j, i, text, ha="center", va="center", color=text_color,
                   fontweight='bold' if text in ['*', '**', '***'] else 'normal', fontsize=14)
    
    plt.title('Variance Equality Tests for Categorical Parameters\n(Levene\'s Test: H₀ = Equal Variances, H₁ = Unequal Variances)', 
              fontsize=18, pad=25)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Variance equality heatmap saved to {save_path}")


def create_random_baseline_variance_heatmap(param_values, metric_values, save_dir='random_baseline_plots', display_names=HEATMAP_DISPLAY_NAMES):
    """
    Create a variance equality heatmap for random baseline results.
    Shows Levene's test results for categorical parameters only.
    """
    # Filter for categorical parameters only
    categorical_params = []
    for param_name in RANDOM_PARAMS_OF_INTEREST:
        param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
        if param_config.get('type') in ['categorical', 'boolean']:
            categorical_params.append(param_name)
    
    if not categorical_params:
        print("No categorical parameters found for random baseline variance heatmap.")
        return
    
    metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS
    
    # Create variance significance matrix
    variance_matrix = np.zeros((len(categorical_params), len(metrics)))
    display_matrix = np.empty((len(categorical_params), len(metrics)), dtype=object)
    
    for i, param_name in enumerate(categorical_params):
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Calculate variance equality for each metric
        for j, metric in enumerate(metrics):
            if param_name in param_values and metric in metric_values:
                param_vals = param_values[param_name]
                metric_vals = metric_values[metric]
                
                # Remove NaN values - use safer checking
                max_idx = min(len(param_vals), len(metric_vals))
                valid_indices = [idx for idx in range(max_idx) 
                               if not (is_invalid_value(param_vals[idx]) or is_invalid_value(metric_vals[idx]))]
                
                if len(valid_indices) > 2:  # Need at least 3 data points
                    valid_params = [param_vals[idx] for idx in valid_indices]
                    valid_metrics = [metric_vals[idx] for idx in valid_indices]
                    
                    # Group data by parameter values for variance testing
                    unique_params = sorted(list(set(valid_params)))
                    if len(unique_params) >= 2:  # Need at least 2 groups
                        groups = []
                        for param_val in unique_params:
                            group_data = [valid_metrics[k] for k in range(len(valid_params)) if valid_params[k] == param_val]
                            if len(group_data) > 0:  # Need at least 1 data point per group
                                groups.append(group_data)
                        
                        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                            try:
                                from scipy.stats import levene
                                levene_stat, var_p_value = levene(*groups)
                                
                                # Convert significance to numeric for heatmap coloring
                                if var_p_value < 0.001:
                                    variance_matrix[i, j] = 3
                                    display_matrix[i, j] = '***'
                                elif var_p_value < 0.01:
                                    variance_matrix[i, j] = 2
                                    display_matrix[i, j] = '**'
                                elif var_p_value < 0.05:
                                    variance_matrix[i, j] = 1
                                    display_matrix[i, j] = '*'
                                else:
                                    variance_matrix[i, j] = 0
                                    display_matrix[i, j] = 'NS'
                            except:
                                variance_matrix[i, j] = 0
                                display_matrix[i, j] = 'No Data'
                        else:
                            variance_matrix[i, j] = 0
                            display_matrix[i, j] = 'No Data'
                    else:
                        variance_matrix[i, j] = 0
                        display_matrix[i, j] = 'No Data'
                else:
                    variance_matrix[i, j] = 0
                    display_matrix[i, j] = 'No Data'
            else:
                variance_matrix[i, j] = 0
                display_matrix[i, j] = 'No Data'
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, len(categorical_params) * 0.8 + 2), constrained_layout=True)
    
    # Remove grid lines
    ax.grid(False)
    
    # Create colormap: 0=NS (white), 1=* (light red), 2=** (medium red), 3=*** (dark red)
    colors = ['white', 'lightcoral', 'red', 'darkred']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Create the heatmap
    im = ax.imshow(variance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(categorical_params)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels([get_plot_param_name(p) for p in categorical_params], fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_label('Levene\'s Test Significance\n(H₀: Equal Variances)', rotation=270, labelpad=20, fontsize=16)
    cbar.ax.set_yticklabels(['NS', '*', '**', '***'], fontsize=14)
    
    # Add text annotations
    for i in range(len(categorical_params)):
        for j in range(len(metrics)):
            text = display_matrix[i, j]
            # Choose text color based on background
            if variance_matrix[i, j] >= 2:
                text_color = "white"
            else:
                text_color = "black"
            
            ax.text(j, i, text, ha="center", va="center", color=text_color,
                   fontweight='bold' if text in ['*', '**', '***'] else 'normal', fontsize=14)
    
    plt.title('Random Baseline: Variance Equality Tests for Categorical Parameters\n(Levene\'s Test: H₀ = Equal Variances, H₁ = Unequal Variances)', 
              fontsize=18, pad=25)
    
    save_path = os.path.join(save_dir, 'random_baseline_variance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Random baseline variance equality heatmap saved to {save_path}")


def extract_correlation_results(result_tuple):
    """
    Helper function to extract correlation results from variable-length return tuple.
    
    Returns:
        dict: Dictionary with correlation and variance test results
    """
    correlation_value, direction, significance, ci_lower, ci_upper = result_tuple[:5]
    
    # Extract variance test results if available (for categorical parameters)
    variance_test_name = result_tuple[5] if len(result_tuple) > 5 else None
    variance_p_value = result_tuple[6] if len(result_tuple) > 6 else None
    variance_significance = result_tuple[7] if len(result_tuple) > 7 else None
    
    return {
        'correlation_value': correlation_value,
        'direction': direction,
        'significance': significance,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'variance_test_name': variance_test_name,
        'variance_p_value': variance_p_value,
        'variance_significance': variance_significance
    }


def calculate_correlation_with_significance(x, y, param_type='continuous', family_ids=None):
    """
    Calculate correlation with direction and statistical significance.
    Always uses family means when family_ids are provided.
    
    Args:
        x: Independent variable values
        y: Dependent variable values  
        param_type: Type of parameter ('continuous', 'categorical', 'boolean', 'range')
        family_ids: Array of family identifiers for family mean calculation
    
    Returns:
        For continuous: (correlation_value, direction, significance, ci_lower, ci_upper)
        For categorical: (correlation_value, direction, significance, ci_lower, ci_upper, variance_test_name, variance_p_value, variance_significance)
    """
    # If family_ids provided, always use family means approach
    if family_ids is not None and len(family_ids) == len(x):
        return calculate_family_mean_correlation(x, y, family_ids, param_type)
    
    # Fallback to traditional methods when no family structure
    # Remove NaN values first
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Check for sufficient data and non-constant arrays after removing NaN
    if len(x) < 3:
        return 0.0, 'none', 'ns', 0.0, 0.0
    
    # Check if arrays are constant (all values the same) after removing NaN
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 'none', 'ns', 0.0, 0.0
    
    # For categorical/boolean parameters, use non-parametric tests
    if param_type in ['categorical', 'boolean']:
        return calculate_categorical_significance(x, y)
    
    # For continuous/range parameters, use Pearson correlation
    return calculate_continuous_correlation(x, y)


def calculate_continuous_correlation(x, y):
    """
    Calculate Pearson correlation for continuous variables.
    """
    try:
        # Check for sufficient data and non-constant arrays
        if len(x) < 3 or len(y) < 3:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Check if arrays are constant (all values the same)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Calculate Pearson correlation coefficient
        correlation, p_value = pearsonr(x, y)
        
        # Check if correlation is nan (shouldn't happen with our checks above, but just in case)
        if np.isnan(correlation):
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # No confidence intervals needed for plotting
        ci_lower, ci_upper = 0.0, 0.0
        
        # Determine direction
        if correlation > 0:
            direction = 'positive'
        elif correlation < 0:
            direction = 'negative'
        else:
            direction = 'none'
        
        # Determine significance based on p-value
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'
        
        return correlation, direction, significance, ci_lower, ci_upper
        
    except:
        return 0.0, 'none', 'ns', 0.0, 0.0


def calculate_family_mean_correlation(param_values, metric_values, family_ids, param_type='continuous'):
    """
    Calculate correlation using family means - much simpler and more reliable than mixed-effects.
    
    This aggregates all observations within each family to their mean, then performs
    simple correlation on the family-level data. This properly accounts for the
    hierarchical structure while being much more robust.
    
    Args:
        param_values: Array of parameter values (one per graph)
        metric_values: Array of metric values (one per graph)
        family_ids: Array of family identifiers (which family each graph belongs to)
        param_type: Type of parameter ('continuous', 'categorical', 'boolean', 'range')
    
    Returns:
        tuple: (correlation_value, direction, significance, ci_lower, ci_upper)
               For categorical: also includes variance test results
    """
    try:
        # Convert to numpy arrays
        param_values = np.array(param_values)
        metric_values = np.array(metric_values)
        family_ids = np.array(family_ids)
        
        # Remove NaN values
        mask = ~(np.isnan(param_values) | np.isnan(metric_values))
        param_values = param_values[mask]
        metric_values = metric_values[mask]
        family_ids = family_ids[mask]
        
        # Check for sufficient data
        if len(param_values) < 3:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Create DataFrame for easier aggregation
        df = pd.DataFrame({
            'parameter': param_values,
            'metric': metric_values,
            'family': family_ids
        })
        
        # Aggregate by family means
        family_means = df.groupby('family').agg({
            'parameter': 'mean',
            'metric': 'mean'
        }).reset_index()
        
        # Check we have enough families
        if len(family_means) < 3:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Extract family-level data
        family_params = family_means['parameter'].values
        family_metrics = family_means['metric'].values
        
        # Check for variability at family level
        if np.std(family_params) == 0 or np.std(family_metrics) == 0:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        print(f"Family means: {len(family_means)} families, param_std={np.std(family_params):.4f}, metric_std={np.std(family_metrics):.4f}")
        
        # For categorical parameters, use non-parametric tests on family means
        if param_type in ['categorical', 'boolean']:
            return calculate_categorical_significance(family_params, family_metrics)
        
        # For continuous parameters, use simple Pearson correlation on family means
        correlation, p_value = pearsonr(family_params, family_metrics)
        
        print(f"Family correlation: r={correlation:.4f}, p={p_value:.6f}")
        
        # No confidence intervals needed for plotting
        ci_lower, ci_upper = 0.0, 0.0
        
        # Determine direction
        if correlation > 0:
            direction = 'positive'
        elif correlation < 0:
            direction = 'negative'
        else:
            direction = 'none'
        
        # Determine significance
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'
        
        return correlation, direction, significance, ci_lower, ci_upper
        
    except Exception as e:
        print(f"Family mean correlation failed: {e}")
        return calculate_continuous_correlation(param_values, metric_values)


def calculate_categorical_significance(x, y):
    """
    Calculate statistical significance for categorical variables using non-parametric tests.
    Uses Mann-Whitney U test for 2 groups and Kruskal-Wallis test for multiple groups.
    Also performs variance equality tests using Levene's test.
    Returns a binary significance indicator (0 or 1) instead of effect size, plus variance test results.
    
    Returns:
        tuple: (correlation_value, direction, significance, ci_lower, ci_upper, variance_test_name, variance_p_value, variance_significance)
    """
    try:
        # Check for sufficient data
        if len(x) < 3 or len(y) < 3:
            return 0.0, 'none', 'ns', 0.0, 0.0, 'None', np.nan, 'ns'
        
        # Group y values by x categories
        unique_x = np.unique(x)
        if len(unique_x) < 2:
            return 0.0, 'none', 'ns', 0.0, 0.0, 'None', np.nan, 'ns'
        
        # Calculate means for each category (for direction determination)
        means_by_category = []
        groups = []
        for x_val in unique_x:
            mask = (x == x_val)
            group_values = y[mask]
            if len(group_values) > 0:
                means_by_category.append(np.mean(group_values))
                groups.append(group_values)
        
        if len(groups) < 2:
            return 0.0, 'none', 'ns', 0.0, 0.0, 'None', np.nan, 'ns'
        
        # Perform variance equality test using Levene's test
        from scipy.stats import levene
        variance_test_name = "Levene's test"
        try:
            levene_stat, variance_p_value = levene(*groups)
            if variance_p_value < 0.001:
                variance_significance = '***'
            elif variance_p_value < 0.01:
                variance_significance = '**'
            elif variance_p_value < 0.05:
                variance_significance = '*'
            else:
                variance_significance = 'ns'
        except:
            variance_p_value = np.nan
            variance_significance = 'ns'
        
        # For 2 categories, use Mann-Whitney U test (non-parametric alternative to t-test)
        if len(groups) == 2:
            from scipy.stats import mannwhitneyu
            
            # Perform Mann-Whitney U test
            # Use alternative='two-sided' for two-tailed test
            try:
                u_stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            except ValueError:
                # Handle case where groups are identical
                return 0.0, 'none', 'ns', 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
            
            # Determine direction based on medians (more robust than means for non-parametric tests)
            # Note: groups are ordered by np.unique(x) which sorts the x values
            
            unique_x_sorted = np.unique(x)  # This is the sorted order
            median_0 = np.median(groups[0])  # median for unique_x_sorted[0]
            median_1 = np.median(groups[1])  # median for unique_x_sorted[1]
            
            # Determine direction by comparing medians
            # The direction should match the visual relationship in individual plots
            
            if len(unique_x_sorted) == 2:
                x_low, x_high = unique_x_sorted[0], unique_x_sorted[1]
                median_low, median_high = median_0, median_1
                
                # Standard interpretation: positive if higher x value leads to higher y value
                if median_high > median_low:
                    direction = 'positive'
                elif median_high < median_low:
                    direction = 'negative'
                else:
                    direction = 'none'
            else:
                # Fallback for non-binary cases
                if median_1 > median_0:
                    direction = 'positive'
                elif median_1 < median_0:
                    direction = 'negative'
                else:
                    direction = 'none'
            
            # Determine significance based on p-value
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            # Return binary significance (1 if significant, 0 if not)
            # Return 1 for *, **, and *** significance
            if significance in ['*', '**', '***']:
                return 1.0, direction, significance, 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
            else:
                return 0.0, direction, significance, 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
        
        # For more than 2 categories, use Kruskal-Wallis test (non-parametric alternative to ANOVA)
        else:
            from scipy.stats import kruskal
            
            # Perform Kruskal-Wallis test
            try:
                h_stat, p_value = kruskal(*groups)
            except ValueError:
                # Handle case where all groups are identical
                return 0.0, 'none', 'ns', 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
            
            # Determine direction based on trend of medians (more robust than means)
            medians_by_category = [np.median(group) for group in groups]
            
            if len(medians_by_category) >= 2:
                # Calculate trend (positive if medians increase with category index)
                trend = np.polyfit(range(len(medians_by_category)), medians_by_category, 1)[0]
                if trend > 0.01:  # Small threshold to avoid noise
                    direction = 'positive'
                elif trend < -0.01:
                    direction = 'negative'
                else:
                    direction = 'none'
            else:
                direction = 'none'
            
            # Determine significance based on p-value
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            # Return binary significance (1 if significant, 0 if not)
            # Return 1 for *, **, and *** significance
            if significance in ['*', '**', '***']:
                return 1.0, direction, significance, 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
            else:
                return 0.0, direction, significance, 0.0, 0.0, variance_test_name, variance_p_value, variance_significance
        
    except Exception as e:
        # More informative error handling for debugging
        print(f"Error in calculate_categorical_significance: {e}")
        return 0.0, 'none', 'ns', 0.0, 0.0, 'None', np.nan, 'ns'


def format_correlation_display(correlation_value, direction, significance, ci_lower, ci_upper, add_confidence_interval=False):
    """
    Format correlation value for display in heatmap.
    
    Args:
        correlation_value: Correlation coefficient, Cohen's d, or eta-squared value
        direction: 'positive', 'negative', or 'none'
        significance: '***', '**', '*', or 'ns'
    
    Returns:
        str: Formatted string for display
    """
    if significance == 'ns':
        return 'NS'
    
    # Format the correlation value
    if abs(correlation_value) < 0.01:
        formatted_value = '<0.01'
    else:
        formatted_value = f'{correlation_value:.2f}'
    
    # Add direction indicator
    if add_confidence_interval:
        if direction == 'positive':
            return f'{formatted_value} ± {abs(ci_upper - correlation_value):.2f} ({significance})'
        elif direction == 'negative':
            return f'-{formatted_value} ± {abs(correlation_value - ci_lower):.2f} ({significance})'
        else:
            return f'{formatted_value} ± {abs(ci_upper - correlation_value):.2f} ({significance})'
    else:
        return f'{formatted_value} ({significance})'


def get_plot_param_name(param_name):
    """
    Get the display name for a parameter in plots.
    Removes underscores and capitalizes words, with special handling for specific parameters.
    
    Args:
        param_name: Original parameter name
    
    Returns:
        str: Formatted parameter name for plotting
    """
    # Special handling for specific parameters
    if param_name == 'degree_center_method':
        return 'Use Degree Community Coupling'
    elif param_name == 'use_dccc_sbm':
        return 'Use DCCC SBM'
    elif param_name == 'edge_propensity_variance':
        return 'Edge Propensity Variance'
    
    # General formatting: remove underscores and capitalize
    return param_name.replace('_', ' ').title()


def get_plot_x_values(param_name, test_values, param_config):
    """
    Get x values for plotting, with special handling for degree_center_method.
    
    Args:
        param_name: Parameter name
        test_values: Original test values
        param_config: Parameter configuration
    
    Returns:
        tuple: (x_values, x_labels) for plotting
    """
    if param_name == 'degree_center_method':
        # Map 'random' to True and 'constant' to False
        # 'random' means using degree community coupling (True)
        # 'constant' means not using degree community coupling (False)
        x_values = [1 if v == 'random' else 0 for v in test_values]
        x_labels = ['True' if v == 'random' else 'False' for v in test_values]
        return x_values, x_labels
    elif param_config['type'] in ['categorical', 'boolean']:
        # For boolean parameters, use consistent mapping: True=1, False=0
        if param_config['type'] == 'boolean':
            x_values = [1 if v else 0 for v in test_values]
            x_labels = ['True' if v else 'False' for v in test_values]
        else:
            # For categorical parameters, use position-based mapping
            x_values = list(range(len(test_values)))
            x_labels = [str(v) for v in test_values]
        return x_values, x_labels
    elif param_config['type'] == 'range':
        # Handle both tuple ranges and midpoint values
        if isinstance(test_values[0], tuple):
            # Original baseline format with (min, max) tuples
            x_values = [(v[0] + v[1]) / 2 for v in test_values]
            x_labels = [f"{v[0]:.2f}-{v[1]:.2f}" for v in test_values]
        else:
            # Randomized format with midpoint values
            x_values = test_values
            x_labels = [f"{v:.2f}" for v in test_values]
        return x_values, x_labels
    else:
        x_values = test_values
        x_labels = [str(v) for v in test_values]
        return x_values, x_labels


def generate_random_baseline_params(n_samples=100, seed=None):
    """
    Generate n_samples random parameter combinations from the allowed ranges.
    We don't fix any specific parameter.
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_samples = []
    
    for sample_idx in range(n_samples):
        # Generate a new seed for each sample to ensure independence
        sample_seed = abs(hash(f"random_baseline_{sample_idx}")) % (2**32)
        np.random.seed(sample_seed)
        
        params = {
            'universe': {},
            'family': {}
        }
        
        # Generate random universe parameters (only for parameters in RANDOM_PARAMS_OF_INTEREST)
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if param_config['level'] == 'universe' and param_name in RANDOM_PARAMS_OF_INTEREST:
                if param_config['type'] == 'continuous':
                    min_val, max_val = param_config['random_range']
                    params['universe'][param_name] = np.random.uniform(min_val, max_val)
                elif param_config['type'] == 'discrete':
                    params['universe'][param_name] = int(np.random.uniform(param_config['random_range'][0], param_config['random_range'][1]))
                elif param_config['type'] == 'categorical':
                    params['universe'][param_name] = np.random.choice(param_config['random_range'])
                elif param_config['type'] == 'range':
                    min_val, max_val = param_config['random_range']
                    range_size = np.random.uniform(0.05, 0.2)  # Random range size
                    start_val = np.random.uniform(min_val, max_val - range_size)
                    end_val = start_val + range_size
                    params['universe'][param_name] = (start_val, end_val)
            elif param_config['level'] == 'universe' and param_name not in RANDOM_PARAMS_OF_INTEREST:
                # For excluded parameters, use baseline values
                if param_name == 'use_dccc_sbm':
                    params['universe'][param_name] = BASELINE_UNIVERSE_PARAMS[param_name]
        
        # Generate random family parameters (only for parameters in RANDOM_PARAMS_OF_INTEREST)
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if param_config['level'] == 'family' and param_name in RANDOM_PARAMS_OF_INTEREST:
                if param_config['type'] == 'continuous':
                    min_val, max_val = param_config['random_range']
                    params['family'][param_name] = np.random.uniform(min_val, max_val)
                elif param_config['type'] == 'discrete':
                    params['family'][param_name] = int(np.random.uniform(param_config['random_range'][0], param_config['random_range'][1]))
                elif param_config['type'] == 'categorical':
                    params['family'][param_name] = np.random.choice(param_config['random_range'])
                elif param_config['type'] == 'range':
                    min_val, max_val = param_config['random_range']
                    range_size = np.random.uniform(0.05, 0.2)  # Random range size
                    start_val = np.random.uniform(min_val, max_val - range_size)
                    end_val = start_val + range_size
                    params['family'][param_name] = (start_val, end_val)
            elif param_config['level'] == 'family' and param_name not in RANDOM_PARAMS_OF_INTEREST:
                # For excluded parameters, use baseline values
                params['family'][param_name] = BASELINE_FAMILY_PARAMS[param_name]
        
        # Handle paired parameters
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if 'paired_with' in param_config:
                paired_param = param_config['paired_with']
                
                if param_name == 'min_n_nodes':
                    # Ensure max > min
                    if param_config['level'] == 'universe':
                        params['universe'][paired_param] = params['universe'][param_name] + np.random.uniform(50, 200)
                    else:
                        params['family'][paired_param] = params['family'][param_name] + np.random.uniform(50, 200)
                elif param_name == 'min_communities':
                    # Ensure max > min
                    if param_config['level'] == 'universe':
                        params['universe'][paired_param] = min(params['universe'][param_name] + np.random.randint(1, 4), UNIVERSE_K)
                    else:
                        params['family'][paired_param] = min(params['family'][param_name] + np.random.randint(1, 4), UNIVERSE_K)
        
        # Fill in any missing parameters with baseline values
        for param_name, default_val in BASELINE_UNIVERSE_PARAMS.items():
            if param_name not in params['universe']:
                params['universe'][param_name] = default_val
        
        for param_name, default_val in BASELINE_FAMILY_PARAMS.items():
            if param_name not in params['family']:
                params['family'][param_name] = default_val
        
        all_samples.append(params)
    
    return all_samples


def run_random_baseline_analysis(n_samples=100, n_repeats_per_sample=3, output_dir='parameter_analysis_results'):
    """
    Run random baseline analysis by generating n_samples random parameter combinations
    and running each n_repeats_per_sample times.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRunning random baseline analysis with {n_samples} samples, {n_repeats_per_sample} repeats each")
    
    # Generate random parameter samples
    random_param_samples = generate_random_baseline_params(n_samples=n_samples)
    
    # Initialize result structure
    random_results = {
        'random_samples': random_param_samples,
        'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
        'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
        'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
        'metadata': []
    }
    
    # Run analysis for each sample
    successful_samples = []
    successful_sample_indices = []
    
    for sample_idx, sample_params in enumerate(tqdm(random_param_samples, desc="Processing random samples")):
        print(f"\nProcessing sample {sample_idx + 1}/{n_samples}")
        
        sample_signals = {metric: [] for metric in SIGNAL_METRICS}
        sample_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
        sample_properties = {metric: [] for metric in PROPERTY_METRICS}
        sample_metadata = []
        sample_successful = False
        
        for repeat in range(n_repeats_per_sample):
            seed = abs(hash(f"random_baseline_{sample_idx}_{repeat}")) % (2**32)
            
            try:
                # Create universe with random params
                universe_params = sample_params['universe']
                universe = GraphUniverse(
                    K=UNIVERSE_K,
                    edge_propensity_variance=universe_params['edge_propensity_variance'],
                    feature_dim=universe_params['feature_dim'],
                    center_variance=universe_params['center_variance'],
                    cluster_variance=universe_params['cluster_variance'],
                    seed=seed
                )
                
                # Create family generator with random params
                family_params = sample_params['family']
                generator = GraphFamilyGenerator(
                    universe=universe,
                    min_n_nodes=family_params['min_n_nodes'],
                    max_n_nodes=family_params['max_n_nodes'],
                    min_communities=family_params['min_communities'],
                    max_communities=family_params['max_communities'],
                    homophily_range=family_params['homophily_range'],
                    avg_degree_range=family_params['avg_degree_range'],
                    use_dccc_sbm=True,
                    degree_heterogeneity=family_params['degree_heterogeneity'],
                    degree_separation_range=family_params['degree_separation_range'],
                    degree_distribution=family_params['degree_distribution'],
                    power_law_exponent_range=family_params['power_law_exponent_range'],
                    max_mean_community_deviation=family_params['max_mean_community_deviation'],
                    seed=seed
                )
                
                # Generate family
                generator.generate_family(
                    n_graphs=GRAPHS_PER_FAMILY,
                    show_progress=False
                )
                
                # Calculate signals
                signals = generator.analyze_graph_family_signals()
                for metric in SIGNAL_METRICS:
                    if metric in signals:
                        metric_values = [v for v in signals[metric] if v is not None]
                        if metric_values:
                            sample_signals[metric].append(metric_values)
                
                # Calculate consistency
                consistency = generator.analyze_graph_family_consistency()
                for metric in CONSISTENCY_METRICS:
                    if metric in consistency:
                        if isinstance(consistency[metric], list):
                            if consistency[metric]:
                                sample_consistency[metric].append(consistency[metric])
                        else:
                            sample_consistency[metric].append([consistency[metric]])
                
                # Calculate properties
                properties = generator.analyze_graph_family_properties()
                for metric in PROPERTY_METRICS:
                    if metric in properties:
                        if isinstance(properties[metric], list):
                            if properties[metric]:
                                sample_properties[metric].append(properties[metric])
                        else:
                            sample_properties[metric].append([properties[metric]])
                
                # Store metadata
                sample_metadata.append({
                    'sample_idx': sample_idx,
                    'repeat': repeat,
                    'seed': seed,
                    'params': sample_params.copy()
                })
                
                sample_successful = True
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}, repeat {repeat}: {e}")
                continue
        
        # Only store results if at least one repeat was successful
        if sample_successful:
            successful_samples.append(sample_params)
            successful_sample_indices.append(sample_idx)
            
            # Store results for this sample
            for metric in SIGNAL_METRICS:
                random_results['signal_metrics'][metric].append(sample_signals[metric])
            for metric in CONSISTENCY_METRICS:
                random_results['consistency_metrics'][metric].append(sample_consistency[metric])
            for metric in PROPERTY_METRICS:
                random_results['property_metrics'][metric].append(sample_properties[metric])
            random_results['metadata'].extend(sample_metadata)
        else:
            print(f"Sample {sample_idx} failed completely - skipping")
    
    # Update results structure with successful samples info
    random_results['random_samples'] = successful_samples
    random_results['successful_sample_indices'] = successful_sample_indices
    random_results['n_successful_samples'] = len(successful_samples)
    random_results['n_total_samples'] = n_samples
    
    # Save results
    output_file = os.path.join(output_dir, 'random_baseline_analysis.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(random_results, f)
    
    print(f"Random baseline analysis completed. Successfully processed {len(successful_samples)}/{n_samples} samples.")
    print(f"Results saved to {output_file}")
    return random_results


def plot_random_baseline_results(random_results, save_dir='random_baseline_plots', display_names=HEATMAP_DISPLAY_NAMES):
    """
    Create plots for random baseline results showing each parameter of interest vs all metrics.
    For categorical parameters: use boxplots with statistical testing
    For continuous parameters: use scatter plots with dark gray dotted correlation lines and error bars
    Similar to baseline analysis but using random samples instead of systematic variation.
    
    Note: For range parameters, we use the midpoint value for analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating random baseline plots in {save_dir}")
    
    # Debug: Print the structure of the results
    print(f"Results structure:")
    print(f"  - random_samples: {len(random_results['random_samples'])} samples")
    if 'n_successful_samples' in random_results:
        print(f"  - successful samples: {random_results['n_successful_samples']}/{random_results['n_total_samples']}")
    print(f"  - signal_metrics: {list(random_results['signal_metrics'].keys())}")
    print(f"  - consistency_metrics: {list(random_results['consistency_metrics'].keys())}")
    print(f"  - property_metrics: {list(random_results['property_metrics'].keys())}")
    
    # Verify data consistency
    n_samples = len(random_results['random_samples'])
    for metric in SIGNAL_METRICS:
        if metric in random_results['signal_metrics']:
            n_metric_samples = len(random_results['signal_metrics'][metric])
            if n_metric_samples != n_samples:
                print(f"WARNING: Mismatch in {metric}: {n_metric_samples} metric samples vs {n_samples} parameter samples")
    
    print(f"Processing {n_samples} successful samples for plotting")
    
    # Extract parameter values from the random samples (only for RANDOM_PARAMS_OF_INTEREST)
    param_values = {}
    for param_name in RANDOM_PARAMS_OF_INTEREST:
        param_values[param_name] = []
    
    # Extract metric values and their standard deviations
    metric_values = {}
    metric_stds = {}
    for metric in SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS:
        metric_values[metric] = []
        metric_stds[metric] = []
    
    # Process the results to extract parameter and metric values
    # Iterate over all samples directly instead of relying on metric data
    for sample_idx in range(len(random_results['random_samples'])):
        # Get the parameters for this sample
        sample_params = random_results['random_samples'][sample_idx]
        
        # Extract parameter values (only for RANDOM_PARAMS_OF_INTEREST)
        for param_name in RANDOM_PARAMS_OF_INTEREST:
            param_config = ALL_VARIABLE_PARAMS[param_name]
            if param_config['level'] == 'universe':
                value = sample_params['universe'].get(param_name)
            else:
                value = sample_params['family'].get(param_name)
            
            # For range parameters, use the midpoint
            if isinstance(value, tuple) and len(value) == 2:
                value = (value[0] + value[1]) / 2
            
            # Special handling for degree_center_method to convert to numeric values for plotting
            if param_name == 'degree_center_method':
                # Convert 'random' to 1 (True - using degree community coupling) and 'constant' to 0 (False)
                value = True if value == 'random' else False
            
            param_values[param_name].append(value)
        
        # Extract metric values and standard deviations (across repeats for this sample)
        for metric in SIGNAL_METRICS:
            if (metric in random_results['signal_metrics'] and 
                sample_idx < len(random_results['signal_metrics'][metric]) and 
                random_results['signal_metrics'][metric][sample_idx]):
                # Flatten all values across repeats and graphs
                all_values = []
                for repeat_data in random_results['signal_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
                metric_stds[metric].append(np.nan)
        
        for metric in CONSISTENCY_METRICS:
            if (metric in random_results['consistency_metrics'] and 
                sample_idx < len(random_results['consistency_metrics'][metric]) and 
                random_results['consistency_metrics'][metric][sample_idx]):
                all_values = []
                for repeat_data in random_results['consistency_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
                metric_stds[metric].append(np.nan)
        
        for metric in PROPERTY_METRICS:
            if (metric in random_results['property_metrics'] and 
                sample_idx < len(random_results['property_metrics'][metric]) and 
                random_results['property_metrics'][metric][sample_idx]):
                all_values = []
                for repeat_data in random_results['property_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
    
    # Debug: Print what we extracted
    print(f"\nExtracted data:")
    print(f"  - Parameters: {len(next(iter(param_values.values()))) if param_values else 0} values per parameter")
    print(f"  - Metrics: {len(next(iter(metric_values.values()))) if metric_values else 0} values per metric")
    
    # Create plots for each parameter of interest (use RANDOM_PARAMS_OF_INTEREST for random baseline)
    n_params = len(RANDOM_PARAMS_OF_INTEREST)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 6.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot each parameter of interest (use RANDOM_PARAMS_OF_INTEREST for random baseline)
    for param_name in RANDOM_PARAMS_OF_INTEREST:
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            
            # Get parameter values and check if it's categorical
            param_config = ALL_VARIABLE_PARAMS[param_name]
            # Treat degree_center_method as categorical since we convert it to True/False
            is_categorical = param_config['type'] in ['categorical', 'boolean'] or param_name == 'degree_center_method'
            
            # Create subplots for all metrics with new organization:
            # Row 1: 3 signal metrics
            # Row 2: 3 consistency metrics  
            # Row 3: 3 main property metrics
            # Row 4: 2 technical metrics (centered)
            total_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(MAIN_PROPERTY_METRICS) + len(TECHNICAL_METRICS)
            metric_axes = []
            
            # Use simple 4 rows, 3 columns grid layout
            n_rows_metric, n_cols_metric = 4, 3
            
            # Create subplot for this parameter with simple grid layout
            fig_width = 4.5 * n_cols_metric
            fig_height = 4.5 * n_rows_metric
            param_fig, param_axes = plt.subplots(n_rows_metric, n_cols_metric, 
                                               figsize=(fig_width, fig_height), constrained_layout=True)
            
            metric_idx = 0
            
            # Create new ordered list of metrics with proper organization:
            # Row 1: Signal metrics (3)
            # Row 2: Consistency metrics (3) 
            # Row 3: Main property metrics (3)
            # Row 4: Technical metrics (2, centered)
            ordered_metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + MAIN_PROPERTY_METRICS + TECHNICAL_METRICS
            metric_types = (['signal'] * len(SIGNAL_METRICS) + 
                           ['consistency'] * len(CONSISTENCY_METRICS) +
                           ['main_property'] * len(MAIN_PROPERTY_METRICS) + 
                           ['technical'] * len(TECHNICAL_METRICS))
            
            # Plot all metrics in simple grid order
            for metric_idx_loop, (metric, metric_type) in enumerate(zip(ordered_metrics, metric_types)):
                # Calculate position in simple 3x4 grid
                row = metric_idx_loop // 3
                col = metric_idx_loop % 3
                
                # Skip if we're beyond our grid
                if row >= n_rows_metric or col >= n_cols_metric:
                    continue
                
                metric_ax = param_axes[row, col]
                
                # Get parameter and metric values
                if param_name in param_values and metric in metric_values:
                    param_vals = param_values[param_name]
                    metric_vals = metric_values[metric]
                    metric_std_vals = metric_stds[metric]
                    
                    # Remove NaN values
                    valid_indices = [i for i in range(len(param_vals)) 
                                   if not (is_invalid_value(param_vals[i]) or is_invalid_value(metric_vals[i]))]
                    
                    if len(valid_indices) > 1:
                        valid_params = [param_vals[i] for i in valid_indices]
                        valid_metrics = [metric_vals[i] for i in valid_indices]
                        valid_stds = [metric_std_vals[i] for i in valid_indices]
                        
                        if is_categorical:
                                # For categorical parameters, use boxplots
                                box_data = []
                                box_positions = []
                                x_labels = []
                                
                                # Group data by parameter values
                                unique_params = sorted(list(set(valid_params)))
                                for i, param_val in enumerate(unique_params):
                                    mask = [p == param_val for p in valid_params]
                                    group_metrics = [valid_metrics[j] for j in range(len(mask)) if mask[j]]
                                    if group_metrics:
                                        box_data.append(group_metrics)
                                        box_positions.append(i)
                                        x_labels.append(str(param_val))
                                
                                # Use appropriate colors based on metric type - colorblind friendly
                                if metric_type == 'signal':
                                    colors = {'face': 'lightblue', 'median': 'blue', 'whiskers': 'blue'}
                                elif metric_type == 'consistency':
                                    colors = {'face': 'lightgreen', 'median': 'darkgreen', 'whiskers': 'darkgreen'}
                                elif metric_type == 'main_property':
                                    colors = {'face': 'mistyrose', 'median': 'firebrick', 'whiskers': 'firebrick'}
                                else:  # technical
                                    colors = {'face': 'lightgray', 'median': 'black', 'whiskers': 'black'}
                                create_categorical_boxplot_with_tests(metric_ax, box_data, box_positions, x_labels, colors, 'lightgray')
                                        
                        else:
                            # For continuous parameters, use scatter plot with error bars - color by metric type
                            if metric_type == 'signal':
                                point_color = 'blue'
                            elif metric_type == 'consistency':
                                point_color = 'darkgreen'
                            elif metric_type == 'main_property':
                                point_color = 'firebrick'
                            else:  # technical
                                point_color = 'black'
                            metric_ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                                fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1, color=point_color)
            
                            # Add dark gray dotted correlation line (no confidence intervals, just simple fit)
                            if len(valid_params) > 1:
                                z = np.polyfit(valid_params, valid_metrics, 1)
                                p = np.poly1d(z)
                                metric_ax.plot(valid_params, p(valid_params), ":", alpha=0.8, linewidth=3, color='darkgray')
                            
                            # Calculate correlation with significance
                            # For randomized data, each sample is its own family
                            family_ids = [f"sample_{idx}" for idx in range(len(valid_params))]
                            corr, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                                                np.array(valid_params), np.array(valid_metrics), param_config.get('type', 'continuous'),
                                                family_ids=np.array(family_ids)
                            )
            
                            # Add correlation info
                            corr_display = format_correlation_display(corr, direction, significance, ci_lower, ci_upper)
                            metric_ax.text(0.05, 0.95, corr_display, 
                                            transform=metric_ax.transAxes, verticalalignment='top',
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
                                                
                            # Use display names from HEATMAP_DISPLAY_NAMES if available
                            metric_title = display_names.get(metric, metric.replace("_", " ").title())
                            metric_ax.set_title(metric_title, fontsize=16, weight='bold')
                            metric_ax.set_xlabel(get_plot_param_name(param_name), fontsize=14)
                            metric_ax.set_ylabel('Value')
                            apply_publication_style(metric_ax)
                    else:
                        metric_ax.text(0.5, 0.5, f'No valid data', 
                                    ha='center', va='center', transform=metric_ax.transAxes)
                        # Use display names from HEATMAP_DISPLAY_NAMES if available
                        metric_title = display_names.get(metric, metric.replace("_", " ").title())
                        metric_ax.set_title(metric_title, fontsize=16, weight='bold')
                                
                metric_idx += 1
    
            # Hide the unused subplot (last position in 3x4 grid)
            if len(ordered_metrics) < n_rows_metric * n_cols_metric:
                param_axes[3, 2].set_visible(False)
            
            # Set the main title for this parameter using display names
            param_title = display_names.get(param_name, get_plot_param_name(param_name))
            param_fig.suptitle(f'{param_title}', fontsize=20, y=0.98, fontweight='bold')
            param_fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
            
            # Save individual parameter plot
            param_output_file = os.path.join(save_dir, f'{param_name}_random_baseline_analysis.png')
            param_fig.savefig(param_output_file, dpi=300, bbox_inches='tight')
            plt.close(param_fig)
            
            # Set the main plot title for this parameter using display names if available
            param_title = HEATMAP_DISPLAY_NAMES.get(param_name, get_plot_param_name(param_name))
            ax.set_title(f'{param_title}')
            # Use display names for parameter in the text as well
            param_title = HEATMAP_DISPLAY_NAMES.get(param_name, get_plot_param_name(param_name))
            ax.text(0.5, 0.5, f'Parameter: {param_title}\nPlot saved as: {param_name}_random_baseline_analysis.png', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plot_idx += 1
    

    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add a note about the new approach
    fig.suptitle('Random Baseline Parameter Analysis Overview\n(Individual parameter plots saved separately)', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the overview plot
    output_file = os.path.join(save_dir, 'random_baseline_overview.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Random baseline overview plot saved to {output_file}")
    print(f"Individual parameter plots saved in {save_dir}")
    
    # Convert random baseline data to summary format and create summary heatmap
    random_baseline_summary_format = convert_random_baseline_to_summary_format(param_values, metric_values)
    heatmap_path = os.path.join(save_dir, 'random_baseline_correlation_heatmap.png')
    create_summary_heatmap(random_baseline_summary_format, heatmap_path, display_names=display_names)
    print(f"Random baseline summary heatmap saved to {heatmap_path}")
    
    # Create variance equality heatmap for categorical parameters
    create_random_baseline_variance_heatmap(param_values, metric_values, save_dir, display_names=display_names)
    
    plt.close()


def convert_random_baseline_to_summary_format(param_values, metric_values):
    """
    Convert random baseline data format (param_values, metric_values) to the format expected by create_summary_heatmap.
    
    Args:
        param_values: Dict of parameter_name -> list of values
        metric_values: Dict of metric_name -> list of values
    
    Returns:
        results_dict: Dict in the format expected by create_summary_heatmap
    """
    results_dict = {}
    
    # Only process parameters that are in RANDOM_PARAMS_OF_INTEREST
    for param_name in RANDOM_PARAMS_OF_INTEREST:
        if param_name not in param_values:
            continue
            
        param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
        param_vals = param_values[param_name]
        
        # We need to handle the fact that in plot_random_baseline_results, we already converted:
        # - Range parameters to midpoint values (single floats)
        # - degree_center_method to True/False booleans
        
        # For range parameters, we need to reconstruct approximate ranges from the midpoints
        # For other parameters, use the values as-is
        if param_config.get('type') == 'range':
            # For range parameters, the param_vals are already midpoints
            # Instead of creating fake ranges, use the midpoints directly as test values
            unique_midpoints = sorted(list(set(param_vals)))
            unique_param_vals = unique_midpoints
        elif param_name == 'degree_center_method':
            # Special handling for degree_center_method which was converted to True/False
            # Convert back to the original string values for compatibility with heatmap logic
            unique_booleans = sorted(list(set(param_vals)))
            unique_param_vals = ['random' if val else 'constant' for val in unique_booleans]
        else:
            # For non-range parameters, use the values directly
            unique_param_vals = sorted(list(set(param_vals)))
        
        # Create the structure expected by create_summary_heatmap
        param_results = {
            'test_values': unique_param_vals,
            'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
            'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
            'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
            'metadata': []
        }
        
        # For each unique parameter value, collect the corresponding metric values
        for test_value in unique_param_vals:
            # Find all indices where this parameter value occurs
            if param_config.get('type') == 'range':
                # For range parameters, test_value is now a midpoint (float), and param_vals contains midpoints
                # Find indices where the midpoint matches this test_value
                value_indices = [i for i, val in enumerate(param_vals) if abs(val - test_value) < 1e-6]
            elif param_name == 'degree_center_method':
                # For degree_center_method, test_value is a string ('random'/'constant') but param_vals contains booleans
                # Convert test_value back to boolean for matching
                target_bool = True if test_value == 'random' else False
                value_indices = [i for i, val in enumerate(param_vals) if val == target_bool]
            else:
                # For non-range parameters, match exactly
                value_indices = [i for i, val in enumerate(param_vals) if val == test_value]
            
            # Skip if no indices found
            if not value_indices:
                # Add empty entries for all metrics
                for metric in SIGNAL_METRICS:
                    param_results['signal_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
                for metric in CONSISTENCY_METRICS:
                    param_results['consistency_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
                for metric in PROPERTY_METRICS:
                    param_results['property_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
                continue
            
            # For each metric type, collect the values at these indices
            for metric in SIGNAL_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [metric_values[metric][i] for i in value_indices 
                                                if i < len(metric_values[metric]) and not is_invalid_value(metric_values[metric][i])]
                    param_results['signal_metrics'][metric].append({
                        'mean': np.mean(metric_vals_for_this_param) if metric_vals_for_this_param else np.nan,
                        'std': np.std(metric_vals_for_this_param) if len(metric_vals_for_this_param) > 1 else 0,
                        'values': metric_vals_for_this_param
                    })
                else:
                    param_results['signal_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            for metric in CONSISTENCY_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [metric_values[metric][i] for i in value_indices 
                                                if i < len(metric_values[metric]) and not is_invalid_value(metric_values[metric][i])]
                    param_results['consistency_metrics'][metric].append({
                        'mean': np.mean(metric_vals_for_this_param) if metric_vals_for_this_param else np.nan,
                        'std': np.std(metric_vals_for_this_param) if len(metric_vals_for_this_param) > 1 else 0,
                        'values': metric_vals_for_this_param
                    })
                else:
                    param_results['consistency_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            for metric in PROPERTY_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [metric_values[metric][i] for i in value_indices 
                                                if i < len(metric_values[metric]) and not is_invalid_value(metric_values[metric][i])]
                    param_results['property_metrics'][metric].append({
                        'mean': np.mean(metric_vals_for_this_param) if metric_vals_for_this_param else np.nan,
                        'std': np.std(metric_vals_for_this_param) if len(metric_vals_for_this_param) > 1 else 0,
                        'values': metric_vals_for_this_param
                    })
                else:
                    param_results['property_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
        
        results_dict[param_name] = param_results
    
    return results_dict


def plot_side_by_side_comparison(baseline_results, random_results, save_dir='side_by_side_plots'):
    """
    Create side-by-side comparison plots for parameters that exist in both baseline and randomized results.
    Uses the exact same plotting logic as the individual methods but arranges them side by side.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Find common parameters between baseline and randomized results
    baseline_params = set(baseline_results.keys()) if baseline_results else set()
    
    # For random results, we need to extract the parameter names from the converted format
    random_params = set()
    if random_results:
        # Check if random_results is in the raw format (from run_random_baseline_analysis)
        if 'random_samples' in random_results:
            # This is raw random results - we need to extract parameter names
            if random_results['random_samples']:
                sample_params = random_results['random_samples'][0]
                for param_name in RANDOM_PARAMS_OF_INTEREST:
                    param_config = ALL_VARIABLE_PARAMS[param_name]
                    if param_config['level'] == 'universe' and param_name in sample_params['universe']:
                        random_params.add(param_name)
                    elif param_config['level'] == 'family' and param_name in sample_params['family']:
                        random_params.add(param_name)
        else:
            # This is already converted format
            random_params = set(random_results.keys())
    
    # Find intersection
    common_params = baseline_params.intersection(random_params)
    
    if not common_params:
        print("No common parameters found between baseline and randomized results.")
        return
    
    print(f"Found {len(common_params)} common parameters: {sorted(common_params)}")
    
    # For randomized data, we need to work with raw data, not converted summary format
    if random_results and 'random_samples' in random_results:
        print("Extracting raw random results data...")
        # Extract parameter and metric values from raw random results (same as working method)
        random_param_values, random_metric_values = extract_random_baseline_data(random_results)
        
        # Also extract metric standard deviations
        random_metric_stds = {}
        for metric in SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS:
            random_metric_stds[metric] = []
        
        # Process the results to extract metric standard deviations
        for sample_idx in range(len(random_results['random_samples'])):
            # Extract metric standard deviations (across repeats for this sample)
            for metric in SIGNAL_METRICS:
                if (metric in random_results['signal_metrics'] and 
                    sample_idx < len(random_results['signal_metrics'][metric]) and 
                    random_results['signal_metrics'][metric][sample_idx]):
                    all_values = []
                    for repeat_data in random_results['signal_metrics'][metric][sample_idx]:
                        if repeat_data:
                            all_values.extend(repeat_data)
                    if all_values:
                        random_metric_stds[metric].append(np.std(all_values))
                    else:
                        random_metric_stds[metric].append(np.nan)
                else:
                    random_metric_stds[metric].append(np.nan)
            
            for metric in CONSISTENCY_METRICS:
                if (metric in random_results['consistency_metrics'] and 
                    sample_idx < len(random_results['consistency_metrics'][metric]) and 
                    random_results['consistency_metrics'][metric][sample_idx]):
                    all_values = []
                    for repeat_data in random_results['consistency_metrics'][metric][sample_idx]:
                        if repeat_data:
                            all_values.extend(repeat_data)
                    if all_values:
                        random_metric_stds[metric].append(np.std(all_values))
                    else:
                        random_metric_stds[metric].append(np.nan)
                else:
                    random_metric_stds[metric].append(np.nan)
            
            for metric in PROPERTY_METRICS:
                if (metric in random_results['property_metrics'] and 
                    sample_idx < len(random_results['property_metrics'][metric]) and 
                    random_results['property_metrics'][metric][sample_idx]):
                    all_values = []
                    for repeat_data in random_results['property_metrics'][metric][sample_idx]:
                        if repeat_data:
                            all_values.extend(repeat_data)
                    if all_values:
                        random_metric_stds[metric].append(np.std(all_values))
                    else:
                        random_metric_stds[metric].append(np.nan)
                else:
                    random_metric_stds[metric].append(np.nan)
        
        # Convert to summary format for baseline compatibility
        random_results_converted = convert_random_baseline_to_summary_format(random_param_values, random_metric_values)
    else:
        random_results_converted = random_results
        random_param_values = None
        random_metric_values = None
        random_metric_stds = None
    
    # Create side-by-side plots for each common parameter
    for param_name in sorted(common_params):
        print(f"Creating side-by-side plot for parameter: {param_name}")
        
        # Get parameter configuration
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Calculate total number of metrics with new organization:
        # Row 1: 3 signal metrics
        # Row 2: 3 consistency metrics  
        # Row 3: 3 main property metrics
        # Row 4: 2 technical metrics (centered)
        total_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(MAIN_PROPERTY_METRICS) + len(TECHNICAL_METRICS)
        n_rows_metric, n_cols_metric = 4, 3
        
        # Create figure with two subplots side by side, each containing a simple 3x4 grid
        fig_width = 5.5 * n_cols_metric * 2  # Width for two side-by-side grids
        fig_height = 4.5 * n_rows_metric + 2  # Extra height for labels
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create two main subplot areas with space for method labels at bottom
        gs_main = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1], wspace=0.15, hspace=0.05)
        
        # Create simple grid layouts for both sides with shared x-axes
        if param_name in ['avg_degree_range', 'homophily_range', 'power_law_exponent_range']:
            gs_left = gs_main[0, 0].subgridspec(n_rows_metric, n_cols_metric, hspace=0.4, wspace=0.3)
            gs_right = gs_main[0, 1].subgridspec(n_rows_metric, n_cols_metric, hspace=0.4, wspace=0.3)
        else:
            gs_left = gs_main[0, 0].subgridspec(n_rows_metric, n_cols_metric, hspace=0.3, wspace=0.3)
            gs_right = gs_main[0, 1].subgridspec(n_rows_metric, n_cols_metric, hspace=0.3, wspace=0.3)
        
        # Bottom row for method labels
        ax_label_left = fig.add_subplot(gs_main[1, 0])
        ax_label_right = fig.add_subplot(gs_main[1, 1])
        
        # Get baseline data
        baseline_param_results = baseline_results[param_name]
        baseline_test_values = baseline_param_results['test_values']
        baseline_x_values, baseline_x_labels = get_plot_x_values(param_name, baseline_test_values, param_config)
        
        # Get random data
        random_param_results = random_results_converted[param_name]
        random_test_values = random_param_results['test_values'] 
        
        random_x_values, random_x_labels = get_plot_x_values(param_name, random_test_values, param_config)
        
        # Create new ordered list of metrics with proper organization:
        # Row 1: Signal metrics (3)
        # Row 2: Consistency metrics (3) 
        # Row 3: Main property metrics (3)
        # Row 4: Technical metrics (2, centered)
        ordered_metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + MAIN_PROPERTY_METRICS + TECHNICAL_METRICS
        metric_types = (['signal'] * len(SIGNAL_METRICS) + 
                       ['consistency'] * len(CONSISTENCY_METRICS) +
                       ['main_property'] * len(MAIN_PROPERTY_METRICS) + 
                       ['technical'] * len(TECHNICAL_METRICS))
        
        # Create axes arrays to track shared x-axes
        axes_left = {}
        axes_right = {}
        
        # Plot all metrics in simple grid order
        for metric_idx, (metric, metric_type) in enumerate(zip(ordered_metrics, metric_types)):
            # Calculate position in simple 3x4 grid
            row = metric_idx // 3
            col = metric_idx % 3
            
            # Skip if we're beyond our grid
            if row >= n_rows_metric or col >= n_cols_metric:
                continue
            
            # Create axes with shared x-axis for columns (share with the axis in the same column, bottom row)
            if row == 0:
                # First row - create new axes
                ax_baseline = fig.add_subplot(gs_left[row, col])
                ax_random = fig.add_subplot(gs_right[row, col])
            else:
                # Subsequent rows - share x-axis with the axis in the same column, first row
                sharex_baseline = axes_left.get((0, col))
                sharex_random = axes_right.get((0, col))
                ax_baseline = fig.add_subplot(gs_left[row, col], sharex=sharex_baseline)
                ax_random = fig.add_subplot(gs_right[row, col], sharex=sharex_random)
            
            # Store axes for sharing reference
            axes_left[(row, col)] = ax_baseline
            axes_right[(row, col)] = ax_random
            
            # Determine if this is the bottom row for this column
            is_bottom_row = (row == n_rows_metric - 1) or (metric_idx + 3 >= len(ordered_metrics))
            
            # Plot baseline data (left side)
            plot_single_metric_data(ax_baseline, param_name, metric, metric_type, param_config,
                                   baseline_param_results, baseline_x_values, baseline_x_labels,
                                   "Baseline", 'lightgray', is_bottom_row=is_bottom_row,
                                   display_names=HEATMAP_DISPLAY_NAMES)
            
            # Plot random data (right side)  
            plot_single_metric_data(ax_random, param_name, metric, metric_type, param_config,
                                   random_param_results, random_x_values, random_x_labels,
                                   "Randomized", 'lightcoral', 
                                   raw_param_values=random_param_values, 
                                   raw_metric_values=random_metric_values,
                                   raw_metric_stds=random_metric_stds,
                                   is_bottom_row=is_bottom_row,
                                   display_names=HEATMAP_DISPLAY_NAMES)
            
            # Set titles using display names
            metric_title = display_names.get(metric, metric.replace("_", " ").title())
            ax_baseline.set_title(metric_title, fontsize=19, fontweight='bold', pad=10)
            ax_random.set_title(metric_title, fontsize=19, fontweight='bold', pad=10)
        
        # # Add overall title
        # if param_name == 'avg_degree_range':
        #     fig.suptitle(f'Parameter Effects: Average Degree Range', 
        #                 fontsize=24, fontweight='bold', )
        # else:
        #     fig.suptitle(f'Parameter Effects: {get_plot_param_name(param_name)}', 
        #                 fontsize=24, fontweight='bold')
        
        # Configure method label axes
        if param_name in ['avg_degree_range', 'homophily_range', 'power_law_exponent_range']:
            ax_label_left.set_title('Systemic Isolated Variation', fontsize=24, fontweight='bold', pad=10, y=-1.2)
            ax_label_left.axis('off')  # Hide axes
        
            ax_label_right.set_title('Randomized Sampling', fontsize=24, fontweight='bold', pad=10, y=-1.2)
            ax_label_right.axis('off')  # Hide axes
        else:
            ax_label_left.set_title('Systemic Isolated Variation', fontsize=24, fontweight='bold', pad=10, y=-1.0)
            ax_label_left.axis('off')  # Hide axes
            
            ax_label_right.set_title('Randomized Sampling', fontsize=24, fontweight='bold', pad=10, y=-1.0)
            ax_label_right.axis('off')  # Hide axes
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{param_name}_side_by_side_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    print(f"Side-by-side comparison plots saved to {save_dir}")


def extract_random_baseline_data(random_results):
    """
    Extract parameter and metric values from raw random baseline results.
    This replicates the extraction logic from plot_random_baseline_results.
    """
    # Extract parameter values from the random samples (only for RANDOM_PARAMS_OF_INTEREST)
    param_values = {}
    for param_name in RANDOM_PARAMS_OF_INTEREST:
        param_values[param_name] = []
    
    # Extract metric values and their standard deviations
    metric_values = {}
    for metric in SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS:
        metric_values[metric] = []
    
    # Process the results to extract parameter and metric values
    for sample_idx in range(len(random_results['random_samples'])):
        # Get the parameters for this sample
        sample_params = random_results['random_samples'][sample_idx]
        
        # Extract parameter values (only for RANDOM_PARAMS_OF_INTEREST)
        for param_name in RANDOM_PARAMS_OF_INTEREST:
            param_config = ALL_VARIABLE_PARAMS[param_name]
            if param_config['level'] == 'universe':
                value = sample_params['universe'].get(param_name)
            else:
                value = sample_params['family'].get(param_name)
            
            # For range parameters, use the midpoint
            if isinstance(value, tuple) and len(value) == 2:
                value = (value[0] + value[1]) / 2
            
            # Special handling for degree_center_method to convert to numeric values for plotting
            if param_name == 'degree_center_method':
                # Convert 'random' to 1 (True - using degree community coupling) and 'constant' to 0 (False)
                value = True if value == 'random' else False
            
            param_values[param_name].append(value)
        
        # Extract metric values (across repeats for this sample)
        for metric in SIGNAL_METRICS:
            if (metric in random_results['signal_metrics'] and 
                sample_idx < len(random_results['signal_metrics'][metric]) and 
                random_results['signal_metrics'][metric][sample_idx]):
                # Flatten all values across repeats and graphs
                all_values = []
                for repeat_data in random_results['signal_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                else:
                    metric_values[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
        
        for metric in CONSISTENCY_METRICS:
            if (metric in random_results['consistency_metrics'] and 
                sample_idx < len(random_results['consistency_metrics'][metric]) and 
                random_results['consistency_metrics'][metric][sample_idx]):
                all_values = []
                for repeat_data in random_results['consistency_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                else:
                    metric_values[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
        
        for metric in PROPERTY_METRICS:
            if (metric in random_results['property_metrics'] and 
                sample_idx < len(random_results['property_metrics'][metric]) and 
                random_results['property_metrics'][metric][sample_idx]):
                all_values = []
                for repeat_data in random_results['property_metrics'][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                else:
                    metric_values[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
    
    return param_values, metric_values


def plot_single_metric_data(ax, param_name, metric, metric_type, param_config, 
                           param_results, x_values, x_labels, method_name, annotation_color,
                           raw_param_values=None, raw_metric_values=None, raw_metric_stds=None,
                           is_bottom_row=False, display_names=HEATMAP_DISPLAY_NAMES):
    """
    Plot a single metric using the exact same logic as the individual plotting methods.
    This function contains the unified plotting logic for both baseline and randomized data.
    """
    # Get metric data from the appropriate dictionary
    if metric_type == 'signal':
        metric_data = param_results['signal_metrics'][metric]
    elif metric_type == 'consistency':
        metric_data = param_results['consistency_metrics'][metric]
    elif metric_type in ['main_property', 'technical']:
        metric_data = param_results['property_metrics'][metric]
    
    # Check if this is a categorical parameter
    # Special handling for degree_center_method which should be treated as categorical
    is_categorical = param_config['type'] in ['categorical', 'boolean'] or param_name == 'degree_center_method'
    
    if is_categorical:
        # For categorical parameters, use boxplots with statistical testing
        box_data = []
        box_positions = []
        box_x_labels = []
        
        for i, data_point in enumerate(metric_data):
            if data_point['values']:
                # The values are already flattened when stored
                all_graph_values = data_point['values']
                
                if all_graph_values:
                    box_data.append(all_graph_values)
                    # Use the correct x_values position, not the iteration index
                    box_positions.append(x_values[i] if i < len(x_values) else i)
                    # Use proper labels
                    if i < len(x_labels):
                        box_x_labels.append(x_labels[i])
                    else:
                        box_x_labels.append(str(param_results['test_values'][i]) if i < len(param_results['test_values']) else str(i))
        
        # Use appropriate colors based on metric type - colorblind friendly
        if metric_type == 'signal':
            colors = {'face': 'lightblue', 'median': 'blue', 'whiskers': 'blue'}
        elif metric_type == 'consistency':
            colors = {'face': 'lightgreen', 'median': 'darkgreen', 'whiskers': 'darkgreen'}
        elif metric_type == 'main_property':
            colors = {'face': 'mistyrose', 'median': 'firebrick', 'whiskers': 'firebrick'}
        else:  # technical
            colors = {'face': 'lightgray', 'median': 'black', 'whiskers': 'black'}
        # Use different annotation colors based on method
        if method_name == "Randomized":
            annotation_color = 'lightgray'  # Use gray for randomized method
        create_categorical_boxplot_with_tests(ax, box_data, box_positions, box_x_labels, colors, annotation_color)
    else:
        # Different plotting logic for baseline vs randomized methods
        if method_name == "Baseline":
            # For baseline continuous parameters, use both error bars AND confidence intervals
            # Each dot represents a family (test value), with shaded confidence intervals
            means = []
            stds = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in metric_data:
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        mean_val = np.mean(all_graph_values)
                        std_val = np.std(all_graph_values)
                        means.append(mean_val)
                        stds.append(std_val)
                        
                        # For baseline method, we'll use standard deviation for shading
                        # (confidence intervals calculated but not used for shading)
                        ci_lower, ci_upper = calculate_confidence_intervals(all_graph_values)[1:3]
                        ci_lowers.append(ci_lower)
                        ci_uppers.append(ci_upper)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                        ci_lowers.append(np.nan)
                        ci_uppers.append(np.nan)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    ci_lowers.append(np.nan)
                    ci_uppers.append(np.nan)
            
            # Plot with standard deviation shading for baseline
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_stds = [stds[i] for i in valid_indices]
                
                # Calculate standard deviation bounds for shading
                valid_std_lower = [valid_means[i] - valid_stds[i] for i in range(len(valid_means))]
                valid_std_upper = [valid_means[i] + valid_stds[i] for i in range(len(valid_means))]
                
                # Get color based on metric type - colorblind friendly
                if metric_type == 'signal':
                    plot_color = 'blue'
                elif metric_type == 'consistency':
                    plot_color = 'darkgreen'
                elif metric_type == 'main_property':
                    plot_color = 'firebrick'
                else:  # technical
                    plot_color = 'black'
                
                # Plot shaded standard deviation areas (no label)
                ax.fill_between(valid_x, valid_std_lower, valid_std_upper, 
                              alpha=0.3, color=plot_color)
                
                # Plot error bars for each family
                ax.errorbar(valid_x, valid_means, yerr=valid_stds, 
                           fmt='o-', alpha=0.8, markersize=6, capsize=3, capthick=1, 
                           color=plot_color, linewidth=2)
        else:
            # For randomized continuous parameters, use EXACT same logic as plot_random_baseline_results
            if raw_param_values is not None and raw_metric_values is not None and raw_metric_stds is not None:
                # Use raw data - EXACT same as working randomized method
                if param_name in raw_param_values and metric in raw_metric_values:
                    param_vals = raw_param_values[param_name]
                    metric_vals = raw_metric_values[metric]
                    metric_std_vals = raw_metric_stds[metric]
                    
                    # Remove NaN values - EXACT same logic as working method
                    valid_indices = [i for i in range(len(param_vals)) 
                                   if not (is_invalid_value(param_vals[i]) or is_invalid_value(metric_vals[i]))]
                    
                    if len(valid_indices) > 1:
                        valid_params = [param_vals[i] for i in valid_indices]
                        valid_metrics = [metric_vals[i] for i in valid_indices]
                        valid_stds = [metric_std_vals[i] for i in valid_indices]
                        
                        # Get color based on metric type - colorblind friendly
                        if metric_type == 'signal':
                            point_color = 'blue'
                        elif metric_type == 'consistency':
                            point_color = 'darkgreen'
                        elif metric_type == 'main_property':
                            point_color = 'firebrick'
                        else:  # technical
                            point_color = 'black'
                        
                        # Plot scatter with error bars - EXACT same as existing randomized method
                        ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                                   fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1, color=point_color)
                        
                        # Add dark gray dotted correlation line (no confidence intervals, just simple fit)
                        if len(valid_params) > 1:
                            z = np.polyfit(valid_params, valid_metrics, 1)
                            p = np.poly1d(z)
                            ax.plot(valid_params, p(valid_params), ":", alpha=0.8, linewidth=3, color='darkgray')
            else:
                # Fallback to processed data if raw data not available
                all_x_vals = []
                all_y_vals = []
                all_stds = []
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values'] and i < len(x_values):
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            mean_val = np.mean(all_graph_values)
                            std_val = np.std(all_graph_values)
                            
                            all_x_vals.append(x_values[i])
                            all_y_vals.append(mean_val)
                            all_stds.append(std_val)
                
                valid_indices = [i for i in range(len(all_x_vals)) if not (is_invalid_value(all_x_vals[i]) or is_invalid_value(all_y_vals[i]))]
                
                if valid_indices:
                    valid_params = [all_x_vals[i] for i in valid_indices]
                    valid_metrics = [all_y_vals[i] for i in valid_indices]
                    valid_stds = [all_stds[i] for i in valid_indices]
                    
                    if metric_type == 'signal':
                        point_color = 'blue'
                    elif metric_type == 'consistency':
                        point_color = 'darkgreen'
                    elif metric_type == 'main_property':
                        point_color = 'firebrick'
                    else:  # technical
                        point_color = 'black'
                    
                    ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                               fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1, color=point_color)
                    
                    if len(valid_params) > 1:
                        z = np.polyfit(valid_params, valid_metrics, 1)
                        p = np.poly1d(z)
                        ax.plot(valid_params, p(valid_params), ":", alpha=0.8, linewidth=3, color='darkgray')
    
    # Calculate and display correlation with significance and direction
    # Skip for categorical parameters since they're handled in the helper function
    if not is_categorical:
        # Use all individual data points, not just means
        all_x_values = []
        all_y_values = []
        all_family_ids = []
        
        for i, data_point in enumerate(metric_data):
            if data_point['values']:
                # The values are already flattened when stored
                all_graph_values = data_point['values']
                
                if all_graph_values:
                    # Add x value for each individual measurement
                    all_x_values.extend([x_values[i]] * len(all_graph_values))
                    all_y_values.extend(all_graph_values)
                    # For baseline data, each test value represents a different family
                    # For randomized data, we need to handle it differently in the calling function
                    if method_name == "Baseline":
                        all_family_ids.extend([f"family_{i}"] * len(all_graph_values))
                    else:  # Randomized
                        # For randomized, each sample is its own family - but we need to map properly
                        all_family_ids.extend([f"sample_{i}"] * len(all_graph_values))
        
        if len(all_x_values) > 2:
            result_tuple = calculate_correlation_with_significance(
                np.array(all_x_values), np.array(all_y_values), param_config['type'],
                family_ids=np.array(all_family_ids)
            )
            results = extract_correlation_results(result_tuple)
            
            # For continuous parameters, show correlation
            if results['significance'] == 'ns':
                display_text = 'NS'
            else:
                if abs(results['correlation_value']) < 0.01:
                    formatted_value = '<0.01'
                else:
                    formatted_value = f'{results["correlation_value"]:.3f}'
                
                display_text = f'{formatted_value} ({results["significance"]})'
            
            # Add significance with larger, more readable text
            ax.text(0.05, 0.95, display_text, 
                   transform=ax.transAxes, verticalalignment='top', fontsize=13, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.95))
    
    # Only show x-axis title on bottom row
    if is_bottom_row:
        ax.set_xlabel(get_plot_param_name(param_name), fontsize=17)
    else:
        ax.set_xlabel('')
    
    # Set appropriate y-axis label and limits based on metric type
    if metric_type == 'signal':
        ax.set_ylabel('Signal Value', fontsize=18)
        ax.set_ylim(0, 1)
    elif metric_type == 'consistency':
        ax.set_ylabel('Consistency Value', fontsize=18)
        ax.set_ylim(0, 1)
    elif metric_type == 'main_property':
        ax.set_ylabel('Property Value', fontsize=18)
        # Set specific limits for main property metrics
        if metric == 'homophily_levels':
            ax.set_ylim(0, 1)
        # avg_degrees and tail_ratio_99 will use automatic scaling
    else:  # technical
        ax.set_ylabel('Technical Value', fontsize=17)
        # Don't fix limits for technical metrics as they have different scales
    
    # Enhanced publication styling
    ax.tick_params(axis='both', which='major', labelsize=15, length=8, width=1.5)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.set_facecolor('white')
    
    maybe_add_legend(ax)
    
    
    # Special handling for x-axis based on method and parameter type
    if method_name == "Randomized" and param_config.get('type') in ['continuous', 'discrete', 'range']:
        # For randomized continuous/discrete/range parameters, use cleaner x-axis with fewer ticks
        if x_values:
            # Set reasonable x-axis limits
            x_min, x_max = min(x_values), max(x_values)
            ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
            
            # Create fewer, evenly spaced ticks
            if len(x_values) > 6:
                # Too many points, use fewer ticks
                n_ticks = 5
                tick_positions = np.linspace(x_min, x_max, n_ticks)
                tick_labels = [f"{pos:.2f}" for pos in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
            else:
                # Few enough points, use all
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
    elif x_labels:
        # Default behavior for other cases
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
        
        # For range parameters (baseline), ensure proper spacing
        if param_config.get('type') == 'range' and len(x_values) > 1:
            ax.set_xlim(min(x_values) - 0.1, max(x_values) + 0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze parameter effects on signals and consistency')
    parser.add_argument('--output-dir', default='parameter_analysis_results',
                       help='Directory to save results')
    parser.add_argument('--params', nargs='+', default=PARAMS_OF_INTEREST,
                       help='Specific parameters to test (default: all)')
    parser.add_argument('--n-repeats', type=int, default=1,
                       help='Number of repeats per value (overrides type-specific defaults)')
    parser.add_argument('--n-graphs', type=int, default=30,
                       help='Number of graphs per family')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot existing results')
    parser.add_argument('--analysis-type', choices=['randomized', 'baseline'], 
                       default='baseline', help='Type of analysis to run')
    parser.add_argument('--n-random-samples', type=int, default=100,
                       help='Number of random parameter samples to generate')
    parser.add_argument('--n-random-repeats-per-sample', type=int, default=1,
                       help='Number of repeats per random sample')
    parser.add_argument('--n-baseline-repeats', type=int, default=1,
                       help='Number of baseline repeats per value (overrides type-specific defaults)')
    parser.add_argument('--n-categorical-repeats', type=int, default=3,
                       help='Number of repeats for categorical parameters (default: 10)')
    parser.add_argument('--n-continuous-repeats', type=int, default=1,
                       help='Number of repeats for continuous parameters (default: 3)')
    parser.add_argument('--plot-side-to-side', action='store_true',
                       help='Create side-by-side comparison plots for parameters run with both baseline and randomized methods')
    
    args = parser.parse_args()
    
    # Always use family means for correlation analysis
    print("Using correlation method: Family means (simplified approach)")
    
    if not args.plot_only:
        # Update global settings
        REPEATS_PER_VALUE = args.n_repeats
        GRAPHS_PER_FAMILY = args.n_graphs
        REPEATS_PER_CATEGORICAL_VALUE = args.n_categorical_repeats
        REPEATS_PER_CONTINUOUS_VALUE = args.n_continuous_repeats
            
        if args.analysis_type == 'baseline':
            # Run only baseline analysis
            baseline_results = run_baseline_analysis(
                params_to_test=args.params,
                output_dir=args.output_dir,
                n_baseline_repeats=args.n_baseline_repeats,
            )
            plot_parameter_effects(baseline_results, os.path.join(args.output_dir, 'baseline_plots'))
            create_summary_heatmap(baseline_results, 
                                 os.path.join(args.output_dir, 'baseline_sensitivity_heatmap.png'))
            create_variance_equality_heatmap(baseline_results,
                                           os.path.join(args.output_dir, 'baseline_variance_equality_heatmap.png'))
              
        elif args.analysis_type == 'randomized':
            # Run only random baseline analysis
            random_results = run_random_baseline_analysis(
                n_samples=args.n_random_samples,
                n_repeats_per_sample=args.n_random_repeats_per_sample,
                output_dir=args.output_dir
            )
            # For random baseline, we'll create special plots
            plot_random_baseline_results(random_results, os.path.join(args.output_dir, 'random_baseline_plots'))
            # Note: Random baseline doesn't use systematic parameter variation, so variance heatmap isn't applicable
    else:

        if args.analysis_type == 'randomized':
            # Try to load random baseline results
            try:
                with open(os.path.join(args.output_dir, 'random_baseline_analysis.pkl'), 'rb') as f:
                    random_results = pickle.load(f)
                plot_random_baseline_results(random_results, os.path.join(args.output_dir, 'random_baseline_plots'))
                # Note: Random baseline doesn't use systematic parameter variation, so variance heatmap isn't applicable
            except FileNotFoundError:
                print("Random baseline analysis results not found.")
                
        elif args.analysis_type == 'baseline':
            results = {}
            
            # Try to load combined baseline results first
            try:
                with open(os.path.join(args.output_dir, 'all_baseline_analysis.pkl'), 'rb') as f:
                    results = pickle.load(f)
                print("Loaded combined baseline analysis results.")
            except FileNotFoundError:
                print("Combined baseline analysis results not found. Loading individual parameter files...")
                
                # Load individual parameter files
                for param_name in PARAMS_OF_INTEREST:
                    baseline_file = os.path.join(args.output_dir, f'{param_name}_baseline_analysis.pkl')
                    
                    if os.path.exists(baseline_file):
                        try:
                            with open(baseline_file, 'rb') as f:
                                results[param_name] = pickle.load(f)
                            print(f"Loaded baseline results for {param_name}")
                        except Exception as e:
                            print(f"Error loading baseline results for {param_name}: {e}")
                
                if not results:
                    print("No baseline analysis results found. Try running the analysis first.")
                    sys.exit(1)
            
            plot_parameter_effects(results, os.path.join(args.output_dir, 'baseline_plots'))
            create_summary_heatmap(results, 
                                 os.path.join(args.output_dir, 'baseline_sensitivity_heatmap.png'))
            create_variance_equality_heatmap(results,
                                           os.path.join(args.output_dir, 'baseline_variance_equality_heatmap.png'))
    
    # Handle side-by-side comparison if requested
    if args.plot_side_to_side:
        print("\nCreating side-by-side comparison plots...")
        
        # Load both baseline and randomized results
        baseline_results = None
        random_results = None
        
        # Try to load baseline results
        try:
            with open(os.path.join(args.output_dir, 'all_baseline_analysis.pkl'), 'rb') as f:
                baseline_results = pickle.load(f)
            print("Loaded baseline results for side-by-side comparison.")
        except FileNotFoundError:
            print("Combined baseline analysis results not found. Loading individual parameter files...")
            baseline_results = {}
            
            # Load individual parameter files
            for param_name in BASELINE_PARAMS_OF_INTEREST:
                baseline_file = os.path.join(args.output_dir, f'{param_name}_baseline_analysis.pkl')
                
                if os.path.exists(baseline_file):
                    try:
                        with open(baseline_file, 'rb') as f:
                            baseline_results[param_name] = pickle.load(f)
                        print(f"Loaded baseline results for {param_name}")
                    except Exception as e:
                        print(f"Error loading baseline results for {param_name}: {e}")
            
            if not baseline_results:
                print("No baseline analysis results found.")
        
        # Try to load randomized results
        try:
            with open(os.path.join(args.output_dir, 'random_baseline_analysis.pkl'), 'rb') as f:
                random_results = pickle.load(f)
            print("Loaded randomized results for side-by-side comparison.")
        except FileNotFoundError:
            print("Random baseline analysis results not found.")
        
        # Create side-by-side plots if both results are available
        if baseline_results and random_results:
            plot_side_by_side_comparison(
                baseline_results, 
                random_results, 
                os.path.join(args.output_dir, 'side_by_side_plots')
            )
        else:
            print("Cannot create side-by-side plots: both baseline and randomized results are required.")
            if not baseline_results:
                print("  - Missing baseline results")
            if not random_results:
                print("  - Missing randomized results")
           
    print("\nAnalysis complete! Check the output directory for results and plots.")