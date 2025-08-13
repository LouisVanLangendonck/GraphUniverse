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

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphFamilyGenerator

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
    Create categorical boxplot with statistical testing - EXACT same logic as randomized method.
    
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
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Calculate non-parametric test significance for categorical parameters
    from scipy.stats import kruskal
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
            significance = 'ns'
        
        # Add significance annotation for location test
        ax.text(0.05, 0.95, f'{test_name} p={p_value:.3f} {significance}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=annotation_color, alpha=0.7))
        
        # Add variance equality test
        try:
            from scipy.stats import levene
            levene_stat, var_p_value = levene(*box_data)
            if var_p_value < 0.001:
                var_significance = '***'
            elif var_p_value < 0.01:
                var_significance = '**'
            elif var_p_value < 0.05:
                var_significance = '*'
            else:
                var_significance = 'ns'
            
            # Use same annotation color for variance test in baseline, lightyellow in randomized
            var_annotation_color = 'lightgray' if annotation_color == 'lightgray' else 'lightyellow'
            ax.text(0.05, 0.85, f"Levene's test p={var_p_value:.3f} {var_significance}", 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=var_annotation_color, alpha=0.7))
        except:
            pass
    except:
        pass


def maybe_add_legend(axis):
    handles, labels = axis.get_legend_handles_labels()
    # Only add a legend if there are non-private labels
    labels = [lab for lab in labels if lab and not lab.startswith("_")]
    if labels:
        axis.legend(frameon=False)


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
BASELINE_PARAMS_OF_INTEREST = ['edge_probability_variance', 'cluster_variance', 
'degree_center_method', 'min_n_nodes', 'homophily_range', 'avg_degree_range', 
'degree_separation_range', 'use_dccc_sbm', 'power_law_exponent_range']

# Parameters for random analysis (excludes use_dccc_sbm)
RANDOM_PARAMS_OF_INTEREST = ['edge_probability_variance', 'cluster_variance', 
'degree_center_method', 'min_n_nodes', 'homophily_range', 'avg_degree_range', 
'degree_separation_range', 'power_law_exponent_range']

# Keep backward compatibility
PARAMS_OF_INTEREST = BASELINE_PARAMS_OF_INTEREST

# All parameters that can be varied
ALL_VARIABLE_PARAMS = {
    # Universe parameters
    'edge_probability_variance': {
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
        'random_range': (0.1, 0.5),
        'level': 'universe'
    },
    'degree_center_method': {
        'type': 'categorical',
        'test_values': ['random', 'constant'],
        'random_range': ['random', 'constant'],
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
        'paired_with': 'min_n_nodes'
    },
    'min_communities': {
        'type': 'discrete',
        'test_values': [2, 4, 6],
        'random_range': (2, 6),
        'level': 'family',
        'paired_with': 'max_communities'
    },
    'max_communities': {
        'type': 'discrete',
        'test_values': [4, 6, 8],
        'random_range': (4, 8),
        'level': 'family',
        'paired_with': 'min_communities'
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
        'random_range': (2.0, 15.0),
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

# Fixed settings
REPEATS_PER_VALUE = 1
GRAPHS_PER_FAMILY = 30
UNIVERSE_K = 15

# Baseline configurations for fixed parameter analysis
BASELINE_UNIVERSE_PARAMS = {
    'edge_probability_variance': 0.5,
    'feature_dim': 15,
    'center_variance': 0.5,
    'cluster_variance': 0.2,
    'degree_center_method': 'random',
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
SIGNAL_METRICS = ['feature_signal', 'degree_signal', 'triangle_signal', 'structure_signal']
CONSISTENCY_METRICS = ['pattern_preservation', 'generation_fidelity', 'degree_consistency']
PROPERTY_METRICS = ['graph_generation_times', 'homophily_levels', 'avg_degrees', 'tail_ratio_99']


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

        elif 'min_communities':
            # Ensure max > min
            if param_config['level'] == 'universe':
                params['universe'][paired_param] = fixed_value + 3
            else:
                params['family'][paired_param] = fixed_value + 3
    
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
                        edge_probability_variance=universe_params['edge_probability_variance'],
                        feature_dim=universe_params['feature_dim'],
                        center_variance=universe_params['center_variance'],
                        cluster_variance=universe_params['cluster_variance'],
                        degree_center_method=universe_params['degree_center_method'],
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
                    graphs = generator.generate_family(
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
                        'n_graphs': len(graphs)
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
        # Create figure with subplots for all metrics
        n_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(PROPERTY_METRICS)
        n_cols = 4
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 5.5 * n_rows))
        axes = axes.flatten()
        
        # Get test values and prepare x-axis
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Use helper function to get x values and labels
        x_values, x_labels = get_plot_x_values(param_name, test_values, param_config)
        
        plot_idx = 0
        
        # Plot signal metrics (first row)
        for metric in SIGNAL_METRICS:
            ax = axes[plot_idx]
            metric_data = param_results['signal_metrics'][metric]
            
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
                
                # Use helper function with signal metric colors (blue)
                signal_colors = {'face': 'lightblue', 'median': 'blue', 'whiskers': 'blue'}
                create_categorical_boxplot_with_tests(ax, box_data, box_positions, x_labels, signal_colors, 'lightgray')
            else:
                # For continuous parameters, use the original line plot with confidence intervals
                # Calculate confidence intervals for each data point
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for data_point in metric_data:
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            mean_val, ci_lower, ci_upper = calculate_confidence_intervals(all_graph_values)
                            means.append(mean_val)
                            ci_lowers.append(ci_lower)
                            ci_uppers.append(ci_upper)
                        else:
                            means.append(np.nan)
                            ci_lowers.append(np.nan)
                            ci_uppers.append(np.nan)
                    else:
                        means.append(np.nan)
                        ci_lowers.append(np.nan)
                        ci_uppers.append(np.nan)
                
                # Plot confidence intervals as shaded areas
                valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                if valid_indices:
                    valid_x = [x_values[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                    valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                    
                    # Plot shaded confidence intervals
                    ax.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                  alpha=0.3, color='blue', label='95% CI')
                    
                    # Plot mean line
                    ax.plot(valid_x, valid_means, 'o-', markersize=8, linewidth=2, 
                           color='blue', alpha=0.8, label='Mean')
                
                # Add individual points
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                            ax.scatter([x_values[i]] * len(all_graph_values), 
                                     np.array(all_graph_values) + y_jitter, 
                                     alpha=0.3, s=30, color='gray')
            
            # Calculate and display correlation with significance and direction
            # Skip for categorical parameters since they're handled in the helper function
            if not is_categorical:
                # Use all individual data points, not just means
                all_x_values = []
                all_y_values = []
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            # Add x value for each individual measurement
                            all_x_values.extend([x_values[i]] * len(all_graph_values))
                            all_y_values.extend(all_graph_values)
                
                if len(all_x_values) > 2:
                    result_tuple = calculate_correlation_with_significance(
                        np.array(all_x_values), np.array(all_y_values), param_config['type']
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
                        
                        display_text = formatted_value
                    
                    # Add significance
                    ax.text(0.05, 0.95, f'{display_text} ({results["significance"]})', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(get_plot_param_name(param_name))
            ax.set_ylabel('Signal Value')
            ax.set_ylim(0, 1)  # Fix y-axis limits for signal metrics
            apply_publication_style(ax)
            maybe_add_legend(ax)
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Plot property metrics (second row)
        for metric in PROPERTY_METRICS:
            ax = axes[plot_idx]
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
                
                # Use helper function with property metric colors (orange)
                property_colors = {'face': 'lightgoldenrodyellow', 'median': 'orange', 'whiskers': 'orange'}
                create_categorical_boxplot_with_tests(ax, box_data, box_positions, x_labels, property_colors, 'lightgray')
            else:
                # For continuous parameters, use the original line plot with confidence intervals
                # Calculate confidence intervals for each data point
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for data_point in metric_data:
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            mean_val, ci_lower, ci_upper = calculate_confidence_intervals(all_graph_values)
                            means.append(mean_val)
                            ci_lowers.append(ci_lower)
                            ci_uppers.append(ci_upper)
                        else:
                            means.append(np.nan)
                            ci_lowers.append(np.nan)
                            ci_uppers.append(np.nan)
                    else:
                        means.append(np.nan)
                        ci_lowers.append(np.nan)
                        ci_uppers.append(np.nan)
                
                # Plot confidence intervals as shaded areas
                valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                if valid_indices:
                    valid_x = [x_values[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                    valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                    
                    # Plot shaded confidence intervals
                    ax.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                  alpha=0.3, color='orange', label='95% CI')
                    
                    # Plot mean line
                    ax.plot(valid_x, valid_means, '^-', markersize=8, linewidth=2, 
                           color='orange', alpha=0.8, label='Mean')
                
                # Add individual points
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                            ax.scatter([x_values[i]] * len(all_graph_values), 
                                     np.array(all_graph_values) + y_jitter, 
                                     alpha=0.3, s=30, color='gold')
            
            # Calculate and display correlation with significance and direction
            # Skip for categorical parameters since they're handled in the helper function
            if not is_categorical:
                # Use all individual data points, not just means
                all_x_values = []
                all_y_values = []
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            # Add x value for each individual measurement
                            all_x_values.extend([x_values[i]] * len(all_graph_values))
                            all_y_values.extend(all_graph_values)
                
                if len(all_x_values) > 2:
                    result_tuple = calculate_correlation_with_significance(
                        np.array(all_x_values), np.array(all_y_values), param_config['type']
                    )
                    results = extract_correlation_results(result_tuple)
                    correlation_value, direction, significance, ci_lower, ci_upper = results['correlation_value'], results['direction'], results['significance'], results['ci_lower'], results['ci_upper']
                    
                    # For continuous parameters, show correlation
                    if significance == 'ns':
                        display_text = 'NS'
                    else:
                        if abs(correlation_value) < 0.01:
                            formatted_value = '<0.01'
                        else:
                            formatted_value = f'{abs(correlation_value):.3f}'
                        
                        if direction == 'positive':
                            display_text = f'{formatted_value}+'
                        elif direction == 'negative':
                            display_text = f'{formatted_value}-'
                        else:
                            display_text = formatted_value
                    
                    # Add confidence interval for continuous parameters only
                    ax.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
                    
                if metric == 'homophily_levels':
                    ax.set_ylim(0, 1)
            
            title = f'{metric.replace("_", " ").title()}'
            if 'Time' in title: 
                title += ' (s)'
            ax.set_title(title)
            ax.set_xlabel(get_plot_param_name(param_name))
            ax.set_ylabel('Property Value')
            apply_publication_style(ax)
            maybe_add_legend(ax)
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Plot consistency metrics (third row)
        for metric in CONSISTENCY_METRICS:
            ax = axes[plot_idx]
            metric_data = param_results['consistency_metrics'][metric]
            
            # Check if this is a categorical parameter
            is_categorical = param_config['type'] in ['categorical', 'boolean']
            
            if is_categorical:
                # For categorical parameters, use boxplots with statistical testing
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
                            
                
                # Use helper function with consistency metric colors (green)
                consistency_colors = {'face': 'lightgreen', 'median': 'darkgreen', 'whiskers': 'darkgreen'}
                create_categorical_boxplot_with_tests(ax, box_data, box_positions, x_labels, consistency_colors, 'lightgray')
            else:
                # For continuous parameters, use the original line plot with confidence intervals
                # Calculate confidence intervals for each data point
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for data_point in metric_data:
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            mean_val, ci_lower, ci_upper = calculate_confidence_intervals(all_graph_values)
                            means.append(mean_val)
                            ci_lowers.append(ci_lower)
                            ci_uppers.append(ci_upper)
                        else:
                            means.append(np.nan)
                            ci_lowers.append(np.nan)
                            ci_uppers.append(np.nan)
                    else:
                        means.append(np.nan)
                        ci_lowers.append(np.nan)
                        ci_uppers.append(np.nan)
                
                # Plot confidence intervals as shaded areas
                valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                if valid_indices:
                    valid_x = [x_values[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                    valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                    
                    # Plot shaded confidence intervals
                    ax.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                  alpha=0.3, color='darkgreen', label='95% CI')
                    
                    # Plot mean line
                    ax.plot(valid_x, valid_means, 's-', markersize=8, linewidth=2, 
                           color='darkgreen', alpha=0.8, label='Mean')
                
                # Add individual points
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                            ax.scatter([x_values[i]] * len(all_graph_values), 
                                     np.array(all_graph_values) + y_jitter, 
                                     alpha=0.3, s=30, color='green')
            
            # Calculate and display correlation with significance and direction
            # Skip for categorical parameters since they're handled in the helper function
            if not is_categorical:
                # Use all individual data points, not just means
                all_x_values = []
                all_y_values = []
                
                for i, data_point in enumerate(metric_data):
                    if data_point['values']:
                        # The values are already flattened when stored
                        all_graph_values = data_point['values']
                        
                        if all_graph_values:
                            # Add x value for each individual measurement
                            all_x_values.extend([x_values[i]] * len(all_graph_values))
                            all_y_values.extend(all_graph_values)
                
                if len(all_x_values) > 2:
                    result_tuple = calculate_correlation_with_significance(
                        np.array(all_x_values), np.array(all_y_values), param_config['type']
                    )
                    results = extract_correlation_results(result_tuple)
                    correlation_value, direction, significance, ci_lower, ci_upper = results['correlation_value'], results['direction'], results['significance'], results['ci_lower'], results['ci_upper']
                    
                    # For continuous parameters, show correlation
                    if significance == 'ns':
                        display_text = 'NS'
                    else:
                        if abs(correlation_value) < 0.01:
                            formatted_value = '<0.01'
                        else:
                            formatted_value = f'{abs(correlation_value):.3f}'
                        
                        if direction == 'positive':
                            display_text = f'{formatted_value}+'
                        elif direction == 'negative':
                            display_text = f'{formatted_value}-'
                        else:
                            display_text = formatted_value
                    
                    # Add confidence interval for continuous parameters only
                    ax.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(get_plot_param_name(param_name))
            ax.set_ylabel('Consistency Value')
            ax.set_ylim(0, 1)  # Fix y-axis limits for consistency metrics
            apply_publication_style(ax)
            maybe_add_legend(ax)
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Remove empty subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        # plt.suptitle(f'Parameter Effects: {get_plot_param_name(param_name)}', 
        #             fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{param_name}_effects.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_heatmap(results_dict, save_path='parameter_sensitivity_heatmap.png'):
    """
    Create a heatmap showing correlation values with significance and direction for all parameter-metric combinations.
    Only shows statistical significance from ** (p < 0.01) and above, treating * (p < 0.05) as NS.
    """
    params = list(results_dict.keys())
    metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS
    
    # Create correlation matrix and display matrix
    correlation_matrix = np.zeros((len(params), len(metrics)))
    display_matrix = np.empty((len(params), len(metrics)), dtype=object)
    significance_matrix = np.empty((len(params), len(metrics)), dtype=object)
    
    for i, param_name in enumerate(params):
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
            
            for k, data_point in enumerate(metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        # Add x value for each individual measurement
                        all_x_values.extend([x_values[k]] * len(all_graph_values))
                        all_y_values.extend(all_graph_values)
            
            if len(all_x_values) > 2 and np.std(all_x_values) > 0:
                # Use new correlation function with significance and direction
                result_tuple = calculate_correlation_with_significance(
                    np.array(all_x_values), np.array(all_y_values), param_config['type']
                )
                results = extract_correlation_results(result_tuple)
                correlation_value, direction, significance, ci_lower, ci_upper = results['correlation_value'], results['direction'], results['significance'], results['ci_lower'], results['ci_upper']
                
                correlation_matrix[i, j] = correlation_value
                significance_matrix[i, j] = significance
                
                # Only show significance from ** and above, treat * as NS
                if significance in ['**', '***']:
                    if param_config['type'] in ['categorical', 'boolean']:
                        # For categorical parameters, show non-parametric test significance
                        if direction != 'none':
                            # Shorten direction indicators: positive -> +, negative -> -
                            short_direction = '+' if direction == 'positive' else '-'
                            display_matrix[i, j] = f'{significance} {short_direction}'
                        else:
                            display_matrix[i, j] = significance
                    else:
                        # For continuous parameters, show correlation
                        display_matrix[i, j] = format_correlation_display(correlation_value, direction, significance, ci_lower, ci_upper)
                else:
                    display_matrix[i, j] = 'NS'
            else:
                correlation_matrix[i, j] = 0.0
                display_matrix[i, j] = 'NS'
                significance_matrix[i, j] = 'ns'
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(18, len(params) * 0.6 + 2), constrained_layout=True)
    
    # Remove grid lines
    ax.grid(False)
    
    # Create a custom colormap that distinguishes between significant and non-significant
    # Create a masked array for non-significant values
    ns_mask = np.zeros_like(correlation_matrix, dtype=bool)
    for i in range(len(params)):
        for j in range(len(metrics)):
            if display_matrix[i, j] == 'NS':
                ns_mask[i, j] = True
    
    # Create the main heatmap with blue-to-red colormap for -1 to +1 correlations
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels with larger, consistent font sizes
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels([get_plot_param_name(p) for p in params], fontsize=14)
    
    # Add colorbar with larger text
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Add text annotations with correlation values or significance stars
    for i in range(len(params)):
        for j in range(len(metrics)):
            correlation_val = correlation_matrix[i, j]
            significance = significance_matrix[i, j]
            param_name = params[i]
            param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
            
            # Choose text color based on correlation magnitude for better contrast
            if abs(correlation_val) < 0.4:
                text_color = "black"
            else:
                text_color = "white"
            
            # Show correlation value only if significant at ** or *** level
            if significance in ['**', '***']:
                # For categorical parameters, show stars instead of correlation values
                if param_config.get('type') in ['categorical', 'boolean']:
                    display_text = display_matrix[i, j]  # This contains the stars
                else:
                    # For continuous parameters, format correlation value with 2 decimal places
                    if abs(correlation_val) < 0.01:
                        display_text = '<0.01'
                    else:
                        display_text = f'{correlation_val:.2f}'
                
                text = ax.text(j, i, display_text,
                             ha="center", va="center", color=text_color,
                             fontweight='bold', fontsize=16)
            else:
                # Show NS for non-significant correlations
                text = ax.text(j, i, 'NS',
                             ha="center", va="center", color="gray", 
                             fontweight='normal', fontsize=16)
    
    plt.title('Parameter Sensitivity Analysis: Correlation (continuous) / Mann-Withney U Test (categorical)\n(Only ** and *** significance shown, * treated as NS)', 
              fontsize=18, pad=25)
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


def create_random_baseline_variance_heatmap(param_values, metric_values, save_dir='random_baseline_plots'):
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


def calculate_correlation_with_significance(x, y, param_type='continuous'):
    """
    Calculate correlation with direction and statistical significance.
    
    For continuous/range parameters: Uses Pearson correlation
    For categorical/boolean parameters: Uses non-parametric tests (Mann-Whitney U or Kruskal-Wallis)
                                       Also performs Levene's test for variance equality
    
    Args:
        x: Independent variable values
        y: Dependent variable values  
        param_type: Type of parameter ('continuous', 'categorical', 'boolean', 'range')
    
    Returns:
        For continuous: (correlation_value, direction, significance, ci_lower, ci_upper)
        For categorical: (correlation_value, direction, significance, ci_lower, ci_upper, variance_test_name, variance_p_value, variance_significance)
    """
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
        
        # Bootstrap for confidence intervals
        n_bootstrap = 100
        r_bootstrap = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(x), len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                r_boot, _ = pearsonr(x_boot, y_boot)
                if not np.isnan(r_boot):
                    r_bootstrap.append(r_boot)
            except:
                continue
        
        if r_bootstrap:
            ci_lower = np.percentile(r_bootstrap, 2.5)
            ci_upper = np.percentile(r_bootstrap, 97.5)
        else:
            ci_lower, ci_upper = correlation, correlation
        
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
            median_0 = np.median(groups[0])
            median_1 = np.median(groups[1])
            
            if median_0 > median_1:
                direction = 'positive'
            elif median_0 < median_1:
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
            # Only return 1 for ** and *** significance
            if significance in ['**', '***']:
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
            # Only return 1 for ** and *** significance
            if significance in ['**', '***']:
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
        x_values = list(range(len(test_values)))
        # Convert boolean values to True/False strings
        x_labels = []
        for v in test_values:
            if isinstance(v, bool):
                x_labels.append('True' if v else 'False')
            else:
                x_labels.append(str(v))
        return x_values, x_labels
    elif param_config['type'] == 'range':
        x_values = [(v[0] + v[1]) / 2 for v in test_values]
        x_labels = [f"{v[0]:.2f}-{v[1]:.2f}" for v in test_values]
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
                        params['universe'][paired_param] = params['universe'][param_name] + np.random.randint(1, 4)
                    else:
                        params['family'][paired_param] = params['family'][param_name] + np.random.randint(1, 4)
        
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
    print(f"Random parameter samples: {random_param_samples}")
    
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
                    edge_probability_variance=universe_params['edge_probability_variance'],
                    feature_dim=universe_params['feature_dim'],
                    center_variance=universe_params['center_variance'],
                    cluster_variance=universe_params['cluster_variance'],
                    degree_center_method=universe_params['degree_center_method'],
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
                graphs = generator.generate_family(
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


def plot_random_baseline_results(random_results, save_dir='random_baseline_plots'):
    """
    Create plots for random baseline results showing each parameter of interest vs all metrics.
    For categorical parameters: use boxplots with red lines between them and non-parametric test significance
    For continuous parameters: use scatter plots with red correlation lines and error bars
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
            
            # Create subplots for all metrics
            n_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(PROPERTY_METRICS)
            metric_axes = []
            
            # Create subplot grid for this parameter
            if n_metrics <= 6:
                # Use 2x3 grid for 6 or fewer metrics
                n_cols_metric = 3
                n_rows_metric = 2
            else:
                # Use 3x4 grid for more metrics
                n_cols_metric = 4
                n_rows_metric = 3
            
            # Create subplot for this parameter
            param_fig, param_axes = plt.subplots(n_rows_metric, n_cols_metric, 
                                               figsize=(18, 4.5 * n_rows_metric))
            if n_rows_metric == 1:
                param_axes = param_axes.reshape(1, -1)
            param_axes = param_axes.flatten()
            
            metric_idx = 0
            
            # Plot signal metrics
            for metric in SIGNAL_METRICS:
                if metric_idx < len(param_axes):
                    metric_ax = param_axes[metric_idx]
                    
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
                                
                                # Use helper function with signal metric colors (blue) - same as baseline
                                signal_colors = {'face': 'lightblue', 'median': 'blue', 'whiskers': 'blue'}
                                create_categorical_boxplot_with_tests(metric_ax, box_data, box_positions, x_labels, signal_colors, 'lightgray')
                                        
                            else:
                                # For continuous parameters, use scatter plot with error bars
                                metric_ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                                    fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1)
                
                                # Add red correlation line (no confidence intervals, just simple fit)
                                if len(valid_params) > 1:
                                    z = np.polyfit(valid_params, valid_metrics, 1)
                                    p = np.poly1d(z)
                                    metric_ax.plot(valid_params, p(valid_params), "r-", alpha=0.8, linewidth=2)
                                
                                # Calculate correlation with significance
                                corr, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                                                    np.array(valid_params), np.array(valid_metrics), param_config.get('type', 'continuous')
                                )
                
                                # Add correlation info
                                corr_display = format_correlation_display(corr, direction, significance, ci_lower, ci_upper)
                                metric_ax.text(0.05, 0.95, corr_display, 
                                                transform=metric_ax.transAxes, verticalalignment='top',
                                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
                                                
                            metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                            metric_ax.set_xlabel(get_plot_param_name(param_name))
                            metric_ax.set_ylabel('Value')
                            apply_publication_style(metric_ax)
                    else:
                        metric_ax.text(0.5, 0.5, f'No valid data', 
                                    ha='center', va='center', transform=metric_ax.transAxes)
                        metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                                
                metric_idx += 1
    
            # Plot property metrics  
            for metric in PROPERTY_METRICS:
                if metric_idx < len(param_axes):
                    metric_ax = param_axes[metric_idx]
                     
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
                            
                            # Use helper function with property metric colors (orange) - same as baseline
                            property_colors = {'face': 'lightgoldenrodyellow', 'median': 'orange', 'whiskers': 'orange'}
                            create_categorical_boxplot_with_tests(metric_ax, box_data, box_positions, x_labels, property_colors, 'lightgray')
                                     
                        else:
                            # For continuous parameters, use scatter plot with error bars
                            metric_ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                                fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1)
        
                            # Add red correlation line (no confidence intervals, just simple fit)
                            if len(valid_params) > 1:
                                z = np.polyfit(valid_params, valid_metrics, 1)
                                p = np.poly1d(z)
                                metric_ax.plot(valid_params, p(valid_params), "r-", alpha=0.8, linewidth=2)
                                
                            # Calculate correlation with significance
                            corr, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                                np.array(valid_params), np.array(valid_metrics), param_config.get('type', 'continuous')
                            )
                            
                            # Add correlation info
                            corr_display = format_correlation_display(corr, direction, significance, ci_lower, ci_upper)
                            metric_ax.text(0.05, 0.95, corr_display, 
                                transform=metric_ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
                             
                        metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                        metric_ax.set_xlabel(get_plot_param_name(param_name))
                        metric_ax.set_ylabel('Value')
                        apply_publication_style(metric_ax)
                else:
                    metric_ax.text(0.5, 0.5, f'No valid data', 
                                ha='center', va='center', transform=metric_ax.transAxes)
                    metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                     
                metric_idx += 1
    
            # Plot consistency metrics
            for metric in CONSISTENCY_METRICS:
                if metric_idx < len(param_axes):
                    metric_ax = param_axes[metric_idx]
                    
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
                                        # Special handling for degree_center_method labels
                                        if param_name == 'degree_center_method':
                                            x_labels.append('True' if param_val == 1 else 'False')
                                        else:
                                            x_labels.append(str(param_val))
                                
                                # Use helper function with consistency metric colors (green) - same as baseline
                                consistency_colors = {'face': 'lightgreen', 'median': 'darkgreen', 'whiskers': 'darkgreen'}
                                create_categorical_boxplot_with_tests(metric_ax, box_data, box_positions, x_labels, consistency_colors, 'lightgray')
                                        
                            else:
                                # For continuous parameters, use scatter plot with error bars
                                metric_ax.errorbar(valid_params, valid_metrics, yerr=valid_stds, 
                                    fmt='o', alpha=0.6, markersize=4, capsize=3, capthick=1)
                
                                # Add red correlation line (no confidence intervals, just simple fit)
                                if len(valid_params) > 1:
                                    z = np.polyfit(valid_params, valid_metrics, 1)
                                    p = np.poly1d(z)
                                    metric_ax.plot(valid_params, p(valid_params), "r-", alpha=0.8, linewidth=2)
                                
                                # Calculate correlation with significance
                                corr, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                                                    np.array(valid_params), np.array(valid_metrics), param_config.get('type', 'continuous')
                                )
                
                                # Add correlation info
                                corr_display = format_correlation_display(corr, direction, significance, ci_lower, ci_upper)
                                metric_ax.text(0.05, 0.95, corr_display, 
                                               transform=metric_ax.transAxes, verticalalignment='top',
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9))
                            
                            metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                            metric_ax.set_xlabel(get_plot_param_name(param_name))
                            metric_ax.set_ylabel('Value')
                            apply_publication_style(metric_ax)
                    else:
                        metric_ax.text(0.5, 0.5, f'No valid data', 
                                        ha='center', va='center', transform=metric_ax.transAxes)
                        metric_ax.set_title(f'{metric.replace("_", " ").title()}')
                        
                    metric_idx += 1
            
            # Hide unused subplots for this parameter
            for i in range(metric_idx, len(param_axes)):
                param_axes[i].set_visible(False)
            
            # Set the main title for this parameter
            param_fig.suptitle(f'{get_plot_param_name(param_name)} vs All Metrics', fontsize=16, y=0.98)
            param_fig.tight_layout()
            
            # Save individual parameter plot
            param_output_file = os.path.join(save_dir, f'{param_name}_random_baseline_analysis.png')
            param_fig.savefig(param_output_file, dpi=300, bbox_inches='tight')
            plt.close(param_fig)
            
            # Set the main plot title for this parameter
            ax.set_title(f'{get_plot_param_name(param_name)}')
            ax.text(0.5, 0.5, f'Parameter: {get_plot_param_name(param_name)}\nPlot saved as: {param_name}_random_baseline_analysis.png', 
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
    
    # Also create a summary correlation heatmap
    create_random_baseline_correlation_heatmap(param_values, metric_values, save_dir)
    
    # Create variance equality heatmap for categorical parameters
    create_random_baseline_variance_heatmap(param_values, metric_values, save_dir)
    
    plt.close()


def create_random_baseline_correlation_heatmap(param_values, metric_values, save_dir):
    """
    Create a correlation heatmap for random baseline results.
    """
    # Calculate correlations between all parameters and metrics
    all_metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS
    # Only include parameters of interest for the random baseline heatmap
    all_params = RANDOM_PARAMS_OF_INTEREST
    
    correlation_matrix = np.zeros((len(all_metrics), len(all_params)))
    
    for i, metric in enumerate(all_metrics):
        for j, param_name in enumerate(all_params):
            if param_name in param_values and metric in metric_values:
                param_vals = param_values[param_name]
                metric_vals = metric_values[metric]
                
                # Remove NaN values - use safer checking
                # Ensure we only process indices that exist in both lists
                max_idx = min(len(param_vals), len(metric_vals))
                valid_indices = [idx for idx in range(max_idx) 
                               if not (is_invalid_value(param_vals[idx]) or is_invalid_value(metric_vals[idx]))]
                
                if len(valid_indices) > 1:  # Need at least 2 data points for correlation
                    valid_params = [param_vals[idx] for idx in valid_indices]
                    valid_metrics = [metric_vals[idx] for idx in valid_indices]
                    
                    if len(valid_params) > 1:
                        corr, _ = pearsonr(valid_params, valid_metrics)
                        correlation_matrix[i, j] = corr
                    else:
                        correlation_matrix[i, j] = np.nan
                else:
                    correlation_matrix[i, j] = np.nan
            else:
                correlation_matrix[i, j] = np.nan
    
    # Determine significance mask (only show ** and ***)
    significance_mask = np.full_like(correlation_matrix, False, dtype=bool)
    significance_values = np.full_like(correlation_matrix, 'ns', dtype=object)
    for i, metric in enumerate(all_metrics):
        for j, param_name in enumerate(all_params):
            if param_name in param_values and metric in metric_values:
                param_vals = param_values[param_name]
                metric_vals = metric_values[metric]
                max_idx = min(len(param_vals), len(metric_vals))
                valid_indices = [idx for idx in range(max_idx) 
                                 if not (is_invalid_value(param_vals[idx]) or is_invalid_value(metric_vals[idx]))]
                if len(valid_indices) > 2:
                    param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
                    param_type = param_config.get('type', 'continuous')
                    
                    corr, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                        np.array([param_vals[idx] for idx in valid_indices], dtype=float),
                        np.array([metric_vals[idx] for idx in valid_indices], dtype=float),
                        param_type
                    )
                    significance_values[i, j] = significance
                    # Only mark if significant at ** or ***
                    if significance in ['**', '***']:
                        significance_mask[i, j] = True
                
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    
    # Remove grid lines
    ax.grid(False)
    
    # Use blue-to-red colormap for full correlation range -1 to +1
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar with larger text
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Set labels with larger, consistent font sizes
    ax.set_xticks(range(len(all_params)))
    ax.set_yticks(range(len(all_metrics)))
    ax.set_xticklabels([get_plot_param_name(p) for p in all_params], rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in all_metrics], fontsize=14)
    
    # Add annotations: show correlation values for significant cells, else NS
    for i in range(len(all_metrics)):
        for j in range(len(all_params)):
            if significance_mask[i, j]:
                val = correlation_matrix[i, j]
                param_name = all_params[j]
                param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
                significance = significance_values[i, j]
                
                # Choose text color based on correlation magnitude for better contrast
                if abs(val) < 0.4:
                    text_color = "black"
                else:
                    text_color = "white"
                
                # For categorical parameters, show stars instead of correlation values
                if param_config.get('type') in ['categorical', 'boolean']:
                    display_text = significance  # Show just the stars (**, ***)
                else:
                    # For continuous parameters, format correlation value with 2 decimal places
                    if abs(val) < 0.01:
                        display_text = '<0.01'
                    else:
                        display_text = f'{val:.2f}'
                
                ax.text(j, i, display_text, ha='center', va='center', color=text_color, fontsize=16, fontweight='bold')
            else:
                ax.text(j, i, 'NS', ha='center', va='center', color='gray', fontsize=16)
    
    ax.set_title('Random Baseline Parameter-Metric Correlations (Only **/***)', fontsize=18, pad=25)
    ax.set_xlabel('Parameters', fontsize=16)
    ax.set_ylabel('Metrics', fontsize=16)
    
    # constrained_layout already active
    
    # Save the heatmap
    output_file = os.path.join(save_dir, 'random_baseline_correlation_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Random baseline correlation heatmap saved to {output_file}")
    
    plt.close()


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
    
    args = parser.parse_args()
    
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
           
    print("\nAnalysis complete! Check the output directory for results and plots.")