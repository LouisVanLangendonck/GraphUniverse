"""
Script to validate that GraphUniverse generation respects property constraints.
Tests homophily, avg_degree, n_nodes, and n_communities ranges.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Any, Optional
import argparse

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphFamilyGenerator

# Properties to validate with their test ranges
PROPERTY_CONFIGS = {
    'homophily': {
        'test_ranges': [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.0, 1.0)],
        'param_name': 'homophily_range',
        'property_key': 'homophily_levels'
    },
    'avg_degree': {
        'test_ranges': [(1.0, 2.0), (2.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 15.0), (0.0, 15.0)],
        'param_name': 'avg_degree_range',
        'property_key': 'avg_degrees'
    },
    'n_nodes': {
        'test_ranges': [(50, 100), (100, 200), (200, 400), (400, 600), (600, 1000), (50, 1000)],
        'param_name': ('min_n_nodes', 'max_n_nodes'),  # tuple for min/max params
        'property_key': 'node_counts'
    },
    'n_communities': {
        'test_ranges': [(2, 5), (5, 12), (12, 16), (2, 16)],
        'param_name': ('min_communities', 'max_communities'),
        'property_key': 'community_counts'
    }
}

# Fixed randomization ranges for other parameters
RANDOMIZATION_RANGES = {
    'edge_probability_variance': (0.0, 1.0),
    'degree_heterogeneity': (0.0, 1.0),
    'degree_separation_range': (0.0, 1.0),
    'feature_dim': [10, 50, 100],  # discrete choices
    'center_variance': (0.5, 2.0),
    'cluster_variance': (0.05, 0.5),
    'degree_center_method': ['linear', 'random', 'constant'],
    'community_cooccurrence_homogeneity': (0.0, 1.0),
    'use_dccc_sbm': [True],
    'degree_distribution': ['power_law'],
    'power_law_exponent_range': (1.5, 5.0),
    'exponential_rate_range': (0.3, 1.0),
    'uniform_min_factor_range': (0.3, 0.7),
    'uniform_max_factor_range': (1.3, 2.0),
    'disable_deviation_limiting': [False],
    'max_mean_community_deviation': (0.05, 0.10),
    'min_edge_density': (0.001, 0.001),
    'max_retries': [5, 10],
    'avg_degree_range': (1.0, 10.0),
    'homophily_range': (0.0, 1.0),
    'min_n_nodes': (50, 200),
    'max_n_nodes': (200, 500),
    'min_communities': (2, 8),
    'max_communities': (8, 16)
}

# Parameters that should be single values (not ranges)
SINGLE_VALUE_PARAMS = {
    'min_n_nodes': (50, 200),
    'max_n_nodes': (200, 1000),
    'min_communities': (2, 8),
    'max_communities': (8, 16),
    'edge_probability_variance': (0.0, 1.0),
    'degree_heterogeneity': (0.0, 1.0),
    'feature_dim': [10, 50, 100],
    'center_variance': (0.5, 2.0),
    'cluster_variance': (0.05, 0.5),
    'degree_center_method': ['linear', 'constant'],
    'community_cooccurrence_homogeneity': (0.0, 1.0),
    'use_dccc_sbm': [True],
    'degree_distribution': ['power_law'],
    'disable_deviation_limiting': [False],
    'max_mean_community_deviation': (0.05, 0.10),
    'min_edge_density': (0.001, 0.002),
    'max_retries': [5, 10]
}

# Parameters that should be ranges (tuples)
RANGE_PARAMS = {
    'avg_degree_range': (1.0, 10.0),
    'homophily_range': (0.0, 1.0),
    'degree_separation_range': (0.0, 1.0),
    'power_law_exponent_range': (1.5, 5.0),
    'exponential_rate_range': (0.3, 1.0),
    'uniform_min_factor_range': (0.3, 0.7),
    'uniform_max_factor_range': (1.3, 2.0)
}

# Experiment settings
REPEATS_PER_RANGE = 3
GRAPHS_PER_FAMILY = 10
UNIVERSE_K = 20  # Total communities in universe


def generate_random_params(fixed_property, fixed_range, seed=None):
    """
    Generate random parameters for all non-fixed properties.
    
    Args:
        fixed_property: The property being validated (e.g., 'homophily')
        fixed_range: The range being tested (e.g., (0.1, 0.3))
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of all parameters needed for GraphFamilyGenerator
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = {}
    
    # Set the fixed parameter
    if fixed_property in PROPERTY_CONFIGS:
        print(f"Fixed property: {fixed_property}")
        param_name = PROPERTY_CONFIGS[fixed_property]['param_name']
        if isinstance(param_name, tuple):
            # Handle min/max parameters
            params[param_name[0]] = fixed_range[0]
            params[param_name[1]] = fixed_range[1]
        else:
            params[param_name] = fixed_range
    
    # Randomize single value parameters
    for param, range_or_choices in SINGLE_VALUE_PARAMS.items():
        # Skip if this parameter is the fixed property's parameter
        fixed_param_name = PROPERTY_CONFIGS.get(fixed_property, {}).get('param_name')
        if isinstance(fixed_param_name, tuple):
            # For tuple parameters (min/max), skip both min and max
            if param in fixed_param_name:
                continue
        elif param == fixed_param_name:
            # For single parameters, skip if it matches
            continue
            
        if isinstance(range_or_choices, tuple):
            # Single continuous value
            params[param] = np.random.uniform(range_or_choices[0], range_or_choices[1])
        elif isinstance(range_or_choices, list):
            # Discrete choices
            params[param] = np.random.choice(range_or_choices)
    
    # Randomize range parameters
    for param, range_or_choices in RANGE_PARAMS.items():
        # Skip if this parameter is the fixed property's parameter
        fixed_param_name = PROPERTY_CONFIGS.get(fixed_property, {}).get('param_name')
        if isinstance(fixed_param_name, tuple):
            # For tuple parameters (min/max), skip both min and max
            if param in fixed_param_name:
                continue
        elif param == fixed_param_name:
            # For single parameters, skip if it matches
            continue
            
        # Generate a tuple range
        min_val = np.random.uniform(range_or_choices[0], range_or_choices[1])
        max_val = np.random.uniform(min_val, range_or_choices[1])
        params[param] = (min_val, max_val)
    
    # Handle special constraints
    # Ensure max > min for range parameters
    if 'min_n_nodes' in params and 'max_n_nodes' not in params:
        params['max_n_nodes'] = params['min_n_nodes'] + np.random.randint(50, 200)
    if 'min_communities' in params and 'max_communities' not in params:
        params['max_communities'] = min(params['min_communities'] + np.random.randint(2, 5), UNIVERSE_K)
    
    # Ensure avg_degree_range makes sense given n_nodes
    if 'avg_degree_range' in params and 'max_n_nodes' in params:
        max_possible_degree = params.get('min_n_nodes', 50) - 1
        if isinstance(params['avg_degree_range'], tuple) and params['avg_degree_range'][1] > max_possible_degree:
            params['avg_degree_range'] = (params['avg_degree_range'][0], 
                                          min(params['avg_degree_range'][1], max_possible_degree))
    
    return params


def run_validation_experiments(output_dir='validation_results'):
    """
    Run all validation experiments and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for property_name, config in PROPERTY_CONFIGS.items():
        print(f"\nValidating {property_name}...")
        
        property_results = {
            'target_ranges': config['test_ranges'],
            'actual_values': [],
            'metadata': []
        }
        
        for range_idx, target_range in enumerate(config['test_ranges']):
            print(f"  Testing range {target_range}...")
            
            range_values = []
            range_metadata = []
            
            for repeat in range(REPEATS_PER_RANGE):
                print(f"    Repeat {repeat + 1}/{REPEATS_PER_RANGE}")
                
                # Generate random parameters
                seed = range_idx * 1000 + repeat
                params = generate_random_params(property_name, target_range, seed=seed)
                
                try:
                    # Create universe
                    universe = GraphUniverse(
                        K=UNIVERSE_K,
                        edge_probability_variance=params.get('edge_probability_variance', 0.5),
                        feature_dim=params.get('feature_dim', 0),
                        center_variance=params.get('center_variance', 1.0),
                        cluster_variance=params.get('cluster_variance', 0.1),
                        degree_center_method=params.get('degree_center_method', 'linear'),
                        community_cooccurrence_homogeneity=params.get('community_cooccurrence_homogeneity', 1.0),
                        seed=seed
                    )
                    
                    # Create family generator
                    generator = GraphFamilyGenerator(
                        universe=universe,
                        min_n_nodes=params.get('min_n_nodes', 50),
                        max_n_nodes=params.get('max_n_nodes', 200),
                        min_communities=params.get('min_communities', 2),
                        max_communities=params.get('max_communities', 5),
                        homophily_range=params.get('homophily_range', (0.0, 1.0)),
                        avg_degree_range=params.get('avg_degree_range', (1.0, 10.0)),
                        use_dccc_sbm=params.get('use_dccc_sbm', False),
                        degree_heterogeneity=params.get('degree_heterogeneity', 0.5),
                        degree_separation_range=params.get('degree_separation_range', (0.5, 0.5)),
                        degree_distribution=params.get('degree_distribution', 'power_law'),
                        power_law_exponent_range=params.get('power_law_exponent_range', (2.0, 3.5)),
                        exponential_rate_range=params.get('exponential_rate_range', (0.3, 1.0)),
                        uniform_min_factor_range=params.get('uniform_min_factor_range', (0.3, 0.7)),
                        uniform_max_factor_range=params.get('uniform_max_factor_range', (1.3, 2.0)),
                        disable_deviation_limiting=params.get('disable_deviation_limiting', False),
                        max_mean_community_deviation=params.get('max_mean_community_deviation', 0.10),
                        min_edge_density=params.get('min_edge_density', 0.005),
                        max_retries=params.get('max_retries', 5),
                        seed=seed
                    )
                    
                    # Generate graphs
                    graphs = generator.generate_family(
                        n_graphs=GRAPHS_PER_FAMILY,
                        show_progress=False
                    )
                    
                    # Use analyze_graph_family_properties to get property values
                    properties = generator.analyze_graph_family_properties()
                    property_key = config['property_key']
                    values = properties[property_key]
                    
                    range_values.extend(values)
                    range_metadata.append({
                        'repeat': repeat,
                        'params': params,
                        'n_graphs_generated': len(graphs),
                        'generation_stats': generator.generation_stats,
                        'all_properties': properties
                    })
                    
                except Exception as e:
                    print(f"      Error in repeat {repeat}: {e}")
                    range_metadata.append({
                        'repeat': repeat,
                        'params': params,
                        'error': str(e)
                    })
            
            property_results['actual_values'].append(range_values)
            property_results['metadata'].append(range_metadata)
        
        all_results[property_name] = property_results
        
        # Save intermediate results
        with open(os.path.join(output_dir, f'{property_name}_results.pkl'), 'wb') as f:
            pickle.dump(property_results, f)
    
    # Save all results
    with open(os.path.join(output_dir, 'all_validation_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results


def plot_property_validation(results_dict, save_path=None):
    """
    Create validation plots from experimental results.
    """
    # Determine number of repeats from the data structure
    num_repeats = 0
    for prop_data in results_dict.values():
        if prop_data['metadata'] and prop_data['metadata'][0]:
            num_repeats = len(prop_data['metadata'][0])
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    properties = ['homophily', 'avg_degree', 'n_nodes', 'n_communities']
    
    for idx, prop in enumerate(properties):
        ax = axes[idx]
        
        if prop not in results_dict:
            ax.text(0.5, 0.5, f'No data for {prop}', ha='center', va='center')
            continue
        
        data = results_dict[prop]
        target_ranges = data['target_ranges']
        actual_values = data['actual_values']
        
        # Sort by range maximum value (right end) in descending order, but put the longest range at the bottom
        range_spans = [target_ranges[i][1] - target_ranges[i][0] for i in range(len(target_ranges))]
        max_span = max(range_spans)
        
        # Sort by maximum value in descending order, but put the longest range at the bottom
        sorted_indices = sorted(range(len(target_ranges)), 
                               key=lambda i: (range_spans[i] == max_span, target_ranges[i][1]), reverse=True)
        
        positions = []
        for i, idx in enumerate(sorted_indices):
            target_range = target_ranges[idx]
            values = actual_values[idx]
            
            if not values:
                continue
            
            # Calculate coverage
            within_range = sum(1 for v in values if target_range[0] <= v <= target_range[1])
            coverage = within_range / len(values) * 100 if values else 0
            
            # Choose color based on coverage
            if coverage >= 90:
                color = 'green'
            elif coverage >= 70:
                color = 'orange'
            else:
                color = 'red'
            
            # Draw target range as horizontal bar
            ax.barh(i, target_range[1] - target_range[0], left=target_range[0], 
                   height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add target range text
            if prop in ['homophily', 'avg_degree']:
                range_text = f"{target_range[0]:.1f}-{target_range[1]:.1f}"
            else:
                range_text = f"{int(target_range[0])}-{int(target_range[1])}"
            
            # Position text in the middle of the bar
            text_x = target_range[0] + (target_range[1] - target_range[0]) / 2
            ax.text(text_x, i, range_text, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
            
            # Add actual generated values with different markers for each repeat
            if values:
                # We need to separate the values by repeat
                # Each repeat generated GRAPHS_PER_FAMILY graphs
                for repeat_idx in range(num_repeats):
                    start_idx = repeat_idx * GRAPHS_PER_FAMILY
                    end_idx = start_idx + GRAPHS_PER_FAMILY
                    repeat_values = values[start_idx:end_idx]
                    
                    if repeat_values:
                        # Choose marker based on repeat - more general for any number of repeats
                        markers = ['x', '+', 'o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h', 'H', 'd']
                        marker = markers[repeat_idx % len(markers)]
                        
                        # Add some jitter to the y-position to avoid overlap
                        y_jitter = np.random.normal(0, 0.1, size=len(repeat_values))
                        ax.scatter(repeat_values, [i + jitter for jitter in y_jitter], 
                                  alpha=0.6, s=20, color='gray', marker=marker, zorder=5)
            
            # Add coverage annotation on the right
            ax.text(target_range[1] + (target_range[1] - target_range[0]) * 0.1, i, 
                   f'{coverage:.0f}%', ha='left', va='center', 
                   fontsize=9, fontweight='bold', color=color)
            
            positions.append(i)
        
        # Set y-axis labels
        ax.set_yticks(positions)
        ax.set_yticklabels([f"Range {i+1}" for i in range(len(positions))])
        
        ax.set_xlabel('Target Range Values')
        ax.set_ylabel('Target Ranges')
        ax.set_title(f'{prop.replace("_", " ").title()} Validation')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove individual legend - will add one for the entire figure
        
        # Set x-axis limits to accommodate all ranges
        all_ranges = [target_ranges[i] for i in sorted_indices]
        if all_ranges:
            min_val = min(r[0] for r in all_ranges)
            max_val = max(r[1] for r in all_ranges)
            padding = (max_val - min_val) * 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
    
    # Add a single legend for the entire figure - dynamic based on number of repeats
    markers = ['x', '+', 'o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h', 'H', 'd']
    legend_elements = []
    for repeat_idx in range(num_repeats):
        marker = markers[repeat_idx % len(markers)]
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='gray', linestyle='', markersize=8, 
                      label=f'Repeat {repeat_idx + 1}')
        )
    
    # Add legend to the figure outside the plot area
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.95, 0.5), 
               title='Individual Families', title_fontsize=10)
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.90)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate graph generation properties')
    parser.add_argument('--output-dir', default='validation_results', 
                       help='Directory to save results')
    parser.add_argument('--n-repeats', type=int, default=5,
                       help='Number of repeats per range')
    parser.add_argument('--n-graphs', type=int, default=20,
                       help='Number of graphs per family')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot existing results')
    
    args = parser.parse_args()
    
    if not args.plot_only:
        # Update global settings
        REPEATS_PER_RANGE = args.n_repeats
        GRAPHS_PER_FAMILY = args.n_graphs
        
        # Run experiments
        results = run_validation_experiments(args.output_dir)
    else:
        # Load existing results
        with open(os.path.join(args.output_dir, 'all_validation_results.pkl'), 'rb') as f:
            results = pickle.load(f)
    
    # Create plots
    plot_property_validation(results, 
                           save_path=os.path.join(args.output_dir, 'validation_plot.png'))
    
    # Print summary statistics
    print("\nValidation Summary:")
    for prop, data in results.items():
        print(f"\n{prop}:")
        for i, (range_val, values) in enumerate(zip(data['target_ranges'], 
                                                    data['actual_values'])):
            if values:
                coverage = sum(1 for v in values if range_val[0] <= v <= range_val[1]) / len(values) * 100
                print(f"  Range {range_val}: {coverage:.1f}% coverage ({len(values)} graphs)") 