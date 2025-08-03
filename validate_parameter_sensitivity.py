"""
Script to analyze how individual parameters affect signal and consistency metrics.
Tests each parameter with narrow ranges/values while randomizing all others.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphFamilyGenerator

INTERESTING_PARAMS_TO_VARIATE = ['edge_probability_variance', 'cluster_variance', 
 'degree_center_method', 'community_cooccurrence_homogeneity', 'min_n_nodes', 'homophily_range', 'avg_degree_range', 
 'degree_separation_range', 'use_dccc_sbm', 'power_law_exponent_range']

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
        'random_range': [10, 50, 100],
        'level': 'universe'
    },
    'center_variance': {
        'type': 'continuous',
        'test_values': [0.1, 0.5, 1.0],
        'random_range': (0.1, 1.0),
        'level': 'universe'
    },
    'cluster_variance': {
        'type': 'continuous',
        'test_values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'random_range': (0.1, 0.5),
        'level': 'universe'
    },
    'degree_center_method': {
        'type': 'categorical',
        'test_values': ['random', 'constant'],
        'random_range': ['random', 'constant'],
        'level': 'universe'
    },
    'community_cooccurrence_homogeneity': {
        'type': 'continuous',
        'test_values': [0.0, 0.25, 0.5, 0.75, 1.0],
        'random_range': (0.0, 1.0),
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
        'test_values': [2, 4, 6, 8],
        'random_range': (2, 8),
        'level': 'family',
        'paired_with': 'max_communities'
    },
    'max_communities': {
        'type': 'discrete',
        'test_values': [4, 8, 12, 16],
        'random_range': (4, 16),
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
        'test_values': [(0.0, 0.05), (0.1, 0.15), (0.2, 0.25), (0.4, 0.45), (0.6, 0.65), (0.8, 0.85)],
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
        'test_values': [0.05, 0.075, 0.10, 0.125, 0.15],
        'random_range': (0.05, 0.15),
        'level': 'family'
    },
}

# Fixed settings
REPEATS_PER_VALUE = 3
GRAPHS_PER_FAMILY = 20
UNIVERSE_K = 20

# Baseline configurations for fixed parameter analysis
BASELINE_UNIVERSE_PARAMS = {
    'edge_probability_variance': 0.5,
    'feature_dim': 15,
    'center_variance': 0.5,
    'cluster_variance': 0.2,
    'degree_center_method': 'random',
    'community_cooccurrence_homogeneity': 1.0
}

BASELINE_FAMILY_PARAMS = {
    'min_n_nodes': 150,
    'max_n_nodes': 150,
    'min_communities': 6,
    'max_communities': 6,
    'homophily_range': (0.50, 0.50),
    'avg_degree_range': (2.5, 2.5),
    'degree_heterogeneity': 0.5,
    'degree_separation_range': (0.6, 0.6),
    'use_dccc_sbm': True,
    'degree_distribution': 'power_law',
    'power_law_exponent_range': (2.2, 2.2),
    'max_mean_community_deviation': 0.05,
}

# Metrics to calculate
SIGNAL_METRICS = ['feature_signal', 'degree_signal', 'triangle_signal', 'structure_signal']
CONSISTENCY_METRICS = ['pattern_preservation', 'generation_fidelity', 'degree_consistency', 'cooccurrence_consistency']
PROPERTY_METRICS = ['graph_generation_times', 'homophily_levels', 'avg_degrees', 'tail_ratio_95']


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


def generate_random_params_for_analysis(fixed_param, fixed_value, all_params, seed=None):
    """
    Generate random parameters for all non-fixed parameters.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = {
        'universe': {},
        'family': {}
    }
    
    # Set the fixed parameter
    param_config = all_params[fixed_param]
    if param_config['level'] == 'universe':
        params['universe'][fixed_param] = fixed_value
    else:
        params['family'][fixed_param] = fixed_value
    
    # Handle paired parameters
    if 'paired_with' in param_config:
        paired_param = param_config['paired_with']
        if param_config['level'] == 'universe':
            params['universe'][paired_param] = fixed_value + 50  # Ensure max > min
        else:
            params['family'][paired_param] = fixed_value + 50
    
    # Randomize all other parameters
    for param_name, config in all_params.items():
        if param_name == fixed_param or param_name == param_config.get('paired_with'):
            continue
        
        if config['type'] == 'continuous':
            value = np.random.uniform(config['random_range'][0], config['random_range'][1])
        elif config['type'] == 'discrete':
            if isinstance(config['random_range'], tuple):
                value = np.random.randint(config['random_range'][0], config['random_range'][1])
            else:
                value = np.random.choice(config['random_range'])
        elif config['type'] == 'categorical' or config['type'] == 'boolean':
            value = np.random.choice(config['random_range'])
        elif config['type'] == 'range':
            # Generate a random range
            min_val = np.random.uniform(config['random_range'][0], config['random_range'][1])
            range_width = (config['random_range'][1] - config['random_range'][0]) * 0.2
            max_val = min(min_val + np.random.uniform(0.1, range_width), config['random_range'][1])
            value = (min_val, max_val)
        
        if config['level'] == 'universe':
            params['universe'][param_name] = value
        else:
            params['family'][param_name] = value
    
    # Ensure constraints are met
    if 'min_n_nodes' in params['family'] and 'max_n_nodes' in params['family']:
        if params['family']['max_n_nodes'] <= params['family']['min_n_nodes']:
            params['family']['max_n_nodes'] = params['family']['min_n_nodes'] + 100
    
    if 'min_communities' in params['family'] and 'max_communities' in params['family']:
        if params['family']['max_communities'] <= params['family']['min_communities']:
            params['family']['max_communities'] = min(params['family']['min_communities'] + 4, UNIVERSE_K)
    
    return params


def run_parameter_analysis(params_to_test=None, output_dir='parameter_analysis_results'):
    """
    Run parameter sensitivity analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if params_to_test is None:
        params_to_test = list(ALL_VARIABLE_PARAMS.keys())
    
    all_results = {}
    
    for param_name in params_to_test:
        if param_name not in ALL_VARIABLE_PARAMS:
            print(f"Warning: {param_name} not in parameter configuration. Skipping.")
            continue
        
        print(f"\nAnalyzing parameter: {param_name}")
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        param_results = {
            'test_values': param_config['test_values'],
            'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
            'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
            'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
            'metadata': []
        }
        
        for value_idx, test_value in enumerate(param_config['test_values']):
            print(f"  Testing value: {test_value}")
            
            value_signals = {metric: [] for metric in SIGNAL_METRICS}
            value_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
            value_properties = {metric: [] for metric in PROPERTY_METRICS}
            value_metadata = []
            
            for repeat in range(REPEATS_PER_VALUE):
                print(f"    Repeat {repeat + 1}/{REPEATS_PER_VALUE}")
                
                # Generate random parameters
                seed = abs(hash(param_name) + value_idx * 1000 + repeat) % (2**32)
                random_params = generate_random_params_for_analysis(
                    param_name, test_value, ALL_VARIABLE_PARAMS, seed=seed
                )
                
                try:
                    # Create universe
                    universe_params = random_params['universe']
                    universe = GraphUniverse(
                        K=UNIVERSE_K,
                        edge_probability_variance=universe_params.get('edge_probability_variance', 0.5),
                        feature_dim=universe_params.get('feature_dim', 50),
                        center_variance=universe_params.get('center_variance', 0.5),
                        cluster_variance=universe_params.get('cluster_variance', 0.1),
                        degree_center_method=universe_params.get('degree_center_method', 'linear'),
                        community_cooccurrence_homogeneity=universe_params.get('community_cooccurrence_homogeneity', 1.0),
                        seed=seed
                    )
                    
                    # Create family generator
                    family_params = random_params['family']
                    generator = GraphFamilyGenerator(
                        universe=universe,
                        min_n_nodes=family_params.get('min_n_nodes', 50),
                        max_n_nodes=family_params.get('max_n_nodes', 200),
                        min_communities=family_params.get('min_communities', 2),
                        max_communities=family_params.get('max_communities', 8),
                        homophily_range=family_params.get('homophily_range', (0.0, 1.0)),
                        avg_degree_range=family_params.get('avg_degree_range', (2.0, 10.0)),
                        use_dccc_sbm=family_params.get('use_dccc_sbm', True),
                        degree_heterogeneity=family_params.get('degree_heterogeneity', 0.5),
                        degree_separation_range=family_params.get('degree_separation_range', (0.5, 0.5)),
                        degree_distribution=family_params.get('degree_distribution', 'power_law'),
                        power_law_exponent_range=family_params.get('power_law_exponent_range', (2.0, 3.5)),
                        max_mean_community_deviation=family_params.get('max_mean_community_deviation', 0.10),
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
                                # Store the original family values, not just the mean
                                value_signals[metric].append(metric_values)
                    
                    # Calculate consistency
                    consistency = generator.analyze_graph_family_consistency()
                    for metric in CONSISTENCY_METRICS:
                        if metric in consistency:
                            if isinstance(consistency[metric], list):
                                if consistency[metric]:
                                    # Store the original family values, not just the mean
                                    value_consistency[metric].append(consistency[metric])
                            else:
                                # For single values like cooccurrence_consistency, store as list
                                value_consistency[metric].append([consistency[metric]])
                    
                    # Calculate properties
                    properties = generator.analyze_graph_family_properties()
                    for metric in PROPERTY_METRICS:
                        if metric in properties:
                            # Store the original family values
                            value_properties[metric].append(properties[metric])
                    
                    value_metadata.append({
                        'repeat': repeat,
                        'all_params': random_params,
                        'n_graphs': len(graphs)
                    })
                    
                except Exception as e:
                    print(f"      Error in repeat {repeat}: {e}")
                    value_metadata.append({
                        'repeat': repeat,
                        'error': str(e)
                    })
            
            # Store values for this test value
            for metric in SIGNAL_METRICS:
                if value_signals[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in value_signals[metric]:
                        all_family_values.extend(family_values)
                    
                    param_results['signal_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    param_results['signal_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            for metric in CONSISTENCY_METRICS:
                if value_consistency[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in value_consistency[metric]:
                        all_family_values.extend(family_values)
                    
                    param_results['consistency_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    param_results['consistency_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            # Store property results
            for metric in PROPERTY_METRICS:
                if value_properties[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in value_properties[metric]:
                        all_family_values.extend(family_values)
                    
                    param_results['property_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    param_results['property_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            param_results['metadata'].append(value_metadata)
        
        all_results[param_name] = param_results
        
        # Save intermediate results
        with open(os.path.join(output_dir, f'{param_name}_analysis.pkl'), 'wb') as f:
            pickle.dump(param_results, f)
    
    # Save all results
    with open(os.path.join(output_dir, 'all_parameter_analysis.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results


def run_combined_analysis(params_to_test=None, output_dir='parameter_analysis_results', 
                         n_random_repeats=10, n_baseline_repeats=3):
    """
    Run both baseline and randomized analyses.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if params_to_test is None:
        params_to_test = list(ALL_VARIABLE_PARAMS.keys())
    
    baseline_results = {}
    randomized_results = {}
    
    for param_name in params_to_test:
        if param_name not in ALL_VARIABLE_PARAMS:
            print(f"Warning: {param_name} not in parameter configuration. Skipping.")
            continue
        
        print(f"\nAnalyzing parameter: {param_name}")
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Initialize result structures
        baseline_param_results = {
            'test_values': param_config['test_values'],
            'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
            'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
            'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
            'metadata': []
        }
        
        randomized_param_results = {
            'test_values': param_config['test_values'],
            'signal_metrics': {metric: [] for metric in SIGNAL_METRICS},
            'consistency_metrics': {metric: [] for metric in CONSISTENCY_METRICS},
            'property_metrics': {metric: [] for metric in PROPERTY_METRICS},
            'metadata': []
        }
        
        for value_idx, test_value in enumerate(param_config['test_values']):
            print(f"  Testing value: {test_value}")
            
            # BASELINE ANALYSIS
            print("    Running baseline analysis...")
            baseline_signals = {metric: [] for metric in SIGNAL_METRICS}
            baseline_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
            baseline_properties = {metric: [] for metric in PROPERTY_METRICS}
            baseline_metadata = []
            
            for repeat in range(n_baseline_repeats):
                seed = abs(hash(param_name + "_baseline") + value_idx * 1000 + repeat) % (2**32)
                baseline_params = generate_baseline_params(param_name, test_value)
                
                try:
                    # Create universe with baseline params
                    universe_params = baseline_params['universe']
                    universe = GraphUniverse(
                        K=UNIVERSE_K,
                        edge_probability_variance=universe_params.get('edge_probability_variance', 0.5),
                        feature_dim=universe_params.get('feature_dim', 50),
                        center_variance=universe_params.get('center_variance', 0.5),
                        cluster_variance=universe_params.get('cluster_variance', 0.1),
                        degree_center_method=universe_params.get('degree_center_method', 'linear'),
                        community_cooccurrence_homogeneity=universe_params.get('community_cooccurrence_homogeneity', 1.0),
                        seed=seed
                    )
                    
                    # Create family generator with baseline params
                    family_params = baseline_params['family']
                    generator = GraphFamilyGenerator(
                        universe=universe,
                        min_n_nodes=family_params.get('min_n_nodes', 50),
                        max_n_nodes=family_params.get('max_n_nodes', 200),
                        min_communities=family_params.get('min_communities', 2),
                        max_communities=family_params.get('max_communities', 8),
                        homophily_range=family_params.get('homophily_range', (0.0, 1.0)),
                        avg_degree_range=family_params.get('avg_degree_range', (2.0, 10.0)),
                        use_dccc_sbm=family_params.get('use_dccc_sbm', True),
                        degree_heterogeneity=family_params.get('degree_heterogeneity', 0.5),
                        degree_separation_range=family_params.get('degree_separation_range', (0.5, 0.5)),
                        degree_distribution=family_params.get('degree_distribution', 'power_law'),
                        power_law_exponent_range=family_params.get('power_law_exponent_range', (2.0, 3.5)),
                        max_mean_community_deviation=family_params.get('max_mean_community_deviation', 0.10),
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
            
            # RANDOMIZED ANALYSIS
            print("    Running randomized analysis...")
            random_signals = {metric: [] for metric in SIGNAL_METRICS}
            random_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
            random_properties = {metric: [] for metric in PROPERTY_METRICS}
            random_metadata = []
            
            for repeat in range(n_random_repeats):
                seed = abs(hash(param_name + "_random") + value_idx * 1000 + repeat) % (2**32)
                random_params = generate_random_params_for_analysis(
                    param_name, test_value, ALL_VARIABLE_PARAMS, seed=seed
                )
                
                try:
                    # Create universe with random params
                    universe_params = random_params['universe']
                    universe = GraphUniverse(
                        K=UNIVERSE_K,
                        edge_probability_variance=universe_params.get('edge_probability_variance', 0.5),
                        feature_dim=universe_params.get('feature_dim', 50),
                        center_variance=universe_params.get('center_variance', 0.5),
                        cluster_variance=universe_params.get('cluster_variance', 0.1),
                        degree_center_method=universe_params.get('degree_center_method', 'linear'),
                        community_cooccurrence_homogeneity=universe_params.get('community_cooccurrence_homogeneity', 1.0),
                        seed=seed
                    )
                    
                    # Create family generator with random params
                    family_params = random_params['family']
                    generator = GraphFamilyGenerator(
                        universe=universe,
                        min_n_nodes=family_params.get('min_n_nodes', 50),
                        max_n_nodes=family_params.get('max_n_nodes', 200),
                        min_communities=family_params.get('min_communities', 2),
                        max_communities=family_params.get('max_communities', 8),
                        homophily_range=family_params.get('homophily_range', (0.0, 1.0)),
                        avg_degree_range=family_params.get('avg_degree_range', (2.0, 10.0)),
                        use_dccc_sbm=family_params.get('use_dccc_sbm', True),
                        degree_heterogeneity=family_params.get('degree_heterogeneity', 0.5),
                        degree_separation_range=family_params.get('degree_separation_range', (0.5, 0.5)),
                        degree_distribution=family_params.get('degree_distribution', 'power_law'),
                        power_law_exponent_range=family_params.get('power_law_exponent_range', (2.0, 3.5)),
                        max_mean_community_deviation=family_params.get('max_mean_community_deviation', 0.10),
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
                                random_signals[metric].append(metric_values)
                    
                    # Calculate consistency
                    consistency = generator.analyze_graph_family_consistency()
                    for metric in CONSISTENCY_METRICS:
                        if metric in consistency:
                            if isinstance(consistency[metric], list):
                                if consistency[metric]:
                                    random_consistency[metric].append(consistency[metric])
                            else:
                                random_consistency[metric].append([consistency[metric]])
                    
                    # Calculate properties
                    properties = generator.analyze_graph_family_properties()
                    for metric in PROPERTY_METRICS:
                        if metric in properties:
                            random_properties[metric].append(properties[metric])
                    
                    random_metadata.append({
                        'repeat': repeat,
                        'all_params': random_params,
                        'n_graphs': len(graphs)
                    })
                    
                except Exception as e:
                    print(f"      Error in random repeat {repeat}: {e}")
                    random_metadata.append({
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
            
            # Store randomized results
            for metric in SIGNAL_METRICS:
                if random_signals[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in random_signals[metric]:
                        all_family_values.extend(family_values)
                    
                    randomized_param_results['signal_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    randomized_param_results['signal_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            for metric in CONSISTENCY_METRICS:
                if random_consistency[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in random_consistency[metric]:
                        all_family_values.extend(family_values)
                    
                    randomized_param_results['consistency_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    randomized_param_results['consistency_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            # Store randomized property results
            for metric in PROPERTY_METRICS:
                if random_properties[metric]:
                    # Flatten all family values across repeats
                    all_family_values = []
                    for family_values in random_properties[metric]:
                        all_family_values.extend(family_values)
                    
                    randomized_param_results['property_metrics'][metric].append({
                        'mean': np.mean(all_family_values),
                        'std': np.std(all_family_values) if len(all_family_values) > 1 else 0,
                        'values': all_family_values
                    })
                else:
                    randomized_param_results['property_metrics'][metric].append({
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    })
            
            randomized_param_results['metadata'].append(random_metadata)
        
        baseline_results[param_name] = baseline_param_results
        randomized_results[param_name] = randomized_param_results
        
        # Save intermediate results
        with open(os.path.join(output_dir, f'{param_name}_baseline_analysis.pkl'), 'wb') as f:
            pickle.dump(baseline_param_results, f)
        with open(os.path.join(output_dir, f'{param_name}_randomized_analysis.pkl'), 'wb') as f:
            pickle.dump(randomized_param_results, f)
    
    # Save all results
    with open(os.path.join(output_dir, 'all_baseline_analysis.pkl'), 'wb') as f:
        pickle.dump(baseline_results, f)
    with open(os.path.join(output_dir, 'all_randomized_analysis.pkl'), 'wb') as f:
        pickle.dump(randomized_results, f)
    
    return baseline_results, randomized_results


def calculate_r2_with_uncertainty(x, y):
    """
    Calculate R² with bootstrap confidence intervals.
    """
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, (0.0, 0.0)
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return 0.0, (0.0, 0.0)
    
    # Calculate base R²
    try:
        r2 = r2_score(y, np.poly1d(np.polyfit(x, y, 1))(x))
    except:
        return 0.0, (0.0, 0.0)
    
    # Bootstrap for confidence intervals
    n_bootstrap = 100
    r2_bootstrap = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        try:
            r2_boot = r2_score(y_boot, np.poly1d(np.polyfit(x_boot, y_boot, 1))(x_boot))
            r2_bootstrap.append(r2_boot)
        except:
            continue
    
    if r2_bootstrap:
        ci_lower = np.percentile(r2_bootstrap, 2.5)
        ci_upper = np.percentile(r2_bootstrap, 97.5)
    else:
        ci_lower, ci_upper = r2, r2
    
    return r2, (ci_lower, ci_upper)


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


def assess_correlation_significance(r2, ci_lower, ci_upper):
    """
    Assess whether a correlation is significant based on R² confidence interval.
    
    Args:
        r2: R² value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
    
    Returns:
        str: Significance level ('***' for p<0.001, '**' for p<0.01, '*' for p<0.05, 'ns' for not significant)
    """
    # If confidence interval includes 0, correlation is not significant
    if ci_lower <= 0 <= ci_upper:
        return 'ns'
    
    # Determine significance based on R² value and confidence interval
    if r2 > 0.5 and ci_lower > 0.3:
        return '***'  # Very strong correlation
    elif r2 > 0.3 and ci_lower > 0.1:
        return '**'   # Strong correlation
    elif r2 > 0.1 and ci_lower > 0:
        return '*'    # Moderate correlation
    else:
        return 'ns'   # Not significant


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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        # Get test values and prepare x-axis
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Convert categorical/boolean to numeric for plotting
        if param_config['type'] in ['categorical', 'boolean']:
            x_values = list(range(len(test_values)))
            x_labels = [str(v) for v in test_values]
        elif param_config['type'] == 'range':
            x_values = [(v[0] + v[1]) / 2 for v in test_values]  # Use midpoint
            x_labels = [f"{v[0]:.2f}-{v[1]:.2f}" for v in test_values]
        else:
            x_values = test_values
            x_labels = None
        
        plot_idx = 0
        
        # Plot signal metrics
        for metric in SIGNAL_METRICS:
            ax = axes[plot_idx]
            metric_data = param_results['signal_metrics'][metric]
            
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
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                # Format display text
                if significance == 'ns':
                    display_text = 'NS'
                else:
                    if correlation_value < 0.01:
                        formatted_value = '<0.01'
                    else:
                        formatted_value = f'{correlation_value:.3f}'
                    
                    if direction == 'positive':
                        display_text = f'{formatted_value}+'
                    elif direction == 'negative':
                        display_text = f'{formatted_value}-'
                    else:
                        display_text = formatted_value
                
                ax.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(param_name.replace("_", " ").title())
            ax.set_ylabel('Signal Value')
            ax.set_ylim(0, 1)  # Fix y-axis limits for signal metrics
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Plot consistency metrics
        for metric in CONSISTENCY_METRICS:
            ax = axes[plot_idx]
            metric_data = param_results['consistency_metrics'][metric]
            
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
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                # Format display text
                if significance == 'ns':
                    display_text = 'NS'
                else:
                    if correlation_value < 0.01:
                        formatted_value = '<0.01'
                    else:
                        formatted_value = f'{correlation_value:.3f}'
                    
                    if direction == 'positive':
                        display_text = f'{formatted_value}+'
                    elif direction == 'negative':
                        display_text = f'{formatted_value}-'
                    else:
                        display_text = formatted_value
                
                ax.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(param_name.replace("_", " ").title())
            ax.set_ylabel('Consistency Value')
            ax.set_ylim(0, 1)  # Fix y-axis limits for consistency metrics
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Plot property metrics
        for metric in PROPERTY_METRICS:
            ax = axes[plot_idx]
            metric_data = param_results['property_metrics'][metric]
            
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
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                # Format display text
                if significance == 'ns':
                    display_text = 'NS'
                else:
                    if correlation_value < 0.01:
                        formatted_value = '<0.01'
                    else:
                        formatted_value = f'{correlation_value:.3f}'
                    
                    if direction == 'positive':
                        display_text = f'{formatted_value}+'
                    elif direction == 'negative':
                        display_text = f'{formatted_value}-'
                    else:
                        display_text = formatted_value
                
                ax.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
            
            title = f'{metric.replace("_", " ").title()}'
            if 'Time' in title: 
                title += ' (s)'
            ax.set_title(title)
            ax.set_xlabel(param_name.replace("_", " ").title())
            ax.set_ylabel('Property Value')
            # Don't fix y-axis limits for property metrics as they can vary widely
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Remove empty subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Parameter Effects: {param_name.replace("_", " ").title()}', 
                    fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{param_name}_effects.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_heatmap(results_dict, save_path='parameter_sensitivity_heatmap.png'):
    """
    Create a heatmap showing correlation values with significance and direction for all parameter-metric combinations.
    """
    params = list(results_dict.keys())
    metrics = SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS
    
    # Create correlation matrix and display matrix
    correlation_matrix = np.zeros((len(params), len(metrics)))
    display_matrix = np.empty((len(params), len(metrics)), dtype=object)
    
    for i, param_name in enumerate(params):
        param_results = results_dict[param_name]
        test_values = param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Get x values
        if param_config['type'] in ['categorical', 'boolean']:
            x_values = list(range(len(test_values)))
        elif param_config['type'] == 'range':
            x_values = [(v[0] + v[1]) / 2 for v in test_values]
        else:
            x_values = test_values
        
        # Calculate correlation for each metric
        for j, metric in enumerate(metrics):
            if metric in SIGNAL_METRICS:
                metric_data = param_results['signal_metrics'][metric]
            elif metric in CONSISTENCY_METRICS:
                metric_data = param_results['consistency_metrics'][metric]
            else:  # PROPERTY_METRICS
                metric_data = param_results['property_metrics'][metric]
            
            means = [d['mean'] for d in metric_data]
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2 and np.std(valid_x) > 0:
                # Use new correlation function with significance and direction
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                correlation_matrix[i, j] = correlation_value
                display_matrix[i, j] = format_correlation_display(correlation_value, direction, significance)
            else:
                correlation_matrix[i, j] = 0.0
                display_matrix[i, j] = 'NS'
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, len(params) * 0.6 + 2))
    
    # Create custom colormap that shows significance levels
    # Use different colors for significant vs non-significant
    significant_mask = np.zeros_like(correlation_matrix, dtype=bool)
    for i in range(len(params)):
        for j in range(len(metrics)):
            if display_matrix[i, j] != 'NS':
                significant_mask[i, j] = True
    
    # Create the heatmap with two different colormaps
    im = ax.imshow(correlation_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Value (R²/η²)', rotation=270, labelpad=15)
    
    # Add text annotations with significance and direction
    for i in range(len(params)):
        for j in range(len(metrics)):
            display_text = display_matrix[i, j]
            correlation_val = correlation_matrix[i, j]
            
            # Choose text color based on background
            if correlation_val < 0.3:
                text_color = "black"
            else:
                text_color = "white"
            
            # Add text with significance indicators
            if display_text == 'NS':
                text = ax.text(j, i, 'NS',
                             ha="center", va="center", color="gray", 
                             fontweight='normal', fontsize=9)
            else:
                # Add significance stars if present
                if '+' in display_text:
                    base_text = display_text.replace('+', '')
                    text = ax.text(j, i, f'{base_text}+',
                                 ha="center", va="center", color=text_color,
                                 fontweight='bold', fontsize=9)
                elif '-' in display_text:
                    base_text = display_text.replace('-', '')
                    text = ax.text(j, i, f'{base_text}-',
                                 ha="center", va="center", color=text_color,
                                 fontweight='bold', fontsize=9)
                else:
                    text = ax.text(j, i, display_text,
                                 ha="center", va="center", color=text_color,
                                 fontweight='bold', fontsize=9)
    
    plt.title('Parameter Sensitivity Analysis: Correlation Values with Significance and Direction', 
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_results(baseline_results, randomized_results, save_dir='combined_plots'):
    """
    Create side-by-side plots comparing baseline and randomized results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for param_name in baseline_results.keys():
        if param_name not in randomized_results:
            continue
            
        print(f"Creating combined plots for {param_name}")
        
        # Create figure with two columns: baseline and randomized
        n_metrics = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + len(PROPERTY_METRICS)
        fig, axes = plt.subplots(n_metrics, 2, figsize=(16, 4 * n_metrics))
        
        baseline_param_results = baseline_results[param_name]
        randomized_param_results = randomized_results[param_name]
        
        # Get test values and prepare x-axis
        test_values = baseline_param_results['test_values']
        param_config = ALL_VARIABLE_PARAMS[param_name]
        
        # Convert categorical/boolean to numeric for plotting
        if param_config['type'] in ['categorical', 'boolean']:
            x_values = list(range(len(test_values)))
            x_labels = [str(v) for v in test_values]
        elif param_config['type'] == 'range':
            x_values = [(v[0] + v[1]) / 2 for v in test_values]  # Use midpoint
            x_labels = [f"{v[0]:.2f}-{v[1]:.2f}" for v in test_values]
        else:
            x_values = test_values
            x_labels = None
        
        # Plot signal metrics
        for i, metric in enumerate(SIGNAL_METRICS):
            # Baseline plot (left column)
            ax_baseline = axes[i, 0] if n_metrics > 1 else axes[0]
            baseline_metric_data = baseline_param_results['signal_metrics'][metric]
            
            # Calculate confidence intervals for baseline
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in baseline_metric_data:
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
            
            # Plot confidence intervals as shaded areas for baseline
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_baseline.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                       alpha=0.3, color='blue', label='95% CI')
                
                # Plot mean line
                ax_baseline.plot(valid_x, valid_means, 'o-', markersize=8, linewidth=2, 
                               color='blue', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(baseline_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_baseline.scatter([x_values[j]] * len(all_graph_values), 
                                         np.array(all_graph_values) + y_jitter, 
                                         alpha=0.3, s=30, color='lightblue')
            
            # Calculate and display correlation with significance and direction for baseline
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            # Check if we have sufficient data and variation
            if len(valid_means) > 2 and len(set(valid_means)) > 1 and len(set(valid_x)) > 1:
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                # Format display text
                if significance == 'ns':
                    display_text = 'NS'
                else:
                    if correlation_value < 0.01:
                        formatted_value = '<0.01'
                    else:
                        formatted_value = f'{correlation_value:.3f}'
                    
                    if direction == 'positive':
                        display_text = f'{formatted_value}+'
                    elif direction == 'negative':
                        display_text = f'{formatted_value}-'
                    else:
                        display_text = formatted_value
                
                ax_baseline.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                               transform=ax_baseline.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            ax_baseline.set_title(f'{metric.replace("_", " ").title()} (Baseline)')
            ax_baseline.set_ylabel('Signal Value')
            ax_baseline.set_ylim(0, 1)  # Fix y-axis limits for signal metrics
            ax_baseline.grid(True, alpha=0.3)
            ax_baseline.legend()
            
            if x_labels:
                ax_baseline.set_xticks(x_values)
                ax_baseline.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Randomized plot (right column)
            ax_random = axes[i, 1] if n_metrics > 1 else axes[1]
            random_metric_data = randomized_param_results['signal_metrics'][metric]
            
            # Calculate confidence intervals for randomized
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in random_metric_data:
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
            
            # Plot confidence intervals as shaded areas for randomized
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_random.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                     alpha=0.3, color='red', label='95% CI')
                
                # Plot mean line
                ax_random.plot(valid_x, valid_means, 's-', markersize=8, linewidth=2, 
                             color='red', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(random_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_random.scatter([x_values[j]] * len(all_graph_values), 
                                       np.array(all_graph_values) + y_jitter, 
                                       alpha=0.3, s=30, color='pink')
            
            # Calculate and display correlation with significance and direction for randomized
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            # Check if we have sufficient data and variation
            if len(valid_means) > 2 and len(set(valid_means)) > 1 and len(set(valid_x)) > 1:
                correlation_value, direction, significance, ci_lower, ci_upper = calculate_correlation_with_significance(
                    np.array(valid_x), np.array(valid_means), param_config['type']
                )
                
                # Format display text
                if significance == 'ns':
                    display_text = 'NS'
                else:
                    if correlation_value < 0.01:
                        formatted_value = '<0.01'
                    else:
                        formatted_value = f'{correlation_value:.3f}'
                    
                    if direction == 'positive':
                        display_text = f'{formatted_value}+'
                    elif direction == 'negative':
                        display_text = f'{formatted_value}-'
                    else:
                        display_text = formatted_value
                
                ax_random.text(0.05, 0.95, f'{display_text} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                             transform=ax_random.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))
            
            ax_random.set_title(f'{metric.replace("_", " ").title()} (Randomized)')
            ax_random.set_ylabel('Signal Value')
            ax_random.set_ylim(0, 1)  # Fix y-axis limits for signal metrics
            ax_random.grid(True, alpha=0.3)
            ax_random.legend()
            
            if x_labels:
                ax_random.set_xticks(x_values)
                ax_random.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Plot consistency metrics
        for i, metric in enumerate(CONSISTENCY_METRICS):
            row_idx = len(SIGNAL_METRICS) + i
            
            # Baseline plot (left column)
            ax_baseline = axes[row_idx, 0] if n_metrics > 1 else axes[0]
            baseline_metric_data = baseline_param_results['consistency_metrics'][metric]
            
            # Calculate confidence intervals for baseline
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in baseline_metric_data:
                if data_point['values']:
                    # Flatten the nested list to get all individual graph values
                    all_graph_values = []
                    for family_values in data_point['values']:
                        if isinstance(family_values, list):
                            all_graph_values.extend(family_values)
                        else:
                            all_graph_values.append(family_values)
                    
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
            
            # Plot confidence intervals as shaded areas for baseline
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_baseline.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                       alpha=0.3, color='darkgreen', label='95% CI')
                
                # Plot mean line
                ax_baseline.plot(valid_x, valid_means, 'o-', markersize=8, linewidth=2, 
                               color='darkgreen', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(baseline_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_baseline.scatter([x_values[j]] * len(all_graph_values), 
                                         np.array(all_graph_values) + y_jitter, 
                                         alpha=0.3, s=30, color='lightgreen')
            
            # Calculate and display R² with significance for baseline
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                r2, (ci_lower, ci_upper) = calculate_r2_with_uncertainty(
                    np.array(valid_x), np.array(valid_means)
                )
                significance = assess_correlation_significance(r2, ci_lower, ci_upper)
                ax_baseline.text(0.05, 0.95, f'R² = {r2:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                               transform=ax_baseline.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            ax_baseline.set_title(f'{metric.replace("_", " ").title()} (Baseline)')
            ax_baseline.set_ylabel('Consistency Value')
            ax_baseline.set_ylim(0, 1)  # Fix y-axis limits for consistency metrics
            ax_baseline.grid(True, alpha=0.3)
            ax_baseline.legend()
            
            if x_labels:
                ax_baseline.set_xticks(x_values)
                ax_baseline.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Randomized plot (right column)
            ax_random = axes[row_idx, 1] if n_metrics > 1 else axes[1]
            random_metric_data = randomized_param_results['consistency_metrics'][metric]
            
            # Calculate confidence intervals for randomized
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in random_metric_data:
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
            
            # Plot confidence intervals as shaded areas for randomized
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_random.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                     alpha=0.3, color='darkred', label='95% CI')
                
                # Plot mean line
                ax_random.plot(valid_x, valid_means, 's-', markersize=8, linewidth=2, 
                             color='darkred', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(random_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_random.scatter([x_values[j]] * len(all_graph_values), 
                                       np.array(all_graph_values) + y_jitter, 
                                       alpha=0.3, s=30, color='lightcoral')
            
            # Calculate and display R² with significance for randomized
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                r2, (ci_lower, ci_upper) = calculate_r2_with_uncertainty(
                    np.array(valid_x), np.array(valid_means)
                )
                significance = assess_correlation_significance(r2, ci_lower, ci_upper)
                ax_random.text(0.05, 0.95, f'R² = {r2:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                             transform=ax_random.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            
            ax_random.set_title(f'{metric.replace("_", " ").title()} (Randomized)')
            ax_random.set_ylabel('Consistency Value')
            ax_random.set_ylim(0, 1)  # Fix y-axis limits for consistency metrics
            ax_random.grid(True, alpha=0.3)
            ax_random.legend()
            
            if x_labels:
                ax_random.set_xticks(x_values)
                ax_random.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Plot property metrics
        for i, metric in enumerate(PROPERTY_METRICS):
            row_idx = len(SIGNAL_METRICS) + len(CONSISTENCY_METRICS) + i
            
            # Baseline plot (left column)
            ax_baseline = axes[row_idx, 0] if n_metrics > 1 else axes[0]
            baseline_metric_data = baseline_param_results['property_metrics'][metric]
            
            # Calculate confidence intervals for baseline
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in baseline_metric_data:
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
            
            # Plot confidence intervals as shaded areas for baseline
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_baseline.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                       alpha=0.3, color='orange', label='95% CI')
                
                # Plot mean line
                ax_baseline.plot(valid_x, valid_means, '^-', markersize=8, linewidth=2, 
                               color='orange', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(baseline_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_baseline.scatter([x_values[j]] * len(all_graph_values), 
                                         np.array(all_graph_values) + y_jitter, 
                                         alpha=0.3, s=30, color='gold')
            
            # Calculate and display R² with significance for baseline
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                r2, (ci_lower, ci_upper) = calculate_r2_with_uncertainty(
                    np.array(valid_x), np.array(valid_means)
                )
                significance = assess_correlation_significance(r2, ci_lower, ci_upper)
                ax_baseline.text(0.05, 0.95, f'R² = {r2:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                               transform=ax_baseline.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
            
            ax_baseline.set_title(f'{metric.replace("_", " ").title()} (Baseline)')
            ax_baseline.set_ylabel('Property Value')
            # Don't fix y-axis limits for property metrics as they can vary widely
            ax_baseline.grid(True, alpha=0.3)
            ax_baseline.legend()
            
            if x_labels:
                ax_baseline.set_xticks(x_values)
                ax_baseline.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Randomized plot (right column)
            ax_random = axes[row_idx, 1] if n_metrics > 1 else axes[1]
            random_metric_data = randomized_param_results['property_metrics'][metric]
            
            # Calculate confidence intervals for randomized
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for data_point in random_metric_data:
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
            
            # Plot confidence intervals as shaded areas for randomized
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if valid_indices:
                valid_x = [x_values[i] for i in valid_indices]
                valid_means = [means[i] for i in valid_indices]
                valid_ci_lower = [ci_lowers[i] for i in valid_indices]
                valid_ci_upper = [ci_uppers[i] for i in valid_indices]
                
                # Plot shaded confidence intervals
                ax_random.fill_between(valid_x, valid_ci_lower, valid_ci_upper, 
                                     alpha=0.3, color='darkorange', label='95% CI')
                
                # Plot mean line
                ax_random.plot(valid_x, valid_means, 's-', markersize=8, linewidth=2, 
                             color='darkorange', alpha=0.8, label='Mean')
            
            # Add individual points
            for j, data_point in enumerate(random_metric_data):
                if data_point['values']:
                    # The values are already flattened when stored
                    all_graph_values = data_point['values']
                    
                    if all_graph_values:
                        y_jitter = np.random.normal(0, 0.005, len(all_graph_values))
                        ax_random.scatter([x_values[j]] * len(all_graph_values), 
                                       np.array(all_graph_values) + y_jitter, 
                                       alpha=0.3, s=30, color='orange')
            
            # Calculate and display R² with significance for randomized
            valid_means = [m for m in means if not np.isnan(m)]
            valid_x = [x for x, m in zip(x_values, means) if not np.isnan(m)]
            
            if len(valid_means) > 2:
                r2, (ci_lower, ci_upper) = calculate_r2_with_uncertainty(
                    np.array(valid_x), np.array(valid_means)
                )
                significance = assess_correlation_significance(r2, ci_lower, ci_upper)
                ax_random.text(0.05, 0.95, f'R² = {r2:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {significance}', 
                             transform=ax_random.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
            
            ax_random.set_title(f'{metric.replace("_", " ").title()} (Randomized)')
            ax_random.set_ylabel('Property Value')
            # Don't fix y-axis limits for property metrics as they can vary widely
            ax_random.grid(True, alpha=0.3)
            ax_random.legend()
            
            if x_labels:
                ax_random.set_xticks(x_values)
                ax_random.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Set common x-axis label
        if n_metrics > 1:
            for i in range(n_metrics):
                axes[i, 0].set_xlabel(param_name.replace("_", " ").title())
                axes[i, 1].set_xlabel(param_name.replace("_", " ").title())
        else:
            axes[0].set_xlabel(param_name.replace("_", " ").title())
            axes[1].set_xlabel(param_name.replace("_", " ").title())
        
        plt.suptitle(f'Parameter Effects Comparison: {param_name.replace("_", " ").title()}', 
                    fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{param_name}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def calculate_correlation_with_significance(x, y, param_type='continuous'):
    """
    Calculate correlation with direction and statistical significance.
    
    Args:
        x: Independent variable values
        y: Dependent variable values  
        param_type: Type of parameter ('continuous', 'categorical', 'boolean', 'range')
    
    Returns:
        tuple: (correlation_value, direction, significance, ci_lower, ci_upper)
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
    
    # For categorical/boolean parameters, use eta-squared (effect size for categorical variables)
    if param_type in ['categorical', 'boolean']:
        return calculate_categorical_correlation(x, y)
    
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
        
        # Calculate R²
        r2 = correlation ** 2
        
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
        
        # Determine significance based on confidence interval
        if ci_lower <= 0 <= ci_upper:
            significance = 'ns'
        elif abs(correlation) > 0.7 and min(abs(ci_lower), abs(ci_upper)) > 0.5:
            significance = '***'
        elif abs(correlation) > 0.5 and min(abs(ci_lower), abs(ci_upper)) > 0.3:
            significance = '**'
        elif abs(correlation) > 0.3 and min(abs(ci_lower), abs(ci_upper)) > 0:
            significance = '*'
        else:
            significance = 'ns'
        
        return r2, direction, significance, ci_lower, ci_upper
        
    except:
        return 0.0, 'none', 'ns', 0.0, 0.0


def calculate_categorical_correlation(x, y):
    """
    Calculate eta-squared (effect size) for categorical variables.
    """
    try:
        # Check for sufficient data
        if len(x) < 3 or len(y) < 3:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # For categorical variables, we calculate eta-squared (proportion of variance explained)
        # This is equivalent to R² for categorical variables
        
        # Group y values by x categories
        unique_x = np.unique(x)
        if len(unique_x) < 2:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Calculate total variance
        total_mean = np.mean(y)
        total_ss = np.sum((y - total_mean) ** 2)
        
        if total_ss == 0:
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Calculate between-group variance
        between_ss = 0
        for x_val in unique_x:
            mask = (x == x_val)
            group_mean = np.mean(y[mask])
            group_size = np.sum(mask)
            between_ss += group_size * (group_mean - total_mean) ** 2
        
        # Calculate eta-squared
        eta_squared = between_ss / total_ss
        
        # Check if eta_squared is nan (shouldn't happen with our checks above, but just in case)
        if np.isnan(eta_squared):
            return 0.0, 'none', 'ns', 0.0, 0.0
        
        # Bootstrap for confidence intervals
        n_bootstrap = 100
        eta_bootstrap = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(x), len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                # Recalculate eta-squared for bootstrap sample
                unique_x_boot = np.unique(x_boot)
                if len(unique_x_boot) < 2:
                    continue
                
                total_mean_boot = np.mean(y_boot)
                total_ss_boot = np.sum((y_boot - total_mean_boot) ** 2)
                
                if total_ss_boot == 0:
                    continue
                
                between_ss_boot = 0
                for x_val in unique_x_boot:
                    mask = (x_boot == x_val)
                    group_mean_boot = np.mean(y_boot[mask])
                    group_size_boot = np.sum(mask)
                    between_ss_boot += group_size_boot * (group_mean_boot - total_mean_boot) ** 2
                
                eta_boot = between_ss_boot / total_ss_boot
                if not np.isnan(eta_boot):
                    eta_bootstrap.append(eta_boot)
            except:
                continue
        
        if eta_bootstrap:
            ci_lower = np.percentile(eta_bootstrap, 2.5)
            ci_upper = np.percentile(eta_bootstrap, 97.5)
        else:
            ci_lower, ci_upper = eta_squared, eta_squared
        
        # For categorical variables, direction is determined by the pattern of means
        # We'll use the trend of means across categories
        means_by_category = []
        for x_val in unique_x:
            mask = (x == x_val)
            means_by_category.append(np.mean(y[mask]))
        
        # Check if there's a clear trend
        if len(means_by_category) >= 2:
            # Calculate trend (positive if means increase with category index)
            trend = np.polyfit(range(len(means_by_category)), means_by_category, 1)[0]
            if trend > 0.01:  # Small threshold to avoid noise
                direction = 'positive'
            elif trend < -0.01:
                direction = 'negative'
            else:
                direction = 'none'
        else:
            direction = 'none'
        
        # Determine significance based on eta-squared and confidence interval
        if ci_lower <= 0.01:  # Small threshold for eta-squared
            significance = 'ns'
        elif eta_squared > 0.25 and ci_lower > 0.15:
            significance = '***'
        elif eta_squared > 0.15 and ci_lower > 0.05:
            significance = '**'
        elif eta_squared > 0.05 and ci_lower > 0:
            significance = '*'
        else:
            significance = 'ns'
        
        return eta_squared, direction, significance, ci_lower, ci_upper
        
    except:
        return 0.0, 'none', 'ns', 0.0, 0.0


def format_correlation_display(correlation_value, direction, significance):
    """
    Format correlation value for display in heatmap.
    
    Args:
        correlation_value: R² or eta-squared value
        direction: 'positive', 'negative', or 'none'
        significance: '***', '**', '*', or 'ns'
    
    Returns:
        str: Formatted string for display
    """
    if significance == 'ns':
        return 'NS'
    
    # Format the correlation value
    if correlation_value < 0.01:
        formatted_value = '<0.01'
    else:
        formatted_value = f'{correlation_value:.2f}'
    
    # Add direction indicator
    if direction == 'positive':
        return f'{formatted_value}+'
    elif direction == 'negative':
        return f'{formatted_value}-'
    else:
        return formatted_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze parameter effects on signals and consistency')
    parser.add_argument('--output-dir', default='parameter_analysis_results',
                       help='Directory to save results')
    parser.add_argument('--params', nargs='+', default=INTERESTING_PARAMS_TO_VARIATE,
                       help='Specific parameters to test (default: all)')
    parser.add_argument('--n-repeats', type=int, default=1,
                       help='Number of repeats per value')
    parser.add_argument('--n-graphs', type=int, default=50,
                       help='Number of graphs per family')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot existing results')
    parser.add_argument('--analysis-type', choices=['randomized', 'baseline', 'combined'], 
                       default='baseline', help='Type of analysis to run')
    parser.add_argument('--n-random-repeats', type=int, default=10,
                       help='Number of randomized repeats per value')
    parser.add_argument('--n-baseline-repeats', type=int, default=1,
                       help='Number of baseline repeats per value')
    
    args = parser.parse_args()
    
    if not args.plot_only:
        # Update global settings
        REPEATS_PER_VALUE = args.n_repeats
        GRAPHS_PER_FAMILY = args.n_graphs
        
        if args.analysis_type == 'combined':
            # Run combined analysis
            baseline_results, randomized_results = run_combined_analysis(
                args.params, args.output_dir, 
                n_random_repeats=args.n_random_repeats,
                n_baseline_repeats=args.n_baseline_repeats
            )
            
            # Create combined plots
            plot_combined_results(baseline_results, randomized_results, 
                               os.path.join(args.output_dir, 'combined_plots'))
            
            # Also create individual plots
            plot_parameter_effects(baseline_results, os.path.join(args.output_dir, 'baseline_plots'))
            plot_parameter_effects(randomized_results, os.path.join(args.output_dir, 'randomized_plots'))
            
            # Create summary heatmaps
            create_summary_heatmap(baseline_results, 
                                 os.path.join(args.output_dir, 'baseline_sensitivity_heatmap.png'))
            create_summary_heatmap(randomized_results, 
                                 os.path.join(args.output_dir, 'randomized_sensitivity_heatmap.png'))
            
        elif args.analysis_type == 'baseline':
            # Run only baseline analysis
            baseline_results, _ = run_combined_analysis(
                args.params, args.output_dir,
                n_random_repeats=0,
                n_baseline_repeats=args.n_baseline_repeats
            )
            plot_parameter_effects(baseline_results, os.path.join(args.output_dir, 'baseline_plots'))
            create_summary_heatmap(baseline_results, 
                                 os.path.join(args.output_dir, 'baseline_sensitivity_heatmap.png'))
            
        elif args.analysis_type == 'randomized':
            # Run only randomized analysis (original method)
            results = run_parameter_analysis(args.params, args.output_dir)
            plot_parameter_effects(results, os.path.join(args.output_dir, 'randomized_plots'))
            create_summary_heatmap(results, 
                                 os.path.join(args.output_dir, 'randomized_sensitivity_heatmap.png'))
    else:
        # Load existing results based on analysis type
        if args.analysis_type == 'combined':
            try:
                with open(os.path.join(args.output_dir, 'all_baseline_analysis.pkl'), 'rb') as f:
                    baseline_results = pickle.load(f)
                with open(os.path.join(args.output_dir, 'all_randomized_analysis.pkl'), 'rb') as f:
                    randomized_results = pickle.load(f)
                
                plot_combined_results(baseline_results, randomized_results, 
                                   os.path.join(args.output_dir, 'combined_plots'))
            except FileNotFoundError:
                print("Combined analysis results not found. Try running the analysis first.")
        elif args.analysis_type == 'baseline':
            try:
                with open(os.path.join(args.output_dir, 'all_baseline_analysis.pkl'), 'rb') as f:
                    results = pickle.load(f)
                plot_parameter_effects(results, os.path.join(args.output_dir, 'baseline_plots'))
                create_summary_heatmap(results, 
                                     os.path.join(args.output_dir, 'baseline_sensitivity_heatmap.png'))
            except FileNotFoundError:
                print("Baseline analysis results not found. Try running the analysis first.")
        elif args.analysis_type == 'randomized':
            try:
                with open(os.path.join(args.output_dir, 'all_randomized_analysis.pkl'), 'rb') as f:
                    results = pickle.load(f)
                plot_parameter_effects(results, os.path.join(args.output_dir, 'randomized_plots'))
                create_summary_heatmap(results, 
                                     os.path.join(args.output_dir, 'randomized_sensitivity_heatmap.png'))
            except FileNotFoundError:
                print("Randomized analysis results not found. Try running the analysis first.")
    
    print("\nAnalysis complete! Check the output directory for results and plots.")