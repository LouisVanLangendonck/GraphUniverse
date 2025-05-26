"""
Script to run multiple inductive learning experiments with parameter sweeps and random sampling.

This script provides a command-line interface for running comprehensive experiment suites
where some parameters are swept systematically and others are randomly sampled.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.multi_config import (
    MultiInductiveExperimentConfig, 
    ParameterRange, 
    create_default_multi_config
)
from experiments.inductive.multi_experiment import run_multi_inductive_experiments, create_analysis_plots
from experiments.inductive.config import InductiveExperimentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple inductive graph learning experiments')
    
    # Configuration source
    parser.add_argument('--config', type=str, default=None,
                        help='Path to multi-experiment configuration file')
    parser.add_argument('--create_default_config', type=str, default=None,
                        help='Create default config file at specified path and exit')
    
    # Experiment setup
    parser.add_argument('--experiment_name', type=str, default='multi_inductive',
                        help='Name for this experiment suite')
    parser.add_argument('--output_dir', type=str, default='multi_inductive_results',
                        help='Base directory to save results')
    parser.add_argument('--n_repetitions', type=int, default=2,
                        help='Number of repetitions for each parameter combination')
    
    # Base experiment parameters
    parser.add_argument('--n_graphs', type=int, default=10,
                        help='Number of graphs in each family')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=120,
                        help='Maximum nodes per graph')
    parser.add_argument('--universe_K', type=int, default=5,
                        help='Number of communities in universe')
    parser.add_argument('--universe_feature_dim', type=int, default=32,
                        help='Feature dimension')
    
    # Sweep parameter definitions
    parser.add_argument('--sweep_homophily', type=float, nargs=3, default=[0.0, 1.0, 0.5],
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='Sweep universe homophily: min max step')
    parser.add_argument('--sweep_density', type=float, nargs=3, default=[0.01, 0.021, 0.02],
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='Sweep universe edge density: min max step')
    parser.add_argument('--no_sweep_homophily', action='store_true',
                        help='Disable homophily sweep')
    parser.add_argument('--no_sweep_density', action='store_true',
                        help='Disable density sweep')
    
    # Graph generation method
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM instead of standard DC-SBM')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution type for DCCC-SBM')
    
    # DCCC-SBM specific parameters
    parser.add_argument('--community_imbalance_range', type=float, nargs=2, default=[0.0, 0.5],
                        help='Range for community size imbalance (DCCC-SBM)')
    parser.add_argument('--degree_separation_range', type=float, nargs=2, default=[0.1, 0.8],
                        help='Range for degree distribution separation (DCCC-SBM)')
    
    # Power law distribution parameters
    parser.add_argument('--power_law_exponent_range', type=float, nargs=2, default=[2.1, 3.5],
                        help='Range for power law exponent')
    parser.add_argument('--power_law_x_min', type=float, default=1.0,
                        help='Minimum value for power law distribution')
    
    # Exponential distribution parameters
    parser.add_argument('--exponential_rate_range', type=float, nargs=2, default=[0.3, 1.0],
                        help='Range for exponential distribution rate parameter')
    
    # Uniform distribution parameters
    parser.add_argument('--uniform_min_factor_range', type=float, nargs=2, default=[0.3, 0.7],
                        help='Range for uniform distribution minimum factor')
    parser.add_argument('--uniform_max_factor_range', type=float, nargs=2, default=[1.3, 2.0],
                        help='Range for uniform distribution maximum factor')
    
    # Random parameter ranges
    parser.add_argument('--random_homophily_range', type=float, nargs=2, default=[0.0, 0.3],
                        help='Random sampling range for homophily_range parameter')
    parser.add_argument('--random_density_range', type=float, nargs=2, default=[0.0, 0.05],
                        help='Random sampling range for density_range parameter')
    parser.add_argument('--random_degree_heterogeneity', type=float, nargs=2, default=[0.1, 1.0],
                        help='Random sampling range for degree_heterogeneity')
    parser.add_argument('--random_edge_noise', type=float, nargs=2, default=[0.0, 0.3],
                        help='Random sampling range for edge_noise')
    parser.add_argument('--random_cluster_count_factor', type=float, nargs=2, default=[0.3, 2.0],
                        help='Random sampling range for cluster_count_factor')
    parser.add_argument('--random_center_variance', type=float, nargs=2, default=[0.05, 1.5],
                        help='Random sampling range for center_variance (log-uniform)')
    parser.add_argument('--random_cluster_variance', type=float, nargs=2, default=[0.05, 0.5],
                        help='Random sampling range for cluster_variance')
    
    # DCCC-SBM presets
    parser.add_argument('--dccc_preset', type=str, default=None,
                        choices=['power_law', 'exponential', 'uniform'],
                        help='Use predefined DCCC-SBM configuration with specified distribution')
    
    # Model and training configuration
    parser.add_argument('--tasks', type=str, nargs='+', default=['community'],
                        choices=['community', 'k_hop_community_counts'],
                        help='Learning tasks to run')
    parser.add_argument('--gnn_types', type=str, nargs='+', default=['gcn', 'sage'],
                        choices=['gcn', 'gat', 'sage'],
                        help='GNN model types to run')
    parser.add_argument('--run_mlp', action='store_true', default=True,
                        help='Run MLP model')
    parser.add_argument('--run_rf', action='store_true', default=True,
                        help='Run Random Forest model')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    
    # Experiment control
    parser.add_argument('--continue_on_failure', action='store_true', default=True,
                        help='Continue running if individual experiments fail')
    parser.add_argument('--consistency_check', action='store_true',
                        help='Enable family consistency checks')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization (slower)')
    
    # Analysis and visualization
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Create analysis plots after experiments')
    parser.add_argument('--analyze_existing', type=str, default=None,
                        help='Skip experiments and analyze existing results at given path')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def create_config_from_args(args) -> MultiInductiveExperimentConfig:
    """Create multi-experiment configuration from command line arguments."""
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Check for DCCC-SBM preset
    if args.dccc_preset:
        from experiments.inductive.multi_config import create_dccc_multi_config
        config = create_dccc_multi_config(args.dccc_preset)
        # Override with any command line arguments
        config.n_repetitions = args.n_repetitions
        config.experiment_name = args.experiment_name
        config.output_dir = args.output_dir
        return config
    
    # Create base configuration
    base_config = InductiveExperimentConfig(
        # Core experiment settings
        n_graphs=args.n_graphs,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=3,
        max_communities=min(6, args.universe_K),
        
        # Universe settings
        universe_K=args.universe_K,
        universe_feature_dim=args.universe_feature_dim,
        universe_edge_density=0.1,  # Will be overridden by sweep
        universe_homophily=0.5,     # Will be overridden by sweep
        universe_randomness_factor=0.0,
        
        # Graph generation method
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,
        
        # DCCC-SBM parameters
        power_law_x_min=args.power_law_x_min,
        
        # Training settings
        epochs=args.epochs,
        patience=args.patience,
        optimize_hyperparams=args.optimize_hyperparams,
        
        # Tasks and models
        tasks=args.tasks,
        gnn_types=args.gnn_types,
        run_gnn=True,
        run_mlp=args.run_mlp,
        run_rf=args.run_rf,
        
        # Experiment control
        require_consistency_check=args.consistency_check,
        collect_family_stats=True,
        
        # Random seed
        seed=args.seed
    )
    
    # Define sweep parameters
    sweep_parameters = {}
    
    if not args.no_sweep_homophily:
        sweep_parameters['universe_homophily'] = ParameterRange(
            min_val=args.sweep_homophily[0],
            max_val=args.sweep_homophily[1],
            step=args.sweep_homophily[2],
            is_sweep=True
        )
    
    if not args.no_sweep_density:
        sweep_parameters['universe_edge_density'] = ParameterRange(
            min_val=args.sweep_density[0],
            max_val=args.sweep_density[1],
            step=args.sweep_density[2],
            is_sweep=True
        )
    
    # Define random parameters
    random_parameters = {
        'homophily_range_0': ParameterRange(  # Will be converted to tuple
            min_val=args.random_homophily_range[0],
            max_val=args.random_homophily_range[1],
            distribution="uniform",
            is_sweep=False
        ),
        'density_range_0': ParameterRange(  # Will be converted to tuple  
            min_val=args.random_density_range[0],
            max_val=args.random_density_range[1],
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=args.random_degree_heterogeneity[0],
            max_val=args.random_degree_heterogeneity[1],
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=args.random_edge_noise[0],
            max_val=args.random_edge_noise[1],
            distribution="uniform",
            is_sweep=False
        ),
        'cluster_count_factor': ParameterRange(
            min_val=args.random_cluster_count_factor[0],
            max_val=args.random_cluster_count_factor[1],
            distribution="uniform",
            is_sweep=False
        ),
        'center_variance': ParameterRange(
            min_val=args.random_center_variance[0],
            max_val=args.random_center_variance[1],
            distribution="log_uniform",
            is_sweep=False
        ),
        'cluster_variance': ParameterRange(
            min_val=args.random_cluster_variance[0],
            max_val=args.random_cluster_variance[1],
            distribution="uniform",
            is_sweep=False
        ),
        'assignment_skewness': ParameterRange(
            min_val=0.0,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'community_exclusivity': ParameterRange(
            min_val=0.5,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'universe_randomness_factor': ParameterRange(
            min_val=0.0,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'max_mean_community_deviation': ParameterRange(
            min_val=0.05,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        'max_max_community_deviation': ParameterRange(
            min_val=0.1,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    # Add DCCC-SBM specific parameters if enabled
    if args.use_dccc_sbm:
        # Community imbalance and degree separation
        random_parameters.update({
            'community_imbalance_range_0': ParameterRange(
                min_val=args.community_imbalance_range[0],
                max_val=args.community_imbalance_range[1],
                distribution="uniform",
                is_sweep=False
            ),
            'degree_separation_range_0': ParameterRange(
                min_val=args.degree_separation_range[0],
                max_val=args.degree_separation_range[1],
                distribution="uniform",
                is_sweep=False
            )
        })
        
        # Distribution-specific parameters
        if args.degree_distribution == "power_law":
            random_parameters.update({
                'power_law_exponent_range_0': ParameterRange(
                    min_val=args.power_law_exponent_range[0],
                    max_val=args.power_law_exponent_range[1],
                    distribution="uniform",
                    is_sweep=False
                )
            })
        
        elif args.degree_distribution == "exponential":
            random_parameters.update({
                'exponential_rate_range_0': ParameterRange(
                    min_val=args.exponential_rate_range[0],
                    max_val=args.exponential_rate_range[1],
                    distribution="uniform",
                    is_sweep=False
                )
            })
        
        elif args.degree_distribution == "uniform":
            random_parameters.update({
                'uniform_min_factor_range_0': ParameterRange(
                    min_val=args.uniform_min_factor_range[0],
                    max_val=args.uniform_min_factor_range[1],
                    distribution="uniform",
                    is_sweep=False
                ),
                'uniform_max_factor_range_0': ParameterRange(
                    min_val=args.uniform_max_factor_range[0],
                    max_val=args.uniform_max_factor_range[1],
                    distribution="uniform",
                    is_sweep=False
                )
            })
    
    # Create multi-experiment configuration
    return MultiInductiveExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=args.n_repetitions,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        continue_on_failure=args.continue_on_failure,
        aggregate_results=True,
        create_summary_plots=args.create_plots
    )


def process_tuple_parameters(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Process parameters that need to be converted to tuples (like ranges)."""
    processed = config_dict.copy()
    
    # Convert range parameters from separate values to tuples
    if 'homophily_range_0' in processed:
        # Create homophily_range tuple
        hr_val = processed.pop('homophily_range_0')
        dr_val = processed.pop('density_range_0', hr_val)  # Use same value if not specified
        processed['homophily_range'] = (0.0, hr_val)  # Range from 0 to sampled value
        processed['density_range'] = (0.0, dr_val)    # Range from 0 to sampled value
    
    # Convert DCCC-SBM range parameters
    if 'community_imbalance_range_0' in processed:
        ci_val = processed.pop('community_imbalance_range_0')
        processed['community_imbalance_range'] = (0.0, ci_val)
    
    if 'degree_separation_range_0' in processed:
        ds_val = processed.pop('degree_separation_range_0')
        processed['degree_separation_range'] = (0.1, ds_val)
    
    # Convert distribution-specific range parameters
    if 'power_law_exponent_range_0' in processed:
        ple_val = processed.pop('power_law_exponent_range_0')
        processed['power_law_exponent_range'] = (2.0, ple_val)
    
    if 'exponential_rate_range_0' in processed:
        er_val = processed.pop('exponential_rate_range_0')
        processed['exponential_rate_range'] = (0.1, er_val)
    
    if 'uniform_min_factor_range_0' in processed:
        umf_val = processed.pop('uniform_min_factor_range_0')
        processed['uniform_min_factor_range'] = (0.1, umf_val)
    
    if 'uniform_max_factor_range_0' in processed:
        umxf_val = processed.pop('uniform_max_factor_range_0')
        processed['uniform_max_factor_range'] = (1.0, umxf_val)
    
    return processed


def main():
    """Main function to run multi-inductive experiments."""
    print("MULTI-INDUCTIVE GRAPH LEARNING EXPERIMENTS")
    print("=" * 60)
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Handle special modes
        if args.create_default_config:
            print(f"Creating default configuration at: {args.create_default_config}")
            if args.dccc_preset:
                from experiments.inductive.multi_config import create_dccc_multi_config
                default_config = create_dccc_multi_config(args.dccc_preset)
            else:
                default_config = create_default_multi_config()
            default_config.save(args.create_default_config)
            print("Default configuration created successfully!")
            return 0
        
        if args.analyze_existing:
            print(f"Analyzing existing results at: {args.analyze_existing}")
            create_analysis_plots(args.analyze_existing)
            print("Analysis complete!")
            return 0
        
        # Create or load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = MultiInductiveExperimentConfig.load(args.config)
        else:
            print("Creating configuration from command line arguments")
            config = create_config_from_args(args)
        
        # Print configuration summary
        print(f"\nExperiment Configuration:")
        print(f"  Experiment name: {config.experiment_name}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Repetitions per combination: {config.n_repetitions}")
        print(f"  Total runs planned: {config.get_total_runs()}")
        print(f"  Base graph family size: {config.base_config.n_graphs}")
        print(f"  Graph generation method: {'DCCC-SBM' if config.base_config.use_dccc_sbm else 'DC-SBM'}")
        
        if config.base_config.use_dccc_sbm:
            print(f"  Degree distribution: {config.base_config.degree_distribution}")
        
        print(f"\nSweep parameters:")
        if config.sweep_parameters:
            for param, param_range in config.sweep_parameters.items():
                values = param_range.get_sweep_values()
                print(f"  {param}: {len(values)} values from {min(values)} to {max(values)}")
        else:
            print("  None")
        
        print(f"\nRandom parameters:")
        for param, param_range in config.random_parameters.items():
            print(f"  {param}: [{param_range.min_val}, {param_range.max_val}] ({param_range.distribution})")
        
        # Confirm before starting
        if config.get_total_runs() > 20:
            response = input(f"\nThis will run {config.get_total_runs()} experiments. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 0
        
        # Run experiments
        print(f"\nStarting multi-experiment suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results = run_multi_inductive_experiments(config)
        
        print(f"\nMulti-experiment suite completed successfully!")
        
        # Print summary
        if 'summary_stats' in results:
            stats = results['summary_stats']
            print(f"\nFinal Summary:")
            print(f"  Total runs attempted: {stats['total_runs_attempted']}")
            print(f"  Successful runs: {stats['successful_runs']}")
            print(f"  Failed runs: {stats['failed_runs']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Total time: {stats['total_time']:.1f}s ({stats['total_time']/3600:.2f}h)")
        
        # Create analysis plots if requested
        if config.create_summary_plots and results['summary_stats']['successful_runs'] > 0:
            print(f"\nCreating analysis plots...")
            output_dir = None
            for result in results['all_results']:
                if 'config_path' in result:
                    output_dir = os.path.dirname(result['config_path'])
                    break
            
            if output_dir and os.path.exists(os.path.join(output_dir, '..', 'results_summary.csv')):
                create_analysis_plots(os.path.join(output_dir, '..'))
                print(f"Analysis plots created!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nMulti-experiment suite interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nMulti-experiment suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)