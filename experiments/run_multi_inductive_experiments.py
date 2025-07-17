"""
Script to run clean multi-experiment sweeps.
Example usage of parameter sweeps with random sampling.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.multi_config import (
    CleanMultiExperimentConfig, 
    ParameterRange,
)
from experiments.inductive.multi_experiment import run_clean_multi_experiments, create_analysis_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run clean multi-experiment parameter sweeps')
    
    # Experiment selection
    parser.add_argument('--preset', type=str, 
                        choices=['custom'],
                        default='custom',
                        help='Preset experiment configuration')
    
    # Custom experiment parameters
    parser.add_argument('--experiment_name', type=str, default='custom_sweep',
                        help='Name for custom experiment')
    parser.add_argument('--output_dir', type=str, default='multi_results',
                        help='Base output directory')
    parser.add_argument('--use_parallel_training', action='store_true', default=False,
                        help='Use parallel training')

    # Continue from intermediate results
    parser.add_argument('--continue_from_intermediate', action='store_true',
                        help='Continue from intermediate results')
    parser.add_argument('--intermediate_result_dir', type=str, default=None,
                        help='Directory containing intermediate results to continue from')
    
    # Task configuration
    parser.add_argument('--tasks', type=str, nargs='+',     
                        default=['community'],
                        choices=['community', 'k_hop_community_counts_k1', 'k_hop_community_counts_k2', 'k_hop_community_counts_k3', 'triangle_count'],
                        help='Learning tasks to run')
    
    # Base experiment settings
    parser.add_argument('--n_graphs', type=int, default=100,
                        help='Number of graphs per family (will be swept over: 10, 30, 50)')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=120,
                        help='Maximum nodes per graph')
    parser.add_argument('--universe_K', type=int, default=15,
                        help='Number of communities in universe')
    parser.add_argument('--min_communities', type=int, default=3,
                        help='Minimum number of communities')
    parser.add_argument('--max_communities', type=int, default=8,
                        help='Maximum number of communities')
    
    # Method selection
    parser.add_argument('--use_dccc_sbm', action='store_true', default=True,
                        help='Use DCCC-SBM (for custom experiments)')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution for DCCC-SBM')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for training (per graph)')
    parser.add_argument('--n_repetitions', type=int, default=3,
                        help='Number of random seed repetitions for statistical robustness')


    # Hyperparameter optimization settings
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--trial_epochs', type=int, default=100,
                        help='Number of epochs for hyperparameter optimization')
    parser.add_argument('--optimization_timeout', type=int, default=600,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Analysis options
    parser.add_argument('--create_plots', action='store_true', default=False,
                        help='Create analysis plots after experiments')
    parser.add_argument('--analyze_existing', type=str, default=None,
                        help='Analyze existing results at given path')
    
    # Model selection
    parser.add_argument('--differentiate_with_and_without_PE', action='store_true', default=False,
                        help='Differentiate between models with and without PE')
    parser.add_argument('--max_pe_dim', type=int, default=8,
                        help='Maximum PE dimension')
    
    parser.add_argument('--gnn_types', type=str, nargs='+', 
                        default=['gcn', 'sage', 'gin', 'gat'],
                        choices=['fagcn', 'gat', 'gcn', 'sage', 'gin'],
                        help='Types of GNN models to run')
    parser.add_argument('--no_gnn', action='store_false', dest='run_gnn',
                        help='Skip GNN models')
    parser.add_argument('--no_mlp', action='store_false', dest='run_mlp',
                        help='Skip MLP model')
    parser.add_argument('--no_rf', action='store_false', dest='run_rf',
                        help='Skip Random Forest model')
    
    # Transformer configuration
    parser.add_argument('--transformer_types', type=str, nargs='+', 
                        default=['graphgps'], 
                        choices=['graphormer', 'graphgps'],
                        help='Types of Graph Transformer models to run')
    parser.add_argument('--run_transformers', action='store_true',
                        help='Run Graph Transformer models')
    parser.add_argument('--transformer_num_heads', type=int, default=8,
                        help='Number of attention heads for transformers')
    parser.add_argument('--transformer_max_nodes', type=int, default=200,
                        help='Maximum nodes for encoding precomputation')
    parser.add_argument('--transformer_max_path_length', type=int, default=10,
                        help='Maximum path length for shortest path encoding')
    parser.add_argument('--transformer_precompute_encodings', action='store_true', default=True,
                        help='Precompute structural encodings for transformers')
    parser.add_argument('--no_precompute_encodings', action='store_false', 
                        dest='transformer_precompute_encodings',
                        help='Disable encoding precomputation')
    parser.add_argument('--transformer_cache_encodings', action='store_true', default=True,
                        help='Cache structural encodings between graphs')
    parser.add_argument('--local_gnn_type', type=str, default='gcn',
                        choices=['gcn', 'sage'],
                        help='Local GNN type for GraphGPS')
    parser.add_argument('--global_model_type', type=str, default='transformer',
                        help='Global model type for GraphGPS')
    parser.add_argument('--transformer_prenorm', action='store_true', default=True,
                        help='Use pre-normalization in transformers')
    
    # Neural Sheaf configuration
    parser.add_argument('--run_neural_sheaf', action='store_true',
                        help='Run Neural Sheaf Diffusion models')
    parser.add_argument('--sheaf_type', type=str, default='diagonal',
                        choices=['diagonal', 'bundle', 'general'],
                        help='Type of sheaf to use')
    parser.add_argument('--sheaf_d', type=int, default=2,
                        help='Sheaf dimension')
    
    # === EVALUATION DISTRIBUTIONAL SHIFT ===
    # Note: These parameters are overridden by the sweep configuration below
    # The sweep creates 7 combinations: no shift + (homophily/density/n_nodes) × (test_only=True/False)
    parser.add_argument('--distributional_shift_in_eval', action='store_true', default=True,
                        help='Use distributional shift in evaluation (overridden by sweep)')
    parser.add_argument('--distributional_shift_in_eval_type', type=str, default='density',
                        choices=['homophily', 'density', 'n_nodes'],
                        help='Type of distributional shift to apply in evaluation (overridden by sweep)')
    parser.add_argument('--distributional_shift_test_only', action='store_true', default=True,
                        help='Use distributional shift in evaluation only for test set (overridden by sweep)')
    
    # Combined sweep parameter for homophily and density ranges
    parser.add_argument('--homophily_density_combinations', type=str, nargs='+',
                        default=['0.1,0.05'],
                        help='Combinations of (homophily_range_max, density_range_max) as comma-separated pairs. Format: "homophily_max,density_max"')
    
    # Fixed parameter values (previously random)
    parser.add_argument('--degree_heterogeneity', type=float, default=1.0,
                        help='Degree heterogeneity parameter')
    parser.add_argument('--edge_noise', type=float, default=0.0,
                        help='Edge noise parameter')
    parser.add_argument('--cluster_count_factor', type=float, default=1.0,
                        help='Cluster count factor parameter')
    parser.add_argument('--cluster_variance', type=float, default=0.5,
                        help='Cluster variance parameter')
    parser.add_argument('--center_variance', type=float, default=0.1,
                        help='Center variance parameter')
    parser.add_argument('--assignment_skewness', type=float, default=0.0,
                        help='Assignment skewness parameter')
    parser.add_argument('--community_exclusivity', type=float, default=1.0,
                        help='Community exclusivity parameter')
    parser.add_argument('--universe_randomness_factor', type=float, default=1.0,
                        help='Universe randomness factor parameter')
    
    # DCCC-specific fixed parameters
    parser.add_argument('--community_imbalance_range_width', type=float, default=0.0,
                        help='Community imbalance range width (DCCC-SBM only)')
    parser.add_argument('--degree_separation_range_width', type=float, default=0.7,
                        help='Degree separation range width (DCCC-SBM only)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def create_custom_experiment(args) -> CleanMultiExperimentConfig:
    """Create a custom experiment configuration from arguments."""
    from experiments.inductive.config import InductiveExperimentConfig
    
    # Base configuration
    base_config = InductiveExperimentConfig(
        n_graphs=args.n_graphs,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        min_communities=args.min_communities,
        max_communities=args.max_communities,
        universe_K=args.universe_K,
        universe_feature_dim=16,
        use_parallel_training=args.use_parallel_training,

        # Method
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,

        # Tasks configuration
        tasks=args.tasks,
        is_regression={},
        is_graph_level_tasks={},
        khop_community_counts_k=1,  # This will be overridden per task
        
        # Tasks and models
        gnn_types=args.gnn_types, # ['fagcn', 'gat', 'gcn', 'sage', 'gin']
        run_gnn=args.run_gnn,
        run_mlp=args.run_mlp,
        run_rf=args.run_rf,
        
        # Transformer configuration
        transformer_types=args.transformer_types,
        run_transformers=args.run_transformers,
        transformer_num_heads=args.transformer_num_heads,
        transformer_max_nodes=args.transformer_max_nodes,
        transformer_max_path_length=args.transformer_max_path_length,
        transformer_precompute_encodings=True,
        transformer_cache_encodings=False,
        local_gnn_type=args.local_gnn_type,
        global_model_type=args.global_model_type,
        transformer_prenorm=getattr(args, 'transformer_prenorm', True),
        max_pe_dim=args.max_pe_dim,
        pe_norm_type=None,
        trial_epochs=args.trial_epochs,

        # Training
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        n_repetitions=args.n_repetitions,
        
        # Hyperparameter optimization
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        
        # Analysis
        collect_signal_metrics=True,
        require_consistency_check=True,

        run_neural_sheaf=args.run_neural_sheaf,
        sheaf_type=args.sheaf_type,
        sheaf_d=args.sheaf_d,
        
        # Evaluation settings
        differentiate_with_and_without_PE=args.differentiate_with_and_without_PE,
        
        # Distributional shift evaluation settings
        distributional_shift_in_eval=args.distributional_shift_in_eval,
        distributional_shift_in_eval_type=args.distributional_shift_in_eval_type,
        distributional_shift_test_only=args.distributional_shift_test_only,

        # Fixed parameters (previously random)
        degree_heterogeneity=args.degree_heterogeneity,
        edge_noise=args.edge_noise,
        cluster_count_factor=args.cluster_count_factor,
        cluster_variance=args.cluster_variance,
        center_variance=args.center_variance,
        assignment_skewness=args.assignment_skewness,
        community_exclusivity=args.community_exclusivity,
        universe_randomness_factor=args.universe_randomness_factor,
        
        # DCCC-specific fixed parameters
        community_imbalance_range=(0.0, args.community_imbalance_range_width),
        degree_separation_range=(0.0, args.degree_separation_range_width),
        
        # Seed
        seed=args.seed
    )

    for task in args.tasks:
        if task == 'community':
            base_config.is_graph_level_tasks['community'] = False
            base_config.is_regression[task] = False
        elif 'k_hop_community_counts_k' in task:
            base_config.is_graph_level_tasks[task] = False
            base_config.is_regression[task] = True
        elif task == 'triangle_count':
            base_config.is_graph_level_tasks[task] = True
            base_config.is_regression[task] = True
        else:
            raise ValueError(f"Invalid task: {task}")
    
    # Define sweep parameters
    sweep_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.5,
            max_val=0.5,
            step=0.3,
            is_sweep=True
        ),
        'universe_edge_density': ParameterRange(
            min_val=0.10,
            max_val=0.10,
            step=0.20,
            is_sweep=True
        ),
        'n_graphs': ParameterRange(
            min_val=100,
            max_val=100,
            step=1,
            is_sweep=True
        ),
        # Combined homophily and density range parameters
        'homophily_density_combination': ParameterRange(
            min_val=0,
            max_val=len(args.homophily_density_combinations) - 1,
            step=1,
            is_sweep=True,
            discrete_values=args.homophily_density_combinations
        ),
        # Distributional shift evaluation sweep parameters
        # Creates 7 combinations: no shift + (homophily/density/n_nodes) × (test_only=True/False)
        # Format: (shift_enabled, shift_type, test_only)
        'distributional_shift_combination': ParameterRange(
            min_val=0,
            max_val=6,  # 7 combinations total
            step=1,
            is_sweep=True,
            discrete_values=[
                (False, None, True),  # No distributional shift (baseline)
                (True, 'homophily', True),  # Homophily shift, test only
                # (True, 'homophily', False),  # Homophily shift, train + test
                # (True, 'density', True),  # Density shift, test only
                # (True, 'density', False),  # Density shift, train + test
                # (True, 'n_nodes', True),  # N_nodes shift, test only
                # (True, 'n_nodes', False)  # N_nodes shift, train + test
            ]
        )
    }
    
    return CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters={},  # No more random parameters
        n_repetitions=args.n_repetitions,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        continue_on_failure=True,
        run_neural_sheaf=args.run_neural_sheaf
    )

def load_intermediate_results(result_dir: str) -> tuple:
    """Load intermediate results and config from a directory."""
    # Load config
    config_path = os.path.join(result_dir, "multi_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert sweep and random parameters back to ParameterRange objects
    if 'sweep_parameters' in config_dict:
        sweep_params = {}
        for param_name, param_dict in config_dict['sweep_parameters'].items():
            # Convert any None values to proper None
            for key, value in param_dict.items():
                if value == "null":
                    param_dict[key] = None
            sweep_params[param_name] = ParameterRange(**param_dict)
        config_dict['sweep_parameters'] = sweep_params
    
    if 'random_parameters' in config_dict:
        random_params = {}
        for param_name, param_dict in config_dict['random_parameters'].items():
            # Convert any None values to proper None
            for key, value in param_dict.items():
                if value == "null":
                    param_dict[key] = None
            random_params[param_name] = ParameterRange(**param_dict)
        config_dict['random_parameters'] = random_params
    
    # Convert base_config to InductiveExperimentConfig
    if 'base_config' in config_dict:
        from experiments.inductive.config import InductiveExperimentConfig
        config_dict['base_config'] = InductiveExperimentConfig(**config_dict['base_config'])
    
    # Load results
    results_path = os.path.join(result_dir, "intermediate_results.json")
    if not os.path.exists(results_path):
        raise ValueError(f"Results file not found at {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return config_dict, results


def get_next_sweep_params(config_dict: dict, completed_results: list) -> dict:
    """Get the next sweep parameter combination to run."""
    # Get all sweep parameter combinations
    config = CleanMultiExperimentConfig(**config_dict)
    all_combinations = config.get_parameter_combinations()
    
    # Get completed sweep parameter combinations
    completed_combos = set()
    for result in completed_results:
        sweep_params = result.get('sweep_parameters', {})
        # Convert to tuple of sorted items for hashability
        param_tuple = tuple(sorted(sweep_params.items()))
        completed_combos.add(param_tuple)
    
    # Find first uncompleted combination
    for combo in all_combinations:
        combo_tuple = tuple(sorted(combo.items()))
        if combo_tuple not in completed_combos:
            return combo
    
    return None

def main():
    """Main function to run multi-experiments."""
    print("CLEAN MULTI-EXPERIMENT PARAMETER SWEEPS")
    print("=" * 60)
    
    try:
        args = parse_args()
        
        # Handle analysis of existing results
        if args.analyze_existing:
            print(f"Analyzing existing results at: {args.analyze_existing}")
            
            # Import the analysis function
            from experiments.inductive.multi_experiment import analyze_saved_multi_experiment_data
            
            # Generate data analysis report
            analysis_report = analyze_saved_multi_experiment_data(args.analyze_existing)
            print("\n" + analysis_report)
            
            # Save the analysis report
            report_path = os.path.join(args.analyze_existing, "data_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(analysis_report)
            print(f"Data analysis report saved to: {report_path}")
            
            # Create plots if requested
            if args.create_plots:
                create_analysis_plots(args.analyze_existing)
                print("Analysis plots created!")
            
            print("Analysis complete!")
            return 0
        
        # Handle continuing from intermediate results
        if args.continue_from_intermediate:
            if not args.intermediate_result_dir:
                raise ValueError("--intermediate_result_dir must be specified when continuing from intermediate results")
            
            print(f"Loading intermediate results from: {args.intermediate_result_dir}")
            config_dict, results = load_intermediate_results(args.intermediate_result_dir)
            
            # Get next sweep parameters to run
            next_combo = get_next_sweep_params(config_dict, results['all_results'])
            if next_combo is None:
                print("All parameter combinations have been completed!")
                return 0
            
            print(f"Continuing from last completed run. Next sweep parameters: {next_combo}")
            
            # Create new config with same settings but new output directory
            config = CleanMultiExperimentConfig(**config_dict)
            config.output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_continued")
            config.experiment_name = f"{args.experiment_name}_continued"
            
            # Override repetitions if specified
            if args.n_repetitions != 2:
                config.n_repetitions = args.n_repetitions
        else:
            # Create configuration based on preset or custom
            print("Creating custom configuration from arguments...")
            config = create_custom_experiment(args)
            
            # Override repetitions if specified
            if args.n_repetitions != 2:
                config.n_repetitions = args.n_repetitions
        
        # Print configuration summary
        print(f"\nMulti-Experiment Configuration:")
        print(f"  Preset: {args.preset}")
        print(f"  Experiment name: {config.experiment_name}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Total runs planned: {config.get_total_runs()}")
        print(f"  Base method: {'DCCC-SBM' if config.base_config.use_dccc_sbm else 'DC-SBM'}")
        
        print(f"\nSweep parameters ({len(config.sweep_parameters)}):")
        for param, param_range in config.sweep_parameters.items():
            if hasattr(param_range, 'get_sweep_values'):
                try:
                    values = param_range.get_sweep_values()
                    if param == 'homophily_density_combination':
                        print(f"  {param}: {len(values)} combinations")
                        for i, combo in enumerate(values):
                            print(f"    Sweep {i}: {combo}")
                    else:
                        print(f"  {param}: {len(values)} values from {min(values)} to {max(values)}")
                except:
                    print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
            else:
                print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
        
        print(f"\nFixed parameters (previously random):")
        print(f"  degree_heterogeneity: {args.degree_heterogeneity}")
        print(f"  edge_noise: {args.edge_noise}")
        print(f"  cluster_count_factor: {args.cluster_count_factor}")
        print(f"  cluster_variance: {args.cluster_variance}")
        print(f"  center_variance: {args.center_variance}")
        print(f"  assignment_skewness: {args.assignment_skewness}")
        print(f"  community_exclusivity: {args.community_exclusivity}")
        print(f"  universe_randomness_factor: {args.universe_randomness_factor}")
        if args.use_dccc_sbm:
            print(f"  community_imbalance_range_width: {args.community_imbalance_range_width}")
            print(f"  degree_separation_range_width: {args.degree_separation_range_width}")
        
        # Confirm before starting
        if config.get_total_runs() > 10:
            response = input(f"\nThis will run {config.get_total_runs()} experiments. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 0
        
        # Run experiments
        print(f"\nStarting multi-experiment suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.continue_from_intermediate:
            results = run_clean_multi_experiments(config, continue_from_results=results)
        else:
            results = run_clean_multi_experiments(config)
        
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
        if args.create_plots and results['summary_stats']['successful_runs'] > 0:
            print(f"\nCreating analysis plots...")
            
            # Find the output directory from results
            output_dir = None
            if results['all_results']:
                # Get the parent directory of the first run
                first_result = results['all_results'][0]
                if 'config_path' in first_result:
                    run_dir = first_result['config_path']
                    output_dir = os.path.dirname(run_dir)
            
            if output_dir and os.path.exists(os.path.join(output_dir, 'results_summary.csv')):
                create_analysis_plots(output_dir)
                print(f"Analysis plots created in: {os.path.join(output_dir, 'plots')}")
            else:
                print("Could not find results for plotting")
        
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
    import sys

    # Run main multi-experiment
    exit_code = main()
    sys.exit(exit_code)