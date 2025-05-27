"""
Script to run clean multi-experiment sweeps.
Example usage of parameter sweeps with random sampling.
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.inductive.multi_config import (
    CleanMultiExperimentConfig, 
    ParameterRange,
    create_homophily_density_sweep,
    create_dccc_method_comparison,
    create_large_scale_benchmark
)
from experiments.inductive.multi_experiment import run_clean_multi_experiments, create_analysis_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run clean multi-experiment parameter sweeps')
    
    # Experiment selection
    parser.add_argument('--preset', type=str, 
                        choices=['homophily_density', 'dccc_comparison', 'benchmark', 'custom'],
                        default='custom',
                        help='Preset experiment configuration')
    
    # Custom experiment parameters
    parser.add_argument('--experiment_name', type=str, default='custom_sweep',
                        help='Name for custom experiment')
    parser.add_argument('--output_dir', type=str, default='multi_results',
                        help='Base output directory')
    parser.add_argument('--n_repetitions', type=int, default=2,
                        help='Number of repetitions per parameter combination')
    
    # Task configuration
    parser.add_argument('--tasks', type=str, nargs='+', 
                        default=['k_hop_community_counts'],
                        choices=['community', 'k_hop_community_counts'],
                        help='Learning tasks to run')
    parser.add_argument('--khop_k', type=int, default=2,
                        help='k value for k-hop community counting task')
    
    # Base experiment settings
    parser.add_argument('--n_graphs', type=int, default=12,
                        help='Number of graphs per family')
    parser.add_argument('--min_n_nodes', type=int, default=80,
                        help='Minimum nodes per graph')
    parser.add_argument('--max_n_nodes', type=int, default=100,
                        help='Maximum nodes per graph')
    parser.add_argument('--universe_K', type=int, default=5,
                        help='Number of communities in universe')
    
    # Method selection
    parser.add_argument('--use_dccc_sbm', action='store_true',
                        help='Use DCCC-SBM (for custom experiments)')
    parser.add_argument('--degree_distribution', type=str, default='power_law',
                        choices=['standard', 'power_law', 'exponential', 'uniform'],
                        help='Degree distribution for DCCC-SBM')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (per graph)')
    
    # Hyperparameter optimization settings
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--optimization_timeout', type=int, default=600,
                        help='Timeout in seconds for hyperparameter optimization')
    
    # Analysis options
    parser.add_argument('--create_plots', action='store_true', default=False,
                        help='Create analysis plots after experiments')
    parser.add_argument('--analyze_existing', type=str, default=None,
                        help='Analyze existing results at given path')
    
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
        min_communities=5,
        max_communities=5,
        universe_K=args.universe_K,
        universe_feature_dim=32,
        
        # Method
        use_dccc_sbm=args.use_dccc_sbm,
        degree_distribution=args.degree_distribution,

        # Tasks configuration
        tasks=args.tasks,
        khop_community_counts_k=args.khop_k,
        is_regression={
            'community': False,
            'k_hop_community_counts': True
        },
        
        # Tasks and models
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=True,
        
        # Training
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        
        # Hyperparameter optimization
        optimize_hyperparams=args.optimize_hyperparams,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        
        # Analysis
        collect_signal_metrics=True,
        require_consistency_check=True,
        
        # Seed
        seed=args.seed
    )
    
    # Define a simple sweep: homophily vs edge density
    sweep_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            step=0.2,
            is_sweep=True
        ),
        'universe_edge_density': ParameterRange(
            min_val=0.02,
            max_val=0.20,
            step=0.05,
            is_sweep=True
        )
    }
    
    # Random parameters
    random_parameters = {
        'homophily_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.2,
            distribution="uniform",
            is_sweep=False
        ),
        'density_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.05,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.1,
            distribution="uniform",
            is_sweep=False
        ),
        'cluster_count_factor': ParameterRange(
            min_val=0.5,
            max_val=1.5,
            distribution="uniform",
            is_sweep=False
        ),
        'cluster_variance': ParameterRange(
            min_val=0.01,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'center_variance': ParameterRange(
            min_val=0.01,
            max_val=1.5,
            distribution="uniform",
            is_sweep=False
        ),
        'assignment_skewness': ParameterRange(
            min_val=0.0,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'community_exclusivity': ParameterRange(
            min_val=0.6,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'universe_randomness_factor': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    # Add DCCC-specific random parameters if needed
    if args.use_dccc_sbm:
        random_parameters.update({
            'community_imbalance_range_width': ParameterRange(
                min_val=0.0,
                max_val=0.5,
                distribution="uniform",
                is_sweep=False
            ),
            'degree_separation_range_width': ParameterRange(
                min_val=0.3,
                max_val=1.0,
                distribution="uniform",
                is_sweep=False
            )
        })
    
    return CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=args.n_repetitions,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        continue_on_failure=True,
        aggregate_results=True,
        create_summary_plots=args.create_plots
    )


def main():
    """Main function to run multi-experiments."""
    print("CLEAN MULTI-EXPERIMENT PARAMETER SWEEPS")
    print("=" * 60)
    
    try:
        args = parse_args()
        
        # Handle analysis of existing results
        if args.analyze_existing:
            print(f"Analyzing existing results at: {args.analyze_existing}")
            create_analysis_plots(args.analyze_existing)
            print("Analysis complete!")
            return 0
        
        # Create configuration based on preset or custom
        if args.preset == 'homophily_density':
            print("Using homophily-density sweep configuration...")
            config = create_homophily_density_sweep()
            
        elif args.preset == 'dccc_comparison':
            print("Using DCCC method comparison configuration...")
            config = create_dccc_method_comparison()
            
        elif args.preset == 'benchmark':
            print("Using large-scale benchmark configuration...")
            config = create_large_scale_benchmark()
            
        elif args.preset == 'custom':
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
                    print(f"  {param}: {len(values)} values from {min(values)} to {max(values)}")
                except:
                    print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
            else:
                print(f"  {param}: {param_range.min_val} to {param_range.max_val}")
        
        print(f"\nRandom parameters ({len(config.random_parameters)}):")
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
        if config.create_summary_plots and results['summary_stats']['successful_runs'] > 0:
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


def demo_experiments():
    """Run a quick demo of different experiment types."""
    print("RUNNING DEMO EXPERIMENTS")
    print("=" * 40)
    
    from experiments.inductive.config import InductiveExperimentConfig
    
    # 1. Quick DC-SBM vs DCCC-SBM comparison
    print("\n1. Quick method comparison (DC-SBM vs DCCC-SBM)")
    
    base_config = InductiveExperimentConfig(
        n_graphs=6,  # Small for demo
        min_n_nodes=60,
        max_n_nodes=80,
        min_communities=3,
        max_communities=4,
        universe_K=4,
        universe_feature_dim=16,
        tasks=['community'],
        gnn_types=['gcn'],
        run_gnn=True,
        run_mlp=True,
        run_rf=False,
        epochs=50,  # Fast for demo
        patience=10,
        collect_signal_metrics=True,
        require_consistency_check=False
    )
    
    # Sweep method type
    sweep_parameters = {
        'use_dccc_sbm': ParameterRange(
            min_val=0,  # False
            max_val=1,  # True
            step=1,
            is_sweep=True
        )
    }
    
    # Random parameters
    random_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.4,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.3,
            max_val=0.7,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    demo_config = CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=2,
        experiment_name="demo_method_comparison",
        output_dir="demo_results",
        continue_on_failure=True
    )
    
    print(f"Running {demo_config.get_total_runs()} experiments...")
    results = run_clean_multi_experiments(demo_config)
    
    # Quick analysis
    if results['all_results']:
        print("\nDemo Results:")
        dc_results = [r for r in results['all_results'] if r['method'] == 'dc_sbm']
        dccc_results = [r for r in results['all_results'] if r['method'] == 'dccc_sbm']
        
        print(f"  DC-SBM runs: {len(dc_results)}")
        print(f"  DCCC-SBM runs: {len(dccc_results)}")
        
        # Compare average performance
        def get_avg_performance(results_list, metric='community_gcn_f1_macro'):
            performances = []
            for r in results_list:
                model_results = r.get('model_results', {})
                community_results = model_results.get('community', {})
                gcn_results = community_results.get('gcn', {})
                test_metrics = gcn_results.get('test_metrics', {})
                if 'f1_macro' in test_metrics:
                    performances.append(test_metrics['f1_macro'])
            return sum(performances) / len(performances) if performances else 0.0
        
        dc_avg = get_avg_performance(dc_results)
        dccc_avg = get_avg_performance(dccc_results)
        
        print(f"  DC-SBM avg F1: {dc_avg:.3f}")
        print(f"  DCCC-SBM avg F1: {dccc_avg:.3f}")
        print(f"  Difference: {dccc_avg - dc_avg:+.3f}")
        
        # Compare signals
        def get_avg_signal(results_list, signal_type='degree_signals'):
            signals = []
            for r in results_list:
                community_signals = r.get('community_signals', {})
                signal_data = community_signals.get(signal_type, {})
                if 'mean' in signal_data:
                    signals.append(signal_data['mean'])
            return sum(signals) / len(signals) if signals else 0.0
        
        dc_degree_signal = get_avg_signal(dc_results)
        dccc_degree_signal = get_avg_signal(dccc_results)
        
        print(f"  DC-SBM avg degree signal: {dc_degree_signal:.3f}")
        print(f"  DCCC-SBM avg degree signal: {dccc_degree_signal:.3f}")
    
    print(f"\nDemo completed! Results saved to: demo_results")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run quick demo
        demo_experiments()
    else:
        # Run main multi-experiment
        exit_code = main()
        sys.exit(exit_code)