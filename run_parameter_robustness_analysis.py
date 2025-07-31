#!/usr/bin/env python3
"""
Parameter Robustness Analysis Runner

This script runs comprehensive parameter robustness analysis to test how well
the graph family generator maintains requested parameter intervals while randomizing
all other parameters.

Usage:
    python run_parameter_robustness_analysis.py [--quick] [--full] [--custom]

Options:
    --quick: Run with reduced parameters for quick testing (default)
    --full: Run with full parameters for comprehensive analysis
    --custom: Run with custom parameters
"""

import argparse
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.parameter_robustness_analysis import ParameterRobustnessAnalyzer

def run_quick_analysis():
    """Run a quick analysis with reduced parameters."""
    print("🚀 Running Quick Parameter Robustness Analysis")
    print("=" * 50)
    
    analyzer = ParameterRobustnessAnalyzer(base_seed=42)
    
    results = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=10,  # Reduced for speed
        timeout_minutes=3.0,       # Shorter timeout
        max_attempts_per_graph=5,  # Fewer attempts
        save_results=True,
        results_dir="quick_parameter_robustness_results",
        skip_signal_calculations=True  # Skip signals for quick runs
    )
    
    return results

def run_full_analysis():
    """Run a full analysis with comprehensive parameters."""
    print("🚀 Running Full Parameter Robustness Analysis")
    print("=" * 50)
    
    analyzer = ParameterRobustnessAnalyzer(base_seed=42)
    
    results = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=100,  # Full analysis
        timeout_minutes=10.0,       # Longer timeout
        max_attempts_per_graph=10,  # More attempts
        save_results=True,
        results_dir="full_parameter_robustness_results",
        skip_signal_calculations=True  # Include signals for full analysis
    )
    
    return results

def run_custom_analysis():
    """Run a custom analysis with user-defined parameters."""
    print("🚀 Running Custom Parameter Robustness Analysis")
    print("=" * 50)
    
    # Get user input for custom parameters
    try:
        n_graphs = int(input("Number of graphs per interval (default: 50): ") or "50")
        timeout = float(input("Timeout in minutes (default: 5.0): ") or "5.0")
        max_attempts = int(input("Max attempts per graph (default: 5): ") or "5")
        results_dir = input("Results directory (default: custom_parameter_robustness_results): ") or "custom_parameter_robustness_results"
        
        # Ask about signal calculations
        skip_signals_input = input("Skip signal calculations for faster execution? (y/n, default: n): ").lower()
        skip_signals = skip_signals_input in ['y', 'yes']
        
    except ValueError:
        print("Invalid input, using defaults")
        n_graphs = 50
        timeout = 5.0
        max_attempts = 5
        results_dir = "custom_parameter_robustness_results"
        skip_signals = False
    
    analyzer = ParameterRobustnessAnalyzer(base_seed=42)
    
    results = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=n_graphs,
        timeout_minutes=timeout,
        max_attempts_per_graph=max_attempts,
        save_results=True,
        results_dir=results_dir,
        skip_signal_calculations=skip_signals
    )
    
    return results

def print_results_summary(results):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("="*60)
    
    summary = results['summary']
    stats = results['experiment_stats']
    
    print(f"\n⏱️  Experiment Statistics:")
    print(f"   Total Time: {stats['total_time']:.1f} seconds")
    print(f"   Total Graphs Generated: {stats['total_graphs']}")
    print(f"   Parameters Tested: {stats['parameters_tested']}")
    if 'signal_calculations_skipped' in stats:
        print(f"   Signal Calculations: {'Skipped' if stats['signal_calculations_skipped'] else 'Included'}")
    
    print(f"\n🎯 Coverage Performance:")
    print(f"   {'Parameter':<20} {'Mean Coverage':<15} {'Std Coverage':<15}")
    print(f"   {'-'*20} {'-'*15} {'-'*15}")
    
    for param, data in summary['overall_coverage'].items():
        print(f"   {param:<20} {data['mean_coverage']:<15.3f} {data['std_coverage']:<15.3f}")
    
    print(f"\n🏆 Top Performing Parameters:")
    for i, (param, coverage) in enumerate(summary['best_performing_parameters'][:3], 1):
        print(f"   {i}. {param}: {coverage:.3f}")
    
    print(f"\n📈 Signal Correlation Highlights:")
    if summary['signal_correlation_summary']:
        for param, signals in summary['signal_correlation_summary'].items():
            if signals:  # Check if signals were calculated
                best_signal = max(signals.items(), key=lambda x: x[1]['r_squared'])
                print(f"   {param}: Best correlation with {best_signal[0]} (R² = {best_signal[1]['r_squared']:.3f})")
    else:
        print("   Signal calculations were skipped for faster execution")
    
    print(f"\n📁 Results saved to: {results.get('results_dir', 'parameter_robustness_results')}")

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Run parameter robustness analysis for graph family generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_parameter_robustness_analysis.py --quick         # Quick test (signals skipped)
  python run_parameter_robustness_analysis.py --full          # Full analysis with signals
  python run_parameter_robustness_analysis.py --full --skip-signals  # Full analysis without signals
  python run_parameter_robustness_analysis.py --custom        # Custom parameters
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick analysis with reduced parameters'
    )
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run full analysis with comprehensive parameters'
    )
    parser.add_argument(
        '--custom', 
        action='store_true',
        help='Run custom analysis with user-defined parameters'
    )
    parser.add_argument(
        '--skip-signals', 
        action='store_true',
        help='Skip signal calculations for faster execution'
    )
    
    args = parser.parse_args()
    
    # Determine which analysis to run
    if args.full:
        results = run_full_analysis()
    elif args.custom:
        results = run_custom_analysis()
    else:
        # Default to quick analysis
        results = run_quick_analysis()
    
    # Override signal calculations if --skip-signals is specified
    if args.skip_signals:
        print("⚡ Signal calculations will be skipped for faster execution")
        # Re-run with skip_signal_calculations=True
        analyzer = ParameterRobustnessAnalyzer(base_seed=42)
        if args.full:
            results = analyzer.run_comprehensive_analysis(
                n_graphs_per_interval=100,
                timeout_minutes=10.0,
                max_attempts_per_graph=10,
                save_results=True,
                results_dir="full_parameter_robustness_results",
                skip_signal_calculations=True
            )
        elif args.custom:
            # For custom, we'll use the user's choice but override if --skip-signals
            results = run_custom_analysis()
        else:
            # Quick analysis already skips signals by default
            pass
    
    # Print summary
    print_results_summary(results)
    
    print(f"\n✅ Analysis complete! Check the results directory for detailed outputs and visualizations.")

if __name__ == "__main__":
    main() 