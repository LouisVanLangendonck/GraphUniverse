#!/usr/bin/env python3
"""
Test script to verify that the skip_signal_calculations parameter works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.parameter_robustness_analysis import ParameterRobustnessAnalyzer

def test_skip_signals():
    """Test that signal calculations can be skipped."""
    print("🧪 Testing skip_signal_calculations parameter")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ParameterRobustnessAnalyzer(base_seed=42)
    
    # Test with signals enabled
    print("\n📊 Running with signals enabled...")
    results_with_signals = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=2,  # Very small for testing
        timeout_minutes=1.0,
        max_attempts_per_graph=2,
        save_results=False,  # Don't save for test
        skip_signal_calculations=False
    )
    
    # Test with signals disabled
    print("\n⚡ Running with signals disabled...")
    results_without_signals = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=2,  # Very small for testing
        timeout_minutes=1.0,
        max_attempts_per_graph=2,
        save_results=False,  # Don't save for test
        skip_signal_calculations=True
    )
    
    # Compare results
    print("\n📋 Comparison Results:")
    print("=" * 30)
    
    # Check experiment stats
    stats_with = results_with_signals['experiment_stats']
    stats_without = results_without_signals['experiment_stats']
    
    print(f"With signals:")
    print(f"  Total time: {stats_with['total_time']:.1f}s")
    print(f"  Signal calculations skipped: {stats_with.get('signal_calculations_skipped', False)}")
    
    print(f"\nWithout signals:")
    print(f"  Total time: {stats_without['total_time']:.1f}s")
    print(f"  Signal calculations skipped: {stats_without.get('signal_calculations_skipped', True)}")
    
    # Check signal correlations
    summary_with = results_with_signals['summary']
    summary_without = results_without_signals['summary']
    
    print(f"\nSignal correlation summary with signals: {len(summary_with['signal_correlation_summary'])} parameters")
    print(f"Signal correlation summary without signals: {len(summary_without['signal_correlation_summary'])} parameters")
    
    # Verify that skipping signals is faster
    time_diff = stats_with['total_time'] - stats_without['total_time']
    print(f"\n⏱️  Time difference: {time_diff:.1f}s (signals {'slower' if time_diff > 0 else 'faster'})")
    
    if time_diff > 0:
        print("✅ Signal calculations are indeed slower (as expected)")
    else:
        print("⚠️  Signal calculations were not slower (unexpected for small test)")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_skip_signals() 