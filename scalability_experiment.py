"""
Scalability Experiment for GraphUniverse

This script benchmarks the graph generation performance across different graph sizes.
Generates 100 graphs per configuration and measures timing statistics.
"""

import time
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_universe import GraphUniverse


def run_scalability_experiment(
    avg_node_sizes: List[int] = [10, 100, 500, 1000],
    n_graphs_per_size: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run scalability experiment across different graph sizes.
    
    Args:
        avg_node_sizes: List of average node counts to test
        n_graphs_per_size: Number of graphs to generate per configuration
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with timing statistics for each configuration
    """
    
    results = []
    
    # Default/average property ranges
    default_params = {
        "K": 10,
        "feature_dim": 15,
        "edge_propensity_variance": 0.5,
        "center_variance": 0.1,
        "cluster_variance": 0.5,
        "homophily_range": (0.1, 0.4),
        "avg_degree_range": (2.0, 5.0),
        "min_communities": 3,
        "max_communities": 8,
        "degree_distribution": "power_law",
        "power_law_exponent_range": (2.0, 2.5),
        "degree_separation_range": (0.7, 1.0),
    }
    
    print("=" * 80)
    print("GRAPH UNIVERSE SCALABILITY EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Graphs per size: {n_graphs_per_size}")
    print(f"  - Node sizes to test: {avg_node_sizes}")
    print(f"  - Random seed: {seed}")
    print(f"\nDefault parameters:")
    for key, value in default_params.items():
        print(f"  - {key}: {value}")
    print("\n" + "=" * 80)
    
    for avg_nodes in avg_node_sizes:
        print(f"\n{'='*80}")
        print(f"Testing with n_nodes = {avg_nodes}")
        print(f"{'='*80}")
        
        try:
            # Create universe
            print("\n[1/3] Creating universe...")
            universe_start = time.time()
            universe = GraphUniverse(
                K=default_params["K"],
                feature_dim=default_params["feature_dim"],
                edge_propensity_variance=default_params["edge_propensity_variance"],
                center_variance=default_params["center_variance"],
                cluster_variance=default_params["cluster_variance"],
                seed=seed,
            )
            universe_time = time.time() - universe_start
            print(f"Universe created in {universe_time:.3f}s")
            
            # Create family generator
            print("\n[2/3] Setting up family generator...")
            generator_start = time.time()
            family_generator = GraphFamilyGenerator(
                universe=universe,
                min_n_nodes=avg_nodes,
                max_n_nodes=avg_nodes,
                min_communities=default_params["min_communities"],
                max_communities=default_params["max_communities"],
                homophily_range=default_params["homophily_range"],
                avg_degree_range=default_params["avg_degree_range"],
                degree_distribution=default_params["degree_distribution"],
                power_law_exponent_range=default_params["power_law_exponent_range"],
                degree_separation_range=default_params["degree_separation_range"],
                seed=seed,
            )
            generator_time = time.time() - generator_start
            print(f"Generator setup completed in {generator_time:.3f}s")
            
            # Generate all graphs at once
            print(f"\n[3/3] Generating {n_graphs_per_size} graphs...")
            generation_start = time.time()
            
            try:
                # Generate all graphs in one call (much more efficient)
                family_generator.generate_family(
                    n_graphs=n_graphs_per_size,
                    show_progress=True,
                    collect_stats=False,
                    timeout_minutes=30,
                )
                
                total_generation_time = time.time() - generation_start
                n_successful = len(family_generator.graphs)
                
            except Exception as e:
                print(f"  Error during generation: {e}")
                total_generation_time = time.time() - generation_start
                n_successful = len(family_generator.graphs) if hasattr(family_generator, 'graphs') else 0
            
            # Calculate statistics
            if n_successful > 0:
                mean_time_per_graph = total_generation_time / n_successful
                throughput = n_successful / total_generation_time
                
                # Get actual graph statistics
                actual_nodes = [g.n_nodes for g in family_generator.graphs]
                actual_edges = [g.graph.number_of_edges() for g in family_generator.graphs]
                
                result = {
                    "avg_target_nodes": avg_nodes,
                    "n_graphs_requested": n_graphs_per_size,
                    "n_graphs_generated": n_successful,
                    "success_rate": n_successful / n_graphs_per_size,
                    "total_time_s": total_generation_time,
                    "mean_time_per_graph_s": mean_time_per_graph,
                    "throughput_graphs_per_s": throughput,
                    "actual_avg_nodes": np.mean(actual_nodes) if actual_nodes else 0,
                    "actual_avg_edges": np.mean(actual_edges) if actual_edges else 0,
                    "universe_creation_time_s": universe_time,
                    "generator_setup_time_s": generator_time,
                }
                
                results.append(result)
                
                # Print summary
                print(f"\n{'─'*80}")
                print(f"RESULTS FOR avg_nodes = {avg_nodes}")
                print(f"{'─'*80}")
                print(f"Graphs generated: {n_successful}/{n_graphs_per_size} "
                      f"({result['success_rate']*100:.1f}% success)")
                print(f"Total time: {total_generation_time:.2f}s")
                print(f"Time per graph: {mean_time_per_graph:.4f}s")
                print(f"Throughput: {throughput:.2f} graphs/s")
                print(f"Actual avg nodes: {result['actual_avg_nodes']:.1f}")
                print(f"Actual avg edges: {result['actual_avg_edges']:.1f}")
                print(f"{'─'*80}")
            else:
                print(f"\nFailed to generate any graphs for avg_nodes = {avg_nodes}")
                
        except Exception as e:
            print(f"\nError during experiment for avg_nodes = {avg_nodes}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame with results
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        print("\nNo successful experiments")
        return pd.DataFrame()


def save_and_display_results(df: pd.DataFrame, output_file: str = "scalability_results.csv"):
    """
    Save results to CSV and display summary.
    
    Args:
        df: DataFrame with experimental results
        output_file: Path to save CSV file
    """
    if df.empty:
        print("\nNo results to save.")
        return
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Display summary table
    print("\nSUMMARY TABLE")
    print("=" * 80)
    
    summary_df = df[[
        "avg_target_nodes",
        "n_graphs_generated",
        "success_rate",
        "total_time_s",
        "mean_time_per_graph_s",
        "throughput_graphs_per_s",
        "actual_avg_nodes",
        "actual_avg_edges",
    ]].copy()
    
    # Format columns for display
    summary_df["success_rate"] = summary_df["success_rate"].apply(lambda x: f"{x*100:.1f}%")
    summary_df["total_time_s"] = summary_df["total_time_s"].apply(lambda x: f"{x:.2f}")
    summary_df["mean_time_per_graph_s"] = summary_df["mean_time_per_graph_s"].apply(
        lambda x: f"{x:.4f}"
    )
    summary_df["throughput_graphs_per_s"] = summary_df["throughput_graphs_per_s"].apply(
        lambda x: f"{x:.2f}"
    )
    summary_df["actual_avg_nodes"] = summary_df["actual_avg_nodes"].apply(lambda x: f"{x:.1f}")
    summary_df["actual_avg_edges"] = summary_df["actual_avg_edges"].apply(lambda x: f"{x:.1f}")
    
    # Rename columns for better display
    summary_df.columns = [
        "Target Nodes",
        "Graphs",
        "Success",
        "Total Time (s)",
        "Time/Graph (s)",
        "Throughput (g/s)",
        "Actual Nodes",
        "Actual Edges",
    ]
    
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    # Additional statistics
    print("\nADDITIONAL STATISTICS")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\nTarget nodes: {row['avg_target_nodes']}")
        print(f"  Total time: {row['total_time_s']:.2f}s")
        print(f"  Time per graph: {row['mean_time_per_graph_s']:.4f}s")
        print(f"  Throughput: {row['throughput_graphs_per_s']:.2f} graphs/s")
        print(f"  Success rate: {row['success_rate']*100:.1f}% "
              f"({row['n_graphs_generated']}/{row['n_graphs_requested']} graphs)")
        print(f"  Actual avg nodes: {row['actual_avg_nodes']:.1f}")
        print(f"  Actual avg edges: {row['actual_avg_edges']:.1f}")


def main():
    """Run the scalability experiment."""
    
    # Configuration
    avg_node_sizes = [10, 100, 500, 1000]
    n_graphs_per_size = 100
    seed = 42
    output_file = "scalability_results.csv"
    
    # Run experiment
    start_time = time.time()
    df = run_scalability_experiment(
        avg_node_sizes=avg_node_sizes,
        n_graphs_per_size=n_graphs_per_size,
        seed=seed,
    )
    total_time = time.time() - start_time
    
    # Save and display results
    save_and_display_results(df, output_file)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiment time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

