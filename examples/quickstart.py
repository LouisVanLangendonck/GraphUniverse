#!/usr/bin/env python3
"""
GraphUniverse Quick Start Notebook Script
=========================================

This script demonstrates all three usage patterns from the GraphUniverse Quick Start (except interactive demo):
1. Via individual classes
2. Via YAML config file  
3. Validation & quality analysis 
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path

# Import GraphUniverse components
from graph_universe import (
    GraphUniverse,
    GraphFamilyGenerator,
    GraphUniverseDataset,
    plot_graph_communities,
)

# =============================================================================
# OPTION 1: VIA INDIVIDUAL CLASSES
# =============================================================================

print("Option 1: Via Individual Classes")
print("-" * 40)

# Create universe
print("Creating GraphUniverse...")
universe = GraphUniverse(K=8, edge_propensity_variance=0.3, feature_dim=10)

# Generate family  
print("Creating GraphFamilyGenerator...")
family = GraphFamilyGenerator(
    universe=universe,
    min_n_nodes=25, 
    max_n_nodes=50,
    min_communities=2,
    max_communities=7,
    homophily_range=(0.2, 0.8),
    avg_degree_range=(2.0, 10.0),
    degree_distribution="power_law",
    power_law_exponent_range=(2.0, 5.0),
    degree_separation_range=(0.1, 0.7),
    seed=42
)

print("Generating 100 graphs...")
family.generate_family(
    n_graphs=30,
    show_progress=True
)

print(f"Generated {len(family.graphs)} graphs!")

# Convert to PyG graphs, ready for training on community detection task
print("🔗 Converting to PyTorch Geometric format...")
try:
    pyg_graphs = family.to_pyg_graphs(tasks=["community_detection"])
    print(f"Created {len(pyg_graphs)} PyG Data objects")
    
    # Show sample properties
    sample_graph = pyg_graphs[0]
    print(f"Sample graph properties:")
    print(f"   - Nodes: {sample_graph.num_nodes}")
    print(f"   - Edges: {sample_graph.num_edges}")
    print(f"   - Features shape: {sample_graph.x.shape}")
    print(f"   - Labels shape: {sample_graph.y.shape}")
    
except Exception as e:
    print(f"PyG conversion failed (this is normal if PyTorch not installed): {e}")
    pyg_graphs = None

# =============================================================================
# OPTION 2: VIA YAML CONFIG FILE
# =============================================================================

print("\nOption 2: Via YAML Config File")
print("-" * 40)

# Create the experiment config as shown in your README
experiment_config = {
    "universe_parameters": {
        "K": 10,
        "edge_propensity_variance": 0.5,
        "feature_dim": 16,
        "center_variance": 1.0,
        "cluster_variance": 0.3,
        "seed": 42
    },
    "family_parameters": {
        "n_graphs": 100,
        "min_n_nodes": 25,
        "max_n_nodes": 200,
        "min_communities": 3,
        "max_communities": 7,
        "homophily_range": [0.1, 0.9],
        "avg_degree_range": [2.0, 8.0],
        "degree_distribution": "power_law",
        "power_law_exponent_range": [2.0, 3.0],
        "degree_separation_range": [0.4, 0.8],
        "seed": 42
    },
    "tasks": ["community_detection", "triangle_counting"]
}

# Save config file
print("Creating experiment config file...")
config_dir = Path("./configs")
config_dir.mkdir(exist_ok=True)

config_path = config_dir / "run.yaml"
with open(config_path, "w") as f:
    yaml.dump(experiment_config, f, default_flow_style=False, indent=2)

print(f"Saved config to: {config_path}")

# Use config-driven workflow (as shown in your README)
print("Loading config and generating dataset...")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

try:
    dataset = GraphUniverseDataset(root="./data", parameters=config)
    print(f"Generated dataset with {len(dataset)} graphs!")
    print(f"Dataset location: {dataset.get_data_dir()}")
    
    # Show sample from dataset
    sample = dataset[0]
    print(f"Sample from dataset:")
    print(f"   - Nodes: {sample.num_nodes}")
    print(f"   - Edges: {sample.num_edges}")
    if hasattr(sample, 'x'):
        print(f"   - Features: {sample.x.shape}")
    if 'tasks' in config:
        for task in config['tasks']:
            if hasattr(sample, task):
                print(f"   - {task}: {getattr(sample, task).shape}")
        
except Exception as e:
    print(f"Dataset creation failed: {e}")
    dataset = None

# =============================================================================
# VALIDATION & QUALITY (Your exact code from README)
# =============================================================================

print("\nValidation & Quality Analysis")
print("-" * 40)

print("GraphUniverse includes comprehensive metrics to validate desired property realization...")

# Using the family from Option 1 for validation
print("Analyzing graph family properties...")

try:
    # Validate Standard Graph Property Generation
    family_properties = family.analyze_graph_family_properties()
    
    print("\nGraph Family Properties:")
    for property_name in ['node_counts', 'avg_degrees', 'homophily_levels', 'mean_edge_probability_deviation']:
        if property_name in family_properties:
            values = family_properties[property_name]
            if isinstance(values, (list, np.ndarray)):
                print(f"   {property_name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
            else:
                print(f"   {property_name}: {values}")
        else:
            print(f"   {property_name}: Not calculated")

    # Calculate Within-graph Community-related Signals
    print("\nCalculating within-graph community-related signals (might take a while. Fitting a RF per graph)...")
    family_signals = family.analyze_graph_family_signals()
    
    print("Graph Family Signals:")
    for signal_metric in ['structure_signal', 'feature_signal', 'degree_signal']:
        if signal_metric in family_signals:
            values = family_signals[signal_metric]
            if isinstance(values, (list, np.ndarray)):
                print(f"   {signal_metric}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
            else:
                print(f"   {signal_metric}: {values}")
        else:
            print(f"   {signal_metric}: Not calculated")

    # Calculate Between-graph Community-related Consistency  
    print("\nCalculating between-graph community-related consistency...")
    family_consistency = family.analyze_graph_family_consistency()
    
    print("Graph Family Consistency:")
    for consistency_metric in ['structure_consistency', 'feature_consistency', 'degree_consistency']:
        if consistency_metric in family_consistency:
            values = family_consistency[consistency_metric]
            if isinstance(values, (list, np.ndarray)):
                print(f"   {consistency_metric}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
            else:
                print(f"   {consistency_metric}: {values}")
        else:
            print(f"   {consistency_metric}: Not calculated")

except Exception as e:
    print(f"Validation analysis failed: {e}")

# =============================================================================
# Bonus: VISUALIZATION OF SAMPLE GRAPHS
# =============================================================================

print("\nBonus: Sample Graph Visualization")
print("-" * 40)

try:
    # Select 3 different graphs from the family for publication-quality visualization
    n_graphs_to_show = min(3, len(family.graphs))
    
    if n_graphs_to_show > 0:
        # Create figure with 3 subplots side by side - publication quality
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot each graph with publication-quality settings
        # Note: plot_graph_communities automatically uses consistent colors based on community IDs
        for i in range(n_graphs_to_show):
            graph = family.graphs[i]
            
            # Plot using viz_utils helper function
            # Colors are automatically consistent: community 0 is always the same color, etc.
            plot_graph_communities(
                graph=graph,
                layout="spring",
                node_size=80,           # Larger nodes for better visibility
                edge_width=1.2,         # Thicker edges for publication quality
                edge_alpha=0.5,         # More visible edges
                with_labels=False,
                ax=axes[i],
                title=f"Graph {i+1}",
            )
            
            # Remove all spines/borders to make subplots seamless
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            
            # Improve title font for publication quality
            axes[i].title.set_fontsize(14)
            axes[i].title.set_fontweight('bold')
        
        # Adjust spacing between subplots for better appearance
        plt.subplots_adjust(wspace=0.05, hspace=0)
        
        # Show plot
        plt.show()
    else:
        print("No graphs available to visualize")
    
except Exception as e:
    print(f"Visualization failed: {e}")
    import traceback
    traceback.print_exc()