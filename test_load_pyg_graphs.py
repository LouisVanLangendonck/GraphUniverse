#!/usr/bin/env python3
"""
Simple test script to load and verify saved PyG graphs and universe.
"""

import pickle
import os
from pathlib import Path

def test_pyg_graphs(family_dir: str, family_id: str):
    """Test loading PyG graphs and universe from saved files."""
    
    family_path = Path(family_dir) / family_id
    print(f"Testing PyG graphs in: {family_path}")
    
    if not family_path.exists():
        print(f"❌ Directory {family_path} does not exist!")
        return
    
    # List all files in the directory
    files = list(family_path.glob("*.pkl"))
    print(f"Found {len(files)} pickle files:")
    for file in files:
        print(f"  - {file.name}")
    
    # Load universe
    universe_file = family_path / "graph_universe.pkl"
    if universe_file.exists():
        print(f"\n📊 Loading universe from: {universe_file}")
        with open(universe_file, "rb") as f:
            universe = pickle.load(f)
        
        print(f"✅ Universe loaded successfully!")
        print(f"   - K (communities): {universe.K}")
        print(f"   - Feature dim: {universe.feature_dim}")
        print(f"   - P matrix shape: {universe.P.shape}")
        print(f"   - Has feature_generator: {hasattr(universe, 'feature_generator') and universe.feature_generator is not None}")
        print(f"   - Degree centers shape: {universe.degree_centers.shape}")
        print(f"   - Edge probability variance: {universe.edge_probability_variance}")
    else:
        print(f"❌ Universe file not found: {universe_file}")
    
    # Load PyG graph files
    pyg_files = [f for f in files if f.name.startswith("pyg_graph_list_")]
    
    for pyg_file in pyg_files:
        # Extract task name from filename
        task_name = pyg_file.stem.replace("pyg_graph_list_", "")
        
        print(f"\n🔗 Loading PyG graphs for task: {task_name}")
        print(f"   File: {pyg_file}")
        
        try:
            with open(pyg_file, "rb") as f:
                graph_list = pickle.load(f)
            
            print(f"✅ Loaded {len(graph_list)} PyG Data objects")
            
            if len(graph_list) > 0:
                # Check first graph
                first_graph = graph_list[0]
                print(f"   First graph properties:")
                print(f"     - Type: {type(first_graph)}")
                print(f"     - Has edge_index: {hasattr(first_graph, 'edge_index')}")
                print(f"     - Has x (features): {hasattr(first_graph, 'x')}")
                print(f"     - Has y (labels): {hasattr(first_graph, 'y')}")
                
                if hasattr(first_graph, 'edge_index'):
                    print(f"     - Edge index shape: {first_graph.edge_index.shape}")
                
                if hasattr(first_graph, 'x'):
                    print(f"     - Features shape: {first_graph.x.shape}")
                
                if hasattr(first_graph, 'y'):
                    print(f"     - Labels shape: {first_graph.y.shape}")
                    print(f"     - Labels type: {type(first_graph.y)}")
                
                # Check a few more graphs
                if len(graph_list) > 1:
                    print(f"   Checking consistency across graphs...")
                    consistent = True
                    for i, graph in enumerate(graph_list[:3]):  # Check first 3
                        if not (hasattr(graph, 'edge_index') and hasattr(graph, 'x') and hasattr(graph, 'y')):
                            print(f"     ❌ Graph {i} missing required attributes")
                            consistent = False
                    
                    if consistent:
                        print(f"     ✅ First 3 graphs have consistent structure")
                
        except Exception as e:
            print(f"❌ Error loading {pyg_file}: {e}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    # Default values - modify these to match your saved files
    family_dir = "graph_family"
    family_id = "family_001"
    
    # You can also pass them as command line arguments
    import sys
    if len(sys.argv) > 1:
        family_dir = sys.argv[1]
    if len(sys.argv) > 2:
        family_id = sys.argv[2]
    
    print("🔍 PyG Graph Loading Test")
    print("=" * 40)
    
    test_pyg_graphs(family_dir, family_id)
    
    print("\n" + "=" * 40)
    print("Usage: python test_load_pyg_graphs.py [family_dir] [family_id]")
    print(f"Current: python test_load_pyg_graphs.py {family_dir} {family_id}")
