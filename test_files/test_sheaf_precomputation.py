#!/usr/bin/env python3
"""
Test script for sheaf Laplacian precomputation functionality.
Tests the new precomputation and caching system for sheaf diffusion models.
"""

import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import copy
from torch_geometric.data import DataLoader, Data

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveSheafDiffusionModel
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders
)
from experiments.inductive.data import precompute_sheaf_laplacian, create_sheaf_dataloaders


def graphsample_to_pyg(graph_sample):
    """Convert a GraphSample to a PyTorch Geometric Data object."""
    import networkx as nx
    from torch_geometric.utils import to_undirected
    
    graph = graph_sample.graph
    n_nodes = graph.number_of_nodes()
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # Use features from GraphSample if available, else identity
    if hasattr(graph_sample, 'features') and graph_sample.features is not None:
        features = torch.tensor(graph_sample.features, dtype=torch.float)
    else:
        features = torch.eye(n_nodes, dtype=torch.float)

    # Use community_labels as y
    if hasattr(graph_sample, 'community_labels') and graph_sample.community_labels is not None:
        y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
    else:
        y = torch.zeros(n_nodes, dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=y)
    return data


def load_graphs():
    """Load graphs from graphs.pkl file."""
    print("Loading graphs from graphs.pkl...")
    
    with open('graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError("No graphs found in graphs.pkl")
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs


def create_test_config():
    """Create a test configuration for the experiment."""
    class TestConfig:
        def __init__(self):
            # Basic settings
            self.seed = 42
            self.k_fold = 1  # Single fold for testing
            self.batch_size = 4
            self.train_graph_ratio = 0.6
            self.val_graph_ratio = 0.2
            
            # Tasks
            self.tasks = ["community"]
            self.is_regression = {"community": False}
            
            # Sheaf settings
            self.sheaf_d = 2
            self.sheaf_type = "bundle"
            self.sheaf_normalised = True
            self.sheaf_deg_normalised = False
            self.sheaf_add_hp = False
            self.sheaf_add_lp = False
            self.sheaf_orth = "cayley"

            self.run_transformers = False
            self.precompute_pe = False
            
            # Allow unseen community combinations for simpler testing
            self.allow_unseen_community_combinations_for_eval = True
    
    return TestConfig()


def test_precomputation_function():
    """Test the precompute_sheaf_laplacian function directly."""
    print("\n" + "="*60)
    print("TESTING PRECOMPUTATION FUNCTION")
    print("="*60)
    
    # Load a few graphs
    graphs = load_graphs()
    if len(graphs) < 3:
        print("Need at least 3 graphs for testing")
        return False
    
    # Convert to PyG format
    pyg_graphs = [graphsample_to_pyg(g) for g in graphs[:3]]
    
    # Test sheaf config
    sheaf_config = {
        'd': 2,
        'sheaf_type': 'bundle',
        'normalised': True,
        'deg_normalised': False,
        'add_hp': False,
        'add_lp': False,
        'orth': 'cayley'
    }
    
    print("Testing precomputation on 3 graphs...")
    
    for i, graph in enumerate(pyg_graphs):
        print(f"  Graph {i}: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges")
        
        # Test precomputation
        start_time = time.time()
        graph_with_cache = precompute_sheaf_laplacian(copy.deepcopy(graph))
        precomp_time = time.time() - start_time
        
        print(f"    Precomputation time: {precomp_time:.4f}s")
        
        # Verify cache was created
        if hasattr(graph_with_cache, 'sheaf_indices_cache'):
            cache = graph_with_cache.sheaf_indices_cache
            print(f"    Cache created: {list(cache.keys())}")
            
            # Verify essential cache components
            required_keys = ['size', 'edge_index', 'full_left_right_idx', 'left_right_idx', 'vertex_tril_idx', 'deg']
            missing_keys = [key for key in required_keys if key not in cache]
            if missing_keys:
                print(f"    WARNING: Missing cache keys: {missing_keys}")
                return False
            else:
                print(f"    ✓ All required cache components present")
        else:
            print(f"    ERROR: No cache created")
            return False
    
    print("✓ Precomputation function test passed")
    return True


def test_data_preparation():
    """Test the data preparation with sheaf-specific splits."""
    print("\n" + "="*60)
    print("TESTING DATA PREPARATION")
    print("="*60)
    
    # Load graphs
    graphs = load_graphs()
    if len(graphs) < 10:
        print("Need at least 10 graphs for testing")
        return False
    
    # Create config
    config = create_test_config()
    
    print(f"Preparing data with {len(graphs)} graphs...")
    
    # Test data preparation
    start_time = time.time()
    inductive_data, sheaf_inductive_data, _ = prepare_inductive_data(graphs, config)
    prep_time = time.time() - start_time
    
    # Create sheaf-specific data
    start_time = time.time()
    sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    sheaf_prep_time = time.time() - start_time
    
    print(f"Data preparation completed in {prep_time:.2f}s")
    print(f"Sheaf data preparation completed in {sheaf_prep_time:.2f}s")
    
    # Verify both data structures were created
    if not inductive_data or not sheaf_inductive_data:
        print("ERROR: Data preparation failed")
        return False
    
    print("✓ Both normal and sheaf data structures created")
    
    # Check that splits are identical
    for task in config.tasks:
        if task not in inductive_data or task not in sheaf_inductive_data:
            print(f"ERROR: Task {task} missing from one of the data structures")
            return False
        
        normal_task = inductive_data[task]
        sheaf_task = sheaf_inductive_data[task]
        
        for fold_name in normal_task.keys():
            if fold_name == 'metadata':
                continue
                
            if fold_name not in sheaf_task:
                print(f"ERROR: Fold {fold_name} missing from sheaf data")
                return False
            
            normal_fold = normal_task[fold_name]
            sheaf_fold = sheaf_task[fold_name]
            
            for split_name in ['train', 'val', 'test']:
                if split_name not in normal_fold or split_name not in sheaf_fold:
                    print(f"ERROR: Split {split_name} missing")
                    return False
                
                normal_split = normal_fold[split_name]
                sheaf_split = sheaf_fold[split_name]
                
                # Check that indices are identical
                if normal_split['indices'] != sheaf_split['indices']:
                    print(f"ERROR: Indices differ for {fold_name}/{split_name}")
                    return False
                
                # Check that sheaf graphs have cache
                for graph in sheaf_split['graphs']:
                    if not hasattr(graph, 'sheaf_indices_cache'):
                        print(f"ERROR: Sheaf graph missing cache")
                        return False
                
                # Check that batch sizes are different
                if normal_split['batch_size'] == sheaf_split['batch_size']:
                    print(f"WARNING: Batch sizes are the same for {fold_name}/{split_name}")
                
                print(f"  ✓ {fold_name}/{split_name}: {len(normal_split['graphs'])} graphs, "
                      f"normal batch={normal_split['batch_size']}, sheaf batch={sheaf_split['batch_size']}")
    
    print("✓ Data preparation test passed")
    return True


def test_dataloader_creation():
    """Test dataloader creation for both normal and sheaf data."""
    print("\n" + "="*60)
    print("TESTING DATALOADER CREATION")
    print("="*60)
    
    # Load graphs and prepare data
    graphs = load_graphs()
    config = create_test_config()
    
    inductive_data, sheaf_inductive_data, _ = prepare_inductive_data(graphs, config)
    
    print("Creating normal dataloaders...")
    start_time = time.time()
    normal_dataloaders = create_inductive_dataloaders(inductive_data, config)
    normal_time = time.time() - start_time
    
    print("Creating sheaf dataloaders...")
    start_time = time.time()
    sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    sheaf_time = time.time() - start_time
    
    print(f"Normal dataloader creation: {normal_time:.4f}s")
    print(f"Sheaf dataloader creation: {sheaf_time:.4f}s")
    
    # Verify dataloaders were created
    if not normal_dataloaders or not sheaf_dataloaders:
        print("ERROR: Dataloader creation failed")
        return False
    
    # Check batch sizes
    for task in config.tasks:
        normal_task = normal_dataloaders[task]
        sheaf_task = sheaf_dataloaders[task]
        
        for fold_name in normal_task.keys():
            normal_fold = normal_task[fold_name]
            sheaf_fold = sheaf_task[fold_name]
            
            for split_name in ['train', 'val', 'test']:
                normal_loader = normal_fold[split_name]
                sheaf_loader = sheaf_fold[split_name]
                
                # Check that sheaf loader has batch size 1
                if sheaf_loader.batch_size != 1:
                    print(f"ERROR: Sheaf loader for {fold_name}/{split_name} has batch size {sheaf_loader.batch_size}, expected 1")
                    return False
                
                print(f"  ✓ {fold_name}/{split_name}: normal batch={normal_loader.batch_size}, sheaf batch={sheaf_loader.batch_size}")
    
    print("✓ Dataloader creation test passed")
    return True


def test_model_with_precomputation():
    """Test the sheaf model with precomputed data."""
    print("\n" + "="*60)
    print("TESTING MODEL WITH PRECOMPUTATION")
    print("="*60)
    
    # Load graphs and prepare data
    graphs = load_graphs()
    config = create_test_config()
    
    inductive_data, sheaf_inductive_data, _ = prepare_inductive_data(graphs, config)
    normal_dataloaders = create_inductive_dataloaders(inductive_data, config)
    sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    
    # Get model parameters
    first_graph = inductive_data['community']['fold_0']['train']['graphs'][0]
    input_dim = first_graph.x.size(1)
    hidden_dim = 32  # Smaller for faster testing
    # Use the correct output dimension from metadata
    output_dim = inductive_data['community']['metadata']['output_dim']
    
    print(f"Model config: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    # Create sheaf model
    model = InductiveSheafDiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        sheaf_type=config.sheaf_type,
        d=config.sheaf_d,
        num_layers=2,
        dropout=0.1,
        input_dropout=0.1,
        is_regression=False,
        is_graph_level_task=False,
        device='cpu',
        normalised=config.sheaf_normalised,
        deg_normalised=config.sheaf_deg_normalised,
        linear=False,
        left_weights=True,
        right_weights=True,
        sparse_learner=False,
        use_act=True,
        sheaf_act="tanh",
        second_linear=False,
        orth=config.sheaf_orth,
        edge_weights=False,
        max_t=1.0,
        add_lp=config.sheaf_add_lp,
        add_hp=config.sheaf_add_hp
    )
    
    print("Testing forward pass with precomputed data...")
    
    # Test with a single graph
    test_graph = sheaf_inductive_data['community']['fold_0']['train']['graphs'][0]
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        out = model(test_graph.x, test_graph.edge_index, graph=test_graph)
        forward_time = time.time() - start_time
    
    print(f"Forward pass completed in {forward_time:.4f}s")
    print(f"Output shape: {out.shape}")
    
    # Verify output is reasonable
    if out.shape[0] != test_graph.x.size(0) or out.shape[1] != output_dim:
        print(f"ERROR: Unexpected output shape {out.shape}")
        return False
    
    print("✓ Model forward pass test passed")
    return True


def test_training_comparison():
    """Compare training speed with and without precomputation."""
    print("\n" + "="*60)
    print("TRAINING SPEED COMPARISON")
    print("="*60)
    
    # Load graphs and prepare data
    graphs = load_graphs()
    config = create_test_config()
    
    inductive_data, sheaf_inductive_data, _ = prepare_inductive_data(graphs, config)
    normal_dataloaders = create_inductive_dataloaders(inductive_data, config)
    sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    
    # Get model parameters
    first_graph = inductive_data['community']['fold_0']['train']['graphs'][0]
    input_dim = first_graph.x.size(1)
    hidden_dim = 32
    # Use the correct output dimension from metadata
    output_dim = inductive_data['community']['metadata']['output_dim']
    
    print(f"Training comparison with {len(inductive_data['community']['fold_0']['train']['graphs'])} train graphs")
    
    # Test 1: Training with normal dataloaders (no precomputation)
    print("\nTesting training with normal dataloaders (no precomputation)...")
    
    model1 = InductiveSheafDiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        sheaf_type=config.sheaf_type,
        d=config.sheaf_d,
        num_layers=2,
        dropout=0.1,
        input_dropout=0.1,
        is_regression=False,
        is_graph_level_task=False,
        device='gpu',
        normalised=config.sheaf_normalised,
        deg_normalised=config.sheaf_deg_normalised,
        linear=False,
        left_weights=True,
        right_weights=True,
        sparse_learner=False,
        use_act=True,
        sheaf_act="tanh",
        second_linear=False,
        orth=config.sheaf_orth,
        edge_weights=False,
        max_t=1.0,
        add_lp=config.sheaf_add_lp,
        add_hp=config.sheaf_add_hp
    )
    
    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    train_loader1 = normal_dataloaders['community']['fold_0']['train']
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    
    model1.train()
    start_time = time.time()
    
    for epoch in range(epochs):  # Just 10 epochs for testing
        total_loss, total_accuracy = 0, 0
        for batch in train_loader1:
            optimizer1.zero_grad()
            out = model1(batch.x, batch.edge_index)
            # Multiclass classification loss multi-class classification loss
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer1.step()
            total_loss += loss.item()

            # Calculate accuracy
            preds = out.argmax(dim=1)
            accuracy = (preds == batch.y).float().mean()
            total_accuracy += accuracy.item()
        
        print(f"  Epoch {epoch}: Loss = {total_loss/len(train_loader1):.4f}")
        print(f"  Epoch {epoch}: Accuracy = {total_accuracy/len(train_loader1):.4f}")
    
    normal_training_time = time.time() - start_time
    print(f"Normal training time: {normal_training_time:.2f}s")
    
    # Test 2: Training with sheaf dataloaders (with precomputation)
    print("\nTesting training with sheaf dataloaders (with precomputation)...")
    
    model2 = InductiveSheafDiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        sheaf_type=config.sheaf_type,
        d=config.sheaf_d,
        num_layers=2,
        dropout=0.1,
        input_dropout=0.1,
        is_regression=False,
        is_graph_level_task=False,
        device='gpu',
        normalised=config.sheaf_normalised,
        deg_normalised=config.sheaf_deg_normalised,
        linear=False,
        left_weights=True,
        right_weights=True,
        sparse_learner=False,
        use_act=True,
        sheaf_act="tanh",
        second_linear=False,
        orth=config.sheaf_orth,
        edge_weights=False,
        max_t=1.0,
        add_lp=config.sheaf_add_lp,
        add_hp=config.sheaf_add_hp
    )
    
    optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
    train_loader2 = sheaf_dataloaders['community']['fold_0']['train']
    criterion = torch.nn.CrossEntropyLoss()

    model2.train()
    start_time = time.time()
    
    for epoch in range(epochs): 
        total_loss, total_accuracy = 0, 0
        for batch in train_loader2:
            optimizer2.zero_grad()
            # Pass the graph object for precomputed data
            out = model2(batch.x, batch.edge_index, graph=batch)
            # Multiclass classification loss
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()

            # Calculate accuracy
            preds = out.argmax(dim=1)
            accuracy = (preds == batch.y).float().mean()
            total_accuracy += accuracy.item()
        
        print(f"  Epoch {epoch}: Loss = {total_loss/len(train_loader2):.4f}")
        print(f"  Epoch {epoch}: Accuracy = {total_accuracy/len(train_loader2):.4f}")
    
    sheaf_training_time = time.time() - start_time
    print(f"Sheaf training time: {sheaf_training_time:.2f}s")
    
    # Compare speeds
    speedup = normal_training_time / sheaf_training_time if sheaf_training_time > 0 else float('inf')
    print(f"\nSpeed comparison:")
    print(f"  Normal training: {normal_training_time:.2f}s")
    print(f"  Sheaf training: {sheaf_training_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("✓ Precomputation provides speedup")
    else:
        print("⚠ Precomputation does not provide speedup (this might be expected for small graphs)")
    
    return True


def main():
    """Main test function."""
    print("=" * 80)
    print("SHEAF LAPLACIAN PRECOMPUTATION TEST SUITE")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        ("Precomputation Function", test_precomputation_function),
        ("Data Preparation", test_data_preparation),
        ("Dataloader Creation", test_dataloader_creation),
        ("Model with Precomputation", test_model_with_precomputation),
        ("Training Speed Comparison", test_training_comparison),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sheaf precomputation functionality is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 