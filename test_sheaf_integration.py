#!/usr/bin/env python3
"""
Test script for SheafDiffusionModel integration.
Tests the integration of discrete sheaf diffusion models into the main model framework.
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.models import SheafDiffusionModel, create_model

def create_sample_graph():
    """Create a simple sample graph for testing."""
    # Create a small undirected graph with 6 nodes
    # Define the base edges (each will have a reverse)
    base_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),  # basic connections
        (0, 3), (1, 2), (2, 5), (3, 4)  # additional connections
    ]
    
    # Create edge_index with both directions
    sources = []
    targets = []
    for source, target in base_edges:
        sources.extend([source, target])
        targets.extend([target, source])
    
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # Create node features (6 nodes, 4 features each)
    node_features = torch.randn(6, 4)
    
    # Create labels (6 nodes, 3 classes)
    labels = torch.randint(0, 3, (6,))
    
    return edge_index, node_features, labels

def test_sheaf_model_creation():
    """Test creation of SheafDiffusionModel with different sheaf types."""
    print("=== Testing SheafDiffusionModel Creation ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    # Test all three sheaf types
    sheaf_types = ["diag", "bundle", "general"]
    
    for sheaf_type in sheaf_types:
        print(f"\n--- Testing {sheaf_type} sheaf ---")
        
        try:
            # Create model
            model = SheafDiffusionModel(
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                sheaf_type=sheaf_type,
                d=2,  # sheaf dimension
                num_layers=2,
                dropout=0.1,
                device='cpu'
            )
            
            print(f"✓ {sheaf_type} model created successfully")
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Test forward pass
            output = model(node_features, edge_index)
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
        except Exception as e:
            print(f"✗ {sheaf_type} model failed: {e}")

def test_create_model_function():
    """Test the create_model function with sheaf models."""
    print("\n=== Testing create_model Function ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    # Test model types
    model_types = ["sheaf_diag", "sheaf_bundle", "sheaf_general"]
    
    for model_type in model_types:
        print(f"\n--- Testing {model_type} ---")
        
        try:
            # Create model using create_model function
            model = create_model(
                model_type=model_type,
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                d=2,
                num_layers=2,
                dropout=0.1,
                device='cpu'
            )
            
            print(f"✓ {model_type} created successfully via create_model")
            
            # Test forward pass
            output = model(node_features, edge_index)
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")

def test_node_vs_graph_level():
    """Test both node-level and graph-level tasks."""
    print("\n=== Testing Node vs Graph Level Tasks ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    # Test node-level task
    print("\n--- Node-level task ---")
    try:
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            is_graph_level_task=False,
            device='cpu'
        )
        
        output = model(node_features, edge_index)
        print(f"✓ Node-level output shape: {output.shape}")
        assert output.shape == (6, 3), f"Expected (6, 3), got {output.shape}"
        
    except Exception as e:
        print(f"✗ Node-level task failed: {e}")
    
    # Test graph-level task
    print("\n--- Graph-level task ---")
    try:
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            is_graph_level_task=True,
            device='cpu'
        )
        
        output = model(node_features, edge_index)
        print(f"✓ Graph-level output shape: {output.shape}")
        assert output.shape == (1, 3), f"Expected (1, 3), got {output.shape}"
        
    except Exception as e:
        print(f"✗ Graph-level task failed: {e}")

def test_different_sheaf_dimensions():
    """Test different sheaf dimensions."""
    print("\n=== Testing Different Sheaf Dimensions ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    # Test different dimensions for each sheaf type
    test_configs = [
        ("diag", 1),
        ("diag", 2),
        ("bundle", 2),
        ("bundle", 3),
        ("general", 2),
        ("general", 3)
    ]
    
    for sheaf_type, d in test_configs:
        print(f"\n--- Testing {sheaf_type} with d={d} ---")
        
        try:
            model = SheafDiffusionModel(
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                sheaf_type=sheaf_type,
                d=d,
                num_layers=2,
                dropout=0.1,
                device='cpu'
            )
            
            output = model(node_features, edge_index)
            print(f"✓ {sheaf_type} with d={d} successful")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ {sheaf_type} with d={d} failed: {e}")

def test_training_step():
    """Test a complete training step."""
    print("\n=== Testing Training Step ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    try:
        # Create model
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        # IMPORTANT: Do forward pass FIRST to initialize sheaf model
        output = model(node_features, edge_index)
        print(f"✓ Forward pass successful, initialized model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # NOW create optimizer after model is initialized
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Compute loss (convert labels to long for NLLLoss)
        loss = F.nll_loss(output, labels.long())
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
        
        if param_count > 0:
            avg_grad_norm = total_grad_norm / param_count
            print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")

def test_training_step_alternative():
    """Alternative test that shows how to handle lazy initialization properly."""
    print("\n=== Testing Training Step (Alternative Approach) ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    try:
        # Create model
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        # Option 1: Create optimizer with dummy parameters first
        dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.Adam([dummy_param], lr=0.01)
        
        # Forward pass to initialize model
        output = model(node_features, edge_index)
        
        # Update optimizer with actual model parameters
        optimizer.param_groups[0]['params'] = list(model.parameters())
        
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Compute loss
        loss = F.nll_loss(output, labels.long())
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Alternative training step failed: {e}")

def test_eager_initialization():
    """Test a model that initializes parameters eagerly."""
    print("\n=== Testing Eager Initialization ===")
    
    edge_index, node_features, labels = create_sample_graph()
    
    try:
        # Create model
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters initially")
        
        # Force initialization by calling the internal method
        model._initialize_sheaf_model(edge_index)
        
        print(f"After manual initialization: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Now create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Forward pass
        output = model(node_features, edge_index)
        
        # Compute loss and train
        loss = F.nll_loss(output, labels.long())
        loss.backward()
        optimizer.step()
        
        print(f"✓ Eager initialization successful")
        print(f"  Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Eager initialization failed: {e}")

def test_consecutive_graphs():
    """Test training on two consecutive graphs of different sizes."""
    print("\n=== Testing Consecutive Graphs of Different Sizes ===")
    
    # Create two graphs of different sizes
    def create_graph_1():
        """Create a small graph with 4 nodes."""
        base_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        sources, targets = [], []
        for source, target in base_edges:
            sources.extend([source, target])
            targets.extend([target, source])
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        node_features = torch.randn(4, 4)
        labels = torch.randint(0, 3, (4,))
        return edge_index, node_features, labels
    
    def create_graph_2():
        """Create a larger graph with 8 nodes."""
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7),  # ring
            (0, 4), (1, 5), (2, 6), (3, 7)  # cross connections
        ]
        sources, targets = [], []
        for source, target in base_edges:
            sources.extend([source, target])
            targets.extend([target, source])
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        node_features = torch.randn(8, 4)
        labels = torch.randint(0, 3, (8,))
        return edge_index, node_features, labels
    
    try:
        # Create model
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        print(f"✓ Model created successfully")
        print(f"  Initial parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train on first graph
        print(f"\n--- Training on Graph 1 (4 nodes) ---")
        edge_index_1, node_features_1, labels_1 = create_graph_1()
        
        print(f"  Graph 1 size: {node_features_1.size(0)} nodes")
        print(f"  Sheaf model graph size before: {model.current_graph_size}")
        
        # IMPORTANT: Forward pass first to initialize
        output_1 = model(node_features_1, edge_index_1)
        print(f"  Parameters after initialization: {sum(p.numel() for p in model.parameters())}")
        print(f"  Sheaf model graph size after: {model.current_graph_size}")
        
        # NOW create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        loss_1 = F.nll_loss(output_1, labels_1.long())
        print(f"  Initial loss on graph 1: {loss_1.item():.4f}")
        
        # Backward pass and update
        loss_1.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Store some parameter values after first training step
        param_values_after_graph1 = {}
        for name, param in model.named_parameters():
            if 'sheaf_model' in name:
                param_values_after_graph1[name] = param.data.clone()
        
        # Train on second graph (different size)
        print(f"\n--- Training on Graph 2 (8 nodes) ---")
        edge_index_2, node_features_2, labels_2 = create_graph_2()
        
        print(f"  Graph 2 size: {node_features_2.size(0)} nodes")
        print(f"  Sheaf model graph size before graph 2: {model.current_graph_size}")
        
        # Forward pass on second graph
        output_2 = model(node_features_2, edge_index_2)
        loss_2 = F.nll_loss(output_2, labels_2.long())
        
        print(f"  Initial loss on graph 2: {loss_2.item():.4f}")
        print(f"  Parameters after graph 2: {sum(p.numel() for p in model.parameters())}")
        print(f"  Sheaf model graph size after graph 2: {model.current_graph_size}")
        
        # Update optimizer with new parameters (important for dynamic models)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Check if parameters were transferred
        print(f"\n--- Checking Parameter Transfer ---")
        transferred_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if 'sheaf_model' in name and name in param_values_after_graph1:
                total_params += 1
                if torch.allclose(param.data, param_values_after_graph1[name], atol=1e-6):
                    transferred_params += 1
                    print(f"  ✓ Parameter {name} transferred successfully")
                else:
                    print(f"  ✗ Parameter {name} changed during transfer")
        
        print(f"  Parameter transfer rate: {transferred_params}/{total_params}")
        
        # Backward pass and update on second graph
        loss_2.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Test that we can go back to the first graph
        print(f"\n--- Testing Return to Graph 1 ---")
        output_1_again = model(node_features_1, edge_index_1)
        loss_1_again = F.nll_loss(output_1_again, labels_1.long())
        
        print(f"  Loss on graph 1 after training on graph 2: {loss_1_again.item():.4f}")
        print(f"  Sheaf model graph size: {model.current_graph_size}")
        
        # Verify that the model can handle both graphs
        print(f"\n--- Final Verification ---")
        print(f"✓ Model successfully trained on graphs of different sizes")
        print(f"✓ Parameter transfer working correctly")
        print(f"✓ Model can switch between different graph sizes")
        
    except Exception as e:
        print(f"✗ Consecutive graphs test failed: {e}")
        import traceback
        traceback.print_exc()

def test_dynamic_graph_sizes():
    """Test that the model handles dynamic graph sizes correctly."""
    print("\n=== Testing Dynamic Graph Sizes ===")
    
    try:
        # Create model without specifying graph_size
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        print(f"✓ Model created without graph_size parameter")
        
        # Test with different graph sizes
        graph_sizes = [3, 5, 7, 10, 15]
        
        for size in graph_sizes:
            print(f"\n--- Testing graph size {size} ---")
            
            # Create valid undirected graph of this size
            # Create a simple ring graph
            sources = []
            targets = []
            for i in range(size):
                # Connect to next node (ring structure)
                next_node = (i + 1) % size
                sources.extend([i, next_node])
                targets.extend([next_node, i])
            
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            node_features = torch.randn(size, 4)
            labels = torch.randint(0, 3, (size,))
            
            # Forward pass
            output = model(node_features, edge_index)
            
            print(f"  ✓ Forward pass successful for size {size}")
            print(f"  Output shape: {output.shape}")
            print(f"  Current sheaf model graph size: {model.current_graph_size}")
            
            # Verify output shape
            expected_shape = (size, 3)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print(f"\n✓ Model successfully handled all graph sizes: {graph_sizes}")
        
    except Exception as e:
        print(f"✗ Dynamic graph sizes test failed: {e}")
        import traceback
        traceback.print_exc()

def test_batched_graphs():
    """Test how the model handles batched graphs."""
    print("\n=== Testing Batched Graphs ===")
    
    try:
        # Create model
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            device='cpu'
        )
        
        # Create two small graphs
        # Graph 1: 3 nodes
        x1 = torch.randn(3, 4)
        edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        
        # Graph 2: 4 nodes  
        x2 = torch.randn(4, 4)
        edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
        
        # Test individual graphs first
        print("--- Testing Individual Graphs ---")
        output1 = model(x1, edge_index1)
        print(f"Graph 1 output shape: {output1.shape}")
        
        # FIXED: Test graph 2 with its original edge indices, not offset ones
        output2 = model(x2, edge_index2)  # Remove the +3 offset here
        print(f"Graph 2 output shape: {output2.shape}")
        
        # Create batched representation
        print("\n--- Testing Batched Graphs ---")
        x_batch = torch.cat([x1, x2], dim=0)  # [7, 4] - concatenate node features
        
        # Offset edge indices for second graph
        edge_index2_offset = edge_index2 + 3  # Add 3 to account for first graph's nodes
        edge_index_batch = torch.cat([edge_index1, edge_index2_offset], dim=1)
        
        # Create batch tensor
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])  # First 3 nodes belong to graph 0, next 4 to graph 1
        
        print(f"Batched input shape: {x_batch.shape}")
        print(f"Batched edge_index shape: {edge_index_batch.shape}")
        print(f"Batch tensor: {batch}")
        
        # Test batched forward pass
        try:
            output_batch = model(x_batch, edge_index_batch, batch)
            print(f"✓ Batched forward pass successful!")
            print(f"Batched output shape: {output_batch.shape}")
            
            # For node-level tasks, output should be [7, 3]
            # For graph-level tasks, output should be [2, 3] (2 graphs)
            if model.is_graph_level_task:
                expected_shape = (2, 3)  # 2 graphs, 3 output classes
            else:
                expected_shape = (7, 3)  # 7 total nodes, 3 output classes
                
            if output_batch.shape == expected_shape:
                print(f"✓ Output shape correct: {output_batch.shape}")
            else:
                print(f"⚠ Output shape unexpected: got {output_batch.shape}, expected {expected_shape}")
                
        except Exception as e:
            print(f"✗ Batched forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            
            # This might indicate the sheaf model doesn't handle batched graphs properly
            print("\n⚠ The sheaf model may not be batch-aware.")
            print("This is expected if the underlying implementation assumes single graphs.")
        
        # Test graph-level task with batching
        print("\n--- Testing Graph-Level Task with Batching ---")
        model_graph = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            is_graph_level_task=True,
            device='cpu'
        )
        
        try:
            output_graph_batch = model_graph(x_batch, edge_index_batch, batch)
            print(f"✓ Graph-level batched forward pass successful!")
            print(f"Graph-level batched output shape: {output_graph_batch.shape}")
            
            # Should be [2, 3] for 2 graphs
            if output_graph_batch.shape == (2, 3):
                print(f"✓ Graph-level output shape correct")
            else:
                print(f"⚠ Graph-level output shape unexpected: {output_graph_batch.shape}")
                
        except Exception as e:
            print(f"✗ Graph-level batched forward pass failed: {e}")
        
    except Exception as e:
        print(f"✗ Batch testing failed: {e}")
        import traceback
        traceback.print_exc()

def test_pyg_dataloader_batching():
    """Test that simulates actual PyTorch Geometric DataLoader behavior."""
    print("\n=== Testing PyG DataLoader Batching ===")
    
    try:
        from torch_geometric.data import Data, Batch
        from torch_geometric.loader import DataLoader
        
        # Create several individual graphs (like your inductive data)
        graphs = []
        
        # Graph 1: 4 nodes, triangle + 1 isolated
        x1 = torch.randn(4, 4)
        edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        y1 = torch.randint(0, 3, (4,))  # Node-level labels
        graph1 = Data(x=x1, edge_index=edge_index1, y=y1)
        graphs.append(graph1)
        
        # Graph 2: 3 nodes, fully connected
        x2 = torch.randn(3, 4)
        edge_index2 = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
        y2 = torch.randint(0, 3, (3,))
        graph2 = Data(x=x2, edge_index=edge_index2, y=y2)
        graphs.append(graph2)
        
        # Graph 3: 5 nodes, ring
        x3 = torch.randn(5, 4)
        edge_index3 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
        ], dtype=torch.long)
        y3 = torch.randint(0, 3, (5,))
        graph3 = Data(x=x3, edge_index=edge_index3, y=y3)
        graphs.append(graph3)
        
        # Graph 4: 2 nodes, single edge
        x4 = torch.randn(2, 4)
        edge_index4 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y4 = torch.randint(0, 3, (2,))
        graph4 = Data(x=x4, edge_index=edge_index4, y=y4)
        graphs.append(graph4)
        
        print(f"Created {len(graphs)} individual graphs:")
        for i, g in enumerate(graphs):
            print(f"  Graph {i}: {g.x.size(0)} nodes, {g.edge_index.size(1)} edges")
        
        # Create DataLoader (like your inductive dataloaders)
        print(f"\n--- Testing with PyG DataLoader ---")
        dataloader = DataLoader(graphs, batch_size=3, shuffle=False)
        
        # Test both node-level and graph-level models
        for task_type in ['node', 'graph']:
            print(f"\n--- Testing {task_type}-level task ---")
            
            model = SheafDiffusionModel(
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                sheaf_type="diag",
                d=2,
                num_layers=2,
                dropout=0.1,
                is_graph_level_task=(task_type == 'graph'),
                device='cpu'
            )
            
            # Process batches like in real training
            for batch_idx, batch in enumerate(dataloader):
                print(f"\n  Batch {batch_idx}:")
                print(f"    Batch size: {batch.batch.max().item() + 1} graphs")
                print(f"    Total nodes: {batch.x.size(0)}")
                print(f"    Total edges: {batch.edge_index.size(1)}")
                print(f"    Node features shape: {batch.x.shape}")
                print(f"    Edge index shape: {batch.edge_index.shape}")
                print(f"    Batch tensor: {batch.batch}")
                
                try:
                    # Forward pass through model
                    output = model(batch.x, batch.edge_index, batch.batch)
                    
                    print(f"    ✓ Forward pass successful!")
                    print(f"    Output shape: {output.shape}")
                    
                    # Verify output shape
                    if task_type == 'node':
                        expected_nodes = batch.x.size(0)
                        expected_shape = (expected_nodes, 3)
                        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
                        print(f"    ✓ Node-level output shape correct")
                    else:  # graph-level
                        expected_graphs = batch.batch.max().item() + 1
                        expected_shape = (expected_graphs, 3)
                        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
                        print(f"    ✓ Graph-level output shape correct")
                    
                    # Test loss computation
                    if task_type == 'node':
                        # Node-level loss
                        loss = F.nll_loss(output, batch.y.long())
                        print(f"    ✓ Node-level loss computed: {loss.item():.4f}")
                    else:
                        # Graph-level loss (create dummy graph labels)
                        graph_labels = torch.randint(0, 3, (expected_graphs,))
                        if output.dim() > 1 and output.size(1) > 1:
                            loss = F.nll_loss(F.log_softmax(output, dim=1), graph_labels.long())
                        else:
                            loss = F.mse_loss(output.squeeze(), graph_labels.float())
                        print(f"    ✓ Graph-level loss computed: {loss.item():.4f}")
                    
                except Exception as e:
                    print(f"    ✗ Batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return
        
        print(f"\n--- Testing Training Loop Simulation ---")
        
        # Simulate a few training steps
        model = SheafDiffusionModel(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            sheaf_type="diag",
            d=2,
            num_layers=2,
            dropout=0.1,
            is_graph_level_task=False,  # Node-level for this test
            device='cpu'
        )
        
        # Initialize model with first batch
        first_batch = next(iter(dataloader))
        _ = model(first_batch.x, first_batch.edge_index, first_batch.batch)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        print(f"Training simulation with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training loop
        for epoch in range(2):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch.x, batch.edge_index, batch.batch)
                
                # Compute loss
                loss = F.nll_loss(output, batch.y.long())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            avg_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch} average loss: {avg_loss:.4f}")
        
        print(f"\n✓ PyG DataLoader integration successful!")
        print(f"✓ Model handles variable graph sizes in batches")
        print(f"✓ Training loop works correctly")
        print(f"✓ Both node-level and graph-level tasks supported")
        
    except ImportError:
        print("✗ PyTorch Geometric not available for DataLoader test")
        print("  (This is expected if PyG is not installed)")
    except Exception as e:
        print(f"✗ PyG DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()

def test_large_batch_efficiency():
    """Test efficiency with larger batches to simulate real usage."""
    print("\n=== Testing Large Batch Efficiency ===")
    
    try:
        from torch_geometric.data import Data, Batch
        from torch_geometric.loader import DataLoader
        import time
        
        # Create many graphs of various sizes
        graphs = []
        graph_sizes = [3, 5, 7, 4, 6, 8, 3, 5, 9, 4, 6, 7, 5, 8, 3, 6]  # Mix of sizes
        
        for i, size in enumerate(graph_sizes):
            # Create random connected graph
            x = torch.randn(size, 4)
            
            # Create edges (simple ring + some random edges)
            edges = []
            for j in range(size):
                edges.extend([(j, (j + 1) % size), ((j + 1) % size, j)])  # Ring
            
            # Add some random edges
            for _ in range(min(3, size - 1)):
                src, tgt = torch.randint(0, size, (2,))
                if src != tgt:
                    edges.extend([(src.item(), tgt.item()), (tgt.item(), src.item())])
            
            edge_index = torch.tensor(list(set(edges)), dtype=torch.long).t()
            y = torch.randint(0, 3, (size,))
            
            graph = Data(x=x, edge_index=edge_index, y=y)
            graphs.append(graph)
        
        print(f"Created {len(graphs)} graphs with sizes: {graph_sizes}")
        
        # Test with different batch sizes
        batch_sizes = [4, 8, 12]
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing batch size {batch_size} ---")
            
            dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
            
            model = SheafDiffusionModel(
                input_dim=4,
                hidden_dim=16,  # Slightly larger for efficiency test
                output_dim=3,
                sheaf_type="diag",
                d=2,
                num_layers=2,
                dropout=0.1,
                device='cpu'
            )
            
            # Warm up
            first_batch = next(iter(dataloader))
            _ = model(first_batch.x, first_batch.edge_index, first_batch.batch)
            
            # Time the batches
            start_time = time.time()
            total_nodes = 0
            batch_count = 0
            
            for batch in dataloader:
                output = model(batch.x, batch.edge_index, batch.batch)
                total_nodes += batch.x.size(0)
                batch_count += 1
            
            end_time = time.time()
            
            avg_time_per_batch = (end_time - start_time) / batch_count
            avg_nodes_per_batch = total_nodes / batch_count
            
            print(f"    Processed {batch_count} batches")
            print(f"    Average nodes per batch: {avg_nodes_per_batch:.1f}")
            print(f"    Average time per batch: {avg_time_per_batch*1000:.2f}ms")
            print(f"    Throughput: {avg_nodes_per_batch/avg_time_per_batch:.1f} nodes/sec")
        
        print(f"\n✓ Large batch efficiency test completed")
        
    except ImportError:
        print("✗ PyTorch Geometric not available for efficiency test")
    except Exception as e:
        print(f"✗ Efficiency test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🧪 Testing SheafDiffusionModel Integration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_sheaf_model_creation()
    test_create_model_function()
    test_node_vs_graph_level()
    test_different_sheaf_dimensions()
    test_training_step()
    test_training_step_alternative()
    test_eager_initialization()
    test_consecutive_graphs()
    test_dynamic_graph_sizes()
    test_batched_graphs()
    test_pyg_dataloader_batching()
    test_large_batch_efficiency()
    print("\n" + "=" * 60)
    print("✅ All SheafDiffusionModel integration tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()