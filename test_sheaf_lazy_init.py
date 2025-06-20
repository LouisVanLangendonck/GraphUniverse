#!/usr/bin/env python3
"""
Quick test to verify SheafDiffusionModel lazy initialization fix.
"""

import torch
import sys
import os

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.models import SheafDiffusionModel

def test_lazy_initialization():
    """Test that SheafDiffusionModel properly handles lazy initialization."""
    print("=== Testing SheafDiffusionModel Lazy Initialization ===")
    
    # Create a simple test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    node_features = torch.randn(3, 4)
    
    # Create model
    model = SheafDiffusionModel(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        sheaf_type="diag",
        d=2,
        num_layers=2,
        dropout=0.1,
        is_regression=False,
        is_graph_level_task=False,
        device='cpu'
    )
    
    print(f"✓ Model created successfully")
    print(f"  Initial parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass (this should initialize the sheaf model)
    output = model(node_features, edge_index)
    
    print(f"✓ Forward pass successful")
    print(f"  Parameters after forward pass: {sum(p.numel() for p in model.parameters())}")
    print(f"  Output shape: {output.shape}")
    
    # Test that we can create an optimizer now
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"✓ Optimizer created successfully")
    print(f"  Optimizer parameter groups: {len(optimizer.param_groups)}")
    
    # Test training step
    labels = torch.randint(0, 3, (3,))
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✅ Lazy initialization test passed!")

if __name__ == "__main__":
    test_lazy_initialization() 