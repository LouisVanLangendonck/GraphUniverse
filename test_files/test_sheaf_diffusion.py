#!/usr/bin/env python3
"""
Simple test script for neural sheaf diffusion models.
Loads graphs from graphs.pkl and runs transductive community prediction.
"""

import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.models import SheafDiffusionModel


def graphsample_to_pyg(graph_sample):
    """
    Convert a GraphSample to a PyTorch Geometric Data object.
    Uses node features from the GraphSample if available, otherwise uses identity features.
    Labels are set to community_labels.
    """
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


def load_first_graph():
    """Load the first graph from graphs.pkl file and convert to PyG format."""
    print("Loading graphs from graphs.pkl...")
    
    with open('graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError("No graphs found in graphs.pkl")
    
    # Get the first graph (GraphSample object)
    graph_sample = graphs[0]
    print(f"Loaded GraphSample with {graph_sample.graph.number_of_nodes()} nodes")
    
    # Convert to PyG format
    pyg_data = graphsample_to_pyg(graph_sample)
    print(f"Converted to PyG format: {pyg_data.x.size(0)} nodes, {pyg_data.x.size(1)} features, {len(torch.unique(pyg_data.y))} classes")
    
    return pyg_data


def prepare_transductive_data(pyg_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """Prepare data for transductive learning with train/val/test splits."""
    print(f"Preparing transductive data with splits: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")
    
    # Extract features and labels from PyG data
    x = pyg_data.x
    y = pyg_data.y
    edge_index = pyg_data.edge_index
    
    # Create node indices
    n_nodes = x.size(0)
    node_indices = np.arange(n_nodes)
    
    # Split indices
    train_idx, temp_idx = train_test_split(
        node_indices, 
        train_size=train_ratio, 
        stratify=y.numpy(), 
        random_state=seed
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_size, 
        stratify=y[temp_idx].numpy(), 
        random_state=seed
    )
    
    # Convert to tensors
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Val nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_classes': len(torch.unique(y))
    }


def train_sheaf_model(model, data, device, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train the sheaf diffusion model."""
    print(f"Training sheaf diffusion model for {epochs} epochs...")
    
    model = model.to(device)
    
    # Move data to device
    x = data['x'].to(device)
    y = data['y'].to(device)
    edge_index = data['edge_index'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Initialize sheaf model parameters with a dummy forward pass
    print("Initializing sheaf model parameters...")
    with torch.no_grad():
        _ = model(x, edge_index)
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Sheaf model initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError("Sheaf model has no parameters after initialization")
    
    # Debug: Check parameter gradients
    print("\nParameter analysis:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    # Create optimizer after parameters are initialized
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_model_state = None
    
    # Debug: Check initial predictions
    model.eval()
    with torch.no_grad():
        # Get the raw sheaf model output (log probabilities)
        sheaf_model = model.get_sheaf_model()
        initial_out = sheaf_model(x)
        initial_pred = initial_out.argmax(dim=1)
        print(f"\nInitial predictions distribution: {torch.bincount(initial_pred)}")
        print(f"True labels distribution: {torch.bincount(y)}")
        print(f"Initial output range: [{initial_out.min():.4f}, {initial_out.max():.4f}]")
        print(f"Initial output std: {initial_out.std():.4f}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass - get raw sheaf model output (log probabilities)
        sheaf_model = model.get_sheaf_model()
        out = sheaf_model(x)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        
        # Debug: Check loss and gradients
        if epoch == 0:
            print(f"\nFirst epoch analysis:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"  Output std: {out.std():.4f}")
            print(f"  Train predictions: {torch.bincount(out[train_mask].argmax(dim=1))}")
        
        # Backward pass
        loss.backward()
        
        # Debug: Check gradients
        if epoch == 0:
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if grad_norm > 0:
                        print(f"  {name} grad norm: {grad_norm:.6f}")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = sheaf_model(x)
            pred = out.argmax(dim=1)
            
            train_acc = accuracy_score(y[train_mask].cpu(), pred[train_mask].cpu())
            val_acc = accuracy_score(y[val_mask].cpu(), pred[val_mask].cpu())
            test_acc = accuracy_score(y[test_mask].cpu(), pred[test_mask].cpu())
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = sheaf_model(x)
        pred = out.argmax(dim=1)
        
        final_train_acc = accuracy_score(y[train_mask].cpu(), pred[train_mask].cpu())
        final_val_acc = accuracy_score(y[val_mask].cpu(), pred[val_mask].cpu())
        final_test_acc = accuracy_score(y[test_mask].cpu(), pred[test_mask].cpu())
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {final_train_acc:.4f}")
    print(f"Val Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    
    return {
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'test_acc': final_test_acc,
        'model': model
    }


def main():
    """Main function to run the test."""
    print("=" * 60)
    print("NEURAL SHEAF DIFFUSION COMMUNITY PREDICTION")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load graph from pickle
        pyg_data = load_first_graph()
        
        # Prepare data
        data = prepare_transductive_data(pyg_data)
        
        # Model parameters
        input_dim = data['x'].size(1)
        hidden_dim = 64
        output_dim = data['num_classes']
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of classes: {data['num_classes']}")
        
        # Test different sheaf types
        sheaf_configs = [
            {'sheaf_type': 'diag', 'd': 2, 'name': 'Diagonal Sheaf (d=2)'},
            {'sheaf_type': 'diag', 'd': 4, 'name': 'Diagonal Sheaf (d=4)'},
            {'sheaf_type': 'general', 'd': 2, 'name': 'General Sheaf (d=2)'},
        ]
        
        results = {}
        
        for config in sheaf_configs:
            print(f"\n" + "="*50)
            print(f"Testing {config['name']}")
            print("="*50)
            
            try:
                # Create model
                model = SheafDiffusionModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    sheaf_type=config['sheaf_type'],
                    d=config['d'],
                    num_layers=2,
                    dropout=0.1,
                    input_dropout=0.1,
                    is_regression=False,
                    is_graph_level_task=False,
                    device=device,
                    normalised=True,
                    deg_normalised=False,
                    linear=False,
                    left_weights=True,
                    right_weights=True,
                    sparse_learner=False,
                    use_act=True,
                    sheaf_act="tanh",
                    second_linear=False,
                    orth="householder",
                    edge_weights=False,
                    max_t=1.0,
                    add_lp=False,
                    add_hp=False
                )
                
                # Train model
                start_time = time.time()
                result = train_sheaf_model(model, data, device, epochs=200)
                training_time = time.time() - start_time
                
                result['training_time'] = training_time
                result['config'] = config
                results[config['name']] = result
                
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        print(f"\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"{name}: ERROR - {result['error']}")
            else:
                print(f"{name}:")
                print(f"  Train Acc: {result['train_acc']:.4f}")
                print(f"  Val Acc: {result['val_acc']:.4f}")
                print(f"  Test Acc: {result['test_acc']:.4f}")
                print(f"  Training Time: {result['training_time']:.2f}s")
        
        print(f"\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 