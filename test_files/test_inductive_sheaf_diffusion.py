#!/usr/bin/env python3
"""
Inductive test script for neural sheaf diffusion models.
Loads graphs from graphs.pkl and runs inductive community prediction with 5 train, 5 val, 5 test graphs.
"""

import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import time

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveSheafDiffusionModel
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders,
    precompute_sheaf_laplacian, 
    create_sheaf_dataloaders
)


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
            
            # Tasks - Use community prediction task
            self.tasks = ["community_prediction"]
            self.is_regression = {"community_prediction": False}
            self.is_graph_level_tasks = {"community_prediction": False}
            
            # Sheaf settings
            self.sheaf_d = 3
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


class SimpleGCN(nn.Module):
    """Simple GCN model for comparison with sheaf diffusion models."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using Xavier initialization."""
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x, edge_index):
        """Forward pass through the GCN model."""
        h = x
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last conv
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Apply classifier
        out = self.classifier(h)
        return F.log_softmax(out, dim=1)


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


def load_graphs():
    """Load original GraphSample objects from graphs.pkl file."""
    print("Loading graphs from graphs.pkl...")
    
    with open('graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError("No graphs found in graphs.pkl")
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Print info for first 5 graphs
    for i, graph_sample in enumerate(graphs[:5]):
        graph = graph_sample.graph
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_communities = len(set(graph_sample.community_labels)) if hasattr(graph_sample, 'community_labels') else 0
        print(f"Graph {i}: {n_nodes} nodes, {n_edges} edges, {n_communities} communities")
    
    return graphs


def prepare_inductive_data(graph_samples, num_train=5, num_val=5, num_test=5, seed=42):
    """Prepare data for inductive learning using the same approach as the comparison script."""
    print(f"Preparing inductive data: {num_train} train, {num_val} val, {num_test} test graphs")
    
    if len(graph_samples) < num_train + num_val + num_test:
        raise ValueError(f"Not enough graphs. Need {num_train + num_val + num_test}, have {len(graph_samples)}")
    
    # Create config
    config = create_test_config()
    
    # Use the same data preparation as the comparison script
    print("Preparing data for precomputed models...")
    start_time = time.time()
    inductive_data, sheaf_inductive_data, fold_indices = prepare_inductive_data(graph_samples, config)
    prep_time = time.time() - start_time
    print(f"Data preparation completed in {prep_time:.2f}s")
    
    # Create dataloaders for precomputed models
    print("Creating dataloaders for precomputed models...")
    start_time = time.time()
    normal_dataloaders = create_inductive_dataloaders(inductive_data, config)
    sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
    dataloader_time = time.time() - start_time
    print(f"Dataloader creation completed in {dataloader_time:.2f}s")
    
    # For message-passing models: convert to PyG and create simple splits
    print("Preparing data for message-passing models...")
    pyg_graphs = [graphsample_to_pyg(g) for g in graph_samples]
    
    # Create simple train/val/test split for message-passing models
    n_graphs = len(pyg_graphs)
    n_train = int(n_graphs * 0.6)
    n_val = int(n_graphs * 0.2)
    n_test = n_graphs - n_train - n_val
    
    # Set random seed for reproducible splits
    np.random.seed(seed)
    indices = np.random.permutation(n_graphs)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_graphs = [pyg_graphs[i] for i in train_indices]
    val_graphs = [pyg_graphs[i] for i in val_indices]
    test_graphs = [pyg_graphs[i] for i in test_indices]
    
    message_passing_data = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'task': 'community_prediction'
    }
    
    precomputed_data = {
        'inductive_data': inductive_data,
        'sheaf_inductive_data': sheaf_inductive_data,
        'normal_dataloaders': normal_dataloaders,
        'sheaf_dataloaders': sheaf_dataloaders,
        'config': config
    }
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Get number of classes from all graphs
    all_classes = set()
    for graph in pyg_graphs:
        all_classes.update(graph.y.numpy())
    num_classes = len(all_classes)
    
    print(f"Total number of classes across all graphs: {num_classes}")
    
    return {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'num_classes': num_classes,
        'precomputed_data': precomputed_data,
        'message_passing_data': message_passing_data
    }


def train_inductive_sheaf_model(model, data, device, epochs=10, lr=0.01, weight_decay=5e-4):
    """Train the inductive sheaf diffusion model."""
    print(f"Training inductive sheaf diffusion model for {epochs} epochs...")
    
    model = model.to(device)
    
    # Use precomputed dataloaders
    train_loader = data['precomputed_data']['sheaf_dataloaders']['community_prediction']['fold_0']['train']
    val_loader = data['precomputed_data']['sheaf_dataloaders']['community_prediction']['fold_0']['val']
    test_loader = data['precomputed_data']['sheaf_dataloaders']['community_prediction']['fold_0']['test']
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Sheaf model initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError("Sheaf model has no parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_nodes = 0
        
        # Training
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass - use precomputed data
            out = model(batch.x, batch.edge_index, graph=batch)
            loss = F.nll_loss(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            total_train_correct += (pred == batch.y).sum().item()
            total_train_nodes += batch.y.size(0)
            total_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_correct = 0
        total_val_nodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, graph=batch)
                pred = out.argmax(dim=1)
                total_val_correct += (pred == batch.y).sum().item()
                total_val_nodes += batch.y.size(0)
        
        train_acc = total_train_correct / total_train_nodes if total_train_nodes > 0 else 0
        val_acc = total_val_correct / total_val_nodes if total_val_nodes > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    total_test_correct = 0
    total_test_nodes = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, graph=batch)
            pred = out.argmax(dim=1)
            total_test_correct += (pred == batch.y).sum().item()
            total_test_nodes += batch.y.size(0)
    
    final_test_acc = total_test_correct / total_test_nodes if total_test_nodes > 0 else 0
    
    print(f"\nFinal Results:")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': final_test_acc,
        'model': model
    }


def train_gcn_model(model, data, device, epochs=10, lr=0.01, weight_decay=5e-4):
    """Train the GCN model."""
    print(f"Training GCN model for {epochs} epochs...")
    
    model = model.to(device)
    
    # Use normal dataloaders for GCN
    train_loader = data['precomputed_data']['normal_dataloaders']['community_prediction']['fold_0']['train']
    val_loader = data['precomputed_data']['normal_dataloaders']['community_prediction']['fold_0']['val']
    test_loader = data['precomputed_data']['normal_dataloaders']['community_prediction']['fold_0']['test']
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"GCN model initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError("GCN model has no parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_nodes = 0
        
        # Training
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            total_train_correct += (pred == batch.y).sum().item()
            total_train_nodes += batch.y.size(0)
            total_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_correct = 0
        total_val_nodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                pred = out.argmax(dim=1)
                total_val_correct += (pred == batch.y).sum().item()
                total_val_nodes += batch.y.size(0)
        
        train_acc = total_train_correct / total_train_nodes if total_train_nodes > 0 else 0
        val_acc = total_val_correct / total_val_nodes if total_val_nodes > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    total_test_correct = 0
    total_test_nodes = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            total_test_correct += (pred == batch.y).sum().item()
            total_test_nodes += batch.y.size(0)
    
    final_test_acc = total_test_correct / total_test_nodes if total_test_nodes > 0 else 0
    
    print(f"\nFinal Results:")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': final_test_acc,
        'model': model
    }


def main():
    """Main function to run the inductive test."""
    print("=" * 60)
    print("INDUCTIVE NEURAL SHEAF DIFFUSION COMMUNITY PREDICTION")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load original GraphSample objects
        graph_samples = load_graphs()
        
        # Prepare inductive data using the same approach as comparison script
        data = prepare_inductive_data(graph_samples, num_train=5, num_val=5, num_test=5)
        
        # Get model parameters from precomputed data
        first_graph = data['precomputed_data']['inductive_data']['community_prediction']['fold_0']['train']['graphs'][0]
        input_dim = first_graph.x.size(1)
        hidden_dim = 64
        output_dim = data['num_classes']
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of classes: {data['num_classes']}")
        
        # Test different sheaf types
        sheaf_configs = [
            {'sheaf_type': 'bundle', 'd': 2, 'name': 'Bundle Sheaf (d=2)', 'orth': 'cayley'},
            {'sheaf_type': 'diag', 'd': 2, 'name': 'Diagonal Sheaf (d=2)', 'orth': 'householder'},
            {'sheaf_type': 'general', 'd': 2, 'name': 'General Sheaf (d=2)', 'orth': 'householder'},
        ]
        
        results = {}
        
        # Test GCN first as baseline
        print(f"\n" + "="*50)
        print(f"Testing Simple GCN (Baseline)")
        print("="*50)
        
        # Create GCN model
        gcn_model = SimpleGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=0.1
        )
        
        # Train GCN model
        start_time = time.time()
        gcn_result = train_gcn_model(gcn_model, data, device, epochs=10)
        gcn_training_time = time.time() - start_time
        
        gcn_result['training_time'] = gcn_training_time
        gcn_result['config'] = {'name': 'Simple GCN (Baseline)'}
        results['Simple GCN (Baseline)'] = gcn_result
        
        print(f"GCN training completed in {gcn_training_time:.2f} seconds")
        
        for config in sheaf_configs:
            print(f"\n" + "="*50)
            print(f"Testing {config['name']}")
            print("="*50)
            
            try:
                # Create model
                model = InductiveSheafDiffusionModel(
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
                    orth=config['orth'],
                    edge_weights=False,
                    max_t=1.0,
                    add_lp=False,
                    add_hp=False
                )
                
                # Train model
                start_time = time.time()
                result = train_inductive_sheaf_model(model, data, device, epochs=10)
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
        
        print(f"\nInductive test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 