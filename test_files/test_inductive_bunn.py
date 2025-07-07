#!/usr/bin/env python3
"""
Inductive test script for InductiveBuNN models.
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

from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveBuNNWrapper, ImprovedInductiveBuNNWrapper


class SimpleGCN(nn.Module):
    """Simple GCN model for comparison with BuNN models."""
    
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
        return out


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
    """Load graphs from graphs.pkl file and convert to PyG format."""
    print("Loading graphs from graphs.pkl...")
    
    with open('graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError("No graphs found in graphs.pkl")
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Convert all graphs to PyG format
    pyg_graphs = []
    for i, graph_sample in enumerate(graphs):
        pyg_data = graphsample_to_pyg(graph_sample)
        pyg_graphs.append(pyg_data)
        if i < 5:  # Print info for first 5 graphs
            print(f"Graph {i}: {pyg_data.x.size(0)} nodes, {pyg_data.x.size(1)} features, {len(torch.unique(pyg_data.y))} classes")
    
    return pyg_graphs


def prepare_inductive_data(graphs, num_train=5, num_val=5, num_test=5, seed=42):
    """Prepare data for inductive learning with separate train/val/test graphs."""
    print(f"Preparing inductive data: {num_train} train, {num_val} val, {num_test} test graphs")
    
    if len(graphs) < num_train + num_val + num_test:
        raise ValueError(f"Not enough graphs. Need {num_train + num_val + num_test}, have {len(graphs)}")
    
    # Set random seed for reproducible splits
    np.random.seed(seed)
    indices = np.random.permutation(len(graphs))
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]
    
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Get number of classes from all graphs
    all_classes = set()
    for graph in graphs:
        all_classes.update(graph.y.numpy())
    num_classes = len(all_classes)
    
    print(f"Total number of classes across all graphs: {num_classes}")
    
    return {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'num_classes': num_classes
    }


def train_inductive_bunn_model(model, data, device, epochs=30, lr=0.01, weight_decay=5e-4):
    """Train the inductive BuNN model."""
    print(f"Training inductive BuNN model for {epochs} epochs...")
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(data['train_graphs'], batch_size=1, shuffle=True)
    val_loader = DataLoader(data['val_graphs'], batch_size=1, shuffle=False)
    test_loader = DataLoader(data['test_graphs'], batch_size=1, shuffle=False)
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"BuNN model initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError("BuNN model has no parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
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
            start_time = time.time()
            out = model(batch.x, batch.edge_index)
            end_time = time.time()
            
            loss = criterion(out, batch.y)
            
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


def train_gcn_model(model, data, device, epochs=10, lr=0.01, weight_decay=5e-4):
    """Train the GCN model."""
    print(f"Training GCN model for {epochs} epochs...")
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(data['train_graphs'], batch_size=5, shuffle=True)
    val_loader = DataLoader(data['val_graphs'], batch_size=5, shuffle=False)
    test_loader = DataLoader(data['test_graphs'], batch_size=5, shuffle=False)
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"GCN model initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError("GCN model has no parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
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
            loss = criterion(out, batch.y)
            
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
    """Main function to run the inductive BuNN test."""
    print("=" * 60)
    print("INDUCTIVE BUNDLE NEURAL NETWORK COMMUNITY PREDICTION")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load graphs from pickle
        graphs = load_graphs()
        
        # Prepare inductive data
        data = prepare_inductive_data(graphs, num_train=5, num_val=5, num_test=5)
        
        # Get model parameters from first graph
        first_graph = data['train_graphs'][0]
        input_dim = first_graph.x.size(1)
        hidden_dim = 64
        output_dim = data['num_classes']
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of classes: {data['num_classes']}")
        
        # Test different BuNN configurations
        bunn_configs = [
            {
                'd': 2, 'num_bundles': 8, 'bundle_method': 'cayley', 
                'heat_method': 'taylor', 'name': 'BuNN Rotation (d=2)'
            },
            {
                'd': 2, 'num_bundles': 16, 'bundle_method': 'cayley', 
                'heat_method': 'taylor', 'name': 'BuNN Rotation Multi-Bundle (d=2)'
            },
            {
                'd': 2, 'num_bundles': 24, 'bundle_method': 'cayley', 
                'heat_method': 'spectral', 'name': 'BuNN Spectral (d=2)'
            },
        ]
        
        # Test improved BuNN configurations
        improved_bunn_configs = [
            {
                'd': 2, 'num_bundles': 8, 'bundle_method': 'rotation', 
                'heat_method': 'adaptive', 'name': 'Improved BuNN Adaptive (d=2)',
                'pos_enc_dim': 8, 'pos_enc_type': 'laplacian'
            },
            {
                'd': 2, 'num_bundles': 16, 'bundle_method': 'rotation', 
                'heat_method': 'adaptive', 'name': 'Improved BuNN Multi-Bundle Adaptive (d=2)',
                'pos_enc_dim': 8, 'pos_enc_type': 'laplacian'
            },
            {
                'd': 2, 'num_bundles': 24, 'bundle_method': 'rotation', 
                'heat_method': 'spectral', 'name': 'Improved BuNN Spectral (d=2)',
                'pos_enc_dim': 8, 'pos_enc_type': 'random_walk'
            },
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
        gcn_result = train_gcn_model(gcn_model, data, device, epochs=30)
        gcn_training_time = time.time() - start_time
        
        gcn_result['training_time'] = gcn_training_time
        gcn_result['config'] = {'name': 'Simple GCN (Baseline)'}
        results['Simple GCN (Baseline)'] = gcn_result
        
        print(f"GCN training completed in {gcn_training_time:.2f} seconds")
        
        for config in bunn_configs:
            print(f"\n" + "="*50)
            print(f"Testing {config['name']}")
            print("="*50)
            
            try:
                # Create BuNN model
                model = InductiveBuNNWrapper(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    d=config['d'],
                    num_bundles=config['num_bundles'],
                    num_layers=2,
                    bundle_method=config['bundle_method'],
                    heat_method=config['heat_method'],
                    max_degree=10,
                    diffusion_times=None,
                    dropout=0.1,
                    input_dropout=0.1,
                    is_regression=False,
                    is_graph_level_task=False,
                    graph_pooling="mean",
                    activation="relu",
                    device=device
                )
                
                # Train model
                start_time = time.time()
                result = train_inductive_bunn_model(model, data, device, epochs=30)
                training_time = time.time() - start_time
                
                result['training_time'] = training_time
                result['config'] = config
                results[config['name']] = result
                
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
                results[config['name']] = {'error': str(e)}
                import traceback
                traceback.print_exc()
        
        # Test improved BuNN models
        for config in improved_bunn_configs:
            print(f"\n" + "="*50)
            print(f"Testing {config['name']}")
            print("="*50)
            
            try:
                # Create improved BuNN model
                model = ImprovedInductiveBuNNWrapper(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    d=config['d'],
                    num_bundles=config['num_bundles'],
                    num_layers=2,
                    bundle_method=config['bundle_method'],
                    heat_method=config['heat_method'],
                    max_degree=8,
                    diffusion_times=None,
                    dropout=0.1,
                    input_dropout=0.1,
                    is_regression=False,
                    is_graph_level_task=False,
                    graph_pooling="mean",
                    activation="relu",
                    device=device,
                    pos_enc_dim=config['pos_enc_dim'],
                    pos_enc_type=config['pos_enc_type'],
                    use_graph_structure=True
                )
                
                # Train model
                start_time = time.time()
                result = train_inductive_bunn_model(model, data, device, epochs=30)
                training_time = time.time() - start_time
                
                result['training_time'] = training_time
                result['config'] = config
                results[config['name']] = result
                
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
                results[config['name']] = {'error': str(e)}
                import traceback
                traceback.print_exc()
        
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
        
        print(f"\nInductive BuNN test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 