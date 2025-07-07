#!/usr/bin/env python3
"""
Comprehensive inductive test script for neural sheaf diffusion models with positional encodings.
Loads graphs from graphs.pkl and runs inductive community prediction with different PE types.
Tests both sheaf models and GCN baseline with each PE type.
Uses clean approach where PEs are saved as graph attributes and models handle concatenation.
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
from collections import defaultdict

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveSheafDiffusionModel
from experiments.models import SheafDiffusionModel, GraphTransformerModel, GNNModel
from experiments.inductive.data import (
    add_positional_encodings_to_data,
    prepare_inductive_data,
    create_inductive_dataloaders,
    create_sheaf_dataloaders
)


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
        graph_samples = pickle.load(f)
    if not graph_samples:
        raise ValueError("No graphs found in graphs.pkl")
    print(f"Loaded {len(graph_samples)} graphs")
    return graph_samples


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
            
            # Tasks - Use community prediction
            self.tasks = ["community"]
            self.is_regression = {"community": False}
            self.is_graph_level_tasks = {"community": False}
            
            # Sheaf settings
            self.sheaf_d = 3
            self.sheaf_type = "bundle"
            self.sheaf_normalised = True
            self.sheaf_deg_normalised = False
            self.sheaf_add_hp = False
            self.sheaf_add_lp = False
            self.sheaf_orth = "cayley"

            self.run_transformers = False
            self.precompute_pe = True  # Enable PE precomputation
            self.max_pe_dim = 16
            
            # Allow unseen community combinations for simpler testing
            self.allow_unseen_community_combinations_for_eval = True
    
    return TestConfig()


class PEModelComparison:
    """Class to handle PE model comparison experiments."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.timing_stats = defaultdict(dict)
        
    def prepare_data(self, graph_samples, pe_types=['laplacian', 'degree', 'rwse'], max_pe_dim=16):
        """Prepare data using the clean approach - PEs are saved as graph attributes."""
        print(f"Preparing data with {len(graph_samples)} graphs...")
        print(f"PE types: {pe_types}")
        print(f"PE dimension: {max_pe_dim}")
        
        # Create config
        config = create_test_config()
        config.precompute_pe = True
        config.max_pe_dim = max_pe_dim
        
        # Prepare data with PE precomputation enabled
        print("Preparing data with PE precomputation...")
        start_time = time.time()
        inductive_data, sheaf_inductive_data, fold_indices = prepare_inductive_data(
            graph_samples, 
            config,
            pe_types=pe_types
        )
        prep_time = time.time() - start_time
        print(f"Data preparation completed in {prep_time:.2f}s")
        
        # Create standard dataloaders (PEs are already saved as graph attributes)
        print("Creating dataloaders...")
        start_time = time.time()
        normal_dataloaders = create_inductive_dataloaders(inductive_data, config)
        sheaf_dataloaders = create_sheaf_dataloaders(sheaf_inductive_data, config)
        dataloader_time = time.time() - start_time
        print(f"Dataloader creation completed in {dataloader_time:.2f}s")
        
        # Verify that PEs are properly saved
        print("Verifying PE attributes...")
        first_graph = inductive_data['community']['fold_0']['train']['graphs'][0]
        pe_attrs = []
        for pe_type in pe_types:
            pe_attr_name = f"{pe_type}_pe"
            if hasattr(first_graph, pe_attr_name):
                pe_attrs.append(pe_attr_name)
                pe_tensor = getattr(first_graph, pe_attr_name)
                print(f"  ✓ {pe_attr_name}: shape {pe_tensor.shape}")
            else:
                print(f"  ✗ {pe_attr_name}: not found")
        
        print(f"Found PE attributes: {pe_attrs}")
        
        return {
            'inductive_data': inductive_data,
            'sheaf_inductive_data': sheaf_inductive_data,
            'normal_dataloaders': normal_dataloaders,
            'sheaf_dataloaders': sheaf_dataloaders,
            'config': config,
            'fold_indices': fold_indices
        }
    
    def train_gnn_model(self, model, data, pe_type, model_name, epochs=50, lr=0.01):
        """Train a GNN model (GCN, GAT, SAGE) with specific PE type."""
        print(f"Training {model_name} with {pe_type if pe_type else 'No'} PE for {epochs} epochs...")
        
        model = model.to(self.device)
        
        # Use standard dataloaders (PEs are already in graph attributes)
        train_loader = data['normal_dataloaders']['community']['fold_0']['train']
        val_loader = data['normal_dataloaders']['community']['fold_0']['val']
        test_loader = data['normal_dataloaders']['community']['fold_0']['test']
        
        # Check parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} has {param_count} parameters")
        
        if param_count == 0:
            raise ValueError(f"{model_name} has no parameters")
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        training_times = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_correct = 0
            total_train_nodes = 0
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass - model will automatically handle PE concatenation
                out = model(batch.x, batch.edge_index, graph=batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                pred = out.argmax(dim=1)
                total_train_correct += (pred == batch.y).sum().item()
                total_train_nodes += batch.y.size(0)
                total_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Validation
            model.eval()
            total_val_correct = 0
            total_val_nodes = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
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
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Time={epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        total_test_correct = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, graph=batch)
                pred = out.argmax(dim=1)
                total_test_correct += (pred == batch.y).sum().item()
                total_test_nodes += batch.y.size(0)
        
        final_test_acc = total_test_correct / total_test_nodes if total_test_nodes > 0 else 0
        
        return {
            'train_acc': train_acc,
            'val_acc': best_val_acc,
            'test_acc': final_test_acc,
            'model': model,
            'avg_epoch_time': np.mean(training_times),
            'total_training_time': sum(training_times)
        }
    
    def train_sheaf_model(self, model, data, pe_type, model_name, epochs=50, lr=0.01):
        """Train a sheaf model with specific PE type."""
        print(f"Training {model_name} with {pe_type if pe_type else 'No'} PE for {epochs} epochs...")
        
        model = model.to(self.device)
        
        # Use standard dataloaders (PEs are already in graph attributes)
        train_loader = data['sheaf_dataloaders']['community']['fold_0']['train']
        val_loader = data['sheaf_dataloaders']['community']['fold_0']['val']
        test_loader = data['sheaf_dataloaders']['community']['fold_0']['test']
        
        # Check parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} has {param_count} parameters")
        
        if param_count == 0:
            raise ValueError(f"{model_name} has no parameters")
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        training_times = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_correct = 0
            total_train_nodes = 0
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass - model will automatically handle PE concatenation
                out = model(batch.x, batch.edge_index, graph=batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                pred = out.argmax(dim=1)
                total_train_correct += (pred == batch.y).sum().item()
                total_train_nodes += batch.y.size(0)
                total_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Validation
            model.eval()
            total_val_correct = 0
            total_val_nodes = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
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
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Time={epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        total_test_correct = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, graph=batch)
                pred = out.argmax(dim=1)
                total_test_correct += (pred == batch.y).sum().item()
                total_test_nodes += batch.y.size(0)
        
        final_test_acc = total_test_correct / total_test_nodes if total_test_nodes > 0 else 0
        
        return {
            'train_acc': train_acc,
            'val_acc': best_val_acc,
            'test_acc': final_test_acc,
            'model': model,
            'avg_epoch_time': np.mean(training_times),
            'total_training_time': sum(training_times)
        }
    
    def run_comparison(self, graph_samples, epochs=50):
        """Run the complete PE comparison."""
        print("=" * 80)
        print("COMPREHENSIVE PE COMPARISON")
        print("=" * 80)
        
        # Prepare data using clean approach
        data = self.prepare_data(
            graph_samples, 
            pe_types=['laplacian', 'degree', 'rwse'], 
            max_pe_dim=16
        )
        
        # Get model parameters from data
        first_graph = data['inductive_data']['community']['fold_0']['train']['graphs'][0]
        input_dim = first_graph.x.size(1)
        hidden_dim = 64
        output_dim = data['inductive_data']['community']['metadata']['output_dim']
        pe_dim = 16
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Output dimension: {output_dim}")
        print(f"PE dimension: {pe_dim}")
        
        # Define PE types to test (including None)
        pe_types = [None, 'laplacian', 'degree', 'rwse']
        
        # Define model configurations to test
        model_configs = [
            # GNN models
            {'type': 'gcn', 'name': 'GCN', 'params': {'gnn_type': 'gcn'}},
            {'type': 'gat', 'name': 'GAT', 'params': {'gnn_type': 'gat', 'heads': 4}},
            {'type': 'sage', 'name': 'SAGE', 'params': {'gnn_type': 'sage'}},
            # GraphGPS model
            {'type': 'graphgps', 'name': 'GraphGPS', 'params': {'transformer_type': 'graphgps', 'num_heads': 8, 'local_gnn_type': 'gcn', 'attn_type': 'multihead'}},
            # Message passing sheaf models
            {'type': 'sheaf', 'name': 'Message Passing Orthogonal Sheaf (d=2)', 'params': {'sheaf_type': 'orthogonal', 'd': 2, 'orthogonal_method': 'euler'}},
            {'type': 'sheaf', 'name': 'Message Passing Diagonal Sheaf (d=2)', 'params': {'sheaf_type': 'diagonal', 'd': 2, 'orthogonal_method': 'euler'}},
            {'type': 'sheaf', 'name': 'Message Passing General Sheaf (d=2)', 'params': {'sheaf_type': 'general', 'd': 2, 'orthogonal_method': 'euler'}},
        ]
        
        results = {}
        
        # Test each PE type
        for pe_type in pe_types:
            pe_name = "No PE" if pe_type is None else f"{pe_type.upper()} PE"
            print(f"\n" + "="*60)
            print(f"Testing with {pe_name}")
            print("="*60)
            
            # Test each model type
            for config in model_configs:
                print(f"\n--- Testing {config['name']} with {pe_name} ---")
                
                try:
                    # Create model based on type
                    if config['type'] in ['gcn', 'gat', 'sage']:
                        # Create GNN model
                        model = GNNModel(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=2,
                            dropout=0.1,
                            is_regression=False,
                            is_graph_level_task=False,
                            pe_type=pe_type,
                            pe_dim=pe_dim,
                            **config['params']
                        )
                        
                        # Train GNN model
                        start_time = time.time()
                        result = self.train_gnn_model(model, data, pe_type, f"{config['name']} with {pe_name}", epochs=epochs)
                        training_time = time.time() - start_time
                        
                    elif config['type'] == 'graphgps':
                        # Create GraphGPS model
                        model = GraphTransformerModel(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=2,
                            dropout=0.1,
                            is_regression=False,
                            is_graph_level_task=False,
                            pe_dim=pe_dim,
                            pe_type=pe_type,
                            pe_norm_type=None,
                            **config['params']
                        )
                        
                        # Train GraphGPS model
                        start_time = time.time()
                        result = self.train_gnn_model(model, data, pe_type, f"{config['name']} with {pe_name}", epochs=epochs)
                        training_time = time.time() - start_time
                        
                    elif config['type'] == 'sheaf':
                        # Create message passing sheaf model
                        model = SheafDiffusionModel(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=2,
                            dropout=0.1,
                            is_regression=False,
                            is_graph_level_task=False,
                            pe_type=pe_type,
                            pe_dim=pe_dim,
                            **config['params']
                        )
                        
                        # Train sheaf model
                        start_time = time.time()
                        result = self.train_sheaf_model(model, data, pe_type, f"{config['name']} with {pe_name}", epochs=epochs)
                        training_time = time.time() - start_time
                    
                    result['training_time'] = training_time
                    result['config'] = {**config, 'pe_type': pe_type}
                    results[f"{config['name']} with {pe_name}"] = result
                    
                    print(f"Training completed in {training_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error with {config['name']} with {pe_name}: {e}")
                    results[f"{config['name']} with {pe_name}"] = {'error': str(e)}
        
        return results
    
    def print_results(self, results):
        """Print comprehensive results comparison."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PE COMPARISON RESULTS")
        print("="*80)
        
        # Create results table
        print(f"{'Model':<40} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Avg Epoch':<10} {'Total Time':<12}")
        print("-" * 92)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"{name:<40} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
            else:
                train_acc = f"{result['train_acc']:.4f}"
                val_acc = f"{result['val_acc']:.4f}"
                test_acc = f"{result['test_acc']:.4f}"
                avg_epoch = f"{result['avg_epoch_time']:.2f}s"
                total_time = f"{result['total_training_time']:.1f}s"
                
                print(f"{name:<40} {train_acc:<10} {val_acc:<10} {test_acc:<10} {avg_epoch:<10} {total_time:<12}")
        
        # Performance analysis
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Find best performing models
        valid_results = {name: result for name, result in results.items() if 'error' not in result}
        
        if valid_results:
            # Best test accuracy (higher is better)
            best_test = max(valid_results.items(), key=lambda x: x[1]['test_acc'])
            print(f"Best Test Accuracy: {best_test[0]} ({best_test[1]['test_acc']:.4f})")
            
            # Fastest training
            fastest = min(valid_results.items(), key=lambda x: x[1]['total_training_time'])
            print(f"Fastest Training: {fastest[0]} ({fastest[1]['total_training_time']:.1f}s)")
            
            # Best validation accuracy (higher is better)
            best_val = max(valid_results.items(), key=lambda x: x[1]['val_acc'])
            print(f"Best Validation Accuracy: {best_val[0]} ({best_val[1]['val_acc']:.4f})")
        
        # PE type comparison
        print("\n" + "="*80)
        print("PE TYPE COMPARISON")
        print("="*80)
        
        pe_results = {}
        for name, result in results.items():
            if 'error' not in result:
                pe_type = result['config'].get('pe_type', 'unknown')
                if pe_type not in pe_results:
                    pe_results[pe_type] = []
                pe_results[pe_type].append(result)
        
        for pe_type in [None, 'laplacian', 'degree', 'rwse']:
            pe_name = "No PE" if pe_type is None else f"{pe_type.upper()} PE"
            if pe_type in pe_results:
                avg_test = np.mean([r['test_acc'] for r in pe_results[pe_type]])
                avg_time = np.mean([r['total_training_time'] for r in pe_results[pe_type]])
                print(f"{pe_name}: Avg Test Acc={avg_test:.4f}, Avg Time={avg_time:.1f}s")
            else:
                print(f"{pe_name}: No results")


def main():
    """Main function to run the PE comparison."""
    print("=" * 80)
    print("PE COMPARISON EXPERIMENT")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load original GraphSample objects
        graph_samples = load_graphs()
        
        # Create comparison object
        comparison = PEModelComparison(device=device)
        
        # Run comparison
        results = comparison.run_comparison(graph_samples, epochs=50)
        
        # Print results
        comparison.print_results(results)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 