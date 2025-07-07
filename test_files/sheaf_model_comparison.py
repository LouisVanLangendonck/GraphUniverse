#!/usr/bin/env python3
"""
Comprehensive comparison of sheaf models vs baseline GNNs.
Compares message-passing sheaf, precomputed sheaf, GCN, and GraphSAGE models
in terms of speed and performance on the graph classification task.
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops, degree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

# Import the message-passing sheaf model from the test file
from test_sheaf_message import NeuralSheafDiffusion, GCN, GraphSAGE

# Import the precomputed sheaf model
from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveSheafDiffusionModel, InductiveContSheafDiffusionModel, InductiveBuNNWrapper
from experiments.inductive.data import (
    prepare_inductive_data, 
    create_inductive_dataloaders,
    precompute_sheaf_laplacian, 
    create_sheaf_dataloaders
)

# Create regression-specific versions of the models (without sigmoid)
class MLPBaseline(nn.Module):
    """Simple MLP baseline that only uses node features without graph structure."""
    
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers=3, dropout=0.3):
        super(MLPBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create MLP layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_channels if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:  # Don't add activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through MLP (ignores edge_index).
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices (ignored for MLP baseline)
            
        Returns:
            Output regression values [num_nodes, output_dim]
        """
        return self.mlp(x)  # Process each node independently

class NeuralSheafDiffusionRegression(NeuralSheafDiffusion):
    """Regression version of NeuralSheafDiffusion without sigmoid activation."""
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete Neural Sheaf Diffusion network.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output regression values [num_nodes, output_dim]
        """
        # Initial projection to stalk structure
        x = self.input_projection(x)  # [num_nodes, d * hidden_channels]
        x = self.dropout(x)
        
        # Apply sheaf diffusion layers
        for layer in self.sheaf_layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # Final projection to output (no sigmoid for regression)
        x = self.output_projection(x)  # [num_nodes, output_dim]
        
        return x  # Return raw output for regression

class GCNRegression(GCN):
    """Regression version of GCN without sigmoid activation."""
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output regression values [num_nodes, output_dim]
        """
        # GCN layers with normalization
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final projection (no sigmoid for regression)
        x = self.output_projection(x)
        
        return x  # Return raw output for regression

class GraphSAGERegression(GraphSAGE):
    """Regression version of GraphSAGE without sigmoid activation."""
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output regression values [num_nodes, output_dim]
        """
        # GraphSAGE layers
        for layer in self.sage_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final projection (no sigmoid for regression)
        x = self.output_projection(x)
        
        return x  # Return raw output for regression


def graphsample_to_pyg(graph_sample):
    """Convert a GraphSample to a PyTorch Geometric Data object."""
    import networkx as nx
    
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
            
            # Tasks - Changed to k_hop_community_counts_k2
            self.tasks = ["k_hop_community_counts_k2"]
            self.is_regression = {"k_hop_community_counts_k2": True}
            self.is_graph_level_tasks = {"k_hop_community_counts_k2": False}
            self.khop_community_counts_k = 2
            
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
            
            # Regression settings
            self.regression_loss = 'mse'  # 'mse' or 'mae'
    
    return TestConfig()


class ModelComparison:
    """Class to handle model comparison experiments."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.timing_stats = defaultdict(dict)
        
    def prepare_data(self, graph_samples):
        """Prepare data using the same approach as test_sheaf_precomputation.py."""
        print(f"Preparing data with {len(graph_samples)} graphs...")
        
        # Create config (same as test_sheaf_precomputation.py)
        config = create_test_config()
        
        # For precomputed models: pass original GraphSample objects to prepare_inductive_data
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
        
        # For k_hop_community_counts_k2 task, we don't need to count classes since it's regression
        # The output dimension will be determined from the data preparation
        print(f"Prepared {len(pyg_graphs)} graphs for message-passing models")
        
        # Create simple train/val/test split for message-passing models
        n_graphs = len(pyg_graphs)
        n_train = int(n_graphs * 0.6)
        n_val = int(n_graphs * 0.2)
        n_test = n_graphs - n_train - n_val
        
        # Set random seed for reproducible splits
        np.random.seed(42)
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
            'task': 'k_hop_community_counts_k2'
        }
        
        precomputed_data = {
            'inductive_data': inductive_data,
            'sheaf_inductive_data': sheaf_inductive_data,
            'normal_dataloaders': normal_dataloaders,
            'sheaf_dataloaders': sheaf_dataloaders,
            'config': config
        }
        
        return message_passing_data, precomputed_data
    
    def train_message_passing_model(self, model, data, model_name, epochs=100, lr=0.01):
        """Train a message-passing model (sheaf, GCN, GraphSAGE)."""
        print(f"Training {model_name} (message-passing) for {epochs} epochs...")
        
        model = model.to(self.device)
        
        # Get dataloaders - Updated to use k_hop_community_counts_k2
        train_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['train']
        val_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['val']
        test_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['test']
        
        # Check parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} has {param_count} parameters")
        
        if param_count == 0:
            raise ValueError(f"{model_name} has no parameters")
        
        # Setup training - Use MSE loss for regression
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.MSELoss()  # Changed from CrossEntropyLoss to MSELoss for regression
        
        # Training loop
        best_val_loss = float('inf')  # Changed from best_val_acc to best_val_loss
        best_model_state = None
        training_times = []
        example_predictions = {
            'train': None,
            'val': None,
            'test': None
        }
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_mse = 0
            total_train_mae = 0
            total_train_nodes = 0
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y.float())  # Ensure y is float for regression
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Calculate MSE and MAE for regression
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())  # L1 loss = MAE
                    total_train_mse += mse.item()
                    total_train_mae += mae.item()
                
                total_train_nodes += batch.y.size(0)
                total_loss += loss.item()
                
                # Store example prediction from first batch of final epoch
                if epoch == epochs - 1 and batch_idx == 0 and example_predictions['train'] is None:
                    example_predictions['train'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Validation
            model.eval()
            total_val_loss = 0
            total_val_mse = 0
            total_val_mae = 0
            total_val_nodes = 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out, batch.y.float())
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())
                    total_val_loss += loss.item()
                    total_val_mse += mse.item()
                    total_val_mae += mae.item()
                    total_val_nodes += batch.y.size(0)
                    
                    # Store example prediction from first batch (final epoch)
                    if i == 0 and example_predictions['val'] is None:
                        example_predictions['val'] = {
                            'target': batch.y.float().cpu().numpy(),
                            'predicted': out.detach().cpu().numpy()
                        }
            
            train_mse = total_train_mse / len(train_loader) if len(train_loader) > 0 else 0
            train_mae = total_train_mae / len(train_loader) if len(train_loader) > 0 else 0
            val_mse = total_val_mse / len(val_loader) if len(val_loader) > 0 else 0
            val_mae = total_val_mae / len(val_loader) if len(val_loader) > 0 else 0
            avg_loss = total_loss / len(train_loader)
            
            if val_mse < best_val_loss:  # Lower MSE is better
                best_val_loss = val_mse
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}, "
                      f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, Time={epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        total_test_mse = 0
        total_test_mae = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                mse = F.mse_loss(out, batch.y.float())
                mae = F.l1_loss(out, batch.y.float())
                total_test_mse += mse.item()
                total_test_mae += mae.item()
                total_test_nodes += batch.y.size(0)
                
                # Store example prediction from first batch (final epoch)
                if i == 0 and example_predictions['test'] is None:
                    example_predictions['test'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
        
        final_test_mse = total_test_mse / len(test_loader) if len(test_loader) > 0 else 0
        final_test_mae = total_test_mae / len(test_loader) if len(test_loader) > 0 else 0
        
        return {
            'train_mse': train_mse,
            'val_mse': best_val_loss,
            'test_mse': final_test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': final_test_mae,
            'model': model,
            'avg_epoch_time': np.mean(training_times),
            'total_training_time': sum(training_times),
            'example_predictions': example_predictions
        }
    
    def train_precomputed_model(self, model, data, model_name, epochs=100, lr=0.01):
        """Train a precomputed sheaf model."""
        print(f"Training {model_name} (precomputed) for {epochs} epochs...")
        
        model = model.to(self.device)
        
        # Get dataloaders - Updated to use k_hop_community_counts_k2
        train_loader = data['sheaf_dataloaders']['k_hop_community_counts_k2']['fold_0']['train']
        val_loader = data['sheaf_dataloaders']['k_hop_community_counts_k2']['fold_0']['val']
        test_loader = data['sheaf_dataloaders']['k_hop_community_counts_k2']['fold_0']['test']
        
        # Check parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} has {param_count} parameters")
        
        if param_count == 0:
            raise ValueError(f"{model_name} has no parameters")
        
        # Setup training - Use MSE loss for regression
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.MSELoss()  # Changed from CrossEntropyLoss to MSELoss for regression
        
        # Training loop
        best_val_loss = float('inf')  # Changed from best_val_acc to best_val_loss
        best_model_state = None
        training_times = []
        example_predictions = {
            'train': None,
            'val': None,
            'test': None
        }
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_mse = 0
            total_train_mae = 0
            total_train_nodes = 0
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Use precomputed data
                out = model(batch.x, batch.edge_index, graph=batch)
                loss = criterion(out, batch.y.float())  # Ensure y is float for regression
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Calculate MSE and MAE for regression
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())  # L1 loss = MAE
                    total_train_mse += mse.item()
                    total_train_mae += mae.item()
                
                total_train_nodes += batch.y.size(0)
                total_loss += loss.item()
                
                # Store example prediction from first batch of final epoch
                if epoch == epochs - 1 and batch_idx == 0 and example_predictions['train'] is None:
                    example_predictions['train'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Validation
            model.eval()
            total_val_loss = 0
            total_val_mse = 0
            total_val_mae = 0
            total_val_nodes = 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index, graph=batch)
                    loss = criterion(out, batch.y.float())
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())
                    total_val_loss += loss.item()
                    total_val_mse += mse.item()
                    total_val_mae += mae.item()
                    total_val_nodes += batch.y.size(0)
                    
                    # Store example prediction from first batch (final epoch)
                    if i == 0 and example_predictions['val'] is None:
                        example_predictions['val'] = {
                            'target': batch.y.float().cpu().numpy(),
                            'predicted': out.detach().cpu().numpy()
                        }
            
            train_mse = total_train_mse / len(train_loader) if len(train_loader) > 0 else 0
            train_mae = total_train_mae / len(train_loader) if len(train_loader) > 0 else 0
            val_mse = total_val_mse / len(val_loader) if len(val_loader) > 0 else 0
            val_mae = total_val_mae / len(val_loader) if len(val_loader) > 0 else 0
            avg_loss = total_loss / len(train_loader)
            
            if val_mse < best_val_loss:  # Lower MSE is better
                best_val_loss = val_mse
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}, "
                      f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, Time={epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        total_test_mse = 0
        total_test_mae = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, graph=batch)
                mse = F.mse_loss(out, batch.y.float())
                mae = F.l1_loss(out, batch.y.float())
                total_test_mse += mse.item()
                total_test_mae += mae.item()
                total_test_nodes += batch.y.size(0)
                
                # Store example prediction from first batch (final epoch)
                if i == 0 and example_predictions['test'] is None:
                    example_predictions['test'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
        
        final_test_mse = total_test_mse / len(test_loader) if len(test_loader) > 0 else 0
        final_test_mae = total_test_mae / len(test_loader) if len(test_loader) > 0 else 0
        
        return {
            'train_mse': train_mse,
            'val_mse': best_val_loss,
            'test_mse': final_test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': final_test_mae,
            'model': model,
            'avg_epoch_time': np.mean(training_times),
            'total_training_time': sum(training_times),
            'example_predictions': example_predictions
        }
    
    def train_mlp_model(self, model, data, model_name, epochs=100, lr=0.01):
        """Train an MLP baseline model."""
        print(f"Training {model_name} (MLP baseline) for {epochs} epochs...")
        
        model = model.to(self.device)
        
        # Get dataloaders - Updated to use k_hop_community_counts_k2
        train_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['train']
        val_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['val']
        test_loader = data['normal_dataloaders']['k_hop_community_counts_k2']['fold_0']['test']
        
        # Check parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{model_name} has {param_count} parameters")
        
        if param_count == 0:
            raise ValueError(f"{model_name} has no parameters")
        
        # Setup training - Use MSE loss for regression
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.MSELoss()  # Changed from CrossEntropyLoss to MSELoss for regression
        
        # Training loop
        best_val_loss = float('inf')  # Changed from best_val_acc to best_val_loss
        best_model_state = None
        training_times = []
        example_predictions = {
            'train': None,
            'val': None,
            'test': None
        }
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_mse = 0
            total_train_mae = 0
            total_train_nodes = 0
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # MLP only uses node features, ignores edge_index
                out = model(batch.x)
                loss = criterion(out, batch.y.float())  # Ensure y is float for regression
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Calculate MSE and MAE for regression
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())  # L1 loss = MAE
                    total_train_mse += mse.item()
                    total_train_mae += mae.item()
                
                total_train_nodes += batch.y.size(0)
                total_loss += loss.item()
                
                # Store example prediction from first batch of final epoch
                if epoch == epochs - 1 and batch_idx == 0 and example_predictions['train'] is None:
                    example_predictions['train'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            # Validation
            model.eval()
            total_val_loss = 0
            total_val_mse = 0
            total_val_mae = 0
            total_val_nodes = 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    out = model(batch.x)  # MLP only uses node features
                    loss = criterion(out, batch.y.float())
                    mse = F.mse_loss(out, batch.y.float())
                    mae = F.l1_loss(out, batch.y.float())
                    total_val_loss += loss.item()
                    total_val_mse += mse.item()
                    total_val_mae += mae.item()
                    total_val_nodes += batch.y.size(0)
                    
                    # Store example prediction from first batch (final epoch)
                    if i == 0 and example_predictions['val'] is None:
                        example_predictions['val'] = {
                            'target': batch.y.float().cpu().numpy(),
                            'predicted': out.detach().cpu().numpy()
                        }
            
            train_mse = total_train_mse / len(train_loader) if len(train_loader) > 0 else 0
            train_mae = total_train_mae / len(train_loader) if len(train_loader) > 0 else 0
            val_mse = total_val_mse / len(val_loader) if len(val_loader) > 0 else 0
            val_mae = total_val_mae / len(val_loader) if len(val_loader) > 0 else 0
            avg_loss = total_loss / len(train_loader)
            
            if val_mse < best_val_loss:  # Lower MSE is better
                best_val_loss = val_mse
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}, "
                      f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, Time={epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        total_test_mse = 0
        total_test_mae = 0
        total_test_nodes = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                out = model(batch.x)  # MLP only uses node features
                mse = F.mse_loss(out, batch.y.float())
                mae = F.l1_loss(out, batch.y.float())
                total_test_mse += mse.item()
                total_test_mae += mae.item()
                total_test_nodes += batch.y.size(0)
                
                # Store example prediction from first batch (final epoch)
                if i == 0 and example_predictions['test'] is None:
                    example_predictions['test'] = {
                        'target': batch.y.float().cpu().numpy(),
                        'predicted': out.detach().cpu().numpy()
                    }
        
        final_test_mse = total_test_mse / len(test_loader) if len(test_loader) > 0 else 0
        final_test_mae = total_test_mae / len(test_loader) if len(test_loader) > 0 else 0
        
        return {
            'train_mse': train_mse,
            'val_mse': best_val_loss,
            'test_mse': final_test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': final_test_mae,
            'model': model,
            'avg_epoch_time': np.mean(training_times),
            'total_training_time': sum(training_times),
            'example_predictions': example_predictions
        }
    
    def run_comparison(self, graph_samples, epochs=100):
        """Run the complete model comparison."""
        print("=" * 80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)
        
        # Prepare data using the same approach as test_sheaf_precomputation.py
        message_passing_data, precomputed_data = self.prepare_data(graph_samples)
        
        # Get model parameters from precomputed data (same as test_sheaf_precomputation.py)
        first_graph = precomputed_data['inductive_data']['k_hop_community_counts_k2']['fold_0']['train']['graphs'][0]
        input_dim = first_graph.x.size(1)
        hidden_channels = 64
        output_dim = precomputed_data['inductive_data']['k_hop_community_counts_k2']['metadata']['output_dim']
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Output dimension: {output_dim}")
        print(f"Task: k_hop_community_counts_k2 (regression)")
        
        # Define models to test
        models = [
            # Message-passing models (regression versions)
            {
                'name': 'Message-Passing Sheaf (Orthogonal)',
                'type': 'message_passing',
                'model_class': NeuralSheafDiffusionRegression,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'd': 3,
                    'num_layers': 3,
                    'restriction_map_type': 'orthogonal',
                    'orthogonal_method': 'euler',
                    'activation': "elu",
                    'dropout': 0.3
                }
            },
            {
                'name': 'Message-Passing Sheaf (Diagonal)',
                'type': 'message_passing',
                'model_class': NeuralSheafDiffusionRegression,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'd': 3,
                    'num_layers': 3,
                    'restriction_map_type': 'diagonal',
                    'orthogonal_method': 'euler',
                    'activation': "elu",
                    'dropout': 0.3
                }
            },
            # {
            #     'name': 'Message-Passing Sheaf (General)',
            #     'type': 'message_passing',
            #     'model_class': NeuralSheafDiffusionRegression,
            #     'params': {
            #         'input_dim': input_dim,
            #         'hidden_channels': hidden_channels,
            #         'output_dim': output_dim,
            #         'd': 3,
            #         'num_layers': 3,
            #         'restriction_map_type': 'general',
            #         'orthogonal_method': 'euler',
            #         'activation': "elu",
            #         'dropout': 0.3
            #     }
            # },
            {
                'name': 'GCN',
                'type': 'message_passing',
                'model_class': GCNRegression,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            {
                'name': 'GraphSAGE (Mean)',
                'type': 'message_passing',
                'model_class': GraphSAGERegression,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'aggr': 'mean'
                }
            },
            {
                'name': 'BuNN (32 bundles)',
                'type': 'precomputed',
                'model_class': InductiveBuNNWrapper,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'd': 2,
                    'num_bundles': 32,
                    'num_layers': 3,
                    'bundle_method': 'rotation',
                    'heat_method': 'taylor',
                    'max_degree': 10,
                    'diffusion_times': None,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'graph_pooling': 'mean',
                    'activation': 'relu',
                    'device': self.device
                }
            },
            {
                'name': 'BuNN (16 bundles)',
                'type': 'precomputed',
                'model_class': InductiveBuNNWrapper,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'd': 2,
                    'num_bundles': 16,
                    'num_layers': 3,
                    'bundle_method': 'rotation',
                    'heat_method': 'taylor',
                    'max_degree': 10,
                    'diffusion_times': None,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'graph_pooling': 'mean',
                    'activation': 'relu',
                    'device': self.device
                }
            },
            # Precomputed models
            {
                'name': 'Precomputed Sheaf (Orthogonal)',
                'type': 'precomputed',
                'model_class': InductiveSheafDiffusionModel,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'sheaf_type': 'bundle',
                    'd': 3,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'device': self.device,
                    'normalised': True,
                    'deg_normalised': False,
                    'linear': False,
                    'left_weights': True,
                    'right_weights': True,
                    'sparse_learner': False,
                    'use_act': True,
                    'sheaf_act': "tanh",
                    'second_linear': False,
                    'orth': 'cayley',
                    'edge_weights': False,
                    'max_t': 1.0,
                    'add_lp': False,
                    'add_hp': False
                }
            },
            # Precomputed Continuous models
            {
                'name': 'Precomputed Continuous Sheaf (Diagonal) - Half Epochs',
                'type': 'precomputed_continuous',
                'model_class': InductiveContSheafDiffusionModel,
                'epochs': epochs // 2,  # Half epochs as requested
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'sheaf_type': 'diag',
                    'd': 3,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'device': self.device,
                    'normalised': True,
                    'deg_normalised': False,
                    'linear': False,
                    'left_weights': True,
                    'right_weights': True,
                    'sparse_learner': False,
                    'use_act': True,
                    'sheaf_act': "tanh",
                    'second_linear': False,
                    'orth': 'cayley',
                    'edge_weights': False,
                    'max_t': 1.0,
                    'add_lp': False,
                    'add_hp': False
                }
            },
            {
                'name': 'Precomputed Continuous Sheaf (Diagonal) - Full Epochs',
                'type': 'precomputed_continuous',
                'model_class': InductiveContSheafDiffusionModel,
                'epochs': epochs,  # Full epochs
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'sheaf_type': 'diag',
                    'd': 3,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'device': self.device,
                    'normalised': True,
                    'deg_normalised': False,
                    'linear': False,
                    'left_weights': True,
                    'right_weights': True,
                    'sparse_learner': False,
                    'use_act': True,
                    'sheaf_act': "tanh",
                    'second_linear': False,
                    'orth': 'cayley',
                    'edge_weights': False,
                    'max_t': 1.0,
                    'add_lp': False,
                    'add_hp': False
                }
            },
            {
                'name': 'Precomputed Continuous Sheaf (Orthogonal) - Half Epochs',
                'type': 'precomputed_continuous',
                'model_class': InductiveContSheafDiffusionModel,
                'epochs': epochs // 2,  # Half epochs as requested
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'sheaf_type': 'bundle',
                    'd': 3,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'device': self.device,
                    'normalised': True,
                    'deg_normalised': False,
                    'linear': False,
                    'left_weights': True,
                    'right_weights': True,
                    'sparse_learner': False,
                    'use_act': True,
                    'sheaf_act': "tanh",
                    'second_linear': False,
                    'orth': 'cayley',
                    'edge_weights': False,
                    'max_t': 1.0,
                    'add_lp': False,
                    'add_hp': False
                }
            },
            {
                'name': 'Precomputed Continuous Sheaf (Orthogonal) - Full Epochs',
                'type': 'precomputed_continuous',
                'model_class': InductiveContSheafDiffusionModel,
                'epochs': epochs,  # Full epochs
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_channels,
                    'output_dim': output_dim,
                    'sheaf_type': 'bundle',
                    'd': 3,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'input_dropout': 0.1,
                    'is_regression': True,
                    'is_graph_level_task': False,
                    'device': self.device,
                    'normalised': True,
                    'deg_normalised': False,
                    'linear': False,
                    'left_weights': True,
                    'right_weights': True,
                    'sparse_learner': False,
                    'use_act': True,
                    'sheaf_act': "tanh",
                    'second_linear': False,
                    'orth': 'cayley',
                    'edge_weights': False,
                    'max_t': 1.0,
                    'add_lp': False,
                    'add_hp': False
                }
            },
            # MLP baseline
            {
                'name': 'MLP Baseline',
                'type': 'mlp',
                'model_class': MLPBaseline,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }
        ]
        
        results = {}
        
        for model_config in models:
            print(f"\n" + "="*60)
            print(f"Testing {model_config['name']}")
            print("="*60)
            
            try:
                # Create model
                model = model_config['model_class'](**model_config['params'])
                
                # Get custom epochs for continuous models, otherwise use default
                custom_epochs = model_config.get('epochs', epochs)
                
                # Train model
                if model_config['type'] == 'message_passing':
                    result = self.train_message_passing_model(
                        model, precomputed_data, model_config['name'], epochs=custom_epochs
                    )
                elif model_config['type'] == 'mlp':
                    result = self.train_mlp_model(
                        model, precomputed_data, model_config['name'], epochs=custom_epochs
                    )
                elif model_config['type'] == 'precomputed_continuous':
                    result = self.train_precomputed_model(
                        model, precomputed_data, model_config['name'], epochs=custom_epochs
                    )
                else:  # precomputed
                    result = self.train_precomputed_model(
                        model, precomputed_data, model_config['name'], epochs=custom_epochs
                    )
                
                results[model_config['name']] = result
                print(f"✓ {model_config['name']} completed successfully")
                
            except Exception as e:
                print(f"✗ Error with {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                results[model_config['name']] = {'error': str(e)}
        
        return results
    
    def print_results(self, results):
        """Print comprehensive results comparison."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS COMPARISON")
        print("="*80)
        
        # Create results table
        print(f"{'Model':<35} {'Train MSE':<10} {'Val MSE':<10} {'Test MSE':<10} {'Train MAE':<10} {'Val MAE':<10} {'Test MAE':<10} {'Avg Epoch':<10} {'Total Time':<12}")
        print("-" * 107)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"{name:<35} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
            else:
                train_mse = f"{result['train_mse']:.4f}"
                val_mse = f"{result['val_mse']:.4f}"
                test_mse = f"{result['test_mse']:.4f}"
                train_mae = f"{result['train_mae']:.4f}"
                val_mae = f"{result['val_mae']:.4f}"
                test_mae = f"{result['test_mae']:.4f}"
                avg_epoch = f"{result['avg_epoch_time']:.2f}s"
                total_time = f"{result['total_training_time']:.1f}s"
                
                print(f"{name:<35} {train_mse:<10} {val_mse:<10} {test_mse:<10} {train_mae:<10} {val_mae:<10} {test_mae:<10} {avg_epoch:<10} {total_time:<12}")
        
        # Performance analysis
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Find best performing models
        valid_results = {name: result for name, result in results.items() if 'error' not in result}
        
        if valid_results:
            # Best test MSE (lower is better)
            best_test = min(valid_results.items(), key=lambda x: x[1]['test_mse'])
            print(f"Best Test MSE: {best_test[0]} ({best_test[1]['test_mse']:.4f})")
            
            # Fastest training
            fastest = min(valid_results.items(), key=lambda x: x[1]['total_training_time'])
            print(f"Fastest Training: {fastest[0]} ({fastest[1]['total_training_time']:.1f}s)")
            
            # Best validation MSE (lower is better)
            best_val = min(valid_results.items(), key=lambda x: x[1]['val_mse'])
            print(f"Best Validation MSE: {best_val[0]} ({best_val[1]['val_mse']:.4f})")
            
            # Speed vs MSE trade-off
            print("\nSpeed vs MSE Trade-off:")
            for name, result in valid_results.items():
                speed_score = 1000 / result['total_training_time']  # Higher is faster
                mse_score = result['test_mse']  # Lower is better
                print(f"  {name}: Speed={speed_score:.1f}, MSE={mse_score:.4f}")
        
        # Example predictions comparison
        print("\n" + "="*80)
        print("DETAILED EXAMPLE PREDICTIONS COMPARISON")
        print("="*80)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"\n{name}: ERROR - {result['error']}")
                continue
                
            if 'example_predictions' not in result:
                print(f"\n{name}: No example predictions available")
                continue
                
            print(f"\n{name}:")
            print("-" * 60)
            
            example_predictions = result['example_predictions']
            
            for split in ['train', 'val', 'test']:
                if example_predictions[split] is not None:
                    target = example_predictions[split]['target']
                    predicted = example_predictions[split]['predicted']
                    
                    print(f"\n{split.upper()} SPLIT:")
                    print(f"  Target shape: {target.shape}")
                    print(f"  Predicted shape: {predicted.shape}")
                    
                    # Show first few nodes
                    num_nodes_to_show = min(3, target.shape[0])
                    for node_idx in range(num_nodes_to_show):
                        print(f"  Node {node_idx}:")
                        print(f"    Target:     {target[node_idx]}")
                        print(f"    Predicted:  {predicted[node_idx]}")
                        
                        # Calculate per-node metrics
                        node_mse = np.mean((target[node_idx] - predicted[node_idx])**2)
                        node_mae = np.mean(np.abs(target[node_idx] - predicted[node_idx]))
                        print(f"    MSE: {node_mse:.4f}, MAE: {node_mae:.4f}")
                    
                    # Overall metrics for this split
                    overall_mse = np.mean((target - predicted)**2)
                    overall_mae = np.mean(np.abs(target - predicted))
                    print(f"  Overall MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}")
                    
                    # Check if predictions are in reasonable range
                    target_range = target.max() - target.min()
                    pred_range = predicted.max() - predicted.min()
                    print(f"  Target range: {target.min():.2f} to {target.max():.2f} (range: {target_range:.2f})")
                    print(f"  Pred range:   {predicted.min():.2f} to {predicted.max():.2f} (range: {pred_range:.2f})")
                    
                    if pred_range < target_range * 0.1:
                        print(f"  WARNING: Predictions have much smaller range than targets!")
                else:
                    print(f"\n{split.upper()} SPLIT: No data available")
        
        # Model type comparison
        print("="*80)
        print("MODEL TYPE COMPARISON")
        print("="*80)
        
        message_passing_models = [name for name in results.keys() if 'Message-Passing' in name or 'GCN' in name or 'GraphSAGE' in name]
        precomputed_models = [name for name in results.keys() if 'Precomputed' in name]
        bunn_models = [name for name in results.keys() if 'BuNN' in name]
        mlp_models = [name for name in results.keys() if 'MLP' in name]
        
        if message_passing_models and precomputed_models:
            mp_results = {name: results[name] for name in message_passing_models if 'error' not in results[name]}
            pc_results = {name: results[name] for name in precomputed_models if 'error' not in results[name]}
            bunn_results = {name: results[name] for name in bunn_models if 'error' not in results[name]}
            mlp_results = {name: results[name] for name in mlp_models if 'error' not in results[name]}
            
            if mp_results:
                mp_avg_test = np.mean([r['test_mse'] for r in mp_results.values()])
                mp_avg_time = np.mean([r['total_training_time'] for r in mp_results.values()])
                print(f"Message-Passing Models (avg): Test MSE={mp_avg_test:.4f}, Time={mp_avg_time:.1f}s")
            
            if pc_results:
                pc_avg_test = np.mean([r['test_mse'] for r in pc_results.values()])
                pc_avg_time = np.mean([r['total_training_time'] for r in pc_results.values()])
                print(f"Precomputed Models (avg): Test MSE={pc_avg_test:.4f}, Time={pc_avg_time:.1f}s")
            
            if bunn_results:
                bunn_avg_test = np.mean([r['test_mse'] for r in bunn_results.values()])
                bunn_avg_time = np.mean([r['total_training_time'] for r in bunn_results.values()])
                print(f"BuNN Models (avg): Test MSE={bunn_avg_test:.4f}, Time={bunn_avg_time:.1f}s")
            
            if mlp_results:
                mlp_avg_test = np.mean([r['test_mse'] for r in mlp_results.values()])
                mlp_avg_time = np.mean([r['total_training_time'] for r in mlp_results.values()])
                print(f"MLP Baseline (avg): Test MSE={mlp_avg_test:.4f}, Time={mlp_avg_time:.1f}s")
            
            # Compare graph models vs MLP baseline
            if mp_results and mlp_results:
                mp_avg_test = np.mean([r['test_mse'] for r in mp_results.values()])
                mlp_avg_test = np.mean([r['test_mse'] for r in mlp_results.values()])
                if mlp_avg_test < mp_avg_test:
                    improvement = (mp_avg_test - mlp_avg_test) / mp_avg_test * 100
                    print(f"Graph models improve over MLP baseline by {improvement:.1f}%")
                else:
                    degradation = (mlp_avg_test - mp_avg_test) / mlp_avg_test * 100
                    print(f"MLP baseline outperforms graph models by {degradation:.1f}%")
            
            if pc_results and mlp_results:
                pc_avg_test = np.mean([r['test_mse'] for r in pc_results.values()])
                mlp_avg_test = np.mean([r['test_mse'] for r in mlp_results.values()])
                if mlp_avg_test < pc_avg_test:
                    improvement = (pc_avg_test - mlp_avg_test) / pc_avg_test * 100
                    print(f"Precomputed models improve over MLP baseline by {improvement:.1f}%")
                else:
                    degradation = (mlp_avg_test - pc_avg_test) / mlp_avg_test * 100
                    print(f"MLP baseline outperforms precomputed models by {degradation:.1f}%")
            
            # Speed comparison
            if mp_results and pc_results:
                mp_avg_time = np.mean([r['total_training_time'] for r in mp_results.values()])
                pc_avg_time = np.mean([r['total_training_time'] for r in pc_results.values()])
                if pc_avg_time < mp_avg_time:
                    speedup = mp_avg_time / pc_avg_time
                    print(f"Precomputation provides {speedup:.2f}x speedup on average")
                else:
                    slowdown = pc_avg_time / mp_avg_time
                    print(f"Precomputation is {slowdown:.2f}x slower on average")

            if bunn_results and mlp_results:
                bunn_avg_test = np.mean([r['test_mse'] for r in bunn_results.values()])
                mlp_avg_test = np.mean([r['test_mse'] for r in mlp_results.values()])
                if mlp_avg_test < bunn_avg_test:
                    improvement = (bunn_avg_test - mlp_avg_test) / bunn_avg_test * 100
                    print(f"BuNN models improve over MLP baseline by {improvement:.1f}%")
                else:
                    degradation = (mlp_avg_test - bunn_avg_test) / mlp_avg_test * 100
                    print(f"MLP baseline outperforms BuNN models by {degradation:.1f}%")

            if bunn_results and pc_results:
                bunn_avg_time = np.mean([r['total_training_time'] for r in bunn_results.values()])
                pc_avg_time = np.mean([r['total_training_time'] for r in pc_results.values()])
                if pc_avg_time < bunn_avg_time:
                    speedup = bunn_avg_time / pc_avg_time
                    print(f"Precomputed models are {speedup:.2f}x faster than BuNN models on average")
                else:
                    slowdown = pc_avg_time / bunn_avg_time
                    print(f"BuNN models are {slowdown:.2f}x faster than precomputed models on average")
    
    def create_visualizations(self, results):
        """Create visualizations of the results."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        # Filter out errors
        valid_results = {name: result for name, result in results.items() if 'error' not in result}
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Test MSE comparison
        names = list(valid_results.keys())
        test_mses = [valid_results[name]['test_mse'] for name in names]
        
        ax1.bar(range(len(names)), test_mses, color='skyblue', alpha=0.7)
        ax1.set_title('Test MSE Comparison')
        ax1.set_ylabel('Test MSE')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(test_mses):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Test MAE comparison
        test_maes = [valid_results[name]['test_mae'] for name in names]
        
        ax2.bar(range(len(names)), test_maes, color='lightgreen', alpha=0.7)
        ax2.set_title('Test MAE Comparison')
        ax2.set_ylabel('Test MAE')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(test_maes):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Speed vs MSE scatter plot
        training_times = [valid_results[name]['total_training_time'] for name in names]
        
        ax3.scatter(training_times, test_mses, s=100, alpha=0.7)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Test MSE')
        ax3.set_title('Speed vs MSE Trade-off')
        
        # Add model names as annotations
        for i, name in enumerate(names):
            ax3.annotate(name, (training_times[i], test_mses[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. MSE vs MAE scatter plot
        ax4.scatter(test_mses, test_maes, s=100, alpha=0.7)
        ax4.set_xlabel('Test MSE')
        ax4.set_ylabel('Test MAE')
        ax4.set_title('MSE vs MAE Comparison')
        
        # Add model names as annotations
        for i, name in enumerate(names):
            ax4.annotate(name, (test_mses[i], test_maes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'model_comparison_results.png'")
        
        # Create summary table
        print("\nSummary Table:")
        print("-" * 120)
        print(f"{'Model':<35} {'Val MSE':<10} {'Test MSE':<10} {'Val MAE':<10} {'Test MAE':<10} {'Train Time':<12} {'Epoch Time':<12}")
        print("-" * 120)
        
        for name in names:
            result = valid_results[name]
            val_mse = f"{result['val_mse']:.4f}"
            test_mse = f"{result['test_mse']:.4f}"
            val_mae = f"{result['val_mae']:.4f}"
            test_mae = f"{result['test_mae']:.4f}"
            train_time = f"{result['total_training_time']:.1f}s"
            epoch_time = f"{result['avg_epoch_time']:.2f}s"
            
            print(f"{name:<35} {val_mse:<10} {test_mse:<10} {val_mae:<10} {test_mae:<10} {train_time:<12} {epoch_time:<12}")


def main():
    """Main function to run the comparison."""
    print("=" * 80)
    print("SHEAF MODEL COMPARISON EXPERIMENT")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load original GraphSample objects (not converted to PyG yet)
        graph_samples = load_graphs()
        
        # Create comparison object
        comparison = ModelComparison(device=device)
        
        # Run comparison with original graph samples
        results = comparison.run_comparison(graph_samples, epochs=100)  # Reduced epochs for faster testing
        
        # Print results
        comparison.print_results(results)
        
        # Create visualizations
        comparison.create_visualizations(results)
        
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