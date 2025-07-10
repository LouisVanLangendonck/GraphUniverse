"""
CORA Dataset Benchmark for Neural Sheaf Diffusion Model

This script benchmarks the neural sheaf diffusion model on the CORA dataset
with positional encoding and hyperparameter optimization using Optuna.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import os
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, degree
from scipy.sparse.linalg import eigsh
from collections import defaultdict
import optuna
from optuna import create_study, Trial
import copy
import random

# Import our neural sheaf diffusion model
from nsd_mp import SheafDiffusionModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrthogonalMaps(nn.Module):
    """
    Creates orthogonal restriction maps using different parameterizations.
    Based on the Orthogonal class from the paper's codebase.
    """
    def __init__(self, d: int, method: str = "householder"):
        super().__init__()
        self.d = d
        self.method = method
        
        # Initialize timing statistics
        self.timing_stats = defaultdict(float)
        self.timing_counts = defaultdict(int)
        
        if method == "householder":
            # Use Householder reflections for orthogonal matrices
            # Need d(d-1)/2 parameters for d x d orthogonal matrix
            self.num_params = d * (d - 1) // 2
        elif method == "euler":
            # Euler angles (only works for d=2,3)
            if d == 2:
                self.num_params = 1
            elif d == 3:
                self.num_params = 3
            else:
                raise ValueError("Euler method only supports d=2 or d=3")
        else:
            raise ValueError(f"Unknown orthogonal method: {method}")
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameters to orthogonal matrices.
        
        Args:
            params: [..., num_params] tensor of parameters
            
        Returns:
            [..., d, d] tensor of orthogonal matrices
        """
        batch_dims = params.shape[:-1]
        
        if self.method == "householder":
            return self._householder_transform(params)
        elif self.method == "euler":
            if self.d == 2:
                return self._euler_2d(params)
            elif self.d == 3:
                return self._euler_3d(params)
    
    def _householder_transform(self, params: torch.Tensor) -> torch.Tensor:
        """Householder reflection based orthogonal matrix generation."""
        batch_dims = params.shape[:-1]
        device = params.device
        
        # Time matrix creation
        start_time = time.time()
        tril_indices = torch.tril_indices(self.d, self.d, offset=-1, device=device)
        A = torch.zeros(*batch_dims, self.d, self.d, device=device)
        A[..., tril_indices[0], tril_indices[1]] = params
        eye = torch.eye(self.d, device=device).expand(*batch_dims, -1, -1)
        A = A + eye
        self.timing_stats['householder_matrix_setup'] += time.time() - start_time
        self.timing_counts['householder_matrix_setup'] += 1
        
        # Time QR decomposition (most expensive part)
        start_time = time.time()
        Q, _ = torch.linalg.qr(A)
        self.timing_stats['householder_qr'] += time.time() - start_time
        self.timing_counts['householder_qr'] += 1
        
        return Q
    
    def _euler_2d(self, params: torch.Tensor) -> torch.Tensor:
        """2D rotation matrices from angle parameter."""
        angles = params[..., 0] * 2 * math.pi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        batch_dims = params.shape[:-1]
        Q = torch.zeros(*batch_dims, 2, 2, device=params.device)
        Q[..., 0, 0] = cos_a
        Q[..., 0, 1] = -sin_a
        Q[..., 1, 0] = sin_a
        Q[..., 1, 1] = cos_a
        
        return Q
    
    def _euler_3d(self, params: torch.Tensor) -> torch.Tensor:
        """3D rotation matrices from Euler angles."""
        alpha = params[..., 0] * 2 * math.pi
        beta = params[..., 1] * 2 * math.pi
        gamma = params[..., 2] * 2 * math.pi
        
        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta), torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)
        
        batch_dims = params.shape[:-1]
        Q = torch.zeros(*batch_dims, 3, 3, device=params.device)
        
        Q[..., 0, 0] = cos_a * cos_b
        Q[..., 0, 1] = cos_a * sin_b * sin_g - sin_a * cos_g
        Q[..., 0, 2] = cos_a * sin_b * cos_g + sin_a * sin_g
        Q[..., 1, 0] = sin_a * cos_b
        Q[..., 1, 1] = sin_a * sin_b * sin_g + cos_a * cos_g
        Q[..., 1, 2] = sin_a * sin_b * cos_g - cos_a * sin_g
        Q[..., 2, 0] = -sin_b
        Q[..., 2, 1] = cos_b * sin_g
        Q[..., 2, 2] = cos_b * cos_g
        
        return Q

class PositionalEncodingComputer:
    """Compute various types of positional encodings for graphs."""
    
    def __init__(self, max_pe_dim: int = 16, pe_types: List[str] = None):
        """
        Initialize PE computer.
        
        Args:
            max_pe_dim: Maximum PE dimension
            pe_types: List of PE types to compute ['laplacian', 'degree', 'rwse']
        """
        self.max_pe_dim = max_pe_dim
        self.pe_types = pe_types or ['laplacian']
    
    def compute_degree_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Degree-based PE - most transferable across graphs."""
        degrees = degree(edge_index[0], num_nodes=num_nodes).float()
        pe = torch.zeros(num_nodes, self.max_pe_dim)
        
        for i in range(min(self.max_pe_dim, 8)):
            pe[:, i] = (degrees ** (i / 4.0)) / (1 + degrees ** (i / 4.0))
        
        return pe
    
    def compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Random Walk Structural Encoding - landing probabilities after k steps."""
        try:
            # Get node degrees
            degrees = degree(edge_index[0], num_nodes=num_nodes).float()
            
            # Handle isolated nodes
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
            
            # Create adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            # Transition matrix: P[i,j] = A[i,j] / degree[i]
            P = adj / degrees.unsqueeze(1)
            
            # Compute powers of transition matrix for different walk lengths
            rwse = torch.zeros(num_nodes, self.max_pe_dim)
            P_power = torch.eye(num_nodes)  # P^0 = I
            
            for k in range(self.max_pe_dim):
                if k > 0:
                    P_power = P_power @ P  # P^k
                
                # Use diagonal entries (return probabilities) as features
                rwse[:, k] = P_power.diag()
            
            return rwse
            
        except Exception as e:
            print(f"Warning: RWSE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)
    
    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Laplacian Positional Encoding using eigenvectors."""
        try:
            # Handle empty/trivial graphs
            if edge_index.shape[1] == 0 or num_nodes <= 1:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            # Get normalized Laplacian
            edge_index_lap, edge_weight = get_laplacian(
                edge_index, 
                edge_weight=None,
                normalization='sym', 
                num_nodes=num_nodes
            )
            
            # Convert to scipy sparse matrix
            L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
            
            # Compute eigenvalues/eigenvectors
            k = min(self.max_pe_dim, num_nodes - 2)
            if k <= 0:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            try:
                eigenvals, eigenvecs = eigsh(
                    L, 
                    k=k, 
                    which='SM',  # Smallest eigenvalues
                    return_eigenvectors=True,
                    tol=1e-6
                )
            except:
                # Fallback for small graphs
                L_dense = L.toarray()
                eigenvals, eigenvecs = np.linalg.eigh(L_dense)
                idx = np.argsort(eigenvals)
                eigenvecs = eigenvecs[:, idx[1:k+1]]  # Skip first (constant) eigenvector
            
            # Handle sign ambiguity
            for i in range(eigenvecs.shape[1]):
                if eigenvecs[0, i] < 0:
                    eigenvecs[:, i] *= -1
            
            # Pad or truncate to max_pe_dim
            if eigenvecs.shape[1] < self.max_pe_dim:
                pad_width = self.max_pe_dim - eigenvecs.shape[1]
                eigenvecs = np.pad(eigenvecs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                eigenvecs = eigenvecs[:, :self.max_pe_dim]
            
            return torch.tensor(eigenvecs, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: Laplacian PE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)
    
    def compute_all_pe(self, edge_index: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
        """Compute all requested PE types."""
        pe_dict = {}
        
        for pe_type in self.pe_types:
            if pe_type == 'laplacian':
                pe = self.compute_laplacian_pe(edge_index, num_nodes)
                pe_dict['laplacian_pe'] = pe
                    
            elif pe_type == 'degree':
                pe = self.compute_degree_pe(edge_index, num_nodes)
                pe_dict['degree_pe'] = pe
                    
            elif pe_type == 'rwse':
                pe = self.compute_rwse(edge_index, num_nodes)
                pe_dict['rwse_pe'] = pe
                    
            elif pe_type == 'random_walk':  # Legacy support
                pe = self.compute_degree_pe(edge_index, num_nodes)  # Use degree PE as fallback
                pe_dict['random_walk_pe'] = pe
        
        return pe_dict

def add_positional_encodings_to_data(data: Data, pe_types: List[str] = ['laplacian', 'degree', 'rwse'], max_pe_dim: int = 16) -> Data:
    """
    Add precomputed positional encodings to a PyG data object.
    
    Args:
        data: PyG Data object
        pe_types: Types of PE to compute ['laplacian', 'degree', 'rwse']
        max_pe_dim: Maximum PE dimension
    
    Returns:
        Updated data object with PE added
    """
    pe_computer = PositionalEncodingComputer(
        max_pe_dim=max_pe_dim, 
        pe_types=pe_types
    )
    
    # Compute PE for this graph
    pe_dict = pe_computer.compute_all_pe(data.edge_index, data.x.size(0))
    
    # Add PE to data object
    for pe_name, pe_tensor in pe_dict.items():
        setattr(data, pe_name, pe_tensor)
    
    return data

def load_cora_dataset(root: str = './data', split_idx: int = 0) -> Tuple[Data, int, int]:
    """
    Load CORA dataset with geom-gcn split and positional encoding.
    
    Args:
        root: Root directory for dataset
        split_idx: Index of the split to use (0-9 for geom-gcn)
        
    Returns:
        Tuple of (data, num_classes, num_features)
    """
    print("Loading CORA dataset with geom-gcn split...")
    
    # Load dataset with geom-gcn split
    dataset = Planetoid(root=root, name='Cora', split='geom-gcn')
    data = dataset[0]
    
    # Select a specific split from the 10 available splits
    print(f"Using split {split_idx} from geom-gcn splits")
    data.train_mask = data.train_mask[:, split_idx]
    data.val_mask = data.val_mask[:, split_idx]
    data.test_mask = data.test_mask[:, split_idx]
    
    print(f"CORA dataset loaded with geom-gcn split {split_idx}:")
    print(f"  Nodes: {data.x.size(0)}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Training nodes: {data.train_mask.sum()}")
    print(f"  Validation nodes: {data.val_mask.sum()}")
    print(f"  Test nodes: {data.test_mask.sum()}")
    
    # Add positional encodings
    print("Adding positional encodings...")
    data = add_positional_encodings_to_data(
        data, 
        pe_types=['laplacian', 'degree', 'rwse'],
        max_pe_dim=16
    )
    
    # Debug: Print data structure information
    print(f"Data structure debug info:")
    print(f"  train_mask shape: {data.train_mask.shape}")
    print(f"  val_mask shape: {data.val_mask.shape}")
    print(f"  test_mask shape: {data.test_mask.shape}")
    print(f"  y shape: {data.y.shape}")
    print(f"  x shape: {data.x.shape}")
    
    return data, dataset.num_classes, data.x.size(1)

def create_cora_dataloaders(data: Data, batch_size: int = 1) -> Dict[str, DataLoader]:
    """
    Create dataloaders for CORA dataset using the geom-gcn train/val/test splits.
    
    Args:
        data: CORA data object with geom-gcn splits
        batch_size: Batch size (typically 1 for node classification)
        
    Returns:
        Dictionary with train/val/test dataloaders
    """
    from torch_geometric.loader import DataLoader
    
    # Create separate datasets for each split
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
    
    # Use the geom-gcn splits directly (already properly formatted as 1D masks)
    train_data.train_mask = data.train_mask
    train_data.val_mask = data.val_mask
    train_data.test_mask = data.test_mask
    
    val_data.train_mask = data.train_mask
    val_data.val_mask = data.val_mask
    val_data.test_mask = data.test_mask
    
    test_data.train_mask = data.train_mask
    test_data.val_mask = data.val_mask
    test_data.test_mask = data.test_mask
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader([train_data], batch_size=batch_size, shuffle=True),
        'val': DataLoader([val_data], batch_size=batch_size, shuffle=False),
        'test': DataLoader([test_data], batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders



def compute_metrics_gpu(y_true: torch.Tensor, y_pred: torch.Tensor, is_regression: bool = False) -> Dict[str, float]:
    """
    Compute metrics on GPU for efficiency.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary of metrics
    """
    if is_regression:
        # Regression metrics
        mse = torch.mean((y_true - y_pred) ** 2)
        mae = torch.mean(torch.abs(y_true - y_pred))
        
        # R² score
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'r2': r2.item()
        }
    else:
        # Classification metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Move to CPU for sklearn metrics
        y_true_cpu = y_true.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        accuracy = accuracy_score(y_true_cpu, y_pred_cpu)
        f1_macro = f1_score(y_true_cpu, y_pred_cpu, average='macro')
        f1_micro = f1_score(y_true_cpu, y_pred_cpu, average='micro')
        precision_macro = precision_score(y_true_cpu, y_pred_cpu, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_cpu, y_pred_cpu, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }

def train_sheaf_model(
    model: SheafDiffusionModel,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int = 500,
    patience: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    param_optization: bool = False
) -> Dict[str, Any]:
    """
    Train the neural sheaf diffusion model on CORA.
    
    Args:
        model: Sheaf diffusion model
        dataloaders: Dictionary with train/val/test dataloaders
        device: Device to use for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Dictionary with training results
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"Training for {epochs} epochs with patience {patience}")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in dataloaders['train']:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, graph=batch)
            
            # Only use training nodes for loss
            train_mask = batch.train_mask
            if train_mask.sum() > 0:
                loss = criterion(out[train_mask], batch.y[train_mask])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(out[train_mask], 1)
                train_total += train_mask.sum().item()
                train_correct += (predicted == batch.y[train_mask]).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in dataloaders['val']:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, graph=batch)
                
                # Only use validation nodes
                val_mask = batch.val_mask
                if val_mask.sum() > 0:
                    loss = criterion(out[val_mask], batch.y[val_mask])
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(out[val_mask], 1)
                    val_total += val_mask.sum().item()
                    val_correct += (predicted == batch.y[val_mask]).sum().item()
        
        # Calculate metrics
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
    
    # Test evaluation
    if not param_optization:
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in dataloaders['test']:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, graph=batch)
                
                # Only use test nodes
                test_mask = batch.test_mask
                if test_mask.sum() > 0:
                    _, predicted = torch.max(out[test_mask], 1)
                    test_total += test_mask.sum().item()
                    test_correct += (predicted == batch.y[test_mask]).sum().item()
        
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        print(f"Final test accuracy: {test_accuracy:.4f}")
    else:
        test_accuracy = 0.0
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_accuracy,
        'final_epoch': epoch
    }

def optimize_hyperparameters(
    data: Data,
    num_classes: int,
    num_features: int,
    device: torch.device,
    n_trials: int = 50,
    timeout: Optional[int] = 1800,  # 30 minutes
    pe_types: List[str] = ['laplacian', 'degree', 'rwse']
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        data: CORA data object
        num_classes: Number of classes
        num_features: Number of features
        device: Device to use
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        pe_types: Positional encoding types to try
        
    Returns:
        Dictionary with optimization results
    """
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    def objective(trial: Trial) -> float:
        # Hyperparameters to optimize
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        sheaf_type = trial.suggest_categorical('sheaf_type', ['diagonal', 'orthogonal', 'general'])
        
        # Constrain d based on sheaf_type
        if sheaf_type == 'orthogonal':
            d = trial.suggest_int('d', 2, 3)  # Euler method only supports d=2,3
        else:
            d = trial.suggest_int('d', 2, 4)  # Other methods support d=2,3,4
            
        pe_type = trial.suggest_categorical('pe_type', [None] + pe_types)
        pe_dim = trial.suggest_int('pe_dim', 8, 32)
        
        # Ensure hidden_dim is divisible by d
        if hidden_dim % d != 0:
            hidden_dim = (hidden_dim // d) * d
        
        # Validate sheaf_type and d compatibility
        if sheaf_type == 'orthogonal' and d > 3:
            raise ValueError(f"Orthogonal sheaf type only supports d <= 3, got d={d}")
        
        # Create model
        model = SheafDiffusionModel(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            d=d,
            num_layers=num_layers,
            sheaf_type=sheaf_type,
            activation='elu',
            dropout=dropout,
            is_regression=False,
            is_graph_level_task=False,
            pe_type=pe_type,
            pe_dim=pe_dim
        ).to(device)
        
        # Create dataloaders using geom-gcn splits
        dataloaders = create_cora_dataloaders(data, batch_size=1)
        
        # Quick training for optimization
        results = train_sheaf_model(
            model=model,
            dataloaders=dataloaders,
            device=device,
            epochs=100,  # Shorter for optimization
            patience=100,
            lr=lr,
            weight_decay=weight_decay
        )
        
        return results['best_val_accuracy']
    
    # Create study
    study = create_study(direction='maximize')
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Optimization completed!")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'study': study
    }

def main():
    """Main function to run CORA benchmark."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CORA dataset
    data, num_classes, num_features = load_cora_dataset()
    
    # Move data to device
    data = data.to(device)
    
    # Hyperparameter optimization
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION (using geom-gcn splits)")
    print("="*50)
    
    opt_results = optimize_hyperparameters(
        data=data,
        num_classes=num_classes,
        num_features=num_features,
        device=device,
        n_trials=30,  # Reduced for faster execution
        timeout=900,  # 15 minutes
        pe_types=['laplacian', 'degree', 'rwse']
    )
    
    # Final training with best parameters
    print("\n" + "="*50)
    print("FINAL TRAINING WITH BEST PARAMETERS (using geom-gcn splits)")
    print("="*50)
    
    best_params = opt_results['best_params']
    
    # Create final model with best parameters
    final_model = SheafDiffusionModel(
        input_dim=num_features,
        hidden_dim=best_params['hidden_dim'],
        output_dim=num_classes,
        d=best_params['d'],
        num_layers=best_params['num_layers'],
        sheaf_type=best_params['sheaf_type'],
        activation='elu',
        dropout=best_params['dropout'],
        is_regression=False,
        is_graph_level_task=False,
        pe_type=best_params['pe_type'],
        pe_dim=best_params['pe_dim']
    ).to(device)
    
    # Create dataloaders using geom-gcn splits
    dataloaders = create_cora_dataloaders(data, batch_size=1)
    
    # Train final model
    final_results = train_sheaf_model(
        model=final_model,
        dataloaders=dataloaders,
        device=device,
        epochs=200,
        patience=50,
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best validation accuracy: {final_results['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Training completed in {final_results['final_epoch']} epochs")
    print(f"Best hyperparameters: {best_params}")
    
    # Save results
    results = {
        'optimization_results': opt_results,
        'final_results': final_results,
        'best_params': best_params,
        'model_config': {
            'input_dim': num_features,
            'hidden_dim': best_params['hidden_dim'],
            'output_dim': num_classes,
            'd': best_params['d'],
            'num_layers': best_params['num_layers'],
            'sheaf_type': best_params['sheaf_type'],
            'pe_type': best_params['pe_type'],
            'pe_dim': best_params['pe_dim']
        }
    }
    
    # Save to file
    import json
    with open('cora_benchmark_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"\nResults saved to 'cora_benchmark_results.json'")
    
    return results

if __name__ == "__main__":
    main() 