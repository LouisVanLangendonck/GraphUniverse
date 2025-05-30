"""
Training utilities for graph learning experiments with hyperparameter optimization.

This module provides functions for training and evaluating models on graph data,
including hyperparameter optimization capabilities using Bayesian Optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# Import for hyperparameter optimization
import optuna
from optuna import create_study, Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from experiments.core.metrics import (
    compute_metrics,
    compute_loss,
    compute_accuracy,
    evaluate_node_classification,
    model_performance_summary,
    compute_classification_metrics,
    compute_regression_metrics
)
from experiments.core.config import ExperimentConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiments.core.models import GNNModel, MLPModel, SklearnModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_hyperparameters(
    model_creator: Callable,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    model_type: str = "gnn",
    gnn_type: Optional[str] = None,
    n_trials: int = 20,
    max_epochs: int = 200,
    timeout: Optional[int] = 600,  # 10 minutes per optimization
    device: Optional[torch.device] = None,
    is_regression: bool = False
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model using Optuna.
    For classification tasks, maximizes F1 score.
    For regression tasks, minimizes MSE.
    """
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device once
    features = features.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    # Enable mixed precision training if using CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def objective(trial):
        # Common hyperparameters for all models
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = trial.suggest_int('patience', 10, 50)
        
        # Model-specific hyperparameters
        if model_type == "gnn":
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.7)
            
            # GNN-specific parameters
            if gnn_type == "gat":
                heads = trial.suggest_int('heads', 1, 8)
                concat_heads = trial.suggest_categorical('concat_heads', [True, False])
            else:
                heads = 1
                concat_heads = True
            
            residual = trial.suggest_categorical('residual', [True, False])
            norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
            agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
            
            # Create GNN model
            model = model_creator(
                input_dim=features.shape[1],
                hidden_dim=hidden_dim,
                output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type,
                residual=residual,
                norm_type=norm_type,
                agg_type=agg_type,
                heads=heads,
                concat_heads=concat_heads,
                is_regression=is_regression
            ).to(device)
            
        elif model_type == "mlp":
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.7)
            
            # Create MLP model
            model = model_creator(
                input_dim=features.shape[1],
                hidden_dim=hidden_dim,
                output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
                num_layers=num_layers,
                dropout=dropout,
                is_regression=is_regression
            ).to(device)
            
        else:  # sklearn model
            # Create sklearn model with required parameters
            model = model_creator(
                input_dim=features.shape[1],
                output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
                is_regression=is_regression
            )
        
        # Initialize optimizer for PyTorch models
        if model_type in ["gnn", "mlp"]:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.MSELoss() if is_regression else torch.nn.CrossEntropyLoss()
            
            # Training loop
            best_val_metric = float('inf') if is_regression else 0.0
            patience_counter = 0
            
            for epoch in range(max_epochs):
                model.train()
                optimizer.zero_grad()
                
                # Use mixed precision training if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(features, edge_index) if model_type == "gnn" else model(features)
                        if is_regression:
                            # For regression, ensure output matches target shape
                            loss = criterion(out[train_idx], labels[train_idx])
                        else:
                            loss = criterion(out[train_idx], labels[train_idx])
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(features, edge_index) if model_type == "gnn" else model(features)
                    if is_regression:
                        # For regression, ensure output matches target shape
                        loss = criterion(out[train_idx], labels[train_idx])
                    else:
                        loss = criterion(out[train_idx], labels[train_idx])
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    out = model(features, edge_index) if model_type == "gnn" else model(features)
                    val_loss = criterion(out[val_idx], labels[val_idx])
                    
                    if is_regression:
                        # For regression, use MSE as the metric (lower is better)
                        val_metric = val_loss.item()
                    else:
                        # For classification, use F1 score (higher is better)
                        val_pred = out[val_idx].argmax(dim=1)
                        val_true = labels[val_idx]
                        val_metric = compute_metrics(val_true.cpu().numpy(), val_pred.cpu().numpy())['f1_macro']
                
                # Early stopping
                if is_regression:
                    # For regression, lower MSE is better
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    # For classification, higher F1 is better
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            # For regression, return negative MSE (to maximize)
            # For classification, return F1 score (already maximizing)
            return -best_val_metric if is_regression else best_val_metric
            
        else:  # sklearn model
            # Convert data to numpy for sklearn
            X_train = features[train_idx].cpu().numpy()
            y_train = labels[train_idx].cpu().numpy()
            X_val = features[val_idx].cpu().numpy()
            y_val = labels[val_idx].cpu().numpy()
            
            # Train and evaluate
            model.fit(X_train, y_train)
            if is_regression:
                # For regression, use MSE
                y_pred = model.predict(X_val)
                val_metric = np.mean((y_val - y_pred) ** 2)
                return -val_metric  # Return negative MSE to maximize
            else:
                # For classification, use F1 score
                y_pred = model.predict(X_val)
                val_metric = compute_metrics(y_val, y_pred)['f1_macro']
                return val_metric
    
    # Create study
    study = optuna.create_study(direction='maximize')  # Always maximize (negative MSE for regression)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    
    # Create final model with best parameters
    if model_type == "gnn":
        final_model = model_creator(
            input_dim=features.shape[1],
            hidden_dim=best_params['hidden_dim'],
            output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            gnn_type=gnn_type,
            residual=best_params.get('residual', False),
            norm_type=best_params.get('norm_type', 'none'),
            agg_type=best_params.get('agg_type', 'mean'),
            heads=best_params.get('heads', 1),
            concat_heads=best_params.get('concat_heads', True),
            is_regression=is_regression
        ).to(device)
    elif model_type == "mlp":
        final_model = model_creator(
            input_dim=features.shape[1],
            hidden_dim=best_params['hidden_dim'],
            output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            is_regression=is_regression
        ).to(device)
    else:  # sklearn model
        final_model = model_creator(
            input_dim=features.shape[1],
            output_dim=labels.shape[1] if is_regression else labels.max().item() + 1,  # Use number of communities for regression
            is_regression=is_regression
        )
    
    return {
        'best_params': best_params,
        'best_value': -study.best_value if is_regression else study.best_value,  # Convert back to MSE for regression
        'n_trials': len(study.trials),
        'model': final_model
    }


def train_model(
    model: Union[GNNModel, MLPModel, SklearnModel],
    data: Dict[str, Any],
    config: Any,
    is_regression: bool = False
) -> Dict[str, Any]:
    """
    Train a model and return results.
    """
    # For scikit-learn models, use a different training procedure
    if isinstance(model, SklearnModel):
        print("Using CPU for scikit-learn model")
        start_time = time.time()
        
        # Convert data to numpy arrays
        X_train = data['features'][data['train_idx']].cpu().numpy()
        y_train = data['labels'][data['train_idx']].cpu().numpy()
        X_val = data['features'][data['val_idx']].cpu().numpy()
        y_val = data['labels'][data['val_idx']].cpu().numpy()
        X_test = data['features'][data['test_idx']].cpu().numpy()
        y_test = data['labels'][data['test_idx']].cpu().numpy()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_pred_train, is_regression)
        val_metrics = compute_metrics(y_val, y_pred_val, is_regression)
        test_metrics = compute_metrics(y_test, y_pred_test, is_regression)
        
        # Store results
        results = {
            'train_time': time.time() - start_time,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
        }
        
        return results
    
    # For PyTorch models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Mixed Precision Training: Enabled")
    
    model = model.to(device)
    
    # Move data to device
    features = data['features'].to(device)
    edge_index = data['edge_index'].to(device)
    labels = data['labels'].to(device)
    train_idx = data['train_idx'].to(device)
    val_idx = data['val_idx'].to(device)
    test_idx = data['test_idx'].to(device)
    
    print(f"Data moved to {device}")
    print(f"Features shape: {features.shape}, device: {features.device}")
    print(f"Edge index shape: {edge_index.shape}, device: {edge_index.device}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Initialize loss function
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    # Enable mixed precision training if using CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Initialize early stopping
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Use mixed precision training if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                if isinstance(model, GNNModel):
                    out = model(features, edge_index)
                else:
                    out = model(features)
                
                # Compute loss
                loss = criterion(out[train_idx], labels[train_idx])
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            if isinstance(model, GNNModel):
                out = model(features, edge_index)
            else:
                out = model(features)
            
            # Compute loss
            loss = criterion(out[train_idx], labels[train_idx])
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if isinstance(model, GNNModel):
                out = model(features, edge_index)
            else:
                out = model(features)
            
            # Compute losses
            train_loss = criterion(out[train_idx], labels[train_idx])
            val_loss = criterion(out[val_idx], labels[val_idx])
            
            # Compute accuracies
            if is_regression:
                train_acc = 1 - train_loss / torch.var(labels[train_idx])
                val_acc = 1 - val_loss / torch.var(labels[val_idx])
            else:
                train_acc = (out[train_idx].argmax(dim=1) == labels[train_idx]).float().mean()
                val_acc = (out[val_idx].argmax(dim=1) == labels[val_idx]).float().mean()
        
        # Store metrics
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:4d}: Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Store final results
    results = {
        'train_time': time.time() - start_time,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc.item()
    }
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        if isinstance(model, GNNModel):
            out = model(features, edge_index)
        else:
            out = model(features)
        
        # Compute metrics for all sets
        train_metrics = compute_metrics(labels[train_idx], out[train_idx], is_regression)
        val_metrics = compute_metrics(labels[val_idx], out[val_idx], is_regression)
        test_metrics = compute_metrics(labels[test_idx], out[test_idx], is_regression)
        
        results['metrics'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
    
    print(f"\nTraining completed in {results['train_time']:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc.item():.4f} at epoch {best_epoch}")
    
    return results


def train_and_evaluate(
    model: Union[GNNModel, MLPModel, SklearnModel],
    data: Dict[str, Any],
    config: ExperimentConfig,
    is_regression: bool = False
) -> Dict[str, Any]:
    """
    Train and evaluate a model.
    
    Returns:
        Dictionary with structure:
        {
                'train': {
                    'metric1': value,
                    'metric2': value,
                    ...
                },
                'val': {
                    'metric1': value,
                    'metric2': value,
                    ...
                },
                'test': {
                    'metric1': value,
                    'metric2': value,
                    ...
                }
            }
        }
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not config.force_cpu else "cpu")
    
    # Move model to device if it's a PyTorch model
    if isinstance(model, (GNNModel, MLPModel)):
        model = model.to(device)
    
    # Get data
    features = data['features'].to(device)
    edge_index = data['edge_index'].to(device)
    labels = data['labels'].to(device)
    train_idx = data['train_idx'].to(device)
    val_idx = data['val_idx'].to(device)
    test_idx = data['test_idx'].to(device)
    
    # Initialize optimizer if it's a PyTorch model
    if isinstance(model, (GNNModel, MLPModel)):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up loss function
        if is_regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_metric = float('inf') if is_regression else 0.0  # Lower MSE is better, higher F1 is better
        patience_counter = 0
        best_model_state = None
        
        # Initialize metrics tracking
        metrics_history = {
            'train': {'loss': [], 'metric': []},
            'val': {'loss': [], 'metric': []}
        }
        
        start_time = time.time()
        
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, GNNModel):
                out = model(features, edge_index)
            else:  # MLPModel
                out = model(features)
            
            # Compute loss
            loss = criterion(out[train_idx], labels[train_idx])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                if isinstance(model, GNNModel):
                    out = model(features, edge_index)
                else:  # MLPModel
                    out = model(features)
                
                # Compute metrics for train and val sets
                for idx, name in [(train_idx, 'train'), (val_idx, 'val')]:
                    current_loss = criterion(out[idx], labels[idx]).item()
                    metrics_history[name]['loss'].append(current_loss)
                    
                    if is_regression:
                        # For regression, use MSE as metric
                        current_metric = current_loss
                    else:
                        # For classification, use F1 score
                        y_pred = out[idx].argmax(dim=1)
                        y_true = labels[idx]
                        current_metric = compute_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())['f1_macro']
                    
                    metrics_history[name]['metric'].append(current_metric)
                
                # Model selection based on validation metric
                val_metric = metrics_history['val']['metric'][-1]
                if (is_regression and val_metric < best_val_metric) or \
                   (not is_regression and val_metric > best_val_metric):
                    best_val_metric = val_metric
                    best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        break
        
        train_time = time.time() - start_time
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on all sets
        model.eval()
        with torch.no_grad():
            if isinstance(model, GNNModel):
                out = model(features, edge_index)
            else:  # MLPModel
                out = model(features)
            
            results = {}
            
            # Evaluate on all sets
            for idx, name in [(train_idx, 'train'), (val_idx, 'val'), (test_idx, 'test')]:
                if is_regression:
                    # Regression metrics
                    mse = criterion(out[idx], labels[idx]).item()
                    r2 = 1 - mse / torch.var(labels[idx])
                    results[name] = {
                        'mse': mse,
                        'r2': r2.item()
                    }
                else:
                    # Classification metrics
                    y_pred = out[idx].argmax(dim=1)
                    y_true = labels[idx]
                    y_score = out[idx].softmax(dim=1)
                    
                    # Calculate metrics
                    metrics = compute_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    
                    results[name] = {
                        'accuracy': metrics['accuracy'],
                        'f1_macro': metrics['f1_macro'],
                        'roc_auc': metrics['roc_auc'] if 'roc_auc' in metrics else 0.0
                    }
            
            # Add training time
            results['train_time'] = train_time
            
            return results
    
    else:  # SklearnModel
        # Convert data to numpy arrays
        X_train = features[train_idx].cpu().numpy()
        y_train = labels[train_idx].cpu().numpy()
        X_val = features[val_idx].cpu().numpy()
        y_val = labels[val_idx].cpu().numpy()
        X_test = features[test_idx].cpu().numpy()
        y_test = labels[test_idx].cpu().numpy()
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        results = {}
        
        # Evaluate on all sets
        for X, y, name in [(X_train, y_train, 'train'), 
                          (X_val, y_val, 'val'),
                          (X_test, y_test, 'test')]:
            y_pred = model.predict(X)
            
            if is_regression:
                # Regression metrics
                mse = np.mean((y - y_pred) ** 2)
                r2 = 1 - mse / np.var(y)
                results[name] = {
                    'mse': mse,
                    'r2': r2
                }
            else:
                # Classification metrics
                y_score = model.predict_proba(X)
                metrics = compute_metrics(y, y_pred)
                roc_auc = compute_metrics(y, y_score)['roc_auc']
                
                results[name] = {
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'roc_auc': roc_auc
                }
        
        # Add training time
        results['train_time'] = train_time
        
        return results


def train_gnn_model(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20,
    verbose: bool = True,
    optimize: bool = True,
    n_trials: int = 20,
    timeout: Optional[int] = 300  # 5 minutes per optimization
) -> Dict[str, Any]:
    """
    Train a GNN model with optional hyperparameter optimization.
    
    Args:
        model: PyTorch GNN model
        features: Node features [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        epochs: Maximum number of training epochs
        lr: Learning rate (used if optimize=False)
        weight_decay: Weight decay (L2 penalty) (used if optimize=False)
        patience: Patience for early stopping (used if optimize=False)
        verbose: Whether to print progress
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials if optimize=True
        timeout: Timeout in seconds for optimization
        
    Returns:
        Dictionary with training results and model performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create label remapping to handle missing classes
    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)
    label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
    
    if verbose:
        print("\nLabel remapping:")
        print(f"Original labels: {sorted(label_map.keys())}")
        print(f"Remapped to: {sorted(label_map.values())}")
    
    # Remap labels to contiguous range
    remapped_labels = torch.tensor([label_map[int(label)] for label in labels], device=labels.device)
    
    # Update model's output dimension if needed
    if hasattr(model, 'output_dim') and model.output_dim != n_classes:
        if verbose:
            print(f"Adjusting model output dimension from {model.output_dim} to {n_classes}")
        
        # Get the last layer's input dimension
        if hasattr(model, 'convs'):
            # For GNN models
            old_conv = model.convs[-1]
            if isinstance(old_conv, GCNConv):
                in_dim = old_conv.in_channels
            elif isinstance(old_conv, GATConv):
                in_dim = old_conv.in_channels
            elif isinstance(old_conv, SAGEConv):
                in_dim = old_conv.in_channels
            else:
                in_dim = model.hidden_dim  # Fallback to hidden dimension
                
            # Create new output layer
            model.convs[-1] = type(old_conv)(in_dim, n_classes)
            model.output_dim = n_classes
        else:
            # For MLP models
            old_linear = model.layers[-1]
            new_linear = nn.Linear(old_linear.in_features, n_classes, bias=old_linear.bias is not None)
            model.layers[-1] = new_linear
            
        model.output_dim = n_classes
    
    if verbose:
        # Print label distribution after remapping
        print("\nLabel distribution (after remapping):")
        for i in range(n_classes):
            train_count = (remapped_labels[train_idx] == i).sum().item()
            val_count = (remapped_labels[val_idx] == i).sum().item()
            test_count = (remapped_labels[test_idx] == i).sum().item()
            print(f"Class {i}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    try:
        # Move data to device
        features = features.to(device)
        edge_index = edge_index.to(device)
        remapped_labels = remapped_labels.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        
        # Hyperparameter optimization if requested
        hyperopt_results = None
        if optimize:
            if verbose:
                print("\nPerforming hyperparameter optimization...")
            
            # Define model creator function for optimization
            def model_creator(**kwargs):
                from experiments.core.models import GNNModel
                return GNNModel(**kwargs)
            
            # Get GNN type from model if available
            gnn_type = model.gnn_type if hasattr(model, 'gnn_type') else "gcn"
            
            # Run optimization
            hyperopt_results = optimize_hyperparameters(
                model_creator=model_creator,
                features=features,
                edge_index=edge_index,
                labels=remapped_labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                model_type="gnn",
                gnn_type=gnn_type,
                n_trials=n_trials,
                max_epochs=epochs,
                timeout=timeout,
                device=device,
                is_regression=False
            )
            
            # Update hyperparameters based on optimization
            if "best_params" in hyperopt_results and hyperopt_results["best_params"]:
                best_params = hyperopt_results["best_params"]
                lr = best_params.get("lr", lr)
                weight_decay = best_params.get("weight_decay", weight_decay)
                patience = best_params.get("patience", patience)
                
                # Recreate model with optimized parameters
                from experiments.core.models import GNNModel
                hidden_dim = best_params.get("hidden_dim", 64)
                num_layers = best_params.get("num_layers", 2)
                dropout = best_params.get("dropout", 0.5)
                
                # Handle GAT-specific parameters
                if gnn_type == "gat" and "heads" in best_params:
                    heads = best_params.get("heads", 1)
                    concat_heads = best_params.get("concat_heads", True)
                    # For GAT, adjust hidden_dim based on concat_heads
                    effective_hidden_dim = hidden_dim * heads if concat_heads else hidden_dim
                    model = GNNModel(
                        input_dim=features.shape[1],
                        hidden_dim=effective_hidden_dim,
                        output_dim=n_classes,
                        num_layers=num_layers,
                        dropout=dropout,
                        gnn_type=gnn_type,
                        residual=best_params.get("residual", False),
                        norm_type=best_params.get("norm_type", "none"),
                        agg_type=best_params.get("agg_type", "mean"),
                        heads=heads,
                        concat_heads=concat_heads
                    ).to(device)
                else:
                    model = GNNModel(
                        input_dim=features.shape[1],
                        hidden_dim=hidden_dim,
                        output_dim=n_classes,
                        num_layers=num_layers,
                        dropout=dropout,
                        gnn_type=gnn_type,
                        residual=best_params.get("residual", False),
                        norm_type=best_params.get("norm_type", "none"),
                        agg_type=best_params.get("agg_type", "mean")
                    ).to(device)
                
                if verbose:
                    print(f"Using optimized hyperparameters: {best_params}")
            else:
                if verbose:
                    print("Optimization failed or returned empty. Using default parameters.")
        
        # Move model to device
        model = model.to(device)
        
        # Rest of the training code (same as before)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Track training time
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(features, edge_index)
            loss = criterion(out[train_idx], remapped_labels[train_idx])
            
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                train_out = model(features, edge_index)
                train_loss = criterion(train_out[train_idx], remapped_labels[train_idx]).item()
                train_pred = train_out[train_idx].argmax(dim=1)
                train_acc = (train_pred == remapped_labels[train_idx]).float().mean().item()
                
                val_loss = criterion(train_out[val_idx], remapped_labels[val_idx]).item()
                val_pred = train_out[val_idx].argmax(dim=1)
                val_acc = (val_pred == remapped_labels[val_idx]).float().mean().item()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
            
            if epoch - best_epoch >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_out = model(features, edge_index)
            test_loss = criterion(test_out[test_idx], remapped_labels[test_idx]).item()
            test_pred = test_out[test_idx].argmax(dim=1)
            test_acc = (test_pred == remapped_labels[test_idx]).float().mean().item()
            
            # Calculate detailed metrics
            y_true = remapped_labels[test_idx].cpu().numpy()
            y_pred = test_pred.cpu().numpy()
            y_score = test_out[test_idx].softmax(dim=1).cpu().numpy()
            
            metrics = evaluate_node_classification(y_true, y_pred, y_score)
            
            # Add label mapping to metrics for reference
            metrics['label_mapping'] = label_map
        
        result = {
            'model': model,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_time': train_time,
            'history': history,
            'metrics': metrics,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_score': y_score,
                'label_mapping': label_map
            }
        }
        
        # Add hyperopt results if available
        if hyperopt_results:
            result['hyperopt_results'] = {
                'best_params': hyperopt_results.get('best_params', {}),
                'best_value': hyperopt_results.get('best_value', 0.0),
                'n_trials': hyperopt_results.get('n_trials', 0)
            }
        
        return result
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA out of memory. Falling back to CPU.")
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retry on CPU
            return train_gnn_model(
                model, features, edge_index, labels,
                train_idx, val_idx, test_idx,
                epochs, lr, weight_decay, patience, verbose,
                optimize, n_trials, timeout
            )
        else:
            raise


def train_mlp_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20,
    verbose: bool = True,
    optimize: bool = True,
    n_trials: int = 20,
    timeout: Optional[int] = 300
) -> Dict[str, Any]:
    """
    Train an MLP model with optional hyperparameter optimization.
    
    Args:
        model: PyTorch MLP model
        features: Node features [num_nodes, num_features]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        epochs: Maximum number of training epochs
        lr: Learning rate (used if optimize=False)
        weight_decay: Weight decay (L2 penalty) (used if optimize=False)
        patience: Patience for early stopping (used if optimize=False)
        verbose: Whether to print progress
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials if optimize=True
        timeout: Timeout in seconds for optimization
        
    Returns:
        Dictionary with training results and model performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add diagnostic prints for shapes and indices
    if verbose:
        print("\nDiagnostic Information:")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of unique labels: {len(torch.unique(labels))}")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        print(f"Train indices range: [{train_idx.min()}, {train_idx.max()}] (length: {len(train_idx)})")
        print(f"Val indices range: [{val_idx.min()}, {val_idx.max()}] (length: {len(val_idx)})")
        print(f"Test indices range: [{test_idx.min()}, {test_idx.max()}] (length: {len(test_idx)})")
    
    # Create label remapping to handle missing classes
    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)
    label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
    
    if verbose:
        print("\nLabel remapping:")
        print(f"Original labels: {sorted(label_map.keys())}")
        print(f"Remapped to: {sorted(label_map.values())}")
    
    # Remap labels to contiguous range
    remapped_labels = torch.tensor([label_map[int(label)] for label in labels], device=labels.device)
    
    # Update model's output dimension if needed
    if hasattr(model, 'output_dim') and model.output_dim != n_classes:
        if verbose:
            print(f"Adjusting model output dimension from {model.output_dim} to {n_classes}")
        # Assuming last layer is Linear
        old_linear = list(model.modules())[-1]
        new_linear = nn.Linear(old_linear.in_features, n_classes, bias=old_linear.bias is not None)
        # Replace the last layer
        model.lin = new_linear  # Adjust based on model architecture
        model.output_dim = n_classes
    
    if verbose:
        # Print label distribution after remapping
        print("\nLabel distribution (after remapping):")
        for i in range(n_classes):
            train_count = (remapped_labels[train_idx] == i).sum().item()
            val_count = (remapped_labels[val_idx] == i).sum().item()
            test_count = (remapped_labels[test_idx] == i).sum().item()
            print(f"Class {i}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    # Hyperparameter optimization if requested
    hyperopt_results = None
    if optimize:
        if verbose:
            print("\nPerforming hyperparameter optimization for MLP...")
        
        # Define model creator function for optimization
        def model_creator(**kwargs):
            from experiments.core.models import MLPModel
            return MLPModel(**kwargs)
        
        # Run optimization
        hyperopt_results = optimize_hyperparameters(
            model_creator=model_creator,
            features=features,
            edge_index=None,  # MLP doesn't use edge_index
            labels=remapped_labels,  # Use remapped labels
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_type="mlp",
            n_trials=n_trials,
            max_epochs=epochs,
            timeout=timeout,
            device=device,
            is_regression=False
        )
        
        # Update hyperparameters based on optimization
        if "best_params" in hyperopt_results and hyperopt_results["best_params"]:
            best_params = hyperopt_results["best_params"]
            lr = best_params.get("lr", lr)
            weight_decay = best_params.get("weight_decay", weight_decay)
            patience = best_params.get("patience", patience)
            
            # Recreate model with optimized parameters
            from experiments.core.models import MLPModel
            hidden_dim = best_params.get("hidden_dim", 64)
            num_layers = best_params.get("num_layers", 2)
            dropout = best_params.get("dropout", 0.5)
            
            model = MLPModel(
                input_dim=features.shape[1],
                hidden_dim=hidden_dim,
                output_dim=n_classes,  # Use n_classes instead of len(torch.unique(labels))
                num_layers=num_layers,
                dropout=dropout
            )
            
            if verbose:
                print(f"Using optimized hyperparameters: {best_params}")
        else:
            if verbose:
                print("Optimization failed or returned empty. Using default parameters.")
    
    # Move data to device
    features = features.to(device)
    remapped_labels = remapped_labels.to(device)  # Use remapped labels
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Track training time
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(features)
        
        # Calculate loss with remapped labels
        loss = criterion(out[train_idx], remapped_labels[train_idx])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            train_out = model(features)
            train_loss = criterion(train_out[train_idx], remapped_labels[train_idx]).item()
            train_pred = train_out[train_idx].argmax(dim=1)
            train_acc = (train_pred == remapped_labels[train_idx]).float().mean().item()
            
            # Evaluate on validation set
            val_loss = criterion(train_out[val_idx], remapped_labels[val_idx]).item()
            val_pred = train_out[val_idx].argmax(dim=1)
            val_acc = (val_pred == remapped_labels[val_idx]).float().mean().item()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
        
        # Early stopping
        if epoch - best_epoch >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_out = model(features)
        test_loss = criterion(test_out[test_idx], remapped_labels[test_idx]).item()
        test_pred = test_out[test_idx].argmax(dim=1)
        test_acc = (test_pred == remapped_labels[test_idx]).float().mean().item()
        
        # Calculate detailed metrics
        y_true = remapped_labels[test_idx].cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        y_score = test_out[test_idx].softmax(dim=1).cpu().numpy()
        
        metrics = evaluate_node_classification(y_true, y_pred, y_score)
        
        # Add label mapping to metrics for reference
        metrics['label_mapping'] = label_map
    
    # Combine results
    result = {
        'model': model,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'train_time': train_time,
        'history': history,
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'label_mapping': label_map
        }
    }
    
    # Add hyperopt results if available
    if hyperopt_results:
        result['hyperopt_results'] = {
            'best_params': hyperopt_results.get('best_params', {}),
            'best_value': hyperopt_results.get('best_value', 0.0),
            'n_trials': hyperopt_results.get('n_trials', 0)
        }
    
    return result


def train_sklearn_model(
    model: Any,
    features: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    train_idx: Union[np.ndarray, torch.Tensor],
    val_idx: Union[np.ndarray, torch.Tensor],
    test_idx: Union[np.ndarray, torch.Tensor],
    verbose: bool = True,
    optimize: bool = True,
    n_trials: int = 15,
    timeout: Optional[int] = 180
) -> Dict[str, Any]:
    """
    Train a scikit-learn model with optional hyperparameter optimization.
    
    Args:
        model: Scikit-learn model
        features: Node features [num_nodes, num_features]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        verbose: Whether to print progress
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials if optimize=True
        timeout: Timeout in seconds for optimization
        
    Returns:
        Dictionary with training results and model performance
    """
    # Convert tensors to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.cpu().numpy()
    if isinstance(val_idx, torch.Tensor):
        val_idx = val_idx.cpu().numpy()
    if isinstance(test_idx, torch.Tensor):
        test_idx = test_idx.cpu().numpy()
    
    # Perform hyperparameter optimization if requested
    hyperopt_results = None
    if optimize:
        if verbose:
            print("\nPerforming hyperparameter optimization for Random Forest...")
        
        def objective(trial: Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            # Create and train model
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(**params, random_state=42)
            rf_model.fit(features[train_idx], labels[train_idx])
            
            # Evaluate on validation set
            val_acc = rf_model.score(features[val_idx], labels[val_idx])
            return val_acc
        
        # Create and run optimization study
        study = create_study(direction="maximize", sampler=TPESampler(seed=42))
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            if verbose:
                print(f"Best hyperparameters: {study.best_params}")
                print(f"Best validation accuracy: {study.best_value:.4f}")
            
            hyperopt_results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }
            
            # Update model with best parameters
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**study.best_params, random_state=42)
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            if verbose:
                print(f"Optimization error: {str(e)}. Using default model.")
    
    # Track training time
    start_time = time.time()
    
    # Train model
    model.fit(features[train_idx], labels[train_idx])
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(features[test_idx])
    y_score = model.predict_proba(features[test_idx])
    
    # Calculate metrics
    y_true = labels[test_idx]
    metrics = evaluate_node_classification(y_true, y_pred, y_score)
    
    # Calculate test accuracy
    test_acc = (y_pred == y_true).mean()
    
    # Combine results
    result = {
        'model': model,
        'test_acc': test_acc,
        'train_time': train_time,
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score
        }
    }
    
    # Add hyperopt results if available
    if hyperopt_results:
        result['hyperopt_results'] = hyperopt_results
    
    return result


def split_node_indices(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split node indices into train/val/test sets.
    
    Args:
        num_nodes: Number of nodes
        train_ratio: Ratio of nodes for training
        val_ratio: Ratio of nodes for validation
        test_ratio: Ratio of nodes for testing
        stratify: Labels for stratified splitting
        random_state: Random seed
        
    Returns:
        Tuple of (train_idx, val_idx, test_idx)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create indices
    indices = np.arange(num_nodes)
    
    try:
        # First split: train vs. rest
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=stratify,
            random_state=random_state
        )
        
        # Second split: val vs. test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=relative_val_ratio,
            stratify=stratify[temp_idx] if stratify is not None else None,
            random_state=random_state
        )
        
    except ValueError as e:
        # Fall back to non-stratified splitting if any class has < 2 samples
        print(f"Warning: {str(e)}")
        print("Falling back to random splitting")
        
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        indices = np.random.RandomState(random_state).permutation(num_nodes)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx