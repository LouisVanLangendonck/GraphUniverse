"""
Training utilities for graph learning experiments.

This module provides functions for training and evaluating models on graph data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.model_selection import train_test_split

from experiments.core.metrics import evaluate_node_classification, model_performance_summary


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
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a GNN model and evaluate performance.
    
    Args:
        model: PyTorch GNN model
        features: Node features [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: Weight decay (L2 penalty)
        patience: Patience for early stopping
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results and model performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add diagnostic prints for shapes and indices
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
    print("\nLabel remapping:")
    print(f"Original labels: {sorted(label_map.keys())}")
    print(f"Remapped to: {sorted(label_map.values())}")
    
    # Remap labels to contiguous range
    remapped_labels = torch.tensor([label_map[int(label)] for label in labels], device=labels.device)
    
    # Update model's output dimension if needed
    if hasattr(model, 'output_dim') and model.output_dim != n_classes:
        print(f"Adjusting model output dimension from {model.output_dim} to {n_classes}")
        # Assuming last layer is Linear
        old_linear = list(model.modules())[-1]
        new_linear = nn.Linear(old_linear.in_features, n_classes, bias=old_linear.bias is not None)
        # Replace the last layer
        if hasattr(model, 'convs'):
            model.convs[-1] = new_linear
        else:
            model.lin = new_linear  # Adjust this based on your model architecture
        model.output_dim = n_classes
    
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
        
        # Rest of the training code remains the same, but use remapped_labels instead of labels
        model = model.to(device)
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
        
        return {
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
                epochs, lr, weight_decay, patience, verbose
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
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train an MLP model and evaluate performance.
    
    Args:
        model: PyTorch MLP model
        features: Node features [num_nodes, num_features]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: Weight decay (L2 penalty)
        patience: Patience for early stopping
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results and model performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
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
        
        # Calculate loss
        loss = criterion(out[train_idx], labels[train_idx])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            train_out = model(features)
            train_loss = criterion(train_out[train_idx], labels[train_idx]).item()
            train_pred = train_out[train_idx].argmax(dim=1)
            train_acc = (train_pred == labels[train_idx]).float().mean().item()
            
            # Evaluate on validation set
            val_loss = criterion(train_out[val_idx], labels[val_idx]).item()
            val_pred = train_out[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()
        
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
        test_loss = criterion(test_out[test_idx], labels[test_idx]).item()
        test_pred = test_out[test_idx].argmax(dim=1)
        test_acc = (test_pred == labels[test_idx]).float().mean().item()
        
        # Calculate detailed metrics
        y_true = labels[test_idx].cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        y_score = test_out[test_idx].softmax(dim=1).cpu().numpy()
        
        metrics = evaluate_node_classification(y_true, y_pred, y_score)
    
    # Combine results
    results = {
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
            'y_score': y_score
        }
    }
    
    return results


def train_sklearn_model(
    model: Any,
    features: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    train_idx: Union[np.ndarray, torch.Tensor],
    val_idx: Union[np.ndarray, torch.Tensor],
    test_idx: Union[np.ndarray, torch.Tensor],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a scikit-learn model and evaluate performance.
    
    Args:
        model: Scikit-learn model
        features: Node features [num_nodes, num_features]
        labels: Node labels [num_nodes]
        train_idx: Indices of training nodes
        val_idx: Indices of validation nodes
        test_idx: Indices of test nodes
        verbose: Whether to print progress
        
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
    results = {
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
    
    return results


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