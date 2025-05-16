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

from experiments.core.metrics import evaluate_node_classification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model using Optuna.
    
    Args:
        model_creator: Function that creates a model given parameters
        features: Node features
        edge_index: Graph connectivity
        labels: Node labels
        train_idx: Training node indices
        val_idx: Validation node indices
        test_idx: Test node indices
        model_type: Type of model ("gnn", "mlp", "rf")
        gnn_type: Type of GNN if model_type is "gnn"
        n_trials: Number of optimization trials
        max_epochs: Maximum number of training epochs per trial
        timeout: Timeout in seconds for the optimization
        device: Device to run on
        
    Returns:
        Dictionary with best hyperparameters and model
    """
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    features = features.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    
    # Define the objective function for optimization
    def objective(trial: Trial) -> float:
        # Define hyperparameters to optimize based on model type
        if model_type == "gnn" or model_type == "mlp":
            # Common parameters for neural networks
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.6)
            hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            
            # Model-specific parameters
            if model_type == "gnn":
                # GNN-specific parameters
                residual = trial.suggest_categorical("residual", [True, False])
                # For GAT, don't allow any normalization
                if gnn_type == "gat":
                    norm_type = "none"  # Force no normalization for GAT
                else:
                    norm_type = trial.suggest_categorical("norm_type", ["none", "batch", "layer"])
                agg_type = trial.suggest_categorical("agg_type", ["mean", "sum", "max"])
                
                if gnn_type == "gat":
                    heads = trial.suggest_int("heads", 1, 4)
                    concat_heads = trial.suggest_categorical("concat_heads", [True, False])
                    # For GAT, we don't need to adjust hidden_dim as it's handled in the model
                    model = model_creator(
                        input_dim=features.shape[1],
                        hidden_dim=hidden_dim,
                        output_dim=len(torch.unique(labels)),
                        num_layers=num_layers,
                        dropout=dropout,
                        gnn_type=gnn_type,
                        residual=residual,
                        norm_type=norm_type,  # Will always be "none" for GAT
                        agg_type=agg_type,
                        heads=heads,
                        concat_heads=concat_heads
                    ).to(device)
                else:
                    model = model_creator(
                        input_dim=features.shape[1],
                        hidden_dim=hidden_dim,
                        output_dim=len(torch.unique(labels)),
                        num_layers=num_layers,
                        dropout=dropout,
                        gnn_type=gnn_type,
                        residual=residual,
                        norm_type=norm_type,
                        agg_type=agg_type
                    ).to(device)
            else:  # MLP
                model = model_creator(
                    input_dim=features.shape[1],
                    hidden_dim=hidden_dim,
                    output_dim=len(torch.unique(labels)),
                    num_layers=num_layers,
                    dropout=dropout
                ).to(device)
            
            # Optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # Training with early stopping
            best_val_acc = 0.0
            early_stop_counter = 0
            early_stop_patience = trial.suggest_int("patience", 30, 100)
            
            for epoch in range(max_epochs):
                # Training step
                model.train()
                optimizer.zero_grad()
                
                if model_type == "gnn":
                    out = model(features, edge_index)
                else:  # MLP
                    out = model(features)
                
                loss = criterion(out[train_idx], labels[train_idx])
                loss.backward()
                optimizer.step()
                
                # Validation step
                model.eval()
                with torch.no_grad():
                    if model_type == "gnn":
                        out = model(features, edge_index)
                    else:  # MLP
                        out = model(features)
                    
                    val_loss = criterion(out[val_idx], labels[val_idx]).item()
                    val_pred = out[val_idx].argmax(dim=1)
                    val_acc = (val_pred == labels[val_idx]).float().mean().item()
                
                # Report intermediate metric to pruner
                trial.report(val_acc, epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                # Early stopping logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= early_stop_patience:
                    break
            
            return best_val_acc
            
        elif model_type == "rf":
            # Random Forest parameters
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            
            # Move data to CPU for sklearn
            X_train = features[train_idx].cpu().numpy()
            y_train = labels[train_idx].cpu().numpy()
            X_val = features[val_idx].cpu().numpy()
            y_val = labels[val_idx].cpu().numpy()
            
            # Create and train model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            val_acc = model.score(X_val, y_val)
            return val_acc
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Create a study object and optimize the objective function
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = TPESampler(seed=42)
    study = create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Best trial for {model_type}{f' ({gnn_type})' if gnn_type else ''}:")
        logger.info(f"  Value: {study.best_trial.value:.4f}")
        logger.info(f"  Params: {study.best_trial.params}")
        
        return {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "study": study
        }
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        return {
            "best_params": {},
            "best_value": 0.0,
            "error": str(e)
        }


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
                device=device
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
            device=device
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