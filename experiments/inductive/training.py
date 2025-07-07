"""
Training utilities for inductive graph learning experiments.

This module provides functions for training models on graph families
where training and testing happen on different graphs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import copy
import random

from experiments.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel
from experiments.neural_sheaf_diffusion.inductive_sheaf_wrapper import InductiveSheafDiffusionModel
from experiments.transductive.metrics import compute_metrics_gpu, compute_metrics
from experiments.inductive.config import InductiveExperimentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optuna storage utilities
def get_optuna_storage_path(experiment_name: str, run_id: int) -> str:
    """Get the storage path for optuna study."""
    storage_dir = os.path.join('optuna_storage', experiment_name)
    os.makedirs(storage_dir, exist_ok=True)
    return f"sqlite:///{storage_dir}/study_{run_id:04d}.db"

def start_optuna_dashboard(experiment_name: Optional[str] = None, port: int = 8080):
    """Start optuna dashboard server.
    
    Args:
        experiment_name: Optional experiment name to filter studies
        port: Port to run the dashboard on
    """
    import subprocess
    import webbrowser
    
    # Build storage URL
    if experiment_name:
        storage_url = f"sqlite:///optuna_storage/{experiment_name}/"
    else:
        storage_url = "sqlite:///optuna_storage/"
    
    # Start dashboard server
    cmd = [
        "optuna-dashboard",
        storage_url,
        "--port", str(port)
    ]
    
    print(f"Starting optuna-dashboard on port {port}...")
    print(f"Storage URL: {storage_url}")
    print("Press Ctrl+C to stop the server")
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}")
    
    # Run server
    subprocess.run(cmd)

def get_total_classes_from_dataloaders(dataloaders: Dict[str, DataLoader]) -> int:
    """Get the total number of classes based on the universe's K."""
    # Get metadata which contains the output dimension
    if 'metadata' in dataloaders:
        output_dim = dataloaders['metadata']['output_dim']
        print(f"Using output dimension from metadata: {output_dim}")
        # Ensure output_dim is always an int
        if isinstance(output_dim, torch.Tensor):
            output_dim = output_dim.cpu().numpy()
            if hasattr(output_dim, 'size') and output_dim.size > 1:
                output_dim = int(output_dim[0])
            else:
                output_dim = int(output_dim)
        elif isinstance(output_dim, (list, tuple)):
            output_dim = int(output_dim[0])
        else:
            output_dim = int(output_dim)
        return output_dim
    
    # Fallback: Get from first batch if metadata not available
    first_fold_name = list(dataloaders.keys())[0]
    sample_batch = next(iter(dataloaders[first_fold_name]['train']))
    if hasattr(sample_batch, 'universe_K'):
        output_dim = sample_batch.universe_K
        print(f"Using universe K from batch: {output_dim}")
        # Ensure output_dim is always an int
        if isinstance(output_dim, torch.Tensor):
            output_dim = output_dim.cpu().numpy()
            if hasattr(output_dim, 'size') and output_dim.size > 1:
                output_dim = int(output_dim[0])
            else:
                output_dim = int(output_dim)
        elif isinstance(output_dim, (list, tuple)):
            output_dim = int(output_dim[0])
        else:
            output_dim = int(output_dim)
        return output_dim
    
    # Last resort: Try to infer from labels
    all_labels = set()
    for fold_name, fold_data in dataloaders.items():
        if fold_name == 'metadata':
            continue
        for split_name, dataloader in fold_data.items():
            if split_name == 'metadata':
                continue
            for batch in dataloader:
                labels = batch.y
                all_labels.update(labels.cpu().numpy().tolist())
    
    output_dim = max(all_labels) + 1
    print(f"Warning: Inferring output dimension from labels: {output_dim}")
    return int(output_dim)

def train_inductive_model(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    model_name: str,
    dataloaders: Dict[str, Dict[str, DataLoader]],
    config: InductiveExperimentConfig,
    task: str,
    device: torch.device,
    finetuning: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Train a model for inductive learning on graph families.
    Assumes all data is already loaded to GPU for efficiency.
    
    Args:
        model: Model to train
        dataloaders: Dictionary with train/val/test dataloaders (GPU-resident)
        config: Experiment configuration
        task: Current task name
        device: Device to use for training
        
    Returns:
        Dictionary with training results and metrics
    """
    is_regression = config.is_regression.get(task, False)
    
    # Handle sklearn models separately
    if isinstance(model, SklearnModel):
        return train_sklearn_inductive(model, dataloaders, config, is_regression)    
    
    # Get the effective patience value
    effective_patience = config.patience

    training_history = {}
    for fold_name, fold_data in dataloaders.items():
        training_history[fold_name] = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
    }
    fold_train_time = {}
    fold_test_metrics = {}
    fold_best_val_metrics = {}
    
    print(f"\nStarting training with patience={effective_patience}")
    print("Early stopping will trigger if validation metric doesn't improve for", effective_patience, "epochs")
    print(f"Using GPU-resident data on {device}")
    
    # Check if the model is graph-based
    if model_name in ['mlp', 'sklearn', 'rf']:
        graph_based_model = False
        transformer_based_model = False
        sheaf_based_model = False
    elif model_name == 'graphgps':
        transformer_based_model = True
        graph_based_model = False
        sheaf_based_model = False
    elif model_name == 'sheaf_diffusion':
        sheaf_based_model = True
        graph_based_model = False
        transformer_based_model = False
    else:
        graph_based_model = True
        transformer_based_model = False
        sheaf_based_model = False

    # Make deep copy of model to always start from for each fold
    model_copy = copy.deepcopy(model)

    # Track initial GPU memory
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        # Train for each fold
        for fold_name, fold_data in dataloaders.items():
            start_time = time.time()
            print("--------------------------------")
            print(f"--- Training fold {fold_name} ---")
            print("--------------------------------\n")
            train_loader = fold_data['train']
            val_loader = fold_data['val']
            test_loader = fold_data['test']

            model = copy.deepcopy(model_copy)
            
            # Set up loss function
            if is_regression:
                if config.regression_loss == 'mae':
                    criterion = torch.nn.L1Loss()
                elif config.regression_loss == 'mse':
                    criterion = torch.nn.MSELoss()
                else:
                    raise ValueError(f"Invalid regression loss: {config.regression_loss}")
            else:
                criterion = torch.nn.CrossEntropyLoss()
            
            best_val_metric = float('inf') if is_regression else 0.0
            patience_counter = 0
            best_model_state = None

            # Check if all data is on device
            for batch in train_loader:
                if batch.x.device != device:
                    all_data_on_device = False
                    break
                else:
                    all_data_on_device = True

            # PyTorch models - data is already on GPU
            model = model.to(device)

            # Handle lazy initialization for sheaf models
            optimizer = None
            if sheaf_based_model:
                # For sheaf models, we need to do a dummy forward pass first to initialize parameters
                # print("Initializing sheaf model parameters...")
                # with torch.no_grad():
                #     # Get first batch to initialize the model
                #     first_batch = next(iter(train_loader))
                #     # Ensure data is on the same device as the model
                #     if not all_data_on_device:
                #         first_batch = first_batch.to(device)
                #     if hasattr(first_batch, 'batch') and first_batch.batch is not None:
                #         _ = model(first_batch.x, first_batch.edge_index, first_batch.batch)
                #     else:
                #         _ = model(first_batch.x, first_batch.edge_index)
                
                # Now create optimizer after parameters are initialized
                param_count = sum(p.numel() for p in model.parameters())
                print(f"Sheaf model initialized with {param_count} parameters")
                
                if param_count == 0:
                    raise ValueError("Sheaf model has no parameters after initialization")
                
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            else:
                # For non-sheaf models, create optimizer immediately
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            
            for epoch in range(config.epochs):
                # Training
                model.train()
                train_loss = 0.0
                train_predictions = []
                train_targets = []
                
                for batch_idx, batch in enumerate(train_loader):
                    # Data is already on GPU - no need for .to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass - check if model requires edge_index
                    if graph_based_model or sheaf_based_model:
                        out = model(batch.x, batch.edge_index)
                    elif transformer_based_model:
                        out = model(batch.x, batch.edge_index, data=batch, batch=batch.batch)
                    # elif sheaf_based_model:
                    #     out = model(batch.x, batch.edge_index, graph=batch)
                    else:  # MLPModel or other non-GNN model
                        out = model(batch.x)
                    
                    # Compute loss
                    loss = criterion(out, batch.y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Store predictions for metrics - keep on GPU for now
                    if is_regression:
                        train_predictions.append(out.detach())
                        train_targets.append(batch.y.detach())
                    else:
                        train_predictions.append(out.argmax(dim=1).detach())
                        train_targets.append(batch.y.detach())
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Data is already on GPU - no need for .to(device)
                        
                        if graph_based_model or sheaf_based_model:
                            out = model(batch.x, batch.edge_index)
                        elif transformer_based_model:
                            out = model(batch.x, batch.edge_index, data=batch, batch=batch.batch)
                        # elif sheaf_based_model:
                        #     out = model(batch.x, batch.edge_index, graph=batch)
                        else:
                            out = model(batch.x)
                        
                        loss = criterion(out, batch.y)
                        val_loss += loss.item()
                        
                        # Keep predictions on GPU for now
                        if is_regression:
                            val_predictions.append(out.detach())
                        else:
                            val_predictions.append(out.argmax(dim=1).detach())
                        
                        val_targets.append(batch.y.detach())
                
                # Calculate metrics - use GPU-based computation to avoid CPU transfer bottleneck
                train_pred = torch.cat(train_predictions, dim=0)
                train_true = torch.cat(train_targets, dim=0)
                val_pred = torch.cat(val_predictions, dim=0)
                val_true = torch.cat(val_targets, dim=0)
                
                # Use GPU-based metrics computation
                train_metrics = compute_metrics_gpu(train_true, train_pred, is_regression)
                val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                
                # Clear prediction lists
                train_predictions.clear()
                train_targets.clear()
                val_predictions.clear()
                val_targets.clear()
                
                # Get primary metrics
                if is_regression:
                    if config.regression_loss == 'mae':
                        train_metric = train_metrics['mae']
                        val_metric = val_metrics['mae']
                    elif config.regression_loss == 'mse':
                        train_metric = train_metrics['mse']
                        val_metric = val_metrics['mse']
                    else:
                        raise ValueError(f"Invalid regression loss: {config.regression_loss}")
                else:
                    train_metric = train_metrics['f1_macro']
                    val_metric = val_metrics['f1_macro']
                
                # Store metrics
                training_history[fold_name]['train_loss'].append(train_loss / len(train_loader))
                training_history[fold_name]['val_loss'].append(val_loss / len(val_loader))
                training_history[fold_name]['train_metric'].append(train_metric)
                training_history[fold_name]['val_metric'].append(val_metric)
                
                # Print progress
                if epoch % 10 == 0 or epoch == config.epochs - 1:
                    print_str = f"FOLD {fold_name} - Epoch {epoch:3d}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, " \
                            f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}"
                    
                    # Add relative error for k_hop task
                    if task.startswith('k_hop_community_counts'):
                        train_rel_error = train_metrics.get('relative_error', 0.0)
                        val_rel_error = val_metrics.get('relative_error', 0.0)
                        print_str += f", Train Rel Error: {train_rel_error:.4f}, Val Rel Error: {val_rel_error:.4f}"
                        
                        # Print random node's target and predicted vectors
                        if len(val_predictions) > 0:
                            # Get a random batch
                            random_batch_idx = np.random.randint(0, len(val_predictions))
                            random_pred = val_predictions[random_batch_idx]
                            random_true = val_targets[random_batch_idx]
                            
                            # Get a random node from that batch
                            random_node_idx = np.random.randint(0, len(random_pred))
                            
                            print(f"\nRandom node vectors at epoch {epoch}:")
                            print(f"Target vector: {random_true[random_node_idx].cpu().numpy()}")
                            # Rounded and minimum 0:
                            print(f"Predicted vector (rounded): {np.round(np.maximum(random_pred[random_node_idx].cpu().numpy(), 0))}")
                    
                    print(print_str)
                
                # Model selection
                improved = False
                if is_regression:
                    if config.regression_loss == 'mae':
                        improved = val_metric < best_val_metric
                    elif config.regression_loss == 'mse':
                        improved = val_metric < best_val_metric
                    else:
                        raise ValueError(f"Invalid regression loss: {config.regression_loss}")
                else:
                    improved = val_metric > best_val_metric
                
                if improved:
                    best_val_metric = val_metric
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= effective_patience:
                        print(f"Early stopping triggered at epoch {epoch}!")
                        break
            
            train_time = time.time() - start_time
        
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("Loaded best model weights")
            
            # Final evaluation on test set
            test_metrics = evaluate_inductive_model_gpu_resident(model, graph_based_model, transformer_based_model, sheaf_based_model, test_loader, is_regression, device, finetuning)

            fold_train_time[fold_name] = train_time
            fold_best_val_metrics[fold_name] = best_val_metric
            fold_test_metrics[fold_name] = test_metrics
            
            # Clear model from GPU after fold
            model = model.cpu()
            del model
            torch.cuda.empty_cache()
            
    finally:
        # Print final GPU memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            print(f"GPU memory used: {(final_memory - initial_memory) / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
        
    return {
        'fold_train_time': fold_train_time,
        'fold_best_val_metrics': fold_best_val_metrics,
        'fold_test_metrics': fold_test_metrics,
        'training_history': training_history,
    }

def train_sklearn_inductive(
    model: SklearnModel,
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    is_regression: bool
) -> Dict[str, Any]:
    """
    Train a sklearn model for inductive learning.
    
    Args:
        model: Sklearn model to train
        dataloaders: Dictionary with dataloaders
        config: Experiment configuration
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary with training results
    """
    # Extract features and labels from all graphs in training set
    train_features = []
    train_labels = []
    
    for batch in dataloaders['train']:
        # Convert batch to numpy
        batch_features = batch.x.cpu().numpy()
        batch_labels = batch.y.cpu().numpy()
        
        # Ensure labels have correct shape
        if len(batch_labels.shape) == 1:
            batch_labels = batch_labels.reshape(-1, 1)
        
        # Reshape features if needed (handle batch dimension)
        if len(batch_features.shape) > 2:
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        train_features.append(batch_features)
        train_labels.append(batch_labels)
    
    # Concatenate all training data
    X_train = np.vstack(train_features)
    y_train = np.vstack(train_labels)
    
    # Verify shapes match
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Feature and label dimensions don't match: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # If regression task, flatten y_train if it's 2D
    if is_regression and len(y_train.shape) > 1:
        y_train = y_train.ravel()
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate on test set
    test_features = []
    test_labels = []
    
    for batch in dataloaders['test']:
        batch_features = batch.x.cpu().numpy()
        batch_labels = batch.y.cpu().numpy()
        
        # Ensure labels have correct shape
        if len(batch_labels.shape) == 1:
            batch_labels = batch_labels.reshape(-1, 1)
        
        # Reshape features if needed (handle batch dimension)
        if len(batch_features.shape) > 2:
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        test_features.append(batch_features)
        test_labels.append(batch_labels)
    
    X_test = np.vstack(test_features)
    y_test = np.vstack(test_labels)
    
    # Verify shapes match
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Feature and label dimensions don't match: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # If regression task, flatten y_test if it's 2D
    if is_regression and len(y_test.shape) > 1:
        y_test = y_test.ravel()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions have correct shape for metrics
    if is_regression and len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Compute metrics
    test_metrics = compute_metrics(y_test, y_pred, is_regression)
    
    return {
        'train_time': train_time,
        'test_metrics': test_metrics,
        'model': model
    }

def evaluate_inductive_model_gpu_resident(
    model: Union[GNNModel, MLPModel],
    graph_based_model: bool,
    transformer_based_model: bool,
    sheaf_based_model: bool,
    test_loader: DataLoader,
    is_regression: bool,
    device: torch.device,
    finetuning: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set for inductive learning.
    Assumes all data is already on GPU for efficiency.
    
    Args:
        model: Trained model
        test_loader: Test data loader (GPU-resident)
        is_regression: Whether this is a regression task
        device: Device to use for evaluation
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Data is already on GPU - no need for .to(device)
            
            # Forward pass - check if model requires edge_index and extra batch info (for PE)
            if graph_based_model or sheaf_based_model:
                out = model(batch.x, batch.edge_index)
            elif transformer_based_model:
                out = model(batch.x, batch.edge_index, data=batch, batch=batch.batch)
            # elif sheaf_based_model:
            #     out = model(batch.x, batch.edge_index, graph=batch)
            else:  # MLPModel or other non-GNN model
                out = model(batch.x)
            
            # Store predictions - keep on GPU for now
            if is_regression:
                all_predictions.append(out.detach())
            else:
                all_predictions.append(out.argmax(dim=1).detach())
            
            all_targets.append(batch.y.detach())
    
    # Concatenate all predictions and compute metrics on GPU
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Use GPU-based metrics computation
    metrics = compute_metrics_gpu(targets, predictions, is_regression)
    
    return metrics

def optimize_finetuning_hyperparameters(
    pretrained_model,
    metadata: Dict[str, Any],
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    output_dim: int,
    is_regression: bool,
    n_trials: int = 10,  # Fewer trials for fine-tuning
    timeout: Optional[int] = 300,  # Shorter timeout
    device: Optional[torch.device] = None,
    experiment_name: Optional[str] = None,
    run_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters specifically for fine-tuning.
    Only optimizes task-specific parameters, not pre-trained encoder architecture.
    """
    import optuna
    from optuna import create_study, Trial
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def objective(trial: Trial) -> float:
        # ONLY optimize fine-tuning specific parameters
        
        # 1. Learning rate (most important for fine-tuning)
        base_lr = trial.suggest_float('base_lr', 1e-5, 1e-2, log=True)
        
        # 2. Learning rate multiplier for encoder vs head
        if not config.freeze_encoder:
            encoder_lr_ratio = trial.suggest_float('encoder_lr_ratio', 0.01, 0.5)
        else:
            encoder_lr_ratio = 0.0  # Frozen encoder
        
        # 3. Weight decay
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 4. Dropout for the task-specific head only
        head_dropout = trial.suggest_float('head_dropout', 0.0, 0.7)
        
        # 5. Patience for early stopping
        patience = config.patience
        
        # 6. Optional: Task head architecture
        if not is_regression:
            # For classification, optionally add a hidden layer in the head
            use_hidden_head = trial.suggest_categorical('use_hidden_head', [True, False])
            if use_hidden_head:
                head_hidden_dim = trial.suggest_int('head_hidden_dim', 32, 128)
            else:
                head_hidden_dim = None
        else:
            use_hidden_head = False
            head_hidden_dim = None
        
        # Create fine-tuning model with optimized head
        model = create_optimized_finetuning_model(
            pretrained_model, metadata, output_dim, is_regression,
            config.freeze_encoder, head_dropout, use_hidden_head, head_hidden_dim
        ).to(device)
        
        # Setup optimizer with different learning rates
        if config.freeze_encoder:
            # Only optimize head parameters
            optimizer = torch.optim.Adam(
                model.head.parameters(),
                lr=base_lr,
                weight_decay=weight_decay
            )
        else:
            # Different learning rates for encoder and head
            optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'lr': base_lr * encoder_lr_ratio},
                {'params': model.head.parameters(), 'lr': base_lr}
            ], weight_decay=weight_decay)
        
        # Quick training loop with early stopping
        if is_regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        max_epochs = min(30, config.epochs // 3)  # Much shorter for fine-tuning
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            for batch in dataloaders['train']:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in dataloaders['val']:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    
                    if is_regression:
                        val_predictions.append(out.detach())
                    else:
                        val_predictions.append(out.argmax(dim=1).detach())
                    
                    val_targets.append(batch.y.detach())
            
            # Calculate validation metric
            val_pred = torch.cat(val_predictions, dim=0)
            val_true = torch.cat(val_targets, dim=0)
            val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
            
            if is_regression:
                if config.regression_loss == 'mae':
                    val_metric = val_metrics['mae']
                    if val_metric < best_val_metric:  # Lower is better for MAE
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                elif config.regression_loss == 'mse':
                    val_metric = val_metrics['mse']
                    if val_metric < best_val_metric:  # Lower is better for MSE
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    val_metric = val_metrics['r2']
                    if val_metric > best_val_metric:  # Higher is better for R²
                        best_val_metric = val_metric
                        patience_counter = 0
            else:
                # For classification, use F1 score
                val_metric = val_metrics['f1_macro']
                if val_metric > best_val_metric:  # Higher is better for F1
                    best_val_metric = val_metric
                    patience_counter = 0
            
            if patience_counter >= patience:
                break
        
        # Return metric for optimization
        if is_regression:
            if config.regression_loss == 'mae':
                final_metric = -best_val_metric  # Negative because we minimize MAE
            elif config.regression_loss == 'mse':
                final_metric = -best_val_metric  # Negative because we minimize MSE
            else:
                final_metric = best_val_metric  # Positive because we maximize R²
        else:
            # For classification, use F1 score
            final_metric = best_val_metric  # Positive because we maximize F1
        
        return final_metric
    
    # ADD EARLY TERMINATION
    def should_terminate_finetuning(study, trial):
        """Check if we should terminate fine-tuning optimization early."""
        if is_regression:
            # For MSE, perfect score is 0, but we return negative values
            return trial.value >= -1e-6  # Essentially 0 MSE
        else:
            # For F1, perfect score is 1.0
            return trial.value >= 0.999

    # Create study with storage if experiment info provided
    study_name = f"finetuning_{metadata['config']['gnn_type']}_{metadata['config']['pretraining_task']}"
    if experiment_name and run_id is not None:
        storage_path = get_optuna_storage_path(experiment_name, run_id)
        study = create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',  # We always maximize (negative MSE for regression)
            load_if_exists=True
        )
    else:
        study = create_study(direction='maximize')
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        callbacks=[lambda study, trial: study.stop() if should_terminate_finetuning(study, trial) else None]
    )
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': study_name,
        'optimization_type': 'fine_tuning'
    }

def create_optimized_finetuning_model(
    pretrained_model,
    metadata: Dict[str, Any],
    output_dim: int,
    is_regression: bool,
    freeze_encoder: bool,
    head_dropout: float = 0.0,
    use_hidden_head: bool = False,
    head_hidden_dim: Optional[int] = None
) -> nn.Module:
    """Create fine-tuning model with optimized head architecture."""
    
    # Extract encoder
    if hasattr(pretrained_model, 'encoder'):
        encoder = pretrained_model.encoder
    else:
        encoder = pretrained_model
    
    encoder_dim = metadata['config']['hidden_dim']
    
    # Create optimized head
    if use_hidden_head and head_hidden_dim and not is_regression:
        # Multi-layer head for classification
        head = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, output_dim)
        )
    else:
        # Simple linear head
        if head_dropout > 0:
            head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(encoder_dim, output_dim)
            )
        else:
            head = nn.Linear(encoder_dim, output_dim)
    
    # Combine into fine-tuning model
    class FineTuningModel(nn.Module):
        def __init__(self, encoder, head, freeze_encoder=False):
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.gnn_type = getattr(encoder, 'gnn_type', 'gcn')
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        def forward(self, x, edge_index=None):
            # Get embeddings from encoder
            if hasattr(self.encoder, 'convs'):  # This is a GNN model
                if edge_index is None:
                    raise ValueError("GNN model requires edge_index")
                embeddings = self.encoder(x, edge_index)
            else:  # This is an MLP or other non-GNN model
                embeddings = self.encoder(x)
            
            # Apply head
            return self.head(embeddings)
    
    return FineTuningModel(encoder, head, freeze_encoder)

def get_hyperparameter_space(trial, model_type: str, is_regression: bool) -> Dict[str, Any]:
    """Get hyperparameter space for different model types."""
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'patience': trial.suggest_int('patience', 20, 50),
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.7)
    }
    
    if model_type in ['gat', 'fagcn']:
        # For GAT and FAGCN, add specific parameters
        params.update({
            'heads': trial.suggest_int('heads', 1, 8),
            'concat_heads': trial.suggest_categorical('concat_heads', [True, False]),
            'residual': trial.suggest_categorical('residual', [True, False]),
            'norm_type': trial.suggest_categorical('norm_type', ['none', 'batch', 'layer']),
            'agg_type': trial.suggest_categorical('agg_type', ['mean', 'sum'])
        })
        
        # For GAT with concatenated heads, ensure hidden_dim is divisible by heads
        if model_type == 'gat' and params['concat_heads']:
            # Adjust hidden_dim to be divisible by heads
            base_dim = params['hidden_dim']
            params['hidden_dim'] = (base_dim // params['heads']) * params['heads']
            if params['hidden_dim'] < 32:  # Ensure minimum dimension
                params['hidden_dim'] = params['heads'] * 32
    
    return params

def train_and_evaluate_inductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],  # Updated type hint
    model_name: str,
    dataloaders: Dict[str, Dict[str, DataLoader]],
    config: InductiveExperimentConfig,
    task: str,
    device: torch.device,
    optimize_hyperparams: bool = False,
    experiment_name: Optional[str] = None,
    run_id: Optional[int] = None, 
    finetuning: Optional[bool] = False,
) -> Dict[str, Any]:
    """Complete training and evaluation pipeline for inductive learning."""
    
    is_regression = config.is_regression.get(task, False)
    print(f"🔄 Task: {task}, is_regression: {is_regression}")

    if finetuning:
        print(f"Finetuning model. No hyperparameter optimization.")
    else:
        print(f"Training from scratch")

        # Handle Graph Transformer models specifically
        if model_name in ['graphormer', 'graphgps']:
            print(f"🔄 Training Graph Transformer: {model.transformer_type}")
            
            # Update hyperparameter optimization for transformers
            if optimize_hyperparams and not isinstance(model, SklearnModel):
                print(f"🎯 Optimizing Graph Transformer hyperparameters...")
                
                # Get model creator function for transformers
                model_creator = lambda **kwargs: GraphTransformerModel(**kwargs)
                
                # Get sample batch to determine dimensions
                first_fold_name = list(dataloaders.keys())[0]
                sample_batch = next(iter(dataloaders[first_fold_name]['train']))
                input_dim = sample_batch.x.shape[1]

                if sample_batch.x.device != device:
                    print("Batches are not on device")
                    all_data_on_device = False
                else:
                    print("All batches are already on device")
                    all_data_on_device = True
                
                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)
                
                # Create a single optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"transformer_{model.transformer_type}_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                # Store transformer type for the objective function
                transformer_type = model.transformer_type
                
                # Use only the first fold for optimization
                opt_fold = first_fold_name
                print(f"Using fold {opt_fold} for optimization")
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    # Use config patience instead of optimizing it
                    patience = config.patience
                    
                    # Transformer-specific hyperparameters
                    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
                    base_dim = trial.suggest_int('base_dim', 4, 32)
                    hidden_dim = base_dim * num_heads
                    if task.startswith('k_hop_community_counts'):
                        k = int(task.split('_k')[-1])
                        num_layers = trial.suggest_int('num_layers', k + 1, k + 2)
                    else:
                        num_layers = config.num_layers
                    dropout = trial.suggest_float('dropout', 0.0, 0.3)
                    
                    # Transformer-type specific parameters
                    if transformer_type == "graphgps":
                        # Only suggest parameters that GraphTransformerModel actually accepts
                        local_gnn_type = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
                        pe_type = trial.suggest_categorical('pe_type', ['laplacian', 'random_walk', 'shortest_path'])
                        pe_norm_type = trial.suggest_categorical('pe_norm_type', ['layer', 'graph', None])
                        attn_type = trial.suggest_categorical('attn_type', ['performer', 'multihead'])
                        
                    
                    # Create transformer model with only accepted parameters for GraphGPS
                    trial_model = GraphTransformerModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        transformer_type=transformer_type,
                        num_layers=num_layers,
                        dropout=dropout,
                        num_heads=num_heads,
                        is_regression=is_regression,
                        is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count'),
                        local_gnn_type=local_gnn_type,
                        attn_type=attn_type,
                        pe_dim=config.max_pe_dim,
                        pe_type=pe_type,
                        pe_norm_type=pe_norm_type,
                    ).to(device)
                    
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        elif config.regression_loss == 'mse':
                            criterion = torch.nn.MSELoss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0

                    # Train and evaluate on the optimization fold
                    fold_dataloader = dataloaders[opt_fold]

                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in fold_dataloader['train']:
                            if not all_data_on_device:
                                batch = batch.to(device)
                            optimizer.zero_grad()
                            out = trial_model(batch.x, batch.edge_index, data=batch, batch=batch.batch)
                            loss = criterion(out, batch.y)
                            loss.backward()
                            optimizer.step()
                        
                        # Validation
                        trial_model.eval()
                        val_predictions = []
                        val_targets = []
                        
                        with torch.no_grad():
                            for batch in fold_dataloader['val']:
                                out = trial_model(batch.x, batch.edge_index, data=batch, batch=batch.batch)
                                
                                if is_regression:
                                    val_predictions.append(out.detach())
                                else:
                                    val_predictions.append(out.argmax(dim=1).detach())
                                
                                val_targets.append(batch.y.detach())
                        
                        # Calculate validation metric
                        val_pred = torch.cat(val_predictions, dim=0)
                        val_true = torch.cat(val_targets, dim=0)
                        val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)

                        if is_regression:
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            elif config.regression_loss == 'mse':
                                val_metric = val_metrics['mse']
                                if val_metric < best_val_metric:  # Lower is better for MSE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            else:
                                val_metric = val_metrics['r2']
                                if val_metric > best_val_metric:  # Higher is better for R²
                                    best_val_metric = val_metric
                                    patience_counter = 0
                        else:
                            val_metric = val_metrics['f1_macro']
                            if val_metric > best_val_metric:  # Higher is better for F1
                                best_val_metric = val_metric
                                patience_counter = 0
                        
                        if patience_counter >= patience:
                            break
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss in ['mae', 'mse']:
                            final_metric = -best_val_metric  # Negative because we minimize MAE
                        else:
                            final_metric = best_val_metric  # Positive because we maximize R²
                    else:
                        final_metric = best_val_metric  # Positive because we maximize F1
                    
                    return final_metric
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    base_dim = best_params['base_dim']
                    num_heads = best_params['num_heads']
                    hidden_dim = base_dim * num_heads
                    
                    if transformer_type == "graphgps":
                        # GraphGPS parameters - only use accepted parameters
                        model = GraphTransformerModel(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            transformer_type=transformer_type,
                            num_layers=best_params.get('num_layers', config.num_layers),
                            dropout=best_params.get('dropout', 0.1),
                            num_heads=num_heads,
                            is_regression=is_regression,
                            is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count'),
                            local_gnn_type=best_params['local_gnn_type'],
                            attn_type=best_params['attn_type'],
                            pe_dim=config.max_pe_dim,
                            pe_type=best_params['pe_type'],
                            pe_norm_type=best_params['pe_norm_type'],
                        ).to(device)
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
                    config.num_layers = best_params.get('num_layers', config.num_layers)
                    config.dropout = best_params.get('dropout', config.dropout)
                    config.hidden_dim = best_params.get('hidden_dim', config.hidden_dim)
                    config.num_heads = best_params.get('num_heads', config.num_heads)
                    config.local_gnn_type = best_params.get('local_gnn_type', config.local_gnn_type)
                    config.attn_type = best_params.get('attn_type', config.attn_type)
                    config.pe_dim = best_params.get('pe_dim', config.pe_dim)
                    config.pe_type = best_params.get('pe_type', config.pe_type)
                    config.pe_norm_type = best_params.get('pe_norm_type', config.pe_norm_type)
        
        if model_name == 'sheaf_diffusion':
            print(f"🔄 Training Sheaf Diffusion")
            # Update hyperparameter optimization for sheaf diffusion
            if optimize_hyperparams:
                print(f"🎯 Optimizing Sheaf Diffusion hyperparameters...")
                
                # Get sample batch to determine dimensions
                first_fold_name = list(dataloaders.keys())[0]
                sample_batch = next(iter(dataloaders[first_fold_name]['train']))
                input_dim = sample_batch.x.shape[1]

                if sample_batch.x.device != device:
                    print("Batches are not on device")
                    all_data_on_device = False
                else:
                    print("All batches are already on device")
                    all_data_on_device = True

                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)

                # Create optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"sheaf_diffusion_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    # Use config patience instead of optimizing it
                    patience = config.patience

                    # Sheaf Diffusion-specific hyperparameters
                    hidden_dim = config.hidden_dim
                    if task.startswith('k_hop_community_counts'):
                        num_layers = trial.suggest_int('num_layers', config.khop_community_counts_k + 1, config.khop_community_counts_k + 2)
                    else:
                        num_layers = trial.suggest_int('num_layers', config.num_layers - 1, config.num_layers + 1)
                    dropout = trial.suggest_float('dropout', 0.1, 0.5)
                    d = trial.suggest_int('d', 2, 3)
                    # Make sure hidden_dim is divisible by d
                    if hidden_dim % d != 0:
                        hidden_dim = hidden_dim // d * d
                        trial.suggest_int('hidden_dim', hidden_dim, hidden_dim)
                        print(f"Warning: hidden_dim is not divisible by d, setting hidden_dim to {hidden_dim}")
                    # sheaf_type = trial.suggest_categorical('sheaf_type', ['diag', 'bundle', 'general'])
                    sheaf_type = trial.suggest_categorical('sheaf_type', ['diagonal', 'orthogonal']) # Not general because it's too UNSTABLE!

                    # Create Sheaf Diffusion model with all parameters
                    # trial_model = InductiveSheafDiffusionModel(
                    #     input_dim=input_dim,
                    #     hidden_dim=hidden_dim,
                    #     output_dim=output_dim,
                    #     sheaf_type=sheaf_type,
                    #     d=d,
                    #     num_layers=num_layers,
                    #     dropout=dropout,
                    #     input_dropout=0.1,
                    #     is_regression=is_regression,
                    #     is_graph_level_task=False,
                    #     device=device
                    # ).to(device)
                

                    trial_model = SheafDiffusionModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        sheaf_type=sheaf_type,
                        num_layers=num_layers,
                        dropout=dropout,
                        d=d,
                        is_regression=is_regression,
                        is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count')
                        ).to(device)
                    #     input_dim=input_dim,
                    #     hidden_dim=hidden_dim,
                    #     output_dim=output_dim,
                    #     sheaf_type=sheaf_type,
                    #     num_layers=num_layers,
                    #     dropout=dropout,
                    #     d=d,
                    #     is_regression=is_regression,
                    #     is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count')
                    # ).to(device)
                    
                    # Initialize sheaf model parameters with a dummy forward pass
                    # with torch.no_grad():
                    #     first_batch = next(iter(dataloaders[first_fold_name]['train']))
                    #     # Ensure data is on the same device as the model
                    #     if not all_data_on_device:
                    #         first_batch = first_batch.to(device)
                    #     if hasattr(first_batch, 'batch') and first_batch.batch is not None:
                    #         _ = trial_model(first_batch.x, first_batch.edge_index, first_batch.batch)
                    #     else:
                    #         _ = trial_model(first_batch.x, first_batch.edge_index)
                    
                    # Now create optimizer after parameters are initialized
                    # param_count = sum(p.numel() for p in trial_model.parameters())
                    # if param_count == 0:
                    #     return float('-inf')  # Return worst possible score if no parameters
                    
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        elif config.regression_loss == 'mse':
                            criterion = torch.nn.MSELoss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()

                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0
                    
                    fold_dataloader = dataloaders[first_fold_name]
                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in fold_dataloader['train']:
                            if not all_data_on_device:  
                                print("Warning: Not all data is on device. Moving batch to device...")
                                batch = batch.to(device)
                            optimizer.zero_grad()
                            out = trial_model(batch.x, batch.edge_index) #, graph=batch)
                            loss = criterion(out, batch.y)
                            loss.backward()
                            optimizer.step()
                        
                        # Validation
                        trial_model.eval()
                        val_predictions = []
                        val_targets = []
                        
                        with torch.no_grad():
                            for batch in fold_dataloader['val']:
                                if not all_data_on_device:
                                    batch = batch.to(device)
                                out = trial_model(batch.x, batch.edge_index) #, graph=batch)
                                
                                if is_regression:
                                    val_predictions.append(out.detach())
                                else:
                                    val_predictions.append(out.argmax(dim=1).detach())
                                
                                val_targets.append(batch.y.detach())
                        
                        # Calculate validation metric
                        val_pred = torch.cat(val_predictions, dim=0)
                        val_true = torch.cat(val_targets, dim=0)
                        val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                        
                        if is_regression:
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            elif config.regression_loss == 'mse':
                                val_metric = val_metrics['mse']
                                if val_metric < best_val_metric:  # Lower is better for MSE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                        else:
                            # For classification, use F1 score
                            val_metric = val_metrics['f1_macro']
                            if val_metric > best_val_metric:  # Higher is better for F1
                                best_val_metric = val_metric
                                patience_counter = 0
                            else:
                                patience_counter += 1

                        if patience_counter >= patience:
                            break
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss in ['mae', 'mse']:
                            final_metric = -best_val_metric  # Negative because we minimize MAE
                        else:
                            final_metric = best_val_metric  # Positive because we maximize F1
                    else:
                        final_metric = best_val_metric  # Positive because we maximize F1
                    
                    return final_metric
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)

                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")

                    # Recreate model with optimized parameters
                    # model = InductiveSheafDiffusionModel(
                    #     input_dim=input_dim,
                    #     hidden_dim=best_params.get('hidden_dim', config.hidden_dim),
                    #     output_dim=output_dim,
                    #     sheaf_type=best_params['sheaf_type'],
                    #     num_layers=best_params['num_layers'],
                    #     dropout=best_params['dropout'],
                    #     d=best_params['d'],
                    #     is_regression=is_regression,
                    #     is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count'),
                    #     device=device
                    # )

                    model = SheafDiffusionModel(
                        input_dim=input_dim,
                        hidden_dim=best_params.get('hidden_dim', config.hidden_dim),
                        output_dim=output_dim,
                        sheaf_type=best_params['sheaf_type'],
                        num_layers=best_params['num_layers'],
                        dropout=best_params['dropout'],
                        d=best_params['d'],
                        is_regression=is_regression,
                        is_graph_level_task=config.is_graph_level_tasks.get(task, task == 'triangle_count')
                        ).to(device)
                    
                    # # Initialize the sheaf model parameters
                    # print("Initializing optimized sheaf model parameters...")
                    # with torch.no_grad():
                    #     first_batch = next(iter(dataloaders[first_fold_name]['train']))
                    #     # Ensure data is on the same device as the model
                    #     first_batch = first_batch.to(device)
                    #     if hasattr(first_batch, 'batch') and first_batch.batch is not None:
                    #         _ = model(first_batch.x, first_batch.edge_index, first_batch.batch)
                    #     else:
                    #         _ = model(first_batch.x, first_batch.edge_index)
                    
                    # param_count = sum(p.numel() for p in model.parameters())
                    # print(f"Optimized sheaf model initialized with {param_count} parameters")
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
                    config.num_layers = best_params.get('num_layers', config.num_layers)
                    config.dropout = best_params.get('dropout', config.dropout)
                    config.hidden_dim = best_params.get('hidden_dim', config.hidden_dim)
        
        # Handle GNN models
        elif model_name in ['gat', 'fagcn', 'gin', 'gcn', 'sage']:
            print(f"🔄 Training GNN: {model_name}")
            
            # Update hyperparameter optimization for GNNs
            if optimize_hyperparams and not isinstance(model, SklearnModel):
                print(f"🎯 Optimizing GNN hyperparameters...")
                
                # Get model creator function for GNNs
                model_creator = lambda **kwargs: GNNModel(**kwargs)
                
                # Get sample batch to determine dimensions
                first_fold_name = list(dataloaders.keys())[0]
                sample_batch = next(iter(dataloaders[first_fold_name]['train']))
                input_dim = sample_batch.x.shape[1]
                
                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)
                
                # Create a single optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"gnn_{model_name}_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                # Store GNN type for the objective function
                gnn_type = model_name
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    # Use config patience instead of optimizing it
                    patience = config.patience
                    
                    # GNN-specific hyperparameters
                    hidden_dim = config.hidden_dim #trial.suggest_int('hidden_dim', 32, 256)
                    if task.startswith('k_hop_community_counts'):
                        num_layers = trial.suggest_int('num_layers', config.khop_community_counts_k + 1, config.khop_community_counts_k + 2)
                    else:
                        num_layers = config.num_layers

                    dropout = trial.suggest_float('dropout', 0.1, 0.5)
                    
                    # Initialize default values
                    heads = 1
                    concat_heads = True
                    eps = 0.3
                    residual = False
                    norm_type = 'none'
                    agg_type = 'sum'
                    
                    # GNN-specific parameters
                    if gnn_type == "gat":
                        # First suggest number of heads
                        heads = trial.suggest_int('heads', 1, 8)
                        # Then suggest a base dimension and multiply by heads to ensure divisibility
                        base_dim = trial.suggest_int('base_dim', 8, 32)
                        hidden_dim = base_dim * heads
                        concat_heads = trial.suggest_categorical('concat_heads', [True, False])
                        residual = trial.suggest_categorical('residual', [True, False])
                        norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                        agg_type = trial.suggest_categorical('agg_type', ['mean', 'sum'])
                    elif gnn_type == "fagcn":
                        eps = trial.suggest_float('eps', 0.0, 1.0)
                    elif gnn_type in ["gcn", "sage", "gin"]:
                        residual = trial.suggest_categorical('residual', [True, False])
                        norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                        if gnn_type in ["sage", "gcn"]:
                            agg_type = trial.suggest_categorical('agg_type', ['mean', 'sum'])
                    
                    # Create GNN model with all parameters
                    trial_model = GNNModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        gnn_type=gnn_type,
                        residual=residual,
                        norm_type=norm_type,
                        agg_type=agg_type,
                        heads=heads,
                        concat_heads=concat_heads,
                        eps=eps,
                        is_regression=is_regression,
                        is_graph_level_task=task == 'triangle_count'
                    ).to(device)
                    
                    # Train model with reduced epochs for speed
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        elif config.regression_loss == 'mse':
                            criterion = torch.nn.MSELoss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0

                    fold_dataloader = dataloaders[first_fold_name]

                    # Check for one batch if already on device (which means all are on device already)
                    for batch in fold_dataloader['train']:
                        if batch.x.device == device:
                            all_data_on_device = True
                            break
                        else:
                            all_data_on_device = False
                            break

                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in fold_dataloader['train']:
                            if not all_data_on_device:
                                batch = batch.to(device)
                            optimizer.zero_grad()
                            out = trial_model(batch.x, batch.edge_index)
                            loss = criterion(out, batch.y)
                            loss.backward()
                            optimizer.step()
                        
                        # Validation
                        trial_model.eval()
                        val_predictions = []
                        val_targets = []
                        
                        with torch.no_grad():
                            for batch in fold_dataloader['val']:
                                if not all_data_on_device:
                                    batch = batch.to(device)
                                out = trial_model(batch.x, batch.edge_index)
                                
                                if is_regression:
                                    val_predictions.append(out.detach())
                                else:
                                    val_predictions.append(out.argmax(dim=1).detach())
                                
                                val_targets.append(batch.y.detach())
                        
                        # Calculate validation metric
                        val_pred = torch.cat(val_predictions, dim=0)
                        val_true = torch.cat(val_targets, dim=0)
                        val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                        
                        if is_regression:
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            elif config.regression_loss == 'mse':
                                val_metric = val_metrics['mse']
                                if val_metric < best_val_metric:  # Lower is better for MSE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            else:
                                val_metric = val_metrics['r2']
                                if val_metric > best_val_metric:  # Higher is better for R²
                                    best_val_metric = val_metric
                                    patience_counter = 0
                        else:
                            val_metric = val_metrics['f1_macro']
                            if val_metric > best_val_metric:  # Higher is better for F1
                                best_val_metric = val_metric
                                patience_counter = 0
                        
                        if patience_counter >= patience:
                            break
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss in ['mae', 'mse']:
                            final_metric = -best_val_metric  # Negative because we minimize MAE
                        else:
                            final_metric = best_val_metric  # Positive because we maximize R²
                    else:
                        final_metric = best_val_metric  # Positive because we maximize F1
                    
                    return final_metric
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    if gnn_type == 'gat':
                        # For GAT, ensure hidden_dim is properly calculated from base_dim and heads
                        base_dim = best_params.get('base_dim', 8)
                        heads = best_params.get('heads', 1)
                        hidden_dim = base_dim * heads
                    else:
                        hidden_dim = best_params.get('hidden_dim', 64)

                    model = GNNModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        num_layers=best_params.get('num_layers', 2),
                        dropout=best_params.get('dropout', 0.5),
                        gnn_type=gnn_type,
                        residual=best_params.get('residual', False),
                        norm_type=best_params.get('norm_type', 'none'),
                        agg_type=best_params.get('agg_type', 'sum'),
                        heads=best_params.get('heads', 1),
                        concat_heads=best_params.get('concat_heads', True),
                        eps=best_params.get('eps', 0.3),
                        is_regression=is_regression,
                        is_graph_level_task=task == 'triangle_count'
                    ).to(device)
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
                    config.num_layers = best_params.get('num_layers', config.num_layers)
        
        # Handle MLP models
        elif isinstance(model, MLPModel):
            print(f"🔄 Training MLP")
            
            # Update hyperparameter optimization for MLPs
            if optimize_hyperparams:
                print(f"🎯 Optimizing MLP hyperparameters...")
                
                # Get sample batch to determine dimensions
                first_fold_name = list(dataloaders.keys())[0]
                sample_batch = next(iter(dataloaders[first_fold_name]['train']))
                input_dim = sample_batch.x.shape[1]

                if sample_batch.x.device != device:
                    print("Batches are not on device")
                    all_data_on_device = False
                else:
                    print("All batches are already on device")
                    all_data_on_device = True
                
                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)
                
                # Create optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"mlp_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    # Use config patience instead of optimizing it
                    patience = config.patience
                    
                    # MLP-specific hyperparameters
                    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
                    num_layers = trial.suggest_int('num_layers', 1, 4)
                    dropout = trial.suggest_float('dropout', 0.1, 0.7)
                    
                    # Create MLP model with all parameters
                    trial_model = MLPModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        is_regression=is_regression
                    ).to(device)
                    
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        elif config.regression_loss == 'mse':
                            criterion = torch.nn.MSELoss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0
                    
                    fold_dataloader = dataloaders[first_fold_name]

                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in fold_dataloader['train']:
                            if not all_data_on_device:
                                batch = batch.to(device)
                            optimizer.zero_grad()
                            out = trial_model(batch.x)
                            loss = criterion(out, batch.y)
                            loss.backward()
                            optimizer.step()
                        
                        # Validation
                        trial_model.eval()
                        val_predictions = []
                        val_targets = []
                        
                        with torch.no_grad():
                            for batch in fold_dataloader['val']:
                                if not all_data_on_device:
                                    batch = batch.to(device)
                                out = trial_model(batch.x)
                                
                                if is_regression:
                                    val_predictions.append(out.detach())
                                else:
                                    val_predictions.append(out.argmax(dim=1).detach())
                                
                                val_targets.append(batch.y.detach())
                        
                        # Calculate validation metric
                        val_pred = torch.cat(val_predictions, dim=0)
                        val_true = torch.cat(val_targets, dim=0)
                        val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                        
                        if is_regression:
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            elif config.regression_loss == 'mse':
                                val_metric = val_metrics['mse']
                                if val_metric < best_val_metric:  # Lower is better for MSE
                                    best_val_metric = val_metric
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                            else:
                                val_metric = val_metrics['r2']
                                if val_metric > best_val_metric:  # Higher is better for R²
                                    best_val_metric = val_metric
                                    patience_counter = 0
                        else:
                            val_metric = val_metrics['f1_macro']
                            if val_metric > best_val_metric:  # Higher is better for F1
                                best_val_metric = val_metric
                                patience_counter = 0
                        
                        if patience_counter >= patience:
                            break
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss in ['mae', 'mse']:
                            final_metric = -best_val_metric  # Negative because we minimize MAE
                        else:
                            final_metric = best_val_metric  # Positive because we maximize R²
                    else:
                        final_metric = best_val_metric  # Positive because we maximize F1
                    
                    return final_metric
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    model = MLPModel(
                        input_dim=input_dim,
                        hidden_dim=best_params.get('hidden_dim', 64),
                        output_dim=output_dim,
                        num_layers=best_params.get('num_layers', 2),
                        dropout=best_params.get('dropout', 0.5),
                        is_regression=is_regression
                    ).to(device)
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
        
        # Handle sklearn models
        elif isinstance(model, SklearnModel):
            print(f"🔄 Training sklearn model: {model.model_type}")
            
            # Update hyperparameter optimization for sklearn models
            if optimize_hyperparams:
                print(f"🎯 Optimizing sklearn model hyperparameters...")
                
                # Get sample batch to determine dimensions
                first_fold_name = list(dataloaders.keys())[0]
                sample_batch = next(iter(dataloaders[first_fold_name]['train']))
                input_dim = sample_batch.x.shape[1]
                
                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)
                
                # Create optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"sklearn_{model.model_type}_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                def objective(trial: Trial) -> float:
                    # Model-specific hyperparameters
                    if model.model_type == "rf":  # Random Forest
                        n_estimators = trial.suggest_int('n_estimators', 50, 500)
                        max_depth = trial.suggest_int('max_depth', 3, 20)
                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                        
                        trial_model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="rf",
                            is_regression=is_regression,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf
                        )
                    
                    elif model.model_type == "svm":  # Support Vector Machine
                        C = trial.suggest_float('C', 0.1, 10.0, log=True)
                        gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
                        
                        trial_model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="svm",
                            is_regression=is_regression,
                            C=C,
                            gamma=gamma
                        )
                    
                    elif model.model_type == "knn":  # K-Nearest Neighbors
                        n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
                        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                        
                        trial_model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="knn",
                            is_regression=is_regression,
                            n_neighbors=n_neighbors,
                            weights=weights
                        )
                    
                    else:  # Default to basic model
                        trial_model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type=model.model_type,
                            is_regression=is_regression
                        )
                    
                    # Train and evaluate model
                    fold_dataloader = dataloaders[first_fold_name]
                    results = train_sklearn_inductive(trial_model, fold_dataloader, config, is_regression)
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss == 'mae':
                            final_metric = -results['test_metrics']['mae']  # Negative because we minimize MAE
                        else:
                            final_metric = results['test_metrics']['r2']  # Positive because we maximize R²
                    else:
                        final_metric = results['test_metrics']['f1_macro']  # Positive because we maximize F1
                    
                    return final_metric
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    if model.model_type == "rf":
                        model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="rf",
                            is_regression=is_regression,
                            n_estimators=best_params.get('n_estimators', 100),
                            max_depth=best_params.get('max_depth', None),
                            min_samples_split=best_params.get('min_samples_split', 2),
                            min_samples_leaf=best_params.get('min_samples_leaf', 1)
                        )
                    elif model.model_type == "svm":
                        model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="svm",
                            is_regression=is_regression,
                            C=best_params.get('C', 1.0),
                            gamma=best_params.get('gamma', 'scale')
                        )
                    elif model.model_type == "knn":
                        model = SklearnModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type="knn",
                            is_regression=is_regression,
                            n_neighbors=best_params.get('n_neighbors', 5),
                            weights=best_params.get('weights', 'uniform')
                        )
        
    # Train the model (either original or optimized)
    results = train_inductive_model(
        model=model,
        model_name=model_name,
        dataloaders=dataloaders,
        config=config,
        task=task,
        device=device,
        finetuning=finetuning,
    )
    
    # Combine results
    if optimize_hyperparams:
        results['optimal_hyperparams'] = best_params
    else:
        results['optimal_hyperparams'] = {}
    
    return results

def cleanup_gpu_dataloaders(dataloaders: Dict[str, Dict[str, Any]], device: torch.device):
    """
    Clean up GPU memory by removing all data from GPU.
    
    Args:
        dataloaders: GPU-resident dataloaders to clean up
        device: Device to clean up
    """
    print(f"🧹 Cleaning up GPU memory on {device}...")
    
    for task, task_data in dataloaders.items():
        for fold_name, fold_data in task_data.items():
            for split_name, dataloader in fold_data.items():
                if split_name == 'metadata':
                    continue
                
                # Clear all batches from GPU
                for batch in dataloader:
                    # Move batch data to CPU and delete
                    if hasattr(batch, 'x'):
                        batch.x = batch.x.cpu()
                    if hasattr(batch, 'edge_index'):
                        batch.edge_index = batch.edge_index.cpu()
                    if hasattr(batch, 'y'):
                        batch.y = batch.y.cpu()
                    if hasattr(batch, 'batch'):
                        batch.batch = batch.batch.cpu()
                    
                    # Clear any PE tensors
                    for attr_name in dir(batch):
                        if attr_name.endswith('_pe'):
                            pe_tensor = getattr(batch, attr_name)
                            if hasattr(pe_tensor, 'cpu'):
                                setattr(batch, attr_name, pe_tensor.cpu())
                
                # Clear the dataloader itself
                del dataloader
    
    # Force GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print(f"✅ GPU memory cleaned up")