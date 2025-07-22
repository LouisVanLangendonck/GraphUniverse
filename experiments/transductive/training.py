"""
Training utilities for transductive graph learning experiments.
Based on inductive training but adapted for single-graph transductive learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import copy
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from experiments.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel
from experiments.metrics import compute_metrics
from experiments.transductive.config import TransductiveExperimentConfig
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_transductive_model(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Train a model for transductive learning on a single graph.
    
    Args:
        model: Model to train
        task_data: Dictionary with features, edge_index, labels, and splits
        config: Experiment configuration
        task: Current task name
        device: Device to use for training
        
    Returns:
        Dictionary with training results and metrics
    """
    is_regression = config.is_regression.get(task, False)
    
    # Handle sklearn models separately
    if isinstance(model, SklearnModel):
        return train_sklearn_transductive(model, task_data, config, is_regression)
    
    # PyTorch models
    model = model.to(device)
    
    # Move data to device
    features = task_data['features'].to(device)
    labels = task_data['labels'].to(device)
    train_idx = task_data['train_idx'].to(device)
    val_idx = task_data['val_idx'].to(device)
    test_idx = task_data['test_idx'].to(device)
    
    # Handle edge_index for graph models
    edge_index = None
    if 'edge_index' in task_data:
        edge_index = task_data['edge_index'].to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Set up loss function
    if is_regression:
        if config.regression_loss == 'mae':
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_metric = float('-inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    start_time = time.time()
    
    print(f"\nStarting training with patience={config.patience}")
    
    for epoch in tqdm(range(config.epochs), desc="Training Model"):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass - check if model requires edge_index
        if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
            if edge_index is None:
                raise ValueError("Graph models require edge_index")
            out = model(features, edge_index)
        else:  # MLPModel
            out = model(features)
        
        # Compute loss on training nodes
        loss = criterion(out[train_idx], labels[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
                out = model(features, edge_index)
            else:
                out = model(features)
            
            # Compute losses
            train_loss = criterion(out[train_idx], labels[train_idx])
            val_loss = criterion(out[val_idx], labels[val_idx])
            
            # Compute metrics
            if is_regression:
                train_pred = out[train_idx].cpu().numpy()
                train_true = labels[train_idx].cpu().numpy()
                val_pred = out[val_idx].cpu().numpy()
                val_true = labels[val_idx].cpu().numpy()
            else:
                train_pred = out[train_idx].argmax(dim=1).cpu().numpy()
                train_true = labels[train_idx].cpu().numpy()
                val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                val_true = labels[val_idx].cpu().numpy()
            
            train_metrics = compute_metrics(train_true, train_pred, is_regression)
            val_metrics = compute_metrics(val_true, val_pred, is_regression)
            
            # Get primary metrics
            if is_regression:
                if config.regression_loss == 'mae':
                    train_metric = train_metrics['mae']
                    val_metric = val_metrics['mae']
                else:
                    train_metric = train_metrics['r2']
                    val_metric = val_metrics['r2']
            else:
                train_metric = train_metrics['f1_macro']
                val_metric = val_metrics['f1_macro']
        
        # Store metrics
        training_history['train_loss'].append(train_loss.item())
        training_history['val_loss'].append(val_loss.item())
        training_history['train_metric'].append(train_metric)
        training_history['val_metric'].append(val_metric)
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                  f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}")
        
        # Model selection
        improved = False
        if is_regression:
            if config.regression_loss == 'mae':
                # For MAE, lower is better
                if val_metric < best_val_metric or best_val_metric == float('-inf'):
                    improved = True
            else:
                # For R², higher is better
                if val_metric > best_val_metric:
                    improved = True
        else:
            if val_metric > best_val_metric:  # Higher F1 is better
                improved = True
        
        if improved:
            best_val_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break
    
    train_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")
    
    # Final evaluation on test set
    test_metrics = evaluate_transductive_model(
        model, task_data, is_regression, device
    )
    
    return {
        'train_time': train_time,
        'best_val_metric': best_val_metric,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'model': model
    }

def train_sklearn_transductive(
    model: SklearnModel,
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    is_regression: bool
) -> Dict[str, Any]:
    """
    Train a sklearn model for transductive learning.
    
    Args:
        model: Sklearn model to train
        task_data: Dictionary with features, labels, and splits
        config: Experiment configuration
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary with training results
    """
    # Extract features and labels for training nodes
    features = task_data['features'].cpu().numpy()
    labels = task_data['labels'].cpu().numpy()
    train_idx = task_data['train_idx'].cpu().numpy()
    test_idx = task_data['test_idx'].cpu().numpy()
    
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]
    
    # Handle label shape for sklearn
    if len(y_train.shape) > 1 and not is_regression:
        y_train = y_train.squeeze()
    if len(y_test.shape) > 1 and not is_regression:
        y_test = y_test.squeeze()
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    test_metrics = compute_metrics(y_test, y_pred, is_regression)
    
    return {
        'train_time': train_time,
        'test_metrics': test_metrics,
        'model': model
    }

def evaluate_transductive_model(
    model: Union[GNNModel, MLPModel, GraphTransformerModel],
    task_data: Dict[str, Any],
    is_regression: bool,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set for transductive learning.
    
    Args:
        model: Trained model
        task_data: Dictionary with features, edge_index, labels, and splits
        is_regression: Whether this is a regression task
        device: Device to use for evaluation
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    
    features = task_data['features'].to(device)
    labels = task_data['labels'].to(device)
    test_idx = task_data['test_idx'].to(device)
    
    # Handle edge_index for graph models
    edge_index = None
    if 'edge_index' in task_data:
        edge_index = task_data['edge_index'].to(device)
    
    with torch.no_grad():
        # Forward pass
        if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
            if edge_index is None:
                raise ValueError("Graph models require edge_index")
            out = model(features, edge_index)
        else:  # MLPModel
            out = model(features)
        
        # Extract test predictions
        if is_regression:
            test_pred = out[test_idx].cpu().numpy()
        else:
            test_pred = out[test_idx].argmax(dim=1).cpu().numpy()
        
        test_true = labels[test_idx].cpu().numpy()
    
    # Compute metrics
    metrics = compute_metrics(test_true, test_pred, is_regression)
    
    return metrics

def train_transductive_model_gpu_resident(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Train a model for transductive learning on a single graph with GPU-resident data.
    Assumes all data is already on GPU to avoid repeated CPU-GPU transfers.
    
    Args:
        model: Model to train
        task_data: Dictionary with features, edge_index, labels, and splits (all on GPU)
        config: Experiment configuration
        task: Current task name
        device: Device to use for training
        
    Returns:
        Dictionary with training results and metrics
    """
    is_regression = config.is_regression.get(task, False)
    
    # Handle sklearn models separately
    if isinstance(model, SklearnModel):
        return train_sklearn_transductive(model, task_data, config, is_regression)
    
    # PyTorch models - data is already on GPU
    model = model.to(device)
    
    # Data is already on GPU - no need for .to(device)
    features = task_data['features']  # Already on GPU
    labels = task_data['labels']  # Already on GPU
    train_idx = task_data['train_idx']  # Already on GPU
    val_idx = task_data['val_idx']  # Already on GPU
    test_idx = task_data['test_idx']  # Already on GPU
    
    # Handle edge_index for graph models
    edge_index = None
    if 'edge_index' in task_data:
        edge_index = task_data['edge_index']  # Already on GPU
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Set up loss function
    if is_regression:
        if config.regression_loss == 'mae':
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_metric = float('-inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    start_time = time.time()
    
    print(f"\nStarting GPU-resident training with patience={config.patience}")
    
    for epoch in tqdm(range(config.epochs), desc="Training Model"):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass - check if model requires edge_index
        if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
            if edge_index is None:
                raise ValueError("Graph models require edge_index")
            out = model(features, edge_index)
        else:  # MLPModel
            out = model(features)
        
        # Compute loss on training nodes
        loss = criterion(out[train_idx], labels[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
                out = model(features, edge_index)
            else:
                out = model(features)
            
            # Compute losses
            train_loss = criterion(out[train_idx], labels[train_idx])
            val_loss = criterion(out[val_idx], labels[val_idx])
            
            # Compute metrics on GPU to avoid CPU transfers
            if is_regression:
                train_pred = out[train_idx]
                train_true = labels[train_idx]
                val_pred = out[val_idx]
                val_true = labels[val_idx]
            else:
                train_pred = out[train_idx].argmax(dim=1)
                train_true = labels[train_idx]
                val_pred = out[val_idx].argmax(dim=1)
                val_true = labels[val_idx]
            
            # Use comprehensive GPU-based metrics computation
            from experiments.metrics import compute_metrics_gpu
            train_metrics = compute_metrics_gpu(train_true, train_pred, is_regression)
            val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
            
            # Get primary metrics
            if is_regression:
                if config.regression_loss == 'mae':
                    train_metric = train_metrics['mae']
                    val_metric = val_metrics['mae']
                else:
                    train_metric = train_metrics['mse']
                    val_metric = val_metrics['mse']
            else:
                # For classification, use F1 macro as primary metric
                train_metric = train_metrics['f1_macro']
                val_metric = val_metrics['f1_macro']
            
            # Store history
            training_history['train_loss'].append(train_loss.item())
            training_history['val_loss'].append(val_loss.item())
            training_history['train_metric'].append(train_metric)
            training_history['val_metric'].append(val_metric)
            
            # Early stopping
            if is_regression:
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
            out = model(features, edge_index)
        else:
            out = model(features)
        
        # Compute test metrics on GPU
        if is_regression:
            test_pred = out[test_idx]
            test_true = labels[test_idx]
        else:
            test_pred = out[test_idx].argmax(dim=1)
            test_true = labels[test_idx]
        
        # Use full metrics for final evaluation
        from experiments.metrics import compute_metrics_gpu
        test_metrics = compute_metrics_gpu(test_true, test_pred, is_regression)
    
    training_time = time.time() - start_time
    
    return {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'best_val_metric': best_val_metric,
        'final_epoch': epoch + 1
    }

def evaluate_transductive_model_gpu_resident(
    model: Union[GNNModel, MLPModel, GraphTransformerModel],
    task_data: Dict[str, Any],
    is_regression: bool,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set for transductive learning with GPU-resident data.
    Assumes all data is already on GPU for efficiency.
    
    Args:
        model: Trained model
        task_data: Dictionary with features, edge_index, labels, and splits (all on GPU)
        is_regression: Whether this is a regression task
        device: Device to use for evaluation
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    
    # Data is already on GPU - no need for .to(device)
    features = task_data['features']  # Already on GPU
    labels = task_data['labels']  # Already on GPU
    test_idx = task_data['test_idx']  # Already on GPU
    
    # Handle edge_index for graph models
    edge_index = None
    if 'edge_index' in task_data:
        edge_index = task_data['edge_index']  # Already on GPU
    
    with torch.no_grad():
        # Forward pass
        if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type'):
            if edge_index is None:
                raise ValueError("Graph models require edge_index")
            out = model(features, edge_index)
        else:  # MLPModel
            out = model(features)
        
        # Extract test predictions - keep on GPU
        if is_regression:
            test_pred = out[test_idx]
        else:
            test_pred = out[test_idx].argmax(dim=1)
        
        test_true = labels[test_idx]
    
    # Compute metrics on GPU
    from experiments.metrics import compute_metrics_gpu
    metrics = compute_metrics_gpu(test_true, test_pred, is_regression)
    
    return metrics

def tune_hyperparameters_transductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    model_name: str = 'gcn',
    model_creator: Callable = None
) -> dict:
    """Run Optuna hyperparameter tuning for transductive (CPU) models and return best hyperparameters."""
    import copy
    import optuna
    from optuna import create_study, Trial
    from experiments.metrics import compute_metrics
    is_regression = task != 'community'
    is_graph_level_task = False
    def instantiate_model(hp):
        if model_creator is not None:
            return model_creator(
                model_name,
                task_data['input_dim'],
                task_data['metadata']['output_dim'],
                is_regression,
                is_graph_level_task,
                hp
            ).to(device)
        else:
            ModelClass = type(model)
            return ModelClass(
                input_dim=task_data['input_dim'],
                hidden_dim=hp.get('hidden_dim', getattr(model, 'hidden_dim', 32)),
                output_dim=task_data['metadata']['output_dim'],
                num_layers=hp.get('num_layers', getattr(model, 'num_layers', 2)),
                dropout=hp.get('dropout', getattr(model, 'dropout', 0.5)),
                gnn_type=model_name if hasattr(model, 'gnn_type') else None,
                is_regression=is_regression,
                is_graph_level_task=is_graph_level_task,
                pe_type=hp.get('pe_type', getattr(model, 'pe_type', 'laplacian')),
                pe_dim=hp.get('pe_dim', getattr(model, 'pe_dim', 8)),
            ).to(device)
    def objective(trial: Trial) -> float:
        hp = {}
        hp['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        hp['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hp['hidden_dim'] = trial.suggest_int('hidden_dim', 16, 64, step=16)
        hp['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        hp['dropout'] = trial.suggest_float('dropout', 0.0, 0.7)
        hp['pe_type'] = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
        hp['pe_dim'] = trial.suggest_categorical('pe_dim', [8])
        if model_name in ['gat']:
            hp['heads'] = trial.suggest_int('heads', 1, 8)
            hp['concat_heads'] = trial.suggest_categorical('concat_heads', [True, False])
        if model_name in ['fagcn', 'gin']:
            hp['eps'] = trial.suggest_float('eps', 0.0, 1.0)
        if model_name in ['graphgps']:
            hp['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            hp['base_dim'] = trial.suggest_int('base_dim', 4, 32)
            hp['hidden_dim'] = hp['base_dim'] * hp['num_heads']
            hp['local_gnn_type'] = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
            hp['attn_type'] = trial.suggest_categorical('attn_type', ['performer', 'multihead'])
        trial_model = instantiate_model(hp)
        optimizer = torch.optim.Adam(trial_model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        if is_regression:
            criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        max_epochs = min(100, config.epochs // 4)
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        best_model_state = None
        graph = task_data['pyg_graph']
        train_idx, val_idx = task_data['train_idx'], task_data['val_idx']
        labels = task_data['labels']
        for epoch in range(max_epochs):
            trial_model.train()
            optimizer.zero_grad()
            out = trial_model(graph.x, graph.edge_index, graph=graph)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            trial_model.eval()
            with torch.no_grad():
                out = trial_model(graph.x, graph.edge_index, graph=graph)
                if is_regression:
                    val_pred = out[val_idx].cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    val_metrics = compute_metrics(val_true, val_pred, is_regression)
                    val_metric = val_metrics['mae'] if config.regression_loss == 'mae' else val_metrics['mse']
                    improved = val_metric < best_val_metric
                else:
                    val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    val_metrics = compute_metrics(val_true, val_pred, is_regression)
                    val_metric = val_metrics['f1_macro']
                    improved = val_metric > best_val_metric
                if improved:
                    best_val_metric = val_metric
                    best_model_state = copy.deepcopy(trial_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            if patience_counter >= config.patience:
                break
        return best_val_metric if is_regression else -best_val_metric
    direction = 'minimize' if is_regression else 'maximize'
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=20, timeout=600)
    best_hyperparams = study.best_params
    if not is_regression:
        best_hyperparams = study.best_trial.params
    return best_hyperparams

def train_and_evaluate_transductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    model_name: str = 'gcn',
    hyperparams: dict = None,
    model_creator: Callable = None
) -> Dict[str, Any]:
    """Train and evaluate a model on a single split using provided hyperparameters (no Optuna inside)."""
    import copy
    import time
    from experiments.metrics import compute_metrics
    is_regression = task != 'community'
    is_graph_level_task = False
    hp = hyperparams if hyperparams is not None else {}
    def instantiate_model(hp):
        if model_creator is not None:
            return model_creator(
                model_name,
                task_data['input_dim'],
                task_data['metadata']['output_dim'],
                is_regression,
                is_graph_level_task,
                hp
            ).to(device)
        else:
            ModelClass = type(model)
            return ModelClass(
                input_dim=task_data['input_dim'],
                hidden_dim=hp.get('hidden_dim', getattr(model, 'hidden_dim', 32)),
                output_dim=task_data['metadata']['output_dim'],
                num_layers=hp.get('num_layers', getattr(model, 'num_layers', 2)),
                dropout=hp.get('dropout', getattr(model, 'dropout', 0.5)),
                gnn_type=model_name if hasattr(model, 'gnn_type') else None,
                is_regression=is_regression,
                is_graph_level_task=is_graph_level_task,
                pe_type=hp.get('pe_type', getattr(model, 'pe_type', 'laplacian')),
                pe_dim=hp.get('pe_dim', getattr(model, 'pe_dim', 8)),
            ).to(device)
    model = instantiate_model(hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.get('learning_rate', 0.01), weight_decay=hp.get('weight_decay', 5e-4))
    if is_regression:
        criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    best_val_metric = float('inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    graph = task_data['pyg_graph']
    train_idx, val_idx, test_idx = task_data['train_idx'], task_data['val_idx'], task_data['test_idx']
    labels = task_data['labels']
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index, graph=graph)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph=graph)
            if is_regression:
                val_pred = out[val_idx].cpu().numpy()
                val_true = labels[val_idx].cpu().numpy()
                val_metrics = compute_metrics(val_true, val_pred, is_regression)
                val_metric = val_metrics['mae'] if config.regression_loss == 'mae' else val_metrics['mse']
                improved = val_metric < best_val_metric
            else:
                val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                val_true = labels[val_idx].cpu().numpy()
                val_metrics = compute_metrics(val_true, val_pred, is_regression)
                val_metric = val_metrics['f1_macro']
                improved = val_metric > best_val_metric
            if improved:
                best_val_metric = val_metric
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
        if patience_counter >= config.patience:
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph=graph)
        if is_regression:
            test_pred = out[test_idx].cpu().numpy()
            test_true = labels[test_idx].cpu().numpy()
            test_metrics = compute_metrics(test_true, test_pred, is_regression)
            test_metric = test_metrics['mae'] if config.regression_loss == 'mae' else test_metrics['mse']
        else:
            test_pred = out[test_idx].argmax(dim=1).cpu().numpy()
            test_true = labels[test_idx].cpu().numpy()
            test_metrics = compute_metrics(test_true, test_pred, is_regression)
            test_metric = test_metrics['f1_macro']
    train_time = time.time() - start_time
    print(f"Test metric: {test_metric}")
    return {
        'test_metrics': test_metrics,
        'train_time': train_time,
        'optimal_hyperparams': hp
    }

def tune_hyperparameters_transductive_gpu_resident(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    model_name: str = 'gcn',
    model_creator: Callable = None
) -> dict:
    """Run Optuna hyperparameter tuning for transductive (GPU) models and return best hyperparameters."""
    import copy
    import optuna
    from optuna import create_study, Trial
    from experiments.metrics import compute_metrics_gpu
    is_regression = task != 'community'
    print(f"is_regression: {is_regression}")
    is_graph_level_task = False
    def instantiate_model(hp):
        if model_creator is not None:
            return model_creator(
                model_name,
                task_data['input_dim'],
                task_data['metadata']['output_dim'],
                is_regression,
                is_graph_level_task,
                hp
            ).to(device)
        else:
            ModelClass = type(model)
            # Patch: For GAT, set readout_in_dim based on concat_heads
            if model_name == 'gat':
                heads = hp.get('heads', getattr(model, 'heads', 1))
                concat_heads = hp.get('concat_heads', getattr(model, 'concat_heads', True))
                hidden_dim = hp.get('hidden_dim', getattr(model, 'hidden_dim', 32))
                if concat_heads:
                    readout_in_dim = hidden_dim
                else:
                    readout_in_dim = hidden_dim
                # Note: The GNNModel expects hidden_dim as the input to the readout, but the encoder will output hidden_dim * heads if concat_heads is True
                # So, we need to patch GNNModel to accept readout_in_dim as a parameter if not already
                # For now, we keep as is, but this is where the fix would go if needed
            return ModelClass(
                input_dim=task_data['input_dim'],
                hidden_dim=hp.get('hidden_dim', getattr(model, 'hidden_dim', 32)),
                output_dim=task_data['metadata']['output_dim'],
                num_layers=hp.get('num_layers', getattr(model, 'num_layers', 2)),
                dropout=hp.get('dropout', getattr(model, 'dropout', 0.5)),
                gnn_type=model_name if hasattr(model, 'gnn_type') else None,
                is_regression=is_regression,
                is_graph_level_task=is_graph_level_task,
                heads=hp.get('heads', getattr(model, 'heads', 1)),
                concat_heads=hp.get('concat_heads', getattr(model, 'concat_heads', True)),
                eps=hp.get('eps', getattr(model, 'eps', 0.3)),
                pe_type=hp.get('pe_type', getattr(model, 'pe_type', 'laplacian')),
                pe_dim=hp.get('pe_dim', getattr(model, 'pe_dim', 8)),
            ).to(device)
    def objective(trial: Trial) -> float:
        hp = {}
        hp['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        hp['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hp['hidden_dim'] = trial.suggest_int('hidden_dim', 16, 64, step=16)
        if task.startswith('k_hop_community_counts'):
            hp['num_layers'] = trial.suggest_int('num_layers', config.khop_community_counts_k + 1, config.khop_community_counts_k + 2)
        else:
            hp['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        hp['dropout'] = trial.suggest_float('dropout', 0.0, 0.7)
        hp['pe_type'] = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
        hp['pe_dim'] = trial.suggest_categorical('pe_dim', [8])
        if model_name in ['gat']:
            hp['heads'] = trial.suggest_int('heads', 1, 8)
            hp['concat_heads'] = trial.suggest_categorical('concat_heads', [True, False])
            # Ensure hidden_dim is divisible by heads if concat_heads is True
            if hp['concat_heads']:
                # Adjust hidden_dim to be divisible by heads
                hp['hidden_dim'] = (hp['hidden_dim'] // hp['heads']) * hp['heads']
                if hp['hidden_dim'] < hp['heads']:
                    hp['hidden_dim'] = hp['heads']
        if model_name in ['fagcn', 'gin']:
            hp['eps'] = trial.suggest_float('eps', 0.0, 1.0)
        if model_name in ['graphgps']:
            hp['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
            hp['base_dim'] = trial.suggest_int('base_dim', 4, 32)
            hp['hidden_dim'] = hp['base_dim'] * hp['num_heads']
            hp['local_gnn_type'] = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
            hp['attn_type'] = trial.suggest_categorical('attn_type', ['performer', 'multihead'])
        if model_name in ['neural_sheaf']:
            hp['sheaf_type'] = trial.suggest_categorical('sheaf_type', ['diagonal', 'orthogonal', 'general'])
            hp['d'] = trial.suggest_int('d', 2, 3)
        trial_model = instantiate_model(hp)
        optimizer = torch.optim.Adam(trial_model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        if is_regression:
            criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        max_epochs = min(100, config.epochs // 4)
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        best_model_state = None
        graph = task_data['pyg_graph']
        train_idx, val_idx = task_data['train_idx'], task_data['val_idx']
        labels = task_data['labels']
        for epoch in range(max_epochs):
            trial_model.train()
            optimizer.zero_grad()
            out = trial_model(graph.x, graph.edge_index, graph=graph)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            trial_model.eval()
            with torch.no_grad():
                out = trial_model(graph.x, graph.edge_index, graph=graph)
                if is_regression:
                    val_pred = out[val_idx]
                    val_true = labels[val_idx]
                    val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                    val_metric = val_metrics['mae'] if config.regression_loss == 'mae' else val_metrics['mse']
                    improved = val_metric < best_val_metric
                else:
                    val_pred = out[val_idx].argmax(dim=1)
                    val_true = labels[val_idx]
                    val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                    val_metric = val_metrics['f1_macro']
                    improved = val_metric > best_val_metric
                if improved:
                    best_val_metric = val_metric
                    best_model_state = copy.deepcopy(trial_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            if patience_counter >= config.patience:
                break
        return -best_val_metric if is_regression else best_val_metric
    study = create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=600)
    best_hyperparams = study.best_params
    if not is_regression:
        best_hyperparams = study.best_trial.params
    return best_hyperparams

def train_and_evaluate_transductive_gpu_resident(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    model_name: str = 'gcn',
    hyperparams: dict = None,
    model_creator: Callable = None
) -> Dict[str, Any]:
    """Train and evaluate a model on a single split with GPU-resident data using provided hyperparameters (no Optuna inside)."""
    import copy
    import time
    from experiments.metrics import compute_metrics_gpu
    is_regression = task != 'community'
    is_graph_level_task = False
    hp = hyperparams if hyperparams is not None else {}
    def instantiate_model(hp):
        if model_creator is not None:
            return model_creator(
                model_name,
                task_data['input_dim'],
                task_data['metadata']['output_dim'],
                is_regression,
                is_graph_level_task,
                hp
            ).to(device)
        else:
            ModelClass = type(model)
            return ModelClass(
                input_dim=task_data['input_dim'],
                hidden_dim=hp.get('hidden_dim', getattr(model, 'hidden_dim', 32)),
                output_dim=task_data['metadata']['output_dim'],
                num_layers=hp.get('num_layers', getattr(model, 'num_layers', 2)),
                dropout=hp.get('dropout', getattr(model, 'dropout', 0.5)),
                gnn_type=model_name if hasattr(model, 'gnn_type') else None,
                is_regression=is_regression,
                is_graph_level_task=is_graph_level_task,
                pe_type=hp.get('pe_type', getattr(model, 'pe_type', 'laplacian')),
                pe_dim=hp.get('pe_dim', getattr(model, 'pe_dim', 8)),
            ).to(device)
    model = instantiate_model(hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.get('learning_rate', 0.01), weight_decay=hp.get('weight_decay', 5e-4))
    if is_regression:
        criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    best_val_metric = float('inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    graph = task_data['pyg_graph']
    train_idx, val_idx, test_idx = task_data['train_idx'], task_data['val_idx'], task_data['test_idx']
    labels = task_data['labels']
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index, graph=graph)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph=graph)
            if is_regression:
                val_pred = out[val_idx]
                val_true = labels[val_idx]
                val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                val_metric = val_metrics['mae'] if config.regression_loss == 'mae' else val_metrics['mse']
                improved = val_metric < best_val_metric
            else:
                val_pred = out[val_idx].argmax(dim=1)
                val_true = labels[val_idx]
                val_metrics = compute_metrics_gpu(val_true, val_pred, is_regression)
                val_metric = val_metrics['f1_macro']
                improved = val_metric > best_val_metric
            if improved:
                best_val_metric = val_metric
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
        if patience_counter >= config.patience:
            break
    if best_model is not None:
        model = best_model
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph=graph)
        if is_regression:
            test_pred = out[test_idx]
            test_true = labels[test_idx]
            test_metrics = compute_metrics_gpu(test_true, test_pred, is_regression)
            test_metric = test_metrics['mae'] if config.regression_loss == 'mae' else test_metrics['mse']
        else:
            test_pred = out[test_idx].argmax(dim=1)
            test_true = labels[test_idx]
            test_metrics = compute_metrics_gpu(test_true, test_pred, is_regression)
            test_metric = test_metrics['f1_macro']
    train_time = time.time() - start_time
    print(f"Test metric: {test_metric}")
    return {
        'test_metrics': test_metrics,
        'train_time': train_time,
        'optimal_hyperparams': hp
    }