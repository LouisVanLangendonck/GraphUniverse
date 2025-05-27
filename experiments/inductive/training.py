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

from experiments.core.models import GNNModel, MLPModel, SklearnModel
from experiments.core.metrics import compute_metrics, compute_loss
from experiments.inductive.config import InductiveExperimentConfig
from experiments.core.training import optimize_hyperparameters

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
    """Get the total number of unique classes across all data splits."""
    all_labels = set()
    
    # Skip metadata entry
    for split_name, dataloader in dataloaders.items():
        if split_name == 'metadata':
            continue
            
        # Process all batches in this dataloader
        for batch in dataloader:
            # Get labels from batch
            labels = batch.y
            all_labels.update(labels.cpu().numpy().tolist())
    
    # Get the maximum label value and add 1 to get number of classes
    # This ensures we handle non-contiguous labels correctly
    n_classes = max(all_labels) + 1
    
    print(f"Total unique classes found across all splits: {sorted(all_labels)}")
    print(f"Number of classes for model: {n_classes}")
    print(f"Label range: {min(all_labels)} to {max(all_labels)}")
    
    # Verify we have all classes
    expected_classes = set(range(n_classes))
    missing_classes = expected_classes - all_labels
    if missing_classes:
        print(f"Warning: Missing classes in data: {missing_classes}")
    
    return n_classes


def train_inductive_model(
    model: Union[GNNModel, MLPModel, SklearnModel],
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    task: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Train a model for inductive learning on graph families.
    
    Args:
        model: Model to train
        dataloaders: Dictionary with train/val/test dataloaders
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
    
    # PyTorch models
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    # Training loop
    best_val_metric = float('-inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, GNNModel):
                out = model(batch.x, batch.edge_index)
            else:  # MLPModel
                out = model(batch.x)
            
            # Compute loss
            loss = criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            if is_regression:
                train_predictions.append(out.detach().cpu())
                train_targets.append(batch.y.detach().cpu())
            else:
                train_predictions.append(out.argmax(dim=1).detach().cpu())
                train_targets.append(batch.y.detach().cpu())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                if isinstance(model, GNNModel):
                    out = model(batch.x, batch.edge_index)
                else:  # MLPModel
                    out = model(batch.x)
                
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                
                if is_regression:
                    val_predictions.append(out.detach().cpu())
                    val_targets.append(batch.y.detach().cpu())
                else:
                    val_predictions.append(out.argmax(dim=1).detach().cpu())
                    val_targets.append(batch.y.detach().cpu())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Concatenate predictions
        train_pred = torch.cat(train_predictions, dim=0).numpy()
        train_true = torch.cat(train_targets, dim=0).numpy()
        val_pred = torch.cat(val_predictions, dim=0).numpy()
        val_true = torch.cat(val_targets, dim=0).numpy()
        
        # Compute performance metrics
        train_metrics = compute_metrics(train_true, train_pred, is_regression)
        val_metrics = compute_metrics(val_true, val_pred, is_regression)
        
        if is_regression:
            train_metric = train_metrics['r2']
            val_metric = val_metrics['r2']
        else:
            train_metric = train_metrics['f1_macro']
            val_metric = val_metrics['f1_macro']
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_metric'].append(train_metric)
        training_history['val_metric'].append(val_metric)
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}")
        
        # Model selection
        improved = False
        if is_regression:
            if val_metric > best_val_metric:  # Higher R² is better
                improved = True
        else:
            if val_metric > best_val_metric:  # Higher F1 is better
                improved = True
        
        if improved:
            best_val_metric = val_metric
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    train_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_metrics = evaluate_inductive_model(model, dataloaders['test'], is_regression, device)
    
    return {
        'train_time': train_time,
        'best_val_metric': best_val_metric,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'model': model
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
        
        train_features.append(batch_features)
        train_labels.append(batch_labels)
    
    # Concatenate all training data
    X_train = np.vstack(train_features)
    y_train = np.hstack(train_labels)
    
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
        
        test_features.append(batch_features)
        test_labels.append(batch_labels)
    
    X_test = np.vstack(test_features)
    y_test = np.hstack(test_labels)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    test_metrics = compute_metrics(y_test, y_pred, is_regression)
    
    return {
        'train_time': train_time,
        'test_metrics': test_metrics,
        'model': model
    }


def evaluate_inductive_model(
    model: Union[GNNModel, MLPModel],
    test_loader: DataLoader,
    is_regression: bool,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set for inductive learning.
    
    Args:
        model: Trained model
        test_loader: Test data loader
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
            batch = batch.to(device)
            
            # Forward pass
            if isinstance(model, GNNModel):
                out = model(batch.x, batch.edge_index)
            else:  # MLPModel
                out = model(batch.x)
            
            # Store predictions
            if is_regression:
                all_predictions.append(out.detach().cpu())
            else:
                all_predictions.append(out.argmax(dim=1).detach().cpu())
            
            all_targets.append(batch.y.detach().cpu())
    
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    metrics = compute_metrics(targets, predictions, is_regression)
    
    return metrics


def optimize_inductive_hyperparameters(
    model_creator: Callable,
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    model_type: str = "gnn",
    gnn_type: Optional[str] = None,
    n_trials: int = 20,
    timeout: Optional[int] = 600,
    device: Optional[torch.device] = None,
    is_regression: bool = False,
    experiment_name: Optional[str] = None,
    run_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for inductive learning using validation graphs.
    
    Args:
        model_creator: Function that creates model instances
        dataloaders: Dictionary with train/val dataloaders
        config: Experiment configuration
        model_type: Type of model ("gnn", "mlp", "rf")
        gnn_type: GNN type if model_type is "gnn"
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        device: Device to use
        is_regression: Whether this is regression
        experiment_name: Name of the experiment for optuna storage
        run_id: ID of the current run for optuna storage
        
    Returns:
        Dictionary with optimization results
    """
    import optuna
    from optuna import create_study, Trial
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get sample batch to determine dimensions
    sample_batch = next(iter(dataloaders['train']))
    input_dim = sample_batch.x.shape[1]
    
    if is_regression:
        output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
    else:
        output_dim = len(torch.unique(sample_batch.y))
    
    def objective(trial: Trial) -> float:
        # Common hyperparameters
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
            
            # Create model
            model = model_creator(
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
                is_regression=is_regression
            ).to(device)
            
        elif model_type == "mlp":
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.7)
            
            # Create model
            model = model_creator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                is_regression=is_regression
            ).to(device)
        
        else:  # sklearn
            # Create sklearn model
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                is_regression=is_regression
            )
            
            # Train and evaluate sklearn model quickly
            return train_sklearn_inductive(model, dataloaders, config, is_regression)['test_metrics']['r2' if is_regression else 'f1_macro']
        
        # Train PyTorch model with reduced epochs for speed
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        
        # Quick training loop
        max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
        best_val_metric = float('-inf') if is_regression else 0.0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            for batch in dataloaders['train']:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                if isinstance(model, GNNModel):
                    out = model(batch.x, batch.edge_index)
                else:
                    out = model(batch.x)
                
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
                    
                    if isinstance(model, GNNModel):
                        out = model(batch.x, batch.edge_index)
                    else:
                        out = model(batch.x)
                    
                    if is_regression:
                        val_predictions.append(out.detach().cpu())
                    else:
                        val_predictions.append(out.argmax(dim=1).detach().cpu())
                    
                    val_targets.append(batch.y.detach().cpu())
            
            # Calculate validation metric
            val_pred = torch.cat(val_predictions, dim=0).numpy()
            val_true = torch.cat(val_targets, dim=0).numpy()
            val_metrics = compute_metrics(val_true, val_pred, is_regression)
            
            if is_regression:
                val_metric = val_metrics['r2']
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                val_metric = val_metrics['f1_macro']
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return best_val_metric
    
    # Create study with storage if experiment info provided
    study_name = f"{model_type}_{gnn_type if gnn_type else ''}_{'regression' if is_regression else 'classification'}"
    if experiment_name and run_id is not None:
        storage_path = get_optuna_storage_path(experiment_name, run_id)
        study = create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            load_if_exists=True
        )
        print(f"Using optuna storage: {storage_path}")
    else:
        study = create_study(direction='maximize')
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': study_name
    }


def train_and_evaluate_inductive(
    model: Union[GNNModel, MLPModel, SklearnModel],
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    task: str,
    device: torch.device,
    optimize_hyperparams: bool = False,
    experiment_name: Optional[str] = None,
    run_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline for inductive learning.
    
    Args:
        model: Model to train
        dataloaders: Data loaders for train/val/test
        config: Experiment configuration
        task: Task name
        device: Device to use
        optimize_hyperparams: Whether to optimize hyperparameters (overrides config setting)
        experiment_name: Name of the experiment for optuna storage
        run_id: ID of the current run for optuna storage
        
    Returns:
        Complete results dictionary
    """
    is_regression = config.is_regression.get(task, False)
    
    # Use config setting if optimize_hyperparams not explicitly set
    if optimize_hyperparams is None:
        optimize_hyperparams = config.optimize_hyperparams
    
    # Hyperparameter optimization if requested
    hyperopt_results = None
    if optimize_hyperparams and not isinstance(model, SklearnModel):
        print(f"Optimizing hyperparameters for inductive learning (n_trials={config.n_trials}, timeout={config.optimization_timeout}s)...")
        
        # Get model creator function
        if isinstance(model, GNNModel):
            model_creator = lambda **kwargs: GNNModel(**kwargs)
            model_type = "gnn"
            gnn_type = model.gnn_type
        else:  # MLPModel
            model_creator = lambda **kwargs: MLPModel(**kwargs)
            model_type = "mlp"
            gnn_type = None
        
        hyperopt_results = optimize_inductive_hyperparameters(
            model_creator=model_creator,
            dataloaders=dataloaders,
            config=config,
            model_type=model_type,
            gnn_type=gnn_type,
            n_trials=config.n_trials,
            timeout=config.optimization_timeout,
            device=device,
            is_regression=is_regression,
            experiment_name=experiment_name,
            run_id=run_id
        )
        
        # Update model with best parameters
        if hyperopt_results and 'best_params' in hyperopt_results:
            best_params = hyperopt_results['best_params']
            print(f"Using optimized parameters: {best_params}")
            
            # Recreate model with best parameters
            sample_batch = next(iter(dataloaders['train']))
            input_dim = sample_batch.x.shape[1]
            
            if is_regression:
                output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
            else:
                output_dim = get_total_classes_from_dataloaders(dataloaders)
            
            if isinstance(model, GNNModel):
                # Handle GAT-specific parameters
                if gnn_type == "gat":
                    heads = best_params.get('heads', 1)
                    concat_heads = best_params.get('concat_heads', True)
                    # For GAT, adjust hidden_dim based on concat_heads
                    effective_hidden_dim = best_params.get('hidden_dim', config.hidden_dim) * heads if concat_heads else best_params.get('hidden_dim', config.hidden_dim)
                else:
                    heads = 1
                    concat_heads = True
                    effective_hidden_dim = best_params.get('hidden_dim', config.hidden_dim)
                
                model = GNNModel(
                    input_dim=input_dim,
                    hidden_dim=effective_hidden_dim,
                    output_dim=output_dim,
                    num_layers=best_params.get('num_layers', config.num_layers),
                    dropout=best_params.get('dropout', config.dropout),
                    gnn_type=gnn_type,
                    residual=best_params.get('residual', False),
                    norm_type=best_params.get('norm_type', 'none'),
                    agg_type=best_params.get('agg_type', 'mean'),
                    heads=heads,
                    concat_heads=concat_heads,
                    is_regression=is_regression
                )
            else:  # MLPModel
                model = MLPModel(
                    input_dim=input_dim,
                    hidden_dim=best_params.get('hidden_dim', config.hidden_dim),
                    output_dim=output_dim,
                    num_layers=best_params.get('num_layers', config.num_layers),
                    dropout=best_params.get('dropout', config.dropout),
                    is_regression=is_regression
                )
            
            # Update learning rate and weight decay
            config.learning_rate = best_params.get('lr', config.learning_rate)
            config.weight_decay = best_params.get('weight_decay', config.weight_decay)
            config.patience = best_params.get('patience', config.patience)
    
    # Train the model
    results = train_inductive_model(model, dataloaders, config, task, device)
    
    # Add hyperopt results if available
    if hyperopt_results:
        results['hyperopt_results'] = hyperopt_results
    
    return results