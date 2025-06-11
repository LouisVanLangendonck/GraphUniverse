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

from experiments.core.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel
from experiments.core.metrics import compute_metrics
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
        return output_dim
    
    # Fallback: Get from first batch if metadata not available
    sample_batch = next(iter(dataloaders['train']))
    if hasattr(sample_batch, 'universe_K'):
        output_dim = sample_batch.universe_K
        print(f"Using universe K from batch: {output_dim}")
        return output_dim
    
    # Last resort: Try to infer from labels
    all_labels = set()
    for split_name, dataloader in dataloaders.items():
        if split_name == 'metadata':
            continue
        for batch in dataloader:
            labels = batch.y
            all_labels.update(labels.cpu().numpy().tolist())
    
    output_dim = max(all_labels) + 1
    print(f"Warning: Inferring output dimension from labels: {output_dim}")
    return output_dim

def train_inductive_model(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel],
    dataloaders: Dict[str, DataLoader],
    config: InductiveExperimentConfig,
    task: str,
    device: torch.device,
    finetuning: Optional[bool] = False
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
    
    # Get the effective patience value - take max of original and optimized if available
    effective_patience = config.patience
    if hasattr(config, 'optimized_patience') and config.optimized_patience is not None:
        effective_patience = min(30, max(config.patience, config.optimized_patience))
        print(f"Using effective patience of {effective_patience} (max of original {config.patience} and optimized {config.optimized_patience})")
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    start_time = time.time()
    
    print(f"\nStarting training with patience={effective_patience}")
    print("Early stopping will trigger if validation metric doesn't improve for", effective_patience, "epochs")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass - check if model requires edge_index
            if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type') or finetuning:
                out = model(batch.x, batch.edge_index)
            else:  # MLPModel or other non-GNN model
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
                
                if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type') or finetuning:
                    out = model(batch.x, batch.edge_index)
                else:
                    out = model(batch.x)
                
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                
                if is_regression:
                    val_predictions.append(out.detach().cpu())
                else:
                    val_predictions.append(out.argmax(dim=1).detach().cpu())
                
                val_targets.append(batch.y.detach().cpu())
        
        # Calculate metrics
        train_pred = torch.cat(train_predictions, dim=0).numpy()
        train_true = torch.cat(train_targets, dim=0).numpy()
        val_pred = torch.cat(val_predictions, dim=0).numpy()
        val_true = torch.cat(val_targets, dim=0).numpy()
        
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
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['val_loss'].append(val_loss / len(val_loader))
        training_history['train_metric'].append(train_metric)
        training_history['val_metric'].append(val_metric)
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}")
        
        # Model selection
        improved = False
        if is_regression:
            if config.regression_loss == 'mae':
                # For MAE, lower is better
                if val_metric < best_val_metric:
                    improved = True
            elif config.regression_loss == 'r2':
                # For R², higher is better
                if val_metric > best_val_metric:
                    improved = True
            else:
                raise ValueError(f"Invalid regression loss: {config.regression_loss}")
        else:
            if val_metric > best_val_metric:  # Higher F1 is better
                improved = True
        
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
    test_metrics = evaluate_inductive_model(model, dataloaders['test'], is_regression, device, finetuning)
    
    return {
        'train_time': train_time,
        'best_val_metric': best_val_metric,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'model': model,
        'effective_patience': effective_patience
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

def evaluate_inductive_model(
    model: Union[GNNModel, MLPModel],
    test_loader: DataLoader,
    is_regression: bool,
    device: torch.device,
    finetuning: Optional[bool] = False
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
            
            # Forward pass - check if model requires edge_index
            if hasattr(model, 'gnn_type') or hasattr(model, 'transformer_type') or finetuning:
                out = model(batch.x, batch.edge_index)
            else:  # MLPModel or other non-GNN model
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
    transformer_type: Optional[str] = None,  # NEW
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
    
    output_dim = config.universe_K
    
    def objective(trial: Trial) -> float:
        # Common hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = trial.suggest_int('patience', 10, 50)
        
        # Model-specific hyperparameters
        if model_type == "transformer":  # NEW TRANSFORMER OPTIMIZATION
            # First suggest number of heads (must be a power of 2 for efficiency)
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
            
            # Then suggest hidden_dim that is divisible by num_heads
            # We'll suggest a base dimension and multiply by num_heads
            base_dim = trial.suggest_int('base_dim', 8, 32)
            hidden_dim = base_dim * num_heads
            
            num_layers = trial.suggest_int('num_layers', 2, 6)
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            
            # Transformer-specific parameters
            if transformer_type == "graphormer":
                max_path_length = trial.suggest_int('max_path_length', 5, 15)
                precompute_encodings = trial.suggest_categorical('precompute_encodings', [True, False])
            elif transformer_type == "graphgps":
                local_gnn_type = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
                prenorm = trial.suggest_categorical('prenorm', [True, False])
            
            # Create transformer model
            from experiments.core.models import GraphTransformerModel
            model = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                transformer_type=transformer_type,
                num_layers=num_layers,
                dropout=dropout,
                is_regression=is_regression,
                num_heads=num_heads,
                max_path_length=max_path_length if transformer_type == "graphormer" else 10,
                precompute_encodings=precompute_encodings if transformer_type == "graphormer" else True,
                local_gnn_type=local_gnn_type if transformer_type == "graphgps" else "gcn",
                prenorm=prenorm if transformer_type == "graphgps" else True
            ).to(device)
        
        # Model-specific hyperparameters
        elif model_type == "gnn":
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.7)
            
            # Initialize default values
            heads = 1
            concat_heads = True
            eps = 0.3
            residual = False
            norm_type = 'none'
            agg_type = 'mean'
            
            # GNN-specific parameters
            if gnn_type == "gat":
                heads = trial.suggest_int('heads', 1, 8)
                concat_heads = trial.suggest_categorical('concat_heads', [True, False])
                residual = trial.suggest_categorical('residual', [True, False])
                norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
            elif gnn_type == "fagcn":
                # FAGCN-specific parameters
                eps = trial.suggest_float('eps', 0.0, 1.0)
                # Suggest more layers for FAGCN since it can handle more layers
                num_layers = trial.suggest_int('num_layers', 2, 6)
                # Don't optimize irrelevant parameters for FAGCN
            elif gnn_type == "gin":
                # GIN-specific parameters
                eps = trial.suggest_float('eps', 0.0, 1.0)
                residual = trial.suggest_categorical('residual', [True, False])
                norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])

                

            elif gnn_type in ["gcn", "sage", "gat", "gin"]:
                residual = trial.suggest_categorical('residual', [True, False])
                norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                if gnn_type == "sage":
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
                eps=eps,
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

        elif model_type == "rf":
            # Create RF model
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                is_regression=is_regression
            )
        
        else:  # sklearn
            # Create sklearn model
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                is_regression=is_regression
            )
            
            # Train and evaluate sklearn model quickly
            results = train_sklearn_inductive(model, dataloaders, config, is_regression)
            if is_regression:
                return -results['test_metrics']['mae']  # Negative because we minimize MAE
            else:
                return results['test_metrics']['f1_macro']
        
        # Train PyTorch model with reduced epochs for speed
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Set up loss function based on regression type
        if is_regression:
            if config.regression_loss == 'mae':
                criterion = torch.nn.L1Loss()
            else:
                criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Quick training loop
        max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
        best_val_metric = float('inf') if is_regression else 0.0  # Initialize based on task
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            for batch in dataloaders['train']:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass - works for all model types
                if hasattr(model, 'transformer_type'):
                    out = model(batch.x, batch.edge_index)
                elif hasattr(model, 'gnn_type'):
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
                    
                    if hasattr(model, 'transformer_type'):
                        out = model(batch.x, batch.edge_index)
                    elif hasattr(model, 'gnn_type'):
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
                if config.regression_loss == 'mae':
                    val_metric = val_metrics['mae']
                    if val_metric < best_val_metric:  # Lower is better for MAE
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
                        patience_counter += 1
            else:
                val_metric = val_metrics['f1_macro']
                if val_metric > best_val_metric:  # Higher is better for F1
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Return negative MAE for minimization or positive R²/F1 for maximization
        if is_regression:
            if config.regression_loss == 'mae':
                return -best_val_metric  # Negative because we minimize MAE
            else:
                return best_val_metric  # Positive because we maximize R²
        else:
            return best_val_metric  # Positive because we maximize F1
        
    # Create study with early termination
    def should_terminate(study, trial):
        """Check if we should terminate optimization early."""
        if is_regression:
            if config.regression_loss == 'mae':
                # For MAE, perfect score is 0, but we return negative values
                return trial.value >= -1e-6  # Essentially 0 MAE
            else:
                # For R², perfect score is 1.0
                return trial.value >= 0.999
        else:
            # For classification metrics (F1, accuracy), perfect score is 1.0
            return trial.value >= 0.999
    
    # Create study with storage if experiment info provided
    study_name = f"{model_type}_{transformer_type if transformer_type else gnn_type}_{'regression' if is_regression else 'classification'}"
    study = create_study(direction='maximize')
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[lambda study, trial: study.stop() if should_terminate(study, trial) else None])
    
    # study_name = f"{model_type}_{gnn_type if gnn_type else ''}_{'regression' if is_regression else 'classification'}"
    # if experiment_name and run_id is not None:
    #     storage_path = get_optuna_storage_path(experiment_name, run_id)
    #     study = create_study(
    #         study_name=study_name,
    #         storage=storage_path,
    #         direction='maximize',  # We always maximize because we negate MAE
    #         load_if_exists=True
    #     )
    #     print(f"Using optuna storage: {storage_path}")
    # else:
    #     study = create_study(direction='maximize')  # We always maximize because we negate MAE
    
    return {
        'best_params': study.best_params,
        'best_value': -study.best_value if is_regression and config.regression_loss == 'mae' else study.best_value,  # Convert back from negative MAE
        'n_trials': len(study.trials),
        'study_name': study_name
    }

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
        patience = trial.suggest_int('patience', 5, 25)
        
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
                        val_predictions.append(out.detach().cpu())
                    else:
                        val_predictions.append(out.argmax(dim=1).detach().cpu())
                    
                    val_targets.append(batch.y.detach().cpu())
            
            # Calculate validation metric
            val_pred = torch.cat(val_predictions, dim=0).numpy()
            val_true = torch.cat(val_targets, dim=0).numpy()
            
            from experiments.core.metrics import compute_metrics
            val_metrics = compute_metrics(val_true, val_pred, is_regression)
            
            if is_regression:
                val_metric = val_metrics['mse']
                if val_metric < best_val_metric:  # Lower is better for MSE
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
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
            return -best_val_metric  # Negative because we minimize MSE
        else:
            return best_val_metric  # Positive because we maximize F1
    
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

def train_and_evaluate_inductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel],  # Updated type hint
    dataloaders: Dict[str, DataLoader],
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
        if hasattr(model, 'transformer_type'):
            print(f"🔄 Training Graph Transformer: {model.transformer_type}")
            
            # Share precomputed cache if available
            if hasattr(config, 'transformer_caches') and model.transformer_type in config.transformer_caches:
                model._encoding_cache = config.transformer_caches[model.transformer_type]
                print(f"✅ Using precomputed encodings cache with {len(model._encoding_cache)} entries")
            
            # Update hyperparameter optimization for transformers
            if optimize_hyperparams and not isinstance(model, SklearnModel):
                print(f"🎯 Optimizing Graph Transformer hyperparameters...")
                
                # Get model creator function for transformers
                model_creator = lambda **kwargs: GraphTransformerModel(**kwargs)
                
                # Get sample batch to determine dimensions
                sample_batch = next(iter(dataloaders['train']))
                input_dim = sample_batch.x.shape[1]
                
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
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    patience = trial.suggest_int('patience', 10, 50)
                    
                    # Transformer-specific hyperparameters
                    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
                    base_dim = trial.suggest_int('base_dim', 8, 32)
                    hidden_dim = base_dim * num_heads
                    num_layers = trial.suggest_int('num_layers', 2, 6)
                    dropout = trial.suggest_float('dropout', 0.0, 0.3)
                    
                    # Transformer-type specific parameters
                    if transformer_type == "graphormer":
                        max_path_length = trial.suggest_int('max_path_length', 5, 15)
                        precompute_encodings = trial.suggest_categorical('precompute_encodings', [True, False])
                        local_gnn_type = "gcn"  # Default for GraphFormer
                        prenorm = True  # Default for GraphFormer
                    elif transformer_type == "graphgps":
                        max_path_length = 10  # Default for GraphGPS
                        precompute_encodings = True  # Default for GraphGPS
                        local_gnn_type = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
                        prenorm = trial.suggest_categorical('prenorm', [True, False])
                    
                    # Create transformer model with all parameters
                    trial_model = GraphTransformerModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        transformer_type=transformer_type,
                        num_layers=num_layers,
                        dropout=dropout,
                        is_regression=is_regression,
                        num_heads=num_heads,
                        max_path_length=max_path_length,
                        precompute_encodings=precompute_encodings,
                        cache_encodings=config.transformer_cache_encodings,
                        local_gnn_type=local_gnn_type,
                        prenorm=prenorm
                    ).to(device)
                    
                    # Restore cache if available
                    if hasattr(config, 'transformer_caches') and transformer_type in config.transformer_caches:
                        trial_model._encoding_cache = config.transformer_caches[transformer_type]
                    
                    # Train model with reduced epochs for speed
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0
                    
                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in dataloaders['train']:
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
                            for batch in dataloaders['val']:
                                batch = batch.to(device)
                                out = trial_model(batch.x, batch.edge_index)
                                
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
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
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
                                    patience_counter += 1
                        else:
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
                        if config.regression_loss == 'mae':
                            return -best_val_metric  # Negative because we minimize MAE
                        else:
                            return best_val_metric  # Positive because we maximize R²
                    else:
                        return best_val_metric  # Positive because we maximize F1
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    base_dim = best_params.get('base_dim', 8)
                    num_heads = best_params.get('num_heads', 8)
                    hidden_dim = base_dim * num_heads
                    
                    model = GraphTransformerModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        transformer_type=transformer_type,
                        num_layers=best_params.get('num_layers', 2),
                        dropout=best_params.get('dropout', 0.1),
                        is_regression=is_regression,
                        num_heads=num_heads,
                        max_path_length=best_params.get('max_path_length', 10),
                        precompute_encodings=best_params.get('precompute_encodings', True),
                        cache_encodings=config.transformer_cache_encodings,
                        local_gnn_type=best_params.get('local_gnn_type', 'gcn'),
                        prenorm=best_params.get('prenorm', True)
                    ).to(device)
                    
                    # Restore cache if available
                    if hasattr(config, 'transformer_caches') and transformer_type in config.transformer_caches:
                        model._encoding_cache = config.transformer_caches[transformer_type]
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
                    config.patience = best_params.get('patience', config.patience)
                    config.optimized_patience = best_params.get('patience', config.patience)
        
        # Handle GNN models
        elif hasattr(model, 'gnn_type'):
            print(f"🔄 Training GNN: {model.gnn_type}")
            
            # Update hyperparameter optimization for GNNs
            if optimize_hyperparams and not isinstance(model, SklearnModel):
                print(f"🎯 Optimizing GNN hyperparameters...")
                
                # Get model creator function for GNNs
                model_creator = lambda **kwargs: GNNModel(**kwargs)
                
                # Get sample batch to determine dimensions
                sample_batch = next(iter(dataloaders['train']))
                input_dim = sample_batch.x.shape[1]
                
                if is_regression:
                    output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
                else:
                    output_dim = get_total_classes_from_dataloaders(dataloaders)
                
                # Create a single optimization study
                import optuna
                from optuna import create_study, Trial
                
                study_name = f"gnn_{model.gnn_type}_{'regression' if is_regression else 'classification'}"
                study = create_study(direction='maximize')
                
                # Store GNN type for the objective function
                gnn_type = model.gnn_type
                
                def objective(trial: Trial) -> float:
                    # Common hyperparameters
                    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                    patience = trial.suggest_int('patience', 10, 50)
                    
                    # GNN-specific hyperparameters
                    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
                    num_layers = trial.suggest_int('num_layers', 1, 4)
                    dropout = trial.suggest_float('dropout', 0.1, 0.7)
                    
                    # Initialize default values
                    heads = 1
                    concat_heads = True
                    eps = 0.3
                    residual = False
                    norm_type = 'none'
                    agg_type = 'mean'
                    
                    # GNN-specific parameters
                    if gnn_type == "gat":
                        heads = trial.suggest_int('heads', 1, 8)
                        concat_heads = trial.suggest_categorical('concat_heads', [True, False])
                        residual = trial.suggest_categorical('residual', [True, False])
                        norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                        agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
                    elif gnn_type in ["fagcn", "gin"]:
                        eps = trial.suggest_float('eps', 0.0, 1.0)
                    elif gnn_type in ["gcn", "sage", "gat", "gin"]:
                        residual = trial.suggest_categorical('residual', [True, False])
                        norm_type = trial.suggest_categorical('norm_type', ['none', 'layer'])
                        if gnn_type in ["sage", "gcn"]:
                            agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
                    
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
                        is_regression=is_regression
                    ).to(device)
                    
                    # Train model with reduced epochs for speed
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0
                    
                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in dataloaders['train']:
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
                            for batch in dataloaders['val']:
                                batch = batch.to(device)
                                out = trial_model(batch.x, batch.edge_index)
                                
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
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
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
                                    patience_counter += 1
                        else:
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
                        if config.regression_loss == 'mae':
                            return -best_val_metric  # Negative because we minimize MAE
                        else:
                            return best_val_metric  # Positive because we maximize R²
                    else:
                        return best_val_metric  # Positive because we maximize F1
                
                # Run optimization
                study.optimize(objective, n_trials=config.n_trials, timeout=config.optimization_timeout)
                
                # Apply optimized parameters
                if study.best_params:
                    best_params = study.best_params
                    print(f"🎯 Applying optimized parameters:")
                    for key, value in best_params.items():
                        print(f"   {key}: {value}")
                    
                    # Recreate model with optimized parameters
                    model = GNNModel(
                        input_dim=input_dim,
                        hidden_dim=best_params.get('hidden_dim', 64),
                        output_dim=output_dim,
                        num_layers=best_params.get('num_layers', 2),
                        dropout=best_params.get('dropout', 0.5),
                        gnn_type=gnn_type,
                        residual=best_params.get('residual', False),
                        norm_type=best_params.get('norm_type', 'none'),
                        agg_type=best_params.get('agg_type', 'mean'),
                        heads=best_params.get('heads', 1),
                        concat_heads=best_params.get('concat_heads', True),
                        eps=best_params.get('eps', 0.3),
                        is_regression=is_regression
                    ).to(device)
                    
                    # Update config with optimized parameters
                    config.learning_rate = best_params.get('lr', config.learning_rate)
                    config.weight_decay = best_params.get('weight_decay', config.weight_decay)
                    config.patience = best_params.get('patience', config.patience)
                    config.optimized_patience = best_params.get('patience', config.patience)
                
        # Handle MLP models
        elif isinstance(model, MLPModel):
            print(f"🔄 Training MLP")
            
            # Update hyperparameter optimization for MLPs
            if optimize_hyperparams:
                print(f"🎯 Optimizing MLP hyperparameters...")
                
                # Get sample batch to determine dimensions
                sample_batch = next(iter(dataloaders['train']))
                input_dim = sample_batch.x.shape[1]
                
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
                    patience = trial.suggest_int('patience', 10, 50)
                    
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
                    
                    # Train model with reduced epochs for speed
                    optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                    # Set up loss function
                    if is_regression:
                        if config.regression_loss == 'mae':
                            criterion = torch.nn.L1Loss()
                        else:
                            criterion = torch.nn.MSELoss()
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                    
                    # Quick training loop
                    max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
                    best_val_metric = float('inf') if is_regression else 0.0
                    patience_counter = 0
                    
                    for epoch in range(max_epochs):
                        # Training
                        trial_model.train()
                        for batch in dataloaders['train']:
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
                            for batch in dataloaders['val']:
                                batch = batch.to(device)
                                out = trial_model(batch.x)
                                
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
                            if config.regression_loss == 'mae':
                                val_metric = val_metrics['mae']
                                if val_metric < best_val_metric:  # Lower is better for MAE
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
                                    patience_counter += 1
                        else:
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
                        if config.regression_loss == 'mae':
                            return -best_val_metric  # Negative because we minimize MAE
                        else:
                            return best_val_metric  # Positive because we maximize R²
                    else:
                        return best_val_metric  # Positive because we maximize F1
                
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
                    config.patience = best_params.get('patience', config.patience)
                    config.optimized_patience = best_params.get('patience', config.patience)
        
        # Handle sklearn models
        elif isinstance(model, SklearnModel):
            print(f"🔄 Training sklearn model: {model.model_type}")
            
            # Update hyperparameter optimization for sklearn models
            if optimize_hyperparams:
                print(f"🎯 Optimizing sklearn model hyperparameters...")
                
                # Get sample batch to determine dimensions
                sample_batch = next(iter(dataloaders['train']))
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
                    results = train_sklearn_inductive(trial_model, dataloaders, config, is_regression)
                    
                    # Return metric for optimization
                    if is_regression:
                        if config.regression_loss == 'mae':
                            return -results['test_metrics']['mae']  # Negative because we minimize MAE
                        else:
                            return results['test_metrics']['r2']  # Positive because we maximize R²
                    else:
                        return results['test_metrics']['f1_macro']  # Positive because we maximize F1
                
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