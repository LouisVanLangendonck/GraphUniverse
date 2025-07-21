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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from experiments.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel
from experiments.metrics import compute_metrics
from experiments.transductive.config import TransductiveExperimentConfig
from torch.cuda.amp import GradScaler, autocast

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
    
    for epoch in range(config.epochs):
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


def optimize_transductive_hyperparameters(
    model_creator: Callable,
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    model_type: str = "gnn",
    n_trials: int = 20,
    timeout: Optional[int] = 600,
    device: Optional[torch.device] = None,
    is_regression: bool = False
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for transductive learning using validation nodes.
    Model-type-specific, matching inductive code.
    """
    import optuna
    from optuna import create_study, Trial

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = task_data['features'].shape[1]
    output_dim = task_data['metadata']['output_dim']

    def gnn_objective(trial: Trial) -> float:
        # Common GNN hyperparameters
        lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = config.patience
        # GNN-specific
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        pe_type = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
        pe_dim = trial.suggest_categorical('pe_dim', [8, 16, 32])
        heads = 1
        concat_heads = True
        eps = 0.3
        residual = False
        norm_type = 'none'
        agg_type = 'sum'
        # Model-specific
        if model_type == 'gat':
            heads = trial.suggest_int('heads', 1, 8)
            concat_heads = trial.suggest_categorical('concat_heads', [True, False])
            hidden_dim = trial.suggest_int('hidden_dim', 8, 32)
            if hidden_dim % heads != 0:
                hidden_dim = hidden_dim // heads * heads
        elif model_type == 'fagcn':
            eps = trial.suggest_float('eps', 0.0, 1.0)
        elif model_type == 'gin':
            eps = trial.suggest_float('eps', 0.0, 1.0)
        # Build model
        model = model_creator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=model_type,
            heads=heads,
            concat_heads=concat_heads,
            eps=eps,
            residual=residual,
            norm_type=norm_type,
            agg_type=agg_type,
            is_regression=is_regression,
            pe_type=pe_type,
            pe_dim=pe_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.L1Loss() if is_regression and config.regression_loss == 'mae' else (
            torch.nn.MSELoss() if is_regression else torch.nn.CrossEntropyLoss())
        features = task_data['features'].to(device)
        labels = task_data['labels'].to(device)
        train_idx = task_data['train_idx'].to(device)
        val_idx = task_data['val_idx'].to(device)
        edge_index = task_data['edge_index'].to(device)
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        max_epochs = min(50, config.epochs // 4)
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(features, edge_index)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(features, edge_index)
                if is_regression:
                    val_pred = out[val_idx].cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    val_metric = np.mean(np.abs(val_pred - val_true)) if config.regression_loss == 'mae' else np.mean((val_pred - val_true) ** 2)
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    acc = np.mean(val_pred == val_true)
                    if acc > best_val_metric:
                        best_val_metric = acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
            if patience_counter >= 5:
                break
        return -best_val_metric if is_regression else best_val_metric

    def mlp_objective(trial: Trial) -> float:
        lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = config.patience
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        pe_type = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
        pe_dim = trial.suggest_categorical('pe_dim', [8, 16, 32])
        model = model_creator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            is_regression=is_regression,
            pe_type=pe_type,
            pe_dim=pe_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.L1Loss() if is_regression and config.regression_loss == 'mae' else (
            torch.nn.MSELoss() if is_regression else torch.nn.CrossEntropyLoss())
        features = task_data['features'].to(device)
        labels = task_data['labels'].to(device)
        train_idx = task_data['train_idx'].to(device)
        val_idx = task_data['val_idx'].to(device)
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        max_epochs = min(50, config.epochs // 4)
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(features)
                if is_regression:
                    val_pred = out[val_idx].cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    val_metric = np.mean(np.abs(val_pred - val_true)) if config.regression_loss == 'mae' else np.mean((val_pred - val_true) ** 2)
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    acc = np.mean(val_pred == val_true)
                    if acc > best_val_metric:
                        best_val_metric = acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
            if patience_counter >= 5:
                break
        return -best_val_metric if is_regression else best_val_metric

    def sklearn_objective(trial: Trial) -> float:
        if model_type == "rf":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                model_type="rf",
                is_regression=is_regression,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
        elif model_type == "svm":
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                model_type="svm",
                is_regression=is_regression,
                C=C,
                kernel=kernel,
                gamma=gamma
            )
        elif model_type == "knn":
            n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                model_type="knn",
                is_regression=is_regression,
                n_neighbors=n_neighbors,
                weights=weights
            )
        else:
            model = model_creator(
                input_dim=input_dim,
                output_dim=output_dim,
                model_type=model_type,
                is_regression=is_regression
            )
        results = train_sklearn_transductive(model, task_data, config, is_regression)
        if is_regression:
            return -results['test_metrics']['mae']
        else:
            return results['test_metrics']['f1_macro']

    def transformer_objective(trial: Trial) -> float:
        lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = config.patience
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        base_dim = trial.suggest_int('base_dim', 4, 32)
        hidden_dim = base_dim * num_heads
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        local_gnn_type = trial.suggest_categorical('local_gnn_type', ['gcn', 'sage'])
        pe_type = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
        attn_type = trial.suggest_categorical('attn_type', ['performer', 'multihead'])
        model = model_creator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            transformer_type=model_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            is_regression=is_regression,
            local_gnn_type=local_gnn_type,
            attn_type=attn_type,
            pe_dim=config.max_pe_dim,
            pe_type=pe_type,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.L1Loss() if is_regression and config.regression_loss == 'mae' else (
            torch.nn.MSELoss() if is_regression else torch.nn.CrossEntropyLoss())
        features = task_data['features'].to(device)
        labels = task_data['labels'].to(device)
        train_idx = task_data['train_idx'].to(device)
        val_idx = task_data['val_idx'].to(device)
        edge_index = task_data['edge_index'].to(device)
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        max_epochs = min(50, config.epochs // 4)
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(features, edge_index)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(features, edge_index)
                if is_regression:
                    val_pred = out[val_idx].cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    val_metric = np.mean(np.abs(val_pred - val_true)) if config.regression_loss == 'mae' else np.mean((val_pred - val_true) ** 2)
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                    acc = np.mean(val_pred == val_true)
                    if acc > best_val_metric:
                        best_val_metric = acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
            if patience_counter >= 5:
                break
        return -best_val_metric if is_regression else best_val_metric

    # Dispatch to correct objective
    if model_type in ['gcn', 'gat', 'sage', 'gin', 'fagcn']:
        objective = gnn_objective
    elif model_type == 'mlp':
        objective = mlp_objective
    elif model_type in ['rf', 'svm', 'knn']:
        objective = sklearn_objective
    elif model_type in ['graphgps', 'graphormer']:
        objective = transformer_objective
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    study = create_study(direction='maximize' if not is_regression else 'minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': model_type
    }


def train_and_evaluate_transductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel, SheafDiffusionModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    optimize_hyperparams: bool = False,
    model_name: str = 'gcn',
    hyperparams: dict = None
) -> Dict[str, Any]:
    """Train and evaluate a model on a single split. Optionally perform hyperparameter optimization."""
    import copy
    import time
    is_regression = task != 'community'
    is_graph_level_task = False
    best_hyperparams = hyperparams if hyperparams is not None else {}
    # If hyperparam optimization is requested
    if optimize_hyperparams:
        # Define Optuna search space similar to inductive
        import optuna
        from optuna import create_study, Trial
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            params = {}
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            params['hidden_dim'] = trial.suggest_int('hidden_dim', 32, 128, step=32)
            params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
            params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
            params['pe_type'] = trial.suggest_categorical('pe_type', [None, 'laplacian', 'degree', 'rwse'])
            params['pe_dim'] = trial.suggest_categorical('pe_dim', [8, 16, 32])
            # Model-specific
            if model_name in ['gat']:
                params['heads'] = trial.suggest_int('heads', 1, 8)
                params['concat_heads'] = trial.suggest_categorical('concat_heads', [True, False])
            if model_name in ['fagcn', 'gin']:
                params['eps'] = trial.suggest_float('eps', 0.0, 1.0)
            # Instantiate model with these params
            ModelClass = type(model)
            model_args = dict(
                input_dim=task_data['features'].shape[1],
                hidden_dim=params['hidden_dim'],
                output_dim=task_data['metadata']['output_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                gnn_type=model_name if hasattr(model, 'gnn_type') else None,
                is_regression=is_regression,
                is_graph_level_task=is_graph_level_task,
                pe_type=params['pe_type'],
                pe_dim=params['pe_dim'],
            )
            if 'heads' in params:
                model_args['heads'] = params['heads']
            if 'concat_heads' in params:
                model_args['concat_heads'] = params['concat_heads']
            if 'eps' in params:
                model_args['eps'] = params['eps']
            trial_model = ModelClass(**{k: v for k, v in model_args.items() if v is not None}).to(device)
            optimizer = torch.optim.Adam(trial_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            # Loss
            if is_regression:
                criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
            else:
                criterion = torch.nn.CrossEntropyLoss()
            # Training loop (short, e.g. 30 epochs)
            max_epochs = 100
            best_val_metric = float('inf') if is_regression else 0.0
            patience_counter = 0
            for epoch in range(max_epochs):
                trial_model.train()
                optimizer.zero_grad()
                out = trial_model(task_data['features'].to(device), task_data['edge_index'].to(device), graph=task_data['pyg_graph'])
                loss = criterion(out[task_data['train_idx']], task_data['labels'][task_data['train_idx']].to(device))
                loss.backward()
                optimizer.step()
                # Validation
                trial_model.eval()
                with torch.no_grad():
                    out = trial_model(task_data['features'].to(device), task_data['edge_index'].to(device), graph=task_data['pyg_graph'])
                    if is_regression:
                        val_pred = out[task_data['val_idx']].cpu().numpy()
                        val_true = task_data['labels'][task_data['val_idx']].cpu().numpy()
                        val_metric = np.mean(np.abs(val_pred - val_true)) if config.regression_loss == 'mae' else np.mean((val_pred - val_true) ** 2)
                        if val_metric < best_val_metric:
                            best_val_metric = val_metric
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    else:
                        val_pred = out[task_data['val_idx']].argmax(dim=1).cpu().numpy()
                        val_true = task_data['labels'][task_data['val_idx']].cpu().numpy()
                        acc = np.mean(val_pred == val_true)
                        if acc > best_val_metric:
                            best_val_metric = acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                if patience_counter >= 5:
                    break
            # For regression, minimize; for classification, maximize
            return -best_val_metric if is_regression else best_val_metric
        study = create_study(direction='minimize' if is_regression else 'maximize')
        study.optimize(objective, n_trials=20, timeout=600)
        best_hyperparams = study.best_params
    # Instantiate model with best hyperparams
    ModelClass = type(model)
    model_args = dict(
        input_dim=task_data['features'].shape[1],
        hidden_dim=best_hyperparams.get('hidden_dim', getattr(model, 'hidden_dim', 64)),
        output_dim=task_data['metadata']['output_dim'],
        num_layers=best_hyperparams.get('num_layers', getattr(model, 'num_layers', 2)),
        dropout=best_hyperparams.get('dropout', getattr(model, 'dropout', 0.5)),
        gnn_type=model_name if hasattr(model, 'gnn_type') else None,
        is_regression=is_regression,
        is_graph_level_task=is_graph_level_task,
        pe_type=best_hyperparams.get('pe_type', getattr(model, 'pe_type', 'laplacian')),
        pe_dim=best_hyperparams.get('pe_dim', getattr(model, 'pe_dim', 16)),
    )
    if 'heads' in best_hyperparams:
        model_args['heads'] = best_hyperparams['heads']
    if 'concat_heads' in best_hyperparams:
        model_args['concat_heads'] = best_hyperparams['concat_heads']
    if 'eps' in best_hyperparams:
        model_args['eps'] = best_hyperparams['eps']
    model = ModelClass(**{k: v for k, v in model_args.items() if v is not None}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams.get('learning_rate', 0.01), weight_decay=best_hyperparams.get('weight_decay', 5e-4))
    # Loss
    if is_regression:
        criterion = torch.nn.MSELoss() if config.regression_loss == 'mse' else torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # Training loop (full epochs)
    best_val_metric = float('inf') if is_regression else 0.0
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(task_data['features'].to(device), task_data['edge_index'].to(device), graph=task_data['pyg_graph'])
        loss = criterion(out[task_data['train_idx']], task_data['labels'][task_data['train_idx']].to(device))
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(task_data['features'].to(device), task_data['edge_index'].to(device), graph=task_data['pyg_graph'])
            if is_regression:
                val_pred = out[task_data['val_idx']].cpu().numpy()
                val_true = task_data['labels'][task_data['val_idx']].cpu().numpy()
                val_metric = np.mean(np.abs(val_pred - val_true)) if config.regression_loss == 'mae' else np.mean((val_pred - val_true) ** 2)
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                val_pred = out[task_data['val_idx']].argmax(dim=1).cpu().numpy()
                val_true = task_data['labels'][task_data['val_idx']].cpu().numpy()
                acc = np.mean(val_pred == val_true)
                if acc > best_val_metric:
                    best_val_metric = acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
        if patience_counter >= 10:
            break
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Test evaluation
    model.eval()
    with torch.no_grad():
        out = model(task_data['features'].to(device), task_data['edge_index'].to(device), graph=task_data['pyg_graph'])
        if is_regression:
            test_pred = out[task_data['test_idx']].cpu().numpy()
            test_true = task_data['labels'][task_data['test_idx']].cpu().numpy()
            test_metric = np.mean(np.abs(test_pred - test_true)) if config.regression_loss == 'mae' else np.mean((test_pred - test_true) ** 2)
        else:
            test_pred = out[task_data['test_idx']].argmax(dim=1).cpu().numpy()
            test_true = task_data['labels'][task_data['test_idx']].cpu().numpy()
            test_metric = np.mean(test_pred == test_true)
    train_time = time.time() - start_time
    print(f"Test metric: {test_metric}")
    input('Press Enter to continue...')
    return {
        'test_metrics': {'mae' if is_regression else 'accuracy': test_metric},
        'train_time': train_time,
        'optimal_hyperparams': best_hyperparams if optimize_hyperparams else {}
    }