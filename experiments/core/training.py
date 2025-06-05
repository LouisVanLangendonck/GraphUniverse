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

from experiments.core.models import GNNModel, MLPModel, SklearnModel, GraphTransformerModel
from experiments.core.metrics import compute_metrics
from experiments.core.config import TransductiveExperimentConfig
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
    gnn_type: Optional[str] = None,
    transformer_type: Optional[str] = None,
    n_trials: int = 20,
    timeout: Optional[int] = 600,
    device: Optional[torch.device] = None,
    is_regression: bool = False
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for transductive learning using validation nodes.
    """
    import optuna
    from optuna import create_study, Trial
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract dimensions from task data
    input_dim = task_data['features'].shape[1]
    output_dim = task_data['metadata']['output_dim']
    
    def objective(trial: Trial) -> float:
        # Handle sklearn models differently
        if model_type in ["rf", "svm", "knn"]:
            # Create sklearn model with trial parameters
            if model_type == "rf":
                model = model_creator(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type="rf",
                    is_regression=is_regression,
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5)
                )
            elif model_type == "svm":
                model = model_creator(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type="svm",
                    is_regression=is_regression,
                    C=trial.suggest_float('C', 0.1, 10.0, log=True),
                    gamma=trial.suggest_categorical('gamma', ['scale', 'auto', 0.1, 0.01, 0.001])
                )
            elif model_type == "knn":
                model = model_creator(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type="knn",
                    is_regression=is_regression,
                    n_neighbors=trial.suggest_int('n_neighbors', 3, 20),
                    weights=trial.suggest_categorical('weights', ['uniform', 'distance'])
                )
            
            # Train and evaluate sklearn model
            results = train_sklearn_transductive(model, task_data, config, is_regression)
            if is_regression:
                return -results['test_metrics']['mae']  # Negative because we minimize MAE
            else:
                return results['test_metrics']['f1_macro']
        
        # For PyTorch models, continue with existing optimization logic
        # Common hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        patience = trial.suggest_int('patience', 10, 50)
        
        # Model-specific hyperparameters
        if model_type == "transformer":
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
                norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
                agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
            elif gnn_type == "fagcn":
                eps = trial.suggest_float('eps', 0.0, 1.0)
            elif gnn_type in ["gcn", "sage"]:
                residual = trial.suggest_categorical('residual', [True, False])
                norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
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
        
        # Move data to device
        features = task_data['features'].to(device)
        labels = task_data['labels'].to(device)
        train_idx = task_data['train_idx'].to(device)
        val_idx = task_data['val_idx'].to(device)
        
        # Handle edge_index for graph models
        edge_index = None
        if 'edge_index' in task_data:
            edge_index = task_data['edge_index'].to(device)
        
        # Quick training loop
        max_epochs = min(50, config.epochs // 4)  # Reduced epochs for hyperopt
        best_val_metric = float('inf') if is_regression else 0.0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'transformer_type') or hasattr(model, 'gnn_type'):
                if edge_index is None:
                    raise ValueError("Graph models require edge_index")
                out = model(features, edge_index)
            else:
                out = model(features)
            
            # Compute loss on training nodes
            loss = criterion(out[train_idx], labels[train_idx])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'transformer_type') or hasattr(model, 'gnn_type'):
                    out = model(features, edge_index)
                else:
                    out = model(features)
                
                # Compute validation metric
                if is_regression:
                    val_pred = out[val_idx].cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                else:
                    val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx].cpu().numpy()
                
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
    
    # Create study
    study_name = f"{model_type}_{transformer_type if transformer_type else gnn_type}_{'regression' if is_regression else 'classification'}"
    study = create_study(direction='maximize')
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    return {
        'best_params': study.best_params,
        'best_value': -study.best_value if is_regression and config.regression_loss == 'mae' else study.best_value,
        'n_trials': len(study.trials),
        'study_name': study_name
    }


def train_and_evaluate_transductive(
    model: Union[GNNModel, MLPModel, SklearnModel, GraphTransformerModel],
    task_data: Dict[str, Any],
    config: TransductiveExperimentConfig,
    task: str,
    device: torch.device,
    optimize_hyperparams: bool = False
) -> Dict[str, Any]:
    """Complete training and evaluation pipeline for transductive learning."""
    
    # Apply max_training_nodes limit if specified
    if config.max_training_nodes is not None:
        train_idx = task_data['train_idx']
        if len(train_idx) > config.max_training_nodes:
            # Randomly sample max_training_nodes from train_idx
            perm = torch.randperm(len(train_idx))
            task_data['train_idx'] = train_idx[perm[:config.max_training_nodes]]
            print(f"Limited training nodes to {config.max_training_nodes} (from {len(train_idx)})")
    
    is_regression = config.is_regression.get(task, False)
    print(f"🔄 Task: {task}, is_regression: {is_regression}")
    
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
            input_dim = task_data['features'].shape[1]
            output_dim = task_data['metadata']['output_dim']
            
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
                
                # Move data to device
                features = task_data['features'].to(device)
                labels = task_data['labels'].to(device)
                train_idx = task_data['train_idx'].to(device)
                val_idx = task_data['val_idx'].to(device)
                edge_index = task_data['edge_index'].to(device)
                
                for epoch in range(max_epochs):
                    # Training
                    trial_model.train()
                    optimizer.zero_grad()
                    out = trial_model(features, edge_index)
                    loss = criterion(out[train_idx], labels[train_idx])
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    trial_model.eval()
                    with torch.no_grad():
                        out = trial_model(features, edge_index)
                        
                        if is_regression:
                            val_pred = out[val_idx].cpu().numpy()
                            val_true = labels[val_idx].cpu().numpy()
                        else:
                            val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                            val_true = labels[val_idx].cpu().numpy()
                        
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
            else:
                best_params = {}
    
    # Handle GNN models
    elif hasattr(model, 'gnn_type'):
        print(f"🔄 Training GNN: {model.gnn_type}")
        
        # Update hyperparameter optimization for GNNs
        if optimize_hyperparams and not isinstance(model, SklearnModel):
            print(f"🎯 Optimizing GNN hyperparameters...")
            
            # Get model creator function for GNNs
            model_creator = lambda **kwargs: GNNModel(**kwargs)
            
            # Get sample batch to determine dimensions
            input_dim = task_data['features'].shape[1]
            output_dim = task_data['metadata']['output_dim']
            
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
                    norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
                    agg_type = trial.suggest_categorical('agg_type', ['mean', 'max', 'sum'])
                elif gnn_type == "fagcn":
                    eps = trial.suggest_float('eps', 0.0, 1.0)
                elif gnn_type in ["gcn", "sage"]:
                    residual = trial.suggest_categorical('residual', [True, False])
                    norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
                    if gnn_type == "sage":
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
                
                # Move data to device
                features = task_data['features'].to(device)
                labels = task_data['labels'].to(device)
                train_idx = task_data['train_idx'].to(device)
                val_idx = task_data['val_idx'].to(device)
                edge_index = task_data['edge_index'].to(device)
                
                for epoch in range(max_epochs):
                    # Training
                    trial_model.train()
                    optimizer.zero_grad()
                    out = trial_model(features, edge_index)
                    loss = criterion(out[train_idx], labels[train_idx])
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    trial_model.eval()
                    with torch.no_grad():
                        out = trial_model(features, edge_index)
                        
                        if is_regression:
                            val_pred = out[val_idx].cpu().numpy()
                            val_true = labels[val_idx].cpu().numpy()
                        else:
                            val_pred = out[val_idx].argmax(dim=1).cpu().numpy()
                            val_true = labels[val_idx].cpu().numpy()
                        
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
            else:
                best_params = {}
    
    # Handle MLP models
    elif isinstance(model, MLPModel):
        print(f"🔄 Training MLP")
        
        # Update hyperparameter optimization for MLPs
        if optimize_hyperparams:
            print(f"🎯 Optimizing MLP hyperparameters...")
            
            # Get model creator function for MLPs
            model_creator = lambda **kwargs: MLPModel(**kwargs)
            
            # Run optimization
            hyperopt_results = optimize_transductive_hyperparameters(
                model_creator=model_creator,
                task_data=task_data,
                config=config,
                model_type="mlp",
                n_trials=config.n_trials,
                timeout=config.optimization_timeout,
                device=device,
                is_regression=is_regression
            )
            
            # Apply optimized parameters
            if hyperopt_results and hyperopt_results['best_params']:
                best_params = hyperopt_results['best_params']
                print(f"🎯 Applying optimized parameters:")
                for key, value in best_params.items():
                    print(f"   {key}: {value}")
                
                # Recreate model with optimized parameters
                input_dim = task_data['features'].shape[1]
                output_dim = task_data['metadata']['output_dim']
                
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
            else:
                best_params = {}
    
    # Handle sklearn models
    elif isinstance(model, SklearnModel):
        print(f"🔄 Training sklearn model: {model.model_type}")
        
        # Update hyperparameter optimization for sklearn models
        if optimize_hyperparams:
            print(f"🎯 Optimizing sklearn model hyperparameters...")
            
            # Get model creator function for sklearn models
            model_creator = lambda **kwargs: SklearnModel(**kwargs)
            
            # Run optimization
            hyperopt_results = optimize_transductive_hyperparameters(
                model_creator=model_creator,
                task_data=task_data,
                config=config,
                model_type=model.model_type,
                n_trials=config.n_trials,
                timeout=config.optimization_timeout,
                device=device,
                is_regression=is_regression
            )
            
            # Apply optimized parameters
            if hyperopt_results and hyperopt_results['best_params']:
                best_params = hyperopt_results['best_params']
                print(f"🎯 Applying optimized parameters:")
                for key, value in best_params.items():
                    print(f"   {key}: {value}")
                
                # Recreate model with optimized parameters
                input_dim = task_data['features'].shape[1]
                output_dim = task_data['metadata']['output_dim']
                
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
            else:
                best_params = {}
    
    # Train the model (either original or optimized)
    results = train_transductive_model(
        model=model,
        task_data=task_data,
        config=config,
        task=task,
        device=device
    )
    

    results['optimal_hyperparams'] = best_params
    

    return results