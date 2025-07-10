"""
Metrics for evaluating transductive model performance.
Updated to match the inductive experiment metrics system.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score
)
from typing import Dict, List, Optional, Union, Any
import time

def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    is_regression: bool = False
) -> Dict[str, float]:
    """
    Compute metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or values
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    metrics = {}
    
    if is_regression:
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Calculate relative error
        numerator = np.sum(np.abs(y_pred - y_true))
        denominator = np.sum(np.abs(y_true)) + 1e-8  # Add small epsilon to avoid division by zero
        metrics['relative_error'] = numerator / denominator
        
        # Handle R² calculation for potentially problematic cases
        try:
            r2 = r2_score(y_true, y_pred)
            metrics['r2'] = r2 if not np.isnan(r2) else 0.0
        except Exception:
            metrics['r2'] = 0.0
        
        # For multilabel regression, compute metrics per label
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            per_label_metrics = {}
            for i in range(y_true.shape[1]):
                label_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
                per_label_metrics[f'mse_label_{i}'] = label_mse
                per_label_metrics[f'rmse_label_{i}'] = np.sqrt(label_mse)
                per_label_metrics[f'mae_label_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                
                # Calculate relative error per label
                label_numerator = np.sum(np.abs(y_pred[:, i] - y_true[:, i]))
                label_denominator = np.sum(np.abs(y_true[:, i])) + 1e-8
                per_label_metrics[f'relative_error_label_{i}'] = label_numerator / label_denominator
                
                try:
                    label_r2 = r2_score(y_true[:, i], y_pred[:, i])
                    per_label_metrics[f'r2_label_{i}'] = label_r2 if not np.isnan(label_r2) else 0.0
                except Exception:
                    per_label_metrics[f'r2_label_{i}'] = 0.0
                    
            metrics.update(per_label_metrics)
    else:
        # Classification metrics
        # Handle case where predictions might be probabilities
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Store probabilities for ROC-AUC before converting to class indices
            y_score = y_pred
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_score = None
        
        # Ensure both arrays are 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Get unique labels for proper metric calculation
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        
        if len(unique_labels) <= 1:
            # Single class case - set all metrics based on perfect/imperfect prediction
            perfect_prediction = np.all(y_true == y_pred)
            default_score = 1.0 if perfect_prediction else 0.0
            
            metrics.update({
                'precision_macro': default_score,
                'precision_weighted': default_score,
                'recall_macro': default_score,
                'recall_weighted': default_score,
                'f1_macro': default_score,
                'f1_weighted': default_score,
                'roc_auc': default_score
            })
        else:
            # Multi-class case
            try:
                # Macro averages
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                # Weighted averages
                metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
            except Exception as e:
                # Fallback if metric calculation fails
                print(f"Warning: Metric calculation failed: {e}")
                metrics.update({
                    'precision_macro': 0.0,
                    'precision_weighted': 0.0,
                    'recall_macro': 0.0,
                    'recall_weighted': 0.0,
                    'f1_macro': 0.0,
                    'f1_weighted': 0.0
                })
            
            # ROC-AUC if probabilities are available
            if y_score is not None:
                try:
                    if y_score.shape[1] == 2:  # Binary case
                        metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
                    else:  # Multi-class case
                        metrics['roc_auc'] = roc_auc_score(y_true, y_score, multi_class='ovr')
                except Exception:
                    metrics['roc_auc'] = 0.0
            else:
                metrics['roc_auc'] = 0.0
    
    return metrics


def compute_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_regression: bool = False,
    loss_type: str = 'default'
) -> torch.Tensor:
    """
    Compute loss for model training.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or values
        is_regression: Whether this is a regression task
        loss_type: Type of loss ('default', 'mse', 'mae', 'ce')
        
    Returns:
        Loss value
    """
    if is_regression:
        if loss_type == 'mae':
            return torch.nn.functional.l1_loss(y_pred, y_true)
        else:  # Default to MSE for regression
            return torch.nn.functional.mse_loss(y_pred, y_true)
    else:
        # For classification, use cross entropy loss
        return torch.nn.functional.cross_entropy(y_pred, y_true)


def compute_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_regression: bool = False
) -> float:
    """
    Compute accuracy for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or values
        is_regression: Whether this is a regression task
        
    Returns:
        Accuracy value
    """
    if is_regression:
        # For regression, compute R² score as accuracy proxy
        try:
            return r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        except Exception:
            return 0.0
    else:
        # For classification, compute standard accuracy
        return (y_pred.argmax(dim=1) == y_true).float().mean().item()


def evaluate_transductive_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    node_indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate transductive node classification performance.
    
    Args:
        y_true: True labels [num_nodes]
        y_pred: Predicted labels [num_nodes]
        y_score: Predicted class probabilities [num_nodes, num_classes]
        node_indices: Indices of nodes to evaluate (if None, evaluate all)
        
    Returns:
        Dictionary of metrics
    """
    # Filter to specific nodes if provided
    if node_indices is not None:
        y_true = y_true[node_indices]
        y_pred = y_pred[node_indices]
        if y_score is not None:
            y_score = y_score[node_indices]
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handle edge cases for precision, recall, F1
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(unique_labels) <= 1:
        # Single class case
        perfect_prediction = np.all(y_true == y_pred)
        default_score = 1.0 if perfect_prediction else 0.0
        
        metrics['metrics_macro'] = {
            'precision': default_score,
            'recall': default_score,
            'f1': default_score
        }
        metrics['metrics_weighted'] = metrics['metrics_macro'].copy()
        metrics['metrics_per_class'] = {
            'precision': [default_score],
            'recall': [default_score],
            'f1': [default_score],
            'support': [len(y_true)]
        }
    else:
        try:
            # Precision, recall, F1 (macro)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['metrics_macro'] = {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            }
            
            # Precision, recall, F1 (weighted)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['metrics_weighted'] = {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            }
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            metrics['metrics_per_class'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            }
        except Exception as e:
            # If metrics calculation fails, return zeros
            print(f"Warning: Metric calculation failed: {e}")
            metrics['metrics_macro'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            metrics['metrics_weighted'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            metrics['metrics_per_class'] = {
                'precision': [0.0] * len(unique_labels),
                'recall': [0.0] * len(unique_labels),
                'f1': [0.0] * len(unique_labels),
                'support': [0] * len(unique_labels)
            }
    
    # ROC-AUC if probabilities are provided
    if y_score is not None and len(unique_labels) > 1:
        try:
            if y_score.shape[1] == 2:  # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
            else:  # Multi-class case
                metrics['roc_auc'] = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception as e:
            print(f"Warning: ROC-AUC calculation failed: {e}")
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def evaluate_transductive_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate transductive node regression performance.
    
    Args:
        y_true: True values [num_nodes] or [num_nodes, num_targets]
        y_pred: Predicted values [num_nodes] or [num_nodes, num_targets]
        node_indices: Indices of nodes to evaluate (if None, evaluate all)
        
    Returns:
        Dictionary of metrics
    """
    # Filter to specific nodes if provided
    if node_indices is not None:
        y_true = y_true[node_indices]
        y_pred = y_pred[node_indices]
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R² score with error handling
    try:
        r2 = r2_score(y_true, y_pred)
        metrics['r2'] = r2 if not np.isnan(r2) else 0.0
    except Exception:
        metrics['r2'] = 0.0
    
    # For multi-target regression
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        per_target_metrics = {}
        for i in range(y_true.shape[1]):
            target_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            per_target_metrics[f'mse_target_{i}'] = target_mse
            per_target_metrics[f'rmse_target_{i}'] = np.sqrt(target_mse)
            per_target_metrics[f'mae_target_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            
            try:
                target_r2 = r2_score(y_true[:, i], y_pred[:, i])
                per_target_metrics[f'r2_target_{i}'] = target_r2 if not np.isnan(target_r2) else 0.0
            except Exception:
                per_target_metrics[f'r2_target_{i}'] = 0.0
                
        metrics.update(per_target_metrics)
    
    return metrics


def compute_classification_metrics(
    predictions: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray],
    return_all: bool = False
) -> Dict[str, float]:
    """
    Compute classification metrics for transductive learning.
    
    Args:
        predictions: Model predictions (either class indices or probabilities)
        labels: True labels
        return_all: Whether to return all metrics or just accuracy
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        y_pred = predictions.cpu().numpy()
    else:
        y_pred = predictions
        
    if isinstance(labels, torch.Tensor):
        y_true = labels.cpu().numpy()
    else:
        y_true = labels
    
    # Handle probability predictions
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Ensure 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    if return_all:
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception:
            return {
                'accuracy': accuracy,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    return {'accuracy': accuracy}


def compute_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    return_all: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Compute regression metrics for transductive learning.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        return_all: Whether to return all metrics or just R² score
        
    Returns:
        If return_all is False, returns R² score.
        If return_all is True, returns dictionary of metrics.
    """
    # Convert tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    try:
        r2 = r2_score(y_true, y_pred)
        r2 = r2 if not np.isnan(r2) else 0.0
    except Exception:
        r2 = 0.0
    
    if not return_all:
        return r2
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # For multi-target regression
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        per_target_metrics = {}
        for i in range(y_true.shape[1]):
            target_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            per_target_metrics[f'mse_target_{i}'] = target_mse
            per_target_metrics[f'rmse_target_{i}'] = np.sqrt(target_mse)
            per_target_metrics[f'mae_target_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            
            try:
                target_r2 = r2_score(y_true[:, i], y_pred[:, i])
                per_target_metrics[f'r2_target_{i}'] = target_r2 if not np.isnan(target_r2) else 0.0
            except Exception:
                per_target_metrics[f'r2_target_{i}'] = 0.0
                
        metrics.update(per_target_metrics)
    
    return metrics


def model_performance_summary(
    model_results: Dict[str, Any],
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate a comprehensive performance summary for transductive models.
    
    Args:
        model_results: Dictionary of model results, each containing metrics and predictions
        target_names: Optional list of class names
        
    Returns:
        String containing performance summary
    """
    summary_lines = []
    summary_lines.append("TRANSDUCTIVE MODEL PERFORMANCE SUMMARY")
    summary_lines.append("=" * 50)
    
    for model_name, results in model_results.items():
        summary_lines.append(f"\n{model_name.upper()}:")
        summary_lines.append("-" * len(model_name))
        
        # Basic metrics
        test_metrics = results.get('test_metrics', {})
        train_time = results.get('train_time', 0)
        
        if test_metrics:
            # Determine if regression or classification
            is_regression = 'r2' in test_metrics or 'mse' in test_metrics
            
            if is_regression:
                summary_lines.append(f"R² Score: {test_metrics.get('r2', 0):.4f}")
                summary_lines.append(f"MSE: {test_metrics.get('mse', 0):.4f}")
                summary_lines.append(f"MAE: {test_metrics.get('mae', 0):.4f}")
                summary_lines.append(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
                summary_lines.append(f"Relative Error: {test_metrics.get('relative_error', 0):.4f}")
            else:
                summary_lines.append(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                summary_lines.append(f"F1 Macro: {test_metrics.get('f1_macro', 0):.4f}")
                summary_lines.append(f"Precision Macro: {test_metrics.get('precision_macro', 0):.4f}")
                summary_lines.append(f"Recall Macro: {test_metrics.get('recall_macro', 0):.4f}")
                
                if 'roc_auc' in test_metrics:
                    summary_lines.append(f"ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
        else:
            summary_lines.append("No test metrics available")
        
        summary_lines.append(f"Training Time: {train_time:.2f}s")
        
        # Error information if available
        if 'error' in results and results['error']:
            summary_lines.append(f"Error: {results['error']}")
    
    return "\n".join(summary_lines)


def compare_transductive_models(
    results: Dict[str, Dict[str, Any]],
    primary_metric: str = 'auto'
) -> Dict[str, Any]:
    """
    Compare performance of different transductive models.
    
    Args:
        results: Dictionary mapping model names to their results
        primary_metric: Primary metric for comparison ('auto' for automatic selection)
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'models': list(results.keys()),
        'primary_metric': primary_metric,
        'rankings': {},
        'best_model': None,
        'performance_summary': {}
    }
    
    # Auto-select primary metric
    if primary_metric == 'auto':
        # Check if any model has regression metrics
        has_regression = any(
            'r2' in results[model].get('test_metrics', {}) or 'mse' in results[model].get('test_metrics', {})
            for model in results
        )
        
        if has_regression:
            primary_metric = 'r2'
        else:
            primary_metric = 'f1_macro'  # Default to F1 for classification
    
    comparison['primary_metric'] = primary_metric
    
    # Extract performance for each model
    model_scores = {}
    for model_name, model_results in results.items():
        test_metrics = model_results.get('test_metrics', {})
        
        if primary_metric in test_metrics:
            score = test_metrics[primary_metric]
            model_scores[model_name] = score
            
            # Store additional metrics for summary
            comparison['performance_summary'][model_name] = {
                'primary_score': score,
                'train_time': model_results.get('train_time', 0),
                'all_metrics': test_metrics
            }
        else:
            model_scores[model_name] = float('-inf') if primary_metric == 'r2' else 0.0
            comparison['performance_summary'][model_name] = {
                'primary_score': model_scores[model_name],
                'train_time': model_results.get('train_time', 0),
                'error': model_results.get('error', 'Metric not available')
            }
    
    # Rank models
    if model_scores:
        # Sort by score (higher is better for most metrics)
        if primary_metric in ['mse', 'mae', 'rmse']:
            # Lower is better for these metrics
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1])
        else:
            # Higher is better for accuracy, F1, R², etc.
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison['rankings'] = {
            rank + 1: {'model': model, 'score': score}
            for rank, (model, score) in enumerate(ranked_models)
        }
        
        if ranked_models:
            comparison['best_model'] = ranked_models[0][0]
            comparison['best_score'] = ranked_models[0][1]
    
    return comparison


def compute_metrics_gpu(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_regression: bool = False
) -> Dict[str, float]:
    """
    Compute metrics directly on GPU without transferring to CPU.
    This eliminates the bottleneck of CPU transfers during training.
    
    Args:
        y_true: True labels (on GPU)
        y_pred: Predicted labels or values (on GPU)
        is_regression: Whether this is a regression task
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    if is_regression:
        # Regression metrics - all computed on GPU
        mse = torch.nn.functional.mse_loss(y_pred, y_true)
        mae = torch.nn.functional.l1_loss(y_pred, y_true)
        
        metrics['mse'] = mse.item()
        metrics['rmse'] = torch.sqrt(mse).item()
        metrics['mae'] = mae.item()
        
        # Calculate relative error on GPU
        numerator = torch.sum(torch.abs(y_pred - y_true))
        denominator = torch.sum(torch.abs(y_true)) + 1e-8
        metrics['relative_error'] = (numerator / denominator).item()
        
        # R² calculation on GPU
        try:
            # R² = 1 - (SS_res / SS_tot)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics['r2'] = r2.item() if not torch.isnan(r2) else 0.0
        except Exception:
            metrics['r2'] = 0.0
        
        # For multilabel regression, compute metrics per label
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            per_label_metrics = {}
            for i in range(y_true.shape[1]):
                label_mse = torch.nn.functional.mse_loss(y_pred[:, i], y_true[:, i])
                label_mae = torch.nn.functional.l1_loss(y_pred[:, i], y_true[:, i])
                
                per_label_metrics[f'mse_label_{i}'] = label_mse.item()
                per_label_metrics[f'rmse_label_{i}'] = torch.sqrt(label_mse).item()
                per_label_metrics[f'mae_label_{i}'] = label_mae.item()
                
                # Calculate relative error per label
                label_numerator = torch.sum(torch.abs(y_pred[:, i] - y_true[:, i]))
                label_denominator = torch.sum(torch.abs(y_true[:, i])) + 1e-8
                per_label_metrics[f'relative_error_label_{i}'] = (label_numerator / label_denominator).item()
                
                # R² per label
                try:
                    label_ss_res = torch.sum((y_true[:, i] - y_pred[:, i]) ** 2)
                    label_ss_tot = torch.sum((y_true[:, i] - torch.mean(y_true[:, i])) ** 2)
                    label_r2 = 1 - (label_ss_res / (label_ss_tot + 1e-8))
                    per_label_metrics[f'r2_label_{i}'] = label_r2.item() if not torch.isnan(label_r2) else 0.0
                except Exception:
                    per_label_metrics[f'r2_label_{i}'] = 0.0
                    
            metrics.update(per_label_metrics)
    else:
        # Classification metrics - computed on GPU where possible
        # Handle case where predictions might be probabilities
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Store probabilities for later use
            y_score = y_pred
            y_pred_classes = torch.argmax(y_pred, dim=1)
        else:
            y_score = None
            y_pred_classes = y_pred
        
        # Ensure both tensors are 1D
        y_true = y_true.flatten()
        y_pred_classes = y_pred_classes.flatten()
        
        # Basic accuracy on GPU
        correct = torch.sum(y_true == y_pred_classes)
        total = y_true.numel()
        metrics['accuracy'] = (correct / total).item()
        
        # Get unique labels for proper metric calculation
        unique_labels = torch.unique(torch.cat([y_true, y_pred_classes]))
        
        if len(unique_labels) <= 1:
            # Single class case - set all metrics based on perfect/imperfect prediction
            perfect_prediction = torch.all(y_true == y_pred_classes).item()
            default_score = 1.0 if perfect_prediction else 0.0
            
            metrics.update({
                'precision_macro': default_score,
                'precision_weighted': default_score,
                'recall_macro': default_score,
                'recall_weighted': default_score,
                'f1_macro': default_score,
                'f1_weighted': default_score,
                'roc_auc': default_score
            })
        else:
            # Multi-class case - compute some metrics on GPU, others need CPU
            # For now, we'll compute the most important ones on GPU and fall back to CPU for complex ones
            
            # F1 score approximation on GPU (macro average)
            f1_scores = []
            for label in unique_labels:
                tp = torch.sum((y_true == label) & (y_pred_classes == label))
                fp = torch.sum((y_true != label) & (y_pred_classes == label))
                fn = torch.sum((y_true == label) & (y_pred_classes != label))
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                f1_scores.append(f1)
            
            metrics['f1_macro'] = torch.mean(torch.stack(f1_scores)).item()
            
            # For other metrics that are more complex, we'll still use CPU but only when needed
            # This is a trade-off between speed and completeness
            try:
                # Move to CPU only for complex metrics
                y_true_cpu = y_true.cpu().numpy()
                y_pred_cpu = y_pred_classes.cpu().numpy()
                
                # Macro averages
                metrics['precision_macro'] = precision_score(y_true_cpu, y_pred_cpu, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(y_true_cpu, y_pred_cpu, average='macro', zero_division=0)
                
                # Weighted averages
                metrics['precision_weighted'] = precision_score(y_true_cpu, y_pred_cpu, average='weighted', zero_division=0)
                metrics['recall_weighted'] = recall_score(y_true_cpu, y_pred_cpu, average='weighted', zero_division=0)
                metrics['f1_weighted'] = f1_score(y_true_cpu, y_pred_cpu, average='weighted', zero_division=0)
                
            except Exception as e:
                # Fallback if metric calculation fails
                print(f"Warning: Metric calculation failed: {e}")
                metrics.update({
                    'precision_macro': 0.0,
                    'precision_weighted': 0.0,
                    'recall_macro': 0.0,
                    'recall_weighted': 0.0,
                    'f1_weighted': 0.0
                })
            
            # ROC-AUC if probabilities are available
            if y_score is not None:
                try:
                    # ROC-AUC needs CPU for sklearn
                    y_score_cpu = y_score.cpu().numpy()
                    y_true_cpu = y_true.cpu().numpy()
                    
                    if y_score_cpu.shape[1] == 2:  # Binary case
                        metrics['roc_auc'] = roc_auc_score(y_true_cpu, y_score_cpu[:, 1])
                    else:  # Multi-class case
                        metrics['roc_auc'] = roc_auc_score(y_true_cpu, y_score_cpu, multi_class='ovr')
                except Exception:
                    metrics['roc_auc'] = 0.0
            else:
                metrics['roc_auc'] = 0.0
    
    return metrics


def compare_metrics_performance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_regression: bool = False,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Compare performance between CPU and GPU metrics computation.
    
    Args:
        y_true: True labels (on GPU)
        y_pred: Predicted labels or values (on GPU)
        is_regression: Whether this is a regression task
        num_runs: Number of runs for timing comparison
        
    Returns:
        Dictionary with timing results
    """
    # Warm up GPU
    for _ in range(3):
        _ = compute_metrics_gpu(y_true, y_pred, is_regression)
    
    # Time GPU computation
    gpu_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = compute_metrics_gpu(y_true, y_pred, is_regression)
        gpu_times.append(time.time() - start_time)
    
    # Time CPU computation
    cpu_times = []
    for _ in range(num_runs):
        start_time = time.time()
        y_true_cpu = y_true.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        _ = compute_metrics(y_true_cpu, y_pred_cpu, is_regression)
        cpu_times.append(time.time() - start_time)
    
    avg_gpu_time = np.mean(gpu_times)
    avg_cpu_time = np.mean(cpu_times)
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
    
    return {
        'gpu_avg_time_ms': avg_gpu_time * 1000,
        'cpu_avg_time_ms': avg_cpu_time * 1000,
        'speedup': speedup,
        'gpu_std_ms': np.std(gpu_times) * 1000,
        'cpu_std_ms': np.std(cpu_times) * 1000
    }