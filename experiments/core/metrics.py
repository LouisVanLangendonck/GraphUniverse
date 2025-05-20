"""
Metrics for evaluating model performance on node classification and regression tasks.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score
)
from typing import Dict, List, Optional, Union, Any

def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    is_regression: bool = False
) -> Dict[str, float]:
    """
    Compute metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        is_regression: Whether this is a regression task (True) or classification task (False)
        
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
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # For multilabel regression, compute metrics per label
        if len(y_true.shape) > 1:
            per_label_metrics = {}
            for i in range(y_true.shape[1]):
                per_label_metrics[f'mse_label_{i}'] = mean_squared_error(y_true[:, i], y_pred[:, i])
                per_label_metrics[f'rmse_label_{i}'] = np.sqrt(per_label_metrics[f'mse_label_{i}'])
                per_label_metrics[f'mae_label_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                per_label_metrics[f'r2_label_{i}'] = r2_score(y_true[:, i], y_pred[:, i])
            metrics.update(per_label_metrics)
    else:
        # Classification metrics
        # For classification, ensure predictions are class indices
        if len(y_pred.shape) > 1:
            # Store probabilities for ROC-AUC before converting to class indices
            y_score = y_pred
            y_pred = np.argmax(y_pred, axis=1)
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # For multiclass classification, compute macro and weighted averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Compute ROC-AUC if we have probability scores
        if 'y_score' in locals():
            try:
                if y_score.shape[1] == 2:  # Binary case
                    metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
                else:  # Multi-class case
                    metrics['roc_auc'] = roc_auc_score(y_true, y_score, multi_class='ovr')
            except Exception as e:
                metrics['roc_auc'] = 0.0
    
    return metrics


def compute_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_regression: bool = False
) -> torch.Tensor:
    """
    Compute loss for model training.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        is_regression: Whether this is a regression task (True) or classification task (False)
        
    Returns:
        Loss value
    """
    if is_regression:
        # For regression, use MSE loss
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
        y_pred: Predicted labels
        is_regression: Whether this is a regression task (True) or classification task (False)
        
    Returns:
        Accuracy value
    """
    if is_regression:
        # For regression, compute R² score as accuracy proxy
        return r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    else:
        # For classification, compute standard accuracy
        return (y_pred.argmax(dim=1) == y_true).float().mean().item()


def evaluate_node_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate node classification performance.
    
    Args:
        y_true: True labels [num_nodes]
        y_pred: Predicted labels [num_nodes]
        y_score: Predicted class probabilities [num_nodes, num_classes]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handle edge cases for precision, recall, F1
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique_labels) <= 1:
        # Single class case
        metrics['metrics_macro'] = {
            'precision': 1.0 if np.all(y_true == y_pred) else 0.0,
            'recall': 1.0 if np.all(y_true == y_pred) else 0.0,
            'f1': 1.0 if np.all(y_true == y_pred) else 0.0
        }
        metrics['metrics_weighted'] = metrics['metrics_macro'].copy()
        metrics['metrics_per_class'] = {
            'precision': [metrics['metrics_macro']['precision']],
            'recall': [metrics['metrics_macro']['recall']],
            'f1': [metrics['metrics_macro']['f1']],
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
            metrics['metrics_macro'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            metrics['metrics_weighted'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            metrics['metrics_per_class'] = {
                'precision': [0.0] * len(unique_labels),
                'recall': [0.0] * len(unique_labels),
                'f1': [0.0] * len(unique_labels),
                'support': [0] * len(unique_labels)
            }
    
    # Try to compute ROC-AUC if probabilities are provided
    if y_score is not None and len(unique_labels) > 1:
        try:
            if y_score.shape[1] == 2:  # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
            else:  # Multi-class case
                metrics['roc_auc'] = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception as e:
            metrics['roc_auc'] = 0.0
    
    return metrics


def model_performance_summary(
    model_results: Dict[str, Any],
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate a comprehensive performance summary for all models.
    
    Args:
        model_results: Dictionary of model results, each containing metrics and predictions
        target_names: Optional list of class names
        
    Returns:
        String containing performance summary
    """
    summary_lines = []
    summary_lines.append("Model Performance Summary")
    summary_lines.append("=" * 30)
    
    for model_name, results in model_results.items():
        summary_lines.append(f"\n{model_name}:")
        summary_lines.append("-" * len(model_name))
        
        # Basic metrics
        test_acc = results.get('test_acc', 0)
        train_time = results.get('train_time', 0)
        
        summary_lines.append(f"Test Accuracy: {test_acc:.4f}")
        summary_lines.append(f"Training Time: {train_time:.2f}s")
        
        # Detailed metrics if available
        metrics = results.get('metrics', {})
        if metrics:
            if 'metrics_macro' in metrics:
                macro = metrics['metrics_macro']
                summary_lines.append("\nMacro Metrics:")
                summary_lines.append(f"  Precision: {macro.get('precision', 0):.4f}")
                summary_lines.append(f"  Recall: {macro.get('recall', 0):.4f}")
                summary_lines.append(f"  F1: {macro.get('f1', 0):.4f}")
            
            if 'metrics_weighted' in metrics:
                weighted = metrics['metrics_weighted']
                summary_lines.append("\nWeighted Metrics:")
                summary_lines.append(f"  Precision: {weighted.get('precision', 0):.4f}")
                summary_lines.append(f"  Recall: {weighted.get('recall', 0):.4f}")
                summary_lines.append(f"  F1: {weighted.get('f1', 0):.4f}")
            
            if 'class_distribution' in metrics:
                dist = metrics['class_distribution']
                summary_lines.append("\nClass Distribution:")
                for i, (count, frac) in enumerate(zip(dist['counts'], dist['fractions'])):
                    class_name = target_names[i] if target_names else f"Class {i}"
                    summary_lines.append(f"  {class_name}: {count} ({frac:.2%})")
    
    return "\n".join(summary_lines)


def compute_classification_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                                labels: Union[torch.Tensor, np.ndarray],
                                return_all: bool = False) -> Dict[str, float]:
    """
    Compute classification metrics.
    
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
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    if return_all:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {'accuracy': accuracy}


def compute_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    return_all: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
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
    r2 = r2_score(y_true, y_pred)
    
    if not return_all:
        return r2
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # For multilabel regression, compute metrics per label
    if len(y_true.shape) > 1:
        per_label_metrics = {}
        for i in range(y_true.shape[1]):
            per_label_metrics[f'mse_label_{i}'] = mean_squared_error(y_true[:, i], y_pred[:, i])
            per_label_metrics[f'rmse_label_{i}'] = np.sqrt(per_label_metrics[f'mse_label_{i}'])
            per_label_metrics[f'mae_label_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            per_label_metrics[f'r2_label_{i}'] = r2_score(y_true[:, i], y_pred[:, i])
        metrics.update(per_label_metrics)
    
    return metrics 