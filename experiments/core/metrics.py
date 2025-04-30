"""
Metrics and evaluation utilities for graph learning experiments.

This module provides functions for evaluating model performance and computing various metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Optional, Union, Any


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