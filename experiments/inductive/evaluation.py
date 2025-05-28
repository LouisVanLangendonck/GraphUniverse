from typing import Dict, Union
import numpy as np
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def evaluate_inductive_model(
    model: Union[GNNModel, MLPModel, SklearnModel],
    dataloader: DataLoader,
    is_regression: bool,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if isinstance(model, GNNModel):
                out = model(batch.x, batch.edge_index)
            else:
                out = model(batch.x)
            
            all_preds.append(out.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    if is_regression:
        metrics = compute_regression_metrics(y_true, y_pred)
        # Use MAE as the primary metric
        primary_metric = metrics['mae']
    else:
        metrics = compute_classification_metrics(y_true, y_pred)
        primary_metric = metrics['f1']
    
    return {
        'metrics': metrics,
        'primary_metric': primary_metric
    } 