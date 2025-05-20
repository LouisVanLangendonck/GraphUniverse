"""
Model definitions for MMSB graph learning experiments.

This module provides model classes for different types of graph learning models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.base import BaseEstimator
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from experiments.core.data import prepare_data

class GNNModel(torch.nn.Module):
    """Graph Neural Network model for node classification or regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
        is_regression: bool = False,
        residual: bool = False,
        norm_type: str = "none",
        agg_type: str = "mean",
        heads: int = 1,
        concat_heads: bool = True
    ):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes for classification, number of communities for regression)
            gnn_type: Type of GNN to use ("gcn", "gat", or "sage")
            num_layers: Number of GNN layers
            dropout: Dropout rate
            is_regression: Whether this is a regression task (True) or classification task (False)
            residual: Whether to use residual connections
            norm_type: Type of normalization to use ("none", "batch", or "layer")
            agg_type: Type of aggregation to use ("mean", "max", or "sum")
            heads: Number of attention heads for GAT
            concat_heads: Whether to concatenate attention heads for GAT
        """
        super().__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.residual = residual
        self.norm_type = norm_type
        self.agg_type = agg_type
        self.heads = heads
        self.concat_heads = concat_heads
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        if gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=concat_heads))
        elif gnn_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=agg_type))
        
        # Add normalization layer if specified
        if norm_type == "batch":
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads if gnn_type == "gat" and concat_heads else hidden_dim))
        elif norm_type == "layer":
            self.norms.append(torch.nn.LayerNorm(hidden_dim * heads if gnn_type == "gat" and concat_heads else hidden_dim))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim * heads if concat_heads else hidden_dim, 
                                        hidden_dim, heads=heads, concat=concat_heads))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=agg_type))
            
            # Add normalization layer if specified
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads if gnn_type == "gat" and concat_heads else hidden_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(hidden_dim * heads if gnn_type == "gat" and concat_heads else hidden_dim))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_dim * heads if gnn_type == "gat" and concat_heads else hidden_dim, output_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Activation for regression (ReLU to ensure non-negative counts)
        self.regression_activation = torch.nn.ReLU() if is_regression else None
    
    def forward(self, x, edge_index):
        """Forward pass."""
        # GNN layers
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Add residual connection if enabled
            if self.residual and x.shape == identity.shape:
                x = x + identity
        
        # Output layer
        x = self.lin(x)
        
        # For regression, ensure non-negative outputs
        if self.is_regression:
            x = self.regression_activation(x)
        
        return x


class MLPModel(torch.nn.Module):
    """Multi-layer perceptron for node classification or regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        is_regression: bool = False
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes for classification, number of communities for regression)
            num_layers: Number of hidden layers
            dropout: Dropout rate
            is_regression: Whether this is a regression task (True) or classification task (False)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.is_regression = is_regression
        
        # Input layer
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, hidden_dim)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Activation for regression (ReLU to ensure non-negative counts)
        self.regression_activation = torch.nn.ReLU() if is_regression else None
    
    def forward(self, x):
        """Forward pass."""
        # Hidden layers
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        # For regression, ensure non-negative outputs
        if self.is_regression:
            x = self.regression_activation(x)
        
        return x


class SklearnModel:
    """Wrapper for scikit-learn models."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_regression: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42
    ):
        """
        Initialize scikit-learn model.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes for classification, number of communities for regression)
            is_regression: Whether this is a regression task (True) or classification task (False)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            random_state: Random state for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        
        if is_regression:
            # For regression, use RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            # For classification, use RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
    
    def fit(self, X, y):
        """Fit the model."""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            x: Input features [num_samples, input_dim]
            
        Returns:
            Class probabilities [num_nodes, output_dim]
        """
        return self.model.predict_proba(x)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params) -> 'SklearnModel':
        """Set model parameters."""
        self.model.set_params(**params)
        return self 