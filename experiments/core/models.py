"""
Model definitions for MMSB graph learning experiments.

This module provides model classes for different types of graph learning models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, MessagePassing
from torch_geometric.utils import degree
from sklearn.base import BaseEstimator
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from experiments.core.data import prepare_data


class FALayer(MessagePassing):
    """Frequency Adaptive Layer for FAGCN."""
    
    def __init__(self, num_hidden: int, dropout: float):
        super(FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
    
    def forward(self, h, edge_index):
        # Compute edge normalization for current batch/graph
        row, col = edge_index
        norm_degree = degree(row, num_nodes=h.size(0)).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        
        # Compute attention weights
        h2 = torch.cat([h[row], h[col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        
        return self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out


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
        concat_heads: bool = True,
        eps: float = 0.3  # For FAGCN residual connection
    ):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes for classification, number of communities for regression)
            gnn_type: Type of GNN to use ("gcn", "gat", "sage", or "fagcn")
            num_layers: Number of GNN layers
            dropout: Dropout rate
            is_regression: Whether this is a regression task (True) or classification task (False)
            residual: Whether to use residual connections (not applicable for FAGCN)
            norm_type: Type of normalization to use (not applicable for FAGCN)
            agg_type: Type of aggregation to use (not applicable for FAGCN)
            heads: Number of attention heads for GAT (not applicable for FAGCN)
            concat_heads: Whether to concatenate attention heads for GAT (not applicable for FAGCN)
            eps: Residual connection weight for FAGCN (ignored for other models)
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
        self.eps = eps
        
        # FAGCN has a different architecture
        if gnn_type == "fagcn":
            self._init_fagcn(input_dim, hidden_dim, output_dim, dropout)
        else:
            self._init_standard_gnn(input_dim, hidden_dim, output_dim, dropout)
    
    def _init_fagcn(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        """Initialize FAGCN architecture."""
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(FALayer(hidden_dim, dropout))
        
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.output_transform = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_normal_(self.input_transform.weight, gain=1.414)
        nn.init.xavier_normal_(self.output_transform.weight, gain=1.414)
        
        # Activation for regression (ReLU to ensure non-negative counts)
        self.regression_activation = torch.nn.ReLU() if self.is_regression else None
    
    def _init_standard_gnn(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        """Initialize standard GNN architecture (GCN, GAT, SAGE)."""
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.gnn_type == "gat":
            self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=self.heads, concat=self.concat_heads))
        elif self.gnn_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=self.agg_type))
        
        # Add normalization layer if specified
        if self.norm_type == "batch":
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * self.heads if self.gnn_type == "gat" and self.concat_heads else hidden_dim))
        elif self.norm_type == "layer":
            self.norms.append(torch.nn.LayerNorm(hidden_dim * self.heads if self.gnn_type == "gat" and self.concat_heads else hidden_dim))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATv2Conv(hidden_dim * self.heads if self.concat_heads else hidden_dim, 
                                        hidden_dim, heads=self.heads, concat=self.concat_heads))
            elif self.gnn_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=self.agg_type))
            
            # Add normalization layer if specified
            if self.norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim * self.heads if self.gnn_type == "gat" and self.concat_heads else hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(hidden_dim * self.heads if self.gnn_type == "gat" and self.concat_heads else hidden_dim))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_dim * self.heads if self.gnn_type == "gat" and self.concat_heads else hidden_dim, output_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Activation for regression (ReLU to ensure non-negative counts)
        self.regression_activation = torch.nn.ReLU() if self.is_regression else None
    
    def forward(self, x, edge_index):
        """Forward pass."""
        if self.gnn_type == "fagcn":
            return self._forward_fagcn(x, edge_index)
        else:
            return self._forward_standard_gnn(x, edge_index)
    
    def _forward_fagcn(self, x, edge_index):
        """Forward pass for FAGCN."""
        x = self.dropout(x)
        x = torch.relu(self.input_transform(x))
        x = self.dropout(x)
        
        raw = x  # Store initial representation for residual connections
        
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = self.eps * raw + x  # Residual connection with epsilon weighting
        
        x = self.output_transform(x)
        
        # For regression, ensure non-negative outputs
        if self.is_regression:
            x = self.regression_activation(x)
            return x
        else:
            return F.log_softmax(x, dim=1)
    
    def _forward_standard_gnn(self, x, edge_index):
        """Forward pass for standard GNN architectures."""
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