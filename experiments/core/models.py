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

from experiments.core.data import prepare_data

# ... existing code ... 

class GNNModel(nn.Module):
    """
    Base class for GNN models.
    
    Supports various GNN architectures for node classification tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        gnn_type: str = "gcn",
        residual: bool = False,
        batch_norm: bool = False
    ):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ("gcn", "gat", "sage")
            residual: Whether to use residual connections
            batch_norm: Whether to use batch normalization
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Initialize layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(self._create_conv_layer(hidden_dim, output_dim))
            
        # Batch normalization layers
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))
    
    def _create_conv_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a GNN convolutional layer of the specified type."""
        if self.gnn_type == "gcn":
            return GCNConv(in_dim, out_dim)
        elif self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, heads=1)
        elif self.gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings/predictions [num_nodes, output_dim]
        """
        prev_x = None  # For residual connections
        
        for i, conv in enumerate(self.convs):
            # Residual connection
            if self.residual and i > 0 and i < len(self.convs) - 1:
                prev_x = x
            
            # Apply convolution
            x = conv(x, edge_index)
            
            # Final layer doesn't have activation or other operations
            if i < len(self.convs) - 1:
                # Apply batch norm if enabled
                if self.batch_norm:
                    x = self.bns[i](x)
                
                # Apply activation
                x = F.relu(x)
                
                # Apply dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Add residual connection
                if self.residual and prev_x is not None:
                    if prev_x.size() == x.size():
                        x = x + prev_x
        
        return x


class MLPModel(nn.Module):
    """
    Multi-layer perceptron baseline model.
    
    Ignores graph structure and only uses node features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = False
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of MLP layers
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Initialize layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            
        # Batch normalization layers
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the MLP model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Ignored, included for API compatibility with GNN
            
        Returns:
            Node predictions [num_nodes, output_dim]
        """
        for i, layer in enumerate(self.layers):
            # Apply linear layer
            x = layer(x)
            
            # Final layer doesn't have activation or other operations
            if i < len(self.layers) - 1:
                # Apply batch norm if enabled
                if self.batch_norm:
                    x = self.bns[i](x)
                
                # Apply activation
                x = F.relu(x)
                
                # Apply dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SklearnModel:
    """
    Wrapper for scikit-learn models.
    
    Provides a consistent interface with PyTorch models.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_type: str = "random_forest",
        **kwargs
    ):
        """
        Initialize the scikit-learn model.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes)
            model_type: Type of model ("random_forest", etc.)
            **kwargs: Additional arguments for the model
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        
        # Initialize model
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SklearnModel':
        """
        Train the model.
        
        Args:
            x: Input features [num_samples, input_dim]
            y: Labels [num_samples]
            
        Returns:
            Self for chaining
        """
        self.model.fit(x, y)
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input features [num_samples, input_dim]
            
        Returns:
            Predicted class labels [num_nodes]
        """
        return self.model.predict(x)
    
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