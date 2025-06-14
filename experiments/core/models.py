"""
Model definitions for DCCC-SBM graph learning experiments.

This module provides model classes for different types of graph learning models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, GCNConv, SAGEConv, GATv2Conv, MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import deque

def create_model(model_type: str, **kwargs):
    """Extended model creation to include Graph Transformers."""
    if model_type in ['graphormer', 'graphgps']:
        return GraphTransformerModel(transformer_type=model_type, **kwargs)
    elif model_type in ['gcn', 'sage', 'gat', 'fagcn', 'gin']:
        return GNNModel(gnn_type=model_type, **kwargs)
    elif model_type == 'mlp':
        return MLPModel(**kwargs)
    elif model_type == 'rf':
        return SklearnModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class FALayer(MessagePassing):
    """Frequency Adaptive Layer for FAGCN."""
    
    def __init__(self, num_hidden: int, dropout: float):
        super(FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(num_hidden, 1)
        # Add linear transformation for input features
        self.transform = nn.Linear(num_hidden, num_hidden)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        nn.init.xavier_normal_(self.transform.weight, gain=1.414)
    
    def forward(self, h, edge_index):
        # Transform input features
        h = self.transform(h)
        
        # Compute edge normalization for current batch/graph
        row, col = edge_index
        norm_degree = degree(row, num_nodes=h.size(0)).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        
        # Compute attention weights
        g = torch.tanh(self.gate(h[row])).squeeze()
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        
        # Add residual connection
        out = self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)
        return out + h  # Add residual connection
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) layer with optional layer norm and residual connections.
    
    Implements: h_v^(k+1) = MLP((1 + ε) · h_v^(k) + AGG({h_u^(k) : u ∈ N(v)}))
    With optional residual: h_v^(k+1) = h_v^(k+1) + h_v^(k) (if dimensions match)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, eps: float = 0.0, train_eps: bool = False,
                 use_layer_norm: bool = False, use_residual: bool = False):
        super().__init__(aggr='add')  # Sum aggregation for theoretical guarantees
        
        self.use_residual = use_residual and (input_dim == hidden_dim)
        
        # Epsilon parameter for self-loop weighting
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))
        
        # MLP layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization choice
        if use_layer_norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.BatchNorm1d(hidden_dim)
        
        self.activation = nn.ReLU()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Store input for potential residual connection
        identity = x
        
        # Aggregate neighbor features
        out = self.propagate(edge_index, x=x)
        
        # Add self-connection with epsilon weighting
        out = (1 + self.eps) * x + out
        
        # Apply MLP
        out = self.linear1(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.linear2(out)
        
        # Add residual connection if enabled and dimensions match
        if self.use_residual:
            out = out + identity
        
        return out
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function - return neighbor features."""
        return x_j
    

class GNNEncoder(torch.nn.Module):
    """Base GNN encoder without prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
        residual: bool = False,
        norm_type: str = "none",
        agg_type: str = "mean",
        heads: int = 1,
        concat_heads: bool = True,
        eps: float = 0.3  # For FAGCN residual connection
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm_type = norm_type
        self.agg_type = agg_type
        self.heads = heads
        self.concat_heads = concat_heads
        self.eps = eps
        
        # Initialize appropriate GNN type
        if gnn_type == "fagcn":
            self._init_fagcn(input_dim, hidden_dim, dropout)
        elif gnn_type == "gin":
            self._init_gin(input_dim, hidden_dim, dropout)
        else:
            self._init_standard_gnn(input_dim, hidden_dim, dropout)
    
    def _init_fagcn(self, input_dim: int, hidden_dim: int, dropout: float):
        """Initialize FAGCN layers."""
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(FALayer(input_dim, dropout))
        if self.norm_type != "none":
            self.norms.append(self._get_norm_layer(input_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(FALayer(hidden_dim, dropout))
            if self.norm_type != "none":
                self.norms.append(self._get_norm_layer(hidden_dim))
    
    def _init_standard_gnn(self, input_dim: int, hidden_dim: int, dropout: float):
        """Initialize standard GNN layers (GCN, GAT, etc.)."""
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.gnn_type == "gat":
            # For GAT, ensure hidden_dim is divisible by heads
            if self.concat_heads:
                # When concatenating heads, each head's output should be hidden_dim/heads
                out_channels = hidden_dim // self.heads
                if hidden_dim % self.heads != 0:
                    # Adjust hidden_dim to be divisible by heads
                    out_channels = hidden_dim // self.heads
                    hidden_dim = out_channels * self.heads
            else:
                # When not concatenating, each head's output should be hidden_dim
                out_channels = hidden_dim
            self.convs.append(GATv2Conv(input_dim, out_channels, heads=self.heads, concat=self.concat_heads))
        elif self.gnn_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
            
        if self.norm_type != "none":
            # For GAT with multiple heads, we need to handle the concatenated dimension
            norm_dim = hidden_dim if self.gnn_type != "gat" or not self.concat_heads else hidden_dim
            self.norms.append(self._get_norm_layer(norm_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                # For GAT, ensure hidden_dim is divisible by heads
                if self.concat_heads:
                    # When concatenating heads, each head's output should be hidden_dim/heads
                    out_channels = hidden_dim // self.heads
                    if hidden_dim % self.heads != 0:
                        # Adjust hidden_dim to be divisible by heads
                        out_channels = hidden_dim // self.heads
                        hidden_dim = out_channels * self.heads
                else:
                    # When not concatenating, each head's output should be hidden_dim
                    out_channels = hidden_dim
                self.convs.append(GATv2Conv(hidden_dim, out_channels, heads=self.heads, concat=self.concat_heads))
            elif self.gnn_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
            if self.norm_type != "none":
                # For GAT with multiple heads, we need to handle the concatenated dimension
                norm_dim = hidden_dim if self.gnn_type != "gat" or not self.concat_heads else hidden_dim
                self.norms.append(self._get_norm_layer(norm_dim))
    
    def _init_gin(self, input_dim: int, hidden_dim: int, dropout: float):
        """Initialize GIN layers."""
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GINConv(
            input_dim, hidden_dim, 
            eps=0.0, train_eps=True,
            use_layer_norm=(self.norm_type == "layer"),
            use_residual=False
        ))
        
        if self.norm_type == "batch":
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(
                hidden_dim, hidden_dim, 
                eps=0.0, train_eps=True,
                use_layer_norm=(self.norm_type == "layer"),
                use_residual=False
            ))
            if self.norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
    
    def _get_norm_layer(self, dim: int) -> torch.nn.Module:
        """Get normalization layer based on norm_type."""
        if self.norm_type == "batch":
            return torch.nn.BatchNorm1d(dim)
        elif self.norm_type == "layer":
            return torch.nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN encoder."""
        if self.gnn_type == "fagcn":
            return self._forward_fagcn(x, edge_index)
        elif self.gnn_type == "gin":
            return self._forward_gin(x, edge_index)
        else:
            return self._forward_standard_gnn(x, edge_index)
    
    def _forward_fagcn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for FAGCN."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.norm_type != "none":
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def _forward_gin(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for GIN."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.norm_type == "batch":
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def _forward_standard_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for standard GNNs."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.norm_type != "none":
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GNNModel(torch.nn.Module):
    """Full GNN model with prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        residual: bool = False,
        norm_type: str = "none",
        agg_type: str = "mean",
        heads: int = 1,
        concat_heads: bool = True,
        eps: float = 0.3
    ):
        super().__init__()
        
        # Create encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            norm_type=norm_type,
            agg_type=agg_type,
            heads=heads,
            concat_heads=concat_heads,
            eps=eps
        )
        
        # Create prediction head
        self.is_graph_level_task = is_graph_level_task
        self.is_regression = is_regression

        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # Only add sigmoid for classification tasks
        if not is_regression:
            self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        else:
            # Add scaling factor for regression tasks
            self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through full model."""
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index)
        
        # Apply prediction head
        if self.is_graph_level_task:
            # Global mean pooling for graph-level tasks
            if batch is None:
                # Single graph case - add batch dimension
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Batched case - use global_mean_pool to maintain batch dimension
                h = global_mean_pool(h, batch)  # Shape: [batch_size, hidden_dim]
        
        # Apply prediction head
        if self.is_graph_level_task:
            out = self.readout(h).squeeze(-1)  # Shape: [batch_size, output_dim]
        else:
            out = self.readout(h)  # Shape: [batch_size, output_dim]
            
        # Apply scaling factor for regression tasks
        if self.is_regression:
            out = out * self.scale_factor
            
        return out


class GPSConv(torch.nn.Module):
    """
    PyG-based GPS layer implementation.
    Adapted from torch_geometric.nn.conv.gps_conv
    """
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        """Reset all learnable parameters."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of GPS layer."""
        hs = []
        
        # Local MPNN
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x  # Residual connection
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention
        h, mask = to_dense_batch(x, batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        # Combine local and global outputs
        out = sum(hs)

        # MLP
        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out


class LaplacianPE(nn.Module):
    """Laplacian positional encoding."""
    def __init__(self, channels: int, pe_dim: int):
        super().__init__()
        self.linear = nn.Linear(pe_dim, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GraphTransformerEncoder(torch.nn.Module):
    """Base Graph Transformer encoder without prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transformer_type: str = "graphormer",
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 8,
        max_nodes: int = 200,
        max_path_length: int = 10,
        use_edge_features: bool = False,
        prenorm: bool = True,
        local_gnn_type: str = "gcn",
        global_model_type: str = "transformer",
        precompute_encodings: bool = True,
        cache_encodings: bool = True,
        attn_type: str = "multihead",
        pe_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_type = transformer_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_nodes = max_nodes
        self.max_path_length = max_path_length
        self.use_edge_features = use_edge_features
        self.prenorm = prenorm
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.precompute_encodings = precompute_encodings
        self.cache_encodings = cache_encodings
        self.pe_dim = pe_dim
        
        # Encoding cache
        self._encoding_cache = {} if cache_encodings else None
        
        # Initialize appropriate transformer type
        if transformer_type == "graphormer":
            self._init_graphormer(input_dim, hidden_dim, dropout)
        elif transformer_type == "graphgps":
            self._init_graphgps(input_dim, hidden_dim, dropout, local_gnn_type, attn_type)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
    
    def _init_graphormer(self, input_dim: int, hidden_dim: int, dropout: float):
        """Initialize Graphormer architecture."""
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encodings
        self.spatial_encoder = SpatialEncoder(hidden_dim, self.max_path_length)
        self.degree_encoder = DegreeEncoder(hidden_dim, max_degree=100)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, self.num_heads, dropout)
            for _ in range(self.num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _init_graphgps(self, input_dim: int, hidden_dim: int, dropout: float, 
                      local_gnn_type: str, attn_type: str):
        """Initialize GraphGPS architecture using PyG's GPSConv."""
        # Input projection that combines features and PE
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # PE projection
        self.pe_lin = nn.Linear(self.pe_dim, hidden_dim)
        self.pe_norm = nn.BatchNorm1d(self.pe_dim)
        
        # Create local GNN layers based on type
        if local_gnn_type == "gcn":
            from torch_geometric.nn import GCNConv
            conv_fn = lambda: GCNConv(hidden_dim, hidden_dim)
        elif local_gnn_type == "sage":
            from torch_geometric.nn import SAGEConv
            conv_fn = lambda: SAGEConv(hidden_dim, hidden_dim)
        elif local_gnn_type == "gin":
            from torch_geometric.nn import GINConv
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv_fn = lambda: GINConv(mlp)
        else:
            raise ValueError(f"Unknown local GNN type: {local_gnn_type}")
        
        # GPS layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=conv_fn(),
                    heads=self.num_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    norm='batch_norm' if not self.prenorm else 'layer_norm'
                )
            )
    
    def _compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute Laplacian positional encoding."""
        try:
            from torch_geometric.utils import to_dense_adj
            
            # Convert to dense adjacency matrix
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            
            # Compute degree matrix
            deg = adj.sum(dim=1)
            
            # Handle isolated nodes
            deg_inv_sqrt = torch.zeros_like(deg)
            non_zero_mask = deg > 0
            deg_inv_sqrt[non_zero_mask] = deg[non_zero_mask].pow(-0.5)
            
            # Compute normalized Laplacian
            deg_matrix = torch.diag(deg_inv_sqrt)
            normalized_adj = deg_matrix @ adj @ deg_matrix
            laplacian = torch.eye(num_nodes, device=adj.device) - normalized_adj
            
            # Compute eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
            
            # Sort by eigenvalues
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Take smallest eigenvalues (skip first which is ~0)
            pe_dim = min(self.pe_dim, num_nodes - 1)
            if pe_dim > 0:
                pe = eigenvecs[:, 1:pe_dim + 1]
                
                # Pad if necessary
                if pe.size(1) < self.pe_dim:
                    padding = torch.zeros(num_nodes, self.pe_dim - pe.size(1), device=pe.device)
                    pe = torch.cat([pe, padding], dim=1)
            else:
                pe = torch.zeros(num_nodes, self.pe_dim, device=edge_index.device)
                
        except Exception as e:
            # Fallback to zero features
            pe = torch.zeros(num_nodes, self.pe_dim, device=edge_index.device)
        
        return pe
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass through transformer encoder."""
        if self.transformer_type == "graphormer":
            return self._forward_graphormer(x, edge_index, batch)
        elif self.transformer_type == "graphgps":
            return self._forward_graphgps(x, edge_index, batch)
    
    def _forward_graphormer(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass for Graphormer."""
        # Project input
        h = self.input_proj(x)
        
        # Add positional encodings if needed
        if self.precompute_encodings:
            # Compute degree encoding
            deg = degree(edge_index[0], num_nodes=x.size(0), dtype=torch.long)
            degree_emb = self.degree_encoder(deg)
            h = h + degree_emb
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
        
        h = self.layer_norm(h)
        return h
    
    def _forward_graphgps(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass for GraphGPS."""
        # Compute PE if needed
        pe = None
        if self.precompute_encodings:
            # Check cache first
            if self.cache_encodings and hasattr(self, '_pe_cache'):
                cache_key = f"{x.size(0)}_{edge_index.size(1)}"
                if cache_key in self._pe_cache:
                    pe = self._pe_cache[cache_key]
            
            if pe is None:
                pe = self._compute_laplacian_pe(edge_index, x.size(0))
                if self.cache_encodings:
                    if not hasattr(self, '_pe_cache'):
                        self._pe_cache = {}
                    self._pe_cache[f"{x.size(0)}_{edge_index.size(1)}"] = pe
        
        # Project input features
        h = self.input_proj(x)
        
        # Add PE if available
        if pe is not None:
            pe_normalized = self.pe_norm(pe)
            pe_proj = self.pe_lin(pe_normalized)
            h = h + pe_proj
        
        # Apply GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
        
        return h


class GraphTransformerModel(torch.nn.Module):
    """Full Graph Transformer model with prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        transformer_type: str = "graphormer",
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 8,
        max_nodes: int = 200,
        max_path_length: int = 10,
        use_edge_features: bool = False,
        prenorm: bool = True,
        local_gnn_type: str = "gcn",
        global_model_type: str = "transformer",
        precompute_encodings: bool = True,
        cache_encodings: bool = True,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        attn_type: str = "multihead",
        pe_dim: int = 16,
    ):
        super().__init__()
        
        # Store transformer type
        self.transformer_type = transformer_type
        
        # Create encoder
        self.encoder = GraphTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            transformer_type=transformer_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            max_nodes=max_nodes,
            max_path_length=max_path_length,
            use_edge_features=use_edge_features,
            prenorm=prenorm,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            precompute_encodings=precompute_encodings,
            cache_encodings=cache_encodings,
            attn_type=attn_type,
            pe_dim=pe_dim,
        )
        
        # Create prediction head
        self.is_graph_level_task = is_graph_level_task
        self.is_regression = is_regression
        
        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # Only add sigmoid for classification tasks
        if not is_regression:
            self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        else:
            # Add scaling factor for regression tasks
            self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through full model."""
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index, batch)
        
        # Apply prediction head
        if self.is_graph_level_task:
            # Global mean pooling for graph-level tasks
            if batch is None:
                # Single graph case
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Batched case
                h = global_mean_pool(h, batch)
        
        # Apply prediction head
        out = self.readout(h)
        
        # Apply scaling factor for regression tasks
        if self.is_regression:
            out = out * self.scale_factor
            
        return out


class SpatialEncoder(nn.Module):
    """Spatial encoding for Graphormer."""
    def __init__(self, hidden_dim: int, max_path_length: int):
        super().__init__()
        self.max_path_length = max_path_length
        self.spatial_embedding = nn.Embedding(max_path_length, hidden_dim)
    
    def forward(self, spatial_matrix: torch.Tensor) -> torch.Tensor:
        return self.spatial_embedding(spatial_matrix.clamp(0, self.max_path_length-1))


class DegreeEncoder(nn.Module):
    """Degree encoding for Graphormer."""
    def __init__(self, hidden_dim: int, max_degree: int = 100):
        super().__init__()
        self.max_degree = max_degree
        self.degree_embedding = nn.Embedding(max_degree, hidden_dim)
    
    def forward(self, degrees: torch.Tensor) -> torch.Tensor:
        return self.degree_embedding(degrees.clamp(0, self.max_degree-1))


class GraphormerLayer(nn.Module):
    """Single Graphormer transformer layer."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                encodings: Optional[Dict] = None, batch: Optional[torch.Tensor] = None):
        """Forward pass through Graphormer layer."""
        if batch is None:
            # Single graph case
            h_input = h.unsqueeze(0)
            h_attn, _ = self.self_attn(h_input, h_input, h_input)
            h_attn = h_attn.squeeze(0)
            h = self.norm1(h + h_attn)
            h_ffn = self.ffn(h)
            h = self.norm2(h + h_ffn)
        else:
            # Batched case - process each graph separately
            batch_size = batch.max().item() + 1
            outputs = []
            
            for b in range(batch_size):
                mask = (batch == b)
                graph_h = h[mask]
                graph_h_input = graph_h.unsqueeze(0)
                graph_h_attn, _ = self.self_attn(graph_h_input, graph_h_input, graph_h_input)
                graph_h_attn = graph_h_attn.squeeze(0)
                graph_h = self.norm1(graph_h + graph_h_attn)
                graph_h_ffn = self.ffn(graph_h)
                graph_h = self.norm2(graph_h + graph_h_ffn)
                outputs.append(graph_h)
            
            h = torch.cat(outputs, dim=0)
        
        return h


class MLPModel(nn.Module):
    """MLP model for graph-level tasks."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float, is_regression: bool = True, is_graph_level_task = False):
        super(MLPModel, self).__init__()
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Add final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, batch=None):
        if self.is_graph_level_task:
            # For graph-level tasks, we need to aggregate node features
            if batch is not None:
                # Use global mean pooling to get graph-level features
                x = global_mean_pool(x, batch)
        
        # Pass through MLP
        out = self.mlp(x)
        
        # For regression tasks, ensure output is properly shaped
        if self.is_regression and self.is_graph_level_task:
            out = out.squeeze(-1)  # Remove last dimension if it's 1
            
        return out


class SklearnModel:
    """Wrapper for scikit-learn models."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_regression: bool = False,
        model_type: str = 'rf',
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
            model_type: Type of sklearn model to use ('rf' for Random Forest)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            random_state: Random state for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        self.model_type = model_type
        
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


class FineTuningModel(torch.nn.Module):
    """
    Model for fine-tuning pre-trained graph encoders.
    
    This model takes a pre-trained encoder and adds a task-specific prediction head.
    The encoder can be frozen during training if desired.
    """
    
    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        freeze_encoder: bool = False,
        use_hidden_head: bool = False,
        head_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
        # Create prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Add sigmoid for binary classification
        if not is_regression:
            self.head.add_module("sigmoid", nn.Sigmoid())
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges] (optional for non-GNN encoders)
            batch: Batch indices [num_nodes] (optional for graph-level tasks)
            
        Returns:
            Predictions [num_nodes, output_dim] or [batch_size, output_dim] for graph-level tasks
        """
        # Get embeddings from encoder
        if hasattr(self.encoder, 'convs'):  # This is a GNN model
            if edge_index is None:
                raise ValueError("GNN model requires edge_index")
            h = self.encoder(x, edge_index)
        else:  # This is an MLP or other non-GNN model
            h = self.encoder(x)
        
        # For graph-level tasks, perform pooling
        if self.is_graph_level_task:
            if batch is None:
                # If no batch provided, assume single graph
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Pool per graph in batch
                h = global_mean_pool(h, batch)
        
        # Apply prediction head
        out = self.head(h)
        
        # Squeeze the output to match target shape
        if self.is_graph_level_task and self.is_regression:
            out = out.squeeze(-1)  # Remove last dimension for classification tasks
        
        return out
