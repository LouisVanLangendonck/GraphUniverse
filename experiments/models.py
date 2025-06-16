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
import inspect

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
        precompute_encodings: bool = False,
        cache_encodings: bool = True,
        attn_type: str = "multihead",
        pe_dim: int = 16,
        pe_type: str = 'laplacian',  # New parameter
        pe_norm_type: str = 'layer',    # New parameter
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
        self.pe_type = pe_type
        self.pe_norm_type = pe_norm_type
        self.use_precomputed_graph_norm = False
        
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
        """Initialize GraphGPS architecture using enhanced implementation."""
        # Input projection that combines features and PE
        self.input_proj = nn.Linear(input_dim + self.pe_dim, hidden_dim)
        
        # PE normalization - configurable
        if self.pe_norm_type == 'layer':
            self.pe_norm = nn.LayerNorm(self.pe_dim)
        elif self.pe_norm_type == 'batch':
            self.pe_norm = nn.BatchNorm1d(self.pe_dim)
        elif self.pe_norm_type == 'instance':
            self.pe_norm = nn.InstanceNorm1d(self.pe_dim)
        # For graph-normalized PEs, we don't need to normalize since it's precomputed. We just need to select other dataloader
        elif self.pe_norm_type == 'graph':
            self.use_precomputed_graph_norm = True
            self.pe_norm = nn.Identity()
        else:
            self.pe_norm = nn.Identity()
        
        # Create GPS layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                self._create_gps_layer(hidden_dim, self.num_heads, dropout, local_gnn_type)
            )
    
    def _create_gps_layer(self, hidden_dim: int, num_heads: int, dropout: float, local_gnn_type: str):
        """Create a GPS layer with local + global components."""
        # Create local GNN layer
        if local_gnn_type == "gcn":
            local_conv = GCNConv(hidden_dim, hidden_dim)
        elif local_gnn_type == "sage":
            local_conv = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown local GNN type: {local_gnn_type}")
        
        # Create GPS layer components
        return nn.ModuleDict({
            'local_conv': local_conv,
            'global_attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True),
            'norm1': nn.LayerNorm(hidden_dim) if self.prenorm else nn.Identity(),
            'norm2': nn.LayerNorm(hidden_dim) if self.prenorm else nn.Identity(),
            'ffn': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        })
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through transformer encoder."""
        if self.transformer_type == "graphormer":
            return self._forward_graphormer(x, edge_index, batch)
        elif self.transformer_type == "graphgps":
            return self._forward_graphgps(x, edge_index, batch, **kwargs)
    
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
            
            # Compute spatial encoding
            spatial_matrix = self._compute_spatial_matrix(edge_index, x.size(0))
            spatial_emb = self.spatial_encoder(spatial_matrix)
            h = h + spatial_emb
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
        
        h = self.layer_norm(h)
        return h
    
    def _compute_spatial_matrix(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute spatial encoding matrix for Graphormer."""
        try:
            # Convert to dense adjacency matrix
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            
            # Compute shortest path distances
            spatial_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=edge_index.device)
            
            # Use Floyd-Warshall algorithm for shortest paths
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if adj[i,k] and adj[k,j]:
                            if spatial_matrix[i,j] == 0 or spatial_matrix[i,j] > spatial_matrix[i,k] + spatial_matrix[k,j]:
                                spatial_matrix[i,j] = spatial_matrix[i,k] + spatial_matrix[k,j]
            
            return spatial_matrix
            
        except Exception as e:
            # Fallback to zero matrix
            return torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=edge_index.device)
    
    def _forward_graphgps(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs):
        """Enhanced forward pass for GraphGPS."""
        # Get precomputed PE
        if self.use_precomputed_graph_norm:
            pe_attr_name = f"{self.pe_type}_pe_norm"
        else:
            pe_attr_name = f"{self.pe_type}_pe"
        
        if pe_attr_name in kwargs:
            pe = kwargs[pe_attr_name]
        else:
            # Try to get from first graph in batch (assuming homogeneous batch)
            pe = getattr(kwargs.get('data', None), pe_attr_name, None)
            # Print Attribute names from kwargs
            if pe is None:
                raise ValueError(f"PE not found for {pe_attr_name} in kwargs")
        
        # Ensure PE is on same device
        pe = pe.to(x.device)
        
        # Normalize PE
        if self.pe_norm_type == 'batch' and pe.size(0) > 1:
            pe = self.pe_norm(pe)
        elif self.pe_norm_type != 'batch':
            pe = self.pe_norm(pe)
        
        # Combine input features with PE
        h = torch.cat([x, pe], dim=-1)
        h = self.input_proj(h)
        
        # Apply GPS layers
        for layer in self.layers:
            # Local message passing
            h_local = layer['local_conv'](h, edge_index)
            
            # Global attention
            if batch is None:
                # Single graph case
                h_global, _ = layer['global_attn'](h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
                h_global = h_global.squeeze(0)
            else:
                # Batched case - process each graph separately
                h_global = []
                for b in range(batch.max().item() + 1):
                    mask = (batch == b)
                    graph_h = h[mask]
                    graph_h_global, _ = layer['global_attn'](
                        graph_h.unsqueeze(0), graph_h.unsqueeze(0), graph_h.unsqueeze(0)
                    )
                    h_global.append(graph_h_global.squeeze(0))
                h_global = torch.cat(h_global, dim=0)
            
            # Combine and apply residual + norm
            h = layer['norm1'](h + h_local + h_global)
            h = layer['norm2'](h + layer['ffn'](h))
        
        return h


class GraphTransformerModel(torch.nn.Module):
    """Full Graph Transformer model with prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        transformer_type: str = "graphgps",
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
        cache_encodings: bool = False,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        attn_type: str = "multihead",
        pe_dim: int = 16,
        pe_type: str = 'laplacian',  # New parameter for PE type
        pe_norm_type: str = 'layer',    # New parameter for normalization type
    ):
        super().__init__()
        
        # Store transformer type and configuration
        self.transformer_type = transformer_type
        self.pe_type = pe_type
        self.pe_norm_type = pe_norm_type
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
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
            pe_type=pe_type,
            pe_norm_type=pe_norm_type,
        )
        
        # Create prediction head
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass through full model."""
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index, batch, **kwargs)
        
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


def precompute_family_encodings(family_graphs: List, model: GraphTransformerModel) -> None:
    """
    Precompute encodings for a family of graphs to improve training efficiency.
    
    Args:
        family_graphs: List of graphs in the family
        model: GraphTransformerModel instance to use for precomputation
    """
    print(f"Precomputing encodings for {len(family_graphs)} graphs...")
    
    # Get the encoder
    if not hasattr(model, 'encoder'):
        raise ValueError("Model must have an encoder attribute")
    encoder = model.encoder
    
    # Ensure encoder has caching enabled
    if not encoder.cache_encodings:
        print("Warning: Encoder caching is disabled. Enabling for precomputation...")
        encoder.cache_encodings = True
    
    # Initialize appropriate cache based on transformer type
    if encoder.transformer_type == "graphormer":
        if not hasattr(encoder, '_encoding_cache'):
            encoder._encoding_cache = {}
    elif encoder.transformer_type == "graphgps":
        if not hasattr(encoder, '_pe_cache'):
            encoder._pe_cache = {}
    
    # Process each graph
    for i, graph in enumerate(family_graphs):
        if i % 10 == 0:
            print(f"Processing graph {i+1}/{len(family_graphs)}")
        
        try:
            # Get graph data
            x = torch.tensor(graph.features, dtype=torch.float)
            
            # Convert NetworkX edges to tensor format
            edge_list = list(graph.graph.edges())
            if not edge_list:  # Handle empty graph case
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Compute encodings based on transformer type
            if encoder.transformer_type == "graphormer":
                # Compute degree encoding
                deg = degree(edge_index[0], num_nodes=x.size(0), dtype=torch.long)
                _ = encoder.degree_encoder(deg)  # This will be cached
                
                # Compute spatial encoding
                spatial_matrix = encoder._compute_spatial_matrix(edge_index, x.size(0))
                _ = encoder.spatial_encoder(spatial_matrix)  # This will be cached
                
            elif encoder.transformer_type == "graphgps":
                # Compute Laplacian PE
                pe = encoder._compute_laplacian_pe(edge_index, x.size(0))
                cache_key = f"{x.size(0)}_{edge_index.size(1)}"
                encoder._pe_cache[cache_key] = pe
            
        except Exception as e:
            print(f"Warning: Failed to precompute encodings for graph {i}: {str(e)}")
            continue
    
    # Print cache statistics
    if encoder.transformer_type == "graphormer":
        print(f"Precomputation completed. Cache size: {len(encoder._encoding_cache)} entries")
    elif encoder.transformer_type == "graphgps":
        print(f"Precomputation completed. PE cache size: {len(encoder._pe_cache)} entries")
