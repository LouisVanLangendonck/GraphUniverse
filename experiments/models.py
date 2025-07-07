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
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch, unbatch, add_self_loops
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import deque, defaultdict
import inspect
import math
import time

def create_model(model_type: str, **kwargs):
    """Extended model creation to include Sheaf Diffusion models."""
    if model_type == 'graphgps':
        return GraphTransformerModel(transformer_type=model_type, **kwargs)
    elif model_type in ['gcn', 'sage', 'gat', 'fagcn', 'gin']:
        return GNNModel(gnn_type=model_type, **kwargs)
    elif model_type == 'mlp':
        return MLPModel(**kwargs)
    elif model_type == 'rf':
        return SklearnModel(**kwargs)
    elif model_type in ['sheaf_diag', 'sheaf_bundle', 'sheaf_general', 'sheaf_orthogonal']:
        # Map model types to sheaf types
        sheaf_type_map = {
            'sheaf_diag': 'diagonal',
            'sheaf_bundle': 'orthogonal', 
            'sheaf_general': 'general',
            'sheaf_orthogonal': 'orthogonal'
        }
        sheaf_type = sheaf_type_map[model_type]
        return SheafDiffusionModel(sheaf_type=sheaf_type, **kwargs)
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
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def _forward_gin(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for GIN."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.norm_type == "batch":
                h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def _forward_standard_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for standard GNNs."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.norm_type != "none":
                h = self.norms[i](h)
            h = F.elu(h)
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

        self.readout = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, output_dim),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            # torch.nn.Linear(hidden_dim, output_dim)
        )
        
        # Only add sigmoid for classification tasks
        # if not is_regression:
        #     self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        # else:
        #     # Add scaling factor for regression tasks
        #     self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))
    
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
            
        # # Apply scaling factor for regression tasks
        # if self.is_regression:
        #     out = out * self.scale_factor
            
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
        elif attn_type == 'bigbird':
            self.attn = BigBirdAttention(
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
        elif isinstance(self.attn, BigBirdAttention):
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


class BigBirdAttention(nn.Module):
    """
    BigBird attention mechanism implementation.
    Based on the paper "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        num_random_blocks: int = 3,
        block_size: int = 64,
        use_global_attention: bool = True,
        use_window_attention: bool = True,
        use_random_attention: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.num_random_blocks = num_random_blocks
        self.block_size = block_size
        self.use_global_attention = use_global_attention
        self.use_window_attention = use_window_attention
        self.use_random_attention = use_random_attention

        # Projections for Q, K, V
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _get_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create attention mask for BigBird attention pattern."""
        # Initialize full attention mask
        mask = torch.zeros(seq_len, seq_len, device=device)

        # Add window attention
        if self.use_window_attention:
            for i in range(seq_len):
                start = max(0, i - self.block_size // 2)
                end = min(seq_len, i + self.block_size // 2 + 1)
                mask[i, start:end] = 1

        # Add random attention
        if self.use_random_attention:
            for i in range(seq_len):
                # Select random blocks
                random_indices = torch.randperm(seq_len)[:self.num_random_blocks]
                mask[i, random_indices] = 1

        # Add global attention
        if self.use_global_attention:
            # First and last tokens attend to all
            mask[0, :] = 1
            mask[-1, :] = 1
            mask[:, 0] = 1
            mask[:, -1] = 1

        return mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of BigBird attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            mask: Optional padding mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, channels]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, channels]
        k = self.k_proj(x)  # [batch_size, seq_len, channels]
        v = self.v_proj(x)  # [batch_size, seq_len, channels]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Apply BigBird attention pattern
        attn_mask = self._get_attention_mask(seq_len, device)
        scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out


class PerformerAttention(nn.Module):
    """
    Performer attention mechanism implementation.
    Based on the paper "Rethinking Attention with Performers" (Choromanski et al., 2020)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        num_random_features: int = 256,
        use_orthogonal_features: bool = True,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.num_random_features = num_random_features
        self.use_orthogonal_features = use_orthogonal_features
        self.use_softmax = use_softmax
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        
        self._init_random_features()
        self._reset_parameters()
    
    def _init_random_features(self):
        if self.use_orthogonal_features:
            # Use orthogonal random features for better approximation
            self.random_features = nn.Parameter(torch.randn(self.num_random_features, self.channels))
            nn.init.orthogonal_(self.random_features)
        else:
            self.random_features = nn.Parameter(torch.randn(self.num_random_features, self.channels))
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def _get_random_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.random_features.t())
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project inputs
        q = self.q_proj(query)  # [batch_size, seq_len, channels]
        k = self.k_proj(key)    # [batch_size, seq_len, channels]
        v = self.v_proj(value)  # [batch_size, seq_len, channels]
        
        # Get random features
        q_features = self._get_random_features(q)  # [batch_size, seq_len, num_random_features]
        k_features = self._get_random_features(k)  # [batch_size, seq_len, num_random_features]
        
        # Compute attention scores using random features
        attention_scores = torch.matmul(q_features, k_features.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        if self.use_softmax:
            attention_weights = F.softmax(attention_scores / math.sqrt(self.channels), dim=-1)
        else:
            attention_weights = attention_scores / math.sqrt(self.channels)
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [batch_size, seq_len, channels]
        output = self.out_proj(output)
        
        return output


class LaplacianPE(nn.Module):
    """Laplacian positional encoding."""
    def __init__(self, channels: int, pe_dim: int):
        super().__init__()
        self.linear = nn.Linear(pe_dim, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GPSLayer(nn.Module):
    """Fixed GPS layer implementation."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 local_gnn_type: str, attn_type: str):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Local GNN component
        if local_gnn_type == "gcn":
            self.local_conv = GCNConv(hidden_dim, hidden_dim)
        elif local_gnn_type == "sage":
            self.local_conv = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown local GNN type: {local_gnn_type}")
        
        # Global attention component - FIXED
        if attn_type == "multihead":
            self.global_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        elif attn_type == "performer":
            self.global_attn = PerformerAttention(
                channels=hidden_dim, heads=num_heads, dropout=dropout
            )
        elif attn_type == "bigbird":
            self.global_attn = BigBirdAttention(
                channels=hidden_dim, heads=num_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        
        # Normalization and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None):
        """FIXED: Proper GPS layer forward pass."""
        
        residual = h
        
        # 1. Local MPNN - FIXED: Proper residual connection
        h_local = self.local_conv(h, edge_index)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h = self.norm1(h + h_local)  # Residual connection
        
        # 2. Global attention - FIXED: Handle batching properly
        h_global = self._apply_global_attention(h, batch)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        h = self.norm2(h + h_global)  # Residual connection
        
        # 3. Feed-forward network
        h_ffn = self.ffn(h)
        h = self.norm3(h + h_ffn)  # Residual connection
        
        return h
    
    def _apply_global_attention(self, h: torch.Tensor, batch: Optional[torch.Tensor]):
        """Apply global attention with proper batching."""
        
        if isinstance(self.global_attn, nn.MultiheadAttention):
            if batch is None:
                # Single graph case
                h_input = h.unsqueeze(0)  # Add batch dimension
                h_attn, _ = self.global_attn(h_input, h_input, h_input)
                return h_attn.squeeze(0)  # Remove batch dimension
            else:
                # Multiple graphs - use to_dense_batch for proper handling
                h_dense, mask = to_dense_batch(h, batch)
                batch_size, max_nodes, hidden_dim = h_dense.shape
                
                # Apply attention to each graph in the batch
                h_attn = torch.zeros_like(h_dense)
                for i in range(batch_size):
                    graph_mask = mask[i]
                    graph_nodes = graph_mask.sum().item()
                    if graph_nodes > 0:
                        graph_h = h_dense[i, :graph_nodes].unsqueeze(0)
                        attn_out, _ = self.global_attn(graph_h, graph_h, graph_h)
                        h_attn[i, :graph_nodes] = attn_out.squeeze(0)
                
                # Convert back to node-level representation
                return h_attn[mask]
        
        elif isinstance(self.global_attn, PerformerAttention):
            # Handle PerformerAttention specifically
            if batch is None:
                # Single graph case
                h_input = h.unsqueeze(0)  # Add batch dimension
                h_attn = self.global_attn(h_input, h_input, h_input)
                return h_attn.squeeze(0)  # Remove batch dimension
            else:
                # Multiple graphs - use to_dense_batch for proper handling
                h_dense, mask = to_dense_batch(h, batch)
                batch_size, max_nodes, hidden_dim = h_dense.shape
                
                # Apply attention to each graph in the batch
                h_attn = torch.zeros_like(h_dense)
                for i in range(batch_size):
                    graph_mask = mask[i]
                    graph_nodes = graph_mask.sum().item()
                    if graph_nodes > 0:
                        graph_h = h_dense[i, :graph_nodes].unsqueeze(0)
                        attn_out = self.global_attn(graph_h, graph_h, graph_h)
                        h_attn[i, :graph_nodes] = attn_out.squeeze(0)
                
                # Convert back to node-level representation
                return h_attn[mask]
        
        elif isinstance(self.global_attn, BigBirdAttention):
            # Handle BigBirdAttention specifically (different interface)
            if batch is None:
                # Single graph case
                h_input = h.unsqueeze(0)  # Add batch dimension
                h_attn = self.global_attn(h_input)
                return h_attn.squeeze(0)  # Remove batch dimension
            else:
                # Multiple graphs - use to_dense_batch for proper handling
                h_dense, mask = to_dense_batch(h, batch)
                batch_size, max_nodes, hidden_dim = h_dense.shape
                
                # Apply attention to each graph in the batch
                h_attn = torch.zeros_like(h_dense)
                for i in range(batch_size):
                    graph_mask = mask[i]
                    graph_nodes = graph_mask.sum().item()
                    if graph_nodes > 0:
                        graph_h = h_dense[i, :graph_nodes].unsqueeze(0)
                        attn_out = self.global_attn(graph_h, mask=graph_mask.unsqueeze(0))
                        h_attn[i, :graph_nodes] = attn_out.squeeze(0)
                
                # Convert back to node-level representation
                return h_attn[mask]
        
        elif hasattr(self.global_attn, 'forward'):
            # For other custom attention implementations
            if batch is None:
                return self.global_attn(h, h, h)
            else:
                h_dense, mask = to_dense_batch(h, batch)
                h_attn = self.global_attn(h_dense, mask=mask)
                return h_attn[mask]
        
        else:
            raise ValueError("Unknown attention implementation")


class GraphTransformerEncoder(torch.nn.Module):
    """Base Graph Transformer encoder without prediction head."""
    
    def __init__(self, input_dim: int, hidden_dim: int, transformer_type: str = "graphgps", 
                 num_layers: int = 3, dropout: float = 0.1, num_heads: int = 8,
                 pe_dim: int = 16, pe_type: str = 'laplacian', pe_norm_type: str = 'layer',
                 local_gnn_type: str = "gcn", attn_type: str = "multihead", **kwargs):
        super().__init__()
        self.transformer_type = transformer_type
        self.pe_type = pe_type
        self.pe_norm_type = pe_norm_type
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        if transformer_type == "graphgps":
            self._init_graphgps(input_dim, hidden_dim, dropout, local_gnn_type, attn_type, num_heads)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
        
    def _init_graphgps(self, input_dim: int, hidden_dim: int, dropout: float, 
                      local_gnn_type: str, attn_type: str, num_heads: int):
        """Initialize GraphGPS architecture using enhanced implementation."""
        # Input projection that combines features and PE
        self.input_proj = nn.Linear(input_dim + self.pe_dim, hidden_dim)
        
        # PE normalization - configurable
        self.use_precomputed_graph_norm = False
        if self.pe_norm_type == 'layer':
            self.pe_norm = nn.LayerNorm(self.pe_dim)
        elif self.pe_norm_type == 'batch':
            self.pe_norm = nn.BatchNorm1d(self.pe_dim)
        # For graph-normalized PEs, we don't need to normalize since it's precomputed. We just need to select other dataloader
        elif self.pe_norm_type == 'graph':
            self.use_precomputed_graph_norm = True
            self.pe_norm = nn.Identity()
        else:
            self.pe_norm = nn.Identity()
        
        # Create GPS layers with fixed implementation
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                local_gnn_type=local_gnn_type,
                attn_type=attn_type
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through transformer encoder."""
        """Enhanced forward pass for GraphGPS."""
        # Get precomputed PE
        pe_attr_name = f"{self.pe_type}_pe"
        if self.pe_norm_type == 'graph':
            pe_attr_name += "_norm"
        
        pe = None
        if pe_attr_name in kwargs:
            pe = kwargs[pe_attr_name]
        elif 'data' in kwargs:
            pe = getattr(kwargs['data'], pe_attr_name, None)

        if pe is None:
            # Fallback: compute PE on the fly
            print(f"Warning: PE not found, computing on the fly")
            from data import PositionalEncodingComputer
            pe_computer = PositionalEncodingComputer(self.pe_dim, [self.pe_type])
            pe = pe_computer.compute_laplacian_pe(edge_index, x.size(0))
        
        # Ensure PE is on same device and has correct shape
        pe = pe.to(x.device)
        if pe.size(0) != x.size(0):
            raise ValueError(f"PE size {pe.size(0)} doesn't match node count {x.size(0)}")
        if pe.size(1) != self.pe_dim:
            # Pad or truncate PE to correct dimension
            if pe.size(1) < self.pe_dim:
                pad_size = self.pe_dim - pe.size(1)
                pe = torch.cat([pe, torch.zeros(pe.size(0), pad_size, device=pe.device)], dim=1)
            else:
                pe = pe[:, :self.pe_dim]
        
        # Normalize PE
        if self.pe_norm_type == 'batch' and pe.size(0) > 1:
            pe = self.pe_norm(pe.unsqueeze(0)).squeeze(0)
        elif self.pe_norm_type != 'batch':
            pe = self.pe_norm(pe)
        
        # Combine input features with PE
        h = torch.cat([x, pe], dim=-1)
        h = self.input_proj(h)
        
        # Apply GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, batch)
        
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
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        local_gnn_type: str = "gcn",
        attn_type: str = "multihead",
        pe_dim: int = 16,
        pe_type: str = 'laplacian',
        pe_norm_type: str = 'layer',
    ):
        super().__init__()
        
        # Store transformer type and configuration
        self.transformer_type = transformer_type
        self.pe_type = pe_type
        self.pe_norm_type = pe_norm_type
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
        # Create encoder with only the parameters it accepts
        self.encoder = GraphTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            transformer_type=transformer_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            pe_dim=pe_dim,
            pe_type=pe_type,
            pe_norm_type=pe_norm_type,
            local_gnn_type=local_gnn_type,
            attn_type=attn_type,
        )
        
        # Create prediction head
        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, output_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, output_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # Only add sigmoid for classification tasks
        # if not is_regression:
        #     self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        # else:
        #     # Add scaling factor for regression tasks
        #     self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))
    
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



class OrthogonalMaps(nn.Module):
    """
    Creates orthogonal restriction maps using different parameterizations.
    Based on the Orthogonal class from the paper's codebase.
    """
    def __init__(self, d: int, method: str = "householder"):
        super().__init__()
        self.d = d
        self.method = method
        
        # Initialize timing statistics
        self.timing_stats = defaultdict(float)
        self.timing_counts = defaultdict(int)
        
        if method == "householder":
            # Use Householder reflections for orthogonal matrices
            # Need d(d-1)/2 parameters for d x d orthogonal matrix
            self.num_params = d * (d - 1) // 2
        elif method == "euler":
            # Euler angles (only works for d=2,3)
            if d == 2:
                self.num_params = 1
            elif d == 3:
                self.num_params = 3
            else:
                raise ValueError("Euler method only supports d=2 or d=3")
        else:
            raise ValueError(f"Unknown orthogonal method: {method}")
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameters to orthogonal matrices.
        
        Args:
            params: [..., num_params] tensor of parameters
            
        Returns:
            [..., d, d] tensor of orthogonal matrices
        """
        batch_dims = params.shape[:-1]
        
        if self.method == "householder":
            return self._householder_transform(params)
        elif self.method == "euler":
            if self.d == 2:
                return self._euler_2d(params)
            elif self.d == 3:
                return self._euler_3d(params)
    
    def _householder_transform(self, params: torch.Tensor) -> torch.Tensor:
        """Householder reflection based orthogonal matrix generation."""
        batch_dims = params.shape[:-1]
        device = params.device
        
        # Time matrix creation
        start_time = time.time()
        tril_indices = torch.tril_indices(self.d, self.d, offset=-1, device=device)
        A = torch.zeros(*batch_dims, self.d, self.d, device=device)
        A[..., tril_indices[0], tril_indices[1]] = params
        eye = torch.eye(self.d, device=device).expand(*batch_dims, -1, -1)
        A = A + eye
        self.timing_stats['householder_matrix_setup'] += time.time() - start_time
        self.timing_counts['householder_matrix_setup'] += 1
        
        # Time QR decomposition (most expensive part)
        start_time = time.time()
        Q, _ = torch.linalg.qr(A)
        self.timing_stats['householder_qr'] += time.time() - start_time
        self.timing_counts['householder_qr'] += 1
        
        return Q
    
    def _euler_2d(self, params: torch.Tensor) -> torch.Tensor:
        """2D rotation matrices from angle parameter."""
        angles = params[..., 0] * 2 * math.pi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        batch_dims = params.shape[:-1]
        Q = torch.zeros(*batch_dims, 2, 2, device=params.device)
        Q[..., 0, 0] = cos_a
        Q[..., 0, 1] = -sin_a
        Q[..., 1, 0] = sin_a
        Q[..., 1, 1] = cos_a
        
        return Q
    
    def _euler_3d(self, params: torch.Tensor) -> torch.Tensor:
        """3D rotation matrices from Euler angles."""
        alpha = params[..., 0] * 2 * math.pi
        beta = params[..., 1] * 2 * math.pi
        gamma = params[..., 2] * 2 * math.pi
        
        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta), torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)
        
        batch_dims = params.shape[:-1]
        Q = torch.zeros(*batch_dims, 3, 3, device=params.device)
        
        Q[..., 0, 0] = cos_a * cos_b
        Q[..., 0, 1] = cos_a * sin_b * sin_g - sin_a * cos_g
        Q[..., 0, 2] = cos_a * sin_b * cos_g + sin_a * sin_g
        Q[..., 1, 0] = sin_a * cos_b
        Q[..., 1, 1] = sin_a * sin_b * sin_g + cos_a * cos_g
        Q[..., 1, 2] = sin_a * sin_b * cos_g - cos_a * sin_g
        Q[..., 2, 0] = -sin_b
        Q[..., 2, 1] = cos_b * sin_g
        Q[..., 2, 2] = cos_b * cos_g
        
        return Q

class NeuralSheafDiffusionLayer(MessagePassing):
    """
    Neural Sheaf Diffusion Layer implementing message passing with learnable sheaf structure.
    
    This implements the discretized sheaf diffusion process:
    X_{t+1} = X_t - σ(ΔF(t) * (I ⊗ W1) * X_t * W2)
    
    Where ΔF(t) is the normalized sheaf Laplacian learned from node features.
    """
    
    def __init__(
        self,
        d: int,  # stalk dimension
        hidden_channels: int,
        restriction_map_type: str = "orthogonal",  # "orthogonal", "diagonal", "general"
        orthogonal_method: str = "householder",
        activation: str = "elu",
        **kwargs
    ):
        # Initialize timing statistics
        self.timing_stats = defaultdict(float)
        self.timing_counts = defaultdict(int)
        # Use 'add' aggregation and set node_dim=0 for proper tensor indexing
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.d = d
        self.hidden_channels = hidden_channels
        self.restriction_map_type = restriction_map_type
        
        # W1: operates on stalk dimension (d x d)
        self.W1 = nn.Linear(d, d, bias=False)
        
        # W2: operates on feature channels (hidden_channels x hidden_channels)  
        self.W2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        # Epsilon parameter for residual connection (learnable per stalk dimension)
        self.epsilons = nn.Parameter(torch.zeros(d))
        
        # Restriction map generation based on type
        if restriction_map_type == "orthogonal":
            self.orthogonal_maps = OrthogonalMaps(d, orthogonal_method)
            # MLP to generate orthogonal matrix parameters
            restriction_out_dim = self.orthogonal_maps.num_params
        elif restriction_map_type == "diagonal":
            # Only diagonal elements, so d parameters per map
            restriction_out_dim = d
        elif restriction_map_type == "general":
            # Full d x d matrix
            restriction_out_dim = d * d
        else:
            raise ValueError(f"Unknown restriction map type: {restriction_map_type}")
        
        # MLP for generating restriction maps F_ve and F_ue from concatenated features
        # Input: 2 * d * hidden_channels (concatenated transformed features)
        # Output: restriction_out_dim (parameters for restriction map)
        self.restriction_mlp = nn.Sequential(
            nn.Linear(2 * d * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, restriction_out_dim)
        )
        
        # Activation function
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of Neural Sheaf Diffusion.
        
        Args:
            x: Node features [num_nodes, d * hidden_channels]
            edge_index: Edge indices [2, num_edges]
            batch_size: Batch size for batched processing
            
        Returns:
            Updated node features [num_nodes, d * hidden_channels]
        """
        num_nodes = x.size(0)
        
        # Time input reshaping
        start_time = time.time()
        x_reshaped = x.view(num_nodes, self.d, self.hidden_channels)
        x_original = x_reshaped.clone()
        self.timing_stats['input_reshape'] += time.time() - start_time
        self.timing_counts['input_reshape'] += 1
        
        # Time W1 and W2 transformations
        start_time = time.time()
        x_transformed = torch.einsum('ndf,dk->nkf', x_reshaped, self.W1.weight)
        x_transformed = torch.einsum('ndf,fg->ndg', x_transformed, self.W2.weight)
        self.timing_stats['w1_w2_transform'] += time.time() - start_time
        self.timing_counts['w1_w2_transform'] += 1
        
        # Time graph preprocessing
        start_time = time.time()
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.timing_stats['graph_preprocessing'] += time.time() - start_time
        self.timing_counts['graph_preprocessing'] += 1
        
        # Time message passing (sheaf Laplacian computation)
        start_time = time.time()
        x_flat = x_transformed.view(num_nodes, -1)
        sheaf_laplacian_output = self.propagate(
            edge_index, 
            x=x_flat,
            deg_inv_sqrt=deg_inv_sqrt,
            size=(num_nodes, num_nodes)
        )
        self.timing_stats['message_passing'] += time.time() - start_time
        self.timing_counts['message_passing'] += 1
        
        # Time output processing
        start_time = time.time()
        sheaf_laplacian_output = sheaf_laplacian_output.view(num_nodes, self.d, self.hidden_channels)
        sheaf_laplacian_output = self.activation(sheaf_laplacian_output)
        coeff = (1 + torch.tanh(self.epsilons)).unsqueeze(0).unsqueeze(-1)
        x_updated = coeff * x_original - sheaf_laplacian_output
        result = x_updated.view(num_nodes, -1)
        self.timing_stats['output_processing'] += time.time() - start_time
        self.timing_counts['output_processing'] += 1
        
        return result
    
    def message(
        self, 
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        deg_inv_sqrt_i: torch.Tensor,
        deg_inv_sqrt_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages between nodes.
        This is where we generate the restriction maps F_ve and F_ue.
        
        Args:
            x_j: Messages from neighbor nodes [num_edges, d * hidden_channels]
            x_i: Center node features [num_edges, d * hidden_channels]
            deg_inv_sqrt_i: Degree normalization for center nodes [num_edges]
            deg_inv_sqrt_j: Degree normalization for neighbor nodes [num_edges]
            
        Returns:
            Messages incorporating sheaf structure [num_edges, d * hidden_channels]
        """
        # Time feature reshaping
        start_time = time.time()
        x_i_reshaped = x_i.view(-1, self.d, self.hidden_channels)
        x_j_reshaped = x_j.view(-1, self.d, self.hidden_channels)
        x_i_flat = x_i_reshaped.view(x_i_reshaped.size(0), -1)
        x_j_flat = x_j_reshaped.view(x_j_reshaped.size(0), -1)
        self.timing_stats['message_feature_reshape'] += time.time() - start_time
        self.timing_counts['message_feature_reshape'] += 1
        
        # Time feature concatenation
        start_time = time.time()
        concat_features_ve = torch.cat([x_i_flat, x_j_flat], dim=-1)
        concat_features_ue = torch.cat([x_j_flat, x_i_flat], dim=-1)
        self.timing_stats['message_feature_concat'] += time.time() - start_time
        self.timing_counts['message_feature_concat'] += 1
        
        # Time restriction map generation (MLP)
        start_time = time.time()
        F_ve_params = self.restriction_mlp(concat_features_ve)
        F_ue_params = self.restriction_mlp(concat_features_ue)
        self.timing_stats['message_restriction_mlp'] += time.time() - start_time
        self.timing_counts['message_restriction_mlp'] += 1
        
        # Time restriction map conversion
        start_time = time.time()
        F_ve = self._params_to_restriction_map(F_ve_params)
        F_ue = self._params_to_restriction_map(F_ue_params)
        self.timing_stats['message_restriction_conversion'] += time.time() - start_time
        self.timing_counts['message_restriction_conversion'] += 1
        
        # Time normalization and matrix operations
        start_time = time.time()
        norm_i = deg_inv_sqrt_i.view(-1, 1, 1)
        norm_j = deg_inv_sqrt_j.view(-1, 1, 1)
        x_i_norm = norm_i * x_i_reshaped
        x_j_norm = norm_j * x_j_reshaped
        self.timing_stats['message_normalization'] += time.time() - start_time
        self.timing_counts['message_normalization'] += 1
        
        # Time matrix multiplications (most expensive part)
        start_time = time.time()
        F_ve_x_i = torch.einsum('eij,ejf->eif', F_ve, x_i_norm)
        F_ue_x_j = torch.einsum('eij,ejf->eif', F_ue, x_j_norm)
        diff = F_ve_x_i - F_ue_x_j
        message = torch.einsum('eji,ejf->eif', F_ve, diff)
        self.timing_stats['message_matrix_ops'] += time.time() - start_time
        self.timing_counts['message_matrix_ops'] += 1
        
        # Time final reshaping
        start_time = time.time()
        result = message.view(message.size(0), -1)
        self.timing_stats['message_final_reshape'] += time.time() - start_time
        self.timing_counts['message_final_reshape'] += 1
        
        return result
    
    def _params_to_restriction_map(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameters to restriction maps based on the specified type.
        
        Args:
            params: Parameters [num_edges, restriction_out_dim]
            
        Returns:
            Restriction maps [num_edges, d, d]
        """
        num_edges = params.size(0)
        
        if self.restriction_map_type == "orthogonal":
            # Use orthogonal parameterization
            start_time = time.time()
            result = self.orthogonal_maps(params)
            self.timing_stats['orthogonal_maps'] += time.time() - start_time
            self.timing_counts['orthogonal_maps'] += 1
            return result
            
        elif self.restriction_map_type == "diagonal":
            # Create diagonal matrices
            start_time = time.time()
            maps = torch.zeros(num_edges, self.d, self.d, device=params.device)
            diag_elements = torch.sigmoid(params) - 0.5
            maps.diagonal(dim1=-2, dim2=-1).copy_(diag_elements)
            self.timing_stats['diagonal_maps'] += time.time() - start_time
            self.timing_counts['diagonal_maps'] += 1
            return maps
            
        elif self.restriction_map_type == "general":
            # Reshape to full matrices
            start_time = time.time()
            result = params.view(num_edges, self.d, self.d)
            self.timing_stats['general_maps'] += time.time() - start_time
            self.timing_counts['general_maps'] += 1
            return result
    
    def aggregate(
        self, 
        inputs: torch.Tensor, 
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Aggregate messages. Since we're implementing the sheaf Laplacian,
        we just sum the messages (which already include the sheaf structure).
        """
        # Sum messages for each node (standard aggregation)
        return super().aggregate(inputs, index, ptr, dim_size)
    
    def get_timing_stats(self):
        """Get timing statistics for this layer."""
        stats = {}
        for key in self.timing_stats:
            if self.timing_counts[key] > 0:
                stats[key] = {
                    'total_time': self.timing_stats[key],
                    'avg_time': self.timing_stats[key] / self.timing_counts[key],
                    'count': self.timing_counts[key]
                }
        return stats

class NeuralSheafDiffusionEncoder(nn.Module):
    """
    Multi-layer Neural Sheaf Diffusion Network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d: int,  # stalk dimension
        num_layers: int = 2,
        sheaf_type: str = "orthogonal",
        orthogonal_method: str = "euler",
        activation: str = "elu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial projection to d * hidden_channels
        self.input_projection = nn.Linear(input_dim, d * hidden_dim)
        
        # Sheaf diffusion layers
        self.sheaf_layers = nn.ModuleList([
            NeuralSheafDiffusionLayer(
                d=d,
                hidden_channels=hidden_dim,
                restriction_map_type=sheaf_type,
                orthogonal_method=orthogonal_method,
                activation=activation
            ) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete Neural Sheaf Diffusion network.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output log probabilities [num_nodes, output_dim]
        """
        # Initial projection to stalk structure
        x = self.input_projection(x)  # [num_nodes, d * hidden_dim]
        x = self.dropout(x)
        
        # Apply sheaf diffusion layers
        for layer in self.sheaf_layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # # Final projection to output
        # x = self.output_projection(x)  # [num_nodes, output_dim]
        
        # Return sigmoid activation
        return x
    
    def get_timing_stats(self):
        """Get timing statistics from all sheaf layers."""
        all_stats = {}
        for i, layer in enumerate(self.sheaf_layers):
            layer_stats = layer.get_timing_stats()
            for key, value in layer_stats.items():
                all_stats[f'layer_{i}_{key}'] = value
        return all_stats

class SheafDiffusionModel(torch.nn.Module):
    """Full Sheaf Diffusion model with prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        d: int = 3,  # stalk dimension
        num_layers: int = 2,
        sheaf_type: str = "orthogonal",  # "orthogonal", "diagonal", "general"
        orthogonal_method: str = "euler",
        activation: str = "elu",
        dropout: float = 0.0,
        is_regression: bool = False,
        is_graph_level_task: bool = False
    ):
        super().__init__()
        
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
        # Create encoder
        self.encoder = NeuralSheafDiffusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            d=d,
            num_layers=num_layers,
            sheaf_type=sheaf_type,
            orthogonal_method=orthogonal_method,
            activation=activation,
            dropout=dropout
        )
        
        # Create prediction head
        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = torch.nn.Sequential(
                # ELU activation function
                torch.nn.ELU(),
                torch.nn.Linear(d * hidden_dim, output_dim)
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = torch.nn.Sequential(
                torch.nn.ELU(),
                torch.nn.Linear(d * hidden_dim, output_dim)
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # Only add sigmoid for classification tasks
        # if not is_regression:
        #     self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        # else:
        #     # Add scaling factor for regression tasks
        #     self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through full model."""
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index)  # [num_nodes, d * hidden_dim]
        
        # Apply prediction head
        if self.is_graph_level_task:
            # Global mean pooling for graph-level tasks
            if batch is None:
                # Single graph case - add batch dimension
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Batched case - use global_mean_pool to maintain batch dimension
                h = global_mean_pool(h, batch)
        
        # Apply prediction head
        out = self.readout(h)
        
        # # Apply scaling factor for regression tasks
        # if self.is_regression:
        #     out = out * self.scale_factor
            
        return out
    
    def get_timing_stats(self):
        """Get timing statistics from all sheaf layers."""
        all_stats = {}
        for i, layer in enumerate(self.encoder.sheaf_layers):
            layer_stats = layer.get_timing_stats()
            for key, value in layer_stats.items():
                all_stats[f'layer_{i}_{key}'] = value
        return all_stats
