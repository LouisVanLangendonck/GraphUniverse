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
from torch_geometric.utils import degree, to_dense_adj
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import deque


def create_model(model_type: str, **kwargs):
    """Extended model creation to include Graph Transformers."""
    if model_type in ['graphormer', 'graphgps']:
        return GraphTransformerModel(transformer_type=model_type, **kwargs)
    elif model_type in ['gcn', 'sage', 'gat', 'fagcn']:
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


class GraphTransformerModel(torch.nn.Module):
    """Graph Transformer model for node classification or regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        transformer_type: str = "graphormer",
        num_layers: int = 3,
        dropout: float = 0.1,
        is_regression: bool = False,
        num_heads: int = 8,
        max_nodes: int = 200,  # For pre-computing encodings
        max_path_length: int = 10,  # For shortest path encoding
        use_edge_features: bool = False,
        prenorm: bool = True,
        # GraphGPS specific
        local_gnn_type: str = "gcn",
        global_model_type: str = "transformer",
        # Optimization flags
        precompute_encodings: bool = True,
        cache_encodings: bool = True
    ):
        """
        Initialize Graph Transformer model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            transformer_type: "graphormer" or "graphgps"
            num_layers: Number of transformer layers
            dropout: Dropout rate
            is_regression: Whether this is regression
            num_heads: Number of attention heads
            max_nodes: Maximum nodes for encoding pre-computation
            max_path_length: Maximum path length for shortest path encoding
            use_edge_features: Whether to use edge features
            prenorm: Whether to use pre-normalization
            local_gnn_type: Local GNN type for GraphGPS
            global_model_type: Global model type for GraphGPS
            precompute_encodings: Whether to precompute structural encodings
            cache_encodings: Whether to cache encodings between batches
        """
        super().__init__()
        
        self.transformer_type = transformer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.num_heads = num_heads
        self.max_nodes = max_nodes
        self.max_path_length = max_path_length
        self.precompute_encodings = precompute_encodings
        self.cache_encodings = cache_encodings
        
        # Encoding cache
        self._encoding_cache = {} if cache_encodings else None
        
        if transformer_type == "graphormer":
            self._init_graphormer(input_dim, hidden_dim, output_dim, dropout)
        elif transformer_type == "graphgps":
            self._init_graphgps(input_dim, hidden_dim, output_dim, dropout, 
                              local_gnn_type, global_model_type, prenorm)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.regression_activation = nn.ReLU() if is_regression else None
        
    def _init_graphormer(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
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
        
    def _init_graphgps(self, input_dim: int, hidden_dim: int, output_dim: int, 
                      dropout: float, local_gnn_type: str, global_model_type: str, prenorm: bool):
        """Initialize GraphGPS architecture."""
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encodings
        self.pe_encoder = PositionalEncoder(hidden_dim, pe_type="laplacian")
        
        # GraphGPS layers
        self.layers = nn.ModuleList([
            GraphGPSLayer(
                hidden_dim, 
                self.num_heads, 
                dropout, 
                local_gnn_type, 
                global_model_type,
                prenorm
            )
            for _ in range(self.num_layers)
        ])
        
    def _get_graph_hash(self, edge_index: torch.Tensor, num_nodes: int) -> str:
        """Generate hash for graph structure for caching."""
        if not self.cache_encodings:
            return None
        
        # Simple hash based on edges and node count
        edge_hash = hash(edge_index.cpu().numpy().tobytes())
        return f"{num_nodes}_{edge_hash}"
    
    def _compute_structural_encodings(self, edge_index: torch.Tensor, num_nodes: int, batch_size: int = 1):
        """Compute structural encodings for the graph."""
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        
        # Check cache first
        if graph_hash and graph_hash in self._encoding_cache:
            # Move cached encodings to the same device as edge_index
            cached_encodings = self._encoding_cache[graph_hash]
            device = edge_index.device
            return {k: v.to(device) for k, v in cached_encodings.items()}
        
        encodings = {}
        
        if self.transformer_type == "graphormer":
            # Shortest path distances
            encodings['spatial'] = self._compute_shortest_paths_bfs(edge_index, num_nodes)
            # Node degrees
            encodings['degree'] = self._compute_degrees(edge_index, num_nodes)
            
        elif self.transformer_type == "graphgps":
            # Laplacian eigenvectors
            encodings['pe'] = self._compute_laplacian_pe(edge_index, num_nodes)
        
        # Cache the encodings
        if graph_hash:
            self._encoding_cache[graph_hash] = encodings
            
        return encodings
    
    def _compute_shortest_paths_bfs(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute shortest path distances using BFS (no NetworkX dependency)."""
        if num_nodes > self.max_nodes:
            # For large graphs, use approximation
            return self._approximate_shortest_paths(edge_index, num_nodes)
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        edges = edge_index.t().cpu().numpy()
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)  # Undirected graph
        
        # Initialize distance matrix
        distances = torch.full((num_nodes, num_nodes), self.max_path_length, dtype=torch.long)
        
        # Set diagonal to 0
        for i in range(num_nodes):
            distances[i, i] = 0
        
        # BFS from each node to compute shortest paths
        for source in range(num_nodes):
            visited = [False] * num_nodes
            queue = deque([(source, 0)])
            visited[source] = True
            
            while queue:
                node, dist = queue.popleft()
                
                if dist >= self.max_path_length - 1:
                    continue
                
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        new_dist = dist + 1
                        distances[source, neighbor] = min(new_dist, self.max_path_length - 1)
                        queue.append((neighbor, new_dist))
        
        return distances.to(edge_index.device)
    
    def _approximate_shortest_paths(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Approximate shortest paths for large graphs."""
        # Use random sampling of nodes for approximation
        sample_size = min(self.max_nodes, num_nodes)
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        
        # Create distance matrix for sampled nodes
        distances = torch.full((num_nodes, num_nodes), self.max_path_length, dtype=torch.long)
        
        # Set diagonal to 0
        distances.fill_diagonal_(0)
        
        # Set adjacent nodes to distance 1
        distances[edge_index[0], edge_index[1]] = 1
        distances[edge_index[1], edge_index[0]] = 1
        
        # For remaining pairs, use hop-based approximation
        # This is a simplified approximation - in practice you might want more sophisticated methods
        for hop in range(2, min(4, self.max_path_length)):
            # Find nodes at distance hop
            prev_dist_mask = (distances == hop - 1)
            for u in range(num_nodes):
                if hop > 3:  # Limit computation for very large graphs
                    break
                neighbors_at_prev_hop = torch.where(prev_dist_mask[u])[0]
                for neighbor in neighbors_at_prev_hop:
                    # Update distances through this neighbor
                    adj_mask = (distances[neighbor] == 1)
                    update_mask = (distances[u] > hop) & adj_mask
                    distances[u][update_mask] = hop
        
        return distances.to(edge_index.device)
    
    def _compute_degrees(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute node degrees."""
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        return deg.to(edge_index.device)
    
    def _compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute Laplacian positional encoding."""
        try:
            # Convert to dense adjacency matrix
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            
            # Compute degree matrix
            deg = adj.sum(dim=1)
            
            # Handle isolated nodes
            deg_inv_sqrt = torch.zeros_like(deg)
            non_zero_mask = deg > 0
            deg_inv_sqrt[non_zero_mask] = deg[non_zero_mask].pow(-0.5)
            
            # Compute normalized Laplacian: I - D^(-1/2) A D^(-1/2)
            deg_matrix = torch.diag(deg_inv_sqrt)
            normalized_adj = deg_matrix @ adj @ deg_matrix
            laplacian = torch.eye(num_nodes, device=adj.device) - normalized_adj
            
            # Compute eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
            
            # Sort by eigenvalues (ascending)
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Take smallest non-zero eigenvalues (skip the first one which should be ~0)
            start_idx = 1 if eigenvals[0] < 1e-6 else 0
            pe_dim = min(self.hidden_dim // 2, num_nodes - start_idx)
            
            if pe_dim > 0:
                pe = eigenvecs[:, start_idx:start_idx + pe_dim]
                
                # Pad if necessary
                if pe.size(1) < self.hidden_dim // 2:
                    padding = torch.zeros(num_nodes, self.hidden_dim // 2 - pe.size(1), device=pe.device)
                    pe = torch.cat([pe, padding], dim=1)
                elif pe.size(1) > self.hidden_dim // 2:
                    pe = pe[:, :self.hidden_dim // 2]
            else:
                pe = torch.zeros(num_nodes, self.hidden_dim // 2, device=edge_index.device)
                
        except Exception as e:
            # Fallback to random features if eigendecomposition fails
            print(f"Warning: Laplacian PE computation failed ({e}), using random features")
            pe = torch.randn(num_nodes, self.hidden_dim // 2, device=edge_index.device)
        
        return pe
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None):
        """Forward pass."""
        num_nodes = x.size(0)
        
        # Project input features
        h = self.input_proj(x)
        
        # Compute structural encodings if enabled
        if self.precompute_encodings:
            encodings = self._compute_structural_encodings(edge_index, num_nodes)
        else:
            encodings = None
        
        if self.transformer_type == "graphormer":
            h = self._forward_graphormer(h, edge_index, encodings)
        elif self.transformer_type == "graphgps":
            h = self._forward_graphgps(h, edge_index, encodings)
        
        # Output projection
        h = self.output_layer(h)
        
        if self.is_regression:
            h = self.regression_activation(h)
        
        return h
    
    def _forward_graphormer(self, h: torch.Tensor, edge_index: torch.Tensor, encodings: Dict):
        """Forward pass for Graphormer."""
        num_nodes = h.size(0)
        
        # Add positional encodings
        if encodings:
            if 'degree' in encodings:
                degree_emb = self.degree_encoder(encodings['degree'])
                h = h + degree_emb
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, encodings)
        
        h = self.layer_norm(h)
        return h
    
    def _forward_graphgps(self, h: torch.Tensor, edge_index: torch.Tensor, encodings: Dict):
        """Forward pass for GraphGPS."""
        # Add positional encoding
        if encodings and 'pe' in encodings:
            pe = encodings['pe']
            h = h + self.pe_encoder(pe)
        
        # Apply GraphGPS layers
        for layer in self.layers:
            h = layer(h, edge_index)
        
        return h


class SpatialEncoder(nn.Module):
    """Spatial encoding for Graphormer."""
    
    def __init__(self, hidden_dim: int, max_path_length: int):
        super().__init__()
        self.max_path_length = max_path_length
        self.spatial_embedding = nn.Embedding(max_path_length, hidden_dim)
    
    def forward(self, spatial_matrix: torch.Tensor) -> torch.Tensor:
        """Encode spatial relationships."""
        # spatial_matrix: [num_nodes, num_nodes]
        return self.spatial_embedding(spatial_matrix.clamp(0, self.max_path_length-1))


class DegreeEncoder(nn.Module):
    """Degree encoding for Graphormer."""
    
    def __init__(self, hidden_dim: int, max_degree: int = 100):
        super().__init__()
        self.max_degree = max_degree
        self.degree_embedding = nn.Embedding(max_degree, hidden_dim)
    
    def forward(self, degrees: torch.Tensor) -> torch.Tensor:
        """Encode node degrees."""
        return self.degree_embedding(degrees.clamp(0, self.max_degree-1))


class PositionalEncoder(nn.Module):
    """Positional encoding for GraphGPS."""
    
    def __init__(self, hidden_dim: int, pe_type: str = "laplacian"):
        super().__init__()
        self.pe_type = pe_type
        if pe_type == "laplacian":
            self.pe_proj = nn.Linear(hidden_dim // 2, hidden_dim)
    
    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        """Project positional encoding."""
        if self.pe_type == "laplacian":
            return self.pe_proj(pe)
        return pe


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
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, encodings: Dict = None):
        """Forward pass through Graphormer layer."""
        # Add batch dimension for attention
        h_input = h.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # Self-attention
        h_attn, _ = self.self_attn(h_input, h_input, h_input)
        h_attn = h_attn.squeeze(0)
        
        # Residual connection and norm
        h = self.norm1(h + h_attn)
        
        # FFN
        h_ffn = self.ffn(h)
        h = self.norm2(h + h_ffn)
        
        return h


class GraphGPSLayer(nn.Module):
    """GraphGPS layer combining local and global attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 local_gnn_type: str = "gcn", global_model_type: str = "transformer",
                 prenorm: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        
        # Local GNN
        if local_gnn_type == "gcn":
            from torch_geometric.nn import GCNConv
            self.local_gnn = GCNConv(hidden_dim, hidden_dim)
        elif local_gnn_type == "sage":
            from torch_geometric.nn import SAGEConv
            self.local_gnn = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown local GNN type: {local_gnn_type}")
        
        # Global attention
        if global_model_type == "transformer":
            self.global_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
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
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass through GraphGPS layer."""
        residual = h
        
        # Pre-norm if specified
        if self.prenorm:
            h = self.norm1(h)
        
        # Local message passing
        h_local = self.local_gnn(h, edge_index)
        h = residual + self.dropout(h_local)
        
        # Post-norm if not pre-norm
        if not self.prenorm:
            h = self.norm1(h)
        
        # Global attention
        residual = h
        if self.prenorm:
            h = self.norm2(h)
        
        h_input = h.unsqueeze(0)  # Add batch dimension
        h_global, _ = self.global_attn(h_input, h_input, h_input)
        h_global = h_global.squeeze(0)
        h = residual + self.dropout(h_global)
        
        if not self.prenorm:
            h = self.norm2(h)
        
        # FFN
        residual = h
        if self.prenorm:
            h = self.norm3(h)
        
        h_ffn = self.ffn(h)
        h = residual + self.dropout(h_ffn)
        
        if not self.prenorm:
            h = self.norm3(h)
        
        return h


def create_graph_transformer_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    transformer_type: str = "graphormer",
    **kwargs
) -> GraphTransformerModel:
    """Factory function to create Graph Transformer models."""
    return GraphTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        transformer_type=transformer_type,
        **kwargs
    )


def precompute_family_encodings(family_graphs: List, transformer_model: GraphTransformerModel):
    """Precompute encodings for an entire graph family."""
    print("Precomputing structural encodings for graph family...")
    
    for i, graph_sample in enumerate(family_graphs):
        if i % 10 == 0:
            print(f"Processing graph {i+1}/{len(family_graphs)}")
        
        # Convert to PyG format
        try:
            from mmsb.feature_regimes import graphsample_to_pyg
            pyg_data = graphsample_to_pyg(graph_sample)
            
            # Precompute encodings
            transformer_model._compute_structural_encodings(
                pyg_data.edge_index, 
                pyg_data.x.size(0)
            )
        except Exception as e:
            print(f"Warning: Failed to precompute encodings for graph {i}: {e}")
            continue
    
    cache_size = len(transformer_model._encoding_cache) if transformer_model._encoding_cache else 0
    print(f"Precomputed encodings for {len(family_graphs)} graphs")
    print(f"Cache size: {cache_size}")


class GraphTransformerCache:
    """Cache manager for Graph Transformer encodings."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get(self, key: str):
        """Get cached encoding."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value):
        """Cache encoding with LRU eviction."""
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


# Additional utility functions for better integration

def get_transformer_model_info(model: GraphTransformerModel) -> Dict[str, Any]:
    """Get information about a transformer model."""
    info = {
        'transformer_type': model.transformer_type,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'max_nodes': model.max_nodes,
        'max_path_length': model.max_path_length,
        'precompute_encodings': model.precompute_encodings,
        'cache_encodings': model.cache_encodings,
        'cache_size': len(model._encoding_cache) if model._encoding_cache else 0
    }
    return info


def benchmark_transformer_performance(
    model: GraphTransformerModel, 
    sample_graphs: List, 
    device: torch.device
) -> Dict[str, float]:
    """Benchmark transformer performance on sample graphs."""
    import time
    
    model.eval()
    model.to(device)
    
    times = {
        'encoding_time': [],
        'forward_time': [],
        'total_time': []
    }
    
    with torch.no_grad():
        for graph_sample in sample_graphs:
            try:
                from mmsb.feature_regimes import graphsample_to_pyg
                pyg_data = graphsample_to_pyg(graph_sample)
                pyg_data = pyg_data.to(device)
                
                # Time encoding computation
                start_time = time.time()
                encodings = model._compute_structural_encodings(
                    pyg_data.edge_index, 
                    pyg_data.x.size(0)
                )
                encoding_time = time.time() - start_time
                
                # Time forward pass
                start_time = time.time()
                output = model(pyg_data.x, pyg_data.edge_index)
                forward_time = time.time() - start_time
                
                times['encoding_time'].append(encoding_time)
                times['forward_time'].append(forward_time)
                times['total_time'].append(encoding_time + forward_time)
                
            except Exception as e:
                print(f"Warning: Benchmark failed for a graph: {e}")
                continue
    
    # Calculate statistics
    stats = {}
    for key, time_list in times.items():
        if time_list:
            stats[f'{key}_mean'] = np.mean(time_list)
            stats[f'{key}_std'] = np.std(time_list)
            stats[f'{key}_min'] = np.min(time_list)
            stats[f'{key}_max'] = np.max(time_list)
        else:
            stats[f'{key}_mean'] = 0.0
            stats[f'{key}_std'] = 0.0
            stats[f'{key}_min'] = 0.0
            stats[f'{key}_max'] = 0.0
    
    return stats


def optimize_transformer_cache_size(
    family_graphs: List, 
    transformer_type: str = "graphormer",
    max_cache_sizes: List[int] = [100, 500, 1000, 2000]
) -> int:
    """Find optimal cache size for a graph family."""
    best_cache_size = 100
    best_hit_rate = 0.0
    
    for cache_size in max_cache_sizes:
        # Create temporary model with specific cache size
        temp_model = GraphTransformerModel(
            input_dim=32,  # Placeholder
            hidden_dim=64,  # Placeholder
            output_dim=10,  # Placeholder
            transformer_type=transformer_type,
            cache_encodings=True
        )
        temp_model._encoding_cache = {}
        
        hits = 0
        total = 0
        
        # Simulate cache usage
        for graph_sample in family_graphs:
            try:
                from mmsb.feature_regimes import graphsample_to_pyg
                pyg_data = graphsample_to_pyg(graph_sample)
                
                graph_hash = temp_model._get_graph_hash(
                    pyg_data.edge_index, 
                    pyg_data.x.size(0)
                )
                
                if graph_hash:
                    total += 1
                    if graph_hash in temp_model._encoding_cache:
                        hits += 1
                    else:
                        # Simulate adding to cache
                        if len(temp_model._encoding_cache) < cache_size:
                            temp_model._encoding_cache[graph_hash] = True  # Placeholder
                
            except Exception:
                continue
        
        hit_rate = hits / total if total > 0 else 0.0
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_cache_size = cache_size
    
    print(f"Optimal cache size: {best_cache_size} (hit rate: {best_hit_rate:.2%})")
    return best_cache_size


def create_transformer_config_recommendations(
    family_graphs: List,
    max_nodes_in_family: int,
    avg_nodes_in_family: float
) -> Dict[str, Any]:
    """Create recommended configuration for transformers based on graph family."""
    
    recommendations = {
        'transformer_types': [],
        'config_params': {},
        'optimization_suggestions': []
    }
    
    # Choose transformer types based on graph size
    if max_nodes_in_family <= 100:
        recommendations['transformer_types'] = ['graphormer', 'graphgps']
        recommendations['optimization_suggestions'].append("Both transformers suitable for small graphs")
    elif max_nodes_in_family <= 300:
        recommendations['transformer_types'] = ['graphgps']
        recommendations['optimization_suggestions'].append("GraphGPS preferred for medium graphs")
    else:
        recommendations['transformer_types'] = ['graphgps']
        recommendations['optimization_suggestions'].append("Use approximation for large graphs")
    
    # Suggest configuration parameters
    if avg_nodes_in_family <= 50:
        recommendations['config_params'] = {
            'max_nodes': 200,
            'max_path_length': 15,
            'num_heads': 8,
            'num_layers': 4,
            'precompute_encodings': True,
            'cache_encodings': True
        }
    elif avg_nodes_in_family <= 150:
        recommendations['config_params'] = {
            'max_nodes': 300,
            'max_path_length': 10,
            'num_heads': 6,
            'num_layers': 3,
            'precompute_encodings': True,
            'cache_encodings': True
        }
    else:
        recommendations['config_params'] = {
            'max_nodes': 500,
            'max_path_length': 8,
            'num_heads': 4,
            'num_layers': 3,
            'precompute_encodings': False,  # Too expensive for large graphs
            'cache_encodings': False
        }
        recommendations['optimization_suggestions'].append("Consider disabling precomputation for large graphs")
    
    # Memory optimization suggestions
    if len(family_graphs) > 100:
        recommendations['optimization_suggestions'].append("Use LRU cache with limited size")
    
    if max_nodes_in_family > 200:
        recommendations['optimization_suggestions'].append("Enable approximation algorithms")
    
    return recommendations

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