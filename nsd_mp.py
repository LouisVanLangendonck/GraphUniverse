import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool
from collections import defaultdict
import time
from typing import Optional
import math

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
        message = torch.einsum('eji,ejf->eif', F_ue, diff)
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
        is_graph_level_task: bool = False,
        pe_type: str = None,
        pe_dim: int = 16
    ):
        super().__init__()
        
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        
        # PE configuration
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        
        # Adjust input dimension if PE is used
        self.actual_input_dim = input_dim
        if pe_type is not None:
            self.actual_input_dim = input_dim + pe_dim
        
        # Create encoder
        self.encoder = NeuralSheafDiffusionEncoder(
            input_dim=self.actual_input_dim,  # Use adjusted input dimension
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
    
    def _get_pe_from_data(self, data):
        """Extract positional encoding from data object."""
        if self.pe_type is None:
            return None
            
        pe_attr_name = f"{self.pe_type}_pe"
        if hasattr(data, pe_attr_name):
            return getattr(data, pe_attr_name)
        
        print(f"Warning: PE type '{self.pe_type}' not found in data")
        return None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass through full model."""
        # Get PE and concatenate with input features if specified
        if self.pe_type is not None:
            # Get PE from the graph object
            graph = kwargs.get('graph', None)
            if graph is None:
                # Try to get from batch if graph not passed
                graph = kwargs.get('batch', None)
            
            pe = self._get_pe_from_data(graph)
            if pe is not None:
                # Ensure PE is on the same device and has correct shape
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
                
                # Concatenate input features with PE
                x = torch.cat([x, pe], dim=-1)
        
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index)  # [num_nodes, d * hidden_dim]
        
        # Apply prediction head
        if self.is_graph_level_task:
            # For graph-level tasks, we need to aggregate node features
            graph = kwargs.get('graph', None)
            # Get 'batch' from graph if it exists
            if graph is not None:
                batch = graph.batch
            else:
                raise ValueError("Graph not found in kwargs")
            # Use global mean pooling to get graph-level features
            h = global_mean_pool(h, batch)

        # Apply prediction head
        if self.is_graph_level_task:
            out = self.readout(h).squeeze(-1)  # Shape: [batch_size, output_dim]
        else:
            out = self.readout(h)  # Shape: [batch_size, output_dim]
        
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
