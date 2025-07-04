import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Union, Tuple
import math
import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.metrics import accuracy_score
import time
from collections import defaultdict


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

class NeuralSheafDiffusion(nn.Module):
    """
    Multi-layer Neural Sheaf Diffusion Network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        d: int,  # stalk dimension
        num_layers: int = 2,
        restriction_map_type: str = "orthogonal",
        orthogonal_method: str = "householder",
        activation: str = "elu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d = d
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Initial projection to d * hidden_channels
        self.input_projection = nn.Linear(input_dim, d * hidden_channels)
        
        # Sheaf diffusion layers
        self.sheaf_layers = nn.ModuleList([
            NeuralSheafDiffusionLayer(
                d=d,
                hidden_channels=hidden_channels,
                restriction_map_type=restriction_map_type,
                orthogonal_method=orthogonal_method,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Output projection: Elu activation function and linear layer to output_dim
        self.output_projection = nn.Sequential(
            nn.ELU(),
            nn.Linear(d * hidden_channels, output_dim)
        )
        
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
        x = self.input_projection(x)  # [num_nodes, d * hidden_channels]
        x = self.dropout(x)
        
        # Apply sheaf diffusion layers
        for layer in self.sheaf_layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # Final projection to output
        x = self.output_projection(x)  # [num_nodes, output_dim]
        
        # Return sigmoid activation
        return torch.sigmoid(x)
    
    def get_timing_stats(self):
        """Get timing statistics from all sheaf layers."""
        all_stats = {}
        for i, layer in enumerate(self.sheaf_layers):
            layer_stats = layer.get_timing_stats()
            for key, value in layer_stats.items():
                all_stats[f'layer_{i}_{key}'] = value
        return all_stats

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for comparison.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(input_dim, hidden_channels)
        ])
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_channels, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output log probabilities [num_nodes, output_dim]
        """
        # GCN layers with normalization
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final projection
        x = self.output_projection(x)
        
        return torch.sigmoid(x)


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for comparison.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        aggr: str = "mean"  # "mean", "max", "sum"
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.aggr = aggr
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList([
            SAGEConv(input_dim, hidden_channels, aggr=aggr)
        ])
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_channels, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE.
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output log probabilities [num_nodes, output_dim]
        """
        # GraphSAGE layers
        for layer in self.sage_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final projection
        x = self.output_projection(x)
        
        return torch.sigmoid(x)



def graphsample_to_pyg(graph_sample):
    """
    Convert a GraphSample to a PyTorch Geometric Data object.
    Uses node features from the GraphSample if available, otherwise uses identity features.
    Labels are set to community_labels.
    """
    graph = graph_sample.graph
    n_nodes = graph.number_of_nodes()
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # Use features from GraphSample if available, else identity
    if hasattr(graph_sample, 'features') and graph_sample.features is not None:
        features = torch.tensor(graph_sample.features, dtype=torch.float)
    else:
        features = torch.eye(n_nodes, dtype=torch.float)

    # Use community_labels as y
    if hasattr(graph_sample, 'community_labels') and graph_sample.community_labels is not None:
        y = torch.tensor(graph_sample.community_labels, dtype=torch.long)
    else:
        y = torch.zeros(n_nodes, dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=y)
    return data

def load_graphs():
    """Load graphs from graphs.pkl file and convert to PyG format."""
    print("Loading graphs from graphs.pkl...")
    
    with open('graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError("No graphs found in graphs.pkl")
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Convert all graphs to PyG format
    pyg_graphs = []
    for i, graph_sample in enumerate(graphs):
        pyg_data = graphsample_to_pyg(graph_sample)
        pyg_graphs.append(pyg_data)
        if i < 5:  # Print info for first 5 graphs
            print(f"Graph {i}: {pyg_data.x.size(0)} nodes, {pyg_data.x.size(1)} features, {len(torch.unique(pyg_data.y))} classes")
    
    return pyg_graphs

def prepare_inductive_data(graphs, num_train=5, num_val=5, num_test=5, seed=42):
    """Prepare data for inductive learning with separate train/val/test graphs."""
    print(f"Preparing inductive data: {num_train} train, {num_val} val, {num_test} test graphs")
    
    if len(graphs) < num_train + num_val + num_test:
        raise ValueError(f"Not enough graphs. Need {num_train + num_val + num_test}, have {len(graphs)}")
    
    # Set random seed for reproducible splits
    np.random.seed(seed)
    indices = np.random.permutation(len(graphs))
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]
    
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Get number of classes from all graphs
    all_classes = set()
    for graph in graphs:
        all_classes.update(graph.y.numpy())
    num_classes = len(all_classes)
    
    print(f"Total number of classes across all graphs: {num_classes}")
    
    return {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'num_classes': num_classes
    }

def train_model(model, data, device, model_name, epochs=10, lr=0.01, weight_decay=5e-4):
    """Generic training function for any graph neural network model."""
    print(f"Training {model_name} for {epochs} epochs...")
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(data['train_graphs'], batch_size=5, shuffle=True)
    val_loader = DataLoader(data['val_graphs'], batch_size=15, shuffle=False)
    test_loader = DataLoader(data['test_graphs'], batch_size=15, shuffle=False)
    
    # Check if parameters were initialized
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} initialized with {param_count} parameters")
    
    if param_count == 0:
        raise ValueError(f"{model_name} has no parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize timing statistics
    timing_stats = {
        'forward_pass': 0.0,
        'backward_pass': 0.0,
        'optimizer_step': 0.0,
        'data_loading': 0.0,
        'loss_computation': 0.0,
        'accuracy_computation': 0.0,
        'total_training': 0.0
    }
    timing_counts = {
        'forward_pass': 0,
        'backward_pass': 0,
        'optimizer_step': 0,
        'data_loading': 0,
        'loss_computation': 0,
        'accuracy_computation': 0,
        'total_training': 0
    }
    
    # Global debug flag to compare models
    global debug_outputs
    if 'debug_outputs' not in globals():
        debug_outputs = {}
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_nodes = 0
        
        # Training
        for batch in train_loader:
            # Time data loading
            start_time = time.time()
            batch = batch.to(device)
            timing_stats['data_loading'] += time.time() - start_time
            timing_counts['data_loading'] += 1
            
            optimizer.zero_grad()
            
            # Time forward pass
            start_time = time.time()
            out = model(batch.x, batch.edge_index)
            timing_stats['forward_pass'] += time.time() - start_time
            timing_counts['forward_pass'] += 1
            
            # Time loss computation
            start_time = time.time()
            loss = criterion(out, batch.y)
            timing_stats['loss_computation'] += time.time() - start_time
            timing_counts['loss_computation'] += 1
            
            # Time backward pass
            start_time = time.time()
            loss.backward()
            timing_stats['backward_pass'] += time.time() - start_time
            timing_counts['backward_pass'] += 1
            
            # Time optimizer step
            start_time = time.time()
            optimizer.step()
            timing_stats['optimizer_step'] += time.time() - start_time
            timing_counts['optimizer_step'] += 1
            
            # Time accuracy computation
            start_time = time.time()
            
            # Debug tensor properties
            if timing_counts['accuracy_computation'] == 0:  # Only debug first batch
                print(f"\nDEBUG - {model_name} Output Tensor Properties:")
                print(f"  Shape: {out.shape}")
                print(f"  Device: {out.device}")
                print(f"  Dtype: {out.dtype}")
                print(f"  Requires grad: {out.requires_grad}")
                print(f"  Is contiguous: {out.is_contiguous()}")
                print(f"  Grad function: {out.grad_fn}")
                print(f"  Is leaf: {out.is_leaf}")
                
                # Store output tensor for comparison
                debug_outputs[model_name] = {
                    'tensor': out.detach().clone(),
                    'shape': out.shape,
                    'device': out.device,
                    'dtype': out.dtype,
                    'requires_grad': out.requires_grad,
                    'is_contiguous': out.is_contiguous(),
                    'grad_fn_type': type(out.grad_fn).__name__ if out.grad_fn else None
                }
                
                # Check if output is part of complex computational graph
                if out.grad_fn is not None:
                    print(f"  Grad function type: {type(out.grad_fn).__name__}")
                    # Count number of operations in computational graph
                    try:
                        op_count = 0
                        current = out.grad_fn
                        while current is not None:
                            op_count += 1
                            current = current.next_functions[0][0] if current.next_functions else None
                        print(f"  Operations in graph: {op_count}")
                        debug_outputs[model_name]['graph_ops'] = op_count
                    except Exception as e:
                        print(f"  Error counting graph operations: {e}")
                        debug_outputs[model_name]['graph_ops'] = -1
                
                # Check CUDA memory and synchronization
                if out.device.type == 'cuda':
                    try:
                        print(f"  CUDA memory allocated: {torch.cuda.memory_allocated(out.device) / 1024**2:.2f} MB")
                        print(f"  CUDA memory cached: {torch.cuda.memory_reserved(out.device) / 1024**2:.2f} MB")
                        
                        # Force CUDA synchronization to see if that's the issue
                        torch.cuda.synchronize(out.device)
                        print(f"  CUDA synchronized")
                    except Exception as e:
                        print(f"  Error checking CUDA memory: {e}")
                
                # Check if tensor needs to be made contiguous
                if not out.is_contiguous():
                    print(f"  WARNING: Tensor is not contiguous!")
                    step_start = time.time()
                    out_contiguous = out.contiguous()
                    contiguous_time = time.time() - step_start
                    print(f"  Making contiguous time: {contiguous_time:.6f}s")
            
            with torch.no_grad():
                # Debug each step of accuracy computation
                if timing_counts['accuracy_computation'] == 0:  # Only debug first batch
                    print(f"\nDEBUG - {model_name} Accuracy Computation Breakdown:")
                    
                    # Step 1: argmax (with explicit detach)
                    step_start = time.time()
                    pred = out.detach().argmax(dim=1)
                    argmax_time = time.time() - step_start
                    print(f"  argmax time: {argmax_time:.6f}s")
                    print(f"  pred shape: {pred.shape}")
                    print(f"  pred device: {pred.device}")
                    print(f"  pred requires grad: {pred.requires_grad}")
                    
                    # Step 2: comparison
                    step_start = time.time()
                    comparison = (pred == batch.y)
                    comparison_time = time.time() - step_start
                    print(f"  comparison time: {comparison_time:.6f}s")
                    print(f"  comparison shape: {comparison.shape}")
                    
                    # Step 3: sum
                    step_start = time.time()
                    correct_sum = comparison.sum()
                    sum_time = time.time() - step_start
                    print(f"  sum time: {sum_time:.6f}s")
                    print(f"  correct_sum: {correct_sum}")
                    
                    # Step 4: item conversion
                    step_start = time.time()
                    correct_item = correct_sum.item()
                    item_time = time.time() - step_start
                    print(f"  item conversion time: {item_time:.6f}s")
                    print(f"  correct_item: {correct_item}")
                    
                    print(f"  Total debug time: {argmax_time + comparison_time + sum_time + item_time:.6f}s")
                else:
                    # Normal computation for other batches (with explicit detach)
                    pred = out.detach().argmax(dim=1)
                    total_train_correct += (pred == batch.y).sum().item()
            
            total_train_nodes += batch.y.size(0)
            total_loss += loss.item()
            timing_stats['accuracy_computation'] += time.time() - start_time
            timing_counts['accuracy_computation'] += 1
        
        # Validation
        model.eval()
        total_val_correct = 0
        total_val_nodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                pred = out.argmax(dim=1)
                total_val_correct += (pred == batch.y).sum().item()
                total_val_nodes += batch.y.size(0)
        
        train_acc = total_train_correct / total_train_nodes if total_train_nodes > 0 else 0
        val_acc = total_val_correct / total_val_nodes if total_val_nodes > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    total_test_correct = 0
    total_test_nodes = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            total_test_correct += (pred == batch.y).sum().item()
            total_test_nodes += batch.y.size(0)
    
    final_test_acc = total_test_correct / total_test_nodes if total_test_nodes > 0 else 0
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    
    # Print training timing statistics
    print(f"\nTraining Timing Breakdown for {model_name}:")
    print("-" * 50)
    for key in timing_stats:
        if timing_counts[key] > 0:
            avg_time = timing_stats[key] / timing_counts[key]
            print(f"{key}:")
            print(f"  Total time: {timing_stats[key]:.4f}s")
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Count: {timing_counts[key]}")
    print("-" * 50)
    
    # Collect timing statistics for sheaf models
    model_timing_stats = None
    if hasattr(model, 'get_timing_stats'):
        model_timing_stats = model.get_timing_stats()
        if model_timing_stats:
            print(f"\nModel Component Timing for {model_name}:")
            print("-" * 50)
            for key, stats in model_timing_stats.items():
                print(f"{key}:")
                print(f"  Total time: {stats['total_time']:.4f}s")
                print(f"  Average time: {stats['avg_time']:.6f}s")
                print(f"  Count: {stats['count']}")
            print("-" * 50)
    
    return {
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': final_test_acc,
        'model': model,
        'model_timing_stats': model_timing_stats,
        'training_timing_stats': timing_stats
    }

def test_all_models():
    """Main function to test and compare different graph neural network models."""
    print("=" * 60)
    print("COMPARISON OF GRAPH NEURAL NETWORK MODELS")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load graphs from pickle
        graphs = load_graphs()
        
        # Prepare inductive data
        data = prepare_inductive_data(graphs, num_train=20, num_val=15, num_test=15)
        
        # Get model parameters from first graph
        first_graph = data['train_graphs'][0]
        input_dim = first_graph.x.size(1)
        hidden_channels = 64
        output_dim = data['num_classes']
        
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of classes: {data['num_classes']}")
        
        # Define all models to test
        model_configs = [
            # Sheaf models
            {
                'name': 'General Sheaf (d=3)',
                'model_class': NeuralSheafDiffusion,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'd': 3,
                    'num_layers': 3,
                    'restriction_map_type': 'general',
                    'orthogonal_method': 'householder',
                    'activation': "elu",
                    'dropout': 0.3
                }
            },
            {
                'name': 'Diagonal Sheaf (d=3)',
                'model_class': NeuralSheafDiffusion,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'd': 3,
                    'num_layers': 3,
                    'restriction_map_type': 'diagonal',
                    'orthogonal_method': 'householder',
                    'activation': "elu",
                    'dropout': 0.3
                }
            },
            {
                'name': 'Orthogonal Sheaf (d=3)',
                'model_class': NeuralSheafDiffusion,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'd': 3,
                    'num_layers': 3,
                    'restriction_map_type': 'orthogonal',
                    'orthogonal_method': 'euler',
                    'activation': "elu",
                    'dropout': 0.3
                }
            },
            # Baseline models
            {
                'name': 'GCN',
                'model_class': GCN,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            {
                'name': 'GraphSAGE (mean)',
                'model_class': GraphSAGE,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'aggr': 'mean'
                }
            },
            {
                'name': 'GraphSAGE (max)',
                'model_class': GraphSAGE,
                'params': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'output_dim': output_dim,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'aggr': 'max'
                }
            }
        ]
        
        results = {}
        
        for config in model_configs:
            print(f"\n" + "="*50)
            print(f"Testing {config['name']}")
            print("="*50)
            
            try:
                # Create model
                model = config['model_class'](**config['params'])
                
                # Train model
                start_time = time.time()
                result = train_model(model, data, device, config['name'], epochs=50)
                training_time = time.time() - start_time
                
                result['training_time'] = training_time
                result['config'] = config
                results[config['name']] = result
                
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        print(f"\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"{name}: ERROR - {result['error']}")
            else:
                print(f"{name}:")
                print(f"  Train Acc: {result['train_acc']:.4f}")
                print(f"  Val Acc: {result['val_acc']:.4f}")
                print(f"  Test Acc: {result['test_acc']:.4f}")
                print(f"  Training Time: {result['training_time']:.2f}s")
        
        # Print training timing analysis
        print(f"\n" + "="*60)
        print("TRAINING TIMING ANALYSIS")
        print("="*60)
        
        for name, result in results.items():
            if 'training_timing_stats' in result and result['training_timing_stats']:
                print(f"\n{name} - Training Timing Breakdown:")
                print("-" * 40)
                
                # Sort by total time
                sorted_timing = sorted(result['training_timing_stats'].items(), key=lambda x: x[1], reverse=True)
                
                for component, total_time in sorted_timing:
                    print(f"  {component}: {total_time:.4f}s")
        
        # Print model component timing analysis for sheaf models
        print(f"\n" + "="*60)
        print("SHEAF MODEL COMPONENT TIMING ANALYSIS")
        print("="*60)
        
        for name, result in results.items():
            if 'model_timing_stats' in result and result['model_timing_stats']:
                print(f"\n{name} - Model Component Timing:")
                print("-" * 40)
                
                # Group by component type
                component_times = defaultdict(float)
                component_counts = defaultdict(int)
                
                for key, stats in result['model_timing_stats'].items():
                    # Extract component name (e.g., 'message_matrix_ops' from 'layer_0_message_matrix_ops')
                    component = key.split('_', 2)[-1] if '_' in key else key
                    component_times[component] += stats['total_time']
                    component_counts[component] += stats['count']
                
                # Sort by total time
                sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
                
                for component, total_time in sorted_components:
                    avg_time = total_time / component_counts[component] if component_counts[component] > 0 else 0
                    print(f"  {component}:")
                    print(f"    Total: {total_time:.4f}s")
                    print(f"    Average: {avg_time:.6f}s")
                    print(f"    Count: {component_counts[component]}")
        
        # Print debug comparison
        if 'debug_outputs' in globals() and debug_outputs:
            print(f"\n" + "="*60)
            print("DEBUG OUTPUT TENSOR COMPARISON")
            print("="*60)
            
            for model_name, debug_info in debug_outputs.items():
                print(f"\n{model_name}:")
                print(f"  Shape: {debug_info['shape']}")
                print(f"  Device: {debug_info['device']}")
                print(f"  Dtype: {debug_info['dtype']}")
                print(f"  Requires grad: {debug_info['requires_grad']}")
                print(f"  Is contiguous: {debug_info['is_contiguous']}")
                print(f"  Grad function type: {debug_info['grad_fn_type']}")
                if 'graph_ops' in debug_info:
                    print(f"  Graph operations: {debug_info['graph_ops']}")
                
                # Test argmax speed on stored tensor
                tensor = debug_info['tensor']
                start_time = time.time()
                for _ in range(100):  # Test 100 times for better timing
                    pred = tensor.argmax(dim=1)
                argmax_time = (time.time() - start_time) / 100
                print(f"  Argmax time (100x avg): {argmax_time:.6f}s")
        
        print(f"\nAll models test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

# Run the comprehensive test if this script is executed directly
if __name__ == "__main__":
    # First run the basic functionality test
    print("Running basic functionality test...")
    
    # Test parameters
    num_nodes = 100
    input_dim = 16
    hidden_channels = 32
    output_dim = 7  # number of classes
    d = 3  # stalk dimension
    num_edges = 200
    
    # Create random data with proper edge indices (avoid out-of-bounds)
    x = torch.randn(num_nodes, input_dim)
    # Ensure edge indices are within valid range [0, num_nodes)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
    
    # Create model
    model = NeuralSheafDiffusion(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        output_dim=output_dim,
        d=d,
        num_layers=3,
        restriction_map_type="orthogonal",  # Try "diagonal", "general"
        activation="elu",
        dropout=0.1
    )
    
    # Forward pass
    print(f"\nRunning forward pass...")
    output = model(x, edge_index)
    print(f"Output shape: {output.shape}")
    
    # Test different restriction map types
    for map_type in ["orthogonal", "diagonal", "general"]:
        print(f"\nTesting {map_type} restriction maps:")
        try:
            test_model = NeuralSheafDiffusion(
                input_dim=input_dim,
                hidden_channels=hidden_channels,
                output_dim=output_dim,
                d=d,
                num_layers=2,
                restriction_map_type=map_type,
                activation="elu"
            )
            
            test_output = test_model(x, edge_index)
            print(f"  ✓ Output shape: {test_output.shape}")
            print(f"  ✓ Number of parameters: {sum(p.numel() for p in test_model.parameters())}")
            
        except Exception as e:
            print(f"  ✗ Error with {map_type}: {e}")
    
    print("\n" + "="*50)
    print("Basic functionality test completed!")
    print("="*50)
    
    # Now run the comprehensive test on real graphs
    print("\n" + "="*50)
    print("Starting comprehensive test on real graphs...")
    print("="*50)
    
    test_all_models()