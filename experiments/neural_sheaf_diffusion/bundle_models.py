import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree, add_self_loops
import math
from typing import Optional, Union, Tuple
import time


class BundleMapLearner(nn.Module):
    """Learn orthogonal maps for flat vector bundles."""
    
    def __init__(self, input_dim: int, d: int, method: str = "rotation"):
        super().__init__()
        self.d = d
        self.method = method
        
        if method == "rotation" and d == 2:
            # Direct parameterization for 2D rotations
            self.angle_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1)
            )
        elif method == "householder":
            # Householder reflections for general d
            # Number of parameters needed: d(d-1)/2
            num_params = d * (d - 1) // 2
            self.param_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, num_params)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [n, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            O_maps: Orthogonal matrices [n, d, d]
        """
        n = x.size(0)
        
        if self.method == "rotation" and self.d == 2:
            angles = self.angle_mlp(x).squeeze(-1)  # [n]
            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)
            
            # Create 2D rotation matrices
            O_maps = torch.zeros(n, 2, 2, device=x.device, dtype=x.dtype)
            O_maps[:, 0, 0] = cos_theta
            O_maps[:, 0, 1] = -sin_theta
            O_maps[:, 1, 0] = sin_theta
            O_maps[:, 1, 1] = cos_theta
            
            return O_maps
            
        elif self.method == "householder":
            params = self.param_mlp(x)  # [n, num_params]
            return self._householder_to_orthogonal(params)
        
    def _householder_to_orthogonal(self, params: torch.Tensor) -> torch.Tensor:
        """Convert Householder parameters to orthogonal matrices."""
        n, _ = params.shape
        device, dtype = params.device, params.dtype
        
        # Initialize as identity matrices
        O_maps = torch.eye(self.d, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
        
        # Apply Householder reflections
        param_idx = 0
        for i in range(self.d - 1):
            # Get parameters for this reflection
            v_params = params[:, param_idx:param_idx + (self.d - i)]
            param_idx += (self.d - i)
            
            # Create Householder vector
            v = torch.zeros(n, self.d, device=device, dtype=dtype)
            v[:, i:] = v_params
            
            # Normalize
            v_norm = torch.norm(v, dim=1, keepdim=True)
            v_norm = torch.where(v_norm > 1e-8, v_norm, torch.ones_like(v_norm))
            v = v / v_norm
            
            # Apply Householder reflection: H = I - 2*v*v^T
            vvT = torch.bmm(v.unsqueeze(-1), v.unsqueeze(1))
            H = torch.eye(self.d, device=device, dtype=dtype).unsqueeze(0) - 2 * vvT
            
            # Update orthogonal matrices
            O_maps = torch.bmm(H, O_maps)
        
        return O_maps


class GraphHeatKernel(nn.Module):
    """Efficient computation of graph heat kernel."""
    
    def __init__(self, method: str = "taylor", max_degree: int = 10):
        super().__init__()
        self.method = method
        self.max_degree = max_degree
        self._cached_eigendecomposition = None
        
    def forward(self, edge_index: torch.Tensor, num_nodes: int, t: float) -> torch.Tensor:
        """
        Compute graph heat kernel H(t) = exp(-t * L).
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            t: Diffusion time
            
        Returns:
            heat_kernel: [num_nodes, num_nodes]
        """
        if t == 0:
            return torch.eye(num_nodes, device=edge_index.device)
            
        if self.method == "taylor":
            return self._taylor_approximation(edge_index, num_nodes, t)
        elif self.method == "spectral":
            return self._spectral_method(edge_index, num_nodes, t)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_normalized_laplacian(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute normalized graph Laplacian."""
        device = edge_index.device
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute degree
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[row, col] = 1.0
        
        # Normalize: L = I - D^(-1/2) A D^(-1/2)
        deg_matrix = torch.diag(deg_inv_sqrt)
        normalized_adj = torch.mm(torch.mm(deg_matrix, adj), deg_matrix)
        laplacian = torch.eye(num_nodes, device=device) - normalized_adj
        
        return laplacian
    
    def _taylor_approximation(self, edge_index: torch.Tensor, num_nodes: int, t: float) -> torch.Tensor:
        """Taylor series approximation of matrix exponential."""
        L = self._get_normalized_laplacian(edge_index, num_nodes)
        device = L.device
        
        # H(t) = exp(-tL) ≈ sum_{k=0}^K (-tL)^k / k!
        result = torch.eye(num_nodes, device=device)
        term = torch.eye(num_nodes, device=device)
        
        for k in range(1, self.max_degree + 1):
            term = -t * torch.mm(L, term) / k
            result = result + term
            
            # Early stopping if term becomes negligible
            if torch.norm(term) < 1e-8:
                break
                
        return result
    
    def _spectral_method(self, edge_index: torch.Tensor, num_nodes: int, t: float) -> torch.Tensor:
        """Spectral method using eigendecomposition."""
        L = self._get_normalized_laplacian(edge_index, num_nodes)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # H(t) = V * exp(-t * Lambda) * V^T
        exp_eigenvalues = torch.exp(-t * eigenvalues)
        heat_kernel = torch.mm(eigenvectors * exp_eigenvalues.unsqueeze(0), eigenvectors.t())
        
        return heat_kernel


class BuNNLayer(nn.Module):
    """Single Bundle Neural Network layer."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d: int = 2,
        num_bundles: int = 1,
        bundle_method: str = "rotation",
        heat_method: str = "taylor",
        max_degree: int = 10,
        use_bias: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d = d
        self.num_bundles = num_bundles
        self.total_dim = d * num_bundles
        
        # Bundle map learner
        self.bundle_learner = BundleMapLearner(
            input_dim=input_dim,
            d=d,
            method=bundle_method
        )
        
        # Heat kernel computer
        self.heat_kernel = GraphHeatKernel(method=heat_method, max_degree=max_degree)
        
        # Learnable parameters
        self.W = nn.Parameter(torch.randn(self.total_dim, self.total_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.total_dim))
        else:
            self.register_parameter('bias', None)
            
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.orthogonal_(self.W)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        t: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass of BuNN layer.
        
        Args:
            x: Node features [n, input_dim]
            edge_index: Edge indices [2, num_edges]
            t: Diffusion time
            
        Returns:
            Updated node features [n, total_dim]
        """
        n = x.size(0)
        device = x.device
        
        # Step 1: Learn bundle maps
        O_maps = self.bundle_learner(x, edge_index)  # [n, d, d]
        
        # Handle multiple bundles by creating block diagonal structure
        if self.num_bundles > 1:
            # Create block diagonal matrix: each bundle gets its own d x d block
            O_block_diag = torch.zeros(n, self.total_dim, self.total_dim, device=device)
            for i in range(self.num_bundles):
                start_idx = i * self.d
                end_idx = (i + 1) * self.d
                O_block_diag[:, start_idx:end_idx, start_idx:end_idx] = O_maps
            O_maps = O_block_diag
        else:
            # Expand to total_dim if needed
            if self.total_dim > self.d:
                O_expanded = torch.eye(self.total_dim, device=device).unsqueeze(0).repeat(n, 1, 1)
                O_expanded[:, :self.d, :self.d] = O_maps
                O_maps = O_expanded
        
        # Step 2: Encoder - transform to bundle space
        if x.size(1) != self.total_dim:
            # Project input to correct dimension
            x_expanded = torch.zeros(n, self.total_dim, device=device)
            x_expanded[:, :min(x.size(1), self.total_dim)] = x[:, :min(x.size(1), self.total_dim)]
            x = x_expanded
            
        # h_v = O_v^T * W * O_v * x_v + b
        Ox = torch.bmm(O_maps, x.unsqueeze(-1)).squeeze(-1)  # Synchronize
        Wx = torch.mm(Ox, self.W.t())  # Apply weights
        h = torch.bmm(O_maps.transpose(-1, -2), Wx.unsqueeze(-1)).squeeze(-1)  # Desynchronize
        
        if self.bias is not None:
            h = h + self.bias
        
        # Step 3: Message diffusion via heat kernel
        if t > 0:
            # Compute graph heat kernel
            H_graph = self.heat_kernel(edge_index, n, t)  # [n, n]
            
            # Apply flat bundle heat diffusion
            h = self._flat_bundle_diffusion(h, O_maps, H_graph)
        
        # Step 4: Apply activation
        h = self.activation(h)
        
        return h
    
    def _flat_bundle_diffusion(
        self, 
        h: torch.Tensor, 
        O_maps: torch.Tensor, 
        H_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply flat bundle heat diffusion.
        
        Based on Lemma 3.1: (H_B(t)X)_v = sum_u H(t,v,u) * O_v^T * O_u * x_u
        """
        n = h.size(0)
        
        # Synchronize: apply O_u to each node's features
        h_sync = torch.bmm(O_maps, h.unsqueeze(-1)).squeeze(-1)  # [n, total_dim]
        
        # Apply graph heat diffusion
        h_diffused = torch.mm(H_graph, h_sync)  # [n, total_dim]
        
        # Desynchronize: apply O_v^T to get back to local coordinates
        h_result = torch.bmm(O_maps.transpose(-1, -2), h_diffused.unsqueeze(-1)).squeeze(-1)
        
        return h_result


class BuNN(nn.Module):
    """Bundle Neural Network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        d: int = 2,
        num_bundles: int = 1,
        bundle_method: str = "rotation",
        heat_method: str = "taylor",
        max_degree: int = 10,
        diffusion_times: Optional[Union[float, list]] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        is_graph_level_task: bool = False,
        graph_pooling: str = "mean"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.is_graph_level_task = is_graph_level_task
        self.graph_pooling = graph_pooling
        self.total_dim = d * num_bundles
        
        # Handle diffusion times
        if diffusion_times is None:
            self.diffusion_times = [1.0] * num_layers
        elif isinstance(diffusion_times, (int, float)):
            self.diffusion_times = [float(diffusion_times)] * num_layers
        else:
            assert len(diffusion_times) == num_layers
            self.diffusion_times = list(diffusion_times)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.total_dim)
        
        # BuNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = self.total_dim if i == 0 else self.total_dim
            self.layers.append(
                BuNNLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    d=d,
                    num_bundles=num_bundles,
                    bundle_method=bundle_method,
                    heat_method=heat_method,
                    max_degree=max_degree,
                    activation=activation if i < num_layers - 1 else "identity"
                )
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output projection
        if is_graph_level_task:
            self.output_proj = nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.output_proj = nn.Linear(self.total_dim, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of BuNN.
        
        Args:
            x: Node features [n, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level tasks [n]
            
        Returns:
            Output predictions
        """
        # Input projection
        h = self.input_proj(x)
        h = self.dropout(h)
        
        # Apply BuNN layers
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, t=self.diffusion_times[i])
            if i < len(self.layers) - 1:  # Don't apply dropout after last layer
                h = self.dropout(h)
        
        # Graph-level pooling if needed
        if self.is_graph_level_task:
            if batch is None:
                # Single graph case
                if self.graph_pooling == "mean":
                    h = h.mean(dim=0, keepdim=True)
                elif self.graph_pooling == "sum":
                    h = h.sum(dim=0, keepdim=True)
                elif self.graph_pooling == "max":
                    h = h.max(dim=0, keepdim=True)[0]
                else:
                    raise ValueError(f"Unknown pooling: {self.graph_pooling}")
            else:
                # Batched graphs
                if self.graph_pooling == "mean":
                    h = global_mean_pool(h, batch)
                else:
                    raise NotImplementedError(f"Batched {self.graph_pooling} pooling not implemented")
        
        # Output projection
        output = self.output_proj(h)
        
        return output