import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree, add_self_loops, get_laplacian, to_dense_adj
import math
from typing import Optional, Union, Tuple
import time
from torch_geometric.nn import SAGEConv

class InductiveBundleMapLearner(nn.Module):
    """Inductive version: Learn orthogonal maps for flat vector bundles."""
    
    def __init__(self, input_dim: int, d: int, method: str = "rotation", use_graph_structure: bool = True):
        super().__init__()
        self.d = d
        self.method = method
        self.use_graph_structure = use_graph_structure
        
        
        if use_graph_structure:
            # Use GraphSAGE-like architecture as mentioned in paper
            self.graph_conv1 = SAGEConv(input_dim, input_dim // 2)
            self.graph_conv2 = SAGEConv(input_dim // 2, input_dim // 4)
            final_dim = input_dim // 4
        else:
            # Fallback to MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4)
            )
            final_dim = input_dim // 4
        
        if method == "rotation" and d == 2:
            self.angle_head = nn.Linear(final_dim, 1)
        elif method == "householder":
            num_params = d * (d - 1) // 2
            self.param_head = nn.Linear(final_dim, num_params)
        elif method == "cayley":
            num_params = d * (d - 1) // 2
            self.param_head = nn.Linear(final_dim, num_params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [n, input_dim] - n can vary between graphs
            edge_index: Edge indices [2, num_edges] - can vary between graphs
        
        Returns:
            O_maps: Orthogonal matrices [n, d, d]
        """
        n = x.size(0)  # Dynamic graph size
        
        # Use graph structure or MLP
        if self.use_graph_structure:
            h = torch.relu(self.graph_conv1(x, edge_index))
            h = torch.relu(self.graph_conv2(h, edge_index))
        else:
            h = self.mlp(x)
        
        if self.method == "rotation" and self.d == 2:
            angles = self.angle_head(h).squeeze(-1)
            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)
            
            O_maps = torch.zeros(n, 2, 2, device=x.device, dtype=x.dtype)
            O_maps[:, 0, 0] = cos_theta
            O_maps[:, 0, 1] = -sin_theta
            O_maps[:, 1, 0] = sin_theta
            O_maps[:, 1, 1] = cos_theta
            
            return O_maps
        elif self.method == "householder":
            params = self.param_head(h)
            return self._householder_to_orthogonal(params)
        elif self.method == "cayley":
            params = self.param_head(h)
            return self._cayley_to_orthogonal(params)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
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

    @staticmethod
    def _cayley_to_orthogonal(params: torch.Tensor) -> torch.Tensor:
        """Convert Cayley parameters to orthogonal matrices."""
        n, num_params = params.shape
        # Reconstruct skew-symmetric matrix A from params
        d = int((1 + (1 + 8 * num_params) ** 0.5) // 2)
        device, dtype = params.device, params.dtype
        A = torch.zeros(n, d, d, device=device, dtype=dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                A[:, i, j] = params[:, idx]
                A[:, j, i] = -params[:, idx]
                idx += 1
        I = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(n, d, d)
        # Cayley transform: O = (I + A) @ (I - A)^{-1}
        O = torch.linalg.solve(I - A, I + A)
        return O


class InductiveGraphHeatKernel(nn.Module):
    """Inductive version: Efficient computation of graph heat kernel."""
    
    def __init__(self, method: str = "taylor", max_degree: int = 10):
        super().__init__()
        self.method = method
        self.max_degree = max_degree
        # Remove cached eigendecomposition since graphs can change
        
    def forward(self, edge_index: torch.Tensor, num_nodes: int, t: float) -> torch.Tensor:
        """
        Compute graph heat kernel H(t) = exp(-t * L).
        
        Args:
            edge_index: Edge indices [2, num_edges] - can vary between graphs
            num_nodes: Number of nodes - can vary between graphs
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


class InductiveBuNNLayer(nn.Module):
    """Inductive version: Single Bundle Neural Network layer."""
    
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
        self.bundle_learner = InductiveBundleMapLearner(
            input_dim=input_dim,
            d=d,
            method=bundle_method
        )
        
        # Heat kernel computer
        self.heat_kernel = InductiveGraphHeatKernel(method=heat_method, max_degree=max_degree)
        
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
        Forward pass of InductiveBuNN layer.
        
        Args:
            x: Node features [n, input_dim] - n can vary between graphs
            edge_index: Edge indices [2, num_edges] - can vary between graphs
            t: Diffusion time
            
        Returns:
            Updated node features [n, total_dim]
        """
        n = x.size(0)  # Dynamic graph size
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
            # Compute graph heat kernel (handles dynamic graph size)
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


class InductiveBuNN(nn.Module):
    """Inductive version: Bundle Neural Network."""
    
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
        
        # InductiveBuNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = self.total_dim if i == 0 else self.total_dim
            self.layers.append(
                InductiveBuNNLayer(
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
        Forward pass of InductiveBuNN.
        
        Args:
            x: Node features [n, input_dim] - n can vary between graphs
            edge_index: Edge indices [2, num_edges] - can vary between graphs
            batch: Batch vector for graph-level tasks [n]
            
        Returns:
            Output predictions
        """
        # Input projection
        h = self.input_proj(x)
        h = self.dropout(h)
        
        # Apply InductiveBuNN layers
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



import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import get_laplacian, to_dense_adj
import math

class ImprovedBundleMapLearner(nn.Module):
    """Improved bundle map learner using graph structure and positional encodings."""
    
    def __init__(self, input_dim: int, d: int, pos_enc_dim: int = 0, 
                 method: str = "rotation", use_graph_structure: bool = True):
        super().__init__()
        self.d = d
        self.method = method
        self.use_graph_structure = use_graph_structure
        
        # Total input dimension includes node features + positional encodings
        total_input_dim = input_dim + pos_enc_dim
        
        if use_graph_structure:
            # Use GraphSAGE-like architecture as mentioned in paper
            self.graph_conv1 = SAGEConv(total_input_dim, total_input_dim // 2)
            self.graph_conv2 = SAGEConv(total_input_dim // 2, total_input_dim // 4)
            final_dim = total_input_dim // 4
        else:
            # Fallback to MLP
            self.mlp = nn.Sequential(
                nn.Linear(total_input_dim, total_input_dim // 2),
                nn.ReLU(),
                nn.Linear(total_input_dim // 2, total_input_dim // 4)
            )
            final_dim = total_input_dim // 4
        
        if method == "rotation" and d == 2:
            self.angle_head = nn.Linear(final_dim, 1)
        elif method == "householder":
            num_params = d * (d - 1) // 2
            self.param_head = nn.Linear(final_dim, num_params)
        elif method == "cayley":
            num_params = d * (d - 1) // 2
            self.param_head = nn.Linear(final_dim, num_params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                pos_enc: torch.Tensor = None) -> torch.Tensor:
        n = x.size(0)
        
        # Concatenate positional encodings if provided
        if pos_enc is not None:
            x = torch.cat([x, pos_enc], dim=-1)
        
        # Use graph structure or MLP
        if self.use_graph_structure:
            h = torch.relu(self.graph_conv1(x, edge_index))
            h = torch.relu(self.graph_conv2(h, edge_index))
        else:
            h = self.mlp(x)
        
        if self.method == "rotation" and self.d == 2:
            angles = self.angle_head(h).squeeze(-1)
            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)
            
            O_maps = torch.zeros(n, 2, 2, device=x.device, dtype=x.dtype)
            O_maps[:, 0, 0] = cos_theta
            O_maps[:, 0, 1] = -sin_theta
            O_maps[:, 1, 0] = sin_theta
            O_maps[:, 1, 1] = cos_theta
            
            return O_maps
        elif self.method == "householder":
            params = self.param_head(h)
            return self._householder_to_orthogonal(params)
        elif self.method == "cayley":
            params = self.param_head(h)
            return self._cayley_to_orthogonal(params)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ImprovedGraphHeatKernel(nn.Module):
    """Improved heat kernel with adaptive method selection."""
    
    def __init__(self, method: str = "adaptive", max_degree: int = 8, 
                 spectral_threshold: float = 10.0):
        super().__init__()
        self.method = method
        self.max_degree = max_degree
        self.spectral_threshold = spectral_threshold
        self._cached_eigendecomposition = {}
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int, t: float) -> torch.Tensor:
        if t == 0:
            return torch.eye(num_nodes, device=edge_index.device)
        
        # Adaptive method selection as mentioned in paper
        if self.method == "adaptive":
            if t <= self.spectral_threshold:
                return self._taylor_approximation(edge_index, num_nodes, t)
            else:
                return self._spectral_method(edge_index, num_nodes, t)
        elif self.method == "taylor":
            return self._taylor_approximation(edge_index, num_nodes, t)
        elif self.method == "spectral":
            return self._spectral_method(edge_index, num_nodes, t)
    
    def _get_normalized_laplacian(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute normalized Laplacian using PyG utilities."""
        edge_index_with_loops, edge_weight = get_laplacian(
            edge_index, normalization='rw', num_nodes=num_nodes
        )
        
        # Convert to dense matrix
        adj = to_dense_adj(edge_index_with_loops, edge_attr=edge_weight, 
                          max_num_nodes=num_nodes).squeeze(0)
        
        return adj
    
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
        """Spectral method with caching for efficiency."""
        # Create cache key
        edge_key = tuple(edge_index.flatten().tolist())
        cache_key = (edge_key, num_nodes)
        
        if cache_key not in self._cached_eigendecomposition:
            L = self._get_normalized_laplacian(edge_index, num_nodes)
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            self._cached_eigendecomposition[cache_key] = (eigenvalues, eigenvectors)
        
        eigenvalues, eigenvectors = self._cached_eigendecomposition[cache_key]
        exp_eigenvalues = torch.exp(-t * eigenvalues)
        heat_kernel = torch.mm(eigenvectors * exp_eigenvalues.unsqueeze(0), eigenvectors.t())
        
        return heat_kernel


def compute_positional_encodings(edge_index: torch.Tensor, num_nodes: int, 
                               pe_type: str = "laplacian", pe_dim: int = 8) -> torch.Tensor:
    """Compute positional encodings as used in the paper."""
    
    if pe_type == "laplacian":
        # Laplacian Positional Encoding
        L = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)[0]
        L_dense = to_dense_adj(L, max_num_nodes=num_nodes).squeeze(0)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
        # Take smallest non-zero eigenvalues (skip the first zero eigenvalue)
        pe = eigenvectors[:, 1:pe_dim+1]
        
    elif pe_type == "random_walk":
        # Random Walk Structural Encoding (simplified version)
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        deg = adj.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        rw_matrix = adj * deg_inv.unsqueeze(1)
        
        # Compute powers of random walk matrix
        pe_list = []
        rw_power = torch.eye(num_nodes, device=edge_index.device)
        for k in range(pe_dim):
            pe_list.append(rw_power.sum(dim=1, keepdim=True))
            rw_power = torch.mm(rw_power, rw_matrix)
        
        pe = torch.cat(pe_list, dim=1)
    
    else:
        raise ValueError(f"Unknown PE type: {pe_type}")
    
    return pe


class ImprovedInductiveBuNNLayer(nn.Module):
    """Improved BuNN layer with paper-specific configurations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, d: int = 2, 
                 num_bundles: int = 1, pos_enc_dim: int = 8,
                 bundle_method: str = "rotation", heat_method: str = "adaptive",
                 max_degree: int = 8, use_graph_structure: bool = True,
                 activation: str = "relu"):
        super().__init__()
        
        self.d = d
        self.num_bundles = num_bundles
        self.total_dim = d * num_bundles
        
        # Improved bundle learner with graph structure and PE
        self.bundle_learner = ImprovedBundleMapLearner(
            input_dim=input_dim,
            d=d,
            pos_enc_dim=pos_enc_dim,
            method=bundle_method,
            use_graph_structure=use_graph_structure
        )
        
        # Improved heat kernel
        self.heat_kernel = ImprovedGraphHeatKernel(
            method=heat_method, 
            max_degree=max_degree
        )
        
        # Learnable parameters (same as original)
        self.W = nn.Parameter(torch.randn(self.total_dim, self.total_dim))
        self.bias = nn.Parameter(torch.zeros(self.total_dim))
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.W)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                pos_enc: torch.Tensor = None, t: float = 1.0) -> torch.Tensor:
        n = x.size(0)
        device = x.device
        
        # Learn bundle maps with positional encodings
        O_maps = self.bundle_learner(x, edge_index, pos_enc)
        
        # Handle multiple bundles
        if self.num_bundles > 1:
            O_block_diag = torch.zeros(n, self.total_dim, self.total_dim, device=device)
            for i in range(self.num_bundles):
                start_idx = i * self.d
                end_idx = (i + 1) * self.d
                O_block_diag[:, start_idx:end_idx, start_idx:end_idx] = O_maps
            O_maps = O_block_diag
        
        # Project input to correct dimension
        if x.size(1) != self.total_dim:
            x_expanded = torch.zeros(n, self.total_dim, device=device)
            x_expanded[:, :min(x.size(1), self.total_dim)] = x[:, :min(x.size(1), self.total_dim)]
            x = x_expanded
        
        # Bundle encoder step
        Ox = torch.bmm(O_maps, x.unsqueeze(-1)).squeeze(-1)
        Wx = torch.mm(Ox, self.W.t())
        h = torch.bmm(O_maps.transpose(-1, -2), Wx.unsqueeze(-1)).squeeze(-1)
        h = h + self.bias
        
        # Message diffusion via heat kernel
        if t > 0:
            H_graph = self.heat_kernel(edge_index, n, t)
            h = self._flat_bundle_diffusion(h, O_maps, H_graph)
        
        return self.activation(h)
    
    def _flat_bundle_diffusion(self, h: torch.Tensor, O_maps: torch.Tensor, 
                              H_graph: torch.Tensor) -> torch.Tensor:
        """Same as original implementation."""
        h_sync = torch.bmm(O_maps, h.unsqueeze(-1)).squeeze(-1)
        h_diffused = torch.mm(H_graph, h_sync)
        h_result = torch.bmm(O_maps.transpose(-1, -2), h_diffused.unsqueeze(-1)).squeeze(-1)
        return h_result


class ImprovedInductiveBuNN(nn.Module):
    """Improved version: Bundle Neural Network with positional encodings and adaptive heat kernel."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        d: int = 2,
        num_bundles: int = 1,
        bundle_method: str = "rotation",
        heat_method: str = "adaptive",
        max_degree: int = 8,
        diffusion_times: Optional[Union[float, list]] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        is_graph_level_task: bool = False,
        graph_pooling: str = "mean",
        pos_enc_dim: int = 8,
        pos_enc_type: str = "laplacian",
        use_graph_structure: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.is_graph_level_task = is_graph_level_task
        self.graph_pooling = graph_pooling
        self.total_dim = d * num_bundles
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc_type = pos_enc_type
        
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
        
        # ImprovedInductiveBuNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = self.total_dim if i == 0 else self.total_dim
            self.layers.append(
                ImprovedInductiveBuNNLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    d=d,
                    num_bundles=num_bundles,
                    pos_enc_dim=pos_enc_dim,
                    bundle_method=bundle_method,
                    heat_method=heat_method,
                    max_degree=max_degree,
                    use_graph_structure=use_graph_structure,
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
        Forward pass of ImprovedInductiveBuNN.
        
        Args:
            x: Node features [n, input_dim] - n can vary between graphs
            edge_index: Edge indices [2, num_edges] - can vary between graphs
            batch: Batch vector for graph-level tasks [n]
            
        Returns:
            Output predictions
        """
        n = x.size(0)  # Dynamic graph size
        
        # Compute positional encodings
        pos_enc = compute_positional_encodings(
            edge_index, n, 
            pe_type=self.pos_enc_type, 
            pe_dim=self.pos_enc_dim
        )
        
        # Input projection
        h = self.input_proj(x)
        h = self.dropout(h)
        
        # Apply ImprovedInductiveBuNN layers
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, pos_enc=pos_enc, t=self.diffusion_times[i])
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

