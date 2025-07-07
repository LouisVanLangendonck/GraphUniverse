import torch
from torch.nn import Module
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph
from typing import Optional, Union
import torch.nn as nn

from experiments.neural_sheaf_diffusion.inductive_disc_models import (
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion
)

from experiments.neural_sheaf_diffusion.inductive_cont_models import (
    InductiveDiagSheafDiffusion,
    InductiveBundleSheafDiffusion,
    InductiveGeneralSheafDiffusion
)

from experiments.neural_sheaf_diffusion.inductive_bundle_models import InductiveBuNN, ImprovedInductiveBuNN


class InductiveSheafDiffusionModel(Module):
    def __init__(
        self,
        input_dim, hidden_dim, output_dim,
        sheaf_type="diag", d=2, num_layers=2,
        dropout=0.1, input_dropout=0.1,
        is_regression=False, is_graph_level_task=False,
        device="cpu", normalised=True, deg_normalised=False,
        linear=False, left_weights=True, right_weights=True,
        sparse_learner=False, use_act=True, sheaf_act="tanh",
        second_linear=False, orth="cayley",
        edge_weights=False, max_t=1.0,
        add_lp=False, add_hp=False, 
        pe_type=None, pe_dim=16, **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sheaf_type = sheaf_type
        self.d = d
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        self.device = device
        
        # PE configuration
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        
        # Adjust input dimension if PE is used
        self.actual_input_dim = input_dim
        if pe_type is not None:
            self.actual_input_dim = input_dim + pe_dim

        if sheaf_type == "diag":
            assert d >= 1
            self.sheaf_class = InductiveDiscreteDiagSheafDiffusion
        elif sheaf_type == "bundle":
            assert d > 1
            self.sheaf_class = InductiveDiscreteBundleSheafDiffusion
        elif sheaf_type == "general":
            assert d > 1
            self.sheaf_class = InductiveDiscreteGeneralSheafDiffusion
        else:
            raise ValueError(f"Unknown sheaf type: {sheaf_type}")

        self.sheaf_config = {
            'd': d,
            'layers': num_layers,
            'hidden_channels': hidden_dim // d,
            'input_dim': self.actual_input_dim,  # Use adjusted input dimension
            'output_dim': hidden_dim,
            'device': device,
            'normalised': normalised,
            'deg_normalised': deg_normalised,
            'linear': linear,
            'input_dropout': input_dropout,
            'dropout': dropout,
            'left_weights': left_weights,
            'right_weights': right_weights,
            'sparse_learner': sparse_learner,
            'use_act': use_act,
            'sheaf_act': sheaf_act,
            'second_linear': second_linear,
            'orth': orth,
            'edge_weights': edge_weights,
            'max_t': max_t,
            'add_lp': add_lp,
            'add_hp': add_hp,
            'graph_size': None
        }

        # Create the sheaf model immediately (no lazy initialization)
        self.sheaf_model = self.sheaf_class(self.sheaf_config)

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
                torch.nn.Linear(hidden_dim, output_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # if is_regression:
        #     # Add scaling factor for regression tasks
        #     self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))

    def _get_pe_from_data(self, data, **kwargs):
        """Extract positional encoding from data or kwargs."""
        if self.pe_type is None:
            return None
            
        # Try to get PE from kwargs first
        pe_attr_name = f"{self.pe_type}_pe"
        if pe_attr_name in kwargs:
            return kwargs[pe_attr_name]
        
        # Try to get PE from data object
        if hasattr(data, pe_attr_name):
            return getattr(data, pe_attr_name)
        
        # Try to get PE from kwargs with 'data' key
        if 'data' in kwargs and hasattr(kwargs['data'], pe_attr_name):
            return getattr(kwargs['data'], pe_attr_name)
        
        print(f"Warning: PE type '{self.pe_type}' not found in data")
        return None

    def _forward_single_graph(self, x, edge_index, graph=None, **kwargs):
        # Set the current graph for the sheaf model to access precomputed indices
        if graph is not None and hasattr(graph, 'sheaf_indices_cache'):
            self.sheaf_model._current_graph = graph
        else:
            self.sheaf_model._current_graph = None
        
        # Get PE and concatenate with input features if specified
        if self.pe_type is not None:
            pe = self._get_pe_from_data(graph, **kwargs)
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
            
        # Direct forward pass with precomputed indices handled internally
        return self.sheaf_model(x, edge_index)

    def forward(self, x, edge_index, batch=None, **kwargs):
        # batch_graphs: list of graph objects for each item in the batch (for sheaf cache)
        if batch is None:
            graph = kwargs.get('graph', None)
            # Remove graph from kwargs to avoid conflict with _forward_single_graph parameter
            kwargs_without_graph = {k: v for k, v in kwargs.items() if k != 'graph'}
            h = self._forward_single_graph(x, edge_index, graph=graph, **kwargs_without_graph)
        else:
            raise NotImplementedError("Batch processing not implemented for sheaf models")
            # log_probs = self._process_batched_graphs(x, edge_index, batch, batch_graphs=batch_graphs)
        
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
        # if self.is_regression:
        #     out = out * self.scale_factor
            
        return out

    def get_sheaf_model(self):
        return self.sheaf_model

class InductiveContSheafDiffusionModel(Module):
    def __init__(
        self,
        input_dim, hidden_dim, output_dim,
        sheaf_type="diag", d=2, num_layers=2,
        dropout=0.1, input_dropout=0.1,
        is_regression=False, is_graph_level_task=False,
        device="cpu", normalised=True, deg_normalised=False,
        linear=False, left_weights=True, right_weights=True,
        sparse_learner=False, use_act=True, sheaf_act="tanh",
        second_linear=False, orth="cayley",
        edge_weights=False, max_t=1.0,
        add_lp=False, add_hp=False, 
        pe_type=None, pe_dim=16, **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sheaf_type = sheaf_type
        self.d = d
        self.num_layers = num_layers
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        self.device = device
        
        # PE configuration
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        
        # Adjust input dimension if PE is used
        self.actual_input_dim = input_dim
        if pe_type is not None:
            self.actual_input_dim = input_dim + pe_dim

        if sheaf_type == "diag":
            assert d >= 1
            self.sheaf_class = InductiveDiagSheafDiffusion
        elif sheaf_type == "bundle":
            assert d > 1
            self.sheaf_class = InductiveBundleSheafDiffusion
        elif sheaf_type == "general":
            assert d > 1
            self.sheaf_class = InductiveGeneralSheafDiffusion
        else:
            raise ValueError(f"Unknown sheaf type: {sheaf_type}")

        self.sheaf_config = {
            'd': d,
            'layers': num_layers,
            'hidden_channels': hidden_dim // d,
            'input_dim': self.actual_input_dim,  # Use adjusted input dimension
            'output_dim': hidden_dim,
            'device': device,
            'normalised': normalised,
            'deg_normalised': deg_normalised,
            'linear': linear,
            'input_dropout': input_dropout,
            'dropout': dropout,
            'left_weights': left_weights,
            'right_weights': right_weights,
            'sparse_learner': sparse_learner,
            'use_act': use_act,
            'sheaf_act': sheaf_act,
            'second_linear': second_linear,
            'orth': orth,
            'edge_weights': edge_weights,
            'max_t': max_t,
            'add_lp': add_lp,
            'add_hp': add_hp,
            'graph_size': None,
            # ODE solver parameters
            'tol_scale': 10.0,
            'tol_scale_adjoint': 10.0,
            'adjoint': False,
            'int_method': 'rk4', # 'dopri5', 'rk4', 'bosh3' or 'euler' 
            'max_iters': 25,
            'adjoint_method': 'rk4',
        }

        # Create the sheaf model immediately (no lazy initialization)
        self.sheaf_model = self.sheaf_class(self.sheaf_config)

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
                torch.nn.Linear(hidden_dim, output_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, output_dim)
            )
        
        # if is_regression:
        #     # Add scaling factor for regression tasks
        #     self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))

    def _get_pe_from_data(self, data, **kwargs):
        """Extract positional encoding from data or kwargs."""
        if self.pe_type is None:
            return None
            
        # Try to get PE from kwargs first
        pe_attr_name = f"{self.pe_type}_pe"
        if pe_attr_name in kwargs:
            return kwargs[pe_attr_name]
        
        # Try to get PE from data object
        if hasattr(data, pe_attr_name):
            return getattr(data, pe_attr_name)
        
        # Try to get PE from kwargs with 'data' key
        if 'data' in kwargs and hasattr(kwargs['data'], pe_attr_name):
            return getattr(kwargs['data'], pe_attr_name)
        
        print(f"Warning: PE type '{self.pe_type}' not found in data")
        return None

    def _forward_single_graph(self, x, edge_index, graph=None, **kwargs):
        # Set the current graph for the sheaf model to access precomputed indices
        if graph is not None and hasattr(graph, 'sheaf_indices_cache'):
            self.sheaf_model._current_graph = graph
        else:
            self.sheaf_model._current_graph = None
        
        # Get PE and concatenate with input features if specified
        if self.pe_type is not None:
            pe = self._get_pe_from_data(graph, **kwargs)
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
            
        # Direct forward pass with precomputed indices handled internally
        return self.sheaf_model(x, edge_index)

    def forward(self, x, edge_index, batch=None, **kwargs):
        # batch_graphs: list of graph objects for each item in the batch (for sheaf cache)
        if batch is None:
            graph = kwargs.get('graph', None)
            # Remove graph from kwargs to avoid conflict with _forward_single_graph parameter
            kwargs_without_graph = {k: v for k, v in kwargs.items() if k != 'graph'}
            h = self._forward_single_graph(x, edge_index, graph=graph, **kwargs_without_graph)
        else:
            raise NotImplementedError("Batch processing not implemented for sheaf models")
            # log_probs = self._process_batched_graphs(x, edge_index, batch, batch_graphs=batch_graphs)
        
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
        # if self.is_regression:
        #     out = out * self.scale_factor
            
        return out

    def get_sheaf_model(self):
        return self.sheaf_model
    
class InductiveBuNNWrapper(nn.Module):
    """
    Wrapper class for InductiveBuNN that follows the same pattern as your sheaf models.
    This provides a consistent interface for different task types.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        d: int = 2,
        num_bundles: int = 1,
        num_layers: int = 2,
        bundle_method: str = "rotation",
        heat_method: str = "taylor",
        max_degree: int = 10,
        diffusion_times: None = None,
        dropout: float = 0.1,
        input_dropout: float = 0.1,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        graph_pooling: str = "mean",
        activation: str = "relu",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        self.device = device
        
        # Create the core BuNN model
        self.bunn_model = InductiveBuNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden_dim, then use readout head
            num_layers=num_layers,
            d=d,
            num_bundles=num_bundles,
            bundle_method=bundle_method,
            heat_method=heat_method,
            max_degree=max_degree,
            diffusion_times=diffusion_times,
            dropout=dropout,
            activation=activation,
            is_graph_level_task=False,  # Handle pooling in wrapper
            graph_pooling=graph_pooling
        )
        
        # Create prediction head
        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = nn.Linear(hidden_dim, output_dim)
        
        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass following the same pattern as your sheaf models.
        
        Args:
            x: Node features [n, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level tasks [n]
            
        Returns:
            Output predictions
        """
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Forward through BuNN
        h = self.bunn_model(x, edge_index, batch=None)  # Handle batching here
        
        # Graph-level pooling if needed
        if self.is_graph_level_task:
            if batch is None:
                # Single graph case
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Batched case
                h = global_mean_pool(h, batch)
        
        # Apply prediction head
        out = self.readout(h)
        
        if self.is_graph_level_task and not self.is_regression:
            # Remove extra dimension for graph classification
            out = out.squeeze(-1)
            
        return out

class ImprovedInductiveBuNNWrapper(nn.Module):
    """
    Wrapper class for ImprovedInductiveBuNN that follows the same pattern as your sheaf models.
    This provides a consistent interface for different task types with improved features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        d: int = 2,
        num_bundles: int = 1,
        num_layers: int = 2,
        bundle_method: str = "rotation",
        heat_method: str = "adaptive",
        max_degree: int = 8,
        diffusion_times: None = None,
        dropout: float = 0.1,
        input_dropout: float = 0.1,
        is_regression: bool = False,
        is_graph_level_task: bool = False,
        graph_pooling: str = "mean",
        activation: str = "relu",
        device: str = "cpu",
        pos_enc_dim: int = 8,
        pos_enc_type: str = "laplacian",
        use_graph_structure: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_regression = is_regression
        self.is_graph_level_task = is_graph_level_task
        self.device = device
        
        # Create the core improved BuNN model
        self.bunn_model = ImprovedInductiveBuNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden_dim, then use readout head
            num_layers=num_layers,
            d=d,
            num_bundles=num_bundles,
            bundle_method=bundle_method,
            heat_method=heat_method,
            max_degree=max_degree,
            diffusion_times=diffusion_times,
            dropout=dropout,
            activation=activation,
            is_graph_level_task=False,  # Handle pooling in wrapper
            graph_pooling=graph_pooling,
            pos_enc_dim=pos_enc_dim,
            pos_enc_type=pos_enc_type,
            use_graph_structure=use_graph_structure
        )
        
        # Create prediction head
        if is_graph_level_task:
            # Graph-level prediction head
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Node-level prediction head
            self.readout = nn.Linear(hidden_dim, output_dim)
        
        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass following the same pattern as your sheaf models.
        
        Args:
            x: Node features [n, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level tasks [n]
            
        Returns:
            Output predictions
        """
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Forward through improved BuNN
        h = self.bunn_model(x, edge_index, batch=None)  # Handle batching here
        
        # Graph-level pooling if needed
        if self.is_graph_level_task:
            if batch is None:
                # Single graph case
                h = torch.mean(h, dim=0, keepdim=True)
            else:
                # Batched case
                h = global_mean_pool(h, batch)
        
        # Apply prediction head
        out = self.readout(h)
        
        if self.is_graph_level_task and not self.is_regression:
            # Remove extra dimension for graph classification
            out = out.squeeze(-1)
            
        return out