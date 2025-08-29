import torch
from torch.nn import Module

from model_encoders.sheaf_model_utils.inductive_disccrete_models import (
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion
)


class NeuralSheafEncoder(Module):
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
        

    def forward(self, x, edge_index):
        
        return self.sheaf_model(x, edge_index)

    def get_sheaf_model(self):
        return self.sheaf_model