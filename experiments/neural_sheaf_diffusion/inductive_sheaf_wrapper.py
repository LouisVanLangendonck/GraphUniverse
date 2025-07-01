import torch
from torch.nn import Module
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph

from experiments.neural_sheaf_diffusion.inductive_disc_models import (
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion
)


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
        add_lp=False, add_hp=False, **kwargs
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
            'input_dim': input_dim,
            'output_dim': output_dim,
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

    def _forward_single_graph(self, x, edge_index):
        # Direct forward pass - no lazy initialization needed
        return self.sheaf_model(x, edge_index)

    def _process_batched_graphs(self, x, edge_index, batch):
        outputs = []
        num_graphs = int(batch.max().item()) + 1

        for i in range(num_graphs):
            node_mask = (batch == i)
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            graph_x = x[node_idx]
            graph_edge_index, _ = subgraph(node_idx, edge_index, relabel_nodes=True)

            out = self._forward_single_graph(graph_x, graph_edge_index)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(self, x, edge_index, batch=None, **kwargs):
        if batch is None:
            log_probs = self._forward_single_graph(x, edge_index)
        else:
            log_probs = self._process_batched_graphs(x, edge_index, batch)

        if self.is_graph_level_task:
            if batch is None:
                out = torch.mean(log_probs, dim=0, keepdim=True)
            else:
                out = global_mean_pool(log_probs, batch)
            if self.readout:
                out = self.readout(out)
                if self.is_regression:
                    out = out * self.scale_factor
        else:
            # Return log probabilities directly (no exp conversion)
            out = log_probs

        return out

    def get_sheaf_model(self):
        return self.sheaf_model 