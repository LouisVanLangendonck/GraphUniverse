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
        
        # Only add sigmoid for classification tasks
        if not is_regression:
            self.readout.add_module("sigmoid", torch.nn.Sigmoid())
        else:
            # Add scaling factor for regression tasks
            self.scale_factor = torch.nn.Parameter(torch.ones(output_dim))

    def _forward_single_graph(self, x, edge_index, graph=None):
        # Set the current graph for the sheaf model to access precomputed indices
        if graph is not None and hasattr(graph, 'sheaf_indices_cache'):
            self.sheaf_model._current_graph = graph
        else:
            self.sheaf_model._current_graph = None
            
        # Direct forward pass with precomputed indices handled internally
        return self.sheaf_model(x, edge_index)

    # def _process_batched_graphs(self, x, edge_index, batch, batch_graphs=None):
    #     outputs = []
    #     num_graphs = int(batch.max().item()) + 1
    #     for i in range(num_graphs):
    #         node_mask = (batch == i)
    #         node_idx = node_mask.nonzero(as_tuple=False).view(-1)
    #         graph_x = x[node_idx]
    #         graph_edge_index, _ = subgraph(node_idx, edge_index, relabel_nodes=True)
    #         graph_obj = batch_graphs[i] if batch_graphs is not None else None
    #         out = self._forward_single_graph(graph_x, graph_edge_index, graph=graph_obj)
    #         outputs.append(out)
    #     return torch.cat(outputs, dim=0)

    def forward(self, x, edge_index, batch=None, **kwargs):
        # batch_graphs: list of graph objects for each item in the batch (for sheaf cache)
        if batch is None:
            graph = kwargs.get('graph', None)
            h = self._forward_single_graph(x, edge_index, graph=graph)
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
        if self.is_regression:
            out = out * self.scale_factor
            
        return out

    def get_sheaf_model(self):
        return self.sheaf_model 