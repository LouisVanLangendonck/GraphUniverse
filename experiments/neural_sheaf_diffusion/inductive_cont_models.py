# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.neural_sheaf_diffusion.sheaf_base import SheafDiffusion

from torch import nn

from experiments.neural_sheaf_diffusion.laplacian_builders import DiagLaplacianBuilder, GeneralLaplacianBuilder, NormConnectionLaplacianBuilder
from experiments.neural_sheaf_diffusion.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerPyg
import experiments.neural_sheaf_diffusion.laplacian_builders as lb
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


class InductiveLaplacianODEFunc(nn.Module):
    """Inductive version of Laplacian-based diffusion ODE function."""

    def __init__(self,
                 d, sheaf_learner, edge_index, hidden_channels,
                 left_weights=False,
                 right_weights=False,
                 use_act=False,
                 nonlinear=False,
                 weight_learner=None,
                 sheaf_type="diag",
                 normalised=False,
                 deg_normalised=False,
                 add_hp=False,
                 add_lp=False,
                 orth_trans="cayley"):
        """
        Args:
            d: Sheaf dimension
            sheaf_learner: The sheaf learning module
            edge_index: Initial edge index (will be updated dynamically)
            hidden_channels: Number of hidden channels
            sheaf_type: Type of sheaf ("diag", "bundle", "general")
            Other args: Various configuration parameters
        """
        super(InductiveLaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.L = None
        
        # Store sheaf configuration for dynamic laplacian builder creation
        self.sheaf_type = sheaf_type
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.orth_trans = orth_trans
        
        # Calculate final_d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.final_d, self.final_d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def _create_laplacian_builder(self, edge_index, num_nodes, precomputed_indices=None):
        """Create appropriate laplacian builder based on sheaf type."""
        if self.sheaf_type == "diag":
            return DiagLaplacianBuilder(
                num_nodes, edge_index, d=self.d,
                normalised=self.normalised, deg_normalised=self.deg_normalised,
                add_hp=self.add_hp, add_lp=self.add_lp,
                precomputed_indices=precomputed_indices)
        elif self.sheaf_type == "bundle":
            return NormConnectionLaplacianBuilder(
                num_nodes, edge_index, d=self.d,
                add_hp=self.add_hp, add_lp=self.add_lp,
                orth_map=self.orth_trans,
                precomputed_indices=precomputed_indices)
        elif self.sheaf_type == "general":
            return GeneralLaplacianBuilder(
                num_nodes, edge_index, d=self.d,
                normalised=self.normalised, deg_normalised=self.deg_normalised,
                add_hp=self.add_hp, add_lp=self.add_lp,
                precomputed_indices=precomputed_indices)
        else:
            raise ValueError(f"Unknown sheaf type: {self.sheaf_type}")

    def set_current_graph_info(self, edge_index, num_nodes, precomputed_indices=None):
        """Set the current graph information for this forward pass."""
        self.current_edge_index = edge_index
        self.current_num_nodes = num_nodes
        self.current_precomputed_indices = precomputed_indices

    def forward(self, t, x):
        # Get current graph dimensions
        num_nodes = self.current_num_nodes
        edge_index = self.current_edge_index
        
        if self.nonlinear or self.L is None:
            # Update the laplacian at each step.
            x_maps = x.view(num_nodes, -1)
            maps = self.sheaf_learner(x_maps, edge_index)
            
            # Create laplacian builder for current graph
            laplacian_builder = self._create_laplacian_builder(
                edge_index, num_nodes, self.current_precomputed_indices)
            
            if self.weight_learner is not None:
                # Update edge_index for weight learner if needed
                if hasattr(self.weight_learner, 'update_edge_index'):
                    self.weight_learner.update_edge_index(edge_index)
                edge_weights = self.weight_learner(x_maps, edge_index)
                L, _ = laplacian_builder(maps, edge_weights)
            else:
                L, _ = laplacian_builder(maps)
            self.L = L
            
            # Explicitly delete the laplacian_builder to free memory
            del laplacian_builder
        else:
            # Cache the Laplacian obtained at the first layer for the rest of the integration.
            L = self.L

        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, num_nodes * self.final_d).t()

        if self.right_weights:
            x = self.lin_right_weights(x)

        x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), -x)

        if self.use_act:
            x = F.elu(x)

        return x


class InductiveODEBlock(nn.Module):
    """Inductive version of ODE Integration block."""

    def __init__(self, odefunc, t, opt):
        super(InductiveODEBlock, self).__init__()
        self.t = t
        self.opt = opt
        self.odefunc = odefunc
        self.set_tol()

    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def forward(self, x, edge_index, num_nodes, precomputed_indices=None):
        # Set current graph information in the ODE function
        self.odefunc.set_current_graph_info(edge_index, num_nodes, precomputed_indices)
        
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(max_iters=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            z = odeint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol)
        self.odefunc.L = None
        z = z[1]
        
        # Force garbage collection to prevent memory leaks
        import gc
        gc.collect()
        
        return z


class InductiveGraphLaplacianDiffusion(SheafDiffusion):
    """Inductive version of diffusion model based on the weighted graph Laplacian."""

    def __init__(self, config):
        super(InductiveGraphLaplacianDiffusion, self).__init__(None, config)
        assert config['d'] == 1

        self.config = config
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # Initialize with dummy edge_index
        dummy_edge_index = torch.zeros((2, 1), dtype=torch.long)
        self.sheaf_learner = EdgeWeightLearner(self.hidden_dim, dummy_edge_index)

        self.odefunc = InductiveLaplacianODEFunc(
            self.final_d, self.sheaf_learner, dummy_edge_index, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, weight_learner=self.sheaf_learner, sheaf_type="diag",
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp)
        self.odeblock = InductiveODEBlock(self.odefunc, self.time_range, config)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Check if we have precomputed indices
        precomputed_indices = None
        if hasattr(self, '_current_graph') and self._current_graph is not None:
            if hasattr(self._current_graph, 'sheaf_indices_cache'):
                precomputed_indices = self._current_graph.sheaf_indices_cache
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(num_nodes * self.final_d, -1)
            x = self.odeblock(x, edge_index, num_nodes, precomputed_indices)
        x = x.view(num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiagSheafDiffusion(SheafDiffusion):
    """Inductive version of diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, config):
        super(InductiveDiagSheafDiffusion, self).__init__(None, config)

        self.config = config
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)

        # Initialize with dummy edge_index
        dummy_edge_index = torch.zeros((2, 1), dtype=torch.long)
        self.odefunc = InductiveLaplacianODEFunc(
            self.final_d, self.sheaf_learner, dummy_edge_index, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, sheaf_type="diag",
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp)
        self.odeblock = InductiveODEBlock(self.odefunc, self.time_range, config)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Check if we have precomputed indices
        precomputed_indices = None
        if hasattr(self, '_current_graph') and self._current_graph is not None:
            if hasattr(self._current_graph, 'sheaf_indices_cache'):
                precomputed_indices = self._current_graph.sheaf_indices_cache
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(num_nodes * self.final_d, -1)
            x = self.odeblock(x, edge_index, num_nodes, precomputed_indices)
        x = x.view(num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveBundleSheafDiffusion(SheafDiffusion):
    """Inductive version of diffusion using a sheaf Laplacian with bundle structure."""

    def __init__(self, config):
        super(InductiveBundleSheafDiffusion, self).__init__(None, config)
        # Should use diagonal sheaf diffusion instead if d=1.
        assert config['d'] > 1

        self.config = config
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # Initialize weight learner with dummy edge_index if needed
        dummy_edge_index = torch.zeros((2, 1), dtype=torch.long)
        self.weight_learner = EdgeWeightLearner(self.hidden_dim, dummy_edge_index) if self.use_edge_weights else None
        
        self.sheaf_learner = LocalConcatSheafLearnerPyg(self.hidden_dim, out_shape=(self.get_param_size(),),
                                                     sheaf_act=self.sheaf_act)

        self.odefunc = InductiveLaplacianODEFunc(
            self.final_d, self.sheaf_learner, dummy_edge_index, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, weight_learner=self.weight_learner, sheaf_type="bundle",
            add_hp=self.add_hp, add_lp=self.add_lp, orth_trans=self.orth_trans)
        self.odeblock = InductiveODEBlock(self.odefunc, self.time_range, config)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Check if we have precomputed indices
        precomputed_indices = None
        if hasattr(self, '_current_graph') and self._current_graph is not None:
            if hasattr(self._current_graph, 'sheaf_indices_cache'):
                precomputed_indices = self._current_graph.sheaf_indices_cache
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(num_nodes * self.final_d, -1)
            x = self.odeblock(x, edge_index, num_nodes, precomputed_indices)
        x = x.view(num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveGeneralSheafDiffusion(SheafDiffusion):
    """Inductive version of general sheaf diffusion."""

    def __init__(self, config):
        super(InductiveGeneralSheafDiffusion, self).__init__(None, config)
        # Should use diagonal diffusion if d == 1
        assert config['d'] > 1

        self.config = config
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)

        # Initialize with dummy edge_index
        dummy_edge_index = torch.zeros((2, 1), dtype=torch.long)
        self.odefunc = InductiveLaplacianODEFunc(
            self.final_d, self.sheaf_learner, dummy_edge_index, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, sheaf_type="general",
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp)
        self.odeblock = InductiveODEBlock(self.odefunc, self.time_range, config)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Check if we have precomputed indices
        precomputed_indices = None
        if hasattr(self, '_current_graph') and self._current_graph is not None:
            if hasattr(self._current_graph, 'sheaf_indices_cache'):
                precomputed_indices = self._current_graph.sheaf_indices_cache
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(num_nodes * self.final_d, -1)
            x = self.odeblock(x, edge_index, num_nodes, precomputed_indices)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.view(num_nodes, -1)
        x = self.lin2(x)
        return x