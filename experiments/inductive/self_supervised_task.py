"""
Self-supervised learning tasks for graph neural networks.
Updated with missing imports and integration fixes.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, Batch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import roc_auc_score
import os
import json
import copy
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
import random
from experiments.inductive.config import PreTrainingConfig
from experiments.core.models import GNNModel
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph


class LinkPredictionModel(nn.Module):
    """Link prediction model combining encoder and predictor."""
    
    def __init__(self, encoder: nn.Module, link_predictor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.link_predictor = link_predictor
    
    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def predict_links(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link probabilities."""
        edge_embeddings = torch.cat([
            embeddings[edge_index[0]], 
            embeddings[edge_index[1]]
        ], dim=1)
        return self.link_predictor(edge_embeddings).squeeze()


class ContrastiveModel(nn.Module):
    """Contrastive learning model."""
    
    def __init__(self, encoder: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.discriminator = discriminator
    
    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)


class GraphCLModel(nn.Module):
    """GraphCL model with projection head."""
    
    def __init__(self, encoder: nn.Module, projection_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)


class ReadoutFunction(nn.Module):
    """Simple mean readout for graph-level representations."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply mean readout - most stable and widely used."""
        return torch.mean(x, dim=0, keepdim=True)


class EnhancedContrastiveModel(nn.Module):
    """Enhanced contrastive model with learnable readout."""
    
    def __init__(self, encoder: nn.Module, discriminator: nn.Module, readout: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.readout = readout
    
    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

class GraphCorruption:
    """Various graph corruption strategies for contrastive learning."""
    
    @staticmethod
    def feature_shuffle(data: Data) -> Data:
        """Shuffle node features randomly."""
        corrupted = data.clone()
        perm = torch.randperm(data.x.size(0), device=data.x.device)
        corrupted.x = data.x[perm]
        return corrupted
    
    @staticmethod
    def feature_dropout(data: Data, drop_rate: float = 0.2) -> Data:
        """Randomly drop node features."""
        corrupted = data.clone()
        mask = torch.rand_like(corrupted.x) > drop_rate
        corrupted.x = corrupted.x * mask
        return corrupted
    
    @staticmethod
    def feature_noise(data: Data, noise_std: float = 0.1) -> Data:
        """Add Gaussian noise to features."""
        corrupted = data.clone()
        noise = torch.randn_like(corrupted.x) * noise_std
        corrupted.x = corrupted.x + noise
        return corrupted
    
    @staticmethod
    def edge_dropout(data: Data, drop_rate: float = 0.2) -> Data:
        """Randomly drop edges."""
        corrupted = data.clone()
        edge_mask = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > drop_rate
        corrupted.edge_index = data.edge_index[:, edge_mask]
        return corrupted
    
    @staticmethod
    def edge_perturbation(data: Data, perturb_rate: float = 0.1) -> Data:
        """Add random edges."""
        corrupted = data.clone()
        num_nodes = data.x.size(0)
        num_new_edges = int(data.edge_index.size(1) * perturb_rate)
        
        # Generate random edges
        new_edges = torch.randint(0, num_nodes, (2, num_new_edges), device=data.edge_index.device)
        
        # Combine with existing edges
        corrupted.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        return corrupted
    
    @staticmethod
    def subgraph_sampling(data: Data, sample_rate: float = 0.8) -> Data:
        """Sample a subgraph."""
        num_nodes = data.x.size(0)
        num_sample = int(num_nodes * sample_rate)
        
        # Randomly sample nodes
        sampled_nodes = torch.randperm(num_nodes, device=data.x.device)[:num_sample]
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            sampled_nodes, 
            data.edge_index, 
            data.edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        corrupted = Data(
            x=data.x[sampled_nodes],
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        return corrupted
    
    @staticmethod
    def random_walk_sampling(data: Data, walk_length: int = 10, num_walks: int = 5) -> Data:
        """Sample subgraph using random walks."""
        # Simplified random walk sampling
        num_nodes = data.x.size(0)
        visited = set()
        
        for _ in range(num_walks):
            current = random.randint(0, num_nodes - 1)
            for _ in range(walk_length):
                visited.add(current)
                # Find neighbors
                neighbors = data.edge_index[1][data.edge_index[0] == current]
                if len(neighbors) > 0:
                    current = neighbors[random.randint(0, len(neighbors) - 1)].item()
        
        if len(visited) == 0:
            visited = {0}  # Fallback
            
        sampled_nodes = torch.tensor(list(visited), device=data.x.device)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            sampled_nodes,
            data.edge_index,
            data.edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        corrupted = Data(
            x=data.x[sampled_nodes],
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        return corrupted

class SelfSupervisedTask(ABC):
    """Abstract base class for self-supervised learning tasks."""
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
        self.device = self._setup_device()
    
    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        if self.config.force_cpu:
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            if self.config.device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{self.config.device_id}")
                torch.cuda.set_device(self.config.device_id)
                return device
        return torch.device("cpu")
    
    @abstractmethod
    def create_model(self, input_dim: int) -> nn.Module:
        """Create model for this task."""
        pass
    
    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: Data) -> torch.Tensor:
        """Compute loss for this task."""
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on this task."""
        pass

class DeepGraphInfoMaxTask(SelfSupervisedTask):
    """Enhanced Deep Graph InfoMax with better corruption strategies."""
    
    def __init__(self, config):
        super().__init__(config)
        self.corruption_functions = {
            "feature_shuffle": GraphCorruption.feature_shuffle,
            "feature_dropout": GraphCorruption.feature_dropout,
            "feature_noise": GraphCorruption.feature_noise,
            "edge_dropout": GraphCorruption.edge_dropout,
            "edge_perturbation": GraphCorruption.edge_perturbation,
            "subgraph": GraphCorruption.subgraph_sampling,
            "random_walk": GraphCorruption.random_walk_sampling,
        }
        
    def create_model(self, input_dim: int) -> nn.Module:
        """Create enhanced DGI model."""
        # Encoder (same as before but with proper setup)
        if self.config.model_type == "transformer" or self.config.run_transformers:
            from experiments.core.models import GraphTransformerModel
            encoder = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                transformer_type=self.config.transformer_type,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                is_regression=False,
                num_heads=self.config.transformer_num_heads,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                precompute_encodings=self.config.transformer_precompute_encodings,
                cache_encodings=self.config.transformer_cache_encodings,
                local_gnn_type=self.config.local_gnn_type,
                prenorm=self.config.transformer_prenorm
            )
        else:
            from experiments.core.models import GNNModel
            encoder = GNNModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                gnn_type=self.config.gnn_type,
                is_regression=False,
                residual=self.config.residual,
                norm_type=self.config.norm_type,
                agg_type=self.config.agg_type,
                heads=self.config.heads,
                concat_heads=self.config.concat_heads,
            )
        
        # Create readout function as Mean + Sigmoid 
        def info_max_readout(x: torch.Tensor) -> torch.Tensor:
            feature_mean = torch.mean(x, dim=0, keepdim=True)
            return torch.sigmoid(feature_mean)

        readout = info_max_readout
        
        # Create discriminator as Single layer MLP to single output and sigmoid
        discriminator = nn.Sequential(
            nn.Linear(2*self.config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        return EnhancedContrastiveModel(encoder, discriminator, readout)
    
    def _corrupt_graph(self, batch: Data) -> Data:
        """Apply multiple corruption strategies."""
        corruption_type = getattr(self.config, 'corruption_type')
        
        
        corrupted = batch.clone()
        # Apply corruption with appropriate parameters
        if corruption_type == 'feature_dropout':
            drop_rate = getattr(self.config, 'corruption_rate', 0.2)
            corrupted = GraphCorruption.feature_dropout(corrupted, drop_rate)
        elif corruption_type == 'feature_shuffle':
            corrupted = GraphCorruption.feature_shuffle(corrupted)
        elif corruption_type == 'feature_noise':
            noise_std = getattr(self.config, 'noise_std', 0.1)
            corrupted = GraphCorruption.feature_noise(corrupted, noise_std)
        elif corruption_type == 'edge_dropout':
            drop_rate = getattr(self.config, 'corruption_rate', 0.2)
            corrupted = GraphCorruption.edge_dropout(corrupted, drop_rate)
        elif corruption_type == 'edge_perturbation':
            perturb_rate = getattr(self.config, 'perturb_rate', 0.1)
            corrupted = GraphCorruption.edge_perturbation(corrupted, perturb_rate)
        else:
            raise ValueError(f"{corruption_type} not implemented")
        
        return corrupted
    
    def compute_loss(self, model: nn.Module, batch: Data) -> torch.Tensor:
        """Enhanced DGI loss with better negative sampling."""
        # Original graph embeddings
        node_embeddings = model.encoder(batch.x, batch.edge_index)
        
        # Graph-level representation with learnable readout
        graph_embedding = model.readout(node_embeddings)
        
        corrupted_batch = self._corrupt_graph(batch)
        corrupted_embeddings = model.encoder(corrupted_batch.x, corrupted_batch.edge_index)
        corrupted_graph_embedding = model.readout(corrupted_embeddings)
        
        # Positive pairs: (node_embedding, graph_embedding)
        pos_pairs = torch.cat([
            node_embeddings, 
            graph_embedding.repeat(node_embeddings.size(0), 1)
        ], dim=1)
        pos_scores = model.discriminator(pos_pairs).squeeze()
        pos_term = torch.log(pos_scores + 1e-8).mean()  # E over N positive pairs
        
        # Negative pairs: (node_embedding, corrupted_graph_embedding)
        neg_pairs = torch.cat([
            node_embeddings,
            corrupted_graph_embedding.repeat(node_embeddings.size(0), 1)
        ], dim=1)
        neg_scores = model.discriminator(neg_pairs).squeeze()
        neg_term = torch.log(1 - neg_scores + 1e-8).mean()  # E over M negative pairs
        
        # InfoMax loss function: maximize both terms
        loss = -(pos_term + neg_term)  # Negative because we minimize
        
        return loss
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """ADDED: Evaluate Deep Graph InfoMax performance."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Compute loss
                loss = self.compute_loss(model, batch)
                total_loss += loss.item()
                
                n_batches += 1
        
        return {'loss': total_loss / n_batches if n_batches > 0 else 0.0}

class LinkPredictionTask(SelfSupervisedTask):
    """Link prediction self-supervised task."""
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create GNN model with link prediction head."""
        # Check if we should use transformer
        if self.config.model_type == "transformer" or self.config.run_transformers:
            from experiments.core.models import GraphTransformerModel
            encoder = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                transformer_type=self.config.transformer_type,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                is_regression=False,
                num_heads=self.config.transformer_num_heads,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                precompute_encodings=self.config.transformer_precompute_encodings,
                cache_encodings=self.config.transformer_cache_encodings,
                local_gnn_type=self.config.local_gnn_type,
                prenorm=self.config.transformer_prenorm
            )
        else:
            # Base GNN encoder
            encoder = GNNModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,  # Output embeddings
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                gnn_type=self.config.gnn_type,
                residual=self.config.residual,
                norm_type=self.config.norm_type,
                agg_type=self.config.agg_type,
                heads=self.config.heads,
                concat_heads=self.config.concat_heads,
                is_regression=False  # Output continuous embeddings
            )
        
        # Link prediction head
        link_predictor = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        return LinkPredictionModel(encoder, link_predictor)
    
    def compute_loss(self, model: nn.Module, batch: Data) -> torch.Tensor:
        """Compute link prediction loss."""
        # Get node embeddings
        embeddings = model.encoder(batch.x, batch.edge_index)
        
        # Generate negative edges
        neg_edge_index = negative_sampling(
            edge_index=batch.edge_index,
            num_nodes=batch.x.size(0),
            num_neg_samples=int(batch.edge_index.size(1) * self.config.negative_sampling_ratio)
        )
        
        # Positive edge predictions
        pos_edges = batch.edge_index
        pos_edge_embeddings = torch.cat([
            embeddings[pos_edges[0]], 
            embeddings[pos_edges[1]]
        ], dim=1)
        pos_pred = model.link_predictor(pos_edge_embeddings).squeeze()
        
        # Negative edge predictions
        neg_edge_embeddings = torch.cat([
            embeddings[neg_edge_index[0]], 
            embeddings[neg_edge_index[1]]
        ], dim=1)
        neg_pred = model.link_predictor(neg_edge_embeddings).squeeze()
        
        # Compute loss
        pos_labels = torch.ones_like(pos_pred)
        neg_labels = torch.zeros_like(neg_pred)
        
        if self.config.link_pred_loss == "bce":
            criterion = nn.BCELoss()
            loss = criterion(pos_pred, pos_labels) + criterion(neg_pred, neg_labels)
        else:  # margin loss
            criterion = nn.MarginRankingLoss(margin=0.1)
            labels = torch.ones(pos_pred.size(0)).to(pos_pred.device)
            loss = criterion(pos_pred, neg_pred, labels)
        
        return loss
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate link prediction performance."""
        model.eval()
        total_loss = 0.0
        total_auc = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.compute_loss(model, batch)
                total_loss += loss.item()
                
                # Compute AUC
                embeddings = model.encoder(batch.x, batch.edge_index)
                
                # Positive and negative edges
                pos_edges = batch.edge_index
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index,
                    num_nodes=batch.x.size(0),
                    num_neg_samples=pos_edges.size(1)
                )
                
                # Predictions
                pos_pred = model.predict_links(embeddings, pos_edges)
                neg_pred = model.predict_links(embeddings, neg_edge_index)
                
                # AUC calculation
                y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
                y_pred = torch.cat([pos_pred, neg_pred])
                
                try:
                    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    total_auc += auc
                except:
                    # Handle edge case where all predictions are the same
                    total_auc += 0.5
                
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'auc': total_auc / n_batches if n_batches > 0 else 0.0
        }

class ContrastiveTask(SelfSupervisedTask):
    """Contrastive learning task (Deep Graph InfoMax style)."""
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create GNN model with contrastive learning head."""
        # Check if we should use transformer
        if self.config.model_type == "transformer" or self.config.run_transformers:
            from experiments.core.models import GraphTransformerModel
            encoder = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                transformer_type=self.config.transformer_type,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                is_regression=False,
                num_heads=self.config.transformer_num_heads,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                precompute_encodings=self.config.transformer_precompute_encodings,
                cache_encodings=self.config.transformer_cache_encodings,
                local_gnn_type=self.config.local_gnn_type,
                prenorm=self.config.transformer_prenorm
            )
        else:
            # Base GNN encoder
            encoder = GNNModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            gnn_type=self.config.gnn_type,
            is_regression=True,
            residual=self.config.residual,
            norm_type=self.config.norm_type,
            agg_type=self.config.agg_type,
            heads=self.config.heads,
            concat_heads=self.config.concat_heads,
        )
        
        # Global discriminator for graph-level representations
        discriminator = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        return ContrastiveModel(encoder, discriminator)
    
    def _corrupt_graph(self, batch: Data) -> Data:
        """Create corrupted version of the graph."""
        corrupted = batch.clone()
        
        if self.config.corruption_type == "feature_shuffle":
            # Shuffle node features
            perm = torch.randperm(batch.x.size(0))
            corrupted.x = batch.x[perm]
        
        elif self.config.corruption_type == "edge_dropout":
            # Randomly drop edges
            edge_mask = torch.rand(batch.edge_index.size(1)) > self.config.corruption_rate
            corrupted.edge_index = batch.edge_index[:, edge_mask]
        
        return corrupted
    
    def compute_loss(self, model: nn.Module, batch: Data) -> torch.Tensor:
        """Compute contrastive loss."""
        # Original graph embeddings
        node_embeddings = model.encoder(batch.x, batch.edge_index)
        
        # Global graph representation (readout)
        graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
        # Corrupted graph
        corrupted_batch = self._corrupt_graph(batch)
        corrupted_embeddings = model.encoder(corrupted_batch.x, corrupted_batch.edge_index)
        corrupted_graph_embedding = torch.mean(corrupted_embeddings, dim=0, keepdim=True)
        
        # Positive pairs: (node_embedding, graph_embedding)
        pos_pairs = torch.cat([
            node_embeddings, 
            graph_embedding.repeat(node_embeddings.size(0), 1)
        ], dim=1)
        pos_scores = model.discriminator(pos_pairs).squeeze()
        
        # Negative pairs: (node_embedding, corrupted_graph_embedding)
        neg_pairs = torch.cat([
            node_embeddings,
            corrupted_graph_embedding.repeat(node_embeddings.size(0), 1)
        ], dim=1)
        neg_scores = model.discriminator(neg_pairs).squeeze()
        
        # Binary cross-entropy loss
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        criterion = nn.BCELoss()
        loss = criterion(pos_scores, pos_labels) + criterion(neg_scores, neg_labels)
        
        return loss
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate contrastive learning performance."""
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.compute_loss(model, batch)
                total_loss += loss.item()
                
                # Compute accuracy
                node_embeddings = model.encoder(batch.x, batch.edge_index)
                graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
                
                pos_pairs = torch.cat([
                    node_embeddings, 
                    graph_embedding.repeat(node_embeddings.size(0), 1)
                ], dim=1)
                pos_scores = model.discriminator(pos_pairs).squeeze()
                
                corrupted_batch = self._corrupt_graph(batch)
                corrupted_embeddings = model.encoder(corrupted_batch.x, corrupted_batch.edge_index)
                corrupted_graph_embedding = torch.mean(corrupted_embeddings, dim=0, keepdim=True)
                
                neg_pairs = torch.cat([
                    node_embeddings,
                    corrupted_graph_embedding.repeat(node_embeddings.size(0), 1)
                ], dim=1)
                neg_scores = model.discriminator(neg_pairs).squeeze()
                
                # Accuracy calculation
                pos_correct = (pos_scores > 0.5).float().mean()
                neg_correct = (neg_scores <= 0.5).float().mean()
                acc = (pos_correct + neg_correct) / 2
                
                total_acc += acc.item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'accuracy': total_acc / n_batches if n_batches > 0 else 0.0
        }

class GraphCLTask(SelfSupervisedTask):
    """Fixed GraphCL implementation with proper evaluation."""
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create GraphCL model with projection head."""
        # Encoder setup (same as before)
        if self.config.model_type == "transformer" or self.config.run_transformers:
            from experiments.core.models import GraphTransformerModel
            encoder = GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                transformer_type=self.config.transformer_type,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                is_regression=False,
                num_heads=self.config.transformer_num_heads,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                precompute_encodings=self.config.transformer_precompute_encodings,
                cache_encodings=self.config.transformer_cache_encodings,
                local_gnn_type=self.config.local_gnn_type,
                prenorm=self.config.transformer_prenorm
            )
        else:
            from experiments.core.models import GNNModel
            encoder = GNNModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                gnn_type=self.config.gnn_type,
                is_regression=False,
                residual=self.config.residual,
                norm_type=self.config.norm_type,
                agg_type=self.config.agg_type,
                heads=self.config.heads,
                concat_heads=self.config.concat_heads,
            )
        
        # Projection head for contrastive learning
        projection_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2)  # Smaller projection dim
        )
        
        return GraphCLModel(encoder, projection_head)
    
    def _create_node_correspondence_augmentations(self, batch: Data) -> Tuple[Data, Data, torch.Tensor]:
        """Create augmentations that maintain node correspondence."""
        # Only use augmentations that preserve node identity and order
        safe_augmentations = [
            lambda x: GraphCorruption.feature_dropout(x, drop_rate=0.1),
            lambda x: GraphCorruption.feature_noise(x, noise_std=0.05),
            lambda x: GraphCorruption.edge_dropout(x, drop_rate=0.1),
        ]
        
        # Apply different augmentations to create two views
        aug_fn1 = random.choice(safe_augmentations)
        aug_fn2 = random.choice(safe_augmentations)
        
        aug1 = aug_fn1(batch)
        aug2 = aug_fn2(batch)
        
        # Node correspondence is preserved (same number of nodes in same order)
        node_correspondence = torch.arange(batch.x.size(0))
        
        return aug1, aug2, node_correspondence
    
    def compute_loss(self, model: nn.Module, batch: Data) -> torch.Tensor:
        """Fixed GraphCL contrastive loss."""
        # Create two views with preserved node correspondence
        aug1, aug2, correspondence = self._create_node_correspondence_augmentations(batch)
        
        # Get embeddings for both views
        emb1 = model.encoder(aug1.x, aug1.edge_index)
        emb2 = model.encoder(aug2.x, aug2.edge_index)
        
        # Project embeddings
        proj1 = model.projection_head(emb1)
        proj2 = model.projection_head(emb2)
        
        # Normalize embeddings
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        # Temperature parameter
        temperature = getattr(self.config, 'temperature', 0.1)
        
        # Compute InfoNCE loss
        # Positive pairs: corresponding nodes between views
        pos_sim = torch.sum(proj1 * proj2, dim=1) / temperature  # [N]
        
        # For each node, compute similarity with ALL nodes in the other view
        sim_matrix = torch.mm(proj1, proj2.t()) / temperature  # [N, N]
        
        # Labels: each node should match itself (diagonal)
        labels = torch.arange(proj1.size(0)).to(proj1.device)
        
        # InfoNCE loss using cross-entropy
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Fixed evaluation with meaningful metrics."""
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_top5_acc = 0.0
        total_alignment = 0.0
        total_uniformity = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Compute loss
                loss = self.compute_loss(model, batch)
                total_loss += loss.item()
                
                # Create augmented views for evaluation
                aug1, aug2, correspondence = self._create_node_correspondence_augmentations(batch)
                
                # Get normalized projections
                emb1 = model.encoder(aug1.x, aug1.edge_index)
                emb2 = model.encoder(aug2.x, aug2.edge_index)
                
                proj1 = F.normalize(model.projection_head(emb1), dim=1)
                proj2 = F.normalize(model.projection_head(emb2), dim=1)
                
                # Compute similarity matrix
                sim_matrix = torch.mm(proj1, proj2.t())
                
                # Top-1 accuracy: how often does each node's best match correspond to itself?
                pred = torch.argmax(sim_matrix, dim=1)
                labels = correspondence.to(sim_matrix.device)
                acc = (pred == labels).float().mean()
                total_acc += acc.item()
                
                # Top-5 accuracy
                _, top5_pred = torch.topk(sim_matrix, k=min(5, sim_matrix.size(1)), dim=1)
                top5_acc = (top5_pred == labels.unsqueeze(1)).any(dim=1).float().mean()
                total_top5_acc += top5_acc.item()
                
                # Alignment: how well do positive pairs align?
                positive_sim = torch.sum(proj1 * proj2, dim=1)
                alignment = positive_sim.mean()
                total_alignment += alignment.item()
                
                # Uniformity: how uniformly distributed are the embeddings?
                # This measures whether embeddings are spread out on the unit sphere
                pairwise_sim = torch.mm(proj1, proj1.t())
                # Remove diagonal (self-similarity)
                mask = ~torch.eye(pairwise_sim.size(0), dtype=torch.bool, device=pairwise_sim.device)
                uniformity = torch.log(torch.exp(pairwise_sim[mask]).mean())
                total_uniformity += uniformity.item()
                
                n_batches += 1
        
        if n_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'top5_accuracy': 0.0, 'alignment': 0.0, 'uniformity': 0.0}
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,  # Top-1 accuracy
            'top5_accuracy': total_top5_acc / n_batches,  # Top-5 accuracy
            'alignment': total_alignment / n_batches,  # Positive pair similarity
            'uniformity': total_uniformity / n_batches,  # Embedding uniformity
        }

class SafeGraphCorruption:
    """Graph corruption strategies that preserve node correspondence."""
    
    @staticmethod
    def safe_feature_dropout(data: Data, drop_rate: float = 0.1) -> Data:
        """Feature dropout that preserves node count and order."""
        corrupted = data.clone()
        mask = torch.rand_like(corrupted.x) > drop_rate
        corrupted.x = corrupted.x * mask
        return corrupted
    
    @staticmethod
    def safe_feature_noise(data: Data, noise_std: float = 0.05) -> Data:
        """Add small amount of noise to features."""
        corrupted = data.clone()
        noise = torch.randn_like(corrupted.x) * noise_std
        corrupted.x = corrupted.x + noise
        return corrupted
    
    @staticmethod
    def safe_edge_dropout(data: Data, drop_rate: float = 0.1) -> Data:
        """Conservative edge dropout."""
        corrupted = data.clone()
        edge_mask = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > drop_rate
        corrupted.edge_index = data.edge_index[:, edge_mask]
        return corrupted

def create_ssl_task(config: PreTrainingConfig) -> SelfSupervisedTask:
    """Create SSL task based on configuration - enhanced version."""
    if config.pretraining_task == "link_prediction":
        return LinkPredictionTask(config)
    elif config.pretraining_task == "contrastive":
        return ContrastiveTask(config)  # Keep existing for backward compatibility
    elif config.pretraining_task == "dgi":
        return DeepGraphInfoMaxTask(config)  # Enhanced version
    elif config.pretraining_task == "graphcl":
        return GraphCLTask(config)
    else:
        raise ValueError(f"Unknown pretraining task: {config.pretraining_task}")

def create_pretraining_dataloader(
    graphs: List,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create dataloader for pre-training graphs."""
    from mmsb.feature_regimes import graphsample_to_pyg
    
    # Convert graphs to PyG format
    pyg_graphs = []
    for graph_sample in graphs:
        pyg_data = graphsample_to_pyg(graph_sample)
        pyg_graphs.append(pyg_data)
    
    # Create dataloader
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    dataloader = PyGDataLoader(
        pyg_graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    return dataloader

class PreTrainedModelSaver:
    """Handles saving and loading of pre-trained models."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_model(
    self,
    model: torch.nn.Module,
    config: PreTrainingConfig,
    training_history: Dict[str, List[float]],
    metrics: Dict[str, float],
    hyperopt_results: Optional[Dict] = None,
    enhanced_metadata: Optional[Dict] = None,
    model_id: Optional[str] = None
) -> str:
        """Save pre-trained model with enhanced metadata including family references."""
        
        if model_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use transformer_type for transformer models, otherwise use gnn_type
            if config.run_transformers or config.model_type == "transformer":
                model_type = config.transformer_type
            else:
                model_type = config.gnn_type
            model_id = f"{model_type}_{config.pretraining_task}_{timestamp}"
        
        model_dir = os.path.join(self.output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(model_dir, "model.pth")
        model_copy = copy.deepcopy(model)
        torch.save(model_copy.state_dict(), model_path)
        
        # Get the actual encoder from the model
        if hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            encoder = model
        
        # FIXED: Get input_dim from the encoder's actual parameters
        input_dim = None
        if hasattr(encoder, 'input_dim'):
            input_dim = encoder.input_dim
        elif hasattr(encoder, 'convs') and len(encoder.convs) > 0:
            # For GNN models, get from first layer
            first_layer = encoder.convs[0]
            if hasattr(first_layer, 'in_channels'):
                input_dim = first_layer.in_channels
            elif hasattr(first_layer, 'lin_l') and hasattr(first_layer.lin_l, 'in_features'):
                input_dim = first_layer.lin_l.in_features
        elif hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            # For MLP models
            first_layer = encoder.layers[0]
            if hasattr(first_layer, 'in_features'):
                input_dim = first_layer.in_features
        
        # Fallback: use universe_feature_dim from config
        if input_dim is None:
            input_dim = config.universe_feature_dim
            print(f"⚠️  Could not determine input_dim from model, using config.universe_feature_dim = {input_dim}")
        else:
            print(f"✅ Detected input_dim = {input_dim} from model architecture")
        
        # Save model architecture info with CORRECT input_dim
        arch_info = {
            'model_class': model.__class__.__name__,
            'encoder_class': encoder.__class__.__name__,
            'input_dim': input_dim,  # FIXED: Now properly set
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'gnn_type': config.gnn_type,
            'dropout': config.dropout,
            'residual': config.residual,
            'norm_type': config.norm_type,
            'agg_type': config.agg_type,
            'heads': config.heads,
            'concat_heads': config.concat_heads
        }
        
        # Create complete metadata with enhanced info
        metadata = {
            'model_id': model_id,
            'creation_timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'architecture': arch_info,
            'training_history': training_history,
            'final_metrics': metrics,
            'hyperopt_results': hyperopt_results,
            'model_path': model_path
        }
        
        # Add enhanced metadata (family info, etc.)
        if enhanced_metadata:
            metadata['enhanced_info'] = enhanced_metadata
            
            # Special handling for family info
            if 'family_id' in enhanced_metadata:
                metadata['family_id'] = enhanced_metadata['family_id']
                metadata['finetuning_ready'] = enhanced_metadata.get('finetuning_graphs_available', 0) > 0
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save config separately for easy loading
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        # Save family reference file for easy lookup
        if enhanced_metadata and 'family_id' in enhanced_metadata:
            family_ref = {
                'family_id': enhanced_metadata['family_id'],
                'model_id': model_id,
                'finetuning_graphs_available': enhanced_metadata.get('finetuning_graphs_available', 0),
                'pretraining_graphs_used': enhanced_metadata.get('pretraining_graphs_used', 0),
                'creation_timestamp': datetime.now().isoformat()
            }
            
            family_ref_path = os.path.join(model_dir, "family_reference.json")
            with open(family_ref_path, 'w') as f:
                json.dump(family_ref, f, indent=2)
        
        print(f"Saved pre-trained model to: {model_dir}")
        print(f"✅ Architecture saved with input_dim = {input_dim}")
        if enhanced_metadata and 'family_id' in enhanced_metadata:
            print(f"✅ Linked to graph family: {enhanced_metadata['family_id']}")
            print(f"✅ Fine-tuning graphs available: {enhanced_metadata.get('finetuning_graphs_available', 0)}")
        
        return model_id

    def load_model(self, model_id: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict]:
        """Load pre-trained model and metadata."""
        model_dir = os.path.join(self.output_dir, model_id)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Load family reference
        family_ref_path = os.path.join(model_dir, "family_reference.json")
        with open(family_ref_path, 'r') as f:
            family_ref = json.load(f)
        
        config = PreTrainingConfig(**config_dict)
        
        # Recreate model architecture
        task = create_ssl_task(config)
        
        # Need input_dim to create model
        input_dim = metadata['architecture']['input_dim']
        if input_dim is None:
            raise ValueError("Cannot recreate model: input_dim not saved in metadata")
        
        model = task.create_model(input_dim)
        
        # Load weights
        model_path = os.path.join(model_dir, "model.pth")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        return model, metadata, family_ref
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available pre-trained models with enhanced info."""
        models = []
        
        if not os.path.exists(self.output_dir):
            return models
        
        for model_id in os.listdir(self.output_dir):
            model_dir = os.path.join(self.output_dir, model_id)
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_info = {
                        'model_id': model_id,
                        'task': metadata['config']['pretraining_task'],
                        'gnn_type': metadata['config']['gnn_type'],
                        'creation_timestamp': metadata.get('creation_timestamp'),
                        'final_metrics': metadata.get('final_metrics', {}),
                        'model_dir': model_dir,
                        'family_id': metadata.get('family_id', None),
                        'finetuning_ready': metadata.get('finetuning_ready', False)
                    }
                    
                    # Add enhanced info if available
                    if 'enhanced_info' in metadata:
                        enhanced = metadata['enhanced_info']
                        model_info.update({
                            'finetuning_graphs_available': enhanced.get('finetuning_graphs_available', 0),
                            'pretraining_graphs_used': enhanced.get('pretraining_graphs_used', 0),
                            'family_total_graphs': enhanced.get('family_total_graphs', 0)
                        })
                    
                    models.append(model_info)
                except Exception as e:
                    print(f"Warning: Could not load metadata for {model_id}: {e}")
        
        return sorted(models, key=lambda x: x.get('creation_timestamp', ''), reverse=True)
    
    def get_models_by_family(self, family_id: str) -> List[Dict[str, Any]]:
        """Get all models trained on a specific graph family."""
        all_models = self.list_models()
        return [model for model in all_models if model.get('family_id') == family_id]