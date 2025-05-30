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

from experiments.inductive.config import PreTrainingConfig
from experiments.core.models import GNNModel


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


class LinkPredictionTask(SelfSupervisedTask):
    """Link prediction self-supervised task."""
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create GNN model with link prediction head."""
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


def create_ssl_task(config: PreTrainingConfig) -> SelfSupervisedTask:
    """Create SSL task based on configuration."""
    if config.pretraining_task == "link_prediction":
        return LinkPredictionTask(config)
    elif config.pretraining_task == "contrastive":
        return ContrastiveTask(config)
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
            model_id = f"{config.gnn_type}_{config.pretraining_task}_{timestamp}"
        
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
        
        return model, metadata
    
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