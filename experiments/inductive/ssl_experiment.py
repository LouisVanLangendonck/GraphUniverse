class SSLExperiment:
    """Self-supervised learning experiment for graph neural networks."""
    
    def __init__(self, config: PreTrainingConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() and not config.force_cpu else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Initialize graph family
        self.graph_family = None
        self.train_graphs = None
        self.val_graphs = None
        self.test_graphs = None
        
        # Initialize model
        self.model = None
        self.optimizer = None
        
    def _create_model(self, input_dim: int) -> torch.nn.Module:
        """Create the model based on configuration."""
        if self.config.model_type == "transformer":
            return GraphTransformerModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.transformer_num_heads,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                dropout=self.config.dropout,
                local_gnn_type=self.config.local_gnn_type,
                global_model_type=self.config.global_model_type,
                transformer_prenorm=self.config.transformer_prenorm
            )
        else:  # model_type == "gnn"
            return GNNModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                gnn_type=self.config.gnn_type,
                dropout=self.config.dropout
            )
    
    def _precompute_transformer_encodings(self, graphs: List[Data]) -> List[Data]:
        """Precompute transformer encodings for a list of graphs."""
        if not self.config.run_transformers or not self.config.transformer_precompute_encodings:
            return graphs
            
        print("Precomputing transformer encodings...")
        encoded_graphs = []
        for graph in tqdm(graphs):
            encoded_graph = precompute_transformer_encodings(
                graph,
                max_nodes=self.config.transformer_max_nodes,
                max_path_length=self.config.transformer_max_path_length,
                cache=self.config.transformer_cache_encodings
            )
            encoded_graphs.append(encoded_graph)
        return encoded_graphs
    
    def run(self):
        """Run the SSL experiment."""
        # Generate or load graph family
        self._setup_graph_family()
        
        # Split graphs into train/val/test
        self._split_graphs()
        
        # Precompute transformer encodings if needed
        if self.config.run_transformers and self.config.transformer_precompute_encodings:
            self.train_graphs = self._precompute_transformer_encodings(self.train_graphs)
            self.val_graphs = self._precompute_transformer_encodings(self.val_graphs)
            self.test_graphs = self._precompute_transformer_encodings(self.test_graphs)
        
        # Create model
        input_dim = self.train_graphs[0].x.size(1)
        self.model = self._create_model(input_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Train model
        self._train()
        
        # Save results
        self._save_results()
        
    def _train(self):
        """Train the model using the specified pre-training task."""
        if self.config.pretraining_task == "link_prediction":
            self._train_link_prediction()
        elif self.config.pretraining_task == "contrastive":
            self._train_contrastive()
        else:
            raise ValueError(f"Unknown pre-training task: {self.config.pretraining_task}")
    
    def _train_link_prediction(self):
        """Train the model using link prediction."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nStarting pre-training with patience={self.config.patience}")
        print("Early stopping will trigger if validation loss doesn't improve for", self.config.patience, "epochs")
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            total_loss = 0
            for graph in self.train_graphs:
                graph = graph.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                node_embeddings = self.model(graph)
                
                # Sample positive and negative edges
                pos_edge_index, neg_edge_index = sample_edges(
                    graph.edge_index,
                    num_neg_samples=int(graph.edge_index.size(1) * self.config.negative_sampling_ratio)
                )
                
                # Compute loss
                loss = link_prediction_loss(
                    node_embeddings,
                    pos_edge_index,
                    neg_edge_index,
                    loss_type=self.config.link_pred_loss
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Validation
            val_loss = self._validate_link_prediction()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint("best_model.pt")
                # print(f"✅ New best validation loss: {best_val_loss:.4f} (patience reset to 0)")
            else:
                patience_counter += 1
                # print(f"❌ No improvement for {patience_counter}/{self.config.patience} epochs (best: {best_val_loss:.4f})")
                if patience_counter >= self.config.patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch}!")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(self.train_graphs):.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model weights")
    
    def _train_contrastive(self):
        """Train the model using contrastive learning."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nStarting pre-training with patience={self.config.patience}")
        print("Early stopping will trigger if validation loss doesn't improve for", self.config.patience, "epochs")
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            total_loss = 0
            for graph in self.train_graphs:
                graph = graph.to(self.device)
                self.optimizer.zero_grad()
                
                # Create corrupted view
                corrupted_graph = corrupt_graph(
                    graph,
                    corruption_type=self.config.corruption_type,
                    corruption_rate=self.config.corruption_rate
                )
                
                # Forward pass
                node_embeddings = self.model(graph)
                corrupted_embeddings = self.model(corrupted_graph)
                
                # Compute loss
                loss = contrastive_loss(
                    node_embeddings,
                    corrupted_embeddings,
                    temperature=self.config.contrastive_temperature
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Validation
            val_loss = self._validate_contrastive()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint("best_model.pt")
                print(f"✅ New best validation loss: {best_val_loss:.4f} (patience reset to 0)")
            else:
                patience_counter += 1
                print(f"❌ No improvement for {patience_counter}/{self.config.patience} epochs (best: {best_val_loss:.4f})")
                if patience_counter >= self.config.patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch}!")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(self.train_graphs):.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model weights")
    
    def _validate_link_prediction(self) -> float:
        """Validate the model using link prediction."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for graph in self.val_graphs:
                graph = graph.to(self.device)
                
                # Forward pass
                node_embeddings = self.model(graph)
                
                # Sample positive and negative edges
                pos_edge_index, neg_edge_index = sample_edges(
                    graph.edge_index,
                    num_neg_samples=int(graph.edge_index.size(1) * self.config.negative_sampling_ratio)
                )
                
                # Compute loss
                loss = link_prediction_loss(
                    node_embeddings,
                    pos_edge_index,
                    neg_edge_index,
                    loss_type=self.config.link_pred_loss
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_graphs)
    
    def _validate_contrastive(self) -> float:
        """Validate the model using contrastive learning."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for graph in self.val_graphs:
                graph = graph.to(self.device)
                
                # Create corrupted view
                corrupted_graph = corrupt_graph(
                    graph,
                    corruption_type=self.config.corruption_type,
                    corruption_rate=self.config.corruption_rate
                )
                
                # Forward pass
                node_embeddings = self.model(graph)
                corrupted_embeddings = self.model(corrupted_graph)
                
                # Compute loss
                loss = contrastive_loss(
                    node_embeddings,
                    corrupted_embeddings,
                    temperature=self.config.contrastive_temperature
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_graphs)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def _save_config(self):
        """Save experiment configuration."""
        config_dict = asdict(self.config)
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def _save_results(self):
        """Save experiment results."""
        results = {
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=4) 