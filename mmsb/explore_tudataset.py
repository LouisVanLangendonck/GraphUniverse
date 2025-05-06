from torch_geometric.datasets import TUDataset
import torch
import numpy as np
import os

def explore_dataset(name):
    print(f"\nExploring {name} dataset...")
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs(f'./data/{name}', exist_ok=True)
        
        # Load dataset with all attributes
        print("Loading dataset...")
        dataset = TUDataset(
            root=f'./data/{name}',
            name=name,
            use_node_attr=True,
            use_edge_attr=True,
            cleaned=True
        )
        
        print(f"Dataset size: {len(dataset)} graphs")
        
        # Examine first graph
        data = dataset[0]
        print("\nFirst graph structure:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        
        # Check all available attributes
        print("\nAvailable attributes:")
        for attr in dir(data):
            if not attr.startswith('_'):
                value = getattr(data, attr)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        print(f"{attr}: Tensor of shape {value.shape}")
                    else:
                        print(f"{attr}: {type(value)}")
        
        # Examine node features
        if hasattr(data, 'x') and data.x is not None:
            print("\nNode features (x):")
            print(f"Shape: {data.x.shape}")
            print(f"Type: {data.x.dtype}")
            print(f"Sample values:\n{data.x[:5]}")
            
            # Check if features are one-hot encoded
            if data.x.shape[1] > 1:
                print("\nChecking if features are one-hot encoded...")
                unique_per_col = [torch.unique(data.x[:, i]) for i in range(data.x.shape[1])]
                print("Unique values per column:", unique_per_col)
        
        # Examine edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print("\nEdge features (edge_attr):")
            print(f"Shape: {data.edge_attr.shape}")
            print(f"Type: {data.edge_attr.dtype}")
            print(f"Sample values:\n{data.edge_attr[:5]}")
        
        # Examine graph labels
        if hasattr(data, 'y') and data.y is not None:
            print("\nGraph labels (y):")
            print(f"Shape: {data.y.shape}")
            print(f"Type: {data.y.dtype}")
            print(f"Value: {data.y}")
        
        # Check if there are node labels
        if hasattr(data, 'node_labels') and data.node_labels is not None:
            print("\nNode labels:")
            print(f"Shape: {data.node_labels.shape}")
            print(f"Type: {data.node_labels.dtype}")
            print(f"Sample values:\n{data.node_labels[:5]}")
        
        # Print unique values in node features
        if hasattr(data, 'x') and data.x is not None:
            unique_values = torch.unique(data.x)
            print(f"\nUnique values in node features: {unique_values}")
            
            # If one-hot encoded, show the distribution
            if data.x.shape[1] > 1:
                print("\nDistribution of node types (one-hot encoding):")
                for i in range(data.x.shape[1]):
                    count = (data.x[:, i] == 1).sum().item()
                    print(f"Type {i}: {count} nodes")
    
    except Exception as e:
        print(f"Error exploring {name}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting dataset exploration...")
    
    # Explore MUTAG dataset
    explore_dataset('MUTAG')
    
    # Explore ENZYMES dataset
    explore_dataset('ENZYMES') 