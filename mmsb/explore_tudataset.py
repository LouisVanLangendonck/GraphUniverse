from torch_geometric.datasets import PPI, Twitch, MalNetTiny
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import os

def calculate_homophily(edge_index, node_labels):
    """Calculate edge homophily ratio - fraction of edges that connect nodes with same label"""
    if node_labels is None:
        return None
    
    # Get source and target nodes for each edge
    src, dst = edge_index
    
    # Count edges where nodes have same label
    same_label = (node_labels[src] == node_labels[dst]).sum().item()
    total_edges = len(src)
    
    return same_label / total_edges if total_edges > 0 else 0

def explore_ppi():
    print("\nExploring PPI dataset...")
    try:
        # Load PPI dataset with train/val/test splits
        train_dataset = PPI(root='./data/PPI', split='train')
        val_dataset = PPI(root='./data/PPI', split='val')
        test_dataset = PPI(root='./data/PPI', split='test')
        
        print("PPI Inductive Node Classification:")
        print("Number of graphs in training set:", len(train_dataset))
        print("Number of graphs in validation set:", len(val_dataset))
        print("Number of graphs in test set:", len(test_dataset))
        
        # Examine first graph from training set
        data = train_dataset[0]
        print("\nFirst graph structure:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Features per node: {data.num_node_features}")
        print(f"Node classes (multi-label): {data.y.shape[1]}")
        
        # Calculate homophily for first graph
        # For multi-label case, we'll consider nodes with any matching label as similar
        node_labels = data.y.argmax(dim=1)  # Convert multi-label to single label for homophily
        homophily = calculate_homophily(data.edge_index, node_labels)
        print(f"\nEdge homophily ratio: {homophily:.4f}")
        
    except Exception as e:
        print(f"Error exploring PPI: {str(e)}")
        import traceback
        traceback.print_exc()

def explore_twitch():
    print("\nExploring Twitch dataset...")
    try:
        # Load Twitch dataset
        dataset = Twitch(root='./data/Twitch', name='EN')
        data = dataset[0]
        
        print("\nTwitch graph structure:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        
        if hasattr(data, 'y') and data.y is not None:
            print(f"\nNode labels (y):")
            print(f"Shape: {data.y.shape}")
            print(f"Number of unique labels: {len(torch.unique(data.y))}")
            print(f"Label distribution: {torch.bincount(data.y)}")
            
            # Calculate homophily
            homophily = calculate_homophily(data.edge_index, data.y)
            print(f"\nEdge homophily ratio: {homophily:.4f}")
        
        if hasattr(data, 'x') and data.x is not None:
            print(f"\nNode features:")
            print(f"Shape: {data.x.shape}")
            print(f"Type: {data.x.dtype}")
        
    except Exception as e:
        print(f"Error exploring Twitch: {str(e)}")
        import traceback
        traceback.print_exc()

def explore_malnettiny():
    print("\nExploring MalNetTiny dataset...")
    try:
        # Load MalNetTiny dataset
        dataset = MalNetTiny(root='./data/MalNetTiny')
        print(f"Total graphs: {len(dataset)}")
        
        # Examine first graph
        data = dataset[0]
        print("\nFirst graph structure:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        
        # MalNetTiny is a graph classification dataset
        if hasattr(data, 'y') and data.y is not None:
            print(f"\nGraph label (y):")
            print(f"Shape: {data.y.shape}")
            print(f"Value: {data.y.item()}")
        
        if hasattr(data, 'x') and data.x is not None:
            print(f"\nNode features:")
            print(f"Shape: {data.x.shape}")
            print(f"Type: {data.x.dtype}")
            
        # Print some statistics about the dataset
        print("\nDataset statistics:")
        num_nodes_list = [data.num_nodes for data in dataset]
        num_edges_list = [data.num_edges for data in dataset]
        print(f"Average number of nodes: {sum(num_nodes_list) / len(num_nodes_list):.2f}")
        print(f"Average number of edges: {sum(num_edges_list) / len(num_edges_list):.2f}")
        print(f"Min nodes: {min(num_nodes_list)}")
        print(f"Max nodes: {max(num_nodes_list)}")
        print(f"Min edges: {min(num_edges_list)}")
        print(f"Max edges: {max(num_edges_list)}")
        
    except Exception as e:
        print(f"Error exploring MalNetTiny: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting dataset exploration...")
    
    # Explore PPI dataset
    explore_ppi()
    
    # Explore Twitch dataset
    explore_twitch()
    
    # Explore MalNetTiny dataset
    explore_malnettiny() 