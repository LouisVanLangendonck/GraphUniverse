"""
I/O utilities for the MMSB model.

This module provides functions for loading and saving graph data,
handling different file formats, and data conversion between formats.
"""

import os
import pickle
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any

# Try importing optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False


def save_graph(
    graph: nx.Graph,
    filepath: str,
    format: str = "networkx",
    metadata: Optional[Dict] = None
) -> str:
    """
    Save a graph to disk in the specified format.
    
    Args:
        graph: NetworkX graph to save
        filepath: Path to save the graph (without extension)
        format: Format to save in ("networkx", "edgelist", "adjacency", "pyg", "dgl")
        metadata: Additional metadata to save with the graph
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if format == "networkx":
        # Save as NetworkX pickle
        path = f"{filepath}.gpickle"
        nx.write_gpickle(graph, path)
        
        # Save metadata separately if provided
        if metadata is not None:
            meta_path = f"{filepath}_meta.json"
            # Convert numpy arrays to lists for JSON serialization
            meta_json = {}
            for k, v in metadata.items():
                if isinstance(v, np.ndarray):
                    meta_json[k] = v.tolist()
                else:
                    meta_json[k] = v
                    
            with open(meta_path, "w") as f:
                json.dump(meta_json, f, indent=2)
                
        return path
        
    elif format == "edgelist":
        # Save as edge list
        path = f"{filepath}.edgelist"
        nx.write_edgelist(graph, path)
        return path
        
    elif format == "adjacency":
        # Save as adjacency matrix
        path = f"{filepath}.npz"
        A = nx.to_scipy_sparse_array(graph)
        np.savez(
            path,
            data=A.data,
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape
        )
        
        # Save node attributes separately
        node_attrs = {}
        for n, attrs in graph.nodes(data=True):
            node_attrs[n] = attrs
            
        with open(f"{filepath}_node_attrs.pkl", "wb") as f:
            pickle.dump(node_attrs, f)
            
        return path
        
    elif format == "pyg":
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric not installed. Cannot save in PyG format.")
        
        from torch_geometric.data import Data
        
        # Get edges
        edges = list(graph.edges())
        if not edges:
            # Empty graph case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            src, dst = zip(*edges)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Get node features if available
        node_attrs = list(graph.nodes(data=True))
        n_nodes = len(node_attrs)
        
        # Check if all nodes have features with the same dimension
        has_features = True
        feature_dim = None
        
        for _, attrs in node_attrs:
            if "features" in attrs:
                if feature_dim is None:
                    feature_dim = len(attrs["features"])
                elif len(attrs["features"]) != feature_dim:
                    has_features = False
                    break
            else:
                has_features = False
                break
        
        if has_features and feature_dim is not None:
            # Create feature tensor
            x = torch.zeros((n_nodes, feature_dim), dtype=torch.float)
            for i, (node, attrs) in enumerate(node_attrs):
                x[i] = torch.tensor(attrs["features"], dtype=torch.float)
        else:
            # No features or inconsistent features, use one-hot encoding of node ID
            x = torch.eye(n_nodes, dtype=torch.float)
        
        # Check for labels
        has_labels = True
        for _, attrs in node_attrs:
            if "primary_community" not in attrs:
                has_labels = False
                break
        
        if has_labels:
            # Create label tensor
            y = torch.zeros(n_nodes, dtype=torch.long)
            for i, (node, attrs) in enumerate(node_attrs):
                y[i] = attrs["primary_community"]
        else:
            # No labels
            y = None
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index)
        if y is not None:
            data.y = y
        
        # Add memberships if available
        has_memberships = True
        membership_dim = None
        
        for _, attrs in node_attrs:
            if "memberships" in attrs:
                if membership_dim is None:
                    membership_dim = len(attrs["memberships"])
                elif len(attrs["memberships"]) != membership_dim:
                    has_memberships = False
                    break
            else:
                has_memberships = False
                break
        
        if has_memberships and membership_dim is not None:
            # Convert dict format to tensor
            max_community_id = -1
            for _, attrs in node_attrs:
                max_community_id = max(max_community_id, max(attrs["memberships"].keys()))
            
            memberships = torch.zeros((n_nodes, max_community_id + 1), dtype=torch.float)
            for i, (node, attrs) in enumerate(node_attrs):
                for comm, weight in attrs["memberships"].items():
                    memberships[i, comm] = weight
            
            data.memberships = memberships
        
        # Add metadata if provided
        if metadata is not None:
            for key, value in metadata.items():
                if isinstance(value, np.ndarray):
                    setattr(data, key, torch.from_numpy(value))
                else:
                    setattr(data, key, value)
        
        # Save PyG data
        path = f"{filepath}.pt"
        torch.save(data, path)
        return path
        
    elif format == "dgl":
        if not HAS_DGL:
            raise ImportError("DGL not installed. Cannot save in DGL format.")
        
        # Create DGL graph
        if HAS_TORCH:
            g = dgl.graph(([], []))
            g.add_nodes(len(graph))
            
            # Add edges
            if graph.number_of_edges() > 0:
                src, dst = zip(*graph.edges())
                g.add_edges(src, dst)
            
            # Add node features if available
            node_attrs = list(graph.nodes(data=True))
            
            # Check for features
            has_features = True
            feature_dim = None
            
            for _, attrs in node_attrs:
                if "features" in attrs:
                    if feature_dim is None:
                        feature_dim = len(attrs["features"])
                    elif len(attrs["features"]) != feature_dim:
                        has_features = False
                        break
                else:
                    has_features = False
                    break
            
            if has_features and feature_dim is not None:
                # Create feature tensor
                feat = torch.zeros((len(graph), feature_dim), dtype=torch.float)
                for i, (node, attrs) in enumerate(sorted(node_attrs)):
                    feat[i] = torch.tensor(attrs["features"], dtype=torch.float)
                g.ndata["feat"] = feat
            else:
                # No features, use one-hot encoding
                g.ndata["feat"] = torch.eye(len(graph))
            
            # Check for labels
            has_labels = True
            for _, attrs in node_attrs:
                if "primary_community" not in attrs:
                    has_labels = False
                    break
            
            if has_labels:
                # Create label tensor
                labels = torch.zeros(len(graph), dtype=torch.long)
                for i, (node, attrs) in enumerate(sorted(node_attrs)):
                    labels[i] = attrs["primary_community"]
                g.ndata["label"] = labels
            
            # Save metadata as graph attributes
            if metadata is not None:
                for key, value in metadata.items():
                    g.gdata[key] = value
            
            # Save DGL graph
            path = f"{filepath}.dgl"
            dgl.save_graphs(path, [g])
            return path
        else:
            raise ImportError("PyTorch not installed. Cannot save in DGL format.")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_graph(
    filepath: str,
    format: Optional[str] = None,
    load_metadata: bool = True
) -> Union[nx.Graph, Tuple[nx.Graph, Dict]]:
    """
    Load a graph from disk.
    
    Args:
        filepath: Path to the graph file
        format: Format to load from (if None, inferred from file extension)
        load_metadata: Whether to load metadata if available
        
    Returns:
        Loaded graph, or tuple of (graph, metadata) if load_metadata=True
    """
    # Infer format from file extension if not provided
    if format is None:
        ext = os.path.splitext(filepath)[1]
        if ext == ".gpickle":
            format = "networkx"
        elif ext == ".edgelist":
            format = "edgelist"
        elif ext == ".npz":
            format = "adjacency"
        elif ext == ".pt":
            format = "pyg"
        elif ext == ".dgl":
            format = "dgl"
        else:
            raise ValueError(f"Cannot infer format from file extension: {ext}")
    
    # Handle different formats
    if format == "networkx":
        graph = nx.read_gpickle(filepath)
        
        if load_metadata:
            # Check for metadata file
            meta_file = os.path.splitext(filepath)[0] + "_meta.json"
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                return graph, metadata
        
        return graph
    
    elif format == "edgelist":
        graph = nx.read_edgelist(filepath)
        return graph
    
    elif format == "adjacency":
        # Load sparse adjacency matrix
        data = np.load(filepath)
        from scipy import sparse
        A = sparse.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
        
        graph = nx.from_scipy_sparse_array(A)
        
        # Load node attributes if available
        attrs_file = os.path.splitext(filepath)[0] + "_node_attrs.pkl"
        if os.path.exists(attrs_file):
            with open(attrs_file, "rb") as f:
                node_attrs = pickle.load(f)
                
            for node, attrs in node_attrs.items():
                for key, value in attrs.items():
                    graph.nodes[node][key] = value
        
        if load_metadata and os.path.exists(attrs_file):
            return graph, {"format": "adjacency"}
        
        return graph
    
    elif format == "pyg":
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric not installed. Cannot load PyG format.")
        
        # Load PyG data
        data = torch.load(filepath)
        
        # Convert to NetworkX
        edge_index = data.edge_index.numpy()
        n_nodes = data.x.shape[0]
        
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        
        # Add edges
        edges = list(zip(edge_index[0], edge_index[1]))
        graph.add_edges_from(edges)
        
        # Add node features
        for i in range(n_nodes):
            graph.nodes[i]["features"] = data.x[i].numpy()
            
            # Add labels if available
            if hasattr(data, "y"):
                graph.nodes[i]["primary_community"] = int(data.y[i].item())
                
            # Add memberships if available
            if hasattr(data, "memberships"):
                mem_vector = data.memberships[i].numpy()
                # Convert to dict format
                graph.nodes[i]["memberships"] = {
                    j: float(v) for j, v in enumerate(mem_vector) if v > 0
                }
        
        # Extract metadata
        metadata = {}
        for key in data.keys:
            if key not in ["x", "edge_index", "y", "memberships"]:
                metadata[key] = getattr(data, key)
                
        if load_metadata:
            return graph, metadata
            
        return graph
    
    elif format == "dgl":
        if not HAS_DGL:
            raise ImportError("DGL not installed. Cannot load DGL format.")
        
        # Load DGL graph
        gs, _ = dgl.load_graphs(filepath)
        g = gs[0]
        
        # Convert to NetworkX
        graph = nx.Graph()
        graph.add_nodes_from(range(g.num_nodes()))
        
        if g.num_edges() > 0:
            edges = list(zip(g.edges()[0].numpy(), g.edges()[1].numpy()))
            graph.add_edges_from(edges)
        
        # Add node features
        if "feat" in g.ndata:
            for i in range(g.num_nodes()):
                graph.nodes[i]["features"] = g.ndata["feat"][i].numpy()
                
                # Add labels if available
                if "label" in g.ndata:
                    graph.nodes[i]["primary_community"] = int(g.ndata["label"][i].item())
        
        # Extract metadata from graph data
        metadata = {}
        for key in g.gdata:
            metadata[key] = g.gdata[key]
            
        if load_metadata:
            return graph, metadata
            
        return graph
    
    else:
        raise ValueError(f"Unknown format: {format}")


def convert_graph_format(
    graph: Union[nx.Graph, Any],
    input_format: str,
    output_format: str,
    temp_dir: Optional[str] = None
) -> Any:
    """
    Convert a graph between different formats.
    
    Args:
        graph: Input graph in the specified format
        input_format: Format of the input graph
        output_format: Desired output format
        temp_dir: Directory for temporary files (if needed)
        
    Returns:
        Converted graph in the requested format
    """
    # Handle direct conversion if possible
    if input_format == output_format:
        return graph
        
    # For now, use NetworkX as an intermediate format
    if input_format != "networkx":
        if temp_dir is None:
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
        # Save in input format
        temp_file = os.path.join(temp_dir, "temp_graph")
        save_graph(graph, temp_file, format=input_format)
        
        # Load as NetworkX
        nx_graph = load_graph(temp_file + get_extension(input_format), format=input_format)
    else:
        nx_graph = graph
        
    # Convert NetworkX to output format
    if output_format == "networkx":
        return nx_graph
    elif output_format in ["pyg", "dgl", "edgelist", "adjacency"]:
        if temp_dir is None:
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
        # Save in output format
        temp_out = os.path.join(temp_dir, "temp_out")
        save_graph(nx_graph, temp_out, format=output_format)
        
        # Load in output format
        return load_graph(temp_out + get_extension(output_format), format=output_format)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def get_extension(format: str) -> str:
    """
    Get the file extension for a given format.
    
    Args:
        format: Graph format
        
    Returns:
        File extension including the dot
    """
    format_extensions = {
        "networkx": ".gpickle",
        "edgelist": ".edgelist",
        "adjacency": ".npz",
        "pyg": ".pt",
        "dgl": ".dgl"
    }
    
    return format_extensions.get(format, "")


def load_benchmark(
    directory: str,
    load_pretrain: bool = True,
    load_transfer: bool = True,
    format: str = "networkx"
) -> Dict[str, List]:
    """
    Load a benchmark dataset from disk.
    
    Args:
        directory: Base directory containing the benchmark
        load_pretrain: Whether to load pretraining graphs
        load_transfer: Whether to load transfer graphs
        format: Format to load graphs in
        
    Returns:
        Dictionary with "pretrain" and "transfer" lists of loaded graphs
    """
    result = {}
    
    # Load universe metadata
    universe_meta_file = os.path.join(directory, "universe_meta.pkl")
    if os.path.exists(universe_meta_file):
        with open(universe_meta_file, "rb") as f:
            result["universe_meta"] = pickle.load(f)
    
    # Load pretraining graphs
    if load_pretrain:
        pretrain_dir = os.path.join(directory, "pretrain")
        if os.path.exists(pretrain_dir):
            pretrain_graphs = []
            
            # List all graph files
            files = [f for f in os.listdir(pretrain_dir) if f.startswith("pretrain_") and not f.endswith("_meta.pkl")]
            
            for f in sorted(files):
                path = os.path.join(pretrain_dir, f)
                graph = load_graph(path, format=format)
                pretrain_graphs.append(graph)
                
            result["pretrain"] = pretrain_graphs
    
    # Load transfer graphs
    if load_transfer:
        transfer_dir = os.path.join(directory, "transfer")
        if os.path.exists(transfer_dir):
            transfer_graphs = []
            
            # List all graph files
            files = [f for f in os.listdir(transfer_dir) if f.startswith("transfer_") and not f.endswith("_meta.pkl")]
            
            for f in sorted(files):
                path = os.path.join(transfer_dir, f)
                graph = load_graph(path, format=format)
                transfer_graphs.append(graph)
                
            result["transfer"] = transfer_graphs
    
    return result