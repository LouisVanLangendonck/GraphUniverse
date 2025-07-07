import torch
import numpy as np
from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh

class PositionalEncodings:
    def __init__(self, max_pe_dim=6):
        self.max_pe_dim = max_pe_dim
    
    def compute_degree_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Degree-based PE (your original method)."""
        degrees = degree(edge_index[0], num_nodes=num_nodes).float()
        pe = torch.zeros(num_nodes, self.max_pe_dim)
        
        for i in range(min(self.max_pe_dim, 8)):
            pe[:, i] = (degrees ** (i / 4.0)) / (1 + degrees ** (i / 4.0))
        
        return pe
    
    def compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Random Walk Structural Encoding - landing probabilities after k steps."""
        try:
            # Build transition matrix P where P[i,j] = prob of going from i to j
            degrees = degree(edge_index[0], num_nodes=num_nodes).float()
            
            # Handle isolated nodes
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
            
            # Create adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            # Transition matrix: P[i,j] = A[i,j] / degree[i]
            P = adj / degrees.unsqueeze(1)
            
            # Compute powers of transition matrix for different walk lengths
            rwse = torch.zeros(num_nodes, self.max_pe_dim)
            P_power = torch.eye(num_nodes)  # P^0 = I
            
            for k in range(self.max_pe_dim):
                if k > 0:
                    P_power = P_power @ P  # P^k
                
                # Use diagonal entries (return probabilities) as features
                rwse[:, k] = P_power.diag()
            
            return rwse
            
        except Exception as e:
            print(f"Warning: RWSE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)
    
    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Laplacian Positional Encoding using eigenvectors."""
        try:
            # Handle empty/trivial graphs
            if edge_index.shape[1] == 0 or num_nodes <= 1:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            # Get normalized Laplacian
            edge_index_lap, edge_weight = get_laplacian(
                edge_index, 
                edge_weight=None,
                normalization='sym', 
                num_nodes=num_nodes
            )
            
            # Convert to scipy sparse matrix
            L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
            
            # Compute eigenvalues/eigenvectors
            k = min(self.max_pe_dim, num_nodes - 2)
            if k <= 0:
                return torch.zeros(num_nodes, self.max_pe_dim)
            
            try:
                eigenvals, eigenvecs = eigsh(
                    L, 
                    k=k, 
                    which='SM',  # Smallest eigenvalues
                    return_eigenvectors=True,
                    tol=1e-6
                )
            except:
                # Fallback for small graphs
                L_dense = L.toarray()
                eigenvals, eigenvecs = np.linalg.eigh(L_dense)
                idx = np.argsort(eigenvals)
                eigenvecs = eigenvecs[:, idx[1:k+1]]  # Skip first (constant) eigenvector
            
            # Handle sign ambiguity
            for i in range(eigenvecs.shape[1]):
                if eigenvecs[0, i] < 0:
                    eigenvecs[:, i] *= -1
            
            # Pad or truncate to max_pe_dim
            if eigenvecs.shape[1] < self.max_pe_dim:
                pad_width = self.max_pe_dim - eigenvecs.shape[1]
                eigenvecs = np.pad(eigenvecs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                eigenvecs = eigenvecs[:, :self.max_pe_dim]
            
            return torch.tensor(eigenvecs, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: Laplacian PE computation failed: {e}")
            return torch.zeros(num_nodes, self.max_pe_dim)

def create_test_graphs():
    """Create three test graphs with different structures."""
    graphs = {}
    
    # Graph 1: Triangle (3 nodes)
    graphs['triangle'] = {
        'edge_index': torch.tensor([
            [0, 1, 2, 1, 2, 0],  # bidirectional edges
            [1, 2, 0, 0, 1, 2]
        ]),
        'num_nodes': 3,
        'description': 'Triangle: all nodes degree 2, highly connected'
    }
    
    # Graph 2: Square (4 nodes)
    graphs['square'] = {
        'edge_index': torch.tensor([
            [0, 1, 2, 3, 1, 2, 3, 0],  # bidirectional edges
            [1, 2, 3, 0, 0, 1, 0, 3]
        ]),
        'num_nodes': 4,
        'description': 'Square: all nodes degree 2, cycle structure'
    }
    
    # Graph 3: Line + Fork (5 nodes)
    # Structure: 0-1-2-3-4 where 3 connects to both 2 and 4
    graphs['line_fork'] = {
        'edge_index': torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],  # bidirectional edges
            [1, 0, 2, 1, 3, 2, 4, 3]
        ]),
        'num_nodes': 5,
        'description': 'Line+Fork: 0-1-2-3-4, varied degrees (1,2,2,2,1)'
    }
    
    return graphs

def analyze_pe_methods():
    """Compare PE methods across different graph structures."""
    pe_computer = PositionalEncodings(max_pe_dim=6)
    graphs = create_test_graphs()
    
    print("=" * 80)
    print("POSITIONAL ENCODING COMPARISON ACROSS GRAPHS")
    print("=" * 80)
    
    for graph_name, graph_data in graphs.items():
        edge_index = graph_data['edge_index']
        num_nodes = graph_data['num_nodes']
        description = graph_data['description']
        
        print(f"\n{'='*20} {graph_name.upper()} {'='*20}")
        print(f"Description: {description}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edges: {edge_index.t().tolist()}")
        
        # Compute node degrees for reference
        degrees = degree(edge_index[0], num_nodes=num_nodes)
        print(f"Node degrees: {degrees.tolist()}")
        
        # Compute all PE methods
        degree_pe = pe_computer.compute_degree_pe(edge_index, num_nodes)
        rwse_pe = pe_computer.compute_rwse(edge_index, num_nodes)
        laplacian_pe = pe_computer.compute_laplacian_pe(edge_index, num_nodes)
        
        print(f"\n--- DEGREE-BASED PE ---")
        print(f"Shape: {degree_pe.shape}")
        for i in range(num_nodes):
            print(f"Node {i}: {degree_pe[i].numpy().round(3)}")
        
        print(f"\n--- RANDOM WALK SE ---")
        print(f"Shape: {rwse_pe.shape}")
        for i in range(num_nodes):
            print(f"Node {i}: {rwse_pe[i].numpy().round(3)}")
        
        print(f"\n--- LAPLACIAN PE ---")
        print(f"Shape: {laplacian_pe.shape}")
        for i in range(num_nodes):
            print(f"Node {i}: {laplacian_pe[i].numpy().round(3)}")
    
    print("\n" + "=" * 80)
    print("CROSS-GRAPH ANALYSIS")
    print("=" * 80)
    
    # Analyze how similar structural roles get encoded
    print("\nNodes with degree 2 across graphs:")
    
    graphs_list = list(graphs.items())
    for graph_name, graph_data in graphs_list:
        edge_index = graph_data['edge_index']
        num_nodes = graph_data['num_nodes']
        
        degrees = degree(edge_index[0], num_nodes=num_nodes)
        degree_2_nodes = (degrees == 2).nonzero().flatten()
        
        if len(degree_2_nodes) > 0:
            degree_pe = pe_computer.compute_degree_pe(edge_index, num_nodes)
            rwse_pe = pe_computer.compute_rwse(edge_index, num_nodes)
            laplacian_pe = pe_computer.compute_laplacian_pe(edge_index, num_nodes)
            
            print(f"\n{graph_name.upper()} - Degree 2 nodes: {degree_2_nodes.tolist()}")
            
            for node_idx in degree_2_nodes:
                print(f"  Node {node_idx}:")
                print(f"    Degree PE:   {degree_pe[node_idx].numpy().round(3)}")
                print(f"    RWSE:        {rwse_pe[node_idx].numpy().round(3)}")
                print(f"    Laplacian:   {laplacian_pe[node_idx].numpy().round(3)}")
    
    print(f"\n{'='*40}")
    print("KEY OBSERVATIONS:")
    print(f"{'='*40}")
    print("1. DEGREE PE: Identical for nodes with same degree across all graphs")
    print("2. RWSE: Captures local random walk patterns, varies by graph structure")
    print("3. LAPLACIAN PE: Graph-specific basis, hard to compare across graphs")
    print("4. For inductive learning: Degree PE most transferable, RWSE moderate, Laplacian PE least")

def detailed_structural_analysis():
    """Detailed analysis of what each method captures."""
    pe_computer = PositionalEncodings(max_pe_dim=4)  # Smaller for clarity
    graphs = create_test_graphs()
    
    print("\n" + "=" * 60)
    print("DETAILED STRUCTURAL ANALYSIS")
    print("=" * 60)
    
    # Focus on line_fork graph for structural diversity
    graph_data = graphs['line_fork']
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    
    print("Graph: Line+Fork (0-1-2-3-4)")
    print("Node roles:")
    print("  Node 0: Terminal (degree 1)")
    print("  Node 1: Bridge (degree 2)")  
    print("  Node 2: Bridge (degree 2)")
    print("  Node 3: Branch point (degree 2)")
    print("  Node 4: Terminal (degree 1)")
    
    degree_pe = pe_computer.compute_degree_pe(edge_index, num_nodes)
    rwse_pe = pe_computer.compute_rwse(edge_index, num_nodes)
    laplacian_pe = pe_computer.compute_laplacian_pe(edge_index, num_nodes)
    
    print(f"\nPositional Encodings (first 4 dimensions):")
    print(f"{'Node':<4} {'Role':<12} {'Degree PE':<25} {'RWSE':<25} {'Laplacian PE':<25}")
    print("-" * 95)
    
    roles = ['Terminal', 'Bridge', 'Bridge', 'Branch', 'Terminal']
    for i in range(num_nodes):
        deg_str = f"[{', '.join(f'{x:.2f}' for x in degree_pe[i][:4])}]"
        rwse_str = f"[{', '.join(f'{x:.2f}' for x in rwse_pe[i][:4])}]"
        lap_str = f"[{', '.join(f'{x:.2f}' for x in laplacian_pe[i][:4])}]"
        
        print(f"{i:<4} {roles[i]:<12} {deg_str:<25} {rwse_str:<25} {lap_str:<25}")

def test_data_pe_implementation():
    """Test the PE implementation from data.py"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))
    
    from experiments.inductive.data import PositionalEncodingComputer
    
    print("\n" + "=" * 60)
    print("TESTING DATA.PY PE IMPLEMENTATION")
    print("=" * 60)
    
    # Create test graphs
    graphs = create_test_graphs()
    
    # Test the new PE computer
    pe_computer = PositionalEncodingComputer(max_pe_dim=6, pe_types=['laplacian', 'degree', 'rwse'])
    
    for graph_name, graph_data in graphs.items():
        edge_index = graph_data['edge_index']
        num_nodes = graph_data['num_nodes']
        
        print(f"\nTesting {graph_name} graph:")
        print(f"  Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
        
        # Test all PE types
        pe_dict = pe_computer.compute_all_pe(edge_index, num_nodes)
        
        for pe_name, pe_tensor in pe_dict.items():
            print(f"  {pe_name}: shape {pe_tensor.shape}")
            print(f"    Values: {pe_tensor.numpy().round(3)}")
        
        print()

if __name__ == "__main__":
    analyze_pe_methods()
    detailed_structural_analysis()
    test_data_pe_implementation()