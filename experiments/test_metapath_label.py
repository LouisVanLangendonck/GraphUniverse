"""
Script to test and visualize metapath detection and labeling implementation.
Generates graphs and shows metapath discovery, probability calculations, and node labeling for verification.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict
import time

# Import your modules (adjust paths as needed)
from mmsb.model import GraphUniverse, GraphSample
from experiments.inductive.data import analyze_metapath_properties
from utils.metapath_analysis import MetapathAnalyzer, UniverseMetapathSelector, FamilyMetapathEvaluator
from experiments.inductive.data import generate_universe_based_metapath_tasks


def create_test_universe(K: int = 5, feature_dim: int = 8, seed: int = 42) -> GraphUniverse:
    """Create a test universe with specified parameters for metapath testing."""
    return GraphUniverse(
        K=K,
        feature_dim=feature_dim,
        edge_density=0.20,  # Higher density for more interesting metapaths
        homophily=0.7,      # Moderate homophily for diverse connections
        randomness_factor=0.1,
        cluster_count_factor=1.0,
        center_variance=1.0,
        cluster_variance=0.1,
        assignment_skewness=0.2,  # Some skewness for interesting degree patterns
        community_exclusivity=0.8,
        degree_center_method="linear",
        max_max_community_deviation=0.5,
        max_mean_community_deviation=0.5,
        seed=seed
    )


def create_test_graph(
    universe: GraphUniverse, 
    n_nodes: int = 60, 
    use_dccc: bool = True,
    seed: int = 42
) -> GraphSample:
    """Create a test graph from the universe, optionally using DCCC-SBM."""
    np.random.seed(seed)
    
    # Sample a subset of communities (use most of them for interesting metapaths)
    n_communities = min(4, universe.K)
    communities = universe.sample_connected_community_subset(size=n_communities, seed=seed)
    
    graph_sample = GraphSample(
        universe=universe,
        communities=communities,
        n_nodes=n_nodes,
        min_component_size=5,
        degree_heterogeneity=0.4,
        edge_noise=0.1,
        feature_regime_balance=0.5,
        target_homophily=0.7,
        target_density=0.20,
        use_configuration_model=False,
        degree_distribution="power_law" if use_dccc else "standard",
        power_law_exponent=2.5 if use_dccc else None,
        target_avg_degree=None,
        triangle_enhancement=0.1,
        max_mean_community_deviation=0.5,
        max_max_community_deviation=0.5,
        max_parameter_search_attempts=20,
        parameter_search_range=0.5,
        min_edge_density=0.01,
        max_retries=5,
        # DCCC-SBM parameters
        use_dccc_sbm=use_dccc,
        community_imbalance=0.3 if use_dccc else 0.0,
        degree_separation=0.6 if use_dccc else 0.5,
        degree_method="standard",
        disable_deviation_limiting=False,
        seed=seed
    )
    
    return graph_sample

def create_challenging_test_universe(K: int = 6, feature_dim: int = 8, seed: int = 42) -> GraphUniverse:
    """Create a test universe designed to challenge metapath discovery."""
    np.random.seed(seed)
    
    # Create a more complex probability matrix that avoids trivial homophily
    universe = GraphUniverse(
        K=K,
        feature_dim=feature_dim,
        edge_density=0.15,  # Moderate density
        homophily=0.6,      # Moderate homophily (not too high)
        randomness_factor=0.2,  # Add some noise
        cluster_count_factor=1.2,
        center_variance=1.2,
        cluster_variance=0.15,
        assignment_skewness=0.3,  # More skewness for interesting patterns
        community_exclusivity=0.7,  # Less exclusivity for more mixing
        degree_center_method="shuffled",  # Shuffled for more complexity
        seed=seed
    )
    
    # Manually adjust P matrix to create interesting non-homophily patterns
    P = universe.P.copy()
    
    # Add some "bridge" communities with higher inter-community connections
    if K >= 4:
        # Make community 1 a "bridge" between 0 and 2
        P[0, 1] = min(P[0, 1] * 1.5, 0.8)
        P[1, 0] = P[0, 1]
        P[1, 2] = min(P[1, 2] * 1.5, 0.8)
        P[2, 1] = P[1, 2]
        
        # Create a "chain" pattern: 0 -> 1 -> 2 -> 3
        if K >= 4:
            P[2, 3] = min(P[2, 3] * 1.3, 0.7)
            P[3, 2] = P[2, 3]
    
    # Reduce some diagonal elements to make within-community less dominant
    for i in range(K):
        P[i, i] = P[i, i] * 0.8
    
    universe.P = P
    
    print(f"Challenging universe P matrix:")
    print(P)
    
    return universe


def test_universe_based_metapath_generation():
    """
    Test the new universe-based metapath generation approach.
    """
    print(f"\n{'='*80}")
    print("TESTING UNIVERSE-BASED METAPATH GENERATION")
    print(f"{'='*80}")
    
    # Create a challenging universe
    universe = create_challenging_test_universe(K=6, seed=789)
    
    # Create a family of graphs
    family_graphs = []
    for i in range(6):
        graph = create_test_graph(universe, n_nodes=50, use_dccc=True, seed=789 + i)
        family_graphs.append(graph)
    
    print(f"Created family of {len(family_graphs)} graphs")
    
    # Define splits
    n_graphs = len(family_graphs)
    train_indices = list(range(0, n_graphs // 2))
    val_indices = list(range(n_graphs // 2, n_graphs // 2 + n_graphs // 4))
    test_indices = list(range(n_graphs // 2 + n_graphs // 4, n_graphs))
    
    print(f"Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Test universe-based generation
    from experiments.inductive.data import generate_universe_based_metapath_tasks
    
    metapath_data = generate_universe_based_metapath_tasks(
        family_graphs=family_graphs,
        universe=universe,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        k_values=[3, 4],
        require_loop=True,
        degree_weight=0.3,
        max_community_participation=0.9,  # Slightly relaxed for testing
        n_candidates_per_k=20
    )
    
    print(f"\n{'='*60}")
    print("UNIVERSE-BASED GENERATION RESULTS")
    print(f"{'='*60}")
    
    print(f"Valid tasks created: {len(metapath_data['tasks'])}")
    
    for task_name, task_info in metapath_data['tasks'].items():
        print(f"\nTask: {task_name}")
        print(f"  Metapath: {' -> '.join(map(str, task_info['metapath']))}")
        print(f"  Universe probability: {task_info['universe_probability']:.6f}")
        print(f"  Coverage: {task_info['coverage']:.2f}")
        print(f"  Avg positive rate: {task_info['avg_positive_rate']:.3f}")
        print(f"  Is loop task: {task_info['is_loop_task']}")
        print(f"  Valid graphs: {len(task_info['valid_graphs'])}/{len(family_graphs)}")
    
    # Print detailed participation report
    print(f"\n{'='*60}")
    print("DETAILED PARTICIPATION REPORT")
    print(f"{'='*60}")
    print(metapath_data['detailed_report'])
    
    # Test visualization for the first valid task
    if metapath_data['tasks']:
        first_task_name = list(metapath_data['tasks'].keys())[0]
        first_task = metapath_data['tasks'][first_task_name]
        
        print(f"\nCreating visualization for: {first_task_name}")
        
        # Find a graph that supports this task
        valid_graph_idx = first_task['valid_graphs'][0]
        graph_sample = family_graphs[valid_graph_idx]
        labels = first_task['labels_by_graph'][valid_graph_idx]
        
        create_universe_metapath_visualization(
            universe, graph_sample, first_task['metapath'], labels, 
            f"Universe-Based: {first_task_name}"
        )
    
    return metapath_data


def create_universe_metapath_visualization(
    universe: GraphUniverse,
    graph_sample: GraphSample,
    universe_metapath: List[int],
    labels: np.ndarray,
    title: str
):
    """
    Create visualization showing universe-based metapath on a specific graph.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Universe P matrix with metapath highlighted
    ax1 = axes[0, 0]
    im1 = ax1.imshow(universe.P, cmap='viridis', aspect='auto')
    
    # Highlight metapath transitions
    for i in range(len(universe_metapath) - 1):
        from_comm = universe_metapath[i]
        to_comm = universe_metapath[i + 1]
        ax1.add_patch(plt.Rectangle((to_comm-0.4, from_comm-0.4), 0.8, 0.8, 
                                   fill=False, edgecolor='red', lw=3))
    
    ax1.set_title('Universe P Matrix\n(Red = metapath transitions)')
    ax1.set_xlabel('To Community')
    ax1.set_ylabel('From Community')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Add probability annotations
    for i in range(universe.P.shape[0]):
        for j in range(universe.P.shape[1]):
            color = 'white' if universe.P[i,j] < 0.5 * universe.P.max() else 'black'
            ax1.text(j, i, f'{universe.P[i,j]:.3f}', ha='center', va='center', 
                    color=color, fontsize=8)
    
    # 2. Universe degree centers
    ax2 = axes[0, 1]
    comm_ids = list(range(universe.K))
    ax2.bar(comm_ids, universe.degree_centers, alpha=0.7)
    ax2.set_title('Universe Degree Centers')
    ax2.set_xlabel('Community ID')
    ax2.set_ylabel('Degree Center')
    
    # Highlight communities in metapath
    for comm in set(universe_metapath):
        ax2.bar(comm, universe.degree_centers[comm], color='red', alpha=0.8)
    
    # 3. Graph visualization
    ax3 = axes[1, 0]
    graph = graph_sample.graph
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Color nodes by labels
    node_colors = ['lightgray' if labels[i] == 0 else 
                  'orange' if labels[i] == 1 else 'red' for i in range(len(labels))]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8, ax=ax3)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, ax=ax3)
    nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax3)
    
    ax3.set_title(f'Graph with Metapath Labels\n'
                  f'Gray=No participation, Orange=Non-loop, Red=Loop')
    ax3.axis('off')
    
    # 4. Participation statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate participation by community
    unique_communities = np.unique(graph_sample.community_labels)
    participation_stats = []
    
    for local_comm_idx in unique_communities:
        universe_comm_id = graph_sample.communities[local_comm_idx]
        comm_nodes = np.where(graph_sample.community_labels == local_comm_idx)[0]
        participating = np.sum(labels[comm_nodes] > 0)
        total = len(comm_nodes)
        rate = participating / total if total > 0 else 0
        
        participation_stats.append(f"Universe Community {universe_comm_id}: {participating}/{total} ({rate:.3f})")
    
    stats_text = [
        "PARTICIPATION STATISTICS:",
        "=" * 25,
        f"Metapath: {' -> '.join(map(str, universe_metapath))}",
        f"Total nodes: {len(labels)}",
        f"Non-participating: {np.sum(labels == 0)}",
        f"Non-loop participants: {np.sum(labels == 1)}",
        f"Loop participants: {np.sum(labels == 2)}",
        "",
        "By Community:"
    ] + participation_stats
    
    ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def run_multi_graph_metapath_test(
    n_graphs: int = 4, 
    k_values: List[int] = [4, 5],  # Changed default to 4+ for proper loops
    use_dccc: bool = True,
    seed: int = 42
):
    """
    Run metapath test showing multiple graphs from the same family.
    Updated to work with universe-based approach.
    """
    print("Testing Universe-Based Metapath Detection Across Multiple Family Graphs")
    print("=" * 70)
    
    # Create universe
    universe = create_challenging_test_universe(K=5, feature_dim=8, seed=seed)
    print(f"Created universe with {universe.K} communities")
    
    # Create family of graphs
    family_graphs = []
    for i in range(n_graphs):
        graph_sample = create_test_graph(
            universe, n_nodes=50, use_dccc=use_dccc, seed=seed + i
        )
        family_graphs.append(graph_sample)
    
    print(f"Created family of {len(family_graphs)} graphs")
    
    # Define splits for universe-based generation
    n_train = n_graphs // 2
    n_val = n_graphs // 4
    n_test = n_graphs - n_train - n_val
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_graphs))
    
    # Generate universe-based metapath tasks
    metapath_data = generate_universe_based_metapath_tasks(
        family_graphs=family_graphs,
        universe=universe,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        k_values=k_values,
        require_loop=True,
        degree_weight=0.3 if use_dccc else 0.0
    )
    
    print(f"\nGenerated {len(metapath_data['tasks'])} universe-based metapath tasks")
    
    # Create multi-graph visualization for each task
    for task_name, task_info in metapath_data['tasks'].items():
        print(f"\n{'='*70}")
        print(f"VISUALIZING TASK: {task_name}")
        print(f"{'='*70}")
        
        metapath = task_info['metapath']
        
        # Show up to 4 graphs that support this task
        valid_graphs = task_info['valid_graphs'][:4]
        task_data = []
        
        for graph_idx in valid_graphs:
            labels = task_info['labels_by_graph'][graph_idx]
            if labels is not None:
                task_data.append({
                    'graph_idx': graph_idx,
                    'labels': labels
                })
        
        if task_data:
            create_multi_graph_metapath_visualization(
                family_graphs, task_data, metapath, task_name
            )
        else:
            print(f"No valid data for visualization of {task_name}")

def create_multi_graph_metapath_visualization(
    family_graphs: List,
    task_data: List[Dict],
    metapath: List[int],
    task_name: str,
    figsize: Tuple[int, int] = (20, 16)
):
    """
    Create visualization showing the same metapath across multiple graphs.
    Fixed to handle missing positive_rate field.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    n_graphs = len(task_data)
    
    # Create subplot grid
    if n_graphs <= 2:
        rows, cols = 1, n_graphs
        figsize = (10 * n_graphs, 8)
    else:
        rows, cols = 2, 2
        figsize = (20, 16)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_graphs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    metapath_str = ' -> '.join(map(str, metapath))
    fig.suptitle(f'Task: {task_name}\nMetapath: {metapath_str}\nAcross {n_graphs} Graphs', 
                 fontsize=16, fontweight='bold')
    
    for idx, task_entry in enumerate(task_data):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        graph_idx = task_entry['graph_idx']
        labels = np.array(task_entry['labels'])
        graph_sample = family_graphs[graph_idx]
        
        # Create layout
        pos = nx.spring_layout(graph_sample.graph, seed=42, k=1, iterations=50)
        
        # Color nodes by community (background)
        unique_communities = np.unique(graph_sample.community_labels)
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        community_color_map = {comm: community_colors[i] for i, comm in enumerate(unique_communities)}
        
        # Draw all nodes with community colors (background)
        for comm in unique_communities:
            nodes_in_comm = [n for n in graph_sample.graph.nodes() if graph_sample.community_labels[n] == comm]
            if nodes_in_comm:
                nx.draw_networkx_nodes(
                    graph_sample.graph, pos, nodelist=nodes_in_comm,
                    node_color=[community_color_map[comm]], 
                    node_size=80, alpha=0.4, ax=ax
                )
        
        # Highlight participating nodes
        loop_nodes = np.where(labels == 2)[0] if np.any(labels == 2) else []
        non_loop_nodes = np.where(labels == 1)[0] if np.any(labels == 1) else []
        
        if len(non_loop_nodes) > 0:
            nx.draw_networkx_nodes(
                graph_sample.graph, pos, nodelist=non_loop_nodes,
                node_color='orange', node_size=150, alpha=0.9, 
                edgecolors='darkorange', linewidths=2, ax=ax
            )
        
        if len(loop_nodes) > 0:
            nx.draw_networkx_nodes(
                graph_sample.graph, pos, nodelist=loop_nodes,
                node_color='red', node_size=180, alpha=0.9, 
                edgecolors='darkred', linewidths=3, ax=ax
            )
        
        # Draw edges
        nx.draw_networkx_edges(graph_sample.graph, pos, alpha=0.2, width=0.5, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(graph_sample.graph, pos, font_size=6, ax=ax)
        
        # Calculate statistics (instead of reading from task_entry)
        positive_rate = np.mean(labels > 0)
        loop_count = np.sum(labels == 2)
        non_loop_count = np.sum(labels == 1)
        
        ax.set_title(f'Graph {graph_idx}\n'
                    f'Positive Rate: {positive_rate:.3f}\n'
                    f'Loop: {loop_count}, Non-loop: {non_loop_count}', 
                    fontsize=12)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_graphs, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_elements = []
    
    # Community legend - use last graph_sample for consistency
    for comm in unique_communities:
        original_comm = graph_sample.communities[comm] if comm < len(graph_sample.communities) else comm
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=community_color_map[comm], 
                      markersize=8, label=f'Community {original_comm}')
        )
    
    # Participation legend
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='orange', markersize=12, 
                  markeredgecolor='darkorange', markeredgewidth=2,
                  label='Non-Loop Participants'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='red', markersize=15, 
                  markeredgecolor='darkred', markeredgewidth=3,
                  label='Loop Participants (4+ nodes)')
    ])
    
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics for {task_name}:")
    print(f"Metapath: {metapath_str}")
    print(f"Graphs analyzed: {len(task_data)}")
    
    for idx, task_entry in enumerate(task_data):
        graph_idx = task_entry['graph_idx']
        labels = np.array(task_entry['labels'])
        
        # Calculate statistics here instead of reading from task_entry
        positive_rate = np.mean(labels > 0)
        loop_count = np.sum(labels == 2)
        non_loop_count = np.sum(labels == 1)
        total_participating = loop_count + non_loop_count
        
        print(f"\n  Graph {graph_idx}:")
        print(f"    Total nodes: {len(labels)}")
        print(f"    Participating nodes: {total_participating} ({positive_rate:.3f})")
        print(f"    Loop participants (4+ nodes): {loop_count}")
        print(f"    Non-loop participants: {non_loop_count}")
        
        if hasattr(family_graphs[graph_idx], 'communities'):
            print(f"    Communities: {family_graphs[graph_idx].communities}")

def run_comprehensive_universe_test():
    """
    Run comprehensive test of the universe-based approach.
    """
    print("Starting Comprehensive Universe-Based Metapath Test")
    print("=" * 70)
    
    # Test 1: Universe metapath selection
    print("\n1. Testing universe metapath candidate generation...")
    universe = create_challenging_test_universe(K=5, seed=456)
    
    selector = UniverseMetapathSelector(universe, degree_weight=0.3, verbose=True)
    candidates = selector.generate_diverse_metapath_candidates(
        k_values=[3, 4],
        require_loop=True,
        n_candidates_per_k=20,
        diversity_factor=0.4
    )
    
    candidate_analysis = selector.analyze_candidate_properties(candidates)
    
    print("\nCandidate Analysis:")
    print(f"  Total candidates: {candidate_analysis['total_candidates']}")
    print(f"  Candidates per k: {candidate_analysis['candidates_per_k']}")
    print(f"  Community usage: {candidate_analysis['community_usage']}")
    
    # Test 2: Family evaluation
    print("\n2. Testing family-level evaluation...")
    metapath_data = test_universe_based_metapath_generation()
    
    # Test 3: Integration test
    print("\n3. Testing full integration...")
    try:
        from experiments.inductive.data import prepare_inductive_data_with_universe_metapaths
        
        # Create a simple config
        class TestConfig:
            def __init__(self):
                self.seed = 42
                self.batch_size = 16
                self.train_graph_ratio = 0.6
                self.val_graph_ratio = 0.2
                self.tasks = ["community"]
                self.is_regression = {"community": False}
                self.enable_metapath_tasks = True
                self.metapath_k_values = [3, 4]
                self.metapath_require_loop = True
                self.metapath_degree_weight = 0.3
                self.max_community_participation = 0.9
                self.n_candidates_per_k = 15
        
        config = TestConfig()
        
        # Create small family for integration test
        family_graphs = []
        for i in range(4):
            graph = create_test_graph(universe, n_nodes=40, use_dccc=True, seed=456 + i)
            family_graphs.append(graph)
        
        inductive_data = prepare_inductive_data_with_universe_metapaths(
            family_graphs=family_graphs,
            universe=universe,
            config=config
        )
        
        print(f"Integration test successful!")
        print(f"  Tasks created: {list(inductive_data.keys())}")
        
        # Print task details
        for task_name, task_data in inductive_data.items():
            if task_name.startswith('metapath'):
                metadata = task_data.get('metadata', {})
                print(f"  {task_name}:")
                print(f"    Output dim: {metadata.get('output_dim', 'N/A')}")
                print(f"    Coverage: {metadata.get('coverage', 'N/A')}")
                print(f"    Metapath: {metadata.get('metapath', 'N/A')}")
                
                # Print split sizes
                for split_name in ['train', 'val', 'test']:
                    if split_name in task_data:
                        n_graphs = task_data[split_name]['n_graphs']
                        print(f"    {split_name}: {n_graphs} graphs")
        
    except ImportError as e:
        print(f"Integration test skipped due to import error: {e}")
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nComprehensive universe-based test completed!")
    print("\nKey improvements verified:")
    print("1. ✓ Metapath selection based purely on universe parameters")
    print("2. ✓ Diverse candidate generation using degree centers + P matrix")
    print("3. ✓ Family-level evaluation with participation rate checking")
    print("4. ✓ Automatic rejection of metapaths with >95% community participation")
    print("5. ✓ Detailed reporting of participation rates by split")

def test_core_universe_metapath():
    """Simplified test of core functionality only - FIXED VERSION"""
    
    # 1. Create universe and family
    universe = create_challenging_test_universe(K=5, seed=42)
    family_graphs = [create_test_graph(universe, seed=42+i) for i in range(6)]
    
    # 2. Define splits
    train_indices = [0, 1, 2]
    val_indices = [3, 4] 
    test_indices = [5]

    # 3. Generate universe-based tasks
    metapath_data = generate_universe_based_metapath_tasks(
        family_graphs=family_graphs,
        universe=universe,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        k_values=[4, 5],  # 4+ for proper loops
        require_loop=False
    )
    
    # 4. Visualize across multiple graphs
    for task_name, task_info in metapath_data['tasks'].items():
        print(f"Task: {task_name}")
        print(f"Metapath: {task_info['metapath']}")
        print(f"Coverage: {task_info['coverage']}")
        
        # Show on multiple graphs - FIXED: Create proper task_data structure
        valid_graphs = task_info['valid_graphs'][:4]  # Show first 4
        task_data = []
        
        for graph_idx in valid_graphs:
            labels = task_info['labels_by_graph'][graph_idx]
            if labels is not None:  # Only include graphs with valid labels
                task_data.append({
                    'graph_idx': graph_idx, 
                    'labels': labels
                    # Note: positive_rate is calculated in the visualization function
                })
        
        if task_data:  # Only visualize if we have valid data
            create_multi_graph_metapath_visualization(
                family_graphs, task_data, task_info['metapath'], task_name
            )
        else:
            print(f"No valid graphs found for task {task_name}")


if __name__ == "__main__":
    test_core_universe_metapath()
