import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import torch_geometric.data as pyg
import hashlib
from graph_universe.graph_sample import GraphSample
from graph_universe.graph_universe import GraphUniverse

class GraphFamilyGenerator:
    """
    Generates families of graphs sampled from a graph universe.
    Each graph in the family is a subgraph of the universe with its own community structure.

    Initialize the graph family generator.
        
    Args:
        universe: GraphUniverse to sample from
        n_graphs: Number of graphs to generate in the family
        min_n_nodes: Minimum number of nodes per graph
        max_n_nodes: Maximum number of nodes per graph
        n_graphs: Default number of graphs to generate
        min_communities: Minimum number of communities per graph
        max_communities: Maximum number of communities per graph (defaults to universe.K)
        homophily_range: Tuple of (min_homophily, max_homophily) in graph family
        density_range: Tuple of (min_density, max_density) in graph family
        use_dccc_sbm: Whether to use DCCC-SBM model or standard DC-SBM
        degree_separation_range: Range for degree distribution separation (min, max)
        enable_deviation_limiting: Whether to enable deviation checks
        max_mean_community_deviation: If enabled, maximum allowed mean target scaled edge probability community deviation
        degree_distribution: Degree distribution type ("standard", "power_law", "exponential", "uniform")
        power_law_exponent_range: Range for power law exponent (min, max)
        exponential_rate_range: Range for exponential distribution rate (min, max)
        uniform_min_factor_range: Range for uniform distribution min factor (min, max)
        uniform_max_factor_range: Range for uniform distribution max factor (min, max)
        degree_heterogeneity: Fixed degree heterogeneity for all graphs (if use_dccc_sbm is False, this is used)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        universe: GraphUniverse,
        min_n_nodes: int,
        max_n_nodes: int,
        n_graphs: int = 100,
        min_communities: int = 2,
        max_communities: int = None,
        homophily_range: Tuple[float, float] = (0.0, 0.4),  # Homophily range
        avg_degree_range: Tuple[float, float] = (1.0, 3.0),    # Average degree range
        use_dccc_sbm: bool = True, # Whether to use DCCC-SBM or standard DC-SBM
        enable_deviation_limiting: bool = False, # Deviation limiting
        max_mean_community_deviation: float = 0.10,
        degree_distribution: str = "standard", # DCCC distribution-specific parameter ranges
        power_law_exponent_range: Tuple[float, float] = (2.0, 3.5),
        exponential_rate_range: Tuple[float, float] = (0.3, 1.0),
        uniform_min_factor_range: Tuple[float, float] = (0.3, 0.7),
        uniform_max_factor_range: Tuple[float, float] = (1.3, 2.0),
        degree_separation_range: Tuple[float, float] = (0.5, 0.5),    # Range for degree separation
        degree_heterogeneity: float = 0.5,  # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        seed: Optional[int] = 42
    ):
        self.universe = universe
        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes
        self.n_graphs = n_graphs
        self.min_communities = min_communities
        self.max_communities = max_communities if max_communities is not None else universe.K
        self.homophily_range = homophily_range
        self.avg_degree_range = avg_degree_range
        # self.density_range = density_range

        # Whether to use DCCC-SBM or standard DC-SBM
        self.use_dccc_sbm = use_dccc_sbm

        # DCCC distribution-specific parameters
        self.degree_separation_range = degree_separation_range # Range for degree separation
        
        # Community co-occurrence homogeneity
        # self.community_cooccurrence_homogeneity = community_cooccurrence_homogeneity

        # Deviation limiting
        self.enable_deviation_limiting = enable_deviation_limiting
        self.max_mean_community_deviation = max_mean_community_deviation
        
        # DCCC distribution parameters
        self.degree_distribution = degree_distribution
        self.power_law_exponent_range = power_law_exponent_range
        self.exponential_rate_range = exponential_rate_range
        self.uniform_min_factor_range = uniform_min_factor_range
        self.uniform_max_factor_range = uniform_max_factor_range

        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
        self.degree_heterogeneity = degree_heterogeneity
        
        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        # Validate parameters
        self._validate_parameters()
        
        # Storage for generated graphs and metadata
        self.graphs: List[GraphSample] = []
        self.generation_metadata: List[Dict[str, Any]] = []
        self.generation_stats: Dict[str, Any] = {}

        self.family_generated = False

    def generate_family(
        self,
        n_graphs: int | None = None,
        show_progress: bool = True,
        collect_stats: bool = True,
        timeout_minutes: float = 5.0,
        allowed_community_combinations: Optional[List[List[int]]] = None
    ) -> List[GraphSample]:
        """
        Generate a family of graphs from the universe.
        
        Args:
            n_graphs: Number of graphs to generate (overrides initial configuration)
            show_progress: Whether to show progress bar
            collect_stats: Whether to collect generation statistics
            timeout_minutes: Maximum time in minutes to spend generating graphs
            allowed_community_combinations: Optional list of lists of community indices to be sampled from the universe
        Returns:
            List of generated GraphSample objects
        """
        self.n_graphs = n_graphs if n_graphs is not None else self.n_graphs
        start_time = time.time()
        starting_time_new_graph = start_time
        timeout_seconds = timeout_minutes * 60
        self.graphs = []
        self.community_labels_per_graph = []
        self.generation_metadata = []
        self.graph_generation_times = []
        failed_graphs = 0

        # Every time this function is called we need to reset Seed
        # self.seed = np.random.randint(0, 1000000)
        # np.random.seed(self.seed)
        
        # Progress bar setup
        if show_progress:
            pbar = tqdm(total=self.n_graphs, desc="Generating graph family")
        
        while len(self.graphs) < self.n_graphs:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                warnings.warn(f"Timeout reached after {timeout_minutes} minutes. Generated {len(self.graphs)} graphs instead of {self.n_graphs}")
                break
                
            # graph_generated = False
            # attempts = 0
            
            # while not graph_generated and attempts < max_attempts_per_graph:
            #     attempts += 1
                
            #     try:
            # Get a random seed for this graph
            graph_seed = np.random.randint(0, 1000000)

            # Sample parameters for this graph
            params = self._sample_graph_parameters()

            if allowed_community_combinations is not None:
                sampled_community_combination_index = np.random.randint(0, len(allowed_community_combinations))
                sampled_community_combination = allowed_community_combinations[sampled_community_combination_index]
            
            # Create graph sample
            graph_sample = GraphSample(
                # Give GraphUniverse object to sample from
                universe=self.universe,

                # Graph Sample specific parameters
                num_communities=params['n_communities'],
                n_nodes=params['n_nodes'],
                target_homophily=params['target_homophily'],
                target_average_degree=params['target_average_degree'],
                degree_distribution=self.degree_distribution,
                power_law_exponent=params.get('power_law_exponent', None),
                max_mean_community_deviation=self.max_mean_community_deviation,

                # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
                degree_heterogeneity=self.degree_heterogeneity,

                # DCCC-SBM parameters
                use_dccc_sbm=self.use_dccc_sbm,
                degree_separation=params.get('degree_separation', 0.5),
                dccc_global_degree_params=params.get('dccc_global_degree_params', {}),
                enable_deviation_limiting=self.enable_deviation_limiting,

                # Random seed
                seed=graph_seed,

                # Optional Parameter for user-defined communites to be sampled (Such that NO unseen communities are sampled for val or test)
                user_defined_communities=sampled_community_combination if allowed_community_combinations is not None else None
            )
            
            # Store graph and metadata
            self.graphs.append(graph_sample)
            self.community_labels_per_graph.append(np.unique(graph_sample.community_labels_universe_level, axis=0))
            metadata = {
                'graph_id': len(self.graphs) - 1,
                'final_n_nodes': graph_sample.n_nodes,
                'final_n_edges': graph_sample.graph.number_of_edges(),
                'actual_density': graph_sample.graph.number_of_edges() / (graph_sample.n_nodes * (graph_sample.n_nodes - 1) / 2) if graph_sample.n_nodes > 1 else 0,
                'generation_method': graph_sample.generation_method,
                'timing_info': graph_sample.timing_info.copy() if hasattr(graph_sample, 'timing_info') else {},
                **params  # Include sampled parameters
            }
            
            self.generation_metadata.append(metadata)
            # graph_generated = True

            graph_generation_time = time.time() - starting_time_new_graph
            self.graph_generation_times.append(graph_generation_time)
            starting_time_new_graph = time.time()
                    
            if show_progress:
                pbar.update(1)
                    
                # except Exception as e:
                #     tb_str = traceback.format_exc()
                #     # Short error version
                #     # print(f"Failed to generate graph after {attempts} attempts: {e}")

                #     # Long error version
                #     # print(f"Failed to generate graph after {attempts} attempts: {e}\n{tb_str}")

                #     if attempts == max_attempts_per_graph:
                #         warnings.warn(f"Failed to generate graph after {attempts} attempts: {e}")
                #         failed_graphs += 1
                #         # Add empty metadata for failed graph
                #         self.generation_metadata.append({
                #             'graph_id': len(self.graphs),
                #             'attempts': attempts,
                #             'failed': True,
                #             'error': str(e),
                #             'traceback': tb_str
                #         })
                #     # Continue to next attempt

        # Use set of sorted tuples for uniqueness
        unique_set_of_community_combinations = {tuple(sorted(arr)) for arr in self.community_labels_per_graph}

        # Convert back to list of lists
        unique_list_of_community_combinations = [list(tup) for tup in unique_set_of_community_combinations]
        self.seen_community_combinations = unique_list_of_community_combinations

        if show_progress:
            pbar.close()
        
        # Collect generation statistics
        if collect_stats:
            self._collect_generation_stats(start_time, failed_graphs, self.n_graphs)

        # Mark family as generated
        self.family_generated = True



    def to_pyg_graphs(self, tasks: List[str] | None = None) -> List[pyg.data.Data]:
        """
        Convert the graphs to PyG graphs with all specified tasks as properties.
        
        Args:
            tasks: List of task strings to include as properties on each graph.
                If None, include all tasks.

        Returns:
            List of PyG Data objects, each containing all tasks as properties
        """
        # Check if graphs are created
        assert self.family_generated, "Graph family has not been generated yet"
        
        # Check if tasks are valid - updated to handle k-hop tasks
        if tasks is not None:
            valid_task_prefixes = ["community_detection", "triangle_counting", "k_hop_community_counts_k"]
            for task in tasks:
                if not any(task.startswith(prefix) for prefix in valid_task_prefixes):
                    raise ValueError(f"Invalid task specified: {task}")
            self.tasks = tasks
        else:
            # Include all tasks if None
            self.tasks = [
                "community_detection",
                "triangle_counting",
                "k_hop_community_counts_k1",
                "k_hop_community_counts_k2",
                "k_hop_community_counts_k3",
                "k_hop_community_counts_k4",
                "k_hop_community_counts_k5",
            ]

        pyg_graphs = []
        for graph in tqdm(self.graphs, desc="Converting graphs to PyG graphs"):
            pyg_graphs.append(graph.to_pyg_graph(self.tasks))
        
        return pyg_graphs
    
    def get_uniquely_identifying_metadata(self) -> Dict[str, Any]:
        """
        Get uniquely identifying metadata for the graph family.
        """
        uniquely_identifying_metadata = {
            'universe_parameters': {
                'K': self.universe.K,
                'feature_dim': self.universe.feature_dim,
                'center_variance': self.universe.center_variance,
                'cluster_variance': self.universe.cluster_variance,
                'edge_propensity_variance': self.universe.edge_propensity_variance,
                'seed': self.universe.seed,
                # 'community_degree_propensity_vector': self.universe.community_degree_propensity_vector.tolist(),
                # 'propensity_matrix_hash': hash(self.universe.P.tobytes()),
                # 'community_degree_propensity_vector_hash': hash(self.universe.community_degree_propensity_vector.tobytes()),
            },
            'family_parameters': {
                'n_graphs': self.n_graphs,
                'min_n_nodes': self.min_n_nodes,
                'max_n_nodes': self.max_n_nodes,
                'min_communities': self.min_communities,
                'max_communities': self.max_communities,
                'homophily_range': self.homophily_range,
                'avg_degree_range': self.avg_degree_range,
                'degree_heterogeneity': self.degree_heterogeneity,
                'use_dccc_sbm': self.use_dccc_sbm,
                'degree_separation_range': self.degree_separation_range,
                'degree_distribution': self.degree_distribution,
                'seed': self.seed,
            },
        }
        # # Create a hash of all UNIVERSE generation parameters to be able to indentify common universes between different graph families
        # universe_hash_parts = []
        # for key, value in uniquely_identifying_metadata['universe_parameters'].items():
        #     universe_hash_parts.append(f"{key}:{value}")
        
        # universe_hash = hashlib.sha256(str(universe_hash_parts).encode()).hexdigest()
        # uniquely_identifying_metadata['universe_hash'] = universe_hash
        return uniquely_identifying_metadata

    def save_pyg_graphs_and_universe(self, tasks: List[str] | None = None, root_dir: str = "datasets"):
        """
        Save the PyG graphs and universe to a file.
        """
        import os
        from .dataset import GraphUniverseDataset

        uniquely_identifying_metadata = self.get_uniquely_identifying_metadata()

        # Convert the graphs to PyG graphs including tasks
        pyg_graphs = self.to_pyg_graphs(tasks)

        GraphUniverseDataset(
            graph_list=pyg_graphs,
            root=root_dir,
            parameters=uniquely_identifying_metadata
        )
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.min_n_nodes <= 0:
            raise ValueError("min_n_nodes must be positive")
        if self.max_n_nodes < self.min_n_nodes:
            raise ValueError("max_n_nodes must be >= min_n_nodes")
        if self.min_communities < 1:
            raise ValueError("min_communities must be >= 1")
        if self.max_communities > self.universe.K:
            raise ValueError(f"max_communities cannot exceed universe size ({self.universe.K})")
        if self.max_communities < self.min_communities:
            raise ValueError("max_communities must be >= min_communities")
        
        # Validate ranges
        if len(self.homophily_range) != 2 or self.homophily_range[0] > self.homophily_range[1]:
            raise ValueError("homophily_range must be a tuple (min, max) with min <= max")
        if self.homophily_range[0] < 0.0 or self.homophily_range[1] > 1.0:
            raise ValueError("homophily_range values must be between 0.0 and 1.0")
        if len(self.avg_degree_range) != 2 or self.avg_degree_range[0] > self.avg_degree_range[1]:
            raise ValueError("avg_degree_range must be a tuple (min, max) with min <= max")
        if self.avg_degree_range[0] < 0.0 or self.avg_degree_range[1] > 100.0:
            raise ValueError("avg_degree_range values must be between 0.0 and 100.0")
        if len(self.degree_separation_range) != 2 or self.degree_separation_range[0] > self.degree_separation_range[1]:
            raise ValueError("degree_separation_range must be a tuple (min, max) with min <= max")
        
        # Validate DCCC distribution parameter ranges
        if len(self.power_law_exponent_range) != 2 or self.power_law_exponent_range[0] > self.power_law_exponent_range[1]:
            raise ValueError("power_law_exponent_range must be a tuple (min, max) with min <= max")
        if len(self.exponential_rate_range) != 2 or self.exponential_rate_range[0] > self.exponential_rate_range[1]:
            raise ValueError("exponential_rate_range must be a tuple (min, max) with min <= max")
        if len(self.uniform_min_factor_range) != 2 or self.uniform_min_factor_range[0] > self.uniform_min_factor_range[1]:
            raise ValueError("uniform_min_factor_range must be a tuple (min, max) with min <= max")
        if len(self.uniform_max_factor_range) != 2 or self.uniform_max_factor_range[0] > self.uniform_max_factor_range[1]:
            raise ValueError("uniform_max_factor_range must be a tuple (min, max) with min <= max")
        
        # Validate that distribution ranges make sense
        if self.power_law_exponent_range[0] <= 1.0:
            raise ValueError("power_law_exponent_range values must be > 1.0")
        if self.exponential_rate_range[0] <= 0.0:
            raise ValueError("exponential_rate_range values must be > 0.0")
        if self.uniform_min_factor_range[0] <= 0.0:
            raise ValueError("uniform_min_factor_range values must be > 0.0")
        if self.uniform_max_factor_range[0] <= 0.0:
            raise ValueError("uniform_max_factor_range values must be > 0.0")
   
    def _sample_graph_parameters(self) -> Dict[str, Any]:
        """
        Sample parameters for a single graph within the specified ranges.
        
        Returns:
            Dictionary of sampled parameters
        """
        # Sample number of nodes
        n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        
        # Sample number of communities
        n_communities = np.random.randint(self.min_communities, self.max_communities + 1)
        
        # Sample target homophily
        target_homophily = np.random.uniform(self.homophily_range[0], self.homophily_range[1])

        # Sample target average degree
        target_average_degree = np.random.uniform(self.avg_degree_range[0], self.avg_degree_range[1])
        
        # # Sample target density
        # target_density = np.random.uniform(self.density_range[0], self.density_range[1])
        
        # Sample DCCC parameters if using DCCC-SBM
        dccc_params = {}
        if self.use_dccc_sbm:
            # Sample degree separation
            degree_separation = np.random.uniform(
                self.degree_separation_range[0], 
                self.degree_separation_range[1]
            )
            
            # Sample distribution-specific parameters
            distribution_params = {}
            if self.degree_distribution == "power_law":
                power_law_exponent = np.random.uniform(
                    self.power_law_exponent_range[0],
                    self.power_law_exponent_range[1]
                )
                distribution_params = {
                    "exponent": power_law_exponent,
                    "x_min": 1.0
                }
            elif self.degree_distribution == "exponential":
                rate = np.random.uniform(
                    self.exponential_rate_range[0],
                    self.exponential_rate_range[1]
                )
                distribution_params = {"rate": rate}
            elif self.degree_distribution == "uniform":
                min_factor = np.random.uniform(
                    self.uniform_min_factor_range[0],
                    self.uniform_min_factor_range[1]
                )
                max_factor = np.random.uniform(
                    self.uniform_max_factor_range[0],
                    self.uniform_max_factor_range[1]
                )
                # Ensure max_factor > min_factor
                if max_factor <= min_factor:
                    max_factor = min_factor + 0.1
                distribution_params = {
                    "min_degree": min_factor,
                    "max_degree": max_factor
                }
            
            dccc_params = {
                "degree_separation": degree_separation,
                "dccc_global_degree_params": distribution_params,
                "power_law_exponent": distribution_params.get("exponent", None)
            }
        
        # Combine all parameters
        params = {
            'n_nodes': n_nodes,
            'target_homophily': target_homophily,
            'target_average_degree': target_average_degree,
            'n_communities': n_communities,
        }
        
        # Add DCCC parameters
        params.update(dccc_params)
        
        return params
    
    def _generate_single_graph(
        self,
        max_attempts: int = 10,
        timeout_minutes: float = 5.0
    ) -> GraphSample:
        """
        Generate a single graph with progress tracking.
        
        Args:
            max_attempts: Maximum attempts per graph before giving up
            timeout_minutes: Maximum time in minutes to spend generating this graph
            
        Returns:
            Generated GraphSample object
            
        Raises:
            Exception: If graph generation fails after max_attempts
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                raise Exception(f"Timeout reached after {timeout_minutes} minutes")
            
            try:
                # Get a random seed for this graph
                graph_seed = np.random.randint(0, 1000000)
                
                # Sample parameters for this graph
                params = self._sample_graph_parameters()
                
                # Debug: Print parameters being used
                print(f"Attempt {attempts}: Generating graph with params: {params}")
                
                # Create graph sample
                print(f"Attempt {attempts}: Starting GraphSample initialization...")
                
                try:
                    graph_sample = GraphSample(
                        # Give GraphUniverse object to sample from
                        universe=self.universe,

                        # Graph Sample specific parameters
                        num_communities=params['n_communities'],
                        n_nodes=params['n_nodes'],
                        target_homophily=params['target_homophily'],
                        target_average_degree=params['target_average_degree'],
                        degree_distribution=self.degree_distribution,
                        power_law_exponent=params.get('power_law_exponent', None),
                        max_mean_community_deviation=self.max_mean_community_deviation,

                        # Standard DC-SBM parameters (so if use_dccc_sbm is False, this is used)
                        degree_heterogeneity=self.degree_heterogeneity,

                        # DCCC-SBM parameters
                        use_dccc_sbm=self.use_dccc_sbm,
                        degree_separation=params.get('degree_separation', 0.5),
                        dccc_global_degree_params=params.get('dccc_global_degree_params', {}),
                        enable_deviation_limiting=self.enable_deviation_limiting,

                        # Random seed
                        seed=graph_seed,

                        # Optional Parameter for user-defined communites to be sampled
                        user_defined_communities=None
                    )
                    print(f"Attempt {attempts}: GraphSample initialization completed successfully")
                except Exception as e:
                    print(f"Attempt {attempts}: GraphSample initialization failed: {str(e)}")
                    raise e
                
                print(f"Successfully generated graph with {graph_sample.n_nodes} nodes")
                return graph_sample
                
            except Exception as e:
                # print(f"Attempt {attempts} failed: {str(e)}")
                if attempts == max_attempts:
                    raise Exception(f"Failed to generate graph after {attempts} attempts: {e}") from e
                # Continue to next attempt
 
    def _collect_generation_stats(self, start_time: float, failed_graphs: int, n_graphs: int) -> None:
        """Collect statistics about the generation process."""
        total_time = time.time() - start_time
        successful_graphs = len(self.graphs)
        
        # Basic stats
        self.generation_stats = {
            'total_time': total_time,
            'successful_graphs': successful_graphs,
            'failed_graphs': failed_graphs,
            'success_rate': successful_graphs / n_graphs if n_graphs > 0 else 0,
            'avg_time_per_graph': total_time / n_graphs if n_graphs > 0 else 0
        }
        
        if successful_graphs > 0:
            # Get statistics from successful graphs
            successful_metadata = [m for m in self.generation_metadata if not m.get('failed', False)]
            
            node_counts = [m['final_n_nodes'] for m in successful_metadata]
            edge_counts = [m['final_n_edges'] for m in successful_metadata]
            densities = [m['actual_density'] for m in successful_metadata]
            # attempts = [m['attempts'] for m in successful_metadata]
            
            self.generation_stats.update({
                'node_count_stats': {
                    'mean': np.mean(node_counts),
                    'std': np.std(node_counts),
                    'min': np.min(node_counts),
                    'max': np.max(node_counts)
                },
                'edge_count_stats': {
                    'mean': np.mean(edge_counts),
                    'std': np.std(edge_counts),
                    'min': np.min(edge_counts),
                    'max': np.max(edge_counts)
                },
                'density_stats': {
                    'mean': np.mean(densities),
                    'std': np.std(densities),
                    'min': np.min(densities),
                    'max': np.max(densities)
                },
        })
    
    def save_family(self, filepath: str, n_graphs: int = 0) -> None:
        """
        Save the graph family to file.
        
        Args:
            filepath: Path to save file
            include_graphs: Whether to include graph objects (large files)
        """
        import pickle
        
        save_data = {
            'n_graphs': n_graphs,
            'min_n_nodes': self.min_n_nodes,
            'max_n_nodes': self.max_n_nodes,
            'min_communities': self.min_communities,
            'max_communities': self.max_communities,
            'homophily_range': self.homophily_range,
            'density_range': self.density_range,
            'use_dccc_sbm': self.use_dccc_sbm,
            'degree_separation_range': self.degree_separation_range,
            'degree_distribution': self.degree_distribution,
            'power_law_exponent_range': self.power_law_exponent_range,
            }
            
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def __len__(self) -> int:
        """Return number of successfully generated graphs."""
        return len(self.graphs)
    
    def __getitem__(self, index: int) -> GraphSample:
        """Get a specific graph from the family."""
        return self.graphs[index]
    
    def __iter__(self):
        """Iterate over graphs in the family."""
        return iter(self.graphs)
    
    def analyze_graph_family_properties(self) -> Dict[str, Any]:
        """Analyze the properties of the graph family once generated."""
        if self.graphs is None:
            raise ValueError("No graphs in family. Please generate family first before analyzing properties.")
        
        properties = {
            'n_graphs': len(self.graphs),
            'node_counts': [],
            'edge_counts': [],
            'densities': [],
            'avg_degrees': [],
            'clustering_coefficients': [],
            'community_counts': [],
            'homophily_levels': [],
            'nr_of_triangles': [],
            'generation_methods': [],
            'degree_distributions': [],
            'degree_distribution_power_law_exponents': [],
            'tail_ratio_95': [],
            'tail_ratio_99': [],
            'mean_edge_probability_deviation': [],
            'graph_generation_times': self.graph_generation_times
        }
        
        for graph in self.graphs:
            properties['node_counts'].append(graph.n_nodes)
            properties['edge_counts'].append(graph.graph.number_of_edges())
            
            if graph.n_nodes > 1:
                density = graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2)
                properties['densities'].append(density)
            else:
                properties['densities'].append(0.0)
            
            if graph.n_nodes > 0:
                avg_degree = sum(dict(graph.graph.degree()).values()) / (graph.n_nodes)
                properties['avg_degrees'].append(avg_degree)
            else:
                properties['avg_degrees'].append(0.0)
            
            try:
                clustering = nx.average_clustering(graph.graph)
                properties['clustering_coefficients'].append(clustering)
            except:
                properties['clustering_coefficients'].append(0.0)
            
            properties['community_counts'].append(len(np.unique(graph.community_labels)))
            
            # Calculate homophily level
            if graph.n_nodes > 0 and graph.graph.number_of_edges() > 0:
                # Count edges between nodes of same community
                same_community_edges = 0
                for u, v in graph.graph.edges():
                    if graph.community_labels[u] == graph.community_labels[v]:
                        same_community_edges += 1
                homophily = same_community_edges / graph.graph.number_of_edges()
                properties['homophily_levels'].append(homophily)
            else:
                properties['homophily_levels'].append(0.0)
            
            # Calculate number of triangles
            if graph.n_nodes > 0:
                triangle_values = list(nx.triangles(graph.graph).values())
                properties['nr_of_triangles'].append(np.sum(triangle_values)/3)
            else:
                properties['nr_of_triangles'].append(0.0)
            
            # Track generation method
            if hasattr(graph, 'generation_method'):
                properties['generation_methods'].append(graph.generation_method)
            
            # Extract degree distribution and fit power law exponent
            if graph.n_nodes > 0:
                degrees = list(dict(graph.graph.degree()).values())
                # Fit power law exponent
                power_law_exponent = self._fit_power_law_exponent(degrees)
                tail_metrics = self._calculate_degree_tail_metrics(degrees)
                properties['tail_ratio_95'].append(tail_metrics['tail_ratio_95'])
                properties['tail_ratio_99'].append(tail_metrics['tail_ratio_99'])
                properties['degree_distributions'].append(degrees)
                properties['degree_distribution_power_law_exponents'].append(power_law_exponent)
            else:
                properties['degree_distributions'].append([])
                properties['degree_distribution_power_law_exponents'].append(0.0)

            # Calculate mean edge probability deviation
            if graph.n_nodes > 0:
                deviations = graph._calculate_community_deviations(graph.graph, graph.community_labels, graph.P_sub)
                properties['mean_edge_probability_deviation'].append(deviations['mean_deviation'])
            else:
                properties['mean_edge_probability_deviation'].append(0.0)

        # Calculate statistics and convert to native Python types
        for key in ['node_counts', 'edge_counts', 'densities', 'avg_degrees', 'clustering_coefficients', 'community_counts', 'homophily_levels', 'nr_of_triangles', 'graph_generation_times', 'mean_edge_probability_deviation']:
            values = properties[key]
            if values:
                properties[f'{key}_mean'] = float(np.mean(values))
                properties[f'{key}_std'] = float(np.std(values))
                properties[f'{key}_min'] = float(np.min(values))
                properties[f'{key}_max'] = float(np.max(values))
            else:
                properties[f'{key}_mean'] = 0.0
                properties[f'{key}_std'] = 0.0
                properties[f'{key}_min'] = 0.0
                properties[f'{key}_max'] = 0.0
        
        # Add generation method summary
        if properties['generation_methods']:
            from collections import Counter
            method_counts = Counter(properties['generation_methods'])
            properties['generation_method_distribution'] = dict(method_counts)
        
        return properties
   
    def analyze_graph_family_signals(self) -> Dict[str, Any]:
        """Analyze the signals of the graph family."""
        if not self.graphs:
            raise ValueError("No graphs in family. Please generate family first before analyzing signals.")
        
        # Calculate the signals
        signals = {
            'feature_signal': [],
            'degree_signal': [],
            'triangle_signal': [],
            'structure_signal': []
        }
        
        for graph in self.graphs:
            signals['feature_signal'].append(graph.calculate_feature_signal())
            signals['degree_signal'].append(graph.calculate_degree_signal())
            signals['triangle_signal'].append(graph.calculate_triangle_community_signal())
            signals['structure_signal'].append(graph.calculate_structure_signal())
        
        return signals

    def analyze_graph_family_consistency(self) -> Dict[str, Any]:
        """Analyze the consistency of the graph family."""
        if not self.graphs:
            raise ValueError("No graphs in family. Please generate family first before analyzing consistency.")
        
        # Calculate the consistency
        results = {}
        
        # 1. Pattern preservation (do communities RELATIVELY connect more to the communities they are supposed to connect to?)
        try:
            pattern_corrs = self._calculate_pattern_consistency()
            results['structure_consistency'] = pattern_corrs

        except Exception as e:
            results['structure_consistency'] = []

        # # 2. Generation fidelity (do graphs match their scaled probability targets (P_sub)?)
        # try:
        #     generation_fidelity = self._calculate_generation_fidelity()
        #     results['generation_fidelity'] = generation_fidelity
        # except Exception as e:
        #     results['generation_fidelity'] = []

        # 3. Degree consistency (do actual node degrees correlate with universe degree centers?)
        try:
            degree_consistency = self._calculate_degree_consistency()
            results['degree_consistency'] = degree_consistency
        except Exception as e:
            results['degree_consistency'] = []

        return results

    def _calculate_pattern_consistency(self) -> List[float]:
        """
        Measure whether relative community connection patterns are preserved.
        Uses rank correlation to focus on structural patterns rather than absolute values.
        """
        pattern_correlations = []
        universe_P = self.universe.P

        for graph in self.graphs:
            try:
                # Extract relevant submatrix from universe
                P_sub = graph.P_sub
                
                # Get actual connections
                actual_edge_probabilities, community_sizes, connection_counts = graph.calculate_actual_probability_matrix()

                # Calculate rank correlation per row comparing the expected and actual edge probabilities and average them
                if len(P_sub) > 1 and np.std(P_sub) > 0 and np.std(actual_edge_probabilities) > 0:
                    correlations = []
                    for i in range(len(P_sub)):
                        correlation, _ = spearmanr(P_sub[i], actual_edge_probabilities[i])
                        if not np.isnan(correlation):
                            correlations.append(correlation)
                    pattern_correlations.append(np.mean(correlations))
            except Exception as e:
                warnings.warn(f"Error in pattern consistency calculation for graph: {e}")
                continue
        
        return pattern_correlations

    def _calculate_generation_fidelity(self) -> List[float]:
            """
            Measure how well graphs match their scaled probability targets (P_sub).
            """
            fidelity_scores = []
            
            for graph in self.graphs:
                try:
                    # Use the graph's own scaled P_sub as reference
                    expected_matrix = graph.P_sub
                    
                    # Get actual connections
                    actual_analysis = graph.analyze_community_connections()
                    actual_matrix = actual_analysis['actual_matrix']
                    
                    # Calculate correlation between expected and actual
                    expected_flat = expected_matrix.flatten()
                    actual_flat = actual_matrix.flatten()
                    
                    if len(expected_flat) > 1:
                        if np.std(expected_flat) > 0 and np.std(actual_flat) > 0:
                            correlation, _ = pearsonr(expected_flat, actual_flat)
                            if not np.isnan(correlation):
                                fidelity_scores.append(correlation)
                except Exception as e:
                    warnings.warn(f"Error in generation fidelity calculation for graph: {e}")
                    continue
            
            if not fidelity_scores:
                return []
            
            return fidelity_scores
    
    def _measure_ordering_consistency(self, values_a: np.ndarray, values_b: np.ndarray) -> float:
        """
        Fallback ordering consistency: fraction of correctly ordered pairs between two vectors.
        Values that preserve pairwise ordering contribute positively.
        """
        num_items = len(values_a)
        if num_items <= 1:
            return 1.0
        correct_pairs = 0
        total_pairs = 0
        for i in range(num_items):
            for j in range(i + 1, num_items):
                total_pairs += 1
                diff_a = values_a[i] - values_a[j]
                diff_b = values_b[i] - values_b[j]
                if diff_a == 0 and diff_b == 0:
                    # Ties in both are considered consistent
                    correct_pairs += 1
                elif diff_a * diff_b >= 0:
                    correct_pairs += 1
        return (correct_pairs / total_pairs) if total_pairs > 0 else 1.0

    def _calculate_degree_consistency(self) -> List[float]:
        """
        Compare actual node degrees to expected degrees based on universe degree centers
        (within-graph ranking consistency) and also measure cross-graph ranking consistency.
        Final per-graph score is the average of the within-graph score and the cross-graph score.
        """
        degree_centers = self.universe.community_degree_propensity_vector
        universe_num_communities = len(degree_centers)

        # First pass: compute within-graph scores and percentile rank signatures per graph
        within_scores: List[float] = []
        percentile_signatures: List[np.ndarray] = []  # Each is length universe_num_communities with NaNs for absent communities

        for graph in self.graphs:
            num_local_communities = len(graph.communities)

            # Average degree per community (normalized by community size)
            avg_degrees_per_community = np.zeros(num_local_communities)
            community_sizes = np.zeros(num_local_communities)

            for node_idx in range(graph.n_nodes):
                local_community_id = graph.community_labels[node_idx]
                degree = graph.graph.degree[node_idx]
                avg_degrees_per_community[local_community_id] += degree
                community_sizes[local_community_id] += 1

            # Normalize by community size
            for i in range(num_local_communities):
                if community_sizes[i] > 0:
                    avg_degrees_per_community[i] /= community_sizes[i]

            # Degree centers for the communities present in this graph (same local order)
            community_degree_centers = degree_centers[graph.communities]

            # Within-graph score: how well avg degrees respect universe degree centers
            if np.std(community_degree_centers) == 0:
                # For constant degree centers, measure degree homogeneity instead
                if np.mean(avg_degrees_per_community) > 0:
                    cv = np.std(avg_degrees_per_community) / np.mean(avg_degrees_per_community)
                    within_score = 1.0 / (1.0 + cv)
                else:
                    within_score = 1.0  # All degrees are 0, perfectly consistent
            else:
                if num_local_communities > 2:
                    from scipy.stats import spearmanr
                    rank_correlation, _ = spearmanr(avg_degrees_per_community, community_degree_centers)
                    if not np.isnan(rank_correlation):
                        within_score = rank_correlation
                    else:
                        within_score = self._measure_ordering_consistency(
                            avg_degrees_per_community, community_degree_centers
                        )
                elif num_local_communities == 2:
                    degree_order_correct = (
                        (avg_degrees_per_community[0] >= avg_degrees_per_community[1]) ==
                        (community_degree_centers[0] >= community_degree_centers[1])
                    )
                    within_score = 1.0 if degree_order_correct else 0.0
                else:
                    within_score = 1.0

            within_scores.append(within_score)

            # Percentile rank signature anchored to universe communities
            signature = np.full(universe_num_communities, np.nan, dtype=float)
            if num_local_communities == 1:
                # Single community: set neutral percentile
                universe_id = graph.community_id_mapping[0]
                signature[universe_id] = 0.5
            else:
                from scipy.stats import rankdata
                ranks = rankdata(avg_degrees_per_community, method='average')  # 1..K
                percentiles = (ranks - 1.0) / (num_local_communities - 1.0)
                for local_idx, universe_id in graph.community_id_mapping.items():
                    signature[universe_id] = percentiles[local_idx]
            percentile_signatures.append(signature)

        # Second pass: cross-graph ranking consistency using Spearman between percentile signatures
        num_graphs = len(self.graphs)
        cross_scores: List[float] = []
        if num_graphs == 0:
            return []

        from scipy.stats import spearmanr
        for i in range(num_graphs):
            sig_i = percentile_signatures[i]
            pairwise_scores: List[float] = []
            pairwise_weights: List[int] = []
            for j in range(num_graphs):
                if j == i:
                    continue
                sig_j = percentile_signatures[j]
                common_mask = ~np.isnan(sig_i) & ~np.isnan(sig_j)
                overlap = int(np.sum(common_mask))
                # Require sufficient overlap for a meaningful correlation
                if overlap >= 3:
                    corr, _ = spearmanr(sig_i[common_mask], sig_j[common_mask])
                    if not np.isnan(corr):
                        pairwise_scores.append(float(corr))
                        pairwise_weights.append(overlap)
            if pairwise_scores:
                cross_scores.append(float(np.average(pairwise_scores, weights=pairwise_weights)))
            else:
                cross_scores.append(float('nan'))

        # Final score: average within-graph (vs centers) and cross-graph ranking consistency
        final_scores: List[float] = []
        for within_score, cross_score in zip(within_scores, cross_scores):
            if np.isnan(cross_score):
                final_scores.append(float(within_score))
            else:
                final_scores.append(float(0.5 * (within_score + cross_score)))

        return final_scores

    def _calculate_cooccurrence_consistency(self) -> List[float]:
        """
        Measure how well community co-occurrence patterns are preserved.
        """        
        # Calculate how often communities co-occur in the family
        cooccurrence_counts_matrix = np.zeros((self.universe.K, self.universe.K))
        for graph in self.graphs:
            for i in range(len(graph.communities)):
                for j in range(i+1, len(graph.communities)):
                    cooccurrence_counts_matrix[graph.community_id_mapping[i], graph.community_id_mapping[j]] += 1
                    cooccurrence_counts_matrix[graph.community_id_mapping[j], graph.community_id_mapping[i]] += 1

        # Calculate how correlated the cooccurrence counts are with the universe cooccurrence matrix
        correlation, _ = pearsonr(cooccurrence_counts_matrix.flatten(), self.universe.community_cooccurrence_matrix.flatten())
        
        return correlation
        
    def _fit_power_law_exponent(self, degrees: List[int]) -> float:
        """
        Fit a power law exponent to a degree distribution using discrete MLE.
        This method is specifically designed for discrete network degrees.
        
        Args:
            degrees: List of node degrees
            
        Returns:
            float: Power law exponent alpha, or 0 if fitting fails or alpha not in [1, 20]
        """
        if not degrees or len(degrees) < 2:
            return 0.0
            
        # Convert to numpy array and filter out zeros
        degrees_array = np.array(degrees, dtype=int)
        degrees_array = degrees_array[degrees_array > 0]
        
        if len(degrees_array) < 2:
            return 0.0
        
        k_min = np.min(degrees_array)
        n = len(degrees_array)
        
        def negative_log_likelihood(alpha):
            if alpha <= 1.0:
                return np.inf
            try:
                # For discrete power law, the log-likelihood is:
                # L = -alpha * sum(log(k_i)) - n * log(zeta(alpha, k_min))
                # We approximate zeta(alpha, k_min) for computational efficiency
                
                # Approximation: zeta(alpha, k_min) ≈ sum_{k=k_min}^{k_max} k^(-alpha)
                k_max = max(100, np.max(degrees_array) * 2)  # Reasonable upper bound
                k_range = np.arange(k_min, k_max + 1)
                zeta_approx = np.sum(k_range**(-alpha))
                
                if zeta_approx <= 0:
                    return np.inf
                    
                log_likelihood = -alpha * np.sum(np.log(degrees_array)) - n * np.log(zeta_approx)
                return -log_likelihood  # Return negative for minimization
                
            except (OverflowError, ZeroDivisionError, ValueError):
                return np.inf
        
        try:
            # Find MLE estimate using bounded optimization
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(negative_log_likelihood, bounds=(1.01, 10.0), method='bounded')
            if result.success and 1.0 < result.x < 20.0:
                return result.x
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_degree_tail_metrics(self, degrees: List[int]) -> Dict[str, float]:
        """Calculate tail-based degree metrics instead of power law fitting"""        
        if len(degrees) == 0:
            return {'tail_ratio': 0.0, 'cv': 0.0}
        
        tail_95 = np.percentile(degrees, 95)
        tail_99 = np.percentile(degrees, 99) 
        mean_degree = np.mean(degrees)
        
        return {
            'tail_ratio_95': tail_95 / mean_degree if mean_degree > 0 else 0.0,
            'tail_ratio_99': tail_99 / mean_degree if mean_degree > 0 else 0.0,
            'coefficient_variation': np.std(degrees) / mean_degree if mean_degree > 0 else 0.0,
            'max_degree': np.max(degrees),
            'mean_degree': mean_degree
        }

