import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import time
import warnings
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import os

# Import the graph generation classes
from graph_universe.model import GraphUniverse, GraphSample, GraphFamilyGenerator

warnings.filterwarnings('ignore')

@dataclass
class ParameterInterval:
    """Represents a parameter interval for testing."""
    name: str
    intervals: List[Tuple[float, float]]
    param_type: str  # 'range', 'single', 'categorical'
    description: str

class ParameterRobustnessAnalyzer:
    """
    Comprehensive analyzer for testing parameter robustness in graph family generation.
    
    This class implements the experimental design where each parameter is varied individually
    while all other parameters are randomized, then measures both interval coverage and
    signal correlations.
    """
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results = {}
        self.experiment_stats = {}
        
        # Define parameter intervals to test
        self.parameter_intervals = {
            'homophily': ParameterInterval(
                name='homophily',
                intervals=[(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)],
                param_type='range',
                description='Target homophily range'
            ),
            'avg_degree': ParameterInterval(
                name='avg_degree',
                intervals=[(1.0, 3.0), (3.0, 5.0), (5.0, 10.0)],
                param_type='range',
                description='Target average degree range'
            ),
            'n_nodes': ParameterInterval(
                name='n_nodes',
                intervals=[(50, 100), (100, 200), (200, 400)],
                param_type='range',
                description='Number of nodes range'
            ),
            'n_communities': ParameterInterval(
                name='n_communities',
                intervals=[(2, 4), (4, 6), (6, 10)],
                param_type='range',
                description='Number of communities range'
            ),
            'power_law_exponent': ParameterInterval(
                name='power_law_exponent',
                intervals=[(2.0, 2.5), (2.5, 3.0), (3.0, 3.5)],
                param_type='range',
                description='Power law exponent range'
            ),
            'degree_separation': ParameterInterval(
                name='degree_separation',
                intervals=[(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)],
                param_type='range',
                description='Degree separation range'
            )
        }
    
    def run_comprehensive_analysis(
        self,
        n_graphs_per_interval: int = 50,
        timeout_minutes: float = 10.0,
        max_attempts_per_graph: int = 10,
        save_results: bool = True,
        results_dir: str = "parameter_robustness_results",
        skip_signal_calculations: bool = False,
        save_detailed_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive parameter robustness analysis.
        
        Args:
            n_graphs_per_interval: Number of graphs to generate per interval
            timeout_minutes: Timeout for each graph generation
            max_attempts_per_graph: Maximum attempts per graph
            save_results: Whether to save results to disk
            results_dir: Directory to save results
            skip_signal_calculations: Whether to skip signal calculations for faster runs
            save_detailed_data: Whether to save detailed data for each graph family
            
        Returns:
            Dictionary with all analysis results
        """
        print("🎯 Starting Comprehensive Parameter Robustness Analysis")
        if skip_signal_calculations:
            print("⚡ Signal calculations disabled for faster execution")
        print("=" * 60)
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        all_results = {}
        experiment_start = time.time()
        
        # Run analysis for each parameter
        for param_name, param_config in self.parameter_intervals.items():
            print(f"\n📊 Testing Parameter: {param_name}")
            print(f"   Intervals: {param_config.intervals}")
            print(f"   Description: {param_config.description}")
            
            param_results = self._analyze_single_parameter(
                param_name=param_name,
                param_config=param_config,
                n_graphs_per_interval=n_graphs_per_interval,
                timeout_minutes=timeout_minutes,
                max_attempts_per_graph=max_attempts_per_graph,
                skip_signal_calculations=skip_signal_calculations,
                save_detailed_data=save_detailed_data
            )
            
            all_results[param_name] = param_results
            
            # Save intermediate results
            if save_results:
                self._save_parameter_results(param_name, param_results, results_dir)
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary(all_results, skip_signal_calculations)
        
        # Save comprehensive results
        if save_results:
            self._save_comprehensive_results(all_results, summary, results_dir, skip_signal_calculations)
        
        total_time = time.time() - experiment_start
        print(f"\n✅ Analysis Complete! Total time: {total_time:.1f}s")
        print(f"📁 Results saved to: {results_dir}")
        
        return {
            'parameter_results': all_results,
            'summary': summary,
            'experiment_stats': {
                'total_time': total_time,
                'total_families': sum(len(r['families']) for r in all_results.values()),
                'total_graphs': sum(len(r['families']) * 10 for r in all_results.values()),  # 10 graphs per family
                'parameters_tested': len(self.parameter_intervals),
                'signal_calculations_skipped': skip_signal_calculations
            }
        }
    
    def _analyze_single_parameter(
        self,
        param_name: str,
        param_config: ParameterInterval,
        n_graphs_per_interval: int,
        timeout_minutes: float,
        max_attempts_per_graph: int,
        skip_signal_calculations: bool = False,
        save_detailed_data: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single parameter across its intervals.
        """
        results = {
            'parameter_name': param_name,
            'parameter_config': param_config,
            'intervals': [],
            'families': [],
            'coverage_stats': [],
            'signal_correlations': {}
        }
        
        for interval_idx, (interval_min, interval_max) in enumerate(param_config.intervals):
            print(f"   Testing interval {interval_idx + 1}/{len(param_config.intervals)}: [{interval_min:.2f}, {interval_max:.2f}]")
            
            interval_results = self._test_single_interval(
                param_name=param_name,
                interval=(interval_min, interval_max),
                n_families=n_graphs_per_interval,
                n_graphs_per_family=10,  # Generate 10 graphs per family
                timeout_minutes=timeout_minutes,
                max_attempts_per_graph=max_attempts_per_graph,
                skip_signal_calculations=skip_signal_calculations,
                save_detailed_data=save_detailed_data
            )
            
            results['intervals'].append({
                'interval': (interval_min, interval_max),
                'midpoint': (interval_min + interval_max) / 2,
                'results': interval_results
            })
            
            results['families'].extend(interval_results['families'])
            results['coverage_stats'].append(interval_results['coverage_stats'])
        
        # Calculate signal correlations
        if not skip_signal_calculations:
            results['signal_correlations'] = self._calculate_signal_correlations(results)
        
        # Calculate consistency correlations
        results['consistency_correlations'] = self._calculate_consistency_correlations(results)
        
        return results
    
    def _test_single_interval(
        self,
        param_name: str,
        interval: Tuple[float, float],
        n_families: int,
        n_graphs_per_family: int,
        timeout_minutes: float,
        max_attempts_per_graph: int,
        skip_signal_calculations: bool = False,
        save_detailed_data: bool = True
    ) -> Dict[str, Any]:
        """
        Test a single parameter interval with randomized other parameters.
        Generate families of graphs and analyze their properties.
        """
        families = []
        family_properties = []
        family_signals = []
        family_consistencies = []
        detailed_family_data = []  # New: store detailed data for each family
        
        successful_generations = 0
        
        for family_idx in range(n_families):
            try:
                # Generate random parameters for everything except the target parameter
                # Use family-specific seed to ensure independent randomization for each family
                family_seed = self.base_seed + family_idx
                params = self._generate_random_parameters(param_name, interval, seed=family_seed)
                
                # Create universe
                universe = GraphUniverse(
                    K=params['K'],
                    feature_dim=params['feature_dim'],
                    inter_community_variance=params['inter_community_variance'],
                    center_variance=params['center_variance'],
                    cluster_variance=params['cluster_variance'],
                    degree_center_method=params['degree_center_method'],
                    community_cooccurrence_homogeneity=params['community_cooccurrence_homogeneity'],
                    seed=self.base_seed + family_idx
                )
                
                # Create family generator
                family_generator = GraphFamilyGenerator(
                    universe=universe,
                    min_n_nodes=params['min_n_nodes'],
                    max_n_nodes=params['max_n_nodes'],
                    min_communities=params['min_communities'],
                    max_communities=params['max_communities'],
                    min_component_size=params['min_component_size'],
                    homophily_range=params['homophily_range'],
                    avg_degree_range=params['avg_degree_range'],
                    use_dccc_sbm=params['use_dccc_sbm'],
                    community_cooccurrence_homogeneity=params['community_cooccurrence_homogeneity'],
                    disable_deviation_limiting=params['disable_deviation_limiting'],
                    max_mean_community_deviation=params['max_mean_community_deviation'],
                    max_max_community_deviation=params['max_max_community_deviation'],
                    min_edge_density=params['min_edge_density'],
                    degree_distribution=params['degree_distribution'],
                    power_law_exponent_range=params['power_law_exponent_range'],
                    exponential_rate_range=params['exponential_rate_range'],
                    uniform_min_factor_range=params['uniform_min_factor_range'],
                    uniform_max_factor_range=params['uniform_max_factor_range'],
                    degree_separation_range=params['degree_separation_range'],
                    degree_heterogeneity=params['degree_heterogeneity'],
                    max_retries=params['max_retries'],
                    seed=self.base_seed + family_idx
                )
                
                # Generate family of graphs
                family_graphs = family_generator.generate_family(
                    n_graphs=n_graphs_per_family,
                    show_progress=False,
                    collect_stats=False,
                    max_attempts_per_graph=max_attempts_per_graph,
                    timeout_minutes=timeout_minutes
                )
                
                # Analyze family properties
                try:
                    family_props = family_generator.analyze_graph_family_properties()
                    family_properties.append(family_props)
                except Exception as e:
                    print(f"     Warning: Failed to analyze family properties for family {family_idx + 1}: {str(e)}")
                    family_properties.append({})
                
                # Analyze family signals
                if not skip_signal_calculations:
                    try:
                        family_sigs = family_generator.analyze_graph_family_signals()
                        family_signals.append(family_sigs)
                    except Exception as e:
                        print(f"     Warning: Failed to analyze family signals for family {family_idx + 1}: {str(e)}")
                        family_signals.append({})
                
                # Analyze family consistency
                try:
                    family_consistency = family_generator.analyze_graph_family_consistency()
                    family_consistencies.append(family_consistency)
                except Exception as e:
                    print(f"     Warning: Failed to analyze family consistency for family {family_idx + 1}: {str(e)}")
                    family_consistencies.append({})
                
                # Collect detailed data for this family
                if save_detailed_data:
                    detailed_family_info = self._collect_detailed_family_data(
                        family_generator, family_props, family_sigs if not skip_signal_calculations else {}, 
                        family_consistency, params, param_name, interval, family_idx
                    )
                    detailed_family_data.append(detailed_family_info)
                
                families.append(family_generator)
                successful_generations += 1
                
                if (family_idx + 1) % 5 == 0:
                    print(f"     Generated {family_idx + 1}/{n_families} families")
                    
            except Exception as e:
                print(f"     Failed to generate family {family_idx + 1}: {str(e)}")
                continue
        
        # Extract actual parameter values from family properties
        actual_values = self._extract_parameter_values_from_families(family_properties, param_name)
        
        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage_stats(actual_values, interval)
        
        return {
            'families': families,
            'family_properties': family_properties,
            'family_signals': family_signals,
            'family_consistencies': family_consistencies,
            'detailed_family_data': detailed_family_data,  # New: include detailed data
            'actual_values': actual_values,
            'coverage_stats': coverage_stats,
            'success_rate': successful_generations / n_families
        }
    
    def _generate_random_parameters(self, target_param: str, target_interval: Tuple[float, float], seed: int = None) -> Dict[str, Any]:
        """
        Generate random parameters, fixing the target parameter to the specified interval.
        
        Args:
            target_param: The parameter being tested
            target_interval: The interval for the target parameter
            seed: Optional seed for reproducible randomization
        """
        # Set random seed for this parameter generation if provided
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        # Universe parameters - ensure K is large enough
        K = rng.randint(10, 15)  # Increased minimum to avoid max_communities issues
        feature_dim = rng.choice([5, 10, 20])
        inter_community_variance = rng.uniform(0.0, 1.0)
        center_variance = rng.uniform(0.5, 2.0)
        cluster_variance = rng.uniform(0.05, 1.0)
        degree_center_method = rng.choice(['random', 'constant'])
        community_cooccurrence_homogeneity = rng.uniform(0.0, 1.0)
        
        # Graph family parameters - ensure max_communities <= K
        min_n_nodes = rng.randint(30, 100)
        max_n_nodes = rng.randint(min_n_nodes + 50, 300)
        min_communities = rng.randint(2, min(6, K-2))  # Ensure min_communities < K
        max_communities = rng.randint(min_communities + 1, min(10, K))  # Ensure max_communities <= K
        min_component_size = rng.randint(5, 20)
        
        # Set target parameter based on type
        if target_param == 'homophily':
            homophily_range = target_interval
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
        elif target_param == 'avg_degree':
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = target_interval
        elif target_param == 'n_nodes':
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
            min_n_nodes, max_n_nodes = target_interval
        elif target_param == 'n_communities':
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
            min_communities, max_communities = target_interval
        elif target_param == 'power_law_exponent':
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
            power_law_exponent_range = target_interval
        elif target_param == 'degree_separation':
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
            degree_separation_range = target_interval
        else:
            # Default case
            homophily_range = (rng.uniform(0.0, 0.3), rng.uniform(0.3, 0.8))
            avg_degree_range = (rng.uniform(1.0, 3.0), rng.uniform(3.0, 8.0))
        
        # DCCC-SBM parameters
        use_dccc_sbm = rng.choice([True, False])
        degree_distribution = rng.choice(['power_law'])
        
        if degree_distribution == 'power_law':
            power_law_exponent_range = (rng.uniform(2.0, 2.5), rng.uniform(2.5, 3.5))
            exponential_rate_range = (0.5, 0.5)
            uniform_min_factor_range = (0.5, 0.5)
            uniform_max_factor_range = (1.5, 1.5)
        elif degree_distribution == 'exponential':
            power_law_exponent_range = (2.5, 2.5)
            exponential_rate_range = (rng.uniform(0.3, 0.7), rng.uniform(0.7, 1.2))
            uniform_min_factor_range = (0.5, 0.5)
            uniform_max_factor_range = (1.5, 1.5)
        else:  # uniform
            power_law_exponent_range = (2.5, 2.5)
            exponential_rate_range = (0.5, 0.5)
            uniform_min_factor_range = (rng.uniform(0.3, 0.5), rng.uniform(0.5, 0.8))
            uniform_max_factor_range = (rng.uniform(1.2, 1.5), rng.uniform(1.5, 2.0))
        
        degree_separation_range = (rng.uniform(0.0, 0.5), rng.uniform(0.5, 1.0))
        degree_heterogeneity = rng.uniform(0.3, 0.8)
        
        # Deviation limiting - more lenient for robustness testing
        disable_deviation_limiting = rng.choice([True, False], p=[0.7, 0.3])  # Prefer True
        max_mean_community_deviation = rng.uniform(0.15, 0.25)  # More lenient
        max_max_community_deviation = rng.uniform(0.20, 0.35)  # More lenient
        min_edge_density = rng.uniform(0.001, 0.005)  # Lower minimum
        max_retries = rng.randint(5, 12)  # More retries
        
        return {
            'K': K,
            'feature_dim': feature_dim,
            'inter_community_variance': inter_community_variance,
            'center_variance': center_variance,
            'cluster_variance': cluster_variance,
            'degree_center_method': degree_center_method,
            'community_cooccurrence_homogeneity': community_cooccurrence_homogeneity,
            'min_n_nodes': min_n_nodes,
            'max_n_nodes': max_n_nodes,
            'min_communities': min_communities,
            'max_communities': max_communities,
            'min_component_size': min_component_size,
            'homophily_range': homophily_range,
            'avg_degree_range': avg_degree_range,
            'use_dccc_sbm': use_dccc_sbm,
            'degree_distribution': degree_distribution,
            'power_law_exponent_range': power_law_exponent_range,
            'exponential_rate_range': exponential_rate_range,
            'uniform_min_factor_range': uniform_min_factor_range,
            'uniform_max_factor_range': uniform_max_factor_range,
            'degree_separation_range': degree_separation_range,
            'degree_heterogeneity': degree_heterogeneity,
            'disable_deviation_limiting': disable_deviation_limiting,
            'max_mean_community_deviation': max_mean_community_deviation,
            'max_max_community_deviation': max_max_community_deviation,
            'min_edge_density': min_edge_density,
            'max_retries': max_retries
        }
    
    def _fit_power_law_exponent(self, degrees: List[int]) -> float:
        """
        Fit power law exponent to degree distribution using discrete MLE.
        This method is specifically designed for discrete network degrees.
        
        Args:
            degrees: List of node degrees
            
        Returns:
            Fitted power law exponent
        """
        if not degrees or len(degrees) < 2:
            return 2.5  # Default value if insufficient data
        
        # Convert to numpy array and filter out zeros
        degrees_array = np.array(degrees, dtype=int)
        degrees_array = degrees_array[degrees_array > 0]
        
        if len(degrees_array) < 2:
            return 2.5
        
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
                return 2.5  # Default value if optimization fails
        except Exception as e:
            print(f"Warning: Failed to fit power law exponent: {e}")
            return 2.5
    
    def _extract_parameter_values_from_families(self, family_properties: List[Dict[str, Any]], param_name: str) -> List[float]:
        """
        Extract the actual parameter values from family properties.
        """
        actual_values = []
        
        for family_props in family_properties:
            if not family_props:
                actual_values.append(0.0)
                continue
                
            if param_name == 'homophily':
                # Use mean homophily from family
                actual_values.append(family_props.get('homophily_levels_mean', 0.0))
            elif param_name == 'avg_degree':
                # Use mean average degree from family
                actual_values.append(family_props.get('avg_degrees_mean', 0.0))
            elif param_name == 'n_nodes':
                # Use mean node count from family
                actual_values.append(family_props.get('node_counts_mean', 0.0))
            elif param_name == 'n_communities':
                # Use mean community count from family
                actual_values.append(family_props.get('community_counts_mean', 0.0))
            elif param_name == 'power_law_exponent':
                # Extract degree distributions and fit power law
                degree_distributions = family_props.get('degree_distributions', [])
                if degree_distributions:
                    # Fit power law to the combined degree distribution
                    all_degrees = []
                    for deg_dist in degree_distributions:
                        if isinstance(deg_dist, list):
                            all_degrees.extend(deg_dist)
                        elif isinstance(deg_dist, dict):
                            # If it's a dict with degree counts
                            for degree, count in deg_dist.items():
                                all_degrees.extend([int(degree)] * int(count))
                    
                    if all_degrees:
                        actual_values.append(self._fit_power_law_exponent(all_degrees))
                    else:
                        actual_values.append(2.5)
                else:
                    actual_values.append(2.5)
            elif param_name == 'degree_separation':
                # This is harder to extract - return the target value for now
                actual_values.append(0.5)  # Placeholder
            else:
                actual_values.append(0.0)
        
        return actual_values
    
    def _calculate_actual_homophily(self, graph: GraphSample) -> float:
        """
        Calculate actual homophily from the generated graph.
        """
        if graph.n_nodes == 0:
            return 0.0
        
        # Count edges within communities vs between communities
        within_edges = 0
        total_edges = graph.graph.number_of_edges()
        
        if total_edges == 0:
            return 0.0
        
        for u, v in graph.graph.edges():
            if graph.community_labels[u] == graph.community_labels[v]:
                within_edges += 1
        
        return within_edges / total_edges
    
    def _calculate_coverage_stats(self, actual_values: List[float], target_interval: Tuple[float, float]) -> Dict[str, float]:
        """
        Calculate coverage statistics for actual values vs target interval.
        """
        if not actual_values:
            return {
                'coverage_ratio': 0.0,
                'mean_deviation': 0.0,
                'std_deviation': 0.0,
                'min_value': 0.0,
                'max_value': 0.0,
                'mean_value': 0.0
            }
        
        actual_values = np.array(actual_values)
        min_target, max_target = target_interval
        
        # Calculate coverage ratio
        within_interval = np.sum((actual_values >= min_target) & (actual_values <= max_target))
        coverage_ratio = within_interval / len(actual_values)
        
        # Calculate deviations
        deviations = np.abs(actual_values - (min_target + max_target) / 2)
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        
        return {
            'coverage_ratio': coverage_ratio,
            'mean_deviation': mean_deviation,
            'std_deviation': std_deviation,
            'min_value': np.min(actual_values),
            'max_value': np.max(actual_values),
            'mean_value': np.mean(actual_values)
        }
    
    def _calculate_signal_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate correlations between parameter values and signal strengths.
        """
        correlations = {}
        
        # Extract data
        midpoints = [interval['midpoint'] for interval in results['intervals']]
        
        for signal_name in ['structure_signal', 'degree_signal', 'feature_signal', 'triangle_signal']:
            signal_means = []
            signal_stds = []
            
            for interval_result in results['intervals']:
                family_signals = interval_result['results']['family_signals']
                # Extract signal values from family signals
                signal_values = []
                for family_sig in family_signals:
                    if family_sig and signal_name in family_sig:
                        # family_sig[signal_name] is a list of signal values for each graph in the family
                        signal_list = family_sig[signal_name]
                        if isinstance(signal_list, list):
                            # Take the mean of all signal values in the family
                            valid_values = [float(v) for v in signal_list if v is not None and not np.isnan(v)]
                            if valid_values:
                                signal_values.append(np.mean(valid_values))
                        else:
                            # Single value case
                            signal_value = signal_list
                            if signal_value is not None and not np.isnan(signal_value):
                                signal_values.append(float(signal_value))
                
                if signal_values:
                    signal_means.append(np.mean(signal_values))
                    signal_stds.append(np.std(signal_values))
                else:
                    signal_means.append(0.0)
                    signal_stds.append(0.0)
            
            # Calculate correlation
            if len(midpoints) > 1 and len(signal_means) > 1:
                correlation, p_value = stats.pearsonr(midpoints, signal_means)
                r_squared = correlation ** 2
            else:
                correlation = 0.0
                p_value = 1.0
                r_squared = 0.0
            
            correlations[signal_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'r_squared': r_squared,
                'signal_means': signal_means,
                'signal_stds': signal_stds
            }
        
        return correlations
    
    def _calculate_consistency_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate correlations between parameter values and consistency metrics.
        """
        correlations = {}
        
        # Extract data
        midpoints = [interval['midpoint'] for interval in results['intervals']]
        
        for consistency_name in ['pattern_preservation', 'generation_fidelity', 'degree_consistency', 'cooccurrence_consistency']:
            consistency_means = []
            consistency_stds = []
            
            for interval_result in results['intervals']:
                family_consistencies = interval_result['results']['family_consistencies']
                # Extract consistency values from family consistencies
                consistency_values = []
                for family_cons in family_consistencies:
                    if family_cons and consistency_name in family_cons:
                        # family_cons[consistency_name] is a list of consistency values for each graph in the family
                        consistency_list = family_cons[consistency_name]
                        if isinstance(consistency_list, list):
                            # Take the mean of all consistency values in the family
                            valid_values = [float(v) for v in consistency_list if v is not None and not np.isnan(v)]
                            if valid_values:
                                consistency_values.append(np.mean(valid_values))
                        else:
                            # Single value case
                            consistency_value = consistency_list
                            if consistency_value is not None and not np.isnan(consistency_value):
                                consistency_values.append(float(consistency_value))
                
                if consistency_values:
                    consistency_means.append(np.mean(consistency_values))
                    consistency_stds.append(np.std(consistency_values))
                else:
                    consistency_means.append(0.0)
                    consistency_stds.append(0.0)
            
            # Calculate correlation
            if len(midpoints) > 1 and len(consistency_means) > 1:
                correlation, p_value = stats.pearsonr(midpoints, consistency_means)
                r_squared = correlation ** 2
            else:
                correlation = 0.0
                p_value = 1.0
                r_squared = 0.0
            
            correlations[consistency_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'r_squared': r_squared,
                'consistency_means': consistency_means,
                'consistency_stds': consistency_stds
            }
        
        return correlations
    
    def _generate_comprehensive_summary(self, all_results: Dict[str, Any], skip_signal_calculations: bool) -> Dict[str, Any]:
        """
        Generate comprehensive summary of all results.
        """
        summary = {
            'overall_coverage': {},
            'best_performing_parameters': [],
            'signal_correlation_summary': {},
            'experiment_quality_metrics': {}
        }
        
        # Calculate overall coverage for each parameter
        for param_name, results in all_results.items():
            coverage_ratios = [interval['results']['coverage_stats']['coverage_ratio'] 
                             for interval in results['intervals']]
            summary['overall_coverage'][param_name] = {
                'mean_coverage': np.mean(coverage_ratios),
                'min_coverage': np.min(coverage_ratios),
                'max_coverage': np.max(coverage_ratios),
                'std_coverage': np.std(coverage_ratios)
            }
        
        # Find best performing parameters
        coverage_means = {param: data['mean_coverage'] 
                         for param, data in summary['overall_coverage'].items()}
        sorted_params = sorted(coverage_means.items(), key=lambda x: x[1], reverse=True)
        summary['best_performing_parameters'] = sorted_params
        
        # Signal correlation summary
        for param_name, results in all_results.items():
            summary['signal_correlation_summary'][param_name] = {}
            if not skip_signal_calculations:
                for signal_name, signal_data in results['signal_correlations'].items():
                    summary['signal_correlation_summary'][param_name][signal_name] = {
                        'r_squared': signal_data['r_squared'],
                        'correlation': signal_data['correlation'],
                        'p_value': signal_data['p_value']
                    }
        
        # Consistency correlation summary
        summary['consistency_correlation_summary'] = {}
        for param_name, results in all_results.items():
            summary['consistency_correlation_summary'][param_name] = {}
            for consistency_name, consistency_data in results['consistency_correlations'].items():
                summary['consistency_correlation_summary'][param_name][consistency_name] = {
                    'r_squared': consistency_data['r_squared'],
                    'correlation': consistency_data['correlation'],
                    'p_value': consistency_data['p_value']
                }
        
        return summary
    
    def _save_parameter_results(self, param_name: str, results: Dict[str, Any], results_dir: str):
        """
        Save results for a single parameter.
        """
        filename = os.path.join(results_dir, f"{param_name}_results.json")
        
        # Create a copy of results without the family objects
        results_for_json = {
            'parameter_name': results['parameter_name'],
            'parameter_config': results['parameter_config'],
            'intervals': results['intervals'],
            'coverage_stats': results['coverage_stats'],
            'signal_correlations': results['signal_correlations'],
            # Exclude the families list since GraphFamilyGenerator objects are not JSON serializable
            'family_count': len(results.get('families', [])),
            'actual_values': results.get('actual_values', [])
        }
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results_for_json)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable objects to JSON-serializable format.
        """
        if isinstance(obj, dict):
            # Handle numpy types as both keys and values
            new_dict = {}
            for key, value in obj.items():
                # Convert numpy types in keys
                if isinstance(key, np.integer):
                    new_key = int(key)
                elif isinstance(key, np.floating):
                    new_key = float(key)
                elif isinstance(key, np.bool_):
                    new_key = bool(key)
                else:
                    new_key = key
                new_dict[new_key] = self._make_json_serializable(value)
            return new_dict
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dataclass_fields__'):  # Handle dataclasses
            return {
                'name': obj.name,
                'intervals': obj.intervals,
                'param_type': obj.param_type,
                'description': obj.description
            }
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'GraphSample':
            # Handle GraphSample objects by returning a summary
            return f"GraphSample(n_nodes={getattr(obj, 'n_nodes', 0)})"
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'GraphFamilyGenerator':
            # Handle GraphFamilyGenerator objects by returning a summary
            return f"GraphFamilyGenerator(n_families={len(getattr(obj, 'graphs', []))})"
        else:
            return obj
    
    def _collect_detailed_family_data(
        self,
        family_generator,
        family_props: Dict[str, Any],
        family_signals: Dict[str, Any],
        family_consistency: Dict[str, Any],
        params: Dict[str, Any],
        param_name: str,
        interval: Tuple[float, float],
        family_idx: int
    ) -> Dict[str, Any]:
        """
        Collect detailed data for a single graph family.
        """
        detailed_data = {
            'family_id': family_idx,
            'target_parameter': param_name,
            'target_interval': interval,
            'generation_parameters': params,
            'family_properties': family_props,
            'family_signals': family_signals,
            'family_consistency': family_consistency,
            'individual_graphs': []
        }
        
        # Collect data for each individual graph in the family
        for graph_idx, graph in enumerate(family_generator.graphs):
            graph_data = {
                'graph_id': graph_idx,
                'n_nodes': graph.n_nodes,
                'n_edges': graph.graph.number_of_edges(),
                'n_communities': len(set(graph.community_labels)),
                'actual_homophily': self._calculate_actual_homophily(graph),
                'actual_avg_degree': 2 * graph.graph.number_of_edges() / graph.n_nodes if graph.n_nodes > 0 else 0,
                'community_labels': graph.community_labels.tolist() if hasattr(graph, 'community_labels') else [],
                'degree_distribution': dict(zip(*np.unique([d for n, d in graph.graph.degree()], return_counts=True))) if graph.graph.number_of_edges() > 0 else {},
                'edge_density': graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1)) if graph.n_nodes > 1 else 0,
                'is_connected': nx.is_connected(graph.graph) if graph.graph.number_of_edges() > 0 else False,
                'largest_component_size': len(max(nx.connected_components(graph.graph), key=len)) if graph.graph.number_of_edges() > 0 else 0
            }
            
            # Add feature information if available
            if hasattr(graph, 'features') and graph.features is not None:
                graph_data['feature_dim'] = graph.features.shape[1] if len(graph.features.shape) > 1 else 1
                graph_data['feature_stats'] = {
                    'mean': float(np.mean(graph.features)),
                    'std': float(np.std(graph.features)),
                    'min': float(np.min(graph.features)),
                    'max': float(np.max(graph.features))
                }
            
            detailed_data['individual_graphs'].append(graph_data)
        
        return detailed_data
    
    def _save_comprehensive_results(
        self,
        all_results: Dict[str, Any],
        summary: Dict[str, Any],
        results_dir: str,
        skip_signal_calculations: bool
    ):
        """
        Save comprehensive results in a structured JSON format.
        """
        print("\n💾 Saving comprehensive results...")
        
        # Create the comprehensive results structure
        comprehensive_results = {
            'experiment_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'base_seed': self.base_seed,
                'parameters_tested': list(self.parameter_intervals.keys()),
                'signal_calculations_skipped': skip_signal_calculations,
                'parameter_intervals': {
                    name: {
                        'intervals': config.intervals,
                        'param_type': config.param_type,
                        'description': config.description
                    }
                    for name, config in self.parameter_intervals.items()
                }
            },
            'summary_statistics': summary,
            'detailed_results': {}
        }
        
        # Add detailed results for each parameter
        for param_name, results in all_results.items():
            param_detailed_results = {
                'parameter_name': param_name,
                'parameter_config': {
                    'intervals': results['parameter_config'].intervals,
                    'param_type': results['parameter_config'].param_type,
                    'description': results['parameter_config'].description
                },
                'intervals': [],
                'coverage_stats': results['coverage_stats'],
                'signal_correlations': results.get('signal_correlations', {}),
                'consistency_correlations': results.get('consistency_correlations', {}),
                'actual_values': results.get('actual_values', [])
            }
            
            # Add detailed data for each interval
            for interval_idx, interval_data in enumerate(results['intervals']):
                interval_detailed = {
                    'interval_index': interval_idx,
                    'target_interval': interval_data['interval'],
                    'midpoint': interval_data['midpoint'],
                    'coverage_stats': interval_data['results']['coverage_stats'],
                    'success_rate': interval_data['results']['success_rate'],
                    'actual_values': interval_data['results']['actual_values'],
                    'detailed_family_data': interval_data['results'].get('detailed_family_data', [])
                }
                
                param_detailed_results['intervals'].append(interval_detailed)
            
            comprehensive_results['detailed_results'][param_name] = param_detailed_results
        
        # Save the comprehensive results
        comprehensive_file = os.path.join(results_dir, 'comprehensive_results.json')
        
        # Convert to JSON-serializable format
        serializable_results = self._make_json_serializable(comprehensive_results)
        
        with open(comprehensive_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Comprehensive results saved to: {comprehensive_file}")
        
        # Also save a summary file for quick reference
        summary_file = os.path.join(results_dir, 'summary_results.json')
        summary_data = {
            'experiment_metadata': comprehensive_results['experiment_metadata'],
            'summary_statistics': summary,
            'parameter_summaries': {}
        }
        
        for param_name, results in all_results.items():
            param_summary = {
                'mean_coverage': summary['overall_coverage'][param_name]['mean_coverage'],
                'std_coverage': summary['overall_coverage'][param_name]['std_coverage'],
                'best_performing_rank': next((i for i, (p, _) in enumerate(summary['best_performing_parameters']) if p == param_name), -1),
                'total_families': len(results['families']),
                'total_graphs': len(results['families']) * 10  # Assuming 10 graphs per family
            }
            summary_data['parameter_summaries'][param_name] = param_summary
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Summary results saved to: {summary_file}")
        
        # Save individual parameter files (existing functionality)
        for param_name, results in all_results.items():
            self._save_parameter_results(param_name, results, results_dir)
    
    def _create_comprehensive_visualizations(self, all_results: Dict[str, Any], results_dir: str, skip_signal_calculations: bool):
        """
        Create comprehensive visualizations for all results.
        """
        print("\n📊 Creating visualizations...")
        
        # 1. Coverage plots
        self._create_coverage_plots(all_results, results_dir)
        
        # 2. Signal correlation plots
        if not skip_signal_calculations:
            self._create_signal_correlation_plots(all_results, results_dir)
        
        # 3. Consistency correlation plots
        self._create_consistency_correlation_plots(all_results, results_dir)
        
        # 3. Summary dashboard
        self._create_summary_dashboard(all_results, results_dir, skip_signal_calculations)
    
    def _create_coverage_plots(self, all_results: Dict[str, Any], results_dir: str):
        """
        Create grouped boxplots showing actual parameter values for each interval.
        """
        # Only create plots for parameters we can actually measure
        measurable_params = ['homophily', 'avg_degree', 'n_nodes', 'power_law_exponent']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, param_name in enumerate(measurable_params):
            if param_name not in all_results or idx >= len(axes):
                continue
                
            ax = axes[idx]
            results = all_results[param_name]
            
            # Prepare data for boxplots
            boxplot_data = []
            interval_labels = []
            
            for interval in results['intervals']:
                actual_values = interval['results']['actual_values']
                if actual_values and any(v > 0 for v in actual_values):
                    boxplot_data.append(actual_values)
                    interval_label = f"[{interval['interval'][0]:.2f}, {interval['interval'][1]:.2f}]"
                    interval_labels.append(interval_label)
            
            if boxplot_data:
                # Create boxplot
                bp = ax.boxplot(boxplot_data, labels=interval_labels, patch_artist=True)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                # Add target intervals as horizontal lines
                for i, interval in enumerate(results['intervals']):
                    if i < len(boxplot_data):
                        min_val, max_val = interval['interval']
                        ax.axhline(y=min_val, color='red', linestyle='--', alpha=0.5, xmin=0.1, xmax=0.9)
                        ax.axhline(y=max_val, color='red', linestyle='--', alpha=0.5, xmin=0.1, xmax=0.9)
                        # Fill the target interval
                        ax.axhspan(min_val, max_val, alpha=0.1, color='red')
                
                ax.set_title(f'{param_name.replace("_", " ").title()} Actual Values')
                ax.set_ylabel('Actual Parameter Value')
                ax.set_xlabel('Target Interval')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                if len(interval_labels) > 3:
                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param_name.replace("_", " ").title()} Actual Values')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'coverage_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_signal_correlation_plots(self, all_results: Dict[str, Any], results_dir: str):
        """
        Create signal correlation plots showing relationships between parameters and signals.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (param_name, results) in enumerate(all_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract data
            midpoints = [interval['midpoint'] for interval in results['intervals']]
            
            # Plot each signal type
            colors = ['blue', 'green', 'red', 'orange']
            signal_names = ['structure_signal', 'degree_signal', 'feature_signal', 'triangle_signal']
            
            for signal_idx, signal_name in enumerate(signal_names):
                if signal_name in results['signal_correlations']:
                    signal_data = results['signal_correlations'][signal_name]
                    signal_means = signal_data['signal_means']
                    signal_stds = signal_data['signal_stds']
                    
                    # Plot with error bars
                    ax.errorbar(midpoints, signal_means, yerr=signal_stds, 
                              marker='o', label=signal_name.replace('_', ' ').title(),
                              color=colors[signal_idx], capsize=5)
            
            ax.set_title(f'{param_name.replace("_", " ").title()} vs Signals')
            ax.set_xlabel('Parameter Midpoint')
            ax.set_ylabel('Signal Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'signal_correlation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_consistency_correlation_plots(self, all_results: Dict[str, Any], results_dir: str):
        """
        Create consistency correlation plots showing relationships between parameters and consistency metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (param_name, results) in enumerate(all_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract data
            midpoints = [interval['midpoint'] for interval in results['intervals']]
            
            # Plot each consistency type
            colors = ['blue', 'green', 'red', 'orange']
            consistency_names = ['pattern_preservation', 'generation_fidelity', 'degree_consistency', 'cooccurrence_consistency']
            
            for consistency_idx, consistency_name in enumerate(consistency_names):
                if consistency_name in results['consistency_correlations']:
                    consistency_data = results['consistency_correlations'][consistency_name]
                    consistency_means = consistency_data['consistency_means']
                    consistency_stds = consistency_data['consistency_stds']
                    
                    # Plot with error bars
                    ax.errorbar(midpoints, consistency_means, yerr=consistency_stds, 
                              marker='o', label=consistency_name.replace('_', ' ').title(),
                              color=colors[consistency_idx], capsize=5)
            
            ax.set_title(f'{param_name.replace("_", " ").title()} vs Consistency')
            ax.set_xlabel('Parameter Midpoint')
            ax.set_ylabel('Consistency Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'consistency_correlation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_dashboard(self, all_results: Dict[str, Any], results_dir: str, skip_signal_calculations: bool = False):
        """
        Create a comprehensive summary dashboard.
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall coverage summary
        ax1 = fig.add_subplot(gs[0, :2])
        params = list(all_results.keys())
        mean_coverages = []
        for param in params:
            coverage_ratios = [interval['results']['coverage_stats']['coverage_ratio'] 
                             for interval in all_results[param]['intervals']]
            mean_coverages.append(np.mean(coverage_ratios))
        
        bars = ax1.bar(params, mean_coverages, color='lightcoral', alpha=0.7)
        ax1.set_title('Mean Coverage Ratio by Parameter')
        ax1.set_ylabel('Mean Coverage Ratio')
        ax1.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, coverage in zip(bars, mean_coverages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{coverage:.3f}', ha='center', va='bottom')
        
        # 2. Signal correlation heatmap (only if signals were calculated)
        if not skip_signal_calculations:
            ax2 = fig.add_subplot(gs[0, 2:])
            signal_names = ['structure_signal', 'degree_signal', 'feature_signal', 'triangle_signal']
            correlation_matrix = []
            
            for param in params:
                param_correlations = []
                for signal in signal_names:
                    if signal in all_results[param]['signal_correlations']:
                        r_squared = all_results[param]['signal_correlations'][signal]['r_squared']
                        param_correlations.append(r_squared)
                    else:
                        param_correlations.append(0.0)
                correlation_matrix.append(param_correlations)
            
            im = ax2.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto')
            ax2.set_xticks(range(len(signal_names)))
            ax2.set_yticks(range(len(params)))
            ax2.set_xticklabels([s.replace('_', ' ').title() for s in signal_names], rotation=45)
            ax2.set_yticklabels([p.replace('_', ' ').title() for p in params])
            ax2.set_title('Signal Correlation Heatmap (R²)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('R²')
        else:
            # If signals were skipped, show a placeholder
            ax2 = fig.add_subplot(gs[0, 2:])
            ax2.text(0.5, 0.5, 'Signal calculations\nwere skipped\nfor faster execution', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Signal Correlation Heatmap (Skipped)')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 3. Consistency correlation heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        consistency_names = ['pattern_preservation', 'generation_fidelity', 'degree_consistency', 'cooccurrence_consistency']
        consistency_correlation_matrix = []
        
        for param in params:
            param_correlations = []
            for consistency in consistency_names:
                if consistency in all_results[param]['consistency_correlations']:
                    r_squared = all_results[param]['consistency_correlations'][consistency]['r_squared']
                    param_correlations.append(r_squared)
                else:
                    param_correlations.append(0.0)
            consistency_correlation_matrix.append(param_correlations)
        
        im2 = ax3.imshow(consistency_correlation_matrix, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(consistency_names)))
        ax3.set_yticks(range(len(params)))
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in consistency_names], rotation=45)
        ax3.set_yticklabels([p.replace('_', ' ').title() for p in params])
        ax3.set_title('Consistency Correlation Heatmap (R²)')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax3)
        cbar2.set_label('R²')
        
        # 4. Parameter distribution plots
        for idx, param_name in enumerate(params[:2]):  # Show first 2 parameters
            ax = fig.add_subplot(gs[1, 2 + idx])
            
            # Collect all actual values
            all_values = []
            for interval in all_results[param_name]['intervals']:
                all_values.extend(interval['results']['actual_values'])
            
            if all_values:
                ax.hist(all_values, bins=20, alpha=0.7, color='lightblue')
                ax.set_title(f'{param_name.replace("_", " ").title()} Distribution')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Frequency')
        
        # 5. Success rate summary
        ax5 = fig.add_subplot(gs[2, :2])
        success_rates = []
        for param in params:
            rates = [interval['results']['success_rate'] 
                    for interval in all_results[param]['intervals']]
            success_rates.append(np.mean(rates))
        
        bars = ax5.bar(params, success_rates, color='lightgreen', alpha=0.7)
        ax5.set_title('Mean Success Rate by Parameter')
        ax5.set_ylabel('Success Rate')
        ax5.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for param in params:
            coverage_ratios = [interval['results']['coverage_stats']['coverage_ratio'] 
                             for interval in all_results[param]['intervals']]
            success_rates = [interval['results']['success_rate'] 
                           for interval in all_results[param]['intervals']]
            
            summary_data.append([
                param.replace('_', ' ').title(),
                f'{np.mean(coverage_ratios):.3f}',
                f'{np.mean(success_rates):.3f}',
                f'{len(all_results[param]["families"])}'
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Parameter', 'Mean Coverage', 'Mean Success Rate', 'Total Graphs'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        plt.savefig(os.path.join(results_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to run the parameter robustness analysis.
    """
    print("🚀 Parameter Robustness Analysis")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ParameterRobustnessAnalyzer(base_seed=42)
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis(
        n_graphs_per_interval=30,  # Reduced for faster testing
        timeout_minutes=5.0,
        max_attempts_per_graph=5,
        save_results=True,
        results_dir="parameter_robustness_results",
        skip_signal_calculations=False,  # Include signals by default
        save_detailed_data=True  # Save detailed data for each graph family
    )
    
    # Print summary
    print("\n📋 Analysis Summary:")
    print("=" * 30)
    
    summary = results['summary']
    
    print("\n🎯 Overall Coverage by Parameter:")
    for param, data in summary['overall_coverage'].items():
        print(f"  {param}: {data['mean_coverage']:.3f} ± {data['std_coverage']:.3f}")
    
    print("\n🏆 Best Performing Parameters:")
    for param, coverage in summary['best_performing_parameters'][:3]:
        print(f"  {param}: {coverage:.3f}")
    
    print("\n📊 Signal Correlation Summary:")
    if summary['signal_correlation_summary']:
        for param, signals in summary['signal_correlation_summary'].items():
            if signals:  # Check if signals were calculated
                print(f"\n  {param}:")
                for signal, data in signals.items():
                    print(f"    {signal}: R² = {data['r_squared']:.3f} (p = {data['p_value']:.3f})")
    else:
        print("  Signal calculations were skipped for faster execution")
    
    print("\n🔗 Consistency Correlation Summary:")
    if summary['consistency_correlation_summary']:
        for param, consistencies in summary['consistency_correlation_summary'].items():
            if consistencies:  # Check if consistencies were calculated
                print(f"\n  {param}:")
                for consistency, data in consistencies.items():
                    print(f"    {consistency}: R² = {data['r_squared']:.3f} (p = {data['p_value']:.3f})")
    else:
        print("  Consistency calculations were skipped")
    
    print(f"\n⏱️  Total Experiment Time: {results['experiment_stats']['total_time']:.1f}s")
    print(f"📈 Total Graphs Generated: {results['experiment_stats']['total_graphs']}")
    print(f"🔬 Parameters Tested: {results['experiment_stats']['parameters_tested']}")
    if 'signal_calculations_skipped' in results['experiment_stats']:
        print(f"⚡ Signal Calculations: {'Skipped' if results['experiment_stats']['signal_calculations_skipped'] else 'Included'}")
    
    print(f"\n💾 Data saved in comprehensive JSON format:")
    print(f"   - comprehensive_results.json: Complete detailed data")
    print(f"   - summary_results.json: Quick summary statistics")
    print(f"   - Individual parameter files for each tested parameter")

if __name__ == "__main__":
    main() 