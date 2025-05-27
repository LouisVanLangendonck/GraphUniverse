"""
Clean configuration for multiple inductive experiments with parameter sweeps and random sampling.
Removes old parameters and focuses only on DC-SBM and DCCC-SBM methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os
import numpy as np
import itertools
from experiments.inductive.config import InductiveExperimentConfig


@dataclass
class ParameterRange:
    """Defines parameter ranges for sweeping or random sampling."""
    
    # Range definition
    min_val: float
    max_val: float
    step: Optional[float] = None  # For systematic sweeps
    n_samples: Optional[int] = None  # For random sampling
    
    # Distribution type for random sampling
    distribution: str = "uniform"  # "uniform", "normal", "log_uniform"
    
    # Additional parameters for specific distributions
    mean: Optional[float] = None  # For normal distribution
    std: Optional[float] = None   # For normal distribution
    
    # Whether this is a sweep parameter (systematic) or random parameter
    is_sweep: bool = False
    
    def __post_init__(self):
        """Validate parameter range configuration."""
        if self.is_sweep and self.step is None:
            raise ValueError("Sweep parameters must have a step size")
        
        if not self.is_sweep and self.n_samples is None:
            self.n_samples = 1
        
        if self.distribution == "normal" and (self.mean is None or self.std is None):
            self.mean = (self.min_val + self.max_val) / 2
            self.std = (self.max_val - self.min_val) / 6
    
    def get_sweep_values(self) -> List[float]:
        """Get systematic sweep values."""
        if not self.is_sweep:
            raise ValueError("Cannot get sweep values for non-sweep parameter")
        
        if isinstance(self.min_val, int) and isinstance(self.max_val, int) and isinstance(self.step, int):
            return list(range(int(self.min_val), int(self.max_val) + 1, int(self.step)))
        else:
            values = []
            current = self.min_val
            while current <= self.max_val + 1e-10:
                values.append(current)
                current += self.step
            return values
    
    def sample_random(self, n_samples: Optional[int] = None) -> List[float]:
        """Get random samples from the range."""
        if self.is_sweep:
            raise ValueError("Cannot sample random values for sweep parameter")
        
        n = n_samples or self.n_samples
        
        if self.distribution == "uniform":
            return np.random.uniform(self.min_val, self.max_val, n).tolist()
        
        elif self.distribution == "normal":
            samples = np.random.normal(self.mean, self.std, n)
            return np.clip(samples, self.min_val, self.max_val).tolist()
        
        elif self.distribution == "log_uniform":
            log_min = np.log(max(self.min_val, 1e-10))
            log_max = np.log(self.max_val)
            log_samples = np.random.uniform(log_min, log_max, n)
            return np.exp(log_samples).tolist()
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class CleanMultiExperimentConfig:
    """Configuration for running multiple clean inductive experiments with parameter variations."""
    
    # Base configuration for all runs
    base_config: InductiveExperimentConfig = field(default_factory=InductiveExperimentConfig)
    
    # Parameters that are swept systematically
    sweep_parameters: Dict[str, ParameterRange] = field(default_factory=dict)
    
    # Parameters that are randomly sampled for each run
    random_parameters: Dict[str, ParameterRange] = field(default_factory=dict)
    
    # Number of repetitions for each parameter combination
    n_repetitions: int = 1
    
    # Output configuration
    output_dir: str = "multi_inductive_results"
    experiment_name: str = "multi_inductive"
    save_individual_configs: bool = False
    save_individual_results: bool = False
    
    # Experiment control
    max_concurrent_runs: int = 1
    continue_on_failure: bool = True
    
    # Result aggregation
    aggregate_results: bool = True
    create_summary_plots: bool = False
    
    def __post_init__(self):
        """Validate multi-experiment configuration."""
        # Ensure all sweep parameters have is_sweep=True
        for param_name, param_range in self.sweep_parameters.items():
            param_range.is_sweep = True
        
        # Ensure all random parameters have is_sweep=False
        for param_name, param_range in self.random_parameters.items():
            param_range.is_sweep = False
    
    def get_parameter_combinations(self) -> List[Dict[str, float]]:
        """Get all parameter combinations for systematic sweeps."""
        if not self.sweep_parameters:
            return [{}]
        
        # Get all sweep values
        sweep_values = {}
        for param_name, param_range in self.sweep_parameters.items():
            sweep_values[param_name] = param_range.get_sweep_values()
        
        # Generate all combinations
        param_names = list(sweep_values.keys())
        param_value_lists = list(sweep_values.values())
        
        combinations = []
        for combo in itertools.product(*param_value_lists):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def sample_random_parameters(self) -> Dict[str, float]:
        """Sample random values for all random parameters."""
        random_values = {}
        for param_name, param_range in self.random_parameters.items():
            values = param_range.sample_random(1)
            random_values[param_name] = values[0]
        
        return random_values
    
    def create_run_config(
        self,
        sweep_params: Dict[str, float],
        random_params: Dict[str, float],
        run_id: int
    ) -> InductiveExperimentConfig:
        """Create a configuration for a single run."""
        # Start with base config
        config_dict = self.base_config.to_dict()
        
        # Override with sweep parameters
        config_dict.update(sweep_params)
        
        # Override with random parameters
        config_dict.update(random_params)
        
        # Process tuple parameters (ranges)
        config_dict = self._process_tuple_parameters(config_dict)
        
        # Set unique output directory
        config_dict['output_dir'] = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_run_{run_id:04d}"
        )
        
        # Create config object
        return InductiveExperimentConfig.from_dict(config_dict)
    
    def _process_tuple_parameters(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process parameters that need to be converted to tuples (like ranges)."""
        processed = config_dict.copy()
        
        # Convert range parameters from single values to tuples
        range_params = {
            'homophily_range': 'homophily_range_width',
            'density_range': 'density_range_width',
            'community_imbalance_range': 'community_imbalance_range_width',
            'degree_separation_range': 'degree_separation_range_width',
            'power_law_exponent_range': 'power_law_exponent_range_width',
            'exponential_rate_range': 'exponential_rate_range_width',
            'uniform_min_factor_range': 'uniform_min_factor_range_width',
            'uniform_max_factor_range': 'uniform_max_factor_range_width',
            'dccc_target_avg_degree_range': 'dccc_target_avg_degree_range_width'
        }
        
        for range_param, width_param in range_params.items():
            if width_param in processed:
                width = processed.pop(width_param)
                if range_param in processed:
                    center = processed[range_param]
                    if isinstance(center, (list, tuple)):
                        # Already a range, don't modify
                        continue
                    else:
                        # Convert single value to range
                        processed[range_param] = (max(0, center - width/2), center + width/2)
        
        return processed
    
    def get_total_runs(self) -> int:
        """Calculate total number of experiment runs."""
        n_combinations = len(self.get_parameter_combinations())
        return n_combinations * self.n_repetitions
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to serializable format
        config_dict = {
            'base_config': self.base_config.to_dict(),
            'sweep_parameters': {
                name: {
                    'min_val': pr.min_val,
                    'max_val': pr.max_val,
                    'step': pr.step,
                    'is_sweep': pr.is_sweep,
                    'distribution': pr.distribution
                }
                for name, pr in self.sweep_parameters.items()
            },
            'random_parameters': {
                name: {
                    'min_val': pr.min_val,
                    'max_val': pr.max_val,
                    'n_samples': pr.n_samples,
                    'is_sweep': pr.is_sweep,
                    'distribution': pr.distribution,
                    'mean': pr.mean,
                    'std': pr.std
                }
                for name, pr in self.random_parameters.items()
            },
            'n_repetitions': self.n_repetitions,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'save_individual_configs': self.save_individual_configs,
            'save_individual_results': self.save_individual_results,
            'max_concurrent_runs': self.max_concurrent_runs,
            'continue_on_failure': self.continue_on_failure,
            'aggregate_results': self.aggregate_results,
            'create_summary_plots': self.create_summary_plots
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CleanMultiExperimentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct base config
        base_config = InductiveExperimentConfig.from_dict(config_dict['base_config'])
        
        # Reconstruct parameter ranges
        sweep_parameters = {}
        for name, pr_dict in config_dict['sweep_parameters'].items():
            sweep_parameters[name] = ParameterRange(**pr_dict)
        
        random_parameters = {}
        for name, pr_dict in config_dict['random_parameters'].items():
            random_parameters[name] = ParameterRange(**pr_dict)
        
        return cls(
            base_config=base_config,
            sweep_parameters=sweep_parameters,
            random_parameters=random_parameters,
            n_repetitions=config_dict['n_repetitions'],
            output_dir=config_dict['output_dir'],
            experiment_name=config_dict['experiment_name'],
            save_individual_configs=config_dict['save_individual_configs'],
            save_individual_results=config_dict['save_individual_results'],
            max_concurrent_runs=config_dict['max_concurrent_runs'],
            continue_on_failure=config_dict['continue_on_failure'],
            aggregate_results=config_dict['aggregate_results'],
            create_summary_plots=config_dict['create_summary_plots']
        )


# =============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS
# =============================================================================

def create_homophily_density_sweep() -> CleanMultiExperimentConfig:
    """Create configuration for homophily vs density sweep."""
    
    base_config = InductiveExperimentConfig(
        n_graphs=12,
        min_n_nodes=80,
        max_n_nodes=100,
        min_communities=3,
        max_communities=5,
        universe_K=5,
        universe_feature_dim=32,
        use_dccc_sbm=False,  # Standard DC-SBM
        tasks=['community'],
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=True,
        epochs=100,
        patience=20,
        collect_signal_metrics=True,
        require_consistency_check=False
    )
    
    # Sweep homophily and density systematically
    sweep_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.1,
            max_val=0.9,
            step=0.2,
            is_sweep=True
        ),
        'universe_edge_density': ParameterRange(
            min_val=0.05,
            max_val=0.15,
            step=0.05,
            is_sweep=True
        )
    }
    
    # Randomize other parameters
    random_parameters = {
        'homophily_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        'density_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.05,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.1,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.2,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    return CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=2,
        experiment_name="homophily_density_sweep",
        output_dir="results/sweep_experiments",
        continue_on_failure=True
    )


def create_dccc_method_comparison() -> CleanMultiExperimentConfig:
    """Create configuration for comparing DCCC-SBM degree distributions."""
    
    base_config = InductiveExperimentConfig(
        n_graphs=10,
        min_n_nodes=60,
        max_n_nodes=80,
        min_communities=3,
        max_communities=4,
        universe_K=4,
        universe_feature_dim=16,
        use_dccc_sbm=True,  # DCCC-SBM
        tasks=['community'],
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=False,
        epochs=80,
        patience=15,
        collect_signal_metrics=True,
        require_consistency_check=False
    )
    
    # Sweep degree distributions and community imbalance
    sweep_parameters = {
        'degree_distribution': ParameterRange(
            min_val=0,  # Will be mapped to distributions
            max_val=2,
            step=1,
            is_sweep=True
        ),
        'community_imbalance_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.6,
            step=0.2,
            is_sweep=True
        )
    }
    
    # Randomize DCCC parameters
    random_parameters = {
        'degree_separation_range_width': ParameterRange(
            min_val=0.2,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'power_law_exponent_range_width': ParameterRange(
            min_val=0.5,
            max_val=1.5,
            distribution="uniform",
            is_sweep=False
        ),
        'universe_homophily': ParameterRange(
            min_val=0.4,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.15,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    return CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=3,
        experiment_name="dccc_method_comparison",
        output_dir="results/dccc_comparison",
        continue_on_failure=True
    )


def create_large_scale_benchmark() -> CleanMultiExperimentConfig:
    """Create configuration for large-scale benchmarking."""
    
    base_config = InductiveExperimentConfig(
        n_graphs=15,
        universe_K=8,
        universe_feature_dim=32,
        tasks=['community'],
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=True,
        epochs=120,
        patience=25,
        collect_signal_metrics=True,
        require_consistency_check=True
    )
    
    # Sweep method and graph size
    sweep_parameters = {
        'use_dccc_sbm': ParameterRange(
            min_val=0,  # Will be mapped to boolean
            max_val=1,
            step=1,
            is_sweep=True
        ),
        'max_n_nodes': ParameterRange(
            min_val=80,
            max_val=120,
            step=20,
            is_sweep=True
        )
    }
    
    # Randomize many parameters
    random_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.3,
            max_val=0.9,
            distribution="uniform",
            is_sweep=False
        ),
        'universe_edge_density': ParameterRange(
            min_val=0.05,
            max_val=0.15,
            distribution="uniform",
            is_sweep=False
        ),
        'homophily_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        'density_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.05,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.2,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.25,
            distribution="uniform",
            is_sweep=False
        ),
        'community_imbalance_range_width': ParameterRange(
            min_val=0.0,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_separation_range_width': ParameterRange(
            min_val=0.1,
            max_val=0.7,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    return CleanMultiExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=2,
        experiment_name="large_scale_benchmark",
        output_dir="results/benchmark",
        continue_on_failure=True
    )