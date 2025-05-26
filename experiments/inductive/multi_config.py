"""
Configuration for multiple inductive experiment runs with parameter sweeps and random sampling.

This module defines configuration for running many inductive experiments with
systematic parameter variations and random sampling from specified ranges.
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
    """Class to define parameter ranges for sampling or sweeping."""
    
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
            self.n_samples = 1  # Default to single sample
        
        if self.distribution == "normal" and (self.mean is None or self.std is None):
            # Default to mean at center of range, std as 1/6 of range
            self.mean = (self.min_val + self.max_val) / 2
            self.std = (self.max_val - self.min_val) / 6
    
    def get_sweep_values(self) -> List[float]:
        """Get systematic sweep values."""
        if not self.is_sweep:
            raise ValueError("Cannot get sweep values for non-sweep parameter")
        
        if isinstance(self.min_val, int) and isinstance(self.max_val, int) and isinstance(self.step, int):
            # Integer range
            return list(range(int(self.min_val), int(self.max_val) + 1, int(self.step)))
        else:
            # Float range
            values = []
            current = self.min_val
            while current <= self.max_val + 1e-10:  # Small epsilon for float comparison
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
            # Clip to range
            return np.clip(samples, self.min_val, self.max_val).tolist()
        
        elif self.distribution == "log_uniform":
            log_min = np.log(max(self.min_val, 1e-10))
            log_max = np.log(self.max_val)
            log_samples = np.random.uniform(log_min, log_max, n)
            return np.exp(log_samples).tolist()
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class MultiInductiveExperimentConfig:
    """Configuration for running multiple inductive experiments with parameter variations."""
    
    # Base configuration that applies to all runs
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
    save_individual_configs: bool = True
    save_individual_results: bool = True
    
    # Experiment control
    max_concurrent_runs: int = 1  # For potential parallel execution
    continue_on_failure: bool = True
    
    # Result aggregation
    aggregate_results: bool = True
    create_summary_plots: bool = True
    
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
            return [{}]  # Single empty combination if no sweep parameters
        
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
        """
        Create a configuration for a single run.
        
        Args:
            sweep_params: Systematic parameter values
            random_params: Random parameter values
            run_id: Unique identifier for this run
            
        Returns:
            InductiveExperimentConfig for this specific run
        """
        # Start with base config
        config_dict = self.base_config.to_dict()
        
        # Override with sweep parameters
        config_dict.update(sweep_params)
        
        # Override with random parameters
        config_dict.update(random_params)
        
        # Set unique output directory
        config_dict['output_dir'] = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_run_{run_id:04d}"
        )
        
        # Create config object
        return InductiveExperimentConfig.from_dict(config_dict)
    
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
    def load(cls, filepath: str) -> 'MultiInductiveExperimentConfig':
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


def create_default_multi_config() -> MultiInductiveExperimentConfig:
    """Create a default multi-experiment configuration with example parameter ranges."""
    
    # Base configuration
    base_config = InductiveExperimentConfig(
        # Core experiment settings
        n_graphs=15,
        min_n_nodes=80,
        max_n_nodes=120,
        min_communities=5,
        max_communities=5,
        
        # Universe settings
        universe_K=5,
        universe_feature_dim=32,
        universe_edge_density=0.1,
        universe_homophily=0.5,
        
        # Graph generation method - can be changed
        use_dccc_sbm=False,  # Set to True to use DCCC-SBM
        degree_distribution="standard",
        
        # Training settings
        epochs=150,
        patience=30,
        optimize_hyperparams=False,
        
        # Tasks
        tasks=['community'],
        
        # Models
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=True,
        
        # Output
        require_consistency_check=False,
        collect_family_stats=True
    )
    
    # Sweep parameters (systematic variation)
    sweep_parameters = {
        'universe_homophily': ParameterRange(
            min_val=0.0,
            max_val=1.0,
            step=0.1,
            is_sweep=True
        ),
        'universe_edge_density': ParameterRange(
            min_val=0.01,
            max_val=0.021,
            step=0.011,
            is_sweep=True
        )
    }
    
    # Random parameters (sampled for each run)
    random_parameters = {
        # Standard graph parameters
        'homophily_range': ParameterRange(
            min_val=0.0,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        'density_range': ParameterRange(
            min_val=0.0,
            max_val=0.05,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.1,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        'edge_noise': ParameterRange(
            min_val=0.0,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        
        # Feature generation parameters
        'cluster_count_factor': ParameterRange(
            min_val=0.3,
            max_val=2.0,
            distribution="uniform",
            is_sweep=False
        ),
        'center_variance': ParameterRange(
            min_val=0.1,
            max_val=2.0,
            distribution="log_uniform",
            is_sweep=False
        ),
        'cluster_variance': ParameterRange(
            min_val=0.05,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'assignment_skewness': ParameterRange(
            min_val=0.0,
            max_val=0.8,
            distribution="uniform",
            is_sweep=False
        ),
        'community_exclusivity': ParameterRange(
            min_val=0.5,
            max_val=1.0,
            distribution="uniform",
            is_sweep=False
        ),
        
        # Universe parameters
        'universe_randomness_factor': ParameterRange(
            min_val=0.0,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        'max_mean_community_deviation': ParameterRange(
            min_val=0.05,
            max_val=0.3,
            distribution="uniform",
            is_sweep=False
        ),
        'max_max_community_deviation': ParameterRange(
            min_val=0.1,
            max_val=0.5,
            distribution="uniform",
            is_sweep=False
        ),
        
        # DCCC-SBM parameters (only used if use_dccc_sbm=True)
        'community_imbalance_range_0': ParameterRange(  # Will be converted to tuple
            min_val=0.0,
            max_val=0.6,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_separation_range_0': ParameterRange(  # Will be converted to tuple
            min_val=0.1,
            max_val=0.9,
            distribution="uniform",
            is_sweep=False
        )
    }
    
    return MultiInductiveExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=2,
        experiment_name="homophily_density_sweep",
        output_dir="multi_inductive_results",
        continue_on_failure=True,
        aggregate_results=True,
        create_summary_plots=True
    )


def create_dccc_multi_config(degree_distribution: str = "power_law") -> MultiInductiveExperimentConfig:
    """
    Create a multi-experiment configuration focused on DCCC-SBM with specific degree distribution.
    
    Args:
        degree_distribution: Type of degree distribution ("power_law", "exponential", "uniform")
    
    Returns:
        MultiInductiveExperimentConfig configured for DCCC-SBM experiments
    """
    
    # Base configuration with DCCC-SBM enabled
    base_config = InductiveExperimentConfig(
        # Core experiment settings
        n_graphs=12,
        min_n_nodes=80,
        max_n_nodes=100,
        min_communities=5,
        max_communities=5,
        
        # Universe settings
        universe_K=5,
        universe_feature_dim=32,
        universe_edge_density=0.1,
        universe_homophily=0.6,
        
        # DCCC-SBM settings
        use_dccc_sbm=True,
        degree_distribution=degree_distribution,
        
        # Training settings
        epochs=120,
        patience=25,
        optimize_hyperparams=False,
        
        # Tasks
        tasks=['community'],
        
        # Models
        gnn_types=['gcn', 'sage'],
        run_gnn=True,
        run_mlp=True,
        run_rf=False,  # Skip for faster experiments
        
        # Output
        require_consistency_check=False,
        collect_family_stats=True
    )
    
    # Sweep community imbalance and degree separation
    sweep_parameters = {
        'community_imbalance_range_0': ParameterRange(
            min_val=0.0,
            max_val=0.6,
            step=0.2,
            is_sweep=True
        ),
        'degree_separation_range_0': ParameterRange(
            min_val=0.2,
            max_val=0.8,
            step=0.3,
            is_sweep=True
        )
    }
    
    # Random parameters based on degree distribution type
    random_parameters = {
        # Standard parameters
        'universe_homophily': ParameterRange(
            min_val=0.3,
            max_val=0.9,
            distribution="uniform",
            is_sweep=False
        ),
        'degree_heterogeneity': ParameterRange(
            min_val=0.3,
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
    
    # Add distribution-specific parameters
    if degree_distribution == "power_law":
        random_parameters.update({
            'power_law_exponent_range_0': ParameterRange(
                min_val=2.1,
                max_val=3.5,
                distribution="uniform",
                is_sweep=False
            ),
            'dccc_target_avg_degree_range_0': ParameterRange(
                min_val=3.0,
                max_val=8.0,
                distribution="uniform",
                is_sweep=False
            )
        })
    
    elif degree_distribution == "exponential":
        random_parameters.update({
            'exponential_rate_range_0': ParameterRange(
                min_val=0.2,
                max_val=1.2,
                distribution="uniform",
                is_sweep=False
            ),
            'dccc_target_avg_degree_range_0': ParameterRange(
                min_val=2.0,
                max_val=6.0,
                distribution="uniform",
                is_sweep=False
            )
        })
    
    elif degree_distribution == "uniform":
        random_parameters.update({
            'uniform_min_factor_range_0': ParameterRange(
                min_val=0.2,
                max_val=0.8,
                distribution="uniform",
                is_sweep=False
            ),
            'uniform_max_factor_range_0': ParameterRange(
                min_val=1.2,
                max_val=2.5,
                distribution="uniform",
                is_sweep=False
            ),
            'dccc_target_avg_degree_range_0': ParameterRange(
                min_val=3.0,
                max_val=10.0,
                distribution="uniform",
                is_sweep=False
            )
        })
    
    return MultiInductiveExperimentConfig(
        base_config=base_config,
        sweep_parameters=sweep_parameters,
        random_parameters=random_parameters,
        n_repetitions=3,
        experiment_name=f"dccc_{degree_distribution}_study",
        output_dir=f"multi_inductive_results/dccc_{degree_distribution}",
        continue_on_failure=True,
        aggregate_results=True,
        create_summary_plots=True
    )