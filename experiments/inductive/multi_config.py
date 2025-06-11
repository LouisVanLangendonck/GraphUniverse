"""
Clean configuration for multiple inductive experiments with parameter sweeps and random sampling.
Removes old parameters and focuses only on DC-SBM and DCCC-SBM methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
import numpy as np
import itertools
from experiments.inductive.config import InductiveExperimentConfig, PreTrainingConfig

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
    
    # For non-continuous parameters, provide a list of possible values
    discrete_values: Optional[List[Any]] = None
    
    def __post_init__(self):
        """Validate parameter range configuration."""
        if self.is_sweep and self.step is None and self.discrete_values is None:
            raise ValueError("Sweep parameters must have either a step size or discrete_values")
        
        if not self.is_sweep and self.n_samples is None:
            self.n_samples = 1
        
        if self.distribution == "normal" and (self.mean is None or self.std is None):
            self.mean = (self.min_val + self.max_val) / 2
            self.std = (self.max_val - self.min_val) / 6
    
    def get_sweep_values(self) -> List[Any]:
        """Get systematic sweep values."""
        if not self.is_sweep:
            raise ValueError("Cannot get sweep values for non-sweep parameter")
        
        # If discrete values are provided, use those
        if self.discrete_values is not None:
            return self.discrete_values
        
        # Otherwise use numeric range logic
        if isinstance(self.min_val, int) and isinstance(self.max_val, int) and isinstance(self.step, int):
            return list(range(int(self.min_val), int(self.max_val) + 1, int(self.step)))
        else:
            values = []
            current = self.min_val
            while current <= self.max_val + 1e-10:
                values.append(current)
                current += self.step
            return values
    
    def sample_random(self, n_samples: Optional[int] = None) -> List[Any]:
        """Get random samples from the range."""
        if self.is_sweep:
            raise ValueError("Cannot sample random values for sweep parameter")
        
        n = n_samples or self.n_samples
        
        # If discrete values are provided, sample from those
        if self.discrete_values is not None:
            return np.random.choice(self.discrete_values, n).tolist()
        
        # Otherwise use numeric sampling logic
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
    """Configuration for clean multi-experiment sweeps."""
    
    def __init__(
        self,
        output_dir: str = "multi_results",
        experiment_name: str = "multi_sweep",
        
        # Base configuration
        base_config: InductiveExperimentConfig = None,
        
        # Parameter sweeps
        sweep_parameters: Dict[str, ParameterRange] = None,
        
        # Random parameters
        random_parameters: Dict[str, ParameterRange] = None,
        
        # Execution parameters
        n_repetitions: int = 1,
        continue_on_failure: bool = True,
        save_individual_results: bool = True,
        
        # Resource management
        max_concurrent_families: int = 1,  # Number of graph families to generate concurrently
        reuse_families: bool = True,  # Reuse graph families across experiments
        
        # Random seed management
        base_seed: int = 42,
        
        ### Model management
        # Gnn Models
        gnn_models: List[str] = None,
        
        # Transformer Models
        transformer_models: List[str] = None,  # List of transformer models to run
        run_transformers: bool = False,  # Whether to run transformer models
        transformer_params: Dict[str, Any] = None  # Additional transformer parameters
    ):
        """Initialize multi-experiment configuration."""
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.base_config = base_config or InductiveExperimentConfig()
        self.sweep_parameters = sweep_parameters or {}
        self.random_parameters = random_parameters or {}
        self.n_repetitions = n_repetitions
        self.continue_on_failure = continue_on_failure
        self.save_individual_results = save_individual_results
        self.max_concurrent_families = max_concurrent_families
        self.reuse_families = reuse_families
        self.base_seed = base_seed
        
        # Model configuration
        self.gnn_models = gnn_models or ['gcn', 'sage', 'gat', 'fagcn', 'gin']
        self.transformer_models = transformer_models or ['graphormer']
        self.run_transformers = run_transformers
        self.transformer_params = transformer_params or {
            'transformer_num_heads': 8,
            'transformer_max_nodes': 200,
            'transformer_max_path_length': 10,
            'transformer_precompute_encodings': True,
            'transformer_cache_encodings': True,
            'local_gnn_type': 'gcn',
            'global_model_type': 'transformer',
            'transformer_prenorm': True
        }
        
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
            'save_individual_results': self.save_individual_results,
            'max_concurrent_families': self.max_concurrent_families,
            'reuse_families': self.reuse_families,
            'base_seed': self.base_seed,
            'gnn_models': self.gnn_models,
            'transformer_models': self.transformer_models,
            'run_transformers': self.run_transformers,
            'transformer_params': self.transformer_params
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
            continue_on_failure=config_dict['continue_on_failure'],
            save_individual_results=config_dict['save_individual_results'],
            max_concurrent_families=config_dict['max_concurrent_families'],
            reuse_families=config_dict['reuse_families'],
            base_seed=config_dict['base_seed'],
            gnn_models=config_dict['gnn_models'],
            transformer_models=config_dict['transformer_models'],
            run_transformers=config_dict['run_transformers'],
            transformer_params=config_dict['transformer_params']
        )

@dataclass
class SSLMultiExperimentConfig:
    """Configuration for multi-experiment SSL runs."""
    
    def __init__(
        self,
        output_dir: str = "multi_ssl_experiments",
        experiment_name: str = "ssl_sweep",
        
        # Base configuration
        base_config: PreTrainingConfig = None,
        
        # Parameter sweeps
        sweep_parameters: Dict[str, ParameterRange] = None,

        # Model sweep parameters
        model_sweep_parameters: Dict[str, ParameterRange] = None,
        
        # Random parameters
        random_parameters: Dict[str, ParameterRange] = None,
        
        # Execution parameters
        n_repetitions: int = 1,
        continue_on_failure: bool = True,
        save_individual_results: bool = True,
        
        # Resource management
        max_concurrent_families: int = 1,  # Number of graph families to generate concurrently
        reuse_families: bool = True,  # Reuse graph families across experiments
        
        # Random seed management
        base_seed: int = 42,
        
        # Model management
        gnn_models: List[str] = None,
        
        # Transformer configuration
        transformer_models: List[str] = None,  # List of transformer models to run
        run_transformers: bool = False,  # Whether to run transformer models
        transformer_params: Dict[str, Any] = None,  # Additional transformer parameters
        skip_gnn: bool = False,  # Whether to skip running GNN models
        patience: int = 50  # Added patience parameter
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.base_config = base_config
        self.sweep_parameters = sweep_parameters or {}
        self.model_sweep_parameters = model_sweep_parameters or {}
        self.random_parameters = random_parameters or {}
        self.n_repetitions = n_repetitions
        self.continue_on_failure = continue_on_failure
        self.save_individual_results = save_individual_results
        self.max_concurrent_families = max_concurrent_families
        self.reuse_families = reuse_families
        self.base_seed = base_seed
        self.gnn_models = gnn_models or ['sage', 'gcn']
        self.transformer_models = transformer_models or []
        self.run_transformers = run_transformers
        self.transformer_params = transformer_params or {
            'transformer_num_heads': 8,
            'transformer_max_nodes': 200,
            'transformer_max_path_length': 10,
            'transformer_precompute_encodings': True,
            'transformer_cache_encodings': True,
            'local_gnn_type': 'gcn',
            'global_model_type': 'transformer',
            'transformer_prenorm': True
        }
        self.skip_gnn = skip_gnn
        self.patience = patience
    
    def get_family_configurations(self) -> List[Dict[str, Any]]:
        """Get all unique graph family configurations."""
        # Get all combinations of sweep parameters
        sweep_values = {}
        for param, param_range in self.sweep_parameters.items():
            if param_range.is_sweep:
                sweep_values[param] = param_range.get_sweep_values()
            else:
                sweep_values[param] = [param_range.min_val]
        
        # Generate all combinations
        param_names = list(sweep_values.keys())
        param_values = list(sweep_values.values())
        combinations = list(itertools.product(*param_values))
        
        family_configs = []
        for i, combo in enumerate(combinations):
            # Start with base configuration values
            family_config = {
                'family_id': f"family_{self.experiment_name}_{i:03d}",
                'n_graphs': self.base_config.n_graphs,
                'n_extra_graphs': self.base_config.n_extra_graphs_for_finetuning,
                'universe_K': self.base_config.universe_K,
                'universe_feature_dim': self.base_config.universe_feature_dim,
                'use_dccc_sbm': self.base_config.use_dccc_sbm,
                'degree_distribution': self.base_config.degree_distribution,
                'min_n_nodes': self.base_config.min_n_nodes,
                'max_n_nodes': self.base_config.max_n_nodes,
                'min_communities': self.base_config.min_communities,
                'max_communities': self.base_config.max_communities
            }
            
            # Add sweep parameter values
            family_config.update(dict(zip(param_names, combo)))
            
            family_configs.append(family_config)
        
        return family_configs
    
    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """Get all model configurations."""
        # For SSL, we'll use the base config's model parameters
        if not self.base_config:
            return []
        
        # Get all combinations of model sweep parameters
        model_sweep_values = {}
        for param, param_range in self.model_sweep_parameters.items():
            if param_range.is_sweep:
                model_sweep_values[param] = param_range.get_sweep_values()
            else:
                model_sweep_values[param] = [param_range.min_val]
        print(model_sweep_values)
        
        # Generate all combinations
        param_names = list(model_sweep_values.keys())
        param_values = list(model_sweep_values.values())
        combinations = list(itertools.product(*param_values))

        print(combinations)

        model_configs = []
        
        # Add GNN models only if not skipping GNNs and gnn_models is not empty
        if not self.skip_gnn and self.gnn_models:
            for gnn_type in self.gnn_models:
                for combo in combinations:
                    model_config = {
                        'gnn_type': gnn_type,
                        'pretraining_task': self.base_config.pretraining_task,
                        'hidden_dim': self.base_config.hidden_dim,
                        'num_layers': self.base_config.num_layers,
                        'run_transformers': False,
                        'model_type': 'gnn',
                        'negative_sampling_ratio': self.base_config.negative_sampling_ratio,
                        'link_pred_loss': self.base_config.link_pred_loss,
                        'dgi_corruption_type': self.base_config.dgi_corruption_type,
                        'dgi_noise_std': self.base_config.dgi_noise_std,
                        'dgi_perturb_rate': self.base_config.dgi_perturb_rate,
                        'dgi_corruption_rate': self.base_config.dgi_corruption_rate,
                        'graphmae_mask_rate': self.base_config.graphmae_mask_rate,
                        'graphmae_replace_rate': self.base_config.graphmae_replace_rate,
                        'graphmae_gamma': self.base_config.graphmae_gamma,
                        'graphmae_decoder_type': self.base_config.graphmae_decoder_type,
                        'graphmae_decoder_gnn_type': self.base_config.graphmae_decoder_gnn_type,
                    }
                    model_config.update(dict(zip(param_names, combo)))
                    model_configs.append(model_config)
        
        # Add transformer models
        if self.run_transformers:
            for transformer_type in self.transformer_models:
                for combo in combinations:
                    model_config = {
                    'transformer_type': transformer_type,
                    'pretraining_task': self.base_config.pretraining_task,
                    'hidden_dim': self.base_config.hidden_dim,
                    'num_layers': self.base_config.num_layers,
                    'run_transformers': True,
                    'model_type': 'transformer',
                    'negative_sampling_ratio': self.base_config.negative_sampling_ratio,
                    'link_pred_loss': self.base_config.link_pred_loss,
                    'dgi_corruption_type': self.base_config.dgi_corruption_type,
                    'dgi_noise_std': self.base_config.dgi_noise_std,
                    'dgi_perturb_rate': self.base_config.dgi_perturb_rate,
                    'dgi_corruption_rate': self.base_config.dgi_corruption_rate,
                    'graphmae_mask_rate': self.base_config.graphmae_mask_rate,
                    'graphmae_replace_rate': self.base_config.graphmae_replace_rate,
                    'graphmae_gamma': self.base_config.graphmae_gamma,
                    'graphmae_decoder_type': self.base_config.graphmae_decoder_type,
                    'graphmae_decoder_gnn_type': self.base_config.graphmae_decoder_gnn_type,
                    }
                    model_config.update(dict(zip(param_names, combo)))
                    model_configs.append(model_config)
        
        return model_configs
    
    def get_total_experiments(self) -> int:
        """Get total number of experiments."""
        n_families = len(self.get_family_configurations())
        n_models = len(self.get_model_configurations())
        return n_families * n_models * self.n_repetitions
