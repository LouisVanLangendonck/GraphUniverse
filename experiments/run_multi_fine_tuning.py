"""
Script to run fine-tuning experiments for all models in a given SSL sweep directory.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import clean implementations
from experiments.inductive.config import InductiveExperimentConfig
from experiments.inductive.experiment import run_inductive_experiment


def find_model_dirs(sweep_dir: str) -> List[Dict[str, str]]:
    """Find all model directories in the sweep directory."""
    valid_model_dirs = []
    valid_model_ids = []
    valid_model_config = []
    valid_model_metadata = []
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(sweep_dir, "exp_*"))
    print(f"Found {len(exp_dirs)} experiments in {sweep_dir}")
    
    for exp_dir in exp_dirs:
        # Find all model directories in this experiment (no matching pattern)
        model_subdirs = glob.glob(os.path.join(exp_dir, "*"))
        print(model_subdirs)
        
        for model_dir in model_subdirs:
            # Check if this is a valid model directory by looking for config.json
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
                    model_metadata = json.load(f)
                
                valid_model_config.append(model_config)
                valid_model_metadata.append(model_metadata)

                valid_model_dirs.append(model_dir)
                exp_id = os.path.basename(exp_dir)
                valid_model_ids.append(f"{exp_id}/{os.path.basename(model_dir)}")
                
                
    return valid_model_dirs, valid_model_ids, valid_model_config, valid_model_metadata


def create_fine_tuning_config(
    model_id: str,
    model_config: Dict[str, Any],
    model_metadata: Dict[str, Any],
    sweep_dir: str,
    output_dir: str,
    only_pretrained_experiments: bool,
    max_train_graphs_for_finetuning: int,
    calculate_silhouette_score: bool,
    hyperparameter_optimization_trials: int
) -> InductiveExperimentConfig:
    """Create fine-tuning configuration for a specific model."""
    # Create base config
    config = InductiveExperimentConfig(
        # === EXPERIMENT SETUP ===
        output_dir=os.path.join(output_dir, f"finetune_{model_id}"),
        seed=42,
        device_id=0,
        force_cpu=False,
        
        # === SSL FINE-TUNING SETUP ===
        use_pretrained=True,
        pretrained_model_dir=sweep_dir,
        pretrained_model_id=model_id,
        graph_family_id=model_metadata['family_id'],
        graph_family_dir=os.path.join(sweep_dir, "graph_families"),
        auto_load_family=True,
        freeze_encoder=False,
        only_pretrained_experiments=only_pretrained_experiments,
        max_train_graphs_for_finetuning=max_train_graphs_for_finetuning,
        calculate_silhouette_score=calculate_silhouette_score,
        
        # === TASKS ===
        tasks=['community'],
        
        # === ANALYSIS ===
        collect_signal_metrics=True,
        require_consistency_check=False,
        optimize_hyperparams=True,
        n_trials=hyperparameter_optimization_trials
    )
    
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run fine-tuning experiments for all models in an SSL sweep')
    
    # === EXPERIMENT SETUP ===
    parser.add_argument('--ssl_sweep_dir', type=str, required=True,
                        help='Directory containing the SSL sweep experiments')
    parser.add_argument('--output_dir', type=str, default='multi_finetune_results',
                        help='Directory to save fine-tuning results')
    parser.add_argument('--only_pretrained_experiments', action='store_true', default=True,
                        help='Only run fine-tuning experiments, no from-scratch baselines')
    parser.add_argument('--max_train_graphs_for_finetuning', type=int, default=3,
                        help='Maximum number of training graphs for fine-tuning')
    parser.add_argument('--calculate_silhouette_score', action='store_true', default=True,
                        help='Calculate silhouette score of communities of pre-trained models')
    parser.add_argument('--hyperparameter_optimization_trials', type=int, default=10,
                        help='Number of trials for hyperparameter optimization of from-scratch, different architecture models')
    
    return parser.parse_args()


def main():
    """Main function to run fine-tuning experiments."""
    print("MULTI-MODEL FINE-TUNING EXPERIMENTS")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"finetune_sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    print(f"\nScanning sweep directory: {args.ssl_sweep_dir}")
    model_dirs, model_ids, model_config, model_metadata = find_model_dirs(args.ssl_sweep_dir)
    print(f"Found {len(model_dirs)} models to fine-tune")
    
    # Run fine-tuning for each model
    all_results = {}
    successful_runs = 0
    
    for i, model_info in enumerate(model_dirs, 1):
        print(f"\nProcessing model {i}/{len(model_dirs)}")
        print(f"Model ID: {model_ids[i-1]}")
        
        try:
            # Create fine-tuning config
            config = create_fine_tuning_config(
                model_ids[i-1],
                model_config[i-1],
                model_metadata[i-1],
                args.ssl_sweep_dir,
                output_dir,
                args.only_pretrained_experiments,
                args.max_train_graphs_for_finetuning,
                args.calculate_silhouette_score,
                args.hyperparameter_optimization_trials
            )
            
            # Run fine-tuning experiment
            results = run_inductive_experiment(config)
            
            successful_runs += 1
            print(f"✓ Fine-tuning completed successfully")
            
        except Exception as e:
            print(f"✗ Fine-tuning failed: {str(e)}")
            continue
    
    # Save summary
    summary = {
        'total_models': len(model_dirs),
        'successful_runs': successful_runs,
        'success_rate': successful_runs / len(model_dirs) if model_dirs else 0,
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFine-tuning experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Successful runs: {successful_runs}/{len(model_dirs)}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 