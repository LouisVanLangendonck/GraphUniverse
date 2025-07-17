"""
Test file to load saved inductive data and train a GraphSAGE model on the same splits.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from typing import Dict, Any, List
import json

# Import necessary modules
from experiments.models import GNNModel
from experiments.inductive.training import train_and_evaluate_inductive
from experiments.inductive.config import InductiveExperimentConfig


def load_saved_inductive_data(data_dir: str):
    """Load the saved inductive data from pickle files."""
    print(f"Loading data from: {data_dir}")
    
    # Load inductive data
    inductive_data_path = os.path.join(data_dir, "inductive_data.pkl")
    with open(inductive_data_path, 'rb') as f:
        inductive_data = pickle.load(f)
    
    # Load sheaf inductive data
    sheaf_data_path = os.path.join(data_dir, "sheaf_inductive_data.pkl")
    with open(sheaf_data_path, 'rb') as f:
        sheaf_inductive_data = pickle.load(f)
    
    # Load config
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    print(f"✓ Loaded inductive data with {len(inductive_data)} tasks")
    print(f"✓ Loaded sheaf data with {len(sheaf_inductive_data)} tasks")
    print(f"✓ Loaded config")
    
    return inductive_data, sheaf_inductive_data, config_dict


def create_dataloaders_from_saved_data(inductive_data: Dict, config_dict: Dict):
    """Create dataloaders from the saved inductive data."""
    dataloaders = {}
    
    for task_name, task_data in inductive_data.items():
        print(f"\nProcessing task: {task_name}")
        dataloaders[task_name] = {}
        
        # Handle new single split structure
        if 'split' in task_data:
            # New structure: task_data['split']['train']['graphs']
            split_data = task_data['split']
            dataloaders[task_name]['split'] = {}
            
            for split_name, split_info in split_data.items():
                print(f"  Split: {split_name} - {split_info['n_graphs']} graphs")
                
                # Create DataLoader from the saved data
                dataloader = DataLoader(
                    split_info['graphs'],
                    batch_size=split_info['batch_size'],
                    shuffle=(split_name == 'train')
                )
                
                dataloaders[task_name]['split'][split_name] = dataloader
        else:
            # Old fold-based structure (for backward compatibility)
            for fold_name, fold_data in task_data.items():
                if fold_name == 'metadata':
                    continue
                    
                print(f"  Fold: {fold_name}")
                dataloaders[task_name][fold_name] = {}
                
                for split_name, split_data in fold_data.items():
                    if split_name == 'metadata':
                        continue
                        
                    print(f"    Split: {split_name} - {split_data['n_graphs']} graphs")
                    
                    # Create DataLoader from the saved data
                    dataloader = DataLoader(
                        split_data['graphs'],
                        batch_size=split_data['batch_size'],
                        shuffle=(split_name == 'train')
                    )
                    
                    dataloaders[task_name][fold_name][split_name] = dataloader
    
    return dataloaders


def train_graphsage_on_saved_data(dataloaders: Dict, config_dict: Dict):
    """Train a GraphSAGE model on the loaded data."""
    print("\n" + "="*60)
    print("TRAINING GRAPHSAGE ON SAVED DATA")
    print("="*60)
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple config object for training
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Extract relevant config parameters
    training_config = SimpleConfig(
        hidden_dim=config_dict.get('hidden_dim', 64),
        num_layers=config_dict.get('num_layers', 3),
        dropout=config_dict.get('dropout', 0.1),
        learning_rate=config_dict.get('learning_rate', 0.001),
        weight_decay=config_dict.get('weight_decay', 1e-5),
        epochs=config_dict.get('epochs', 100),
        patience=config_dict.get('patience', 20),
        batch_size=config_dict.get('batch_size', 32),
        seed=config_dict.get('seed', 42),
        # Add missing attributes that the training function expects
        is_regression=config_dict.get('is_regression', {}),  # Dictionary mapping task names to booleans
        is_graph_level_tasks=config_dict.get('is_graph_level_tasks', {}),  # Dictionary mapping task names to booleans
        optimize_hyperparams=False,
        use_mixed_precision=False,
        device_id=0,
        force_cpu=False,
        regression_loss=config_dict.get('regression_loss', 'mse'),
        n_trials=config_dict.get('n_trials', 20),
        optimization_timeout=config_dict.get('optimization_timeout', 600)
    )
    
    results = {}
    
    # Train on each task
    for task_name, task_dataloaders in dataloaders.items():
        print(f"\n{'='*40}")
        print(f"TASK: {task_name.upper()}")
        print(f"{'='*40}")
        
        # Get sample batch to determine dimensions
        # Handle new single split structure
        if 'split' in task_dataloaders:
            # New structure: task_dataloaders['split']['train']
            sample_batch = next(iter(task_dataloaders['split']['train']))
        else:
            # Old fold-based structure (for backward compatibility)
            first_fold_name = list(task_dataloaders.keys())[0]
            sample_batch = next(iter(task_dataloaders[first_fold_name]['train']))
        input_dim = sample_batch.x.shape[1]
        
        # Determine if it's regression or classification
        is_regression = task_name.startswith('k_hop_community_counts') or task_name == 'triangle_count'
        is_graph_level_task = task_name == 'triangle_count'
        
        # Update config for this task (using dictionary structure)
        training_config.is_regression[task_name] = is_regression
        training_config.is_graph_level_tasks[task_name] = is_graph_level_task
        
        # Determine output dimension
        if is_regression:
            output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
        else:
            # Count unique classes
            all_labels = []
            # Handle new single split structure
            if 'split' in task_dataloaders:
                for split_name, dataloader in task_dataloaders['split'].items():
                    for batch in dataloader:
                        all_labels.extend(batch.y.cpu().numpy().flatten())
            else:
                # Old fold-based structure (for backward compatibility)
                for fold_name, fold_data in task_dataloaders.items():
                    for split_name, dataloader in fold_data.items():
                        for batch in dataloader:
                            all_labels.extend(batch.y.cpu().numpy().flatten())
            output_dim = len(set(all_labels))
        
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"Regression: {is_regression}, Graph-level: {is_graph_level_task}")
        
        # Create GraphSAGE model
        model = GNNModel(
            input_dim=input_dim,
            hidden_dim=training_config.hidden_dim,
            output_dim=output_dim,
            num_layers=training_config.num_layers,
            dropout=training_config.dropout,
            gnn_type='sage',  # Use GraphSAGE
            is_regression=is_regression,
            is_graph_level_task=is_graph_level_task
        )
        
        # Train and evaluate
        try:
            task_results = train_and_evaluate_inductive(
                model=model,
                model_name='sage_test',
                dataloaders=task_dataloaders,
                config=training_config,
                task=task_name,
                device=device,
                optimize_hyperparams=False,  # Don't optimize for this test
                experiment_name='test_load_inductive_data',
                run_id='test_run'
            )
            
            results[task_name] = task_results
            print(f"✓ Successfully trained GraphSAGE on {task_name}")
            
            # Print some results
            if 'repetition_test_metrics' in task_results:
                # New repetition-based structure
                test_metrics = task_results['repetition_test_metrics']
                for repetition_name, metrics in test_metrics.items():
                    print(f"  {repetition_name}: {metrics}")
            elif 'fold_test_metrics' in task_results:
                # Old fold-based structure (for backward compatibility)
                test_metrics = task_results['fold_test_metrics']
                for fold_name, metrics in test_metrics.items():
                    print(f"  {fold_name}: {metrics}")
            
        except Exception as e:
            print(f"✗ Failed to train on {task_name}: {e}")
            results[task_name] = {'error': str(e)}
    
    return results


def main():
    """Main test function."""
    print("TESTING LOAD OF SAVED INDUCTIVE DATA")
    print("="*60)
    
    # Path to the saved data
    data_dir = "clean_inductive_results/inductive_20250707_180516"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        return
    
    try:
        # Load the saved data
        inductive_data, sheaf_inductive_data, config_dict = load_saved_inductive_data(data_dir)
        
        # Create dataloaders from the saved data
        dataloaders = create_dataloaders_from_saved_data(inductive_data, config_dict)
        
        # Train GraphSAGE on the loaded data
        results = train_graphsage_on_saved_data(dataloaders, config_dict)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        successful_tasks = 0
        total_tasks = len(results)
        
        for task_name, task_results in results.items():
            if 'error' not in task_results:
                successful_tasks += 1
                print(f"✓ {task_name}: Success")
            else:
                print(f"✗ {task_name}: Failed - {task_results['error']}")
        
        print(f"\nSuccess rate: {successful_tasks}/{total_tasks} tasks")
        
        if successful_tasks > 0:
            print("\n✓ SUCCESS: Can load saved inductive data and train models on same splits!")
        else:
            print("\n✗ FAILED: Could not train models on loaded data")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 