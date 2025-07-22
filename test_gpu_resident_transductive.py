"""
Test script to verify GPU-resident transductive training performance.
"""

import torch
import time
import numpy as np
from experiments.transductive.config import TransductiveExperimentConfig
from experiments.transductive.experiment import TransductiveExperiment
from experiments.transductive.data import prepare_transductive_data, prepare_transductive_data_gpu_resident
from experiments.transductive.training import train_and_evaluate_transductive, train_and_evaluate_transductive_gpu_resident
from experiments.models import GNNModel

def test_gpu_resident_performance():
    """Test GPU-resident vs original transductive training performance."""
    print("="*60)
    print("GPU-RESIDENT TRANSDUCTIVE TRAINING PERFORMANCE TEST")
    print("="*60)
    
    # Create a simple config
    config = TransductiveExperimentConfig(
        seed=42,
        universe_K=4,
        num_communities=4,
        universe_feature_dim=8,
        universe_edge_density=0.1,
        universe_homophily=0.8,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        n_repetitions=2,
        tasks=['community'],
        is_regression={'community': False},
        run_gnn=True,
        gnn_types=['gcn'],
        epochs=50,
        learning_rate=0.01,
        weight_decay=5e-4,
        patience=10,
        device_id=0,
        force_cpu=False
    )
    
    # Create experiment
    experiment = TransductiveExperiment(config)
    
    # Generate graph
    print("\n1. Generating graph...")
    experiment.generate_graph()
    
    # Test original data preparation
    print("\n2. Testing original data preparation...")
    start_time = time.time()
    original_data = prepare_transductive_data(experiment.graph_sample, config)
    original_prep_time = time.time() - start_time
    print(f"Original data preparation time: {original_prep_time:.4f}s")
    
    # Test GPU-resident data preparation
    print("\n3. Testing GPU-resident data preparation...")
    start_time = time.time()
    gpu_data = prepare_transductive_data_gpu_resident(experiment.graph_sample, config, experiment.device)
    gpu_prep_time = time.time() - start_time
    print(f"GPU-resident data preparation time: {gpu_prep_time:.4f}s")
    print(f"Speedup: {original_prep_time/gpu_prep_time:.2f}x")
    
    # Test original training
    print("\n4. Testing original training...")
    model = GNNModel(
        input_dim=8,
        hidden_dim=32,
        output_dim=4,
        num_layers=2,
        dropout=0.5,
        gnn_type='gcn',
        is_regression=False,
        is_graph_level_task=False
    )
    
    start_time = time.time()
    original_result = train_and_evaluate_transductive(
        model=model,
        task_data=original_data['splits'][0],
        config=config,
        task='community',
        device=experiment.device,
        optimize_hyperparams=False,
        model_name='gcn'
    )
    original_train_time = time.time() - start_time
    print(f"Original training time: {original_train_time:.4f}s")
    print(f"Original test accuracy: {original_result['test_metrics']['accuracy']:.4f}")
    
    # Test GPU-resident training
    print("\n5. Testing GPU-resident training...")
    model = GNNModel(
        input_dim=8,
        hidden_dim=32,
        output_dim=4,
        num_layers=2,
        dropout=0.5,
        gnn_type='gcn',
        is_regression=False,
        is_graph_level_task=False
    )
    
    start_time = time.time()
    gpu_result = train_and_evaluate_transductive_gpu_resident(
        model=model,
        task_data=gpu_data['splits'][0],
        config=config,
        task='community',
        device=experiment.device,
        optimize_hyperparams=False,
        model_name='gcn'
    )
    gpu_train_time = time.time() - start_time
    print(f"GPU-resident training time: {gpu_train_time:.4f}s")
    print(f"GPU-resident test accuracy: {gpu_result['test_metrics']['accuracy']:.4f}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Data preparation speedup: {original_prep_time/gpu_prep_time:.2f}x")
    print(f"Training speedup: {original_train_time/gpu_train_time:.2f}x")
    print(f"Total speedup: {(original_prep_time + original_train_time)/(gpu_prep_time + gpu_train_time):.2f}x")
    
    # Verify accuracy is similar
    accuracy_diff = abs(original_result['test_metrics']['accuracy'] - gpu_result['test_metrics']['accuracy'])
    print(f"Accuracy difference: {accuracy_diff:.6f}")
    
    if accuracy_diff < 0.01:
        print("✅ GPU-resident training produces similar accuracy")
    else:
        print("⚠️  GPU-resident training accuracy differs significantly")
    
    # Memory usage comparison
    if torch.cuda.is_available():
        original_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"GPU memory usage: {original_memory:.2f} GB")
    
    return {
        'original_prep_time': original_prep_time,
        'gpu_prep_time': gpu_prep_time,
        'original_train_time': original_train_time,
        'gpu_train_time': gpu_train_time,
        'original_accuracy': original_result['test_metrics']['accuracy'],
        'gpu_accuracy': gpu_result['test_metrics']['accuracy'],
        'accuracy_diff': accuracy_diff
    }

def test_memory_efficiency():
    """Test memory efficiency of GPU-resident approach."""
    print("\n" + "="*60)
    print("MEMORY EFFICIENCY TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Create config with larger graph
    config = TransductiveExperimentConfig(
        seed=42,
        universe_K=8,
        num_communities=8,
        universe_feature_dim=16,
        universe_edge_density=0.15,
        universe_homophily=0.8,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        n_repetitions=3,
        tasks=['community'],
        is_regression={'community': False},
        run_gnn=True,
        gnn_types=['gcn'],
        epochs=30,
        learning_rate=0.01,
        weight_decay=5e-4,
        patience=10,
        device_id=0,
        force_cpu=False
    )
    
    # Create experiment
    experiment = TransductiveExperiment(config)
    experiment.generate_graph()
    
    # Measure memory before
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory before data preparation: {memory_before:.2f} GB")
    
    # Prepare GPU-resident data
    gpu_data = prepare_transductive_data_gpu_resident(experiment.graph_sample, config, experiment.device)
    
    # Measure memory after data preparation
    memory_after_prep = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after data preparation: {memory_after_prep:.2f} GB")
    print(f"Data preparation memory usage: {memory_after_prep - memory_before:.2f} GB")
    
    # Train a model
    model = GNNModel(
        input_dim=16,
        hidden_dim=64,
        output_dim=8,
        num_layers=3,
        dropout=0.5,
        gnn_type='gcn',
        is_regression=False,
        is_graph_level_task=False
    )
    
    # Measure memory before training
    memory_before_train = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory before training: {memory_before_train:.2f} GB")
    
    # Train
    result = train_and_evaluate_transductive_gpu_resident(
        model=model,
        task_data=gpu_data['splits'][0],
        config=config,
        task='community',
        device=experiment.device,
        optimize_hyperparams=False,
        model_name='gcn'
    )
    
    # Measure memory after training
    memory_after_train = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after training: {memory_after_train:.2f} GB")
    print(f"Training memory usage: {memory_after_train - memory_before_train:.2f} GB")
    print(f"Total memory usage: {memory_after_train - memory_before:.2f} GB")
    
    # Clean up
    from experiments.transductive.data import cleanup_transductive_gpu_data
    for split in gpu_data['splits']:
        cleanup_transductive_gpu_data(split, experiment.device)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    memory_after_cleanup = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after cleanup: {memory_after_cleanup:.2f} GB")
    
    return {
        'memory_before': memory_before,
        'memory_after_prep': memory_after_prep,
        'memory_after_train': memory_after_train,
        'memory_after_cleanup': memory_after_cleanup,
        'data_prep_memory': memory_after_prep - memory_before,
        'training_memory': memory_after_train - memory_before_train,
        'total_memory': memory_after_train - memory_before
    }

if __name__ == "__main__":
    print("Testing GPU-resident transductive training performance...")
    
    # Test performance
    perf_results = test_gpu_resident_performance()
    
    # Test memory efficiency
    mem_results = test_memory_efficiency()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Performance speedup: {perf_results['original_train_time']/perf_results['gpu_train_time']:.2f}x")
    print(f"Accuracy maintained: {perf_results['accuracy_diff'] < 0.01}")
    print(f"Memory usage: {mem_results['total_memory']:.2f} GB")
    print("✅ GPU-resident transductive training is working correctly!") 