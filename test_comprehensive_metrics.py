#!/usr/bin/env python3
"""
Test script to verify that comprehensive metrics are working correctly.
"""

import torch
import numpy as np
from experiments.metrics import compute_metrics, compute_metrics_gpu

def test_comprehensive_metrics():
    """Test that comprehensive metrics are computed correctly."""
    print("Testing comprehensive metrics...")
    
    # Test classification metrics
    print("\n--- Classification Metrics ---")
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])  # One error
    
    metrics = compute_metrics(y_true, y_pred, is_regression=False)
    print("CPU Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test GPU metrics
    y_true_gpu = torch.tensor(y_true, dtype=torch.long)
    y_pred_gpu = torch.tensor(y_pred, dtype=torch.long)
    
    metrics_gpu = compute_metrics_gpu(y_true_gpu, y_pred_gpu, is_regression=False)
    print("\nGPU Metrics:")
    for key, value in metrics_gpu.items():
        print(f"  {key}: {value:.4f}")
    
    # Test regression metrics
    print("\n--- Regression Metrics ---")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    metrics_reg = compute_metrics(y_true_reg, y_pred_reg, is_regression=True)
    print("CPU Regression Metrics:")
    for key, value in metrics_reg.items():
        print(f"  {key}: {value:.4f}")
    
    # Test GPU regression metrics
    y_true_reg_gpu = torch.tensor(y_true_reg, dtype=torch.float32)
    y_pred_reg_gpu = torch.tensor(y_pred_reg, dtype=torch.float32)
    
    metrics_reg_gpu = compute_metrics_gpu(y_true_reg_gpu, y_pred_reg_gpu, is_regression=True)
    print("\nGPU Regression Metrics:")
    for key, value in metrics_reg_gpu.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Comprehensive metrics test completed!")

if __name__ == "__main__":
    test_comprehensive_metrics() 