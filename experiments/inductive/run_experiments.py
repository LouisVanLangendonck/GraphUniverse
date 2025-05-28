import os
import json
from typing import Dict, Any

def save_results(results: Dict[str, Any], config: InductiveExperimentConfig, output_dir: str):
    """Save experiment results to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'results': {
                'train_time': results['train_time'],
                'best_val_metric': results['best_val_metric'],
                'test_metrics': results['test_metrics'],
                'training_history': results['training_history']
            }
        }, f, indent=2)
    
    # Save training history plot
    plot_file = os.path.join(output_dir, 'training_history.png')
    plot_training_history(results['training_history'], plot_file)
    
    # Print summary
    print("\nExperiment Results:")
    print(f"Training time: {results['train_time']:.2f} seconds")
    print(f"Best validation metric: {results['best_val_metric']:.4f}")
    print("\nTest Metrics:")
    for metric, value in results['test_metrics']['metrics'].items():
        print(f"{metric}: {value:.4f}") 