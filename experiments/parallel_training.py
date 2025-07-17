"""
Parallel GPU training implementation for multiple models.

This module supports two parallelization modes:

1. PER-TASK PARALLELIZATION (default):
   - Processes tasks sequentially
   - Within each task, runs all models in parallel
   - Example: Task1 (GCN, SAGE, GAT in parallel) -> Task2 (GCN, SAGE, GAT in parallel)
   - Good for memory-constrained scenarios

2. CROSS-TASK PARALLELIZATION (--cross_task_parallelization):
   - Runs ALL models across ALL tasks in parallel
   - Example: Task1-GCN, Task1-SAGE, Task2-GCN, Task2-SAGE all in parallel
   - Maximum parallelization but requires more GPU memory
   - Use with --cross_task_parallelization flag

Usage:
  # Per-task parallelization (default)
  python run_inductive_experiments.py --use_parallel_training
  
  # Cross-task parallelization
  python run_inductive_experiments.py --use_parallel_training --cross_task_parallelization
"""

import torch
import threading
import queue
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import psutil

@dataclass
class ModelTrainingJob:
    """Represents a single model training job."""
    model_name: str
    model_creator: callable  # Function that creates the model
    model_kwargs: dict
    dataloaders: dict
    config: Any
    task: str
    job_id: str
    
class ParallelGPUTrainer:
    """Manages parallel training of multiple models on same GPU."""
    
    def __init__(self, device: torch.device, max_parallel_jobs: int = None, cross_task_mode: bool = False):
        self.device = device
        self.cross_task_mode = cross_task_mode
        self.max_parallel_jobs = max_parallel_jobs or self._estimate_max_jobs()
        self.active_jobs = {}
        self.results = {}
        self.gpu_lock = threading.Lock()  # For coordinating GPU access
        self.model_creation_lock = threading.Lock()  # For thread-safe model creation
        
    def _estimate_max_jobs(self) -> int:
        """Estimate maximum parallel jobs based on available GPU memory."""
        if not torch.cuda.is_available():
            return 1
            
        try:
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated(self.device)
            
            # More accurate memory estimation based on model type and config
            # Base memory for model + optimizer + gradients + activations
            # Conservative estimate: 200MB per model for small GNNs
            estimated_memory_per_model = 200 * 1024 * 1024  # 200MB per model
            
            # For cross-task parallelization, we might need more memory per model
            # because different tasks might have different data sizes
            # Increase estimate by 50% for cross-task scenarios
            if hasattr(self, 'cross_task_mode') and self.cross_task_mode:
                estimated_memory_per_model = int(estimated_memory_per_model * 1.5)
                print(f"🔧 Cross-task mode detected - increased memory estimate per model")
            
            # Leave 20% buffer for system operations
            max_jobs = min(20, max(1, int(available_memory * 0.8 / estimated_memory_per_model)))
            print(f"🔧 Estimated max parallel jobs: {max_jobs} (GPU memory: {total_memory/1e9:.1f}GB)")
            print(f"🔧 Memory per model: {estimated_memory_per_model/1e6:.0f}MB")
            return max_jobs
            
        except Exception as e:
            print(f"Warning: Could not estimate GPU capacity: {e}")
            return 2  # Conservative default
    
    def monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitor current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
            
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        
        return {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'total_gb': total / 1e9,
            'utilization': allocated / total
        }
    
    def _train_single_model_parallel(self, job: ModelTrainingJob) -> Dict[str, Any]:
        """Train a single model in parallel-safe manner."""
        try:
            print(f"🚀 Starting parallel training: {job.job_id}")
            
            # Monitor memory before starting
            memory_before = self.monitor_gpu_memory()
            print(f"📊 Memory before {job.job_id}: {memory_before.get('allocated_gb', 0):.2f}GB allocated")
            
            # Create model on device with thread safety
            with self.model_creation_lock:
                model = job.model_creator(**job.model_kwargs)
                model = model.to(self.device)
            
            # Import training function
            from experiments.inductive.training import train_and_evaluate_inductive
            
            # Train model - the existing function is thread-safe for different models
            # Data is already on GPU, so no need for additional device transfers
            results = train_and_evaluate_inductive(
                model=model,
                model_name=job.model_name,
                dataloaders=job.dataloaders,
                config=job.config,
                task=job.task,
                device=self.device,
                optimize_hyperparams=job.config.optimize_hyperparams,
                experiment_name=getattr(job.config, 'experiment_name', None),
                run_id=getattr(job.config, 'run_id', None),
            )
            
            # Clean up GPU memory - only the model, not the data (data is shared)
            with self.gpu_lock:
                del model
                torch.cuda.empty_cache()
                gc.collect()
            
            # Monitor memory after cleanup
            memory_after = self.monitor_gpu_memory()
            print(f"✅ Completed {job.job_id}: {memory_after.get('allocated_gb', 0):.2f}GB allocated")
            
            return {
                'job_id': job.job_id,
                'model_name': job.model_name,
                'task': job.task,
                'results': results,
                'success': True,
                'memory_before': memory_before,
                'memory_after': memory_after
            }
            
        except Exception as e:
            print(f"❌ Failed {job.job_id}: {str(e)}")
            
            # Still clean up on failure
            try:
                with self.gpu_lock:
                    if 'model' in locals():
                        del model
                    torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass
                
            return {
                'job_id': job.job_id,
                'model_name': job.model_name,
                'task': job.task,
                'error': str(e),
                'success': False
            }
    
    def train_models_parallel(self, jobs: List[ModelTrainingJob]) -> Dict[str, Any]:
        """Train multiple models in parallel."""
        print(f"🔄 Starting parallel training of {len(jobs)} models")
        print(f"📈 Max parallel jobs: {self.max_parallel_jobs}")
        
        start_time = time.time()
        all_results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._train_single_model_parallel, job): job 
                for job in jobs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                result = future.result()
                
                all_results[job.job_id] = result
                completed += 1
                
                success_status = "✅" if result['success'] else "❌"
                print(f"{success_status} [{completed}/{len(jobs)}] {job.job_id} completed")
                
                # Print current GPU utilization
                memory_stats = self.monitor_gpu_memory()
                print(f"🖥️  GPU: {memory_stats.get('allocated_gb', 0):.2f}GB allocated")
        
        total_time = time.time() - start_time
        successful = sum(1 for r in all_results.values() if r['success'])
        
        print(f"\n🏁 Parallel training completed:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful: {successful}/{len(jobs)}")
        print(f"   Speedup estimate: {len(jobs) * total_time / self.max_parallel_jobs / total_time:.1f}x")
        
        return {
            'results': all_results,
            'total_time': total_time,
            'successful_jobs': successful,
            'failed_jobs': len(jobs) - successful
        }

def create_model_jobs(task_dataloaders: dict, config: Any, task: str, device: torch.device) -> List[ModelTrainingJob]:
    """Create model training jobs for all configured models (legacy function for backward compatibility)."""
    # This is now a wrapper around the new function
    all_dataloaders = {task: task_dataloaders}
    return create_all_model_jobs_across_tasks(all_dataloaders, config, device)

def create_all_model_jobs_across_tasks(
    all_dataloaders: Dict[str, Dict[str, Any]], 
    config: Any, 
    device: torch.device
) -> List[ModelTrainingJob]:
    """Create model training jobs for ALL tasks and ALL models at once."""
    all_jobs = []
    
    for task in config.tasks:
        if task not in all_dataloaders:
            print(f"Warning: Task {task} not found in dataloaders")
            continue
            
        task_dataloaders = all_dataloaders[task]
        
        # Get dimensions from sample batch
        # Handle new single split structure
        if 'split' in task_dataloaders:
            # New structure: task_dataloaders['split']['train']
            sample_batch = next(iter(task_dataloaders['split']['train']))
        else:
            # Old fold-based structure (for backward compatibility)
            first_fold_name = list(task_dataloaders.keys())[0]
            sample_batch = next(iter(task_dataloaders[first_fold_name]['train']))
        input_dim = sample_batch.x.shape[1]
        
        is_regression = task.startswith('k_hop_community_counts') or task == 'triangle_count'
        is_graph_level_task = task == 'triangle_count'
        
        # Determine output dimension
        if is_regression:
            output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
        else:
            from experiments.inductive.training import get_total_classes_from_dataloaders
            output_dim = get_total_classes_from_dataloaders(task_dataloaders)
        
        # Create GNN model jobs
        if config.run_gnn:
            for gnn_type in config.gnn_types:
                from experiments.models import GNNModel
                
                def create_gnn_model(gnn_type=gnn_type, **kwargs):
                    return GNNModel(
                        input_dim=input_dim,
                        hidden_dim=config.hidden_dim,
                        output_dim=output_dim,
                        num_layers=config.num_layers,
                        dropout=config.dropout,
                        gnn_type=gnn_type,
                        is_regression=is_regression,
                        is_graph_level_task=is_graph_level_task,
                        **kwargs
                    )
                
                all_jobs.append(ModelTrainingJob(
                    model_name=gnn_type,
                    model_creator=create_gnn_model,
                    model_kwargs={},
                    dataloaders=task_dataloaders,
                    config=config,
                    task=task,
                    job_id=f"{task}_{gnn_type}"
                ))
        
        # Create transformer model jobs
        if config.run_transformers:
            for transformer_type in config.transformer_types:
                from experiments.models import GraphTransformerModel
                
                def create_transformer_model(transformer_type=transformer_type, **kwargs):
                    return GraphTransformerModel(
                        input_dim=input_dim,
                        hidden_dim=config.hidden_dim,
                        output_dim=output_dim,
                        transformer_type=transformer_type,
                        num_layers=config.num_layers,
                        dropout=config.dropout,
                        is_regression=is_regression,
                        is_graph_level_task=is_graph_level_task,
                        **kwargs
                    )
                
                all_jobs.append(ModelTrainingJob(
                    model_name=transformer_type,
                    model_creator=create_transformer_model,
                    model_kwargs={},
                    dataloaders=task_dataloaders,
                    config=config,
                    task=task,
                    job_id=f"{task}_{transformer_type}"
                ))
        
        # Create MLP model jobs
        if config.run_mlp:
            from experiments.models import MLPModel
            
            def create_mlp_model(**kwargs):
                return MLPModel(
                    input_dim=input_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=output_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    is_regression=is_regression,
                    is_graph_level_task=is_graph_level_task
                )
            
            all_jobs.append(ModelTrainingJob(
                model_name='mlp',
                model_creator=create_mlp_model,
                model_kwargs={},
                dataloaders=task_dataloaders,
                config=config,
                task=task,
                job_id=f"{task}_mlp"
            ))
        
        # Create Random Forest model jobs
        if config.run_rf:
            from experiments.models import SklearnModel
            
            def create_rf_model(**kwargs):
                return SklearnModel(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type="rf",
                    is_regression=is_regression,
                    is_graph_level_task=is_graph_level_task
                )
            
            all_jobs.append(ModelTrainingJob(
                model_name='rf',
                model_creator=create_rf_model,
                model_kwargs={},
                dataloaders=task_dataloaders,
                config=config,
                task=task,
                job_id=f"{task}_rf"
            ))
    
    return all_jobs

# Modified experiment run_experiments method
def run_experiments_parallel(experiment_instance) -> Dict[str, Any]:
    """Drop-in replacement for run_experiments with parallel training."""
    print("\n" + "="*60)
    print("RUNNING PARALLEL INDUCTIVE EXPERIMENTS")
    print("="*60)
    
    if not hasattr(experiment_instance, 'dataloaders'):
        raise ValueError("Must prepare data before running experiments")
    
    all_results = {}
    
    # Initialize parallel trainer
    parallel_trainer = ParallelGPUTrainer(
        experiment_instance.device, 
        max_parallel_jobs=getattr(experiment_instance.config, 'max_parallel_gpu_jobs', None),
        cross_task_mode=experiment_instance.config.cross_task_parallelization
    )
    
    # Check if we should use cross-task parallelization
    if experiment_instance.config.cross_task_parallelization:
        # CROSS-TASK PARALLELIZATION: Create ALL jobs across ALL tasks at once
        print("🔄 CROSS-TASK PARALLELIZATION MODE")
        print("   All tasks and models will train simultaneously!")
        print("   This maximizes GPU utilization but requires more memory.")
        
        all_jobs = create_all_model_jobs_across_tasks(
            experiment_instance.dataloaders, 
            experiment_instance.config, 
            experiment_instance.device
        )
        
        if not all_jobs:
            print("No models configured for any task")
            return all_results
        
        # Show job breakdown
        print(f"\n📊 JOB BREAKDOWN:")
        task_model_counts = {}
        for job in all_jobs:
            task = job.task
            model = job.model_name
            if task not in task_model_counts:
                task_model_counts[task] = {}
            if model not in task_model_counts[task]:
                task_model_counts[task][model] = 0
            task_model_counts[task][model] += 1
        
        for task, models in task_model_counts.items():
            print(f"   {task}: {', '.join(models.keys())}")
        
        print(f"\n🔄 Created {len(all_jobs)} total training jobs across all tasks")
        print(f"📈 Max parallel jobs: {parallel_trainer.max_parallel_jobs}")
        print(f"⚡ Expected speedup: ~{len(all_jobs) / parallel_trainer.max_parallel_jobs:.1f}x")
        
        # Train ALL models across ALL tasks in parallel
        parallel_results = parallel_trainer.train_models_parallel(all_jobs)
        
        # Organize results back by task
        print(f"\n📋 ORGANIZING RESULTS BY TASK:")
        for job_id, job_result in parallel_results['results'].items():
            if job_result['success']:
                task = job_result['task']
                model_name = job_result['model_name']
                
                if task not in all_results:
                    all_results[task] = {}
                
                all_results[task][model_name] = job_result['results']
                print(f"   ✅ {task}_{model_name}: Success")
            else:
                task = job_result['task']
                model_name = job_result['model_name']
                
                if task not in all_results:
                    all_results[task] = {}
                
                all_results[task][model_name] = {
                    'error': job_result['error'],
                    'test_metrics': {}
                }
                print(f"   ❌ {task}_{model_name}: Failed - {job_result['error']}")
        
        # Print final summary
        successful_jobs = sum(1 for r in parallel_results['results'].values() if r['success'])
        total_jobs = len(parallel_results['results'])
        print(f"\n🏁 CROSS-TASK PARALLELIZATION COMPLETED:")
        print(f"   Total jobs: {total_jobs}")
        print(f"   Successful: {successful_jobs}")
        print(f"   Failed: {total_jobs - successful_jobs}")
        print(f"   Success rate: {successful_jobs/total_jobs:.1%}")
        print(f"   Total time: {parallel_results['total_time']:.2f}s")
    
    else:
        # PER-TASK PARALLELIZATION: Process each task separately (original behavior)
        print("🔄 PER-TASK PARALLELIZATION MODE")
        print("   Tasks will be processed sequentially.")
        print("   Models within each task will train in parallel.")
        
        for task in experiment_instance.config.tasks:
            print(f"\n{'='*40}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*40}")
            
            task_dataloaders = experiment_instance.dataloaders[task]
            
            # Create training jobs for this task
            jobs = create_model_jobs(task_dataloaders, experiment_instance.config, task, experiment_instance.device)
            
            if not jobs:
                print(f"No models configured for task {task}")
                continue
                
            print(f"Created {len(jobs)} training jobs for task {task}")
            print(f"Models: {', '.join([job.model_name for job in jobs])}")
            
            # Train models in parallel for this task
            parallel_results = parallel_trainer.train_models_parallel(jobs)
            
            # Convert results to expected format
            task_results = {}
            for job_id, job_result in parallel_results['results'].items():
                if job_result['success']:
                    model_name = job_result['model_name']
                    task_results[model_name] = job_result['results']
                    print(f"   ✅ {model_name}: Success")
                else:
                    model_name = job_result['model_name']
                    task_results[model_name] = {
                        'error': job_result['error'],
                        'test_metrics': {}
                    }
                    print(f"   ❌ {model_name}: Failed - {job_result['error']}")
            
            all_results[task] = task_results
    
    return all_results