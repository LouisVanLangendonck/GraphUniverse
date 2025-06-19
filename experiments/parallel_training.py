"""
Parallel GPU training implementation for multiple models.
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
    
    def __init__(self, device: torch.device, max_parallel_jobs: int = None):
        self.device = device
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
            
            # Leave 20% buffer for system operations
            max_jobs = min(4, max(1, int(available_memory * 0.8 / estimated_memory_per_model)))
            print(f"🔧 Estimated max parallel jobs: {max_jobs} (GPU memory: {total_memory/1e9:.1f}GB)")
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
    """Create model training jobs for all configured models."""
    jobs = []
    
    # Get dimensions from sample batch
    first_fold_name = list(task_dataloaders.keys())[0]
    sample_batch = next(iter(task_dataloaders[first_fold_name]['train']))
    input_dim = sample_batch.x.shape[1]
    
    is_regression = task in ['k_hop_community_counts', 'triangle_count']
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
            
            jobs.append(ModelTrainingJob(
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
            
            jobs.append(ModelTrainingJob(
                model_name=transformer_type,
                model_creator=create_transformer_model,
                model_kwargs={},
                dataloaders=task_dataloaders,
                config=config,
                task=task,
                job_id=f"{task}_{transformer_type}"
            ))
    
    # MLP and RF models - these are fast, might not need parallelization
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
        
        jobs.append(ModelTrainingJob(
            model_name='mlp',
            model_creator=create_mlp_model,
            model_kwargs={},
            dataloaders=task_dataloaders,
            config=config,
            task=task,
            job_id=f"{task}_mlp"
        ))
    
    return jobs

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
    parallel_trainer = ParallelGPUTrainer(experiment_instance.device)
    
    # Process each task
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
        
        # Train models in parallel
        parallel_results = parallel_trainer.train_models_parallel(jobs)
        
        # Convert results to expected format
        task_results = {}
        for job_id, job_result in parallel_results['results'].items():
            if job_result['success']:
                model_name = job_result['model_name']
                task_results[model_name] = job_result['results']
            else:
                model_name = job_result['model_name']
                task_results[model_name] = {
                    'error': job_result['error'],
                    'test_metrics': {}
                }
        
        all_results[task] = task_results
    
    return all_results