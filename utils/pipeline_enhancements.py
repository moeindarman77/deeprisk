#!/usr/bin/env python3
"""
Pipeline enhancements for improved batch processing and memory management
Extends the core pipeline with advanced features for production use
"""
import os
import time
import torch
import numpy as np
import xarray as xr
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchProcessingPipeline:
    """Enhanced pipeline with optimized batch processing capabilities"""
    
    def __init__(self, base_pipeline, batch_size: int = 4, save_strategy: str = 'immediate'):
        """
        Initialize batch processing wrapper
        
        Args:
            base_pipeline: Base IntegratedPipeline instance
            batch_size: Number of samples to process in each batch
            save_strategy: How to handle saving ('immediate', 'batched', 'async')
        """
        self.pipeline = base_pipeline
        self.batch_size = batch_size
        self.save_strategy = save_strategy
        
        # Async saving components
        self.save_queue = queue.Queue()
        self.save_thread = None
        self.saving_active = False
        
        # Performance tracking
        self.batch_times = []
        self.memory_usage = []
        
    def start_async_saving(self, output_dir: str):
        """Start async saving thread"""
        if self.save_strategy == 'async':
            self.saving_active = True
            self.save_thread = threading.Thread(
                target=self._async_save_worker,
                args=(output_dir,),
                daemon=True
            )
            self.save_thread.start()
    
    def stop_async_saving(self):
        """Stop async saving thread"""
        if self.save_strategy == 'async' and self.saving_active:
            self.saving_active = False
            self.save_queue.put(None)  # Sentinel to stop worker
            if self.save_thread:
                self.save_thread.join(timeout=30)
    
    def _async_save_worker(self, output_dir: str):
        """Worker thread for async saving"""
        while self.saving_active:
            try:
                item = self.save_queue.get(timeout=1)
                if item is None:  # Sentinel to stop
                    break
                
                result, sample_idx, save_intermediate = item
                self.pipeline.save_single_sample(result, sample_idx, output_dir, save_intermediate)
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async saving: {e}")
    
    def process_batch(self, start_idx: int, end_idx: int, output_dir: str) -> List[Dict]:
        """Process a batch of samples with optimized memory management"""
        batch_start = time.time()
        batch_results = []
        
        print(f"Processing batch: samples {start_idx}-{end_idx-1}")
        
        # Track memory before batch
        initial_memory = self._get_memory_usage()
        
        for sample_idx in range(start_idx, end_idx):
            try:
                # Process sample
                l2_output, edm_output, timestamp, original_idx = self.pipeline.run_single_sample(sample_idx)
                
                result = {
                    'sample_idx': sample_idx,
                    'original_idx': original_idx,
                    'timestamp': timestamp,
                    'l2_output': l2_output,
                    'edm_output': edm_output,
                    'success': True
                }
                
                # Handle saving based on strategy
                if self.save_strategy == 'immediate':
                    self.pipeline.save_single_sample(result, sample_idx, output_dir, save_intermediate=True)
                    # Clear from memory immediately
                    result['l2_output'] = None
                    result['edm_output'] = None
                    
                elif self.save_strategy == 'async':
                    # Queue for async saving
                    self.save_queue.put((result.copy(), sample_idx, True))
                    # Clear from memory
                    result['l2_output'] = None  
                    result['edm_output'] = None
                    
                # For 'batched' strategy, keep in memory until batch complete
                
                batch_results.append(result)
                
                # Force cleanup after each sample
                del l2_output, edm_output
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                batch_results.append({
                    'sample_idx': sample_idx,
                    'error': str(e),
                    'success': False
                })
        
        # Handle batched saving
        if self.save_strategy == 'batched':
            self._save_batch(batch_results, output_dir)
            # Clear batch data from memory
            for result in batch_results:
                result['l2_output'] = None
                result['edm_output'] = None
        
        # Track batch performance
        batch_time = time.time() - batch_start
        final_memory = self._get_memory_usage()
        
        self.batch_times.append(batch_time)
        self.memory_usage.append({
            'initial_gb': initial_memory,
            'final_gb': final_memory,
            'batch_size': len(batch_results)
        })
        
        successful = sum(1 for r in batch_results if r['success'])
        print(f"Batch completed: {successful}/{len(batch_results)} samples in {batch_time:.1f}s")
        
        return batch_results
    
    def _save_batch(self, batch_results: List[Dict], output_dir: str):
        """Save a complete batch of results"""
        for result in batch_results:
            if result['success'] and result.get('l2_output') is not None:
                self.pipeline.save_single_sample(result, result['sample_idx'], output_dir, save_intermediate=True)
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def run_batched_pipeline(self, max_samples: int, output_dir: str) -> Dict[str, Any]:
        """Run pipeline with batch processing"""
        print(f"Starting batched pipeline: {max_samples} samples, batch size {self.batch_size}")
        
        # Setup async saving if needed
        if self.save_strategy == 'async':
            self.start_async_saving(output_dir)
        
        total_start = time.time()
        all_results = []
        
        try:
            # Process in batches
            for start_idx in range(0, max_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, max_samples)
                
                batch_results = self.process_batch(start_idx, end_idx, output_dir)
                all_results.extend(batch_results)
        
        finally:
            # Stop async saving
            if self.save_strategy == 'async':
                self.stop_async_saving()
        
        total_time = time.time() - total_start
        successful = sum(1 for r in all_results if r['success'])
        
        # Performance summary
        performance_summary = {
            'total_time': total_time,
            'total_samples': len(all_results),
            'successful_samples': successful,
            'avg_time_per_sample': total_time / len(all_results) if all_results else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'batch_size': self.batch_size,
            'save_strategy': self.save_strategy,
            'memory_efficiency': self._calculate_memory_efficiency()
        }
        
        print(f"Batched pipeline completed: {successful}/{len(all_results)} samples in {total_time:.1f}s")
        
        return {
            'results': all_results,
            'performance': performance_summary,
            'success': successful == len(all_results)
        }
    
    def _calculate_memory_efficiency(self) -> Dict[str, float]:
        """Calculate memory usage efficiency metrics"""
        if not self.memory_usage:
            return {}
        
        initial_memories = [m['initial_gb'] for m in self.memory_usage]
        final_memories = [m['final_gb'] for m in self.memory_usage]
        
        return {
            'avg_initial_memory_gb': np.mean(initial_memories),
            'avg_final_memory_gb': np.mean(final_memories),
            'peak_memory_gb': max(final_memories) if final_memories else 0,
            'memory_growth_gb': np.mean(final_memories) - np.mean(initial_memories) if initial_memories and final_memories else 0
        }


class MemoryOptimizedDataset:
    """Memory-optimized dataset wrapper with lazy loading and caching"""
    
    def __init__(self, base_dataset, cache_size: int = 100, preload: bool = False):
        """
        Initialize memory-optimized dataset
        
        Args:
            base_dataset: Base dataset instance
            cache_size: Number of samples to keep in memory
            preload: Whether to preload frequently accessed samples
        """
        self.base_dataset = base_dataset
        self.cache_size = cache_size
        self.preload = preload
        
        # LRU cache for samples
        self.cache = {}
        self.access_count = {}
        self.cache_order = []
        
        # Preloading
        if preload and len(base_dataset) > 0:
            self._preload_samples()
    
    def _preload_samples(self, n_samples: int = None):
        """Preload frequently accessed samples"""
        if n_samples is None:
            n_samples = min(self.cache_size // 2, len(self.base_dataset))
        
        print(f"Preloading {n_samples} samples...")
        
        # Load evenly distributed samples
        indices = np.linspace(0, len(self.base_dataset) - 1, n_samples, dtype=int)
        
        for idx in indices:
            try:
                sample = self.base_dataset[idx]
                self._add_to_cache(idx, sample)
            except Exception as e:
                print(f"Error preloading sample {idx}: {e}")
    
    def _add_to_cache(self, idx: int, sample):
        """Add sample to cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_idx = self.cache_order.pop(0)
            del self.cache[lru_idx]
            if lru_idx in self.access_count:
                del self.access_count[lru_idx]
        
        self.cache[idx] = sample
        self.cache_order.append(idx)
        self.access_count[idx] = self.access_count.get(idx, 0) + 1
    
    def __getitem__(self, idx: int):
        """Get item with caching"""
        # Check cache first
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Load from base dataset
        sample = self.base_dataset[idx]
        
        # Add to cache
        self._add_to_cache(idx, sample)
        
        return sample
    
    def __len__(self):
        return len(self.base_dataset)
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'total_accesses': sum(self.access_count.values()),
            'unique_samples_accessed': len(self.access_count),
            'cache_hit_rate': len([idx for idx in self.access_count if idx in self.cache]) / len(self.access_count) if self.access_count else 0
        }


class PipelineProfiler:
    """Profiler for pipeline performance analysis"""
    
    def __init__(self):
        self.profiles = []
        self.current_profile = None
    
    def start_profiling(self, profile_name: str):
        """Start a new profiling session"""
        self.current_profile = {
            'name': profile_name,
            'start_time': time.time(),
            'stages': [],
            'memory_snapshots': [],
            'gpu_utilization': []
        }
    
    def log_stage(self, stage_name: str, duration: float = None, memory_usage: float = None):
        """Log a pipeline stage"""
        if not self.current_profile:
            return
        
        stage_info = {
            'stage': stage_name,
            'timestamp': time.time(),
            'duration': duration,
            'memory_usage_gb': memory_usage or self._get_current_memory_usage()
        }
        
        self.current_profile['stages'].append(stage_info)
    
    def snapshot_memory(self):
        """Take memory usage snapshot"""
        if not self.current_profile:
            return
        
        snapshot = {
            'timestamp': time.time(),
            'cpu_memory_gb': self._get_cpu_memory(),
            'gpu_memory_gb': self._get_gpu_memory(),
            'gpu_utilization': self._get_gpu_utilization()
        }
        
        self.current_profile['memory_snapshots'].append(snapshot)
    
    def end_profiling(self):
        """End current profiling session"""
        if not self.current_profile:
            return None
        
        self.current_profile['end_time'] = time.time()
        self.current_profile['total_duration'] = self.current_profile['end_time'] - self.current_profile['start_time']
        
        self.profiles.append(self.current_profile)
        profile = self.current_profile
        self.current_profile = None
        
        return profile
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_cpu_memory(self) -> float:
        """Get CPU memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**3)
        except ImportError:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except (ImportError, Exception):
            return 0.0
    
    def generate_report(self, profile_name: str = None) -> Dict[str, Any]:
        """Generate performance report"""
        target_profiles = [p for p in self.profiles if profile_name is None or p['name'] == profile_name]
        
        if not target_profiles:
            return {}
        
        report = {
            'total_profiles': len(target_profiles),
            'average_duration': np.mean([p['total_duration'] for p in target_profiles]),
            'stage_analysis': self._analyze_stages(target_profiles),
            'memory_analysis': self._analyze_memory(target_profiles),
            'bottlenecks': self._identify_bottlenecks(target_profiles)
        }
        
        return report
    
    def _analyze_stages(self, profiles: List[Dict]) -> Dict[str, Any]:
        """Analyze stage performance across profiles"""
        all_stages = {}
        
        for profile in profiles:
            for stage in profile['stages']:
                stage_name = stage['stage']
                if stage_name not in all_stages:
                    all_stages[stage_name] = []
                
                if stage['duration'] is not None:
                    all_stages[stage_name].append(stage['duration'])
        
        stage_stats = {}
        for stage_name, durations in all_stages.items():
            if durations:
                stage_stats[stage_name] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'std_duration': np.std(durations)
                }
        
        return stage_stats
    
    def _analyze_memory(self, profiles: List[Dict]) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        all_cpu_memory = []
        all_gpu_memory = []
        
        for profile in profiles:
            cpu_memories = [s['cpu_memory_gb'] for s in profile['memory_snapshots']]
            gpu_memories = [s['gpu_memory_gb'] for s in profile['memory_snapshots']]
            
            all_cpu_memory.extend(cpu_memories)
            all_gpu_memory.extend(gpu_memories)
        
        memory_analysis = {}
        
        if all_cpu_memory:
            memory_analysis['cpu'] = {
                'avg_usage_gb': np.mean(all_cpu_memory),
                'peak_usage_gb': np.max(all_cpu_memory),
                'min_usage_gb': np.min(all_cpu_memory)
            }
        
        if all_gpu_memory:
            memory_analysis['gpu'] = {
                'avg_usage_gb': np.mean(all_gpu_memory),
                'peak_usage_gb': np.max(all_gpu_memory),
                'min_usage_gb': np.min(all_gpu_memory)
            }
        
        return memory_analysis
    
    def _identify_bottlenecks(self, profiles: List[Dict]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        stage_analysis = self._analyze_stages(profiles)
        
        bottlenecks = []
        
        # Find slowest stages
        for stage_name, stats in stage_analysis.items():
            if stats['avg_duration'] > 5.0:  # Stages taking more than 5 seconds
                bottlenecks.append({
                    'type': 'slow_stage',
                    'stage': stage_name,
                    'avg_duration': stats['avg_duration'],
                    'suggestion': f"Optimize {stage_name} - taking {stats['avg_duration']:.1f}s on average"
                })
        
        # Check for high memory usage
        memory_analysis = self._analyze_memory(profiles)
        if 'gpu' in memory_analysis and memory_analysis['gpu']['peak_usage_gb'] > 14:
            bottlenecks.append({
                'type': 'high_memory',
                'peak_memory_gb': memory_analysis['gpu']['peak_usage_gb'],
                'suggestion': f"High GPU memory usage: {memory_analysis['gpu']['peak_usage_gb']:.1f}GB - consider batch size reduction"
            })
        
        return bottlenecks


# Example usage functions
def create_enhanced_pipeline(base_pipeline, batch_size: int = 4, save_strategy: str = 'immediate'):
    """Create enhanced pipeline with batch processing"""
    return BatchProcessingPipeline(base_pipeline, batch_size, save_strategy)

def optimize_dataset_loading(base_dataset, cache_size: int = 100, preload: bool = True):
    """Create memory-optimized dataset wrapper"""
    return MemoryOptimizedDataset(base_dataset, cache_size, preload)

def profile_pipeline_run(pipeline, profiler_name: str = "pipeline_run"):
    """Context manager for profiling pipeline runs"""
    class ProfilerContext:
        def __init__(self, profiler, name):
            self.profiler = PipelineProfiler()
            self.name = name
        
        def __enter__(self):
            self.profiler.start_profiling(self.name)
            return self.profiler
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.profiler.end_profiling()
    
    return ProfilerContext(PipelineProfiler(), profiler_name)