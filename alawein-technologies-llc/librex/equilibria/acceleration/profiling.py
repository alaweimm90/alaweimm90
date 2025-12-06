"""
Performance Profiling Module

Tools for profiling and analyzing optimization performance.
"""

import logging
import time
import psutil
import numpy as np
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Comprehensive profiling results"""
    # Timing
    total_runtime: float
    initialization_time: float
    evaluation_time: float
    selection_time: float
    crossover_time: float
    mutation_time: float
    communication_time: float

    # Throughput
    evaluations_per_second: float
    generations_per_second: float

    # Memory
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency: float

    # GPU Utilization
    gpu_utilization: float
    gpu_memory_mb: float
    gpu_compute_efficiency: float

    # Bottlenecks
    primary_bottleneck: str
    bottleneck_percentage: float

    # Scalability
    parallel_efficiency: float
    speedup_factor: float

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    backend: str = ""
    n_devices: int = 1
    problem_dimension: int = 0
    population_size: int = 0


class PerformanceProfiler:
    """
    Profile GPU/CPU utilization and bottlenecks.

    Features:
    - Fine-grained timing of optimization components
    - GPU utilization monitoring
    - Memory profiling
    - Bottleneck identification
    - Performance visualization
    - Comparative analysis
    """

    def __init__(self, backend: str = 'auto', verbose: bool = False):
        """
        Initialize profiler.

        Args:
            backend: Computation backend
            verbose: Whether to print profiling info
        """
        from .gpu_backend import GPUBackend
        self.gpu = GPUBackend(backend)
        self.backend_name = self.gpu.backend_name
        self.verbose = verbose

        # Timing storage
        self._timers = {}
        self._counters = {}
        self._memory_samples = []
        self._gpu_samples = []

        # Setup backend-specific profiling
        if self.backend_name == 'jax':
            self._setup_jax_profiling()
        elif self.backend_name == 'pytorch':
            self._setup_pytorch_profiling()

    def _setup_jax_profiling(self):
        """Setup JAX profiling tools"""
        try:
            import jax.profiler as jax_profiler
            self.jax_profiler = jax_profiler
        except ImportError:
            self.jax_profiler = None

    def _setup_pytorch_profiling(self):
        """Setup PyTorch profiling tools"""
        try:
            import torch.profiler as torch_profiler
            self.torch_profiler = torch_profiler
        except ImportError:
            self.torch_profiler = None

    def profile_optimization(self, method_fn: Callable, problem: Any,
                           config: Dict, warmup_runs: int = 2) -> Dict[str, Any]:
        """
        Profile complete optimization run.

        Args:
            method_fn: Optimization method to profile
            problem: Optimization problem
            config: Method configuration
            warmup_runs: Number of warmup runs before profiling

        Returns:
            Profiling results and optimization output
        """
        # Warmup runs
        if self.verbose:
            logger.info(f"Running {warmup_runs} warmup iterations...")

        for _ in range(warmup_runs):
            _ = method_fn(problem, config)
            self._clear_gpu_memory()

        # Reset counters
        self._reset_profiling()

        # Start profiling
        if self.verbose:
            logger.info("Starting profiling run...")

        profile_result = self._run_with_profiling(method_fn, problem, config)

        return profile_result

    def _run_with_profiling(self, method_fn: Callable, problem: Any, config: Dict) -> Dict[str, Any]:
        """Run optimization with detailed profiling"""
        # Start monitoring threads
        import threading
        stop_monitoring = threading.Event()

        memory_thread = threading.Thread(
            target=self._monitor_memory, args=(stop_monitoring,)
        )
        memory_thread.start()

        if self.gpu.is_gpu:
            gpu_thread = threading.Thread(
                target=self._monitor_gpu, args=(stop_monitoring,)
            )
            gpu_thread.start()

        # Profile based on backend
        if self.backend_name == 'jax' and self.jax_profiler:
            result = self._profile_jax(method_fn, problem, config)
        elif self.backend_name == 'pytorch' and self.torch_profiler:
            result = self._profile_pytorch(method_fn, problem, config)
        else:
            result = self._profile_generic(method_fn, problem, config)

        # Stop monitoring
        stop_monitoring.set()
        memory_thread.join()
        if self.gpu.is_gpu:
            gpu_thread.join()

        # Analyze results
        profile = self._analyze_profile(result)

        return {
            'optimization_result': result,
            'profile': profile,
            'detailed_timers': self._timers,
            'counters': self._counters,
            'memory_samples': self._memory_samples,
            'gpu_samples': self._gpu_samples,
        }

    def _profile_jax(self, method_fn: Callable, problem: Any, config: Dict) -> Dict[str, Any]:
        """Profile using JAX profiler"""
        import tempfile
        import os

        # Create temporary directory for trace
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "jax_trace")

            # Run with tracing
            with self.jax_profiler.trace(trace_path, create_perfetto_trace=True):
                result = self._profile_generic(method_fn, problem, config)

            # Read trace data if needed
            # (In practice, would parse the trace file for detailed analysis)

        return result

    def _profile_pytorch(self, method_fn: Callable, problem: Any, config: Dict) -> Dict[str, Any]:
        """Profile using PyTorch profiler"""
        import torch

        activities = []
        if torch.cuda.is_available():
            activities.append(self.torch_profiler.ProfilerActivity.CUDA)
        activities.append(self.torch_profiler.ProfilerActivity.CPU)

        with self.torch_profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            result = self._profile_generic(method_fn, problem, config)

        # Export profiling data
        self._pytorch_trace = prof.key_averages()

        if self.verbose:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        return result

    def _profile_generic(self, method_fn: Callable, problem: Any, config: Dict) -> Dict[str, Any]:
        """Generic profiling for any backend"""
        # Monkey-patch the method to add timing
        original_method = method_fn

        def profiled_method(problem, config):
            # Start total timer
            self._start_timer('total')

            # Initialization
            self._start_timer('initialization')
            # Setup code would go here
            self._end_timer('initialization')

            # Run the actual method
            result = original_method(problem, config)

            # End total timer
            self._end_timer('total')

            # Extract component times from result if available
            if 'metadata' in result:
                for key in ['evaluation_time', 'selection_time', 'crossover_time', 'mutation_time']:
                    if key in result['metadata']:
                        self._timers[key] = result['metadata'][key]

            return result

        return profiled_method(problem, config)

    def _monitor_memory(self, stop_event):
        """Monitor memory usage in background thread"""
        process = psutil.Process()

        while not stop_event.is_set():
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self._memory_samples.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            })
            time.sleep(0.1)  # Sample every 100ms

    def _monitor_gpu(self, stop_event):
        """Monitor GPU usage in background thread"""
        if self.backend_name == 'pytorch':
            import torch

            while not stop_event.is_set():
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_utilization = torch.cuda.utilization()

                    self._gpu_samples.append({
                        'timestamp': time.time(),
                        'memory_mb': gpu_memory,
                        'utilization': gpu_utilization,
                    })

                time.sleep(0.1)

        elif self.backend_name == 'jax':
            # Try to use nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                while not stop_event.is_set():
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    self._gpu_samples.append({
                        'timestamp': time.time(),
                        'memory_mb': mem_info.used / (1024 * 1024),
                        'utilization': util.gpu,
                    })

                    time.sleep(0.1)

            except Exception:
                pass

    def _analyze_profile(self, result: Dict[str, Any]) -> ProfileResult:
        """Analyze profiling data to create result"""
        # Calculate timing metrics
        total_runtime = self._timers.get('total', 0)
        eval_time = self._timers.get('evaluation', 0)
        sel_time = self._timers.get('selection', 0)
        cross_time = self._timers.get('crossover', 0)
        mut_time = self._timers.get('mutation', 0)

        # Calculate throughput
        n_evaluations = result.get('n_evaluations', 0)
        evaluations_per_second = n_evaluations / total_runtime if total_runtime > 0 else 0

        n_generations = result.get('metadata', {}).get('n_generations', 0)
        generations_per_second = n_generations / total_runtime if total_runtime > 0 else 0

        # Memory analysis
        memory_samples = [s['memory_mb'] for s in self._memory_samples]
        peak_memory = max(memory_samples) if memory_samples else 0
        avg_memory = np.mean(memory_samples) if memory_samples else 0

        # GPU analysis
        gpu_util = 0
        gpu_mem = 0
        if self._gpu_samples:
            gpu_utils = [s['utilization'] for s in self._gpu_samples]
            gpu_mems = [s['memory_mb'] for s in self._gpu_samples]
            gpu_util = np.mean(gpu_utils)
            gpu_mem = np.mean(gpu_mems)

        # Identify bottleneck
        component_times = {
            'evaluation': eval_time,
            'selection': sel_time,
            'crossover': cross_time,
            'mutation': mut_time,
        }

        if component_times:
            bottleneck = max(component_times, key=component_times.get)
            bottleneck_pct = (component_times[bottleneck] / total_runtime * 100) if total_runtime > 0 else 0
        else:
            bottleneck = 'unknown'
            bottleneck_pct = 0

        # Calculate efficiency metrics
        n_devices = self.gpu._device_manager.get_devices_by_type('gpu')
        parallel_efficiency = self._calculate_parallel_efficiency(result, n_devices)
        speedup_factor = self._calculate_speedup(result)

        return ProfileResult(
            total_runtime=total_runtime,
            initialization_time=self._timers.get('initialization', 0),
            evaluation_time=eval_time,
            selection_time=sel_time,
            crossover_time=cross_time,
            mutation_time=mut_time,
            communication_time=self._timers.get('communication', 0),
            evaluations_per_second=evaluations_per_second,
            generations_per_second=generations_per_second,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            memory_efficiency=self._calculate_memory_efficiency(peak_memory),
            gpu_utilization=gpu_util,
            gpu_memory_mb=gpu_mem,
            gpu_compute_efficiency=gpu_util / 100.0 if gpu_util > 0 else 0,
            primary_bottleneck=bottleneck,
            bottleneck_percentage=bottleneck_pct,
            parallel_efficiency=parallel_efficiency,
            speedup_factor=speedup_factor,
            backend=self.backend_name,
            n_devices=len(n_devices) if n_devices else 1,
            problem_dimension=result.get('problem_dimension', 0),
            population_size=result.get('population_size', 0)
        )

    def compare_methods(self, methods: List[Callable], problem: Any,
                       config: Dict) -> Dict[str, Any]:
        """
        Compare performance of multiple methods.

        Args:
            methods: List of optimization methods
            problem: Problem to optimize
            config: Configuration for methods

        Returns:
            Comparative analysis results
        """
        results = {}

        for method in methods:
            method_name = method.__name__ if hasattr(method, '__name__') else str(method)

            if self.verbose:
                logger.info(f"Profiling {method_name}...")

            profile_result = self.profile_optimization(method, problem, config)
            results[method_name] = profile_result

        # Comparative analysis
        comparison = self._compare_profiles(results)

        return {
            'individual_results': results,
            'comparison': comparison,
            'best_method': comparison['best_method'],
        }

    def _compare_profiles(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare profiling results from multiple methods"""
        comparison = {
            'runtime': {},
            'memory': {},
            'throughput': {},
            'efficiency': {},
        }

        best_runtime = float('inf')
        best_method = None

        for method_name, result in results.items():
            profile = result['profile']

            comparison['runtime'][method_name] = profile.total_runtime
            comparison['memory'][method_name] = profile.peak_memory_mb
            comparison['throughput'][method_name] = profile.evaluations_per_second
            comparison['efficiency'][method_name] = profile.parallel_efficiency

            if profile.total_runtime < best_runtime:
                best_runtime = profile.total_runtime
                best_method = method_name

        comparison['best_method'] = best_method
        comparison['speedup_vs_best'] = {
            method: comparison['runtime'][method] / best_runtime
            for method in comparison['runtime']
        }

        return comparison

    def generate_report(self, profile_result: Dict[str, Any],
                       output_path: Optional[str] = None) -> str:
        """
        Generate detailed profiling report.

        Args:
            profile_result: Profiling results
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        profile = profile_result['profile']

        report = f"""
        ============================================
        Librex PERFORMANCE PROFILING REPORT
        ============================================

        Timestamp: {profile.timestamp}
        Backend: {profile.backend}
        Devices: {profile.n_devices}

        TIMING BREAKDOWN
        ----------------
        Total Runtime: {profile.total_runtime:.2f}s
        - Initialization: {profile.initialization_time:.2f}s
        - Evaluation: {profile.evaluation_time:.2f}s
        - Selection: {profile.selection_time:.2f}s
        - Crossover: {profile.crossover_time:.2f}s
        - Mutation: {profile.mutation_time:.2f}s
        - Communication: {profile.communication_time:.2f}s

        THROUGHPUT
        ----------
        Evaluations/sec: {profile.evaluations_per_second:.0f}
        Generations/sec: {profile.generations_per_second:.2f}

        MEMORY USAGE
        ------------
        Peak Memory: {profile.peak_memory_mb:.0f} MB
        Average Memory: {profile.avg_memory_mb:.0f} MB
        Memory Efficiency: {profile.memory_efficiency:.1%}

        GPU UTILIZATION
        ---------------
        GPU Utilization: {profile.gpu_utilization:.1f}%
        GPU Memory: {profile.gpu_memory_mb:.0f} MB
        Compute Efficiency: {profile.gpu_compute_efficiency:.1%}

        BOTTLENECK ANALYSIS
        -------------------
        Primary Bottleneck: {profile.primary_bottleneck}
        Bottleneck Impact: {profile.bottleneck_percentage:.1f}%

        PARALLEL PERFORMANCE
        --------------------
        Parallel Efficiency: {profile.parallel_efficiency:.1%}
        Speedup Factor: {profile.speedup_factor:.2f}x

        RECOMMENDATIONS
        ---------------
        """

        # Add recommendations based on bottleneck
        if profile.primary_bottleneck == 'evaluation':
            report += "- Consider GPU acceleration or parallel evaluation\n"
            report += "- Implement caching for repeated evaluations\n"
        elif profile.primary_bottleneck == 'selection':
            report += "- Optimize selection algorithm\n"
            report += "- Consider vectorized selection operations\n"
        elif profile.gpu_utilization < 50:
            report += "- GPU underutilized - increase batch size\n"
            report += "- Consider using multiple GPU streams\n"

        if profile.memory_efficiency < 0.5:
            report += "- High memory usage - implement chunking\n"
            report += "- Enable gradient checkpointing if applicable\n"

        report += "\n============================================\n"

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

            # Also save JSON data
            json_path = output_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'profile': profile.__dict__,
                    'timers': profile_result['detailed_timers'],
                    'counters': profile_result['counters'],
                }, f, indent=2)

        return report

    def _start_timer(self, name: str):
        """Start a named timer"""
        self._timers[f"{name}_start"] = time.time()

    def _end_timer(self, name: str):
        """End a named timer"""
        if f"{name}_start" in self._timers:
            self._timers[name] = time.time() - self._timers[f"{name}_start"]
            del self._timers[f"{name}_start"]

    def _reset_profiling(self):
        """Reset all profiling data"""
        self._timers.clear()
        self._counters.clear()
        self._memory_samples.clear()
        self._gpu_samples.clear()

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        self.gpu.clear_memory()

    def _calculate_parallel_efficiency(self, result: Dict, n_devices: List) -> float:
        """Calculate parallel efficiency"""
        if not n_devices or len(n_devices) <= 1:
            return 1.0

        # Estimate based on speedup
        speedup = result.get('speedup', 1.0)
        return speedup / len(n_devices)

    def _calculate_speedup(self, result: Dict) -> float:
        """Calculate speedup vs sequential"""
        # Would need baseline sequential time
        # For now, estimate based on parallelism
        return result.get('speedup', 1.0)

    def _calculate_memory_efficiency(self, peak_memory: float) -> float:
        """Calculate memory efficiency"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        return 1.0 - (peak_memory / total_memory) if total_memory > 0 else 0