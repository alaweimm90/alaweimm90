"""
Quick benchmark for ASlib scenario loading performance.

Measures parsing time and throughput for features, runs, and training data.

Usage:
  python -m benchmark.bench_arff_loading [scenario_path]

Defaults to GRAPHS-2015 if no path is provided.
"""

import sys
import time
from pathlib import Path

from benchmark.aslib_parser import ASLibScenarioLoader


def main():
    args = sys.argv[1:]
    run_all = any(a in ('--all', '-a', 'all') for a in args)
    use_arrays = any(a in ('--arrays', '--vectorized') for a in args)
    path_args = [a for a in args if not a.startswith('-') and a != 'all']
    scenario_path = (
        Path(path_args[0]) if path_args
        else Path(__file__).resolve().parents[1] / "aslib_data" / "GRAPHS-2015"
    )

    t0 = time.perf_counter()
    loader = ASLibScenarioLoader(str(scenario_path))
    t_load = time.perf_counter() - t0

    summary = loader.get_summary()
    n_instances = summary.get("n_instances", 0)

    # Measure training data extraction for a representative slice
    slice_size = n_instances if run_all else min(1000, n_instances)
    t1 = time.perf_counter()
    if use_arrays:
        _ = loader.get_training_arrays(max_instances=slice_size)
    else:
        _ = loader.get_training_data(max_instances=slice_size)
    t_train = time.perf_counter() - t1

    print("ASlib Loading Benchmark")
    print(f"Scenario: {summary.get('name')}")
    print(f"Instances: {n_instances}")
    print(f"Algorithms: {summary.get('n_algorithms')}")
    print(f"Features per instance: {summary.get('n_features')}")
    print("---")
    print(f"Load time: {t_load:.3f}s")
    if t_load > 0:
        print(f"  Instances/s (load): {n_instances / t_load:.1f}")
    kind = 'arrays' if use_arrays else 'dicts'
    print(f"Training-data ({kind}) (first {slice_size}) time: {t_train:.3f}s")
    if t_train > 0:
        print(f"  Instances/s (train slice): {slice_size / t_train:.1f}")


if __name__ == "__main__":
    main()
