# Librex Benchmarking System

## Overview

The Librex benchmarking system provides comprehensive performance testing and tracking for all optimization algorithms. It includes automated nightly benchmarks, performance regression detection, and detailed reporting.

## Features

- **Multiple Benchmark Suites**: Smoke tests (5 min), Standard (30 min), Comprehensive (2 hours), and Method Comparison
- **Performance Metrics**: Solution quality, runtime, memory usage, convergence speed, success rate
- **Historical Tracking**: SQLite database for tracking performance over time
- **Regression Detection**: Automatic detection of performance degradations
- **Report Generation**: HTML dashboards, Markdown summaries, and CSV exports
- **CI/CD Integration**: GitHub Actions workflow for automated nightly runs

## Quick Start

### Running Benchmarks Locally

```bash
# Run smoke tests (quick validation, ~5 minutes)
python nightly_benchmark.py --suite smoke

# Run standard benchmark (~30 minutes)
python nightly_benchmark.py --suite standard

# Run comprehensive benchmark (~2 hours)
python nightly_benchmark.py --suite comprehensive

# Run method comparison
python nightly_benchmark.py --suite method_comparison

# Run all enabled suites
python nightly_benchmark.py --all
```

### Testing the System

```bash
# Run system tests
python test_benchmark_system.py
```

## Configuration

Edit `benchmark_config.yaml` to customize:
- Problem types and sizes
- Methods and their parameters
- Metrics to track
- Regression thresholds
- Output formats

## Benchmark Suites

### Smoke Test Suite (~5 minutes)
- Quick validation of core functionality
- Small problem instances (n ≤ 15)
- Limited iterations
- 2 runs per method

### Standard Benchmark (~30 minutes)
- Regular performance testing
- Medium problem instances (n ≤ 50)
- Standard parameters
- 5 runs per method
- Includes QAPLIB instances

### Comprehensive Benchmark (~2 hours)
- Full performance analysis
- Large problem instances (n ≤ 100)
- Parameter sweeps
- 10 runs per method
- Statistical significance testing

### Method Comparison
- Head-to-head comparison
- Standardized parameters
- Statistical significance testing
- QAPLIB benchmark instances

## Performance Metrics

1. **Solution Quality**: Objective value of best solution found
2. **Optimality Gap**: Percentage gap from known optimal (when available)
3. **Runtime**: Total execution time
4. **Memory Usage**: Peak memory consumption
5. **Convergence Speed**: Iterations to reach 90% of final quality
6. **Success Rate**: Percentage of successful runs
7. **Stability**: Standard deviation across runs

## Output Files

Results are saved in `benchmark_results/`:
- `{suite}_{timestamp}.json` - Raw benchmark data
- `{suite}_{timestamp}.html` - Interactive HTML dashboard
- `{suite}_{timestamp}.md` - Markdown summary for GitHub
- `{suite}_{timestamp}.csv` - CSV export for analysis
- `{suite}_latest.json` - Symlink to most recent results
- `history.db` - SQLite database for historical tracking

## HTML Dashboard

The HTML dashboard includes:
- Performance summary cards
- Method comparison charts
- Scalability analysis
- Memory usage distribution
- Performance heatmaps
- Detailed results tables

## Performance Tracking

### Recording History

```python
from benchmark_history import BenchmarkHistory

history = BenchmarkHistory()

# Record a benchmark run
with open('benchmark_results/standard_latest.json', 'r') as f:
    results = json.load(f)
run_id = history.record_run(results)

# Update baselines
history.update_baselines('standard', window_days=7)

# Detect regressions
regressions = history.detect_regressions(run_id)
```

### Analyzing Trends

```python
# Get performance trends
trends = history.get_trend_analysis(
    suite_name='standard',
    method='genetic_algorithm',
    problem_key='qap_chr12a',
    days=30
)

# Generate comparison matrix
comparison = history.generate_comparison_matrix('standard', 'qap')

# Export to CSV
history.export_to_csv(Path('./exports'))
```

## CI/CD Integration

The GitHub Actions workflow (`Librex-nightly-benchmark.yml`) provides:

1. **Nightly Runs**: Automated benchmarks at 2 AM UTC
2. **PR Testing**: Smoke tests on pull requests
3. **Regression Alerts**: Issues created for critical regressions
4. **Performance Comments**: Results posted to PRs
5. **Artifact Storage**: Results stored for 90 days

### Manual Workflow Dispatch

You can manually trigger benchmarks from GitHub Actions:
1. Go to Actions tab
2. Select "Librex Nightly Benchmark"
3. Click "Run workflow"
4. Choose suite: smoke, standard, comprehensive, or all

## Regression Detection

Regressions are detected when:
- Solution quality degrades by >5%
- Runtime increases by >20%
- Memory usage increases by >30%

Alerts are categorized as:
- **Warning**: Single threshold exceeded
- **Critical**: Double threshold exceeded

## Example Usage

```python
from benchmarks.nightly_benchmark import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner('benchmark_config.yaml')

# Run a specific suite
results = runner.run_suite('standard')

# Save results with reports
runner.save_results(results, 'standard')
```

## Adding New Benchmarks

1. **Add Problem Type**: Extend `_generate_problem()` in `nightly_benchmark.py`
2. **Add Method**: Include in `benchmark_config.yaml`
3. **Add Metrics**: Extend `_benchmark_single_method()` for new metrics
4. **Add Reports**: Extend report generators for custom visualizations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Librex is installed: `pip install -e .`
2. **Memory Issues**: Reduce problem sizes or iterations in config
3. **Timeout Issues**: Adjust `max_duration` in suite configuration
4. **Database Errors**: Delete `history.db` to reset tracking

### Debug Mode

```bash
# Run with verbose output
python nightly_benchmark.py --suite smoke --verbose

# Test specific components
python -c "from benchmark_history import BenchmarkHistory; BenchmarkHistory().validate_installation()"
```

## Architecture

```
benchmarks/
├── nightly_benchmark.py      # Main benchmark runner
├── benchmark_config.yaml     # Configuration file
├── benchmark_history.py      # Performance tracking
├── report_generator.py       # Report generation
├── test_benchmark_system.py  # System tests
├── benchmark_results/        # Output directory
│   ├── *.json               # Raw results
│   ├── *.html               # HTML reports
│   ├── *.md                 # Markdown summaries
│   ├── *.csv                # CSV exports
│   └── history.db           # Performance database
└── README.md                # This file
```

## Contributing

1. Add tests for new features
2. Update configuration documentation
3. Ensure backward compatibility
4. Test with all benchmark suites
5. Update this README

## License

Apache 2.0 - See LICENSE file