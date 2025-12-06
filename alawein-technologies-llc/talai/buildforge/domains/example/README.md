# Example Domain Configuration

This is a template domain configuration for BuildForge.

## Setup

1. **Copy this folder**:
   ```bash
   cp -r domains/example domains/your_project_name
   ```

2. **Edit `config.yaml`**:
   - Update `domain` to your research area
   - Add your `keywords`
   - Set `novelty_threshold` (recommended: 65)
   - Add SOTA baselines from literature
   - Define benchmarks and metrics
   - Set release criteria

3. **Run BuildForge**:
   ```bash
   python buildforge.py run-gates --config domains/your_project_name/config.yaml
   ```

## Configuration Fields

### Required Fields

- `domain`: Research domain name
- `keywords`: List of keywords for literature search
- `novelty_threshold`: Minimum novelty score to proceed (0-100)
- `sota_baselines`: Existing methods to compare against
- `benchmarks`: Benchmark datasets and instances
- `evaluation_metrics`: Metrics to measure performance
- `release_criteria`: Conditions that must be met to deploy

### Optional Fields

- `price_point`: Pricing if building a product
- `target_users`: Who will use this
- `timeline`: Estimated timeline per gate
- `budget`: Budget estimates
- `deployment`: Deployment configuration

## Example: QAP Solver

```yaml
domain: "Quadratic Assignment Problem Optimization"
keywords:
  - "quadratic assignment"
  - "combinatorial optimization"
  - "metaheuristic"

novelty_threshold: 70

sota_baselines:
  - name: "Genetic Algorithm (GA)"
    year: 2020
    metric_value: 12.5  # gap from optimal
    runtime_sec: 120

benchmarks:
  dataset: "QAPLIB"
  source_url: "https://qaplib.org"
  instances:
    - "chr12a"
    - "chr15a"
    - "chr20a"
```

## Next Steps

After configuration:

1. **Gate 1** (2 hours): Novelty assessment
2. **Gate 2** (1 week): Theory validation
3. **Gate 3** (1 week): Feasibility analysis
4. **Gate 4** (8 weeks): Build MVP + benchmark
5. **Gate 5** (1 week): Write paper + deploy

Total: ~11 weeks from idea to deployed product + published paper.
