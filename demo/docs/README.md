# ATLAS Comprehensive Demo

This directory contains a complete demonstration environment for the ATLAS system, showcasing its capabilities across agent orchestration, code analysis, refactoring, CLI operations, and KILO integration.

## Overview

The ATLAS demo validates the entire system end-to-end by running comprehensive tests on code quality issues, measuring performance, and demonstrating governance workflows.

## Directory Structure

```
demo/
├── test-repos/           # Test repositories with code quality issues
│   ├── messy-python/     # Python code with poor practices
│   ├── complex-js/       # JavaScript with callback hell
│   └── spaghetti-ts/     # TypeScript with mixed concerns
├── scenarios/            # Specific demo scenarios
│   ├── end-to-end-workflow/    # Complete analysis → optimization cycle
│   ├── cli-demo/               # CLI command demonstrations
│   └── kilo-integration/       # Governance workflow showcase
├── scripts/              # Automation scripts
│   ├── run-demo.sh       # Main demo runner (Bash)
│   ├── run-demo.ps1      # Main demo runner (PowerShell)
│   ├── benchmark.sh      # Performance benchmarking
│   ├── validation-tests.js    # System validation tests
│   ├── validate-agents.cjs    # Agent validation
│   └── logger.js         # Comprehensive logging system
├── dashboards/           # Visual dashboards
│   └── generate-dashboard.js  # Dashboard generator
├── logs/                 # Operation logs and metrics
└── docs/                 # Documentation
```

## Quick Start

### Prerequisites

- Node.js 20+
- ATLAS CLI installed (`npm run atlas` should work)
- Bash or PowerShell for running scripts

### Running the Full Demo

**Option 1: Bash (Linux/Mac)**

```bash
cd demo/scripts
chmod +x run-demo.sh
./run-demo.sh
```

**Option 2: PowerShell (Windows)**

```powershell
cd demo/scripts
.\run-demo.ps1
```

### Running Individual Components

**Agent Validation:**

```bash
node demo/scripts/validate-agents.cjs
```

**Performance Benchmarking:**

```bash
cd demo/scripts
chmod +x benchmark.sh
./benchmark.sh
```

**Validation Tests:**

```bash
node demo/scripts/validation-tests.js
```

**Generate Dashboard:**

```bash
node demo/dashboards/generate-dashboard.js demo/logs/metrics_*.json
```

## Demo Scenarios

### 1. End-to-End Workflow

**File:** `demo/scenarios/end-to-end-workflow/run-scenario.sh`

Demonstrates the complete ATLAS workflow:

1. Initial code analysis
2. Metric collection
3. Optimization recommendations
4. Post-optimization validation
5. Before/after comparisons

### 2. CLI Operations Demo

**File:** `demo/scenarios/cli-demo/run-scenario.sh`

Showcases all major ATLAS CLI commands:

- Repository analysis with different formats
- Complexity analysis
- Chaos analysis
- Quick scanning

### 3. KILO Integration

**File:** `demo/scenarios/kilo-integration/`

Demonstrates governance and compliance checking workflows.

## Test Repositories

### Messy Python (`demo/test-repos/messy-python/`)

- **Issues:** Global variables, long functions, poor separation of concerns
- **ATLAS Focus:** Complexity analysis, refactoring suggestions

### Complex JavaScript (`demo/test-repos/complex-js/`)

- **Issues:** Deep callback nesting, complex async patterns
- **ATLAS Focus:** Chaos analysis, maintainability metrics

### Spaghetti TypeScript (`demo/test-repos/spaghetti-ts/`)

- **Issues:** Mixed concerns, `any` types, large classes
- **ATLAS Focus:** Type safety, architectural analysis

## Validation Checkpoints

The demo validates these key ATLAS capabilities:

✅ **Agent Registration & Capability Mapping**

- Agent discovery and registration
- Capability mapping and routing
- Load balancing and health monitoring

✅ **Code Analysis Accuracy**

- Complexity metric calculation
- Chaos level assessment
- Maintainability scoring

✅ **CLI Completeness**

- All major commands functional
- Multiple output formats supported
- Error handling and validation

✅ **KILO Integration**

- Governance policy enforcement
- Compliance checking
- Audit trail generation

✅ **Safety & Performance**

- All transformations validated as safe
- Performance benchmarks within acceptable ranges
- Comprehensive error handling

## Output Files

After running the demo, these files are generated:

### Logs (`demo/logs/`)

- `demo_*.log` - Main demo execution log
- `*_analysis_*.json` - Repository analysis results
- `*_complexity_*.txt` - Complexity analysis output
- `*_chaos_*.txt` - Chaos analysis results
- `benchmark_*.txt` - Performance metrics
- `validation-report-*.json` - System validation results

### Dashboards (`demo/dashboards/`)

- `dashboard_*.html` - Interactive HTML dashboard
- `metrics_*.json` - Raw metrics data

## Understanding Results

### Complexity Scores

- **0-2:** Good (maintainable)
- **2-5:** Warning (needs attention)
- **5+:** Bad (refactoring required)

### Chaos Levels

- **0-2:** Low chaos (well-structured)
- **2-5:** Moderate chaos (some issues)
- **5+:** High chaos (major refactoring needed)

### Maintainability Index

- **8+:** Good maintainability
- **6-8:** Moderate maintainability
- **<6:** Poor maintainability

## Troubleshooting

### Common Issues

**ATLAS CLI not found:**

```bash
npm install
npm run atlas --version
```

**Permission denied on scripts:**

```bash
chmod +x demo/scripts/*.sh
```

**Dashboard not loading:**

- Ensure modern browser with JavaScript enabled
- Check console for Chart.js loading errors

**Analysis fails:**

- Verify test repositories exist
- Check file permissions
- Ensure Node.js modules are installed

### Performance Considerations

- First run may be slower due to cold starts
- Large repositories may take longer to analyze
- Benchmark results vary by system performance

## Extending the Demo

### Adding New Test Cases

1. Create new directory in `demo/test-repos/`
2. Add problematic code files
3. Update validation scripts if needed
4. Run demo to see ATLAS analysis

### Custom Scenarios

1. Create new directory in `demo/scenarios/`
2. Add scenario-specific scripts
3. Update main demo runner
4. Document the new scenario

### Custom Dashboards

1. Modify `demo/dashboards/generate-dashboard.js`
2. Add new chart types or metrics
3. Update HTML template
4. Test with sample data

## Integration with CI/CD

The demo can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run ATLAS Demo
  run: |
    cd demo/scripts
    ./run-demo.sh

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: atlas-results
    path: demo/logs/
```

## Contributing

To contribute to the ATLAS demo:

1. Follow the existing code patterns
2. Add comprehensive logging
3. Include validation tests
4. Update documentation
5. Test on multiple platforms

## Support

For issues with the ATLAS demo:

1. Check the logs in `demo/logs/`
2. Run validation tests: `node demo/scripts/validation-tests.js`
3. Verify ATLAS installation: `npm run atlas --version`
4. Check system requirements

## License

This demo is part of the ATLAS system and follows the same license terms.
