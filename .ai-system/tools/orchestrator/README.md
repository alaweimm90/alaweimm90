# Workflow Orchestrator

DAG-based workflow execution with parallel processing and dependency management.

## Features

- **DAG Execution** - Dependency-aware step execution
- **Parallel Processing** - Run independent steps concurrently
- **Error Handling** - Graceful failure handling
- **Context Variables** - Pass data between steps
- **Cycle Detection** - Validates workflow before execution

## Usage

### Basic Execution
```bash
python tools/orchestrator/engine.py tools/orchestrator/workflows/example-simple.yaml
```

### With Context Variables
```bash
python tools/orchestrator/engine.py \
  tools/orchestrator/workflows/development-cycle.yaml \
  --context target=librex/equilibria \
  --context env=staging
```

### Control Parallelism
```bash
python tools/orchestrator/engine.py \
  workflow.yaml \
  --parallel 8
```

## Workflow Format

```yaml
name: workflow-name
description: Workflow description

steps:
  - id: step1
    run: command to execute
    
  - id: step2
    run: command with ${context_var}
    depends_on: [step1]
    
  - id: step3
    run: another command
    depends_on: [step1, step2]
```

## Examples

### Simple Linear Workflow
```yaml
steps:
  - id: build
    run: make build
  - id: test
    run: make test
    depends_on: [build]
  - id: deploy
    run: make deploy
    depends_on: [test]
```

### Parallel Testing
```yaml
steps:
  - id: lint
    run: ruff check .
    
  - id: unit-tests
    run: pytest tests/unit
    depends_on: [lint]
    
  - id: integration-tests
    run: pytest tests/integration
    depends_on: [lint]
    
  - id: report
    run: generate-report
    depends_on: [unit-tests, integration-tests]
```

### Context Variables
```yaml
steps:
  - id: test
    run: pytest ${target}
  - id: benchmark
    run: python benchmark.py --target ${target}
    depends_on: [test]
```

## Execution Flow

1. **Load** - Parse workflow YAML
2. **Validate** - Check for cycles
3. **Execute** - Run steps respecting dependencies
4. **Parallel** - Execute independent steps concurrently
5. **Report** - Show results

## Error Handling

- Failed steps stop dependent steps
- Independent branches continue
- Timeout after 5 minutes per step
- Detailed error reporting

## Next Steps

- Add retry logic
- Support conditional execution
- Add step artifacts
- Implement caching
