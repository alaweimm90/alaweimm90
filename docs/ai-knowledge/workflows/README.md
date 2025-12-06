# Workflows Directory

Automated workflows and scripts for common development tasks.

## Categories

### 💻 [development/](development/)
Development workflows:
- `test-driven-refactor.py` - TDD refactoring with validation
- `benchmark-driven-optimization.py` - Optimize based on benchmarks
- `physics-validation.py` - Validate physics correctness

### 🧪 [testing/](testing/)
Testing workflows:
- `property-based-testing.py` - Generate property-based tests
- `regression-suite.py` - Build regression test suite

### 🚀 [deployment/](deployment/)
Deployment workflows:
- `pre-deploy-checks.py` - Run checks before deployment
- `performance-validation.py` - Validate performance metrics

### 🔬 [research/](research/)
Research workflows:
- `experiment-design.py` - Design computational experiments
- `literature-review.py` - Automated literature review

## Usage

### Run a Workflow
```bash
python .ai-knowledge/workflows/development/test-driven-refactor.py \
  --target librex/equilibria/algorithms/gradient_descent.py
```

### Create New Workflow
```bash
python .ai-knowledge/tools/create-workflow.py
```

## Workflow Structure

All workflows follow this pattern:
```python
#!/usr/bin/env python3
"""
Workflow Name

Description of what it does.

Usage:
    python workflow.py --target <path>
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    
    # Workflow logic here
    
if __name__ == "__main__":
    main()
```

## Best Practices

1. **Idempotent**: Can run multiple times safely
2. **Validated**: Check inputs and outputs
3. **Logged**: Print progress and results
4. **Documented**: Clear usage instructions
5. **Tested**: Include test cases

## Contributing

When adding a workflow:
1. Use clear naming: `verb-noun.py`
2. Add docstring with usage
3. Include argument parsing
4. Update this README
5. Run: `python ../tools/update-catalog.py`
