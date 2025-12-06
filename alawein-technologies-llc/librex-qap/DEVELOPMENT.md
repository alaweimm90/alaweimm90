# Librex.QAP-new Development Guide

This guide explains how to actively develop, test, and document Librex.QAP-new.

## Project Structure

```
Librex.QAP-new/
├── Librex.QAP/          # Optimization library - Core development
├── ORCHEX/              # Autonomous research system - Core development
├── tests/              # Test suite - Run frequently during development
├── examples/           # Usage examples - Keep updated with changes
├── data/               # Benchmark data - Reference only
├── docs/               # ACTIVE DEVELOPMENT DOCS (this stays lean & current)
│   ├── development/    # Development notes & architecture
│   └── guides/         # How-to guides for features
├── .archive/           # Historical docs & results (reference only)
├── DEVELOPMENT.md      # This file
├── CHANGELOG.md        # Track changes as you make them
└── CONTRIBUTING.md     # Guidelines for contributors
```

## Development Workflow

### 1. Feature Branch Workflow

```bash
# Always work on a feature branch (if needed)
git checkout -b feature/your-feature-name

# Make changes to Librex.QAP/ or ORCHEX/
# Write tests in tests/
# Update inline documentation

# Before committing
make test          # Run all tests
make lint          # Check code quality
make format        # Auto-format code

# Commit with clear messages
git commit -m "Add feature: description"

# Push to remote
git push origin feature/your-feature-name
```

### 2. During Development

**Keep these updated in real-time:**

1. **Code Comments** - Explain the "why" not "what"
   ```python
   # ❌ Bad
   x = y + 1  # Add 1 to y

   # ✅ Good
   # Increment iteration counter for next optimization step
   iteration_count = current_iteration + 1
   ```

2. **Docstrings** - Document functions/classes
   ```python
   def optimize_qap(problem, method='fft_laplace'):
       """Solve QAP using specified optimization method.

       Args:
           problem: QAP problem instance
           method: Optimization method ('fft_laplace', 'reverse_time', etc.)

       Returns:
           OptimizationResult with best_solution and objective_value
       """
   ```

3. **CHANGELOG.md** - Track changes
   ```markdown
   ## [Unreleased]

   ### Added
   - New optimization method: Quantum-inspired algorithm
   - Integration test suite for ORCHEX + Librex.QAP

   ### Fixed
   - Bug in pipeline dispatcher (issue #42)
   - Memory leak in visualization module

   ### Changed
   - Refactored optimization core for better modularity
   ```

4. **docs/development/** - Architecture notes
   - Current design decisions
   - Known limitations
   - Performance considerations
   - Integration points between systems

### 3. Testing During Development

```bash
# Run specific test file
pytest tests/test_methods.py -v

# Run tests with coverage
pytest --cov=Librex.QAP --cov=ORCHEX tests/

# Run integration tests
pytest tests/test_integration.py -v

# Run all tests before committing
make test
```

### 4. Documentation During Development

**For new features:**

1. Create/update docstrings in code
2. Add inline comments explaining complex logic
3. If significant: create `docs/guides/your-feature.md`
4. Update `CHANGELOG.md` with your changes
5. Update `README.md` if feature is user-facing

**For bug fixes:**

1. Document what was broken
2. Explain the fix
3. Link to test case that validates it
4. Update `CHANGELOG.md`

## Quick Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
make test

# Run with coverage
make coverage

# Format code
make format

# Lint code
make lint

# Run all checks
make check-all

# Build documentation
make docs

# Clean generated files
make clean
```

## Key Development Areas

### Librex.QAP Development

**Location:** `Librex.QAP/`

**Main Components:**
- `core/pipeline.py` - Core optimization pipeline (most critical)
- `methods/novel.py` - Novel optimization methods
- `methods/baselines.py` - Baseline algorithms
- `utils.py` - Core utilities

**When adding new method:**
1. Add implementation to `methods/novel.py` or `methods/baselines.py`
2. Add metadata to `methods/metadata.py`
3. Create test in `tests/test_methods.py`
4. Add example to `examples/05_optimization.py`
5. Document in docstring and `docs/guides/`

### ORCHEX Development

**Location:** `ORCHEX/`

**Main Components:**
- `ORCHEX/brainstorming/` - Hypothesis generation
- `ORCHEX/experimentation/` - Experiment design & execution
- `ORCHEX/learning/` - Learning mechanisms
- `ORCHEX/orchestration/` - Workflow orchestration
- `uaro/` - Universal solver integration

**When adding new capability:**
1. Add implementation to appropriate module
2. Create test in `tests/test_integration.py` if cross-system
3. Update `ORCHEX/protocol.py` if API changes
4. Document in `docs/guides/`

## Code Quality Standards

### 1. Type Hints
```python
# Use type hints for clarity
def optimize(problem: QAPProblem, iterations: int) -> OptimizationResult:
    pass
```

### 2. Error Handling
```python
# Handle errors gracefully
try:
    result = method.optimize(problem)
except ConvergenceError as e:
    logger.warning(f"Method didn't converge: {e}")
    result = fallback_method(problem)
```

### 3. Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Starting optimization with {len(methods)} methods")
logger.debug(f"Iteration {i}: objective={objective}")
logger.warning("Large problem may take time")
logger.error("Failed to load benchmark data")
```

### 4. Testing
- Write tests for new code
- Aim for >80% coverage of critical modules
- Test both happy path and edge cases

## Documentation Best Practices

### Code Documentation
- **Module level:** Explain what the module does
- **Class level:** Explain purpose and usage
- **Function level:** Explain parameters, returns, raises
- **Inline:** Explain "why" for complex logic

### Development Documentation
- Keep in `docs/development/` or `docs/guides/`
- Link from main README when relevant
- Include examples where possible
- Update when architecture changes

### External Documentation
- For public API: docstrings + examples
- For algorithms: mathematical notation + references
- For integration: architecture diagrams (if helpful)

## Performance Monitoring

```python
# Use timing utilities for critical sections
from Librex.QAP.performance_utils import timer

with timer("optimization"):
    result = method.optimize(problem)

# Log performance metrics
logger.info(f"Optimization completed in {elapsed:.2f}s")
```

## Integration Between Systems

### Librex.QAP + ORCHEX

**When ORCHEX validates Librex.QAP methods:**
```python
# ORCHEX can use Librex.QAP optimization
from Librex.QAP.core import OptimizationPipeline

pipeline = OptimizationPipeline(problem_size=20)
results = pipeline.solve(method="fft_laplace")

# ORCHEX validates results
validation_score = atlas_agents.validate(results)
```

**When Librex.QAP uses ORCHEX learning:**
```python
# Librex.QAP can use ORCHEX's Hall of Failures
from ORCHEX.learning import HallOfFailures

failures = HallOfFailures()
past_failures = failures.get_similar(current_problem)

# Adjust strategy based on past failures
```

## Common Development Tasks

### Adding a New Optimization Method

1. Implement in `Librex.QAP/methods/novel.py` or `baselines.py`
2. Add tests in `tests/test_methods.py`
3. Create example in `examples/05_optimization.py`
4. Document:
   - Docstring with algorithm explanation
   - `docs/guides/method-name.md` with algorithm details
   - Update `CHANGELOG.md`

### Fixing a Bug

1. Create test that reproduces bug
2. Fix the bug
3. Verify test passes
4. Update `CHANGELOG.md`
5. Commit with message: "Fix: description (closes #issue-number)"

### Improving Documentation

1. Update files in `docs/development/` or `docs/guides/`
2. Update inline code comments/docstrings
3. Update `CHANGELOG.md`
4. No need to rebuild, changes are immediately visible

### Performance Optimization

1. Profile current code: `make profile`
2. Identify bottleneck
3. Implement optimization
4. Benchmark improvement: `make benchmark`
5. Update `CHANGELOG.md` with performance gains

## Tips for Success

✅ **DO:**
- Write tests before/with code
- Document as you go (not after)
- Keep commits small and focused
- Update CHANGELOG.md
- Run tests frequently
- Use meaningful variable names

❌ **DON'T:**
- Leave code without docstrings
- Commit without running tests
- Make huge commits mixing features
- Forget to update CHANGELOG.md
- Leave debug prints in code
- Make breaking changes without discussion

## Getting Help

- Check `.archive/docs/` for historical reference
- Review examples in `examples/`
- Read docstrings in code
- Check CHANGELOG.md for recent changes
- Ask questions in code comments or issues

## Next Steps

1. Pick a feature/bug to work on
2. Create feature branch if needed
3. Make changes with tests
4. Document as you go
5. Run full test suite: `make test`
6. Commit and push
7. The work is live on the development branch!

---

**Happy coding! The structure is here to support active development while keeping things organized.** 🚀
