# Physics Code Review Prompt

> **Review code with physics correctness as priority**

## Metadata

```yaml
name: physics-code-review
version: 1.0.0
tags: [physics, review, validation, correctness]
tools: [amazon-q, claude, windsurf, cline]
domain: computational-physics
```

## Prompt

```markdown
Review this physics code for correctness and best practices.

## Code
[Paste code here]

## Review Checklist

### Physics Correctness
- [ ] Conservation laws respected (energy, momentum, charge)
- [ ] Units are consistent throughout
- [ ] Boundary conditions properly implemented
- [ ] Physical constraints enforced
- [ ] Symmetries preserved

### Numerical Stability
- [ ] No overflow/underflow risks
- [ ] Stable algorithms used
- [ ] Convergence criteria appropriate
- [ ] Error accumulation minimized

### Performance
- [ ] Vectorized operations
- [ ] Minimal memory allocations
- [ ] Appropriate data structures
- [ ] GPU-friendly if applicable

### Code Quality
- [ ] Clear variable names (physics notation)
- [ ] Docstrings explain physics
- [ ] Type hints present
- [ ] Tests cover edge cases

## Output Format

Provide:
1. Physics issues (if any)
2. Numerical stability concerns
3. Performance suggestions
4. Code quality improvements
```

## Usage

```
@prompt physics-code-review

File: librex/equilibria/algorithms/gradient_descent.py
```
