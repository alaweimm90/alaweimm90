# Development Documentation

Architecture notes, design decisions, and development insights for Librex.QAP-new.

## What Is This Directory?

**docs/development/** contains active development notes, architecture documentation, and design decision records that are created and updated during development.

## Contents

This directory is for:
- Architecture diagrams and notes
- Design decision records
- Component interaction notes
- Performance analysis
- Technical investigations
- Development insights

## Adding Development Notes

When documenting development decisions:

1. **Create file:** `[topic].md`
2. **Include:**
   - What decision was made
   - Why it was made
   - Alternatives considered
   - Impact & implications

3. **Example structure:**
   ```markdown
   # Topic: FFT-Laplace Preconditioning

   ## Decision
   Implement FFT-Laplace preconditioning for O(n² log n) acceleration

   ## Rationale
   - First application to QAP
   - Significant speedup potential
   - Novel research contribution

   ## Alternatives Considered
   - Full matrix preconditioning (too slow)
   - Gradient-based methods (less effective)

   ## Implementation
   - See: Librex.QAP/methods/novel.py
   - Key function: fft_laplace_accelerate()

   ## Performance Impact
   - 100x speedup on medium instances
   - 250x on large instances
   ```

## Template for Development Notes

```markdown
# [Component/Feature Name]

## Overview
Brief description

## Architecture
How it works

## Key Files
- `path/to/implementation.py`
- `tests/test_something.py`

## Design Decisions
Why we chose this approach

## Performance
Timing and resource usage

## Related
Links to other notes or files
```

## Suggested Topics to Document

- Core pipeline architecture
- Method implementation patterns
- Agent coordination strategy
- Learning mechanisms
- Integration points
- Performance optimizations
- Data flow diagrams
- Error handling strategy

## Keep Notes Updated

Notes should be kept current as code changes:
- Add notes as you develop
- Update when architecture changes
- Remove obsolete notes
- Cross-reference with code

## Linking to Code

When referring to code:
```markdown
See `Librex.QAP/core/pipeline.py:line_number` for implementation
See `tests/test_methods.py:test_fft_laplace` for usage example
```

## Keeping It Organized

Current organization:
- Each topic = one file
- Related topics can reference each other
- Keep filenames clear and descriptive

---

Start adding development notes here as you work! 📝
