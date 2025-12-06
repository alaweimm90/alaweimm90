# Claude AI Assistant Guide

## Repository Context

**Name:** Librex  
**Type:** research-library  
**Purpose:** Universal optimization framework for combinatorial and continuous optimization problems.

## Key Files

- `README.md` - Main documentation and quickstart
- `docs/` - Book-style documentation and architecture
- `Librex/` - Core Python library
- `tests/` - Unit, integration, and performance tests

## Development Guidelines

1. Follow existing code style and module layout.
2. Add tests for new features in `tests/`.
3. Update relevant docs under `docs/` when behavior or APIs change.

## Common Tasks

### Running Tests
```bash
pytest
```

### Building
```bash
# Python packaging is configured via pyproject.toml
# Use standard build tooling, for example:
python -m build
```

## Architecture

See `docs/PROJECT_SUMMARY.md` and `docs/MULTI_SOLVER_SUITE_ARCHITECTURE.md` for high-level system design.

## SSOT References

- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/INDEX.md` — master SSOT index
- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/5-TOOLS/ide-integration.md` — IDE integration guide
- `ORGANIZATIONS/alaweimm90-tools/agis/docs/standards/2-PROMPTS/PROMPT_REGISTRY.md` — prompt registry

