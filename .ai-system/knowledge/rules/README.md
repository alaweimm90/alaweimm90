# Rules Directory

Development rules and standards that guide AI-assisted coding.

## Categories

### 🌍 [global/](global/)
Cross-project rules:
- `physics-first.md` - Physics correctness before optimization
- `performance-aware.md` - Always consider performance
- `minimal-code.md` - Write minimal, elegant code

### 🐍 [python/](python/)
Python-specific rules:
- `numpy-style.md` - NumPy/JAX coding standards
- `type-hints.md` - Comprehensive type hints
- `docstring-format.md` - Physics-aware docstrings

### 📘 [typescript/](typescript/)
TypeScript-specific rules:
- `type-safety.md` - Strict type safety
- `functional-style.md` - Functional programming patterns

### ⚛️ [physics/](physics/)
Scientific computing rules:
- `conservation-laws.md` - Verify conservation laws
- `unit-consistency.md` - Check unit consistency
- `numerical-stability.md` - Ensure numerical stability

## How Rules Work

Rules are automatically loaded by AI tools when working in this repository.

### Amazon Q
Rules in `.amazonq/rules/` are auto-loaded.

### Claude / Other IDEs
Reference rules explicitly or they're loaded from context.

## Rule Format

```markdown
# Rule Name

## Principle
[Core principle]

## Example

### Bad
[Code example]

### Good
[Code example]

## When to Apply
[Specific scenarios]
```

## Best Practices

1. **Specific**: Clear, actionable guidance
2. **Examples**: Show good vs bad code
3. **Context**: When to apply the rule
4. **Testable**: Can be validated automatically

## Contributing

When adding a rule:
1. Use clear, descriptive name
2. Include code examples
3. Explain when to apply
4. Update this README
