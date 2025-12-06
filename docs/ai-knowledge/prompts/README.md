# Prompts Directory

Reusable prompt templates for AI-assisted development.

## Categories

### 📝 [superprompts/](superprompts/)
Complex, multi-step prompts for major tasks:
- `optimization-refactor.md` - Refactor optimization code with physics constraints
- `gpu-optimization.md` - Convert NumPy to JAX for GPU acceleration
- `test-driven-development.md` - TDD workflow for scientific computing

### 🔍 [code-review/](code-review/)
Code review prompts with specific focus areas:
- `physics-code-review.md` - Review with physics correctness checks
- `performance-review.md` - Focus on performance bottlenecks
- `security-review.md` - Security-focused review

### 🔧 [refactoring/](refactoring/)
Targeted refactoring prompts:
- `extract-algorithm.md` - Extract optimization algorithm to module
- `simplify-physics.md` - Simplify physics calculations

### 🏗️ [architecture/](architecture/)
System design and architecture prompts:
- `system-design.md` - Design scalable systems
- `api-design.md` - RESTful API design

### 🐛 [debugging/](debugging/)
Debugging workflows:
- `physics-debugging.md` - Debug physics simulations
- `performance-debugging.md` - Profile and fix bottlenecks

## Usage

### In Any IDE
```
@prompt optimization-refactor
```

### Copy to Clipboard
```bash
cat .ai-knowledge/prompts/superprompts/optimization-refactor.md | clip
```

### Create New Prompt
```bash
cp ../templates/new-superprompt.md superprompts/my-prompt.md
```

## Best Practices

1. **Use metadata**: Include tags, version, tools
2. **Provide examples**: Show before/after code
3. **Document context**: When to use this prompt
4. **Version control**: Track changes with version numbers
5. **Test prompts**: Verify they work across different IDEs

## Contributing

When adding a new prompt:
1. Use the template: `../templates/new-superprompt.md`
2. Add to appropriate category
3. Update this README
4. Run: `python ../tools/update-catalog.py`
