# AI Knowledge System - Quick Start

## 🎯 What This Is

A centralized system for managing all your AI prompts, workflows, and rules across Amazon Q, Claude, Windsurf, Cline, and other tools.

## 🚀 Getting Started (5 minutes)

### 1. Sync Your Tools
```bash
python .ai-knowledge/tools/sync-across-tools.py
```

### 2. Use a Prompt
In any AI tool:
```
@prompt optimization-refactor
```

### 3. Run a Workflow
```bash
python .ai-knowledge/workflows/development/test-driven-refactor.py --target librex/equilibria/
```

## 📚 Key Resources

- **Catalog**: [catalog/INDEX.md](catalog/INDEX.md) - Browse all resources
- **Prompts**: [prompts/](prompts/) - Reusable prompt templates
- **Workflows**: [workflows/](workflows/) - Automated workflows
- **Rules**: [rules/](rules/) - Development rules

## 💡 Common Use Cases

### Refactoring Code
```
@prompt optimization-refactor
Context: Gradient Descent in Optilibria
Target: 10x performance improvement
```

### Code Review
```
@prompt physics-code-review
File: librex/equilibria/algorithms/gradient_descent.py
```

### Running Tests
```bash
python .ai-knowledge/workflows/development/test-driven-refactor.py \
  --target librex/equilibria/algorithms/
```

## 🔧 Tool-Specific Setup

### Amazon Q
Prompts auto-load from `~/.aws/amazonq/prompts/`

### Claude
Reference with `@prompt <name>` in any Claude interface

### Windsurf/Cline
Use `@prompt` syntax in chat

## 📖 Next Steps

1. Browse the [catalog](catalog/INDEX.md)
2. Try the `optimization-refactor` superprompt
3. Run the `test-driven-refactor` workflow
4. Create your own prompts in `prompts/`

---

**Questions?** Check [README.md](README.md) for full documentation.
