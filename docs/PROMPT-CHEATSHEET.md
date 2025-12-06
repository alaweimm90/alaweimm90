# AI Prompt System - Quick Reference

## 🚀 Most Used Commands

```bash
# Sync all prompts (do this first!)
python tools/cross-ide-sync/cli.py sync

# Get recommendations
python tools/recommendation-engine/cli.py recommend "your task"

# Validate prompts
python tools/prompt-testing/cli.py validate --all

# View analytics
python tools/analytics/dashboard.py

# Search marketplace
python tools/marketplace/cli.py search "keyword"
```

## 📝 In Your IDE (Amazon Q)

```
@optimization-framework     # For optimization tasks
@agentic-code-review       # For code reviews
@gpu-optimization          # For GPU/CUDA work
@optilibria-optimization   # For Optilibria-specific work
@testing-qa-strategy       # For testing
@api-design-development    # For API design
```

## 🎯 Common Workflows

### Code Review
```bash
# 1. Get recommendation
python tools/recommendation-engine/cli.py recommend "code review"

# 2. Use in IDE
# Type: @agentic-code-review

# 3. Track usage (automatic)
```

### Optimization Work
```bash
# 1. Use Optilibria prompt
# Type: @optilibria-optimization

# 2. Follow the checklist:
# - JIT compile
# - Vectorize
# - Profile
# - Benchmark
```

### Build Feature
```bash
# 1. Get workflow
python tools/recommendation-engine/cli.py workflow "build feature"

# 2. Compose custom workflow
python tools/prompt-composer/cli.py templates/fullstack-workflow.md vars.json

# 3. Follow phases
```

## 🔧 Maintenance

### Daily
```bash
# Run daily routine
daily-prompt-routine.bat
```

### Weekly
```bash
# Save regression baseline
python tools/prompt-testing/cli.py regression --save

# Check patterns
python tools/pattern-extractor/extractor.py
```

### Monthly
```bash
# Publish best prompts
python tools/marketplace/cli.py publish my-prompt path/to/prompt.md

# Review analytics
python tools/analytics/dashboard.py
```

## 💡 Pro Tips

1. **Keep Auto-Sync Running**: `start-auto-sync.bat`
2. **Use Tab Completion**: Type `@` in IDE and search
3. **Rate Prompts**: Help improve recommendations
4. **Create Custom**: Use meta-prompt generator
5. **Check Analytics**: See what works best

## 🆘 Troubleshooting

**Prompts not showing?**
```bash
python tools/cross-ide-sync/cli.py sync
```

**Validation failing?**
```bash
python tools/prompt-testing/cli.py validate path/to/prompt.md
# Check output for issues
```

**Need help?**
- Quick Start: `docs/ai-knowledge/QUICK-START.md`
- Full Docs: `docs/ai-knowledge/FINAL-SUMMARY.md`

## 📊 System Status

Check: `README-SYSTEM.md`

---

**Keep this handy!** Pin to your desktop or print it out.
