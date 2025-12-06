# Quick Start Guide - AI Knowledge Management System

## 🚀 Getting Started in 5 Minutes

### Step 1: Sync Prompts to All IDEs
```bash
cd tools/cross-ide-sync
python cli.py sync
```
**Result**: 67 prompts synced to Amazon Q, Claude, Windsurf, Cline, Cursor

### Step 2: Validate Your Prompts
```bash
cd tools/prompt-testing
python cli.py validate --all
```
**Result**: Quality report for all 67 prompts

### Step 3: Get Recommendations
```bash
cd tools/recommendation-engine
python cli.py recommend "I need to optimize my API performance"
```
**Result**: Top 5 relevant prompts

### Step 4: View Analytics
```bash
cd tools/analytics
python dashboard.py
```
**Result**: Usage statistics and insights

### Step 5: Compose a Workflow
```bash
cd tools/prompt-composer
python cli.py templates/fullstack-workflow.md example-vars.json > my-workflow.md
```
**Result**: Custom workflow prompt

---

## 📋 Common Tasks

### Add a New Prompt
1. Create prompt in `docs/ai-knowledge/prompts/category/`
2. Run: `python tools/cross-ide-sync/cli.py sync`
3. Validate: `python tools/prompt-testing/cli.py validate your-prompt.md`

### Search Marketplace
```bash
cd tools/marketplace
python cli.py search "testing"
```

### Set Your Preferences
```python
from tools.adaptive_prompts.personalizer import PromptPersonalizer

p = PromptPersonalizer()
p.set_preference('language', 'typescript')
p.set_preference('framework', 'react')
p.set_preference('style', 'concise')
```

### Run a Workflow
```bash
cd tools/orchestrator
python engine.py workflows/development-cycle.yaml
```

---

## 🎯 Real-World Examples

### Example 1: Code Review Workflow
```bash
# 1. Get recommendation
python tools/recommendation-engine/cli.py recommend "code review"

# 2. Use the recommended prompt
# Open: docs/ai-knowledge/prompts/code-review/agentic-code-review.md

# 3. Track usage
# Analytics automatically tracks when you use it
```

### Example 2: Build Custom Workflow
```bash
# 1. Create variables file
cat > my-vars.json << EOF
{
  "project_name": "MyAPI",
  "tech_stack": "FastAPI + PostgreSQL",
  "backend_lang": "Python",
  "frontend_lang": "TypeScript",
  "coverage_target": "90"
}
EOF

# 2. Compose workflow
python tools/prompt-composer/cli.py templates/fullstack-workflow.md my-vars.json > my-workflow.md

# 3. Use the workflow
# Open my-workflow.md in your IDE
```

### Example 3: Publish to Marketplace
```bash
# 1. Create your prompt
# docs/ai-knowledge/prompts/custom/my-prompt.md

# 2. Publish
python tools/marketplace/cli.py publish my-prompt docs/ai-knowledge/prompts/custom/my-prompt.md

# 3. Others can install
python tools/marketplace/cli.py install your-username/my-prompt
```

---

## 🔄 Daily Workflow

### Morning Setup
```bash
# Sync latest prompts
python tools/cross-ide-sync/cli.py sync

# Check for updates
python tools/prompt-testing/cli.py regression --check
```

### During Development
```bash
# Get recommendations as needed
python tools/recommendation-engine/cli.py recommend "your task"

# Use prompts in your IDE (already synced!)
```

### End of Day
```bash
# View analytics
python tools/analytics/dashboard.py

# Save regression baseline
python tools/prompt-testing/cli.py regression --save
```

---

## 🛠️ Advanced Usage

### Auto-Sync on File Changes
```bash
cd tools/cross-ide-sync
python cli.py watch
# Leave running in background
```

### Generate New Prompt
```bash
cd tools/meta-prompt
python generator.py "Create a prompt for database optimization"
```

### Extract Patterns
```bash
cd tools/pattern-extractor
python extractor.py
```

### Run Tests
```bash
cd tools/prompt-testing
python cli.py validate --all
python cli.py benchmark optimization-framework 10
```

---

## 📊 Monitor Your System

### Check System Health
```bash
# Validation status
python tools/prompt-testing/cli.py validate --all

# Analytics overview
python tools/analytics/dashboard.py

# Installed prompts
python tools/marketplace/cli.py list
```

### Performance Metrics
- Sync: <1 second for 67 files
- Validation: <1 second for all prompts
- Recommendations: <100ms
- Search: <50ms

---

## 🎓 Learning Path

### Week 1: Basics
- Day 1-2: Sync prompts, explore categories
- Day 3-4: Use recommendations
- Day 5-7: Try prompt composition

### Week 2: Advanced
- Day 1-2: Set up auto-sync
- Day 3-4: Create custom prompts
- Day 5-7: Publish to marketplace

### Week 3: Mastery
- Day 1-2: Build custom workflows
- Day 3-4: Analyze patterns
- Day 5-7: Contribute improvements

---

## 🆘 Troubleshooting

### Prompts not syncing?
```bash
# Check source exists
ls docs/ai-knowledge/prompts/

# Re-sync
python tools/cross-ide-sync/cli.py sync --force
```

### Validation failing?
```bash
# Check specific prompt
python tools/prompt-testing/cli.py validate path/to/prompt.md

# View issues
# Output shows what's missing
```

### Analytics not working?
```bash
# Initialize database
cd tools/analytics
python tracker.py
```

---

## 📞 Get Help

- **Documentation**: `docs/ai-knowledge/`
- **Examples**: `tools/*/test_*.py`
- **Issues**: Check phase completion docs

---

**Ready to go!** Start with Step 1 above. 🚀
