# AI Knowledge Structure

## Philosophy

**Single Source of Truth**: Everything lives in `.ai-knowledge/` on GitHub.  
**IDE-Agnostic**: Works with any AI tool (Amazon Q, Claude, Windsurf, Cline, Cursor, etc.)  
**Version Controlled**: All prompts, workflows, and rules are tracked in Git.

## Directory Layout

```
.ai-knowledge/
│
├── prompts/                    # Reusable prompt templates
│   ├── superprompts/          # Complex multi-step prompts
│   ├── code-review/           # Code review prompts
│   ├── refactoring/           # Refactoring prompts
│   ├── architecture/          # System design prompts
│   ├── debugging/             # Debugging workflows
│   └── README.md              # Prompts documentation
│
├── workflows/                  # Automated Python scripts
│   ├── development/           # Dev workflows
│   ├── testing/               # Testing workflows
│   ├── deployment/            # Deployment workflows
│   ├── research/              # Research workflows
│   └── README.md              # Workflows documentation
│
├── rules/                      # Development standards
│   ├── global/                # Cross-project rules
│   ├── python/                # Python-specific rules
│   ├── typescript/            # TypeScript-specific rules
│   ├── physics/               # Scientific computing rules
│   └── README.md              # Rules documentation
│
├── catalog/                    # Searchable index
│   ├── INDEX.md               # Human-readable catalog
│   ├── prompts.json           # Prompt metadata
│   ├── workflows.json         # Workflow metadata
│   └── README.md              # Catalog documentation
│
├── tools/                      # Automation scripts
│   ├── migrate-prompts.py     # Extract prompts from docs
│   ├── update-catalog.py      # Update catalog
│   ├── create-workflow.py     # Generate new workflow
│   ├── sync-across-tools.py   # Sync to IDEs
│   └── README.md              # Tools documentation
│
├── templates/                  # Quick-start templates
│   └── new-superprompt.md     # Prompt template
│
├── README.md                   # Main documentation
├── QUICKSTART.md              # 5-minute guide
├── CLI.md                     # Command reference
├── SETUP.md                   # Setup guide
├── DONE.md                    # Completion summary
└── STRUCTURE.md               # This file
```

## Design Principles

### 1. Clear Organization
- **prompts/** - What to say to AI
- **workflows/** - What to run automatically
- **rules/** - How to code
- **catalog/** - How to find things
- **tools/** - How to manage it all

### 2. Self-Documenting
Every directory has a README explaining:
- What it contains
- How to use it
- How to contribute

### 3. IDE-Agnostic
Prompts work in any tool:
```
@prompt optimization-refactor
```

Or copy directly:
```bash
cat .ai-knowledge/prompts/superprompts/optimization-refactor.md
```

### 4. Automation-First
Common tasks are automated:
- `migrate-prompts.py` - Find existing prompts
- `update-catalog.py` - Keep index current
- `create-workflow.py` - Generate new workflows
- `sync-across-tools.py` - Sync to IDEs

### 5. Version Controlled
Everything is in Git:
- Track what works
- Revert bad changes
- Share across team
- Document evolution

## Usage Patterns

### Daily Development
```bash
# Use a prompt
@prompt physics-code-review

# Run a workflow
python .ai-knowledge/workflows/development/test-driven-refactor.py --target <file>
```

### Adding New Content
```bash
# Create new prompt
cp .ai-knowledge/templates/new-superprompt.md \
   .ai-knowledge/prompts/superprompts/my-prompt.md

# Update catalog
python .ai-knowledge/tools/update-catalog.py

# Sync to IDEs (optional)
python .ai-knowledge/tools/sync-across-tools.py
```

### Finding Resources
```bash
# Browse catalog
cat .ai-knowledge/catalog/INDEX.md

# Search by tag
grep -r "optimization" .ai-knowledge/catalog/
```

## Extending the System

### Add New IDE
Edit `.ai-knowledge/tools/sync-across-tools.py`:
```python
IDE_DIRS = [
    # ... existing IDEs ...
    ROOT / ".your-ide" / "prompts",  # Your IDE
]
```

### Add New Category
```bash
mkdir .ai-knowledge/prompts/new-category
echo "# New Category" > .ai-knowledge/prompts/new-category/README.md
```

### Add New Tool
```bash
# Create new automation script
touch .ai-knowledge/tools/my-tool.py
chmod +x .ai-knowledge/tools/my-tool.py
```

## Benefits

1. **No Duplication**: One place for all AI knowledge
2. **Works Everywhere**: Any IDE, any AI tool
3. **Easy to Find**: Searchable catalog
4. **Easy to Share**: Just point to GitHub
5. **Easy to Maintain**: Automated updates
6. **Version Controlled**: Track changes over time

---

**This structure is designed to scale with your needs while staying simple and maintainable.**
