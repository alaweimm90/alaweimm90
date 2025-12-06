# Tools Directory

Automation scripts for managing AI knowledge.

## Available Tools

### 📦 migrate-prompts.py
Scan existing documentation for reusable prompts.

```bash
python migrate-prompts.py
# Output: migration-report.md
```

### 📊 update-catalog.py
Auto-update catalog from filesystem.

```bash
python update-catalog.py
# Updates: catalog/prompts.json, catalog/workflows.json
```

### ✨ create-workflow.py
Interactive workflow generator.

```bash
python create-workflow.py
# Follow prompts to create new workflow
```

### 🔄 sync-across-tools.py
Sync prompts to IDE-specific locations.

```bash
python sync-across-tools.py
# Copies to: Amazon Q, Claude, etc.
```

## Usage Patterns

### After Adding New Content
```bash
# 1. Update catalog
python update-catalog.py

# 2. Sync to IDEs (optional)
python sync-across-tools.py
```

### Starting New Project
```bash
# 1. Scan for existing prompts
python migrate-prompts.py

# 2. Create workflows
python create-workflow.py
```

## Extending Tools

### Add New IDE to Sync
Edit `sync-across-tools.py`:
```python
CURSOR = ROOT / ".cursor" / "prompts"
# Add to sync() function
```

### Add New Catalog Type
Edit `update-catalog.py`:
```python
def scan_rules():
    # Scan rules directory
    pass
```

## Best Practices

1. **Run update-catalog.py** after any changes
2. **Check migration-report.md** for valuable prompts
3. **Sync regularly** to keep IDEs updated
4. **Extend tools** as needed for your workflow
