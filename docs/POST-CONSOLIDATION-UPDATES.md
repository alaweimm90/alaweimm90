# Post-Consolidation Updates

## After running `consolidate.bat`, update these paths:

### 1. Update Sync Tool Paths
```bash
# Edit: .ai-system/tools/cross-ide-sync/config.py
# Change: self.source = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge"
# To:     self.source = Path(__file__).parent.parent / "knowledge"
```

### 2. Update Recommendation Engine
```bash
# Edit: .ai-system/tools/recommendation-engine/recommender.py
# Change: prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
# To:     prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
```

### 3. Update Pattern Extractor
```bash
# Edit: .ai-system/tools/pattern-extractor/extractor.py
# Change: prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
# To:     prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
```

### 4. Update Prompt Composer
```bash
# Edit: .ai-system/tools/prompt-composer/composer.py
# Change: prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
# To:     prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
```

### 5. Update Validator
```bash
# Edit: .ai-system/tools/prompt-testing/validator.py
# Change: prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
# To:     prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
```

### 6. Update Batch Files
```bash
# Edit: daily-prompt-routine.bat
# Change all: tools\ → .ai-system\tools\

# Edit: start-auto-sync.bat
# Change: cd tools\cross-ide-sync → cd .ai-system\tools\cross-ide-sync
```

### 7. Update Documentation
```bash
# Edit: README-SYSTEM.md
# Update all paths to reflect new structure

# Edit: PROMPT-CHEATSHEET.md
# Update all command paths
```

## Or Run Auto-Update Script:

```bash
python update-all-paths.py
```
