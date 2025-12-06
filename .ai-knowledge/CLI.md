# AI Knowledge CLI

Quick commands for common tasks.

## Migrate Prompts
```bash
python .ai-knowledge/tools/migrate-prompts.py
```

## Create New Workflow
```bash
python .ai-knowledge/tools/create-workflow.py
```

## Sync Across Tools
```bash
python .ai-knowledge/tools/sync-across-tools.py
```

## Search Catalog
```bash
# Search prompts
grep -r "optimization" .ai-knowledge/catalog/prompts.json

# Search workflows
grep -r "testing" .ai-knowledge/catalog/workflows.json
```

## Use a Prompt
```bash
# In any AI tool
@prompt optimization-refactor

# Or copy to clipboard
cat .ai-knowledge/prompts/superprompts/optimization-refactor.md | clip
```

## Create New Prompt
```bash
cp .ai-knowledge/templates/new-superprompt.md \
   .ai-knowledge/prompts/superprompts/my-prompt.md
```

## Run Workflow
```bash
python .ai-knowledge/workflows/development/test-driven-refactor.py \
  --target librex/equilibria/
```

## Update Catalog
```bash
# After adding new prompts/workflows
python .ai-knowledge/tools/update-catalog.py
```
