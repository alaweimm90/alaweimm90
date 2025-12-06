# Catalog

Searchable index of all AI knowledge resources.

## Files

- **[INDEX.md](INDEX.md)** - Human-readable catalog with search
- **[prompts.json](prompts.json)** - Machine-readable prompt metadata
- **[workflows.json](workflows.json)** - Machine-readable workflow metadata

## Usage

### Browse Catalog
Open [INDEX.md](INDEX.md) and use Ctrl+F to search.

### Search by Tag
```bash
# Find all optimization-related resources
grep -r "optimization" .ai-knowledge/catalog/

# Find all physics prompts
grep -r "physics" .ai-knowledge/catalog/prompts.json
```

### Update Catalog
```bash
# Auto-update from filesystem
python ../tools/update-catalog.py
```

## Catalog Structure

### prompts.json
```json
{
  "prompts": [
    {
      "id": "optimization-refactor",
      "name": "Optimization Refactor",
      "category": "superprompt",
      "tags": ["optimization", "refactoring", "physics"],
      "path": "prompts/superprompts/optimization-refactor.md"
    }
  ]
}
```

### workflows.json
```json
{
  "workflows": [
    {
      "id": "test-driven-refactor",
      "name": "Test-Driven Refactor",
      "category": "development",
      "automated": true,
      "path": "workflows/development/test-driven-refactor.py"
    }
  ]
}
```

## Metadata Fields

### Prompts
- `id`: Unique identifier
- `name`: Display name
- `category`: superprompt, code-review, refactoring, etc.
- `tags`: Searchable tags
- `path`: Relative path from `.ai-knowledge/`

### Workflows
- `id`: Unique identifier
- `name`: Display name
- `category`: development, testing, deployment, research
- `automated`: Boolean (can be run as script)
- `path`: Relative path from `.ai-knowledge/`
