# Meta-Prompt Generator

Generate prompts from natural language requirements.

## Usage

### Basic
```bash
python tools/meta-prompt/generator.py "optimize database queries"
```

### Save to File
```bash
python tools/meta-prompt/generator.py \
  "optimize database queries for better performance" \
  --output docs/ai-knowledge/prompts/superprompts/database-query-optimization.md
```

### Custom Templates
```bash
python tools/meta-prompt/generator.py \
  "implement CI/CD pipeline" \
  --templates tools/meta-prompt/templates
```

## How It Works

1. **Parse** - Extract key information from requirement
2. **Select** - Choose appropriate template based on domain
3. **Fill** - Populate template with extracted data
4. **Validate** - Ensure quality standards

## Templates

- `generic.md` - Default template
- `optimization.md` - Performance optimization
- `testing.md` - Testing strategy
- `architecture.md` - System design
- `debugging.md` - Debugging workflow

## Examples

### Input
```
"optimize database queries for better performance"
```

### Output
Complete prompt with:
- Title: "Optimize Database Queries"
- Purpose: Performance optimization
- Domain: optimization
- Use cases
- Examples
- Success criteria

## Adding Templates

Create new template in `templates/`:

```markdown
# {title}

> **{purpose}**

## Purpose
{purpose}

## When to Use
{use_cases}

## Prompt
[Your template content]

## Examples
{examples}
```

## Quality Standards

Generated prompts must have:
- Clear title
- Purpose statement
- Use cases
- Detailed instructions
- Examples
- Success criteria

## Next Steps

- Add more templates
- Improve parsing
- Add validation rules
- Generate better examples
