# [Prompt Name]

> **[One-line description]**

## Metadata

```yaml
name: [prompt-id]
version: 1.0.0
tags: [tag1, tag2, tag3]
tools: [amazon-q, claude, windsurf, cline]
domain: [computational-physics, ml, devops, etc.]
author: Meshal Alawein
last_updated: 2025-01-XX
```

## Purpose

[Explain what this prompt does and when to use it]

## Prompt

```markdown
[Your actual prompt content here]

## Context
- **[Key Context 1]**: [Description]
- **[Key Context 2]**: [Description]

## Requirements

### 1. [Requirement Category 1]
- [Specific requirement]
- [Specific requirement]

### 2. [Requirement Category 2]
- [Specific requirement]
- [Specific requirement]

## Workflow

1. **[Step 1]**: [Description]
2. **[Step 2]**: [Description]
3. **[Step 3]**: [Description]

## Success Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

## Example Output Format

### Before
```python
# Example before code
```

### After
```python
# Example after code
```
```

## Usage Examples

### Amazon Q
```
@prompt [prompt-id]

Context:
- [Key context]
```

### Claude CLI
```bash
cat .ai-knowledge/prompts/[category]/[prompt-id].md | \
  claude --context "[context]"
```

## Related Resources

- [Related Prompt 1](../path/to/prompt.md)
- [Related Workflow 1](../../workflows/path/to/workflow.py)

## Changelog

- **v1.0.0** (2025-01-XX): Initial version

---

**Pro Tip**: [Helpful tip for using this prompt]
