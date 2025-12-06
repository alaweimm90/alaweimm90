# PromptForge Lite - Offline Prompt Pattern Extraction

Extract reusable prompt patterns from your notes and documentation using regex. No ML required - works completely offline.

## What It Does

Scans text files (markdown, txt) and extracts common prompt patterns:

- **Instructions**: Task/goal statements
- **Role-play**: "You are X" patterns
- **Format**: Output structure specifications
- **Constraints**: Requirements and limitations
- **Examples**: Sample inputs/outputs
- **Step-by-step**: Sequential procedures
- **Conditionals**: If-then logic
- **Context**: Background information

## Installation

```bash
cd promptforge-lite
# No dependencies - pure Python 3.11+
```

## Usage

### Extract from Single File

```bash
python promptforge.py extract --input notes.md --output prompts.json
```

### Scan Entire Directory

```bash
python promptforge.py scan --directory ./my-notes/ --output all_prompts.json
```

### Filter by Confidence

```bash
python promptforge.py scan --directory ./notes/ --output prompts.json --min-confidence 0.6
```

### Retrieve Template by Name

```bash
python promptforge.py template --input prompts.json --name "instruction_analyze_code"
```

### Filter by Type

```bash
python promptforge.py template --input prompts.json --type "role_play"
```

### Filter by Tag

```bash
python promptforge.py template --input prompts.json --tag "coding"
```

## Pattern Types

1. **instruction** - Direct task statements
2. **role_play** - Role assignments ("act as", "you are")
3. **format** - Output structure requirements
4. **constraint** - Limitations and requirements
5. **example** - Sample demonstrations
6. **step_by_step** - Sequential instructions
7. **conditional** - If-then logic
8. **context** - Background information

## Variable Detection

Automatically detects variable placeholders:
- `{variable}`
- `[variable]`
- `<variable>`
- `$variable` or `${variable}`

## Output Format

```json
{
  "total_patterns": 42,
  "files_processed": 15,
  "timestamp": "2025-11-15T10:30:00",
  "patterns": [
    {
      "pattern_type": "instruction",
      "pattern_name": "instruction_analyze_code",
      "template": "Analyze the following code for {quality} issues",
      "variables": ["quality"],
      "example": "e.g., security, performance, style",
      "source_file": "notes/coding-prompts.md",
      "line_number": 42,
      "confidence": 0.8,
      "tags": ["coding", "analysis"]
    }
  ]
}
```

## Confidence Scoring

Patterns are scored 0.0-1.0 based on:
- Presence of variables (+0.2)
- Structured content (+0.1)
- Explicit instructions (+0.1)
- Length penalty for very short templates (-0.2)

## Example Workflow

1. **Extract patterns from your notes**
```bash
python promptforge.py scan --directory ~/Documents/notes/ --output my_prompts.json --min-confidence 0.5
```

2. **Browse top reusable patterns**
```bash
python promptforge.py template --input my_prompts.json
```

3. **Get specific template**
```bash
python promptforge.py template --input my_prompts.json --name "role_play_code_reviewer"
```

4. **Find all coding-related prompts**
```bash
python promptforge.py template --input my_prompts.json --tag "coding"
```

## Use Cases

1. **Personal prompt library** - Extract patterns from your notes
2. **Team knowledge sharing** - Scan team documentation for best practices
3. **Prompt template creation** - Build reusable prompt templates
4. **Workflow documentation** - Discover common instruction patterns
5. **Offline operation** - Works without internet or API calls

## Example Input/Output

**Input** (notes.md):
```markdown
# Code Review Prompts

Instruction: Analyze the code for security vulnerabilities

Role: You are an experienced security engineer

Format: Provide findings in bullet points with severity levels

Example: e.g., SQL injection, XSS, CSRF
```

**Output** (patterns extracted):
- `instruction_analyze_code` - "Analyze the code for security vulnerabilities"
- `role_play_experienced_security` - "You are an experienced security engineer"
- `format_provide_findings` - "Provide findings in bullet points with severity levels"
- `example` - "e.g., SQL injection, XSS, CSRF"

## Current Limitations

- Regex-based (not semantic understanding)
- Works best with structured notes
- May miss implicit patterns
- No context-aware variable inference

## Revenue Model

- **Individual**: $29/month
- **Team** (5 users): $99/month
- **Enterprise**: $299/month (unlimited users)

## Build Info

- Build time: 4 hours
- Credit used: ~$60
- Lines of code: 450
- Status: Functional prototype

## Future Enhancements

- Support for more file formats (YAML, JSON, code comments)
- Semantic similarity clustering
- Template composition (combine patterns)
- Variable type inference
- Integration with prompt testing frameworks
