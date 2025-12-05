# Codemap Generation Guide for Claude Opus

> Instructions for generating and maintaining architectural codemaps using MCP tools and context providers.

**Target Audience:** Claude Opus (or any LLM with MCP access)  
**Output Location:** `docs/governance/CODEMAPS.md`  
**Update Frequency:** After significant architectural changes

---

## Table of Contents

1. [Overview](#1-overview)
2. [MCP Tools for Context Gathering](#2-mcp-tools-for-context-gathering)
3. [Generation Workflow](#3-generation-workflow)
4. [Trace Diagram Format](#4-trace-diagram-format)
5. [Update Procedures](#5-update-procedures)
6. [Quality Checklist](#6-quality-checklist)

---

## 1. Overview

Codemaps are visual trace diagrams that document:

- **Call flows** between functions/methods
- **Data transformations** through pipelines
- **Integration points** between systems
- **File/line references** for navigation

### Target Systems

| System                   | Primary Files                                            | Trace Count |
| ------------------------ | -------------------------------------------------------- | ----------- |
| Unified Automation       | `automation/core/*.ts`, `automation/cli/*.ts`            | 3           |
| MCP Integration          | `tools/cli/mcp.py`, `.metaHub/scripts/integration/*`     | 4           |
| Orchestration Governance | `.metaHub/scripts/orchestration/*`                       | 4           |
| MetaHub Governance       | `.metaHub/scripts/compliance/*`, `.metaHub/scripts/*.py` | 4           |
| DevOps Templates         | `tools/cli/devops*.ts`, `tools/lib/*.ts`                 | 3           |
| ORCHEX Orchestration     | `tools/orchex/orchestration/*`, `tools/orchex/cli/*`     | 3           |

---

## 2. MCP Tools for Context Gathering

### 2.1 Sequential Thinking (mcp13_sequentialthinking)

Use for **complex analysis** that requires multi-step reasoning:

```json
{
  "thought": "Analyzing the MCP server registration flow...",
  "thoughtNumber": 1,
  "totalThoughts": 5,
  "nextThoughtNeeded": true
}
```

**When to use:**

- Breaking down complex call chains
- Identifying hidden dependencies
- Planning trace diagram structure
- Verifying integration points

### 2.2 Memory MCP (mcp9\_\*)

Store and retrieve architectural knowledge:

```json
// Create entity for a system
{
  "entities": [{
    "name": "AutomationCore",
    "entityType": "TypeScript Class",
    "observations": [
      "Entry point for unified automation",
      "Loads 24 agents from YAML",
      "Routes tasks via keyword matching"
    ]
  }]
}

// Create relations between systems
{
  "relations": [{
    "from": "AutomationCore",
    "to": "MCPManager",
    "relationType": "invokes"
  }]
}
```

**Memory operations:**

- `mcp9_create_entities` - Document new components
- `mcp9_create_relations` - Map dependencies
- `mcp9_search_nodes` - Find existing documentation
- `mcp9_read_graph` - Get full system map

### 2.3 Filesystem MCP (mcp4\_\*)

Read source files for accurate tracing:

```json
// Read a specific file
{ "path": "c:/Users/mesha/Desktop/GitHub/automation/core/AutomationCore.ts" }

// Get directory structure
{ "path": "c:/Users/mesha/Desktop/GitHub/tools/cli" }

// Search for patterns
{ "path": "c:/Users/mesha/Desktop/GitHub", "pattern": "**/*.py" }
```

**Key operations:**

- `mcp4_read_text_file` - Read source code
- `mcp4_directory_tree` - Understand structure
- `mcp4_search_files` - Find relevant files

### 2.4 DeepWiki MCP (mcp1\_\*)

Query external library documentation:

```json
// Get wiki structure for a library
{ "repoName": "commander-js/commander.js" }

// Ask specific questions
{
  "repoName": "chalk/chalk",
  "question": "How does chalk handle color detection?"
}
```

### 2.5 Exa MCP (mcp2\_\*)

Search for code patterns and best practices:

```json
// Search for implementation patterns
{
  "query": "TypeScript workflow orchestration DAG execution",
  "tokensNum": 5000
}
```

### 2.6 Puppeteer MCP (mcp12\_\*)

For web-based documentation or dashboards:

```json
// Navigate to local dashboard
{ "url": "http://localhost:3000/dashboard" }

// Take screenshot for documentation
{ "name": "governance-dashboard", "width": 1200, "height": 800 }
```

---

## 3. Generation Workflow

### Step 1: Gather Context

```text
1. Use mcp4_directory_tree to map the target system
2. Use mcp4_read_text_file on key entry points
3. Use mcp9_search_nodes to find existing documentation
4. Use mcp13_sequentialthinking to plan the trace structure
```

### Step 2: Identify Entry Points

For each system, identify:

- **CLI commands** (entry points)
- **Core classes** (orchestration)
- **Configuration loaders** (initialization)
- **External integrations** (boundaries)

### Step 3: Trace Call Chains

Follow the code flow:

```text
Entry Point (CLI/API)
  └── Handler Function
      ├── Validation Layer
      ├── Business Logic
      │   ├── Data Access
      │   └── External Calls
      └── Response/Output
```

### Step 4: Document Line References

Every trace node should include:

- **File path** (relative to repo root)
- **Line number** (for navigation)
- **Function/method name**

Example:

```text
├── loadAgents() <-- AutomationCore.ts:37
│   └── readYamlFile(agents.yaml) <-- AutomationCore.ts:42
```

### Step 5: Create Reference Tables

After each trace diagram, add a reference table:

```markdown
| Location     | Line         | Description       |
| ------------ | ------------ | ----------------- |
| `file.ts:37` | functionName | Brief description |
```

### Step 6: Build Integration Map

Create ASCII diagram showing system interconnections:

```text
┌─────────────┐     ┌─────────────┐
│  System A   │────▶│  System B   │
└─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  System C   │◀────│  System D   │
└─────────────┘     └─────────────┘
```

---

## 4. Trace Diagram Format

### Standard Trace Block

````markdown
### Trace X.Y: [Descriptive Title]

```text
[Flow Name]
├── function1() <-- file.ts:line
│   └── function2() <-- file.ts:line
│       ├── function3() <-- file.ts:line
│       │   └── Return value description
│       └── function4() <-- file.ts:line
└── Final step
```

| Location     | Line      | Description |
| ------------ | --------- | ----------- |
| `file.ts:10` | function1 | Entry point |
| `file.ts:25` | function2 | Processing  |
````

### Tree Characters

Use these Unicode box-drawing characters:

| Character | Usage                           |
| --------- | ------------------------------- |
| `├──`     | Branch with siblings below      |
| `└──`     | Last branch (no siblings below) |
| `│`       | Vertical continuation           |
| `<--`     | File/line reference marker      |

### Indentation Rules

- 4 spaces per nesting level
- Align `<--` markers vertically when possible
- Use `│` to show continuation through nested levels

---

## 5. Update Procedures

### When to Update

1. **New system added** - Create new section with 2-4 traces
2. **Major refactor** - Update affected traces
3. **API changes** - Update line references
4. **New integration** - Update integration map

### Update Checklist

```markdown
- [ ] Read changed files with mcp4_read_text_file
- [ ] Verify line numbers are still accurate
- [ ] Check function signatures haven't changed
- [ ] Update integration map if boundaries changed
- [ ] Run sequential thinking to verify completeness
- [ ] Update memory graph with new relations
```

### Automated Verification

Run these commands to verify codemap accuracy:

```bash
# Check if referenced files exist
grep -oP '`[^`]+\.(ts|py|js):\d+`' docs/governance/CODEMAPS.md | \
  sed 's/`//g' | while read ref; do
    file=$(echo $ref | cut -d: -f1)
    line=$(echo $ref | cut -d: -f2)
    if [ ! -f "$file" ]; then
      echo "Missing: $file"
    fi
  done

# Verify line counts are within range
wc -l automation/core/AutomationCore.ts
```

---

## 6. Quality Checklist

### Completeness

- [ ] All 6 systems have traces
- [ ] Each system has 2-4 traces minimum
- [ ] Entry points are documented
- [ ] Error handling paths included
- [ ] Integration points mapped

### Accuracy

- [ ] Line numbers verified against current code
- [ ] Function names match source
- [ ] File paths are correct
- [ ] Return types documented

### Formatting

- [ ] All code blocks have `text` language specifier
- [ ] Tables are properly aligned (use Prettier)
- [ ] Tree characters render correctly
- [ ] No orphaned branches in diagrams

### Navigation

- [ ] Table of contents links work
- [ ] Cross-references are valid
- [ ] Related documentation linked

---

## Example: Complete Trace Generation

### Input: Generate trace for `tools/cli/mcp.py` ping command

### Step 1: Read the file

```
Use mcp4_read_text_file with path "tools/cli/mcp.py"
```

### Step 2: Identify the flow

```
Use mcp13_sequentialthinking:
- Thought 1: Find CLI entry point (argparse setup)
- Thought 2: Locate ping command handler
- Thought 3: Trace subprocess execution
- Thought 4: Map telemetry recording
- Thought 5: Document return path
```

### Step 3: Create the trace

```text
Server Ping Flow
├── cmd_ping(args, manager) <-- mcp.py:XXX
│   └── manager.ping_server(server_name)
│       ├── Get server config from registry
│       ├── Build command array
│       │   └── [command] + args
│       ├── Execute subprocess with timeout
│       │   └── subprocess.run(cmd, timeout=10)
│       ├── Calculate latency
│       │   └── end_time - start_time
│       ├── Determine status
│       │   ├── SUCCESS → ServerStatus.CONNECTED
│       │   ├── TIMEOUT → ServerStatus.ERROR
│       │   └── NOT_FOUND → ServerStatus.NOT_INSTALLED
│       └── Record telemetry event
│           └── Telemetry.record_event(...)
```

### Step 4: Store in memory

```
Use mcp9_create_entities to document MCPManager
Use mcp9_create_relations to link to Telemetry
```

---

## MCP Tool Quick Reference

| Tool                        | Purpose              | When to Use                |
| --------------------------- | -------------------- | -------------------------- |
| `mcp13_sequentialthinking`  | Multi-step reasoning | Complex analysis, planning |
| `mcp9_create_entities`      | Store knowledge      | New components discovered  |
| `mcp9_create_relations`     | Map dependencies     | Integration points         |
| `mcp9_search_nodes`         | Find existing docs   | Before creating new        |
| `mcp4_read_text_file`       | Read source          | Trace generation           |
| `mcp4_directory_tree`       | Map structure        | System overview            |
| `mcp1_ask_question`         | External docs        | Library usage              |
| `mcp2_get_code_context_exa` | Code patterns        | Best practices             |

---

## Related Documentation

- [CODEMAPS.md](../governance/CODEMAPS.md) - The actual codemaps
- [AI-TOOLS-ORCHESTRATION.md](../AI-TOOLS-ORCHESTRATION.md) - Orchestration details
- [LLM-MODEL-CATALOG.md](../governance/LLM-MODEL-CATALOG.md) - Available models

---

_This guide enables any LLM with MCP access to maintain accurate architectural documentation._
