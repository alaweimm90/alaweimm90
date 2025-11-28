# AI Tools Orchestration Guide

This document describes the multi-agent AI tool orchestration system for the alaweimm90 portfolio governance infrastructure.

## Overview

The orchestration system enables seamless collaboration between 14 AI coding assistants, providing:

- **Intelligent task routing** - Route tasks to the best-fit tool based on intent
- **Handoff context preservation** - Maintain context across tool switches
- **Hallucination prevention** - Three-layer verification cascade
- **Checkpoint-based recovery** - Restore from known-good states on failure
- **Graceful degradation** - Fallback strategies when tools fail
- **Telemetry & metrics** - Track success rates and performance

## Architecture

```
.ai/
├── settings.yaml           # Unified tool configuration with routing
├── context.md              # Shared context for all tools
├── mcp/
│   └── server-registry.yaml  # MCP server definitions
├── rules/                  # Tool-specific rules
└── <tool>/                 # Individual tool configs

.metaHub/
├── policies/
│   ├── orchestration-governance.yaml  # Orchestration policy
│   └── mcp-governance.yaml            # MCP governance policy
├── schemas/
│   └── handoff-envelope-schema.json   # Handoff validation schema
├── orchestration/
│   ├── checkpoints/        # Workflow checkpoints
│   ├── state/              # Workflow state files
│   └── telemetry/          # Metrics data
└── scripts/
    ├── orchestration_checkpoint.py   # Checkpoint management
    ├── orchestration_validator.py    # Handoff validation
    ├── orchestration_telemetry.py    # Metrics collection
    ├── hallucination_verifier.py     # Output verification
    └── self_healing_workflow.py      # Error recovery
```

## Tool Routing

The system routes tasks to the optimal tool based on task type and intent extraction.

### Task Type Mappings

| Task Type | Primary Tool | Secondary Tools | Description |
|-----------|--------------|-----------------|-------------|
| Architecture | claude_code | kilo_code | System design, planning, technical decisions |
| Implementation | cline | cursor, aider | Building features, writing code |
| Rapid Prototyping | blackbox | cursor | Quick snippets, boilerplate |
| Debugging | cline | claude_code, aider | Bug fixes, troubleshooting |
| Refactoring | aider | cline, windsurf | Code improvements, cleanup |
| Testing | cline | aider | Test writing, coverage |
| Research | claude_code | kilo_code, gemini | Codebase exploration, analysis |

### Intent Extraction Keywords

```yaml
architecture: [design, architect, plan, structure, system, diagram]
implementation: [build, implement, create, develop, code, write]
refactoring: [refactor, improve, optimize, clean, simplify]
debugging: [fix, debug, error, issue, bug, broken, failing]
rapid: [quick, fast, prototype, snippet, boilerplate, template]
testing: [test, coverage, spec, unit, integration, e2e]
research: [find, search, understand, explore, analyze, investigate]
```

### Using the Router

```bash
# Route a task to the best tool
python orchestration_validator.py route "implement user authentication"

# Check routing between tools
python orchestration_validator.py check-routing --source cline --target kilo
```

## Handoff Envelopes

When switching between tools, use structured handoff envelopes to preserve context.

### Envelope Structure

```json
{
  "metadata": {
    "source_tool": "claude_code",
    "target_tool": "cline",
    "timestamp": "2025-11-27T10:30:00Z",
    "correlation_id": "uuid-here",
    "workflow_name": "feature-auth"
  },
  "context": {
    "task_description": "Implement user authentication flow",
    "relevant_files": ["src/auth/", "src/api/users.py"],
    "prior_decisions": [
      {"decision": "Use JWT tokens", "rationale": "Stateless, scalable"}
    ],
    "success_criteria": ["Users can login", "Tokens expire correctly"]
  },
  "artifacts": {
    "files_modified": [
      {"path": "src/auth/jwt.py", "action": "created", "lines_changed": 45}
    ],
    "validation_results": {
      "tests_passed": true,
      "lint_passed": true
    }
  },
  "instructions": {
    "expected_action": "Implement login UI component",
    "boundary_conditions": ["Don't modify database schema"],
    "rollback_instructions": "Revert src/auth/ changes"
  }
}
```

### Validating Envelopes

```bash
# Validate a handoff envelope
python orchestration_validator.py validate handoff.json
```

## Checkpoint System

Create checkpoints before risky operations to enable recovery.

### Creating Checkpoints

```bash
# Create checkpoint before handoff
python orchestration_checkpoint.py create \
  --workflow "feature-auth" \
  --tool cline \
  --task "Implement login flow" \
  --files src/auth/login.py src/api/auth.py
```

### Recovery from Checkpoints

```bash
# List available checkpoints
python orchestration_checkpoint.py list --workflow "feature-auth"

# Validate checkpoint integrity
python orchestration_checkpoint.py validate --id abc123

# Restore from checkpoint (dry run)
python orchestration_checkpoint.py restore --id abc123 --dry-run

# Execute restore
python orchestration_checkpoint.py restore --id abc123 --execute
```

## Hallucination Prevention

The system uses a three-layer verification cascade to detect hallucinations.

### Verification Layers

1. **Semantic Grounding** (40% weight)
   - Output must reference provided context
   - Flags absolute language (always, never, clearly)

2. **Entity Verification** (35% weight)
   - Mentioned files must exist
   - Referenced functions must be in codebase

3. **Claim Verification** (25% weight)
   - Flags unverified statistical claims
   - Identifies absolute statements

### Using the Verifier

```bash
# Verify AI output text
python hallucination_verifier.py verify output.json --context context.json

# Check entity references
python hallucination_verifier.py check-entities --text "The login function in auth.py" --workspace .

# Analyze claims
python hallucination_verifier.py analyze-claims --text "This will always work perfectly"
```

### Confidence Threshold

The default confidence threshold is **0.8** (80%). Outputs below this threshold are flagged for review.

## Error Recovery & Self-Healing

The system automatically handles failures with intelligent recovery strategies.

### Error Categories

| Category | Examples | Recovery Action |
|----------|----------|-----------------|
| Transient | timeout, rate_limit | Retry with backoff |
| Context | context_window_exceeded | Compress context |
| Permanent | authentication_failed | Escalate to human |

### Degradation Levels

| Level | Description | Available Tools |
|-------|-------------|-----------------|
| 1 - Full | All capabilities | All 14 tools |
| 2 - Reduced | Sequential execution | All tools, reduced context |
| 3 - Primary Only | Limited tools | claude_code, aider, cline |
| 4 - Safe Mode | Human approval required | aider only |

### Using Self-Healing

```bash
# Execute recovery for a failure
python self_healing_workflow.py recover \
  --workflow "feature-x" \
  --error "timeout connecting to API" \
  --tool cline

# Check workflow status
python self_healing_workflow.py status --workflow "feature-x"

# Set degradation level
python self_healing_workflow.py degrade --workflow "feature-x" --level 2

# Recover from checkpoint
python self_healing_workflow.py from-checkpoint \
  --workflow "feature-x" \
  --checkpoint abc123
```

## Telemetry & Metrics

Track orchestration performance with built-in telemetry.

### Recording Events

```bash
# Record a handoff
python orchestration_telemetry.py record \
  --event handoff \
  --status success \
  --tool "cline->aider" \
  --duration 150

# Record tool invocation
python orchestration_telemetry.py record \
  --event tool_invocation \
  --status success \
  --tool claude_code \
  --duration 5000
```

### Viewing Reports

```bash
# Generate report for last 24 hours
python orchestration_telemetry.py report --period 24h

# Check against target thresholds
python orchestration_telemetry.py check-targets
```

### Target Metrics

| Metric | Target |
|--------|--------|
| Handoff Success Rate | ≥95% |
| Hallucination Rate | ≤2% |
| Workflow Completion | ≥90% |
| First Pass Success | ≥80% |

## MCP Server Integration

The system integrates with Model Context Protocol (MCP) servers for enhanced capabilities.

### Available Servers

| Server | Category | Use Case |
|--------|----------|----------|
| context | infrastructure | Handoff state, memory |
| github | code_management | PRs, issues, workflows |
| filesystem | code_management | File operations |
| puppeteer | browser_automation | E2E testing |
| sqlite/postgres | data_processing | Database operations |
| brave_search | search | Web research |

### Server Registry

See `.ai/mcp/server-registry.yaml` for full server definitions and configurations.

## Agent Framework Integration

The orchestration system integrates with existing multi-agent frameworks:

### MeatheadPhysicist
- **Use for**: Academic research, scientific analysis
- **Agents**: Scientist, Literature, Scout, Theory, Critic

### Turingo
- **Use for**: Performance optimization, algorithm design
- **Agents**: QuantumQuokka, MLMagician, PuzzleProdigy, CodeCowboy

### IdeaForge
- **Use for**: Brainstorming, innovation
- **Agents**: WildCard, ChaosMonkey, Futurist, MetaOrchestrator

### ATLAS
- **Use for**: Complex multi-step analysis
- **Agents**: Coordinator, Analyst, Synthesizer

## Best Practices

### 1. Always Create Checkpoints Before Handoffs
```bash
# Before switching tools
python orchestration_checkpoint.py create --workflow "my-feature" --tool current_tool
```

### 2. Use Structured Handoff Envelopes
Don't just switch tools - pass context through validated envelopes.

### 3. Monitor Hallucination Rates
Regularly check telemetry for hallucination spikes:
```bash
python orchestration_telemetry.py check-targets
```

### 4. Progressive Degradation
Let the system degrade gracefully instead of failing completely.

### 5. Review Flagged Claims
When the verifier flags claims, review before proceeding.

## Troubleshooting

### High Hallucination Rate
- Check if context is being passed correctly
- Verify file references exist
- Review flagged claims in output

### Frequent Handoff Failures
- Check tool availability
- Verify envelope schema compliance
- Review context size (max 500KB)

### Checkpoint Validation Failures
- Run integrity check: `python orchestration_checkpoint.py validate --id X`
- Check file snapshots exist
- Verify checksum matches

### Degradation Level Stuck
- Reset workflow: `python self_healing_workflow.py reset --workflow X`
- Check error patterns in recovery log

## Related Documentation

- [ROOT_STRUCTURE_CONTRACT.md](ROOT_STRUCTURE_CONTRACT.md) - Repository structure
- [.ai/settings.yaml](../.ai/settings.yaml) - Tool configurations
- [orchestration-governance.yaml](../.metaHub/policies/orchestration-governance.yaml) - Full policy

---

*Last updated: 2025-11-27*
