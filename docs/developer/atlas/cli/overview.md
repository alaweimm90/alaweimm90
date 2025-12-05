# CLI Overview

Complete guide to the ORCHEX Command Line Interface (CLI), including command structure, concepts, usage patterns, and getting started.

---

## CLI Philosophy

The ORCHEX CLI is designed with these core principles:

- **Intuitive**: Commands follow natural language patterns
- **Composable**: Commands can be chained and scripted
- **Helpful**: Comprehensive help and auto-completion
- **Fast**: Optimized for quick execution and feedback
- **Safe**: Built-in safeguards and confirmation prompts

---

## Installation

### Quick Install

```bash
npm install -g @ORCHEX/cli
```

### Verify Installation

```bash
ORCHEX --version
# ORCHEX CLI v1.0.0

ORCHEX --help
# Display help information
```

### Auto-Completion Setup

#### Bash

```bash
echo 'eval "$(ORCHEX completion bash)"' >> ~/.bashrc
source ~/.bashrc
```

#### Zsh

```bash
echo 'eval "$(ORCHEX completion zsh)"' >> ~/.zshrc
source ~/.zshrc
```

#### Fish

```bash
ORCHEX completion fish > ~/.config/fish/completions/ORCHEX.fish
```

#### PowerShell

```powershell
ORCHEX completion powershell >> $PROFILE
```

---

## Command Structure

### Global Options

All commands support these global options:

```bash
ORCHEX [command] [subcommand] [options]

Global Options:
  -h, --help          Show help information
  -v, --version       Show version number
  --verbose           Enable verbose output
  --quiet             Suppress non-essential output
  --json              Output in JSON format
  --config <file>     Use specific config file
  --profile <name>    Use specific profile
  --no-color          Disable colored output
```

### Command Hierarchy

```
ORCHEX
├── agent          # Agent management
│   ├── register   # Register new agent
│   ├── list       # List agents
│   ├── show       # Show agent details
│   ├── update     # Update agent
│   ├── remove     # Remove agent
│   └── health     # Check agent health
├── task           # Task management
│   ├── submit     # Submit new task
│   ├── status     # Check task status
│   ├── list       # List tasks
│   ├── result     # Get task result
│   ├── cancel     # Cancel task
│   └── retry      # Retry failed task
├── analyze        # Repository analysis
│   ├── repo       # Analyze repository
│   ├── status     # Check analysis status
│   ├── report     # Get analysis report
│   └── compare    # Compare analyses
├── refactor       # Refactoring operations
│   ├── apply      # Apply refactoring
│   ├── status     # Check refactoring status
│   ├── list       # List opportunities
│   └── rollback   # Rollback refactoring
├── optimize       # Continuous optimization
│   ├── start      # Start optimization
│   ├── status     # Check optimization status
│   ├── stop       # Stop optimization
│   └── report     # Get optimization report
├── metrics        # Metrics and monitoring
│   ├── show       # Show system metrics
│   ├── agent      # Show agent metrics
│   ├── export     # Export metrics
│   └── dashboard  # Open metrics dashboard
├── config         # Configuration management
│   ├── show       # Show configuration
│   ├── set        # Set configuration value
│   ├── get        # Get configuration value
│   ├── reset      # Reset configuration
│   └── validate   # Validate configuration
├── bridge         # KILO integration
│   ├── status     # Check bridge status
│   ├── test       # Test bridge connectivity
│   ├── configure  # Configure bridges
│   └── logs       # Show bridge logs
└── system         # System management
    ├── health     # System health check
    ├── status     # System status
    ├── logs       # System logs
    ├── restart    # Restart services
    └── update     # Update ORCHEX
```

---

## Core Concepts

### Agents

Agents are AI models configured for specific tasks:

```bash
# Register an agent
ORCHEX agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --capabilities code_generation,code_review

# List agents
ORCHEX agent list

# Check agent health
ORCHEX agent health claude-sonnet-4
```

### Tasks

Tasks are units of work submitted to agents:

```bash
# Submit a task
ORCHEX task submit \
  --type code_generation \
  --description "Create authentication endpoint"

# Check status
ORCHEX task status task_abc123

# Get result
ORCHEX task result task_abc123
```

### Analysis

Repository analysis identifies improvement opportunities:

```bash
# Analyze repository
ORCHEX analyze repo . --type full

# Get analysis report
ORCHEX analyze report analysis_xyz789
```

### Refactoring

Automated code improvements with safety checks:

```bash
# Apply refactoring
ORCHEX refactor apply opp_123 --create-pr

# Check status
ORCHEX refactor status refactor_456
```

### Optimization

Continuous repository improvement:

```bash
# Start optimization
ORCHEX optimize start --schedule daily

# Check status
ORCHEX optimize status
```

---

## Usage Patterns

### Interactive Usage

For exploratory work and one-off tasks:

```bash
# Get help
ORCHEX --help
ORCHEX task --help
ORCHEX task submit --help

# Interactive task submission
ORCHEX task submit
# Follows prompts for type, description, etc.
```

### Scripting and Automation

For CI/CD pipelines and automated workflows:

```bash
#!/bin/bash
# Submit code review task
TASK_ID=$(ORCHEX task submit \
  --type code_review \
  --description "Review authentication changes" \
  --files src/auth.js \
  --json | jq -r '.data.task_id')

# Wait for completion
while true; do
  STATUS=$(ORCHEX task status $TASK_ID --json | jq -r '.data.status')
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  sleep 5
done

# Get result
ORCHEX task result $TASK_ID --json | jq '.data.result'
```

### Batch Processing

For processing multiple items:

```bash
# Process multiple files
for file in src/*.js; do
  ORCHEX task submit \
    --type code_review \
    --description "Review $file" \
    --file-path "$file" \
    --async
done

# List all running tasks
ORCHEX task list --status running --json | jq '.data.tasks[].task_id'
```

### Pipeline Integration

For CI/CD integration:

```yaml
# .github/workflows/ORCHEX.yml
name: ORCHEX Code Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install -g @ORCHEX/cli
      - run: ORCHEX agent register claude-sonnet-4 --api-key ${{ secrets.ANTHROPIC_KEY }}
      - run: ORCHEX analyze repo . --format json > analysis.json
      - run: ORCHEX task submit --type code_review --description "Review PR changes" --pr ${{ github.event.number }}
```

---

## Output Formats

### Human-Readable (Default)

```bash
ORCHEX task list
# ┌─────────────────┬─────────────────┬──────────┐
# │ Task ID         │ Type            │ Status   │
# ├─────────────────┼─────────────────┼──────────┤
# │ task_abc123     │ code_generation │ completed│
# │ task_def456     │ code_review     │ running  │
# └─────────────────┴─────────────────┴──────────┘
```

### JSON Format

```bash
ORCHEX task list --json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "task_id": "task_abc123",
        "type": "code_generation",
        "status": "completed",
        "created_at": "2025-11-29T21:00:00Z"
      }
    ]
  }
}
```

### Custom Formatting

```bash
# Custom columns
ORCHEX task list --columns task_id,type,status,duration

# Filter and format
ORCHEX task list --status completed --format table

# Export to file
ORCHEX metrics show --period 24h --output metrics.json
```

---

## Configuration

### Global Configuration

```bash
# Show current configuration
ORCHEX config show

# Set configuration values
ORCHEX config set log.level debug
ORCHEX config set cost.max_per_task 1.0

# Use different config file
ORCHEX --config ./custom-config.json task list
```

### Profiles

```bash
# Create profile for different environments
ORCHEX config profile create production
ORCHEX config profile set production api.endpoint https://api.orchex-platform.com

# Use profile
ORCHEX --profile production task list
```

### Environment Variables

```bash
# Set API keys
export ANTHROPIC_API_KEY="your-key"
export ATLAS_LOG_LEVEL="info"

# Override config
export ATLAS_CONFIG_FILE="./config.json"
```

---

## Error Handling

### Exit Codes

- `0`: Success
- `1`: General error
- `2`: Authentication error
- `3`: Validation error
- `4`: Network error
- `5`: Rate limit error

### Error Messages

```bash
ORCHEX task submit --type invalid_type
# Error: Invalid task type 'invalid_type'. Must be one of: code_generation, code_review, debugging, refactoring, documentation, testing, architecture, security_analysis
```

### Debugging

```bash
# Enable verbose output
ORCHEX --verbose task submit --type code_generation --description "test"

# Show debug information
ORCHEX config set log.level debug

# View logs
ORCHEX system logs --tail 50
```

---

## Advanced Features

### Command Chaining

```bash
# Chain commands with &&
ORCHEX agent register claude-sonnet-4 --capabilities code_generation && \
ORCHEX task submit --type code_generation --description "test"

# Use command output in scripts
TASK_ID=$(ORCHEX task submit --type code_generation --description "test" --json | jq -r '.data.task_id')
```

### Background Execution

```bash
# Run in background
ORCHEX analyze repo . --background

# Check background jobs
ORCHEX job list

# Stop background job
ORCHEX job stop <job-id>
```

### Aliases

```bash
# Create command alias
ORCHEX alias create review "task submit --type code_review"

# Use alias
ORCHEX review --description "Review this code" --files src/main.js

# List aliases
ORCHEX alias list
```

### Plugins

```bash
# Install plugin
ORCHEX plugin install @ORCHEX/plugin-gitlab

# List plugins
ORCHEX plugin list

# Use plugin commands
ORCHEX gitlab merge-request review 123
```

---

## Best Practices

### 1. Use Appropriate Task Types

```bash
# Good: Specific task type
ORCHEX task submit --type code_review --description "Security review"

# Bad: Generic description
ORCHEX task submit --type code_generation --description "Do everything"
```

### 2. Provide Context

```bash
# Good: Detailed context
ORCHEX task submit \
  --type code_generation \
  --description "Create REST API for user management" \
  --context language=typescript,framework=express,database=postgresql

# Bad: Minimal context
ORCHEX task submit --type code_generation --description "API"
```

### 3. Monitor Costs

```bash
# Check costs before submitting
ORCHEX metrics costs --period 24h

# Set cost limits
ORCHEX config set cost.max_per_task 1.0
ORCHEX config set cost.max_per_day 50.0
```

### 4. Use Batch Operations

```bash
# Process multiple files
ORCHEX analyze repo . --include "**/*.ts" --batch-size 10

# Submit multiple tasks
for file in $(find src -name "*.js"); do
  ORCHEX task submit --type code_review --file-path "$file" --async
done
```

### 5. Handle Errors Gracefully

```bash
#!/bin/bash
set -e

# Submit task with error handling
if ! TASK_ID=$(ORCHEX task submit --type code_generation --description "test" --json 2>/dev/null | jq -r '.data.task_id' 2>/dev/null); then
  echo "Failed to submit task"
  exit 1
fi

# Wait for completion with timeout
timeout=300
while [ $timeout -gt 0 ]; do
  status=$(ORCHEX task status $TASK_ID --json | jq -r '.data.status')
  if [ "$status" = "completed" ]; then
    break
  elif [ "$status" = "failed" ]; then
    echo "Task failed"
    exit 1
  fi
  sleep 5
  timeout=$((timeout - 5))
done

if [ $timeout -le 0 ]; then
  echo "Task timed out"
  exit 1
fi
```

---

## Getting Help

### Built-in Help

```bash
# General help
ORCHEX --help

# Command help
ORCHEX task --help
ORCHEX task submit --help

# Contextual help
ORCHEX task submit --help --type code_generation
```

### Community Resources

- **Documentation**: [Full Documentation](../README.md)
- **Community Forum**: [community.orchex-platform.com](https://community.orchex-platform.com)
- **GitHub Issues**: [Report bugs](https://github.com/ORCHEX-platform/ORCHEX/issues)
- **Discord**: [Real-time help](https://discord.gg/ORCHEX-platform)

### Enterprise Support

For enterprise customers:

- **Dedicated Support**: enterprise@ORCHEX-platform.com
- **SLA**: 1-hour response time
- **Training**: On-site and virtual sessions
- **Consulting**: Architecture reviews and optimization

---

This CLI overview provides the foundation for using ORCHEX effectively. Each command group has detailed documentation in the following sections. Start with [Agent Management](agents.md) to register your first AI agent!</instructions>
