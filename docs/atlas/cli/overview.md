# CLI Overview

Complete guide to the ATLAS Command Line Interface (CLI), including command structure, concepts, usage patterns, and getting started.

---

## CLI Philosophy

The ATLAS CLI is designed with these core principles:

- **Intuitive**: Commands follow natural language patterns
- **Composable**: Commands can be chained and scripted
- **Helpful**: Comprehensive help and auto-completion
- **Fast**: Optimized for quick execution and feedback
- **Safe**: Built-in safeguards and confirmation prompts

---

## Installation

### Quick Install

```bash
npm install -g @atlas/cli
```

### Verify Installation

```bash
atlas --version
# ATLAS CLI v1.0.0

atlas --help
# Display help information
```

### Auto-Completion Setup

#### Bash
```bash
echo 'eval "$(atlas completion bash)"' >> ~/.bashrc
source ~/.bashrc
```

#### Zsh
```bash
echo 'eval "$(atlas completion zsh)"' >> ~/.zshrc
source ~/.zshrc
```

#### Fish
```bash
atlas completion fish > ~/.config/fish/completions/atlas.fish
```

#### PowerShell
```powershell
atlas completion powershell >> $PROFILE
```

---

## Command Structure

### Global Options

All commands support these global options:

```bash
atlas [command] [subcommand] [options]

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
atlas
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
    └── update     # Update ATLAS
```

---

## Core Concepts

### Agents

Agents are AI models configured for specific tasks:

```bash
# Register an agent
atlas agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --capabilities code_generation,code_review

# List agents
atlas agent list

# Check agent health
atlas agent health claude-sonnet-4
```

### Tasks

Tasks are units of work submitted to agents:

```bash
# Submit a task
atlas task submit \
  --type code_generation \
  --description "Create authentication endpoint"

# Check status
atlas task status task_abc123

# Get result
atlas task result task_abc123
```

### Analysis

Repository analysis identifies improvement opportunities:

```bash
# Analyze repository
atlas analyze repo . --type full

# Get analysis report
atlas analyze report analysis_xyz789
```

### Refactoring

Automated code improvements with safety checks:

```bash
# Apply refactoring
atlas refactor apply opp_123 --create-pr

# Check status
atlas refactor status refactor_456
```

### Optimization

Continuous repository improvement:

```bash
# Start optimization
atlas optimize start --schedule daily

# Check status
atlas optimize status
```

---

## Usage Patterns

### Interactive Usage

For exploratory work and one-off tasks:

```bash
# Get help
atlas --help
atlas task --help
atlas task submit --help

# Interactive task submission
atlas task submit
# Follows prompts for type, description, etc.
```

### Scripting and Automation

For CI/CD pipelines and automated workflows:

```bash
#!/bin/bash
# Submit code review task
TASK_ID=$(atlas task submit \
  --type code_review \
  --description "Review authentication changes" \
  --files src/auth.js \
  --json | jq -r '.data.task_id')

# Wait for completion
while true; do
  STATUS=$(atlas task status $TASK_ID --json | jq -r '.data.status')
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  sleep 5
done

# Get result
atlas task result $TASK_ID --json | jq '.data.result'
```

### Batch Processing

For processing multiple items:

```bash
# Process multiple files
for file in src/*.js; do
  atlas task submit \
    --type code_review \
    --description "Review $file" \
    --file-path "$file" \
    --async
done

# List all running tasks
atlas task list --status running --json | jq '.data.tasks[].task_id'
```

### Pipeline Integration

For CI/CD integration:

```yaml
# .github/workflows/atlas.yml
name: ATLAS Code Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install -g @atlas/cli
      - run: atlas agent register claude-sonnet-4 --api-key ${{ secrets.ANTHROPIC_KEY }}
      - run: atlas analyze repo . --format json > analysis.json
      - run: atlas task submit --type code_review --description "Review PR changes" --pr ${{ github.event.number }}
```

---

## Output Formats

### Human-Readable (Default)

```bash
atlas task list
# ┌─────────────────┬─────────────────┬──────────┐
# │ Task ID         │ Type            │ Status   │
# ├─────────────────┼─────────────────┼──────────┤
# │ task_abc123     │ code_generation │ completed│
# │ task_def456     │ code_review     │ running  │
# └─────────────────┴─────────────────┴──────────┘
```

### JSON Format

```bash
atlas task list --json
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
atlas task list --columns task_id,type,status,duration

# Filter and format
atlas task list --status completed --format table

# Export to file
atlas metrics show --period 24h --output metrics.json
```

---

## Configuration

### Global Configuration

```bash
# Show current configuration
atlas config show

# Set configuration values
atlas config set log.level debug
atlas config set cost.max_per_task 1.0

# Use different config file
atlas --config ./custom-config.json task list
```

### Profiles

```bash
# Create profile for different environments
atlas config profile create production
atlas config profile set production api.endpoint https://api.atlas-platform.com

# Use profile
atlas --profile production task list
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
atlas task submit --type invalid_type
# Error: Invalid task type 'invalid_type'. Must be one of: code_generation, code_review, debugging, refactoring, documentation, testing, architecture, security_analysis
```

### Debugging

```bash
# Enable verbose output
atlas --verbose task submit --type code_generation --description "test"

# Show debug information
atlas config set log.level debug

# View logs
atlas system logs --tail 50
```

---

## Advanced Features

### Command Chaining

```bash
# Chain commands with &&
atlas agent register claude-sonnet-4 --capabilities code_generation && \
atlas task submit --type code_generation --description "test"

# Use command output in scripts
TASK_ID=$(atlas task submit --type code_generation --description "test" --json | jq -r '.data.task_id')
```

### Background Execution

```bash
# Run in background
atlas analyze repo . --background

# Check background jobs
atlas job list

# Stop background job
atlas job stop <job-id>
```

### Aliases

```bash
# Create command alias
atlas alias create review "task submit --type code_review"

# Use alias
atlas review --description "Review this code" --files src/main.js

# List aliases
atlas alias list
```

### Plugins

```bash
# Install plugin
atlas plugin install @atlas/plugin-gitlab

# List plugins
atlas plugin list

# Use plugin commands
atlas gitlab merge-request review 123
```

---

## Best Practices

### 1. Use Appropriate Task Types

```bash
# Good: Specific task type
atlas task submit --type code_review --description "Security review"

# Bad: Generic description
atlas task submit --type code_generation --description "Do everything"
```

### 2. Provide Context

```bash
# Good: Detailed context
atlas task submit \
  --type code_generation \
  --description "Create REST API for user management" \
  --context language=typescript,framework=express,database=postgresql

# Bad: Minimal context
atlas task submit --type code_generation --description "API"
```

### 3. Monitor Costs

```bash
# Check costs before submitting
atlas metrics costs --period 24h

# Set cost limits
atlas config set cost.max_per_task 1.0
atlas config set cost.max_per_day 50.0
```

### 4. Use Batch Operations

```bash
# Process multiple files
atlas analyze repo . --include "**/*.ts" --batch-size 10

# Submit multiple tasks
for file in $(find src -name "*.js"); do
  atlas task submit --type code_review --file-path "$file" --async
done
```

### 5. Handle Errors Gracefully

```bash
#!/bin/bash
set -e

# Submit task with error handling
if ! TASK_ID=$(atlas task submit --type code_generation --description "test" --json 2>/dev/null | jq -r '.data.task_id' 2>/dev/null); then
  echo "Failed to submit task"
  exit 1
fi

# Wait for completion with timeout
timeout=300
while [ $timeout -gt 0 ]; do
  status=$(atlas task status $TASK_ID --json | jq -r '.data.status')
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
atlas --help

# Command help
atlas task --help
atlas task submit --help

# Contextual help
atlas task submit --help --type code_generation
```

### Community Resources

- **Documentation**: [Full Documentation](../README.md)
- **Community Forum**: [community.atlas-platform.com](https://community.atlas-platform.com)
- **GitHub Issues**: [Report bugs](https://github.com/atlas-platform/atlas/issues)
- **Discord**: [Real-time help](https://discord.gg/atlas-platform)

### Enterprise Support

For enterprise customers:
- **Dedicated Support**: enterprise@atlas-platform.com
- **SLA**: 1-hour response time
- **Training**: On-site and virtual sessions
- **Consulting**: Architecture reviews and optimization

---

This CLI overview provides the foundation for using ATLAS effectively. Each command group has detailed documentation in the following sections. Start with [Agent Management](agents.md) to register your first AI agent!</instructions>