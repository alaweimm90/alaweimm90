# Quick Start Guide

Get started with ORCHEX in 5 minutes! This guide will walk you through installing ORCHEX, registering your first AI agent, and submitting your first task.

---

## Prerequisites

Before you begin, ensure you have:

- **Node.js 16+** installed ([download here](https://nodejs.org/))
- **npm** or **yarn** package manager
- **API keys** for at least one AI provider (Anthropic, OpenAI, or Google)

---

## Step 1: Install ORCHEX CLI

Install the ORCHEX command-line interface globally:

```bash
npm install -g @ORCHEX/cli
```

Verify the installation:

```bash
ORCHEX --version
# Should output: ORCHEX CLI v1.0.0
```

---

## Step 2: Initialize ORCHEX

Navigate to your project directory and initialize ORCHEX:

```bash
cd your-project-directory
ORCHEX init
```

This creates the necessary configuration files and directories:

- `.orchex/` - ORCHEX configuration directory
- `.orchex/config.json` - Main configuration file
- `.orchex/agents/` - Agent registry storage

---

## Step 3: Register Your First Agent

Register an AI agent with ORCHEX. Choose from Claude, GPT-4, or Gemini:

### Option A: Register Claude (Anthropic)

```bash
ORCHEX agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --model claude-sonnet-4.5 \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_ANTHROPIC_API_KEY
```

### Option B: Register GPT-4 (OpenAI)

```bash
ORCHEX agent register gpt-4-turbo \
  --name "GPT-4 Turbo" \
  --provider openai \
  --model gpt-4-turbo \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_OPENAI_API_KEY
```

### Option C: Register Gemini (Google)

```bash
ORCHEX agent register gemini-pro \
  --name "Gemini Pro" \
  --provider google \
  --model gemini-pro \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_GOOGLE_API_KEY
```

**Security Note:** For production use, set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
ORCHEX agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --model claude-sonnet-4.5 \
  --capabilities code_generation,code_review,refactoring,debugging
```

---

## Step 4: Verify Agent Registration

Check that your agent was registered successfully:

```bash
ORCHEX agent list
```

You should see output similar to:

```
Registered Agents:
┌─────────────────┬─────────────────┬──────────┬─────────┐
│ Agent ID        │ Name            │ Provider │ Status  │
├─────────────────┼─────────────────┼──────────┼─────────┤
│ claude-sonnet-4 │ Claude Sonnet 4 │ anthropic│ healthy │
└─────────────────┴─────────────────┴──────────┴─────────┘
```

Test the agent health:

```bash
ORCHEX agent health claude-sonnet-4
```

---

## Step 5: Submit Your First Task

Submit a code generation task:

```bash
ORCHEX task submit \
  --type code_generation \
  --description "Create a REST API endpoint for user authentication in Node.js/Express" \
  --context language=javascript,framework=express \
  --priority high
```

ORCHEX will:

1. Analyze your request
2. Select the best agent (Claude Sonnet 4)
3. Execute the task
4. Return the generated code

---

## Step 6: Monitor Task Progress

Check the status of your task:

```bash
ORCHEX task status <task-id>
# Replace <task-id> with the ID returned from the submit command
```

Or list all tasks:

```bash
ORCHEX task list --status running
```

---

## Step 7: View Results

Once the task completes, view the results:

```bash
ORCHEX task result <task-id>
```

The output will include:

- Generated code
- Implementation explanation
- Usage examples
- Best practices recommendations

---

## Step 8: Try Advanced Features

### Code Review Task

```bash
ORCHEX task submit \
  --type code_review \
  --description "Review this authentication endpoint for security vulnerabilities" \
  --file-path src/auth.js \
  --priority medium
```

### Repository Analysis

```bash
ORCHEX analyze repo . --type quick
```

This performs a quick analysis of your codebase and provides:

- Code complexity metrics
- Technical debt assessment
- Refactoring opportunities

---

## Next Steps

🎉 **Congratulations!** You've successfully set up ORCHEX and completed your first AI-assisted development task.

### What to Explore Next

1. **[Register More Agents](first-tasks.md#multi-agent-setup)** - Add multiple AI agents for better task routing
2. **[Explore Task Types](first-tasks.md#task-types)** - Try debugging, refactoring, and documentation tasks
3. **[Configure Advanced Settings](configuration.md)** - Set up cost limits, performance monitoring, and custom routing rules
4. **[Integrate with CI/CD](integration/cicd-integration.md)** - Automate code quality checks in your pipeline
5. **[Set Up KILO Integration](integration/kilo-integration.md)** - Enable governance and compliance validation

### Useful Commands

```bash
# Get help on any command
ORCHEX --help
ORCHEX task --help

# View system status
ORCHEX status

# Check agent performance metrics
ORCHEX metrics agents

# View recent task history
ORCHEX task list --limit 10
```

---

## Troubleshooting

### Common Issues

**"Command not found: ORCHEX"**

- Ensure npm global packages are in your PATH
- Try `npx @ORCHEX/cli` instead of `ORCHEX`

**"Agent registration failed"**

- Verify your API key is valid
- Check your internet connection
- Ensure the API key has sufficient permissions

**"Task submission failed"**

- Verify an agent is registered and healthy
- Check your task description is clear and specific
- Ensure you have sufficient API quota

### Getting Help

- **Documentation**: [Full Documentation](../README.md)
- **CLI Help**: `ORCHEX --help` or `ORCHEX <command> --help`
- **Community**: Join our [Discord community](https://discord.gg/ORCHEX-platform)
- **Issues**: Report bugs on [GitHub](https://github.com/ORCHEX-platform/ORCHEX/issues)

---

**Ready for more?** Continue to [First Tasks](first-tasks.md) to explore ORCHEX capabilities in depth!</instructions>
