# Quick Start Guide

Get started with ATLAS in 5 minutes! This guide will walk you through installing ATLAS, registering your first AI agent, and submitting your first task.

---

## Prerequisites

Before you begin, ensure you have:

- **Node.js 16+** installed ([download here](https://nodejs.org/))
- **npm** or **yarn** package manager
- **API keys** for at least one AI provider (Anthropic, OpenAI, or Google)

---

## Step 1: Install ATLAS CLI

Install the ATLAS command-line interface globally:

```bash
npm install -g @atlas/cli
```

Verify the installation:

```bash
atlas --version
# Should output: ATLAS CLI v1.0.0
```

---

## Step 2: Initialize ATLAS

Navigate to your project directory and initialize ATLAS:

```bash
cd your-project-directory
atlas init
```

This creates the necessary configuration files and directories:

- `.atlas/` - ATLAS configuration directory
- `.atlas/config.json` - Main configuration file
- `.atlas/agents/` - Agent registry storage

---

## Step 3: Register Your First Agent

Register an AI agent with ATLAS. Choose from Claude, GPT-4, or Gemini:

### Option A: Register Claude (Anthropic)

```bash
atlas agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --model claude-sonnet-4.5 \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_ANTHROPIC_API_KEY
```

### Option B: Register GPT-4 (OpenAI)

```bash
atlas agent register gpt-4-turbo \
  --name "GPT-4 Turbo" \
  --provider openai \
  --model gpt-4-turbo \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_OPENAI_API_KEY
```

### Option C: Register Gemini (Google)

```bash
atlas agent register gemini-pro \
  --name "Gemini Pro" \
  --provider google \
  --model gemini-pro \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_GOOGLE_API_KEY
```

**Security Note:** For production use, set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
atlas agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --model claude-sonnet-4.5 \
  --capabilities code_generation,code_review,refactoring,debugging
```

---

## Step 4: Verify Agent Registration

Check that your agent was registered successfully:

```bash
atlas agent list
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
atlas agent health claude-sonnet-4
```

---

## Step 5: Submit Your First Task

Submit a code generation task:

```bash
atlas task submit \
  --type code_generation \
  --description "Create a REST API endpoint for user authentication in Node.js/Express" \
  --context language=javascript,framework=express \
  --priority high
```

ATLAS will:

1. Analyze your request
2. Select the best agent (Claude Sonnet 4)
3. Execute the task
4. Return the generated code

---

## Step 6: Monitor Task Progress

Check the status of your task:

```bash
atlas task status <task-id>
# Replace <task-id> with the ID returned from the submit command
```

Or list all tasks:

```bash
atlas task list --status running
```

---

## Step 7: View Results

Once the task completes, view the results:

```bash
atlas task result <task-id>
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
atlas task submit \
  --type code_review \
  --description "Review this authentication endpoint for security vulnerabilities" \
  --file-path src/auth.js \
  --priority medium
```

### Repository Analysis

```bash
atlas analyze repo . --type quick
```

This performs a quick analysis of your codebase and provides:

- Code complexity metrics
- Technical debt assessment
- Refactoring opportunities

---

## Next Steps

🎉 **Congratulations!** You've successfully set up ATLAS and completed your first AI-assisted development task.

### What to Explore Next

1. **[Register More Agents](first-tasks.md#multi-agent-setup)** - Add multiple AI agents for better task routing
2. **[Explore Task Types](first-tasks.md#task-types)** - Try debugging, refactoring, and documentation tasks
3. **[Configure Advanced Settings](configuration.md)** - Set up cost limits, performance monitoring, and custom routing rules
4. **[Integrate with CI/CD](integration/cicd-integration.md)** - Automate code quality checks in your pipeline
5. **[Set Up KILO Integration](integration/kilo-integration.md)** - Enable governance and compliance validation

### Useful Commands

```bash
# Get help on any command
atlas --help
atlas task --help

# View system status
atlas status

# Check agent performance metrics
atlas metrics agents

# View recent task history
atlas task list --limit 10
```

---

## Troubleshooting

### Common Issues

**"Command not found: atlas"**

- Ensure npm global packages are in your PATH
- Try `npx @atlas/cli` instead of `atlas`

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
- **CLI Help**: `atlas --help` or `atlas <command> --help`
- **Community**: Join our [Discord community](https://discord.gg/atlas-platform)
- **Issues**: Report bugs on [GitHub](https://github.com/atlas-platform/atlas/issues)

---

**Ready for more?** Continue to [First Tasks](first-tasks.md) to explore ATLAS capabilities in depth!</instructions>
