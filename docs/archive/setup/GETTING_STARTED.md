# Getting Started: MCP & Agents Setup

Complete checklist to get MCP and Agents running in your monorepo.

## Pre-requisites

- [ ] Node.js 20 or higher (`node --version`)
- [ ] pnpm installed (`pnpm --version`)
- [ ] Git repository initialized
- [ ] ~5 minutes of time

## Step 1: Review Architecture (2 min)

- [ ] Read [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- [ ] Understand the 3-layer architecture (MCPs → Agents → Orchestration)
- [ ] Know the 5 core packages

```
mcp-core → agent-core → context-provider
           → workflow-templates
           → issue-library
```

## Step 2: Initialize MCP (1 min)

```bash
# Run MCP setup script
node scripts/mcp-setup.js --install

# This will:
# ✅ Create .claude/ directory
# ✅ Create mcp-config.json
# ✅ Create agents.json
# ✅ Create orchestration.json
```

Verify:
```bash
ls -la .claude/
cat .claude/mcp-config.json
```

## Step 3: Initialize Agents (1 min)

```bash
# Run agent setup script
node scripts/agent-setup.js

# This will:
# ✅ Create agents/ directory with agent definitions
# ✅ Create workflows/ directory with workflow definitions
# ✅ Set up code-review and bug-fix workflows
```

Verify:
```bash
ls -la .claude/agents/
ls -la .claude/workflows/
cat .claude/agents/code-agent.json
```

## Step 4: Install Dependencies (1 min)

```bash
# Install all packages
pnpm install

# Build all packages
pnpm build

# Verify build
ls packages/mcp-core/dist/
ls packages/agent-core/dist/
```

## Step 5: Verify Setup

```bash
# Check MCP configuration
cat .claude/mcp-config.json | head -20

# Check agents
cat .claude/agents.json

# Check orchestration rules
cat .claude/orchestration.json

# Run a test
pnpm -F @monorepo/mcp-core test 2>/dev/null || echo "Tests not configured yet"
```

## What You Now Have

✅ **Configuration Files**
- `.claude/mcp-config.json` - MCP server config
- `.claude/agents.json` - Agent definitions
- `.claude/orchestration.json` - Orchestration rules

✅ **Agent Definitions**
- Code Agent - for code manipulation
- Analysis Agent - for testing/analysis

✅ **Workflows**
- Code Review Workflow
- Bug Fix Workflow

✅ **Core Packages** (ready to build on)
- @monorepo/mcp-core
- @monorepo/agent-core
- @monorepo/context-provider
- @monorepo/issue-library
- @monorepo/workflow-templates

✅ **Documentation**
- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - Overview
- [docs/QUICK_START.md](./docs/QUICK_START.md) - 5-minute guide
- [docs/MCP_AGENTS_ORCHESTRATION.md](./docs/MCP_AGENTS_ORCHESTRATION.md) - Complete reference
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - Design patterns
- [MCP_SERVERS_GUIDE.md](./MCP_SERVERS_GUIDE.md) - All 50+ MCPs

## Next: Customize Your Setup

### Enable More MCPs

Edit `.claude/mcp-config.json`:
```json
{
  "enabled": ["filesystem", "git", "fetch", "github"],
  "disabled": ["postgres", "brave-search"]
}
```

Common MCPs to enable:
- `github` - GitHub API
- `fetch` - Web content retrieval
- `postgres` - Database access
- `brave-search` - Web search

### Create Custom Workflow

1. Create `.claude/workflows/my-workflow.json`:
```json
{
  "id": "my-workflow",
  "name": "My Custom Workflow",
  "enabled": true,
  "steps": [
    {
      "id": "step-1",
      "name": "Check Lint",
      "type": "task",
      "agentId": "code-agent",
      "action": "lint"
    },
    {
      "id": "step-2",
      "name": "Run Tests",
      "type": "task",
      "agentId": "code-agent",
      "action": "test"
    }
  ]
}
```

2. Use in code or Claude Code:
```typescript
const orchestrator = new AgentOrchestrator();
await orchestrator.executeWorkflow('my-workflow');
```

### Create Custom Agent

1. Create agent class in `packages/my-agents/src/my-agent.ts`:
```typescript
import { BaseAgent, AgentTask, AgentResult } from '@monorepo/agent-core';

export class MyAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult> {
    // Your implementation
    return {
      success: true,
      data: { /* your result */ },
      duration: Date.now()
    };
  }
}
```

2. Register with orchestrator:
```typescript
orchestrator.registerAgent(new MyAgent(config));
```

### Use in Claude Code

Just mention what you want:
```
@Claude: Run code review workflow
@Claude: Execute the bug-fix workflow
@Claude: Analyze this code
```

Claude Code will use your configured MCPs and agents!

## Common Issues

### ❌ "MCP not found" error

```bash
# Check configuration
cat .claude/mcp-config.json

# Reinstall MCP
npm install -g @modelcontextprotocol/server-filesystem

# Make sure it's in 'enabled' array
```

### ❌ "Agent not registered"

```bash
# Check agents.json
cat .claude/agents.json

# Verify agent ID in your code matches config
# Make sure you're calling orchestrator.registerAgent()
```

### ❌ "Workflow not found"

```bash
# Check workflow files
ls .claude/workflows/

# Check workflow ID matches exactly
# Verify JSON syntax
cat .claude/workflows/my-workflow.json | jq .
```

## Recommended Reading Order

1. **IMPLEMENTATION_SUMMARY.md** (5 min) - Understand what was built
2. **docs/QUICK_START.md** (5 min) - Complete workflow
3. **docs/MCP_AGENTS_ORCHESTRATION.md** (15 min) - Deep dive
4. **docs/ARCHITECTURE.md** (10 min) - Design patterns
5. **MCP_SERVERS_GUIDE.md** (browsing) - Available MCPs

## Key Commands

```bash
# Setup
node scripts/mcp-setup.js --install
node scripts/agent-setup.js

# Development
pnpm install
pnpm build
pnpm -F @monorepo/mcp-core build
pnpm -F @monorepo/agent-core build

# Testing
pnpm test

# Linting
pnpm lint
pnpm lint:fix

# Check configuration
cat .claude/mcp-config.json
cat .claude/agents.json
cat .claude/orchestration.json
```

## DevContainer Usage

The devcontainer comes pre-configured!

**In VS Code:**
```
Cmd+Shift+P → Dev Containers: Reopen in Container
```

**Why use devcontainer:**
- ✅ Everything pre-installed
- ✅ MCP servers ready
- ✅ Consistent environment
- ✅ No local setup needed

## What's Next?

- [ ] Run setup scripts
- [ ] Enable additional MCPs
- [ ] Create custom workflows
- [ ] Try Claude Code with workflows
- [ ] Create custom agents
- [ ] Integrate into your team's process

## Documentation Map

```
GETTING_STARTED.md (you are here)
├── For 5-minute setup → docs/QUICK_START.md
├── For complete guide → docs/MCP_AGENTS_ORCHESTRATION.md
├── For architecture → docs/ARCHITECTURE.md
├── For MCP selection → MCP_SERVERS_GUIDE.md
├── For implementation details → IMPLEMENTATION_SUMMARY.md
└── For package docs → packages/*/README.md
```

## Support

**Having issues?**
1. Check [docs/QUICK_START.md](./docs/QUICK_START.md) troubleshooting section
2. Review your configuration files
3. Check that all JSON files are valid: `cat file.json | jq .`
4. Run setup scripts again

**Want to customize?**
1. Review [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for patterns
2. Check package READMEs for APIs
3. Look at example configs in `.claude/`

## Timeline

| Task | Time | Status |
|------|------|--------|
| Review architecture | 2 min | ⏱️ |
| Run setup scripts | 1 min | ⏱️ |
| Install dependencies | 1 min | ⏱️ |
| Verify setup | 1 min | ⏱️ |
| **Total** | **5 min** | ✅ |

**Extra (optional):**
- Enable more MCPs: 2 min
- Create custom workflow: 5 min
- Create custom agent: 15 min

---

**Ready?** Run `node scripts/mcp-setup.js --install` and let's go! 🚀
