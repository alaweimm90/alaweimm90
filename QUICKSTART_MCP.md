# Quick Start: MCP & Agents

## 30-Second Setup

```bash
pnpm install
pnpm run build
chmod +x scripts/setup-all.sh
./scripts/setup-all.sh
```

Done! MCPs and agents are configured.

---

## What You Got

### 5 Packages
- `@monorepo/mcp-core` - MCP management
- `@monorepo/agent-core` - Agent framework
- `@monorepo/context-provider` - Shared context
- `@monorepo/issue-library` - Issue templates
- `@monorepo/workflow-templates` - Workflows

### Configuration Files
- `.claude/mcp-config.json` - MCPs for Claude Code
- `.claude/agents.json` - Available agents
- `.claude/orchestration.json` - Automation rules

### Pre-built Workflows
- `code-review-workflow` - Lint → Type → Test → Security
- `bug-fix-workflow` - Issue → Reproduce → Fix → Verify
- `feature-development-workflow` - Full lifecycle
- `security-audit-workflow` - Comprehensive scan

### Pre-built Agents
- `code-agent` - Code tasks (review, fix, refactor)
- `analysis-agent` - Analysis (lint, test, security)
- Orchestrator - Coordinates agents

---

## Use It

### Run a Workflow
```typescript
import { AgentOrchestrator } from '@monorepo/agent-core';

const orchestrator = new AgentOrchestrator();
await orchestrator.initialize(context);

const results = await orchestrator.executeWorkflow('code-review-workflow');
```

### Create an Issue
```typescript
import { IssueManager } from '@monorepo/issue-library';

const manager = new IssueManager();
const issue = manager.createIssue('bug-report', {
  title: 'Login broken',
  description: 'Cannot log in on mobile',
  // ... other fields
});
```

### Add an MCP
```typescript
import { MCPConfigManager, MCPCategory } from '@monorepo/mcp-core';

const config = new MCPConfigManager();
config.addServer({
  name: 'github',
  command: 'npx',
  args: ['@modelcontextprotocol/server-github'],
  enabled: true,
  category: MCPCategory.DEVELOPMENT
}, 'local');
```

### Create Custom Agent
```typescript
import { BaseAgent, AgentTask, AgentResult } from '@monorepo/agent-core';

class MyAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult> {
    // Your logic
  }
}

orchestrator.registerAgent(new MyAgent(config));
```

---

## Key Files

```
.claude/mcp-config.json        ← MCPs for Claude Code
.claude/agents.json             ← Agent definitions
.claude/orchestration.json      ← Automation rules
.devcontainer/Dockerfile        ← MCP servers in container
packages/mcp-core/              ← MCP management
packages/agent-core/            ← Agent framework
packages/context-provider/      ← Shared context
packages/issue-library/         ← Issue templates
packages/workflow-templates/    ← Workflows
```

---

## Learn More

- **Full Setup Guide**: `MCP_AND_AGENTS_SETUP.md`
- **Claude Code Integration**: `CLAUDE_CODE_MCP_GUIDE.md`
- **MCP Catalog**: `MCP_SERVERS_GUIDE.md` (50+ MCPs)
- **Implementation Details**: `MCP_IMPLEMENTATION_SUMMARY.md`

---

## Common Tasks

### Enable GitHub MCP
Edit `.claude/mcp-config.json`:
```json
{
  "mcpServers": {
    "github": {
      "name": "github",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "enabled": true,
      "category": "development"
    }
  },
  "enabled": ["filesystem", "git", "github"]
}
```

### Add Custom Workflow
```typescript
import { WorkflowTemplate } from '@monorepo/workflow-templates';

const workflow: WorkflowTemplate = {
  id: 'my-workflow',
  name: 'My Workflow',
  version: '1.0.0',
  category: 'development',
  enabled: true,
  steps: [
    { id: 's1', name: 'Step 1', type: 'task', agentId: 'code-agent' }
  ]
};

workflowManager.registerTemplate(workflow);
```

### Register Custom Agent
```typescript
const agent = new MyCustomAgent(config);
orchestrator.registerAgent(agent);
```

---

## Troubleshooting

**MCPs not working?**
1. Run setup again: `./scripts/setup-all.sh`
2. Check config files exist and are valid JSON
3. Verify MCPs are installed: `npm ls -g @modelcontextprotocol/server-*`

**Agent not executing?**
1. Verify agent is registered
2. Check task type matches capabilities
3. Ensure required MCPs are enabled

**Workflow failing?**
1. Check all agents are initialized
2. Verify step agent IDs exist
3. Review workflow definition

---

## What's Next?

1. ✅ Install and configure (done!)
2. ✅ Review documentation (30 minutes)
3. ✅ Configure MCPs for your needs
4. ✅ Create custom agents
5. ✅ Define workflows for your team
6. ✅ Integrate with CI/CD

---

## Reference

**Packages** (in `packages/`)
- `mcp-core` - MCPRegistryManager, MCPConfigManager
- `agent-core` - BaseAgent, CodeAgent, AnalysisAgent, AgentOrchestrator
- `context-provider` - ContextProvider (singleton)
- `issue-library` - IssueManager, IssueTemplate
- `workflow-templates` - WorkflowManager, WorkflowTemplate

**Configuration** (in `.claude/`)
- `mcp-config.json` - MCP servers
- `agents.json` - Agent definitions
- `orchestration.json` - Automation rules

**Setup** (in `scripts/`)
- `setup-mcp.ts` - Configure MCPs
- `setup-agents.ts` - Configure agents
- `setup-all.sh` - Complete setup

**Documentation**
- This file (quick start)
- `MCP_AND_AGENTS_SETUP.md` (detailed)
- `CLAUDE_CODE_MCP_GUIDE.md` (Claude Code)
- `MCP_SERVERS_GUIDE.md` (MCP catalog)

---

**You're ready to go!** 🚀

