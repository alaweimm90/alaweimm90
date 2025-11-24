# MCP & Agent Architecture Setup Guide

## Overview

This monorepo now includes a comprehensive, abstract MCP (Model Context Protocol) and Agent framework that can be used across all projects and integrated with Claude Code, devcontainers, and other environments.

## Architecture Components

### 1. Core Packages

#### `packages/mcp-core`
Manages MCP server configurations and registry.

**Key Classes:**
- `MCPRegistryManager` - Registry of available MCP servers
- `MCPConfigManager` - Configuration management for different environments

**Usage:**
```typescript
import { MCPRegistryManager, MCPConfigManager } from '@monorepo/mcp-core';

const registry = new MCPRegistryManager();
const config = new MCPConfigManager();

// Add an MCP server
config.addServer({
  name: 'github',
  description: 'GitHub API access',
  command: 'npx',
  args: ['@modelcontextprotocol/server-github'],
  enabled: true,
  category: MCPCategory.DEVELOPMENT
}, 'local');
```

#### `packages/agent-core`
Framework for creating and orchestrating agents.

**Key Classes:**
- `BaseAgent` - Abstract base class for all agents
- `CodeAgent` - Specialized for code tasks
- `AnalysisAgent` - Specialized for analysis tasks
- `AgentOrchestrator` - Manages agent execution and workflows

**Usage:**
```typescript
import { BaseAgent, AgentOrchestrator, AgentTask, TaskStatus } from '@monorepo/agent-core';

const orchestrator = new AgentOrchestrator();
const codeAgent = new CodeAgent({
  id: 'code-agent',
  name: 'Code Agent',
  type: AgentType.CODE,
  capabilities: ['lint', 'test', 'review'],
  enabled: true
});

orchestrator.registerAgent(codeAgent);

const task: AgentTask = {
  id: 'task-1',
  name: 'Lint Code',
  type: 'lint',
  status: TaskStatus.PENDING
};

const result = await orchestrator.executeTask(task);
```

#### `packages/context-provider`
Shared context provider as a singleton.

**Key Classes:**
- `ContextProvider` - Singleton for shared context across agents

**Usage:**
```typescript
import { ContextProvider } from '@monorepo/context-provider';

const context = ContextProvider.getInstance();
const workspaceRoot = context.getWorkspaceRoot();
context.setMetadata('projectId', 'my-project');
```

#### `packages/issue-library`
Issue template management system.

**Key Classes:**
- `IssueManager` - Manages issue templates and creation
- `createDefaultTemplates()` - Creates standard templates (Bug, Feature, Refactor)

**Usage:**
```typescript
import { IssueManager, IssueType, IssuePriority } from '@monorepo/issue-library';

const issueManager = new IssueManager();
const issue = issueManager.createIssue('bug-report', {
  title: 'Login button not working',
  description: 'Users cannot login',
  steps: '1. Click login\n2. See error',
  expected: 'Should redirect to dashboard',
  actual: 'Shows error message'
});
```

#### `packages/workflow-templates`
Workflow and orchestration templates.

**Key Classes:**
- `WorkflowManager` - Manages workflow templates
- `createDefaultTemplates()` - Creates standard workflows

**Included Workflows:**
- `code-review-workflow` - Lint, type check, test, security scan
- `bug-fix-workflow` - Issue creation, reproduction, fix, verification
- `feature-development-workflow` - Full feature development lifecycle
- `security-audit-workflow` - Comprehensive security audit

**Usage:**
```typescript
import { WorkflowManager } from '@monorepo/workflow-templates';

const workflowManager = new WorkflowManager();
const workflow = workflowManager.createWorkflow('code-review-workflow', {
  name: 'PR Code Review'
});

await orchestrator.executeWorkflow(workflow.id);
```

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
cd c:\Users\mesha\Desktop\GitHub
pnpm install
```

### Step 2: Build All Packages

```bash
pnpm run build
```

### Step 3: Configure MCP Servers (Local)

Create `c:\Users\mesha\Desktop\GitHub\.claude\mcp-config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "name": "filesystem",
      "description": "Secure file operations",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem"],
      "enabled": true,
      "category": "core"
    },
    "git": {
      "name": "git",
      "description": "Git operations",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git"],
      "enabled": true,
      "category": "core"
    }
  },
  "enabled": ["filesystem", "git"],
  "disabled": []
}
```

### Step 4: Configure Agents

Create `c:\Users\mesha\Desktop\GitHub\.claude\agents.json`:

```json
{
  "agents": [
    {
      "id": "code-agent",
      "name": "Code Agent",
      "type": "code",
      "capabilities": ["code-review", "code-fix", "refactor"],
      "enabled": true
    },
    {
      "id": "analysis-agent",
      "name": "Analysis Agent",
      "type": "analysis",
      "capabilities": ["analyze", "test", "security-scan"],
      "enabled": true
    }
  ]
}
```

### Step 5: Configure Orchestration

Create `c:\Users\mesha\Desktop\GitHub\.claude\orchestration.json`:

```json
{
  "version": "1.0.0",
  "rules": [
    {
      "id": "code-review-rule",
      "name": "Code Review on PR",
      "trigger": "pull_request_opened",
      "actions": [
        {
          "type": "execute",
          "target": "code-review-workflow"
        }
      ],
      "enabled": true
    }
  ]
}
```

---

## DevContainer Setup

### Update Dockerfile

The `.devcontainer/Dockerfile` should include MCP server support:

```dockerfile
# Install MCP server tools
RUN npm install -g @modelcontextprotocol/sdk

# Install specific MCP servers (optional, can be done per-project)
RUN npm install -g \
  @modelcontextprotocol/server-filesystem \
  @modelcontextprotocol/server-git
```

### Environment Variables

Add to `.devcontainer/.env`:

```env
MCP_SERVERS=filesystem,git,fetch
WORKSPACE_ROOT=/workspace
```

---

## Claude Code Integration

### Configuration Flow

1. **Local Config** (`.claude/mcp-config.json`) - Project-specific MCPs
2. **DevContainer Config** (`.devcontainer/mcp-config.json`) - Container-specific setup
3. **Claude Code Config** (`~/.config/trae/claude-code-mcp.json` on macOS/Linux or `%APPDATA%\Trae\User\claude-code-mcp.json` on Windows)

Configurations are merged with priority: Local > DevContainer > Global

### Using MCPs with Claude Code

Once configured, MCPs become available in Claude Code:
- File operations via `filesystem` MCP
- Git operations via `git` MCP
- Web content via `fetch` MCP
- etc.

---

## Creating Custom MCPs

### Add a New MCP Server

```typescript
import { MCPConfigManager, MCPCategory } from '@monorepo/mcp-core';

const configManager = new MCPConfigManager();

configManager.addServer({
  name: 'my-custom-mcp',
  description: 'My custom MCP server',
  command: 'node',
  args: ['./my-mcp-server.js'],
  enabled: true,
  category: MCPCategory.SPECIALIZED,
  version: '1.0.0'
}, 'local');
```

### Register in MCP Config

```json
{
  "mcpServers": {
    "my-custom-mcp": {
      "name": "my-custom-mcp",
      "command": "node",
      "args": ["./my-mcp-server.js"],
      "enabled": true,
      "category": "specialized"
    }
  },
  "enabled": ["my-custom-mcp"]
}
```

---

## Creating Custom Agents

### Extend BaseAgent

```typescript
import { BaseAgent, AgentConfig, AgentTask, AgentResult } from '@monorepo/agent-core';

export class MyCustomAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult> {
    const startTime = Date.now();
    try {
      // Implement custom logic
      const result = await this.doSomething(task);
      return {
        success: true,
        data: result,
        duration: Date.now() - startTime
      };
    } catch (error) {
      return {
        success: false,
        error: error as Error,
        duration: Date.now() - startTime
      };
    }
  }

  private async doSomething(task: AgentTask): Promise<unknown> {
    // Custom implementation
    return { message: 'Done' };
  }
}
```

### Register with Orchestrator

```typescript
const customAgent = new MyCustomAgent({
  id: 'my-agent',
  name: 'My Custom Agent',
  type: 'custom',
  capabilities: ['my-capability'],
  enabled: true
});

orchestrator.registerAgent(customAgent);
```

---

## Creating Custom Workflows

### Define Workflow Template

```typescript
import { WorkflowTemplate } from '@monorepo/workflow-templates';

const myWorkflow: WorkflowTemplate = {
  id: 'my-workflow',
  name: 'My Workflow',
  description: 'My custom workflow',
  version: '1.0.0',
  category: 'development',
  tags: ['custom'],
  difficulty: 'beginner',
  enabled: true,
  steps: [
    {
      id: 'step-1',
      name: 'Step 1',
      type: 'task',
      agentId: 'code-agent',
      action: 'my-action'
    },
    {
      id: 'step-2',
      name: 'Step 2',
      type: 'task',
      agentId: 'analysis-agent',
      action: 'my-analysis'
    }
  ]
};
```

### Register Workflow

```typescript
import { WorkflowManager } from '@monorepo/workflow-templates';

const workflowManager = new WorkflowManager();
workflowManager.registerTemplate(myWorkflow);

// Execute
const results = await orchestrator.executeWorkflow(myWorkflow.id);
```

---

## Monorepo Integration

### Using Packages in Your Apps

In any app's `package.json`:

```json
{
  "dependencies": {
    "@monorepo/mcp-core": "workspace:*",
    "@monorepo/agent-core": "workspace:*",
    "@monorepo/context-provider": "workspace:*",
    "@monorepo/issue-library": "workspace:*",
    "@monorepo/workflow-templates": "workspace:*"
  }
}
```

### Turbo Tasks

All packages support:
- `pnpm run build` - Build all packages
- `pnpm run test` - Test all packages
- `pnpm run lint` - Lint all packages
- `pnpm run type-check` - Type check all packages

---

## Example: Complete Workflow

```typescript
import { ContextProvider } from '@monorepo/context-provider';
import { MCPConfigManager } from '@monorepo/mcp-core';
import { AgentOrchestrator, CodeAgent, AnalysisAgent } from '@monorepo/agent-core';
import { WorkflowManager } from '@monorepo/workflow-templates';

async function main() {
  // 1. Initialize context
  const context = ContextProvider.getInstance();

  // 2. Load MCP configuration
  const configManager = new MCPConfigManager();
  const mcpConfig = configManager.mergeConfigs();
  console.log('MCPs enabled:', mcpConfig.enabled);

  // 3. Create orchestrator
  const orchestrator = new AgentOrchestrator();
  await orchestrator.initialize(context.getContext());

  // 4. Register agents
  orchestrator.registerAgent(new CodeAgent({
    id: 'code-agent',
    name: 'Code Agent',
    type: 'code',
    capabilities: ['code-review', 'test'],
    enabled: true,
    version: '1.0.0'
  }));

  orchestrator.registerAgent(new AnalysisAgent({
    id: 'analysis-agent',
    name: 'Analysis Agent',
    type: 'analysis',
    capabilities: ['analyze', 'security-scan'],
    enabled: true,
    version: '1.0.0'
  }));

  // 5. Register workflows
  const workflowManager = new WorkflowManager();
  const workflows = workflowManager.getAllTemplates();
  for (const workflow of workflows) {
    orchestrator.registerWorkflow(workflow);
  }

  // 6. Execute a workflow
  const results = await orchestrator.executeWorkflow('code-review-workflow');
  console.log('Workflow results:', results);
}

main().catch(console.error);
```

---

## Troubleshooting

### MCP Servers Not Found

1. Ensure MCPs are installed: `npm install @modelcontextprotocol/server-*`
2. Check configuration file paths
3. Verify `enabled` flag is `true`

### Agent Not Executing

1. Check agent is registered: `orchestrator.getAgent(agentId)`
2. Verify task type matches agent capabilities
3. Check required MCPs are enabled

### Workflow Steps Failing

1. Ensure all agents are initialized
2. Check step dependencies are met
3. Verify step agent IDs exist

---

## Next Steps

1. **Install specific MCPs** as needed for your projects
2. **Create custom agents** for domain-specific tasks
3. **Define workflows** for your development processes
4. **Integrate with CI/CD** for automated orchestration
5. **Add custom issue templates** for your workflows

