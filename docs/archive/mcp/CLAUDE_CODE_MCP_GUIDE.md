# Claude Code MCP & Agent Integration Guide

## Quick Start

### 1. Install and Build
```bash
pnpm install
pnpm run build
```

### 2. Run Setup Script
```bash
chmod +x scripts/setup-all.sh
./scripts/setup-all.sh
```

This will:
- Install all dependencies
- Build all packages
- Configure MCP servers
- Set up agents and workflows
- Run type checks

### 3. Verify Configuration
Check that these files were created:
- `.claude/mcp-config.json` - MCP server configuration
- `.claude/agents.json` - Agent definitions
- `.claude/orchestration.json` - Orchestration rules

---

## Architecture Overview

### 5 Core Packages

#### 1. **mcp-core** - MCP Management
Handles MCP server registration, configuration, and management across environments.

```typescript
import { MCPConfigManager, MCPCategory } from '@monorepo/mcp-core';

const configManager = new MCPConfigManager();
configManager.addServer({
  name: 'github',
  description: 'GitHub API',
  command: 'npx',
  args: ['@modelcontextprotocol/server-github'],
  enabled: true,
  category: MCPCategory.DEVELOPMENT
}, 'local');
```

#### 2. **agent-core** - Agent Framework
Provides base classes and orchestration for creating intelligent agents.

```typescript
import { BaseAgent, AgentOrchestrator } from '@monorepo/agent-core';

class MyAgent extends BaseAgent {
  async execute(task) { /* ... */ }
}

const orchestrator = new AgentOrchestrator();
orchestrator.registerAgent(new MyAgent(config));
```

#### 3. **context-provider** - Shared Context
Singleton context provider for accessing workspace info and shared state.

```typescript
import { ContextProvider } from '@monorepo/context-provider';

const context = ContextProvider.getInstance();
context.getWorkspaceRoot();
context.setMetadata('key', value);
```

#### 4. **issue-library** - Issue Templates
Pre-defined issue templates for common workflows (Bug, Feature, Refactor).

```typescript
import { IssueManager } from '@monorepo/issue-library';

const issueManager = new IssueManager();
const issue = issueManager.createIssue('bug-report', {
  title: 'Bug title',
  description: 'Bug description'
});
```

#### 5. **workflow-templates** - Workflow Definitions
Ready-to-use workflow templates:
- Code Review Workflow
- Bug Fix Workflow
- Feature Development Workflow
- Security Audit Workflow

```typescript
import { WorkflowManager } from '@monorepo/workflow-templates';

const manager = new WorkflowManager();
const workflow = manager.createWorkflow('code-review-workflow');
await orchestrator.executeWorkflow(workflow.id);
```

---

## MCP Server Configuration

### Understanding the 3-Level Configuration System

```
Claude Code Global (~/AppData/Roaming/Trae/User)
         ↑
         ↑ Merges with
         ↑
DevContainer Config (.devcontainer/mcp-config.json)
         ↑
         ↑ Merges with
         ↑
Local Project Config (.claude/mcp-config.json) ← Highest priority
```

### Available MCP Servers

**Core MCPs:**
- `filesystem` - File operations
- `git` - Git repository access
- `fetch` - Web content retrieval

**Optional MCPs:**
- `github` - GitHub API
- `postgres` - PostgreSQL database
- `brave-search` - Web search
- And 50+ more (see MCP_SERVERS_GUIDE.md)

### Adding a New MCP

```json
{
  "mcpServers": {
    "my-mcp": {
      "name": "my-mcp",
      "description": "My custom MCP",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-my-mcp"],
      "enabled": true,
      "category": "specialized"
    }
  },
  "enabled": ["my-mcp"]
}
```

---

## Agent Framework

### Creating Custom Agents

```typescript
import { BaseAgent, AgentTask, AgentResult, TaskStatus } from '@monorepo/agent-core';

export class CustomAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult> {
    const startTime = Date.now();
    try {
      this.updateTaskStatus(task.id, TaskStatus.RUNNING);

      const result = await this.performWork(task);

      this.updateTaskStatus(task.id, TaskStatus.COMPLETED);
      return {
        success: true,
        data: result,
        duration: Date.now() - startTime
      };
    } catch (error) {
      this.updateTaskStatus(task.id, TaskStatus.FAILED);
      return {
        success: false,
        error: error as Error,
        duration: Date.now() - startTime
      };
    }
  }

  private async performWork(task: AgentTask) {
    // Your custom logic here
  }
}
```

### Registering with Orchestrator

```typescript
import { AgentOrchestrator } from '@monorepo/agent-core';

const orchestrator = new AgentOrchestrator();
const agent = new CustomAgent(config);

await orchestrator.initialize(context);
orchestrator.registerAgent(agent);
```

### Executing Tasks

```typescript
const task: AgentTask = {
  id: 'task-1',
  name: 'My Task',
  type: 'my-task-type',
  status: TaskStatus.PENDING,
  agentId: 'my-agent'
};

const result = await orchestrator.executeTask(task);
console.log(result.success ? 'Done!' : result.error);
```

---

## Workflow System

### Pre-built Workflows

#### Code Review Workflow
```
1. Lint Check → 2. Type Check → 3. Test → 4. Security Scan
```

#### Bug Fix Workflow
```
1. Create Issue → 2. Reproduce → 3. Fix → 4. Verify
```

#### Feature Development Workflow
```
1. Create Issue → 2. Design Review → 3. Implementation →
4. Tests → 5. Code Review → 6. Documentation
```

#### Security Audit Workflow
```
1. Dependency Scan → 2. Code Security → 3. Config Review → 4. Report
```

### Creating Custom Workflows

```typescript
import { WorkflowTemplate, WORKFLOW_CATEGORIES } from '@monorepo/workflow-templates';

const customWorkflow: WorkflowTemplate = {
  id: 'my-workflow',
  name: 'My Workflow',
  description: 'Description',
  version: '1.0.0',
  category: WORKFLOW_CATEGORIES.DEVELOPMENT,
  difficulty: 'beginner',
  tags: ['custom'],
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
      action: 'analyze'
    }
  ]
};

workflowManager.registerTemplate(customWorkflow);
```

### Executing Workflows

```typescript
// Execute a built-in workflow
const results = await orchestrator.executeWorkflow('code-review-workflow');

// Execute custom workflow
const custom = workflowManager.createWorkflow('my-workflow');
await orchestrator.executeWorkflow(custom.id);
```

---

## Integration Points

### With Claude Code

MCPs are automatically discovered by Claude Code when configured in:
- `~/.config/trae/claude-code-mcp.json` (Mac/Linux)
- `%APPDATA%\Trae\User\claude-code-mcp.json` (Windows)
- `.devcontainer/mcp-config.json` (in container)
- `.claude/mcp-config.json` (project-specific)

### With DevContainers

The Dockerfile installs core MCP servers. Add more in postCreateCommand:
```json
"postCreateCommand": "pnpm install && npm install -g @modelcontextprotocol/server-github"
```

### With GitHub Actions

Create a workflow file:
```yaml
name: Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: pnpm/action-setup@v2
      - name: Setup
        run: ./scripts/setup-all.sh
      - name: Run Workflow
        run: npx ts-node -e "orchestrator.executeWorkflow('code-review-workflow')"
```

---

## Configuration Files Explained

### `.claude/mcp-config.json`
Defines which MCP servers are available and enabled.

```json
{
  "mcpServers": {
    "filesystem": { /* ... */ },
    "git": { /* ... */ }
  },
  "enabled": ["filesystem", "git"],
  "disabled": []
}
```

### `.claude/agents.json`
Defines available agents and their capabilities.

```json
{
  "agents": [
    {
      "id": "code-agent",
      "name": "Code Agent",
      "type": "code",
      "capabilities": ["review", "fix"],
      "enabled": true
    }
  ]
}
```

### `.claude/orchestration.json`
Defines rules that trigger workflows automatically.

```json
{
  "rules": [
    {
      "id": "pr-review",
      "name": "Review on PR",
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

## Usage Examples

### Example 1: Running a Code Review

```typescript
import { AgentOrchestrator } from '@monorepo/agent-core';

const orchestrator = new AgentOrchestrator();
await orchestrator.initialize(context);

// Execute code review workflow
const results = await orchestrator.executeWorkflow('code-review-workflow');
console.log('Code review completed:', results);
```

### Example 2: Creating an Issue

```typescript
import { IssueManager, IssueType, IssuePriority } from '@monorepo/issue-library';

const issueManager = new IssueManager();
const issue = issueManager.createIssue('bug-report', {
  title: 'Login fails on iOS',
  description: 'Users on iOS devices cannot log in',
  steps: '1. Open app on iOS\n2. Try to log in\n3. See error',
  expected: 'Should log in successfully',
  actual: 'Shows authentication error',
  environment: 'iOS 16.7, latest app version'
});

console.log('Issue created:', issue.id);
```

### Example 3: Custom Workflow

```typescript
import { WorkflowManager } from '@monorepo/workflow-templates';

const manager = new WorkflowManager();
const workflow = manager.createWorkflow('feature-development-workflow', {
  name: 'Add Dark Mode Feature'
});

const results = await orchestrator.executeWorkflow(workflow.id);
for (const [stepId, result] of results) {
  console.log(`${stepId}: ${result}`);
}
```

### Example 4: Multi-Agent Coordination

```typescript
import { AgentOrchestrator, CodeAgent, AnalysisAgent } from '@monorepo/agent-core';

const orchestrator = new AgentOrchestrator();

// Register multiple agents
orchestrator.registerAgent(new CodeAgent(codeAgentConfig));
orchestrator.registerAgent(new AnalysisAgent(analysisAgentConfig));

// Execute tasks that require coordination
const codeReviewTask = /* ... */;
const securityTask = /* ... */;

const [codeResult, secResult] = await Promise.all([
  orchestrator.executeTask(codeReviewTask),
  orchestrator.executeTask(securityTask)
]);

console.log('Code Review:', codeResult.success);
console.log('Security:', secResult.success);
```

---

## Troubleshooting

### MCPs Not Available in Claude Code

1. Check configuration files exist and have correct syntax
2. Verify MCP servers are installed: `npm list -g @modelcontextprotocol/server-*`
3. Ensure `enabled` list includes your MCPs
4. Restart Claude Code and devcontainer

### Agent Execution Failures

1. Verify agent is registered: `orchestrator.getAgent(agentId)`
2. Check task type matches agent capabilities
3. Ensure required MCPs are enabled
4. Check logs for specific errors

### Workflow Not Progressing

1. Verify all steps have valid agentIds
2. Check step dependencies are met
3. Review workflow definition for syntax errors
4. Enable verbose logging for debugging

---

## Next Steps

1. **Customize MCPs** - Add specific servers your project needs
2. **Create Domain Agents** - Build agents for your specific workflows
3. **Define Workflows** - Create workflows matching your development process
4. **Set Up Orchestration** - Configure rules to automate tasks
5. **Integrate with CI/CD** - Use with GitHub Actions or GitLab CI

---

## References

- **MCP Servers Guide**: See `MCP_SERVERS_GUIDE.md` for 50+ available MCPs
- **Architecture Details**: See `MCP_AND_AGENTS_SETUP.md` for complete setup guide
- **Package Documentation**: Each package has detailed API docs in source

---

## Support

For issues or questions:
1. Check package-specific documentation
2. Review example implementations
3. Open an issue in the repository
4. Check troubleshooting section above
