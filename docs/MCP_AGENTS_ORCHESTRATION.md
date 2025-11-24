# Mcp Agents Orchestration
This document describes the complete architecture for Model Context Protocol (MCP) servers, Agents, and Orchestration in the monorepo.
## Overview
The system is built on three core abstractions:
1. **MCP (Model Context Protocol)** - Secure, sandboxed access to tools and data
2. **Agents** - Autonomous entities that use MCPs to perform tasks
3. **Orchestration** - Coordination of agents and workflows
## Architecture
### 1. MCP (Model Context Protocol)
MCPs provide Claude Code and agents with secure, controlled access to:
- File systems
- Git repositories
- External APIs
- Databases
- Web services
#### Core MCP Servers
| Server | Purpose | Status |
|--------|---------|--------|
| **filesystem** | File operations with path restrictions | Essential |
| **git** | Git repository operations | Essential |
| **fetch** | Web content retrieval | Recommended |
| **github** | GitHub API integration | Optional |
| **postgres** | PostgreSQL database access | Optional |
| **brave-search** | Web search capabilities | Optional |
#### Configuration
MCP servers are configured in `.claude/mcp-config.json`:
```json
{
  "mcpServers": {
    "filesystem": {
      "name": "filesystem",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem"],
      "enabled": true,
      "category": "core"
    }
  },
  "enabled": ["filesystem", "git", "fetch"],
  "disabled": []
}
```
### 2. Agent Framework
The agent framework provides abstractions for autonomous agents that:
- Execute tasks
- Use MCPs as tools
- Maintain state
- Report results
#### Core Packages
- **@monorepo/agent-core** - Base agent classes and orchestrator
- **@monorepo/context-provider** - Shared context management
- **@monorepo/issue-library** - Issue templates
- **@monorepo/workflow-templates** - Workflow definitions
#### Agent Types
```typescript
enum AgentType {
  BASE = 'base',           // Generic base agent
  CODE = 'code',           // Code manipulation and analysis
  ANALYSIS = 'analysis',   // Testing and analysis
  ORCHESTRATOR = 'orchestrator',  // Coordinates other agents
  CUSTOM = 'custom'        // Custom implementation
}
```
#### Built-in Agents
##### Code Agent
- **Purpose**: Code manipulation, review, and generation
- **Capabilities**: code-review, code-fix, refactor, test
- **Required MCPs**: filesystem, git
##### Analysis Agent
- **Purpose**: Code analysis, security scanning, testing
- **Capabilities**: analyze, security-scan, test, lint, type-check
- **Required MCPs**: filesystem, git
### 3. Orchestration
Orchestration coordinates agents and workflows through:
- **Workflows** - Sequences of steps executed by agents
- **Rules** - Trigger-action pairs for automation
- **Context** - Shared state across agents
#### Workflows
Workflows define a sequence of steps to be executed:
```json
{
  "id": "code-review-workflow",
  "name": "Code Review",
  "steps": [
    {
      "id": "step-1",
      "name": "Lint Check",
      "type": "task",
      "action": "lint",
      "agentId": "code-agent"
    }
  ]
}
```
#### Orchestration Rules
Rules trigger workflows based on events:
```json
{
  "id": "code-review-rule",
  "name": "Code Review on PR",
  "trigger": "pull_request_opened",
  "actions": [
    {
      "type": "execute",
      "target": "code-review-workflow"
    }
  ]
}
```
## File Structure
```
monorepo/
├── packages/
│   ├── mcp-core/              # MCP abstractions
│   ├── agent-core/            # Agent framework
│   ├── context-provider/      # Context management
│   ├── issue-library/         # Issue templates
│   └── workflow-templates/    # Workflow definitions
│
├── .claude/                   # Claude Code configuration
│   ├── mcp-config.json        # MCP configuration
│   ├── agents.json            # Agent definitions
│   ├── orchestration.json     # Orchestration rules
│   ├── agents/                # Individual agent configs
│   └── workflows/             # Individual workflow configs
│
├── .devcontainer/
│   ├── Dockerfile             # Includes MCP servers
│   └── devcontainer.json      # Dev environment config
│
└── scripts/
    ├── mcp-setup.js           # Initialize MCP
    └── agent-setup.js         # Initialize agents
```
## Setup & Installation
### Quick Start
1. **Initialize MCP Configuration**
   ```bash
   node scripts/mcp-setup.js --install
   ```
2. **Set up Agents and Workflows**
   ```bash
   node scripts/agent-setup.js
   ```
3. **Install Dependencies**
   ```bash
   pnpm install
   pnpm build
   ```
4. **Verify Setup**
   ```bash
   cat .claude/mcp-config.json
   cat .claude/agents.json
   ```
### In DevContainer
The devcontainer automatically:
1. Installs core MCP servers (filesystem, git, fetch)
2. Sets up environment variables
3. Initializes workspace
### Local Development
For local development (not in devcontainer):
1. Install Node.js 20+
2. Run MCP setup: `node scripts/mcp-setup.js`
3. Install MCPs globally: `npm install -g @modelcontextprotocol/server-*`
## Core Packages
### @monorepo/mcp-core
Manages MCP servers and configuration.
```typescript
import { MCPRegistryManager, MCPConfigManager } from '@monorepo/mcp-core';
// Register and manage MCP servers
const registry = new MCPRegistryManager();
const enabled = registry.getEnabledServers();
// Configure MCPs
const config = new MCPConfigManager();
config.addServer(serverConfig, 'local');
```
### @monorepo/agent-core
Defines agent framework and orchestration.
```typescript
import { BaseAgent, AgentOrchestrator, CodeAgent } from '@monorepo/agent-core';
// Create and use agents
const agent = new CodeAgent(config);
await agent.initialize(context);
// Orchestrate workflows
const orchestrator = new AgentOrchestrator();
orchestrator.registerAgent(agent);
await orchestrator.executeWorkflow(workflowId);
```
### @monorepo/context-provider
Provides shared context across agents and MCPs.
```typescript
import { ContextProvider } from '@monorepo/context-provider';
const context = ContextProvider.getInstance();
context.setMetadata('key', value);
const subContext = context.createSubContext(taskId, taskData);
```
### @monorepo/issue-library
Manages issue templates and creation.
```typescript
import { IssueManager, IssueType } from '@monorepo/issue-library';
const issueManager = new IssueManager();
const issue = issueManager.createIssue('bug-report', {
  title: 'Bug title',
  description: 'Bug description',
  steps: 'Steps to reproduce',
});
```
### @monorepo/workflow-templates
Defines and manages workflow templates.
```typescript
import { WorkflowManager } from '@monorepo/workflow-templates';
const workflowManager = new WorkflowManager();
const workflows = workflowManager.getAllTemplates();
const workflow = workflowManager.createWorkflow('code-review', overrides);
```
## Configuration Files
### .claude/mcp-config.json
Configuration for MCP servers. Specifies which servers are enabled and their command-line arguments.
### .claude/agents.json
Defines available agents and their capabilities. References which MCPs are required.
### .claude/orchestration.json
Defines orchestration rules that trigger workflows based on events.
### .claude/agents/*.json
Individual agent configuration files with specific settings.
### .claude/workflows/*.json
Individual workflow definitions with step sequences.
## Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `MCP_SERVERS` | Enabled MCP servers | `filesystem,git,fetch` |
| `WORKSPACE_ROOT` | Monorepo root directory | `/workspace` or `/home/user/project` |
| `NODE_ENV` | Environment | `development` or `production` |
| `CLAUDE_MODEL` | Claude model to use | `claude-opus` |
## Common Tasks
### Add a New MCP Server
1. Update `.claude/mcp-config.json`:
   ```json
   {
     "my-server": {
       "name": "my-server",
       "command": "npx",
       "args": ["@modelcontextprotocol/server-my-server"],
       "enabled": true,
       "category": "custom"
     }
   }
   ```
2. Install the MCP: `npm install -g @modelcontextprotocol/server-my-server`
3. Enable in agents: Update agent `requiredMcps`
### Create a Custom Agent
1. Create a new class extending `BaseAgent`:
   ```typescript
   export class MyAgent extends BaseAgent {
     async execute(task: AgentTask): Promise<AgentResult> {
       // Implementation
     }
   }
   ```
2. Register with orchestrator:
   ```typescript
   const agent = new MyAgent(config);
   orchestrator.registerAgent(agent);
   ```
3. Define in `.claude/agents/my-agent.json`
### Define a New Workflow
1. Create workflow in `.claude/workflows/my-workflow.json`:
   ```json
   {
     "id": "my-workflow",
     "name": "My Workflow",
     "steps": [
       {
         "id": "step-1",
         "name": "Step 1",
         "type": "task",
         "agentId": "code-agent",
         "action": "my-action"
       }
     ]
   }
   ```
2. Register with orchestrator or trigger via rules
### Trigger a Workflow
Via orchestration rule:
```json
{
  "id": "my-rule",
  "trigger": "pull_request_opened",
  "actions": [
    {
      "type": "execute",
      "target": "my-workflow"
    }
  ]
}
```
Via Claude Code:
```typescript
const orchestrator = new AgentOrchestrator();
await orchestrator.executeWorkflow('my-workflow');
```
## Best Practices
### MCP Usage
- ✅ Use MCPs for all external operations (files, git, APIs)
- ✅ Keep MCP calls isolated in agents
- ❌ Don't bypass MCPs for security-critical operations
- ❌ Don't hardcode MCP server paths
### Agent Development
- ✅ Extend BaseAgent or specific agent types
- ✅ Validate task inputs
- ✅ Report clear results
- ❌ Don't share state between agents
- ❌ Don't assume MCPs are available
### Workflow Design
- ✅ Keep steps focused and single-purpose
- ✅ Define dependencies explicitly
- ✅ Use conditional steps for branching
- ❌ Don't create circular dependencies
- ❌ Don't hard-code agent IDs
### Orchestration Rules
- ✅ Keep triggers specific
- ✅ Handle errors gracefully
- ✅ Log rule execution
- ❌ Don't create infinite loops
- ❌ Don't ignore rule failures
## Troubleshooting
### MCP Server Not Found
```bash
# Check if MCP is installed
npm list -g @modelcontextprotocol/server-xxx
# Reinstall if needed
npm install -g @modelcontextprotocol/server-xxx
# Check configuration
cat .claude/mcp-config.json
```
### Agent Execution Fails
```bash
# Check agent configuration
cat .claude/agents.json
# Verify required MCPs are enabled
cat .claude/mcp-config.json
# Check logs for detailed error messages
```
### Workflow Hangs
```bash
# Check workflow definition
cat .claude/workflows/my-workflow.json
# Verify agent IDs are correct
# Check for circular dependencies
# Ensure all agents are registered
```
## Resources
- [MCP Servers Guide](./MCP_SERVERS_GUIDE.md)
- [Agent Development Guide](./AGENT_DEVELOPMENT.md)
- [Workflow Tutorial](./WORKFLOW_TUTORIAL.md)
- [Architecture Decisions](./ARCHITECTURE.md)
## Contributing
When adding new MCPs, agents, or workflows:
1. Follow the established patterns
2. Add comprehensive documentation
3. Update configuration files
4. Test with devcontainer
5. Update this guide
## License
See LICENSE file in repository root.
