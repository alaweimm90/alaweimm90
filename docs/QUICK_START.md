# Quick Start
Get up and running with MCPs and Agents in 5 minutes.
## Prerequisites
- Node.js 20+
- pnpm or npm
- Git
- Claude Code or VS Code with Claude Code extension
## Installation
### 1. Install MCP Infrastructure
```bash
# Navigate to project root
cd /path/to/monorepo
# Run MCP setup
node scripts/mcp-setup.js --install
# Run agent setup
node scripts/agent-setup.js
# Install dependencies
pnpm install
# Build packages
pnpm build
```
### 2. Verify Installation
```bash
# Check MCP configuration
cat .claude/mcp-config.json
# Check agents
cat .claude/agents.json
# Check workflows
ls -la .claude/workflows/
```
## Basic Usage
### Using Claude Code with MCPs
Claude Code automatically uses available MCPs. You can:
```
@Claude: Review this file for security issues
@Claude: Create a new feature following the workflow
@Claude: Run tests and generate a report
```
### Running a Workflow
```bash
# Execute a workflow programmatically
pnpm exec ts-node -e "
import { AgentOrchestrator } from '@monorepo/agent-core';
const orchestrator = new AgentOrchestrator();
await orchestrator.executeWorkflow('code-review-workflow');
"
```
### Using an Agent
```typescript
import { CodeAgent } from '@monorepo/agent-core';
import { ContextProvider } from '@monorepo/context-provider';
const agent = new CodeAgent({
  id: 'my-agent',
  name: 'My Agent',
  type: 'code',
  capabilities: ['code-review'],
});
const context = ContextProvider.getInstance().getContext();
await agent.initialize(context);
const result = await agent.execute({
  id: 'task-1',
  name: 'Review Code',
  type: 'code-review',
  input: { filePath: 'src/index.ts' },
});
```
## Common Tasks
### Enable an Additional MCP
1. Edit `.claude/mcp-config.json`
2. Change `"enabled": false` to `"enabled": true` for desired server
3. Add server name to `enabled` array
4. Restart Claude Code
```json
{
  "mcpServers": {
    "github": {
      "enabled": true
    }
  },
  "enabled": ["filesystem", "git", "fetch", "github"]
}
```
### Create a Custom Workflow
1. Create `.claude/workflows/my-workflow.json`:
```json
{
  "id": "my-workflow",
  "name": "My Custom Workflow",
  "enabled": true,
  "steps": [
    {
      "id": "step-1",
      "name": "Check Code",
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
2. Use in Claude Code or code:
```typescript
const orchestrator = new AgentOrchestrator();
await orchestrator.executeWorkflow('my-workflow');
```
### Add Custom Agent
1. Create agent class:
```typescript
// packages/my-agents/src/custom-agent.ts
import { BaseAgent, AgentTask, AgentResult } from '@monorepo/agent-core';
export class CustomAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult> {
    try {
      // Your implementation here
      return {
        success: true,
        data: { result: 'success' },
        duration: Date.now(),
      };
    } catch (error) {
      return {
        success: false,
        error: error as Error,
        duration: Date.now(),
      };
    }
  }
}
```
2. Register agent:
```typescript
import { CustomAgent } from '@monorepo/my-agents';
const agent = new CustomAgent({
  id: 'custom-agent',
  name: 'Custom Agent',
  type: 'custom',
  capabilities: ['custom-task'],
});
orchestrator.registerAgent(agent);
```
## Development in DevContainer
The devcontainer comes pre-configured with MCPs. To use it:
### VS Code
1. Install Remote - Containers extension
2. Open project in container: `Cmd+Shift+P` → "Dev Containers: Reopen in Container"
3. Run setup scripts (they're automatically called on creation)
### Command Line
```bash
# Open devcontainer shell
devcontainer open .
# Run inside container
node scripts/mcp-setup.js
pnpm install
pnpm build
```
## Troubleshooting
### MCPs Not Available
```bash
# Reinstall MCPs
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-git
# Check .claude/mcp-config.json has correct enabled array
cat .claude/mcp-config.json
```
### Agent Not Found
```bash
# Check agents.json
cat .claude/agents.json
# Verify agent ID matches your code
# Check logs for error details
```
### Workflow Execution Fails
```bash
# Check workflow JSON syntax
cat .claude/workflows/my-workflow.json | jq .
# Verify all agent IDs exist
# Check MCP requirements are met
```
## What's Next?
1. **Explore MCPs**: [MCP_SERVERS_GUIDE.md](./MCP_SERVERS_GUIDE.md)
2. **Learn Agents**: [MCP_AGENTS_ORCHESTRATION.md](./MCP_AGENTS_ORCHESTRATION.md)
3. **Build Workflows**: [WORKFLOW_TUTORIAL.md](./WORKFLOW_TUTORIAL.md)
4. **Package Documentation**: Check `packages/*/README.md`
## Resources
- Official MCP: https://www.anthropic.com/news/model-context-protocol
- GitHub MCPs: https://github.com/modelcontextprotocol/servers
- Community MCPs: https://mcp.so
## Support
For issues or questions:
1. Check documentation in `docs/`
2. Review example configurations in `.claude/`
3. Check package READMEs in `packages/*/`
4. Open an issue on GitHub
