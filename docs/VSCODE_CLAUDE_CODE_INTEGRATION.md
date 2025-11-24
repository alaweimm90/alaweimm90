# Vscode Claude Code Integration
Complete guide to integrate MCPs and Agents with VS Code and Claude Code.
## Prerequisites
- VS Code 1.84+
- Claude Code extension installed
- Node.js 20+
- All MCPs configured in `.claude/mcp-config.json`
## Step 1: Configure Claude Code for MCPs
### Option A: Automatic Configuration (Recommended)
Claude Code automatically detects MCPs from `.claude/mcp-config.json`.
1. Open VS Code
2. Install Claude Code extension (if not already installed)
3. Configure in Claude Code settings:
```json
{
  "claude.mcp.configPath": ".claude/mcp-config.json",
  "claude.mcp.autoLoad": true,
  "claude.mcp.enableLogging": true
}
```
### Option B: Manual Configuration
Edit `.vscode/settings.json`:
```json
{
  "claude.mcp": {
    "enabled": true,
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem"]
      },
      "git": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-git"]
      },
      "github": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-github"],
        "env": {
          "GITHUB_TOKEN": "${env:GITHUB_TOKEN}"
        }
      }
    },
    "defaultMcps": ["filesystem", "git"]
  }
}
```
## Step 2: Configure Agents
Create `.vscode/claude-agents.json`:
```json
{
  "agents": [
    {
      "id": "code-reviewer",
      "name": "Code Reviewer",
      "description": "Reviews code for quality and best practices",
      "command": "claude",
      "args": ["review"],
      "enabled": true
    },
    {
      "id": "bug-fixer",
      "name": "Bug Fixer",
      "description": "Identifies and fixes bugs",
      "command": "claude",
      "args": ["fix"],
      "enabled": true
    },
    {
      "id": "documenter",
      "name": "Documenter",
      "description": "Generates documentation",
      "command": "claude",
      "args": ["document"],
      "enabled": true
    }
  ]
}
```
## Step 3: Create Custom Commands
Create `.vscode/tasks.json` with Claude Code tasks:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Claude: Review Code",
      "type": "shell",
      "command": "echo",
      "args": ["@Claude: Run code-review-workflow"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always"
      }
    },
    {
      "label": "Claude: Run Security Audit",
      "type": "shell",
      "command": "echo",
      "args": ["@Claude: Run security-audit-workflow"],
      "group": "test"
    },
    {
      "label": "Claude: Analyze Performance",
      "type": "shell",
      "command": "echo",
      "args": ["@Claude: Run performance-analysis-workflow"],
      "group": "test"
    },
    {
      "label": "Claude: Generate Docs",
      "type": "shell",
      "command": "echo",
      "args": ["@Claude: Run documentation-generation-workflow"],
      "group": "build"
    }
  ]
}
```
## Step 4: Configure Keybindings
Create `.vscode/keybindings.json`:
```json
[
  {
    "key": "ctrl+shift+r",
    "command": "workbench.action.tasks.runTask",
    "args": "Claude: Review Code",
    "when": "editorTextFocus"
  },
  {
    "key": "ctrl+shift+s",
    "command": "workbench.action.tasks.runTask",
    "args": "Claude: Run Security Audit",
    "when": "explorerViewletFocus"
  },
  {
    "key": "ctrl+shift+d",
    "command": "workbench.action.tasks.runTask",
    "args": "Claude: Generate Docs",
    "when": "editorTextFocus"
  }
]
```
## Step 5: Extension Settings
Update `.vscode/settings.json`:
```json
{
  "claude.enabled": true,
  "claude.apiKey": "${env:CLAUDE_API_KEY}",
  "claude.model": "claude-opus",
  "claude.temperature": 0.3,
  "claude.maxTokens": 4096,
  "claude.mcp": {
    "enabled": true,
    "autoLoad": true,
    "configPath": ".claude/mcp-config.json"
  },
  "claude.agents": {
    "enabled": true,
    "configPath": ".claude/agents.json"
  },
  "claude.workflows": {
    "enabled": true,
    "configPath": ".claude/orchestration.json"
  },
  "claude.logging": {
    "level": "info",
    "file": "${workspaceFolder}/.claude/logs/claude.log"
  }
}
```
## Step 6: Using Claude Code
### Direct Commands
In Claude Code input:
```
@Claude: Review this file for best practices
@Claude: Run code-review-workflow
@Claude: Execute security-audit-workflow
@Claude: Analyze performance regressions
@Claude: Generate API documentation
```
### With Agents
```
@CodeReviewAgent: Check for security issues in auth.ts
@BugFixerAgent: Find and fix the bug in database.ts
@DocumenterAgent: Create API docs for handlers/
```
### With MCPs
Claude Code automatically uses configured MCPs:
```
@Claude: Check git history for this file
@Claude: Search GitHub for similar implementations
@Claude: Fetch documentation from the web
```
## Step 7: Troubleshooting
### MCPs Not Loading
```bash
# Check MCP configuration
cat .claude/mcp-config.json
# Verify paths
ls -la .claude/
# Check Claude Code logs
tail -f .claude/logs/claude.log
```
### Agents Not Available
```bash
# Verify agents.json
cat .claude/agents.json | jq .
# Check syntax
node -e "console.log(require('./.claude/agents.json'))"
```
### Workflows Not Running
```bash
# Check orchestration.json
cat .claude/orchestration.json | jq .
# Verify workflow files
ls -la .claude/workflows/
# Test validation
node scripts/validate-setup.js
```
## Step 8: Advanced Integration
### Pre-commit Hooks
Create `.husky/pre-commit`:
```bash
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"
# Run Claude Code review
echo "Running Claude Code review..."
# (Integration depends on Claude Code CLI)
```
### CI/CD Integration
See [GitHub Actions Integration](./GITHUB_ACTIONS_INTEGRATION.md)
## Step 9: Environment Variables
Create `.env` for Claude Code:
```bash
# Claude API
CLAUDE_API_KEY=your_key_here
CLAUDE_MODEL=claude-opus
# GitHub
GITHUB_TOKEN=your_token_here
# MCPs
BRAVE_API_KEY=your_brave_key
SENTRY_AUTH_TOKEN=your_sentry_token
SLACK_BOT_TOKEN=your_slack_token
# Database (if using)
DATABASE_URL=postgresql://...
MONGODB_URI=mongodb://...
```
## Step 10: Team Configuration
Share with team via `.vscode/extensions.json`:
```json
{
  "recommendations": [
    "anthropic.claude-code",
    "ms-python.python",
    "ms-typescript.vscode-typescript-next",
    "eamodio.gitlens"
  ]
}
```
## Usage Examples
### Code Review
```
User: @Claude: Review this file
Claude: I'll run the code-review-workflow...
  → Lints code
  → Type checks
  → Tests
  → Security scan
  → Provides feedback
```
### Bug Fixing
```
User: @Claude: Fix the authentication bug
Claude: I'll use the bug-fix-workflow...
  → Analyzes issue
  → Implements fix
  → Runs tests
  → Provides recommendations
```
### Documentation
```
User: @Claude: Generate API docs
Claude: I'll run documentation-generation-workflow...
  → Analyzes exports
  → Extracts comments
  → Finds gaps
  → Generates docs
  → Updates files
```
## Performance Tips
1. **Lazy Load MCPs** - Only enable needed MCPs
2. **Cache Results** - Configure caching for faster reuse
3. **Parallel Execution** - Use parallel workflow steps
4. **Limit Scope** - Target specific files/directories
5. **Monitor Logs** - Track performance in `.claude/logs/`
## Security
1. **Store Secrets in `.env`** - Never commit API keys
2. **Use Claude Code Secrets** - Built-in secret management
3. **Restrict MCP Access** - Limit filesystem paths
4. **Audit Logs** - Review orchestration logs
5. **Code Review** - Have humans review Claude suggestions
## Next Steps
1. Configure `.claude/mcp-config.json` for your needs
2. Set up environment variables in `.env`
3. Test commands in Claude Code
4. Create team guidelines
5. Integrate with CI/CD
## Support
- Check logs: `.claude/logs/claude.log`
- Validate setup: `node scripts/validate-setup.js`
- Review docs: [MCP_AGENTS_ORCHESTRATION.md](./MCP_AGENTS_ORCHESTRATION.md)
- GitHub Issues: [anthropics/claude-code](https://github.com/anthropics/claude-code)
