# 🎯 LOCAL AI/ML/LLM ORCHESTRATION SUPERPROMPT

**For**: Claude Instance 3 (Local Development Environment Setup)
**Context**: While ATLAS Engine and Libria Suite are being built in parallel
**Goal**: Unified AI/ML/LLM instruction framework for local WSL/PowerShell development
**Duration**: 2-4 hours for initial setup, ongoing maintenance

---

## MISSION BRIEF

Create a **comprehensive, unified instruction framework** for managing AI/ML/LLM workflows across your local development environment (WSL/PowerShell), including:

1. **IDEs with integrated LLM support** (VSCode, JetBrains, Vim, etc.)
2. **Internal AI agents** for local development tasks
3. **MCP (Model Context Protocol) servers** for CLI augmentation
4. **Workflow orchestration** for multi-tool coordination
5. **Consolidated configuration** for consistency across tools

---

## CURRENT STATE ANALYSIS

### Existing Locations (To Explore)
- `C:\Users\mesha\` - Main Windows user directory
- `\\wsl.localhost\Ubuntu\home\mesha\` - WSL filesystem
- `/mnt/c/Users/mesha/` - WSL mount of Windows home
- **TBD**: Exact locations of existing AI/LLM configs

### What We Know Works
- Claude Code CLI functioning
- Multi-Claude coordination (ATLAS + Libria)
- Git workflow management
- Docker/Docker-compose support

---

## DELIVERABLES EXPECTED

### 1. **Unified Configuration Framework** (15-20 minutes)

Create a master configuration structure:

```
~/.ai-orchestration/
├── config/
│   ├── global.yaml              # Global settings
│   ├── ides.yaml                # IDE configurations
│   ├── agents.yaml              # Internal agent definitions
│   ├── mcps.yaml                # MCP server configurations
│   └── workflows.yaml           # Workflow definitions
├── agents/
│   ├── code-reviewer/
│   ├── documentation/
│   ├── test-generator/
│   ├── dependency-checker/
│   ├── performance-analyzer/
│   └── security-scanner/
├── mcps/
│   ├── filesystem-agent/
│   ├── git-agent/
│   ├── npm-agent/
│   ├── docker-agent/
│   └── system-agent/
├── workflows/
│   ├── pre-commit.yaml
│   ├── code-review.yaml
│   ├── documentation.yaml
│   ├── testing.yaml
│   └── deployment.yaml
├── templates/
│   ├── agent-template.yaml
│   ├── mcp-template.yaml
│   ├── workflow-template.yaml
│   └── script-template.sh
└── scripts/
    ├── init.sh                  # Initialize setup
    ├── install-mcps.sh          # Install MCP servers
    ├── validate-config.sh       # Validate configuration
    └── orchestrate.sh           # Run workflows
```

### 2. **Internal AI Agents** (20-30 minutes)

Define **5-8 specialized internal agents** for local development:

**Agent Types** (with examples):
- **CodeReviewerAgent** - Reviews code changes before commit
- **DocumentationAgent** - Generates/updates documentation
- **TestGeneratorAgent** - Creates test cases automatically
- **DependencyCheckerAgent** - Analyzes dependencies for conflicts
- **PerformanceAnalyzerAgent** - Profile code performance
- **SecurityScannerAgent** - Security vulnerability analysis
- **LogAnalyzerAgent** - Parse and analyze logs
- **ConsolidationAgent** - Consolidates findings from other agents

**Agent Schema** (similar to ATLAS):
```yaml
agent:
  id: code-reviewer-001
  type: code-reviewer
  capabilities:
    - static-analysis
    - style-checking
    - complexity-analysis
    - best-practices
  tools:
    - ruff
    - pylint
    - black
    - mypy
  triggers:
    - pre-commit
    - on-pr
    - manual
  output:
    format: markdown
    destination: console | file
```

### 3. **MCP (Model Context Protocol) Server Setup** (20-30 minutes)

Configure MCPs for CLI augmentation:

**Standard MCPs to Integrate**:
- **filesystem** - File operations
- **git** - Git workflow
- **npm/pip** - Package management
- **docker** - Container operations
- **system** - Shell commands
- **web** - HTTP requests
- **database** - SQL operations
- **claude-code** - Claude Code CLI integration

**MCP Configuration Example**:
```yaml
mcp_servers:
  filesystem:
    command: mcp-filesystem
    config:
      sandbox: true
      allowed_paths:
        - ~/projects
        - /tmp

  git:
    command: mcp-git
    config:
      auto_commit_format: "Agent: {agent_name} - {message}"

  claude-code:
    command: claude code
    config:
      auto_run: false
      parallel: true
```

### 4. **Workflow Orchestration** (15-20 minutes)

Create workflows that coordinate multiple agents:

**Example Workflows**:

a) **Pre-Commit Workflow**:
```yaml
name: pre-commit
trigger: pre-commit
agents:
  - code-reviewer (parallel)
  - test-generator (parallel)
  - documentation-updater
  - security-scanner
consolidation: consolidation-agent
success_criteria:
  - all_tests_pass
  - no_security_issues
  - code_review_approved
```

b) **Code Review Workflow**:
```yaml
name: code-review
trigger: manual | pr-created
agents:
  - code-reviewer
  - performance-analyzer
  - security-scanner
  - test-generator
consolidation: consolidation-agent
output: pr-comment | stdout
```

c) **Documentation Workflow**:
```yaml
name: update-docs
trigger: manual | push-to-main
agents:
  - documentation-agent
  - code-reviewer (validate-docs)
  - security-scanner (check-sensitive-info)
output: git-commit
```

### 5. **IDE Integration** (20-30 minutes)

**VSCode Configuration** (`.vscode/settings.json`):
```json
{
  "ai.agents": {
    "enabled": true,
    "autoActivate": ["code-reviewer", "test-generator"],
    "config": "~/.ai-orchestration/config/ides.yaml"
  },
  "mcp.servers": {
    "filesystem": {"command": "mcp-filesystem"},
    "git": {"command": "mcp-git"}
  }
}
```

**JetBrains Configuration** (IDE settings):
```yaml
ides:
  jetbrains:
    plugins:
      - claude-assistant
      - mcp-integration
    agents: [code-reviewer, test-generator]
    workflows: [pre-commit]
```

**Vim/Neovim Configuration** (init.vim / init.lua):
```vim
" AI orchestration
let g:ai_agents_enabled = 1
let g:ai_config_path = '~/.ai-orchestration/config/ides.yaml'
let g:mcp_servers = ['git', 'filesystem', 'system']
```

### 6. **Consolidated Instructions** (30-45 minutes)

Create **comprehensive instruction sets** for each combination:

**Example Structure**:
```
instructions/
├── GLOBAL_SETUP.md          # Universal setup
├── VSCODE_INSTRUCTIONS.md   # VSCode-specific
├── JETBRAINS_INSTRUCTIONS.md
├── VIM_INSTRUCTIONS.md
├── WSL_SETUP.md             # WSL-specific
├── POWERSHELL_SETUP.md      # PowerShell-specific
├── AGENTS_GUIDE.md          # How agents work
├── MCP_INTEGRATION.md       # MCP setup
└── WORKFLOW_RECIPES.md      # Common workflows
```

### 7. **Orchestration Framework** (30-45 minutes)

Create scripts/tools to coordinate everything:

**Main Orchestrator** (`orchestrate.sh`):
```bash
#!/bin/bash
# Main orchestration script

COMMAND=$1
CONFIG_PATH=~/.ai-orchestration/config/

case $COMMAND in
  init)           initialize_setup ;;
  agent)          run_agent $2 ;;
  workflow)       run_workflow $2 ;;
  validate)       validate_configuration ;;
  list-agents)    list_all_agents ;;
  list-mcps)      list_all_mcps ;;
  list-workflows) list_all_workflows ;;
  *)              show_help ;;
esac
```

**Usage Examples**:
```bash
orchestrate init                          # Initial setup
orchestrate agent code-reviewer           # Run specific agent
orchestrate workflow pre-commit           # Run workflow
orchestrate validate                      # Validate config
orchestrate list-agents                   # Show all agents
```

---

## SPECIFIC INSTRUCTIONS

### Step 1: Discover Existing Configurations
- [ ] Search `C:\Users\mesha\` for existing `.vscode`, `.idea`, `.vim`, `.config`
- [ ] Check WSL home: `ls -la ~` for hidden config directories
- [ ] Catalog all IDE installations and their configurations
- [ ] Document any existing LLM/AI tool integrations

### Step 2: Create Unified Configuration Framework
- [ ] Create `~/.ai-orchestration/` directory structure
- [ ] Write global configuration (`config/global.yaml`)
- [ ] Create IDE configuration templates
- [ ] Document configuration schema

### Step 3: Define Internal Agents
- [ ] Create agent definition schema (similar to ATLAS ResearchAgent)
- [ ] Implement 5-8 core agents with clear purposes
- [ ] Document each agent's capabilities and triggers
- [ ] Create agent templates for easy addition

### Step 4: Configure MCP Servers
- [ ] Research available MCPs for your use cases
- [ ] Create MCP configuration templates
- [ ] Implement integration with each IDE
- [ ] Document MCP capabilities and usage

### Step 5: Design Workflows
- [ ] Create workflow schema
- [ ] Define 3-5 core workflows (pre-commit, code-review, etc.)
- [ ] Implement agent coordination in workflows
- [ ] Create workflow templates for reuse

### Step 6: IDE Integration
- [ ] Integrate with VSCode
- [ ] Integrate with JetBrains IDEs
- [ ] Integrate with Vim/Neovim
- [ ] Create IDE-specific instructions

### Step 7: Orchestration Implementation
- [ ] Create main orchestration script
- [ ] Implement agent runner
- [ ] Implement workflow runner
- [ ] Create validation tools

### Step 8: Documentation & Instructions
- [ ] Write comprehensive setup instructions
- [ ] Create IDE-specific guides
- [ ] Document all agents and MCPs
- [ ] Create workflow recipe book
- [ ] Write troubleshooting guide

---

## REFERENCE SPECIFICATIONS

### Agent Schema (Match ATLAS Style)
```yaml
ResearchAgent:
  agent_id: string
  agent_type: string
  capabilities: List[string]
  tools: List[string]
  triggers: List[string]
  model: string
  config: Dict
```

### MCP Server Schema
```yaml
MCPServer:
  name: string
  command: string
  config: Dict
  capabilities: List[string]
  auto_start: bool
```

### Workflow Schema
```yaml
Workflow:
  name: string
  trigger: string | List[string]
  agents: List[AgentRef]
  consolidation: string
  success_criteria: List[string]
  output: string
```

---

## INTEGRATION WITH ATLAS/LIBRIA

**Connection Points**:
- Local agents can test ATLAS Engine before deployment
- Workflows can orchestrate multi-Claude tasks
- MCPs can provide system context to remote instances
- Configuration can define when to use local vs remote agents

**Example**: Pre-commit workflow could:
1. Run LocalCodeReviewerAgent locally
2. Optionally invoke ATLAS Engine for complex analysis
3. Use git MCP to commit results

---

## DELIVERABLE CHECKLIST

- [ ] Unified configuration framework created
- [ ] 5-8 internal agents defined
- [ ] MCP servers configured
- [ ] Workflows orchestrated
- [ ] IDE integration complete
- [ ] Comprehensive instructions written
- [ ] Orchestration framework implemented
- [ ] Setup script created
- [ ] Validation script created
- [ ] Example workflows included
- [ ] Troubleshooting guide written
- [ ] Integration with ATLAS/Libria documented

---

## SUCCESS CRITERIA

- ✅ Single source of truth for all AI/LLM configurations
- ✅ Consistent agent definitions across tools
- ✅ Seamless MCP integration with CLI
- ✅ Workflows coordinate multiple agents automatically
- ✅ Easy to add new agents, MCPs, workflows
- ✅ Works across WSL, PowerShell, and all IDEs
- ✅ Clear documentation for each component
- ✅ Integration points with ATLAS/Libria defined

---

## BONUS FEATURES (If Time Allows)

1. **Dashboard**: Web UI to visualize agents, workflows, MCPs
2. **Monitoring**: Track agent execution, performance metrics
3. **Logging**: Centralized logging for all agents
4. **Analytics**: Usage statistics, performance trends
5. **Auto-Update**: Automatic configuration sync
6. **Cloud Sync**: Backup/sync configurations to cloud
7. **Version Control**: Git tracking of configurations
8. **Community Recipes**: Share workflows with others

---

## REFERENCES

- **ATLAS_LIBRIA_INTEGRATION_SPEC.md** - Integration patterns
- **IMPLEMENTATION_MASTER_PLAN.md** - Architecture patterns
- **MCP Documentation** - Model Context Protocol spec
- **Claude Code CLI** - CLI reference

---

## SUCCESS HANDOFF

Once complete, this framework will enable:
- Instant agent provisioning for new tasks
- Consistent workflows across all tools
- Seamless local-to-cloud escalation
- Comprehensive development environment
- Easy onboarding for new developers

---

**Next Steps After Setup**:
1. Test with local projects
2. Integrate with ATLAS/Libria
3. Share configurations across team
4. Iterate based on usage
5. Expand agent library as needed

---

**Estimated Effort**: 2-4 hours
**Complexity**: Medium (well-defined scope)
**Value**: Very High (unified development environment)

---

**Start with Step 1: Discover Existing Configurations**
Then proceed sequentially through Steps 2-8.

Good luck! 🚀
