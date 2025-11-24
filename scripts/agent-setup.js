#!/usr/bin/env node

/**
 * Agent Setup Script - Initialize agent and orchestration framework
 * Usage: node scripts/agent-setup.js [options]
 */

const fs = require('fs');
const path = require('path');

class AgentSetup {
  constructor(options = {}) {
    this.options = options;
    this.rootDir = process.cwd();
    this.claudeDir = path.join(this.rootDir, '.claude');
  }

  log(message) {
    console.log(`[Agent Setup] ${message}`);
  }

  /**
   * Create agent directory structure
   */
  createAgentStructure() {
    const agentDir = path.join(this.claudeDir, 'agents');
    const workflowDir = path.join(this.claudeDir, 'workflows');

    if (!fs.existsSync(agentDir)) {
      fs.mkdirSync(agentDir, { recursive: true });
      this.log(`Created agents directory: ${agentDir}`);
    }

    if (!fs.existsSync(workflowDir)) {
      fs.mkdirSync(workflowDir, { recursive: true });
      this.log(`Created workflows directory: ${workflowDir}`);
    }
  }

  /**
   * Create default agent definitions
   */
  createDefaultAgents() {
    const agents = {
      'code-agent.json': {
        id: 'code-agent',
        name: 'Code Agent',
        description: 'Handles code manipulation, analysis, and generation',
        version: '1.0.0',
        type: 'code',
        capabilities: ['code-review', 'code-fix', 'code-feature', 'refactor', 'test'],
        requiredMcps: ['filesystem', 'git'],
        enabled: true,
        config: {
          lintOnExecute: true,
          typeCheckOnExecute: true,
          testOnExecute: false,
        },
      },
      'analysis-agent.json': {
        id: 'analysis-agent',
        name: 'Analysis Agent',
        description: 'Performs code analysis, testing, and security scanning',
        version: '1.0.0',
        type: 'analysis',
        capabilities: ['analyze', 'security-scan', 'test', 'lint', 'type-check', 'performance-analysis'],
        requiredMcps: ['filesystem', 'git'],
        enabled: true,
        config: {
          performSecurityScan: true,
          checkPerformance: true,
          generateReports: true,
        },
      },
    };

    const agentDir = path.join(this.claudeDir, 'agents');
    for (const [filename, agentDef] of Object.entries(agents)) {
      const filepath = path.join(agentDir, filename);
      fs.writeFileSync(filepath, JSON.stringify(agentDef, null, 2), 'utf-8');
      this.log(`Created agent definition: ${filepath}`);
    }
  }

  /**
   * Create default workflows
   */
  createDefaultWorkflows() {
    const workflows = {
      'code-review.json': {
        id: 'code-review-workflow',
        name: 'Code Review Workflow',
        description: 'Standard code review process',
        version: '1.0.0',
        enabled: true,
        steps: [
          {
            id: 'step-lint',
            name: 'Lint Check',
            type: 'task',
            action: 'lint',
            agentId: 'code-agent',
          },
          {
            id: 'step-type',
            name: 'Type Check',
            type: 'task',
            action: 'type-check',
            agentId: 'code-agent',
          },
          {
            id: 'step-test',
            name: 'Test Execution',
            type: 'task',
            action: 'test',
            agentId: 'code-agent',
          },
          {
            id: 'step-security',
            name: 'Security Scan',
            type: 'task',
            action: 'security-scan',
            agentId: 'analysis-agent',
          },
        ],
      },
      'bug-fix.json': {
        id: 'bug-fix-workflow',
        name: 'Bug Fix Workflow',
        description: 'Process for identifying and fixing bugs',
        version: '1.0.0',
        enabled: true,
        steps: [
          {
            id: 'step-reproduce',
            name: 'Reproduce Bug',
            type: 'task',
            action: 'test',
            agentId: 'code-agent',
          },
          {
            id: 'step-analyze',
            name: 'Analyze Issue',
            type: 'task',
            action: 'analyze',
            agentId: 'analysis-agent',
          },
          {
            id: 'step-fix',
            name: 'Implement Fix',
            type: 'task',
            action: 'code-fix',
            agentId: 'code-agent',
          },
          {
            id: 'step-verify',
            name: 'Verify Fix',
            type: 'task',
            action: 'test',
            agentId: 'code-agent',
          },
        ],
      },
    };

    const workflowDir = path.join(this.claudeDir, 'workflows');
    for (const [filename, workflow] of Object.entries(workflows)) {
      const filepath = path.join(workflowDir, filename);
      fs.writeFileSync(filepath, JSON.stringify(workflow, null, 2), 'utf-8');
      this.log(`Created workflow: ${filepath}`);
    }
  }

  /**
   * Run full setup
   */
  run() {
    try {
      this.log('Starting agent and workflow setup...');

      // Create directory structure
      this.createAgentStructure();

      // Create default agents
      this.createDefaultAgents();

      // Create default workflows
      this.createDefaultWorkflows();

      this.log('Agent setup completed successfully!');
      this.printSummary();
    } catch (error) {
      console.error('[Agent Setup ERROR]', error.message);
      process.exit(1);
    }
  }

  printSummary() {
    console.log('\n--- Agent Setup Summary ---');
    console.log('Created structure:');
    console.log('  .claude/');
    console.log('    ├── agents/');
    console.log('    │   ├── code-agent.json');
    console.log('    │   └── analysis-agent.json');
    console.log('    └── workflows/');
    console.log('        ├── code-review.json');
    console.log('        └── bug-fix.json');
    console.log('\nNext steps:');
    console.log('1. Review agent definitions: cat .claude/agents/*.json');
    console.log('2. Review workflows: cat .claude/workflows/*.json');
    console.log('3. Read documentation: docs/AGENTS_AND_WORKFLOWS.md');
  }
}

const setup = new AgentSetup();
setup.run();
