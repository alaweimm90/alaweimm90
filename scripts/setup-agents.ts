#!/usr/bin/env ts-node

/**
 * Agent Setup Script
 * Initializes agents and orchestration for Claude Code
 */

import { AgentOrchestrator, CodeAgent, AnalysisAgent, AgentType } from '@monorepo/agent-core';
import { ContextProvider } from '@monorepo/context-provider';
import { createDefaultTemplates as createDefaultWorkflows } from '@monorepo/workflow-templates';
import * as fs from 'fs';
import * as path from 'path';

async function setupAgents(): Promise<void> {
  console.log('🤖 Setting up Agents and Orchestration...\n');

  // Initialize context
  const context = ContextProvider.getInstance();
  console.log(`📍 Workspace: ${context.getWorkspaceRoot()}`);
  console.log(`📦 Project: ${context.getProjectName()}\n`);

  // Create orchestrator
  const orchestrator = new AgentOrchestrator();
  await orchestrator.initialize(context.getContext());

  // Register agents
  console.log('👥 Registering agents...');

  const codeAgent = new CodeAgent({
    id: 'code-agent',
    name: 'Code Agent',
    description: 'Handles code manipulation, review, and generation',
    version: '1.0.0',
    type: AgentType.CODE,
    capabilities: ['code-review', 'code-fix', 'code-feature', 'refactor', 'test', 'lint'],
    enabled: true,
  });

  const analysisAgent = new AnalysisAgent({
    id: 'analysis-agent',
    name: 'Analysis Agent',
    description: 'Handles code analysis, testing, and reporting',
    version: '1.0.0',
    type: AgentType.ANALYSIS,
    capabilities: ['analyze', 'security-scan', 'test', 'performance-analysis', 'type-check'],
    enabled: true,
  });

  orchestrator.registerAgent(codeAgent);
  orchestrator.registerAgent(analysisAgent);
  console.log('  ✓ Code Agent registered');
  console.log('  ✓ Analysis Agent registered');

  // Register workflows
  console.log('\n📋 Registering workflows...');

  const defaultWorkflows = createDefaultWorkflows();

  for (const workflow of defaultWorkflows) {
    orchestrator.registerWorkflow(workflow);
    console.log(`  ✓ ${workflow.name}`);
  }

  // Save agent configuration
  console.log('\n💾 Saving agent configuration...');
  const agentConfig = {
    agents: orchestrator.getAgents().map(agent => agent.getConfig()),
    workflows: defaultWorkflows,
    timestamp: new Date().toISOString(),
  };

  const claudeDir = path.join(context.getWorkspaceRoot(), '.claude');
  if (!fs.existsSync(claudeDir)) {
    fs.mkdirSync(claudeDir, { recursive: true });
  }

  fs.writeFileSync(
    path.join(claudeDir, 'agents.json'),
    JSON.stringify(agentConfig, null, 2),
    'utf-8'
  );
  console.log('  ✓ Agent configuration saved to .claude/agents.json');

  console.log('\n✨ Agent setup complete!\n');
  console.log('Summary:');
  console.log(`  - ${orchestrator.getAgents().length} agents registered`);
  console.log(`  - ${defaultWorkflows.length} workflows registered`);
  console.log('\nNext steps:');
  console.log('1. Create custom agents by extending BaseAgent');
  console.log('2. Define custom workflows for your development process');
  console.log('3. Set up orchestration rules in .claude/orchestration.json\n');
}

setupAgents().catch(error => {
  console.error('❌ Error setting up agents:', error);
  process.exit(1);
});
