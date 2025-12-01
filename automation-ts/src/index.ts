// Automation TypeScript Library
// Main entry point for programmatic usage

export * from './types';
export * from './utils/file';
export * from './validation';
export * from './executor';
export * from './deployment';
export * from './crews';

import * as path from 'path';
import * as fs from 'fs';
import { getAutomationPath, readYamlFile, listFilesRecursive } from './utils/file';
import type { Agent, Workflow, Prompt, OrchestrationPattern } from './types';

/**
 * Load all agents from the automation folder
 */
export function loadAgents(): Map<string, Agent> {
  const automationPath = getAutomationPath();
  const agentsPath = path.join(automationPath, 'agents', 'config', 'agents.yaml');

  const data = readYamlFile<{ agents: Record<string, Agent> }>(agentsPath);
  if (!data?.agents) {
    return new Map();
  }

  return new Map(Object.entries(data.agents));
}

/**
 * Load a specific agent by name
 */
export function loadAgent(name: string): Agent | null {
  const agents = loadAgents();
  return agents.get(name) || null;
}

/**
 * Load all workflows from the automation folder
 */
export function loadWorkflows(): Map<string, Workflow> {
  const automationPath = getAutomationPath();
  const workflowsPath = path.join(automationPath, 'workflows', 'config', 'workflows.yaml');

  const data = readYamlFile<{ workflows: Record<string, Workflow> }>(workflowsPath);
  if (!data?.workflows) {
    return new Map();
  }

  return new Map(Object.entries(data.workflows));
}

/**
 * Load a specific workflow by name
 */
export function loadWorkflow(name: string): Workflow | null {
  const workflows = loadWorkflows();
  return workflows.get(name) || null;
}

/**
 * Load all prompts from the automation folder
 */
export function loadPrompts(): Prompt[] {
  const automationPath = getAutomationPath();
  const promptsPath = path.join(automationPath, 'prompts');

  const prompts: Prompt[] = [];
  const categories: Array<'system' | 'project' | 'tasks'> = ['system', 'project', 'tasks'];

  for (const category of categories) {
    const categoryPath = path.join(promptsPath, category);
    const files = listFilesRecursive(categoryPath, '.md');

    for (const file of files) {
      const stats = fs.statSync(file);
      prompts.push({
        path: file,
        name: path.basename(file, '.md'),
        category,
        size: stats.size
      });
    }
  }

  return prompts;
}

/**
 * Load orchestration patterns
 */
export function loadPatterns(): OrchestrationPattern[] {
  const automationPath = getAutomationPath();
  const patternsPath = path.join(automationPath, 'orchestration', 'patterns');

  const patterns: OrchestrationPattern[] = [];
  const files = listFilesRecursive(patternsPath, '.yaml');

  for (const file of files) {
    const data = readYamlFile<OrchestrationPattern>(file);
    if (data) {
      patterns.push(data);
    }
  }

  return patterns;
}

/**
 * Route a task to appropriate tools based on description
 */
export function routeTask(description: string): {
  task_type: string;
  confidence: number;
  recommended_tools: string[];
  suggested_agents: string[];
} {
  const lowerDesc = description.toLowerCase();

  // Task type detection
  const patterns: Array<{ keywords: string[]; type: string; tools: string[]; agents: string[] }> = [
    {
      keywords: ['debug', 'fix', 'error', 'bug', 'issue'],
      type: 'debugging',
      tools: ['cline', 'cursor', 'claude_code'],
      agents: ['debugger_agent', 'coder_agent']
    },
    {
      keywords: ['implement', 'create', 'build', 'develop', 'feature'],
      type: 'development',
      tools: ['cursor', 'claude_code', 'copilot'],
      agents: ['coder_agent', 'architect_agent']
    },
    {
      keywords: ['refactor', 'clean', 'optimize', 'improve'],
      type: 'refactoring',
      tools: ['kilo_code', 'cursor'],
      agents: ['coder_agent', 'reviewer_agent']
    },
    {
      keywords: ['review', 'check', 'audit', 'analyze'],
      type: 'review',
      tools: ['claude_code', 'cline'],
      agents: ['reviewer_agent', 'critic_agent']
    },
    {
      keywords: ['test', 'spec', 'coverage'],
      type: 'testing',
      tools: ['cursor', 'copilot'],
      agents: ['qa_engineer_agent', 'coder_agent']
    },
    {
      keywords: ['document', 'readme', 'docs', 'explain'],
      type: 'documentation',
      tools: ['claude_code', 'cursor'],
      agents: ['writer_agent', 'technical_writer_agent']
    },
    {
      keywords: ['deploy', 'release', 'ci', 'cd', 'pipeline'],
      type: 'devops',
      tools: ['cline', 'claude_code'],
      agents: ['devops_agent', 'mlops_agent']
    },
    {
      keywords: ['research', 'investigate', 'explore', 'study'],
      type: 'research',
      tools: ['claude_code', 'perplexity'],
      agents: ['scientist_agent', 'scout_agent']
    }
  ];

  let bestMatch = { type: 'general', confidence: 0.3, tools: ['cursor', 'copilot'], agents: ['coder_agent'] };

  for (const pattern of patterns) {
    const matches = pattern.keywords.filter(kw => lowerDesc.includes(kw)).length;

    // If any keyword matches, consider it a match
    if (matches > 0) {
      const confidence = Math.min(0.5 + (matches * 0.2), 1.0);

      if (confidence > bestMatch.confidence) {
        bestMatch = {
          type: pattern.type,
          confidence,
          tools: pattern.tools,
          agents: pattern.agents
        };
      }
    }
  }

  return {
    task_type: bestMatch.type,
    confidence: bestMatch.confidence,
    recommended_tools: bestMatch.tools,
    suggested_agents: bestMatch.agents
  };
}
