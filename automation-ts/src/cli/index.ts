#!/usr/bin/env node

import { Command } from 'commander';
import {
  loadAgents,
  loadWorkflows,
  loadPrompts,
  loadPatterns,
  routeTask,
  getAutomationPath
} from '../index';
import { validateAll } from '../validation';
import { executeWorkflow } from '../executor';
import { listProjects, getProjectStats, listTemplates } from '../deployment';
import { loadCrews, listCrews } from '../crews';

const program = new Command();

program
  .name('automation')
  .description('TypeScript CLI for AI automation asset management')
  .version('1.0.0');

// Info command
program
  .command('info')
  .description('Show system information')
  .action(() => {
    const automationPath = getAutomationPath();
    const prompts = loadPrompts();
    const agents = loadAgents();
    const workflows = loadWorkflows();
    const patterns = loadPatterns();
    const crews = loadCrews();

    console.log('\n📊 Automation System Info\n');
    console.log(`Version: 1.0.0`);
    console.log(`Automation Path: ${automationPath}`);
    console.log(`\nAssets:`);
    console.log(`  Prompts:   ${prompts.length}`);
    console.log(`  Agents:    ${agents.size}`);
    console.log(`  Workflows: ${workflows.size}`);
    console.log(`  Patterns:  ${patterns.length}`);
    console.log(`  Crews:     ${crews.size}`);
    console.log('');
  });

// Prompts commands
const promptsCmd = program
  .command('prompts')
  .description('Manage prompts');

promptsCmd
  .command('list')
  .description('List all prompts')
  .option('-c, --category <category>', 'Filter by category (system, project, tasks)')
  .action((options) => {
    const prompts = loadPrompts();
    let filtered = prompts;

    if (options.category) {
      filtered = prompts.filter(p => p.category === options.category);
    }

    console.log('\n📝 Prompts\n');

    const byCategory = new Map<string, typeof prompts>();
    for (const prompt of filtered) {
      const list = byCategory.get(prompt.category) || [];
      list.push(prompt);
      byCategory.set(prompt.category, list);
    }

    for (const [category, categoryPrompts] of byCategory) {
      console.log(`\n${category.toUpperCase()} (${categoryPrompts.length}):`);
      for (const prompt of categoryPrompts) {
        const sizeKB = (prompt.size / 1024).toFixed(1);
        console.log(`  - ${prompt.name} (${sizeKB} KB)`);
      }
    }
    console.log(`\nTotal: ${filtered.length} prompts\n`);
  });

// Agents commands
const agentsCmd = program
  .command('agents')
  .description('Manage agents');

agentsCmd
  .command('list')
  .description('List all agents')
  .action(() => {
    const agents = loadAgents();

    console.log('\n🤖 Agents\n');
    for (const [name, agent] of agents) {
      console.log(`  ${name}:`);
      console.log(`    Role: ${agent.role}`);
      console.log(`    Goal: ${agent.goal.substring(0, 60)}...`);
      if (agent.tools) {
        console.log(`    Tools: ${agent.tools.join(', ')}`);
      }
      console.log('');
    }
    console.log(`Total: ${agents.size} agents\n`);
  });

// Workflows commands
const workflowsCmd = program
  .command('workflows')
  .description('Manage workflows');

workflowsCmd
  .command('list')
  .description('List all workflows')
  .action(() => {
    const workflows = loadWorkflows();

    console.log('\n⚡ Workflows\n');
    for (const [name, workflow] of workflows) {
      console.log(`  ${name}:`);
      console.log(`    Name: ${workflow.name}`);
      console.log(`    Pattern: ${workflow.pattern}`);
      console.log(`    Stages: ${workflow.stages?.length || 0}`);
      console.log('');
    }
    console.log(`Total: ${workflows.size} workflows\n`);
  });

// Route command
program
  .command('route <task>')
  .description('Route a task to appropriate tools')
  .action((task) => {
    const result = routeTask(task);

    console.log('\n🎯 Task Routing\n');
    console.log(`Task: "${task}"`);
    console.log(`\nDetected Type: ${result.task_type} (${(result.confidence * 100).toFixed(0)}% confidence)`);
    console.log(`Recommended Tools: ${result.recommended_tools.join(', ')}`);
    console.log(`Suggested Agents: ${result.suggested_agents.join(', ')}`);
    console.log('');
  });

// Patterns command
program
  .command('patterns')
  .description('List orchestration patterns')
  .action(() => {
    const patterns = loadPatterns();

    console.log('\n🔄 Orchestration Patterns\n');

    // Also show built-in patterns
    const builtIn = [
      { name: 'prompt_chaining', description: 'Sequential processing with output passing' },
      { name: 'routing', description: 'Conditional branching based on input' },
      { name: 'parallelization', description: 'Concurrent execution of independent tasks' },
      { name: 'orchestrator_workers', description: 'Central coordinator with worker agents' },
      { name: 'evaluator_optimizer', description: 'Iterative improvement with feedback' }
    ];

    console.log('Built-in Patterns:');
    for (const pattern of builtIn) {
      console.log(`  - ${pattern.name}: ${pattern.description}`);
    }

    if (patterns.length > 0) {
      console.log('\nCustom Patterns:');
      for (const pattern of patterns) {
        console.log(`  - ${pattern.name}: ${pattern.description}`);
      }
    }

    console.log(`\nTotal: ${builtIn.length + patterns.length} patterns\n`);
  });

// Validate command
program
  .command('validate')
  .description('Validate all automation assets')
  .action(() => {
    console.log('\n🔍 Validating assets...\n');

    const result = validateAll();

    if (result.errors.length > 0) {
      console.log('❌ Errors:');
      for (const error of result.errors) {
        console.log(`  - [${error.type}] ${error.message}`);
        if (error.path) console.log(`    Path: ${error.path}`);
      }
    }

    if (result.warnings.length > 0) {
      console.log('\n⚠️  Warnings:');
      for (const warning of result.warnings) {
        console.log(`  - [${warning.type}] ${warning.message}`);
        if (warning.path) console.log(`    Path: ${warning.path}`);
      }
    }

    console.log(`\n📊 Summary: ${result.errors.length} errors, ${result.warnings.length} warnings`);
    console.log(result.valid ? '✅ Validation passed\n' : '❌ Validation failed\n');

    process.exit(result.valid ? 0 : 1);
  });

// Execute command
program
  .command('execute <workflow>')
  .description('Execute a workflow')
  .option('-i, --input <json>', 'Input data as JSON')
  .action(async (workflowName, options) => {
    console.log(`\n⚡ Executing workflow: ${workflowName}\n`);

    let inputs = {};
    if (options.input) {
      try {
        inputs = JSON.parse(options.input);
      } catch {
        console.error('Invalid JSON input');
        process.exit(1);
      }
    }

    const result = await executeWorkflow(workflowName, inputs);

    if (result.success) {
      console.log('✅ Workflow completed successfully');
      console.log(`Duration: ${result.duration_ms}ms`);
      console.log(`Stages completed: ${result.stages_completed.join(' → ')}`);
    } else {
      console.log('❌ Workflow failed');
      console.log(`Error: ${result.error}`);
    }
    console.log('');
  });

// Deploy commands
const deployCmd = program
  .command('deploy')
  .description('Deployment management');

deployCmd
  .command('list')
  .description('List all projects')
  .option('-o, --org <organization>', 'Filter by organization')
  .option('-t, --type <type>', 'Filter by type')
  .action((options) => {
    let projects = listProjects();

    if (options.org) {
      projects = projects.filter(p => p.organization === options.org);
    }
    if (options.type) {
      projects = projects.filter(p => p.type === options.type);
    }

    console.log('\n📦 Projects\n');
    for (const project of projects) {
      console.log(`  ${project.name}:`);
      console.log(`    Path: ${project.path}`);
      console.log(`    Type: ${project.type}`);
      console.log(`    Org: ${project.organization}`);
      console.log(`    Tech: ${project.technologies.join(', ')}`);
      console.log('');
    }
    console.log(`Total: ${projects.length} projects\n`);
  });

deployCmd
  .command('stats')
  .description('Show deployment statistics')
  .action(() => {
    const stats = getProjectStats();

    console.log('\n📊 Deployment Statistics\n');
    console.log(`Total Projects: ${stats.total}\n`);

    console.log('By Type:');
    for (const [type, count] of Object.entries(stats.byType)) {
      console.log(`  ${type}: ${count}`);
    }

    console.log('\nBy Organization:');
    for (const [org, count] of Object.entries(stats.byOrganization)) {
      console.log(`  ${org}: ${count}`);
    }

    console.log('\nBy Technology:');
    for (const [tech, count] of Object.entries(stats.byTechnology)) {
      console.log(`  ${tech}: ${count}`);
    }
    console.log('');
  });

deployCmd
  .command('templates')
  .description('List deployment templates')
  .action(() => {
    const templates = listTemplates();

    console.log('\n📋 Deployment Templates\n');
    for (const template of templates) {
      console.log(`  ${template.name}:`);
      console.log(`    Description: ${template.description}`);
      console.log(`    Platform: ${template.platform}`);
      console.log(`    Files: ${template.files.join(', ')}`);
      console.log('');
    }
    console.log(`Total: ${templates.length} templates\n`);
  });

// Crews commands
const crewsCmd = program
  .command('crews')
  .description('Manage crews');

crewsCmd
  .command('list')
  .description('List all crews')
  .action(() => {
    const crewNames = listCrews();
    const crews = loadCrews();

    console.log('\n👥 Crews\n');
    for (const name of crewNames) {
      const crew = crews.get(name);
      if (crew) {
        console.log(`  ${name}:`);
        console.log(`    Description: ${crew.description}`);
        console.log(`    Agents: ${crew.agents?.length || 0}`);
        console.log(`    Tasks: ${crew.tasks?.length || 0}`);
        console.log('');
      }
    }
    console.log(`Total: ${crewNames.length} crews\n`);
  });

program.parse();
