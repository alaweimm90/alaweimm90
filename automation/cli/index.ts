#!/usr/bin/env node

import { Command } from 'commander';
import { AutomationCore, unifiedExecutor } from '../core';
import chalk from 'chalk';

const program = new Command();

program
  .name('automation')
  .description('Unified CLI for AI automation asset management')
  .version('1.0.0');

// Route command - replaces both Python and TS routing functionality
program
  .command('route <task>')
  .description('Route a task to appropriate workflow and tools')
  .option('-c, --confidence <threshold>', 'Minimum confidence threshold', parseFloat, 0.6)
  .action(async (task: string, options: { confidence: number }) => {
    try {
      const core = new AutomationCore();
      const route = core.routeTask(task);

      console.log(chalk.blue(`Task: "${task}"`));
      console.log(chalk.green(`Routed to: ${route.task_type}`));
      console.log(chalk.yellow(`Confidence: ${(route.confidence * 100).toFixed(1)}%`));

      if (route.confidence >= options.confidence) {
        console.log(chalk.green('✓ Meets confidence threshold'));
        console.log(chalk.cyan(`Recommended tools: ${route.recommended_tools.join(', ')}`));
        console.log(chalk.cyan(`Suggested agents: ${route.suggested_agents.join(', ')}`));
      } else {
        console.log(chalk.red('✗ Below confidence threshold'));
        console.log(chalk.gray('Please provide more specific task description'));
      }
    } catch (error) {
      console.error(chalk.red('Error routing task:'), error);
      process.exit(1);
    }
  });

// Execute command - unified workflow execution
program
  .command('execute <workflow>')
  .description('Execute a workflow with optional inputs')
  .option('-d, --dry-run', 'Perform dry run without actual execution')
  .option('-i, --input <json>', 'JSON input parameters')
  .action(async (workflow: string, options: { dryRun?: boolean; input?: string }) => {
    try {
      let inputs = {};

      if (options.input) {
        try {
          inputs = JSON.parse(options.input);
        } catch (error) {
          console.error(chalk.red('Invalid JSON input:'), error);
          process.exit(1);
        }
      }

      console.log(chalk.blue(`Executing workflow: ${workflow}`));
      if (options.dryRun) {
        console.log(chalk.yellow('[DRY RUN MODE]'));
      }

      const context = await unifiedExecutor(
        workflow,
        {
          dryRun: options.dryRun || false,
        },
        inputs
      );

      console.log(chalk.green(`✓ Execution completed: ${context.status}`));
      console.log(
        chalk.gray(
          `Duration: ${context.endTime ? context.endTime.getTime() - context.startTime.getTime() + 'ms' : 'N/A'}`
        )
      );
    } catch (error) {
      console.error(chalk.red('Error executing workflow:'), error);
      process.exit(1);
    }
  });

// List command - show available assets
program
  .command('list')
  .description('List available automation assets')
  .option('-a, --agents', 'List agents')
  .option('-w, --workflows', 'List workflows')
  .option('-p, --prompts', 'List prompts')
  .action(async (options: { agents?: boolean; workflows?: boolean; prompts?: boolean }) => {
    try {
      const core = new AutomationCore();

      if (options.agents || (!options.agents && !options.workflows && !options.prompts)) {
        const agents = Array.from(core.getAllAgents().keys());
        console.log(chalk.blue('\n📋 Available Agents:'));
        agents.forEach((agent) => console.log(chalk.gray(`  • ${agent}`)));
      }

      if (options.workflows || (!options.agents && !options.workflows && !options.prompts)) {
        const workflows = Array.from(core.getAllWorkflows().keys());
        console.log(chalk.blue('\n⚡ Available Workflows:'));
        workflows.forEach((workflow) => console.log(chalk.gray(`  • ${workflow}`)));
      }

      if (options.prompts || (!options.agents && !options.workflows && !options.prompts)) {
        const prompts = core.getAllPrompts();
        console.log(chalk.blue('\n📝 Available Prompts:'));
        prompts.forEach((prompt) =>
          console.log(chalk.gray(`  • ${prompt.name} (${prompt.category})`))
        );
      }
    } catch (error) {
      console.error(chalk.red('Error listing assets:'), error);
      process.exit(1);
    }
  });

// Info command - get detailed information about assets
program
  .command('info <name>')
  .description('Get detailed information about an asset')
  .option('-t, --type <type>', 'Asset type (agent|workflow)', 'agent')
  .action(async (name: string, options: { type: string }) => {
    try {
      const core = new AutomationCore();

      if (options.type === 'agent') {
        const agent = core.getAgent(name);
        if (agent) {
          console.log(chalk.blue(`\n🤖 Agent: ${name}`));
          console.log(chalk.gray(`Role: ${agent.role}`));
          console.log(chalk.gray(`Goal: ${agent.goal}`));
          if (agent.tools) {
            console.log(chalk.gray(`Tools: ${agent.tools.join(', ')}`));
          }
        } else {
          console.log(chalk.red(`Agent '${name}' not found`));
        }
      } else if (options.type === 'workflow') {
        const workflow = core.getWorkflow(name);
        if (workflow) {
          console.log(chalk.blue(`\n⚡ Workflow: ${name}`));
          console.log(chalk.gray(`Description: ${workflow.description}`));
          console.log(chalk.gray(`Stages: ${workflow.stages.length}`));
          console.log(chalk.gray(`Pattern: ${workflow.pattern}`));
        } else {
          console.log(chalk.red(`Workflow '${name}' not found`));
        }
      }
    } catch (error) {
      console.error(chalk.red('Error getting asset info:'), error);
      process.exit(1);
    }
  });

// Catch-all command for natural language task routing
program
  .command('run <task>')
  .description('Execute a task using natural language routing')
  .option('-d, --dry-run', 'Perform dry run without execution')
  .action(async (task: string, options: { dryRun?: boolean }) => {
    try {
      console.log(chalk.blue(`🎯 Processing task: "${task}"`));

      const context = await unifiedExecutor(task, {
        dryRun: options.dryRun || false,
      });

      console.log(chalk.green(`✓ Task completed: ${context.status}`));
    } catch (error) {
      console.error(chalk.red('Error executing task:'), error);
      process.exit(1);
    }
  });

// Handle no command provided - show help
program.action(() => {
  console.log(chalk.cyan('\n🚀 Unified Automation CLI'));
  console.log(chalk.gray('Consolidated Python and TypeScript automation systems\n'));
  program.help();
});

// Error handling
process.on('SIGTERM', () => {
  console.log(chalk.yellow('\n⚠️  Received SIGTERM, shutting down gracefully…'));
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log(chalk.yellow('\n⚠️  Received SIGINT, shutting down gracefully…'));
  process.exit(0);
});

// Parse command line arguments
program.parse();
