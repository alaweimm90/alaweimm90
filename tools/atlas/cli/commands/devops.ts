/**
 * ATLAS DevOps CLI Commands
 * Commands for managing DevOps agents and workflows
 */

import { Command } from 'commander';
import chalk from 'chalk';
import {
  loadDevOpsAgents,
  listDevOpsAgents,
  getDevOpsAgent,
  getAgentsByCategory,
  executeDevOpsWorkflow,
  CI_CD_PIPELINE,
  SECURE_RELEASE,
  INCIDENT_RESPONSE,
  type DevOpsWorkflow,
} from '@atlas/orchestration/devops-agents.js';
import { output, getDefaultStyle } from '@atlas/cli/utils.js';

const NO_COLOR = Boolean(process.env.NO_COLOR);
const cyan = (s: string): string => (NO_COLOR ? s : chalk.cyan(s));
const gray = (s: string): string => (NO_COLOR ? s : chalk.gray(s));
const green = (s: string): string => (NO_COLOR ? s : chalk.green(s));
const yellow = (s: string): string => (NO_COLOR ? s : chalk.yellow(s));
const red = (s: string): string => (NO_COLOR ? s : chalk.red(s));
const bold = (s: string): string => (NO_COLOR ? s : chalk.bold(s));

// Pre-built workflows registry
const WORKFLOWS: Record<string, DevOpsWorkflow> = {
  'ci-cd-pipeline': CI_CD_PIPELINE,
  'secure-release': SECURE_RELEASE,
  'incident-response': INCIDENT_RESPONSE,
};

// Default circuit breaker config
const DEFAULT_CB_CONFIG = {
  failureThreshold: 5,
  successThreshold: 3,
  timeout: 60000,
  halfOpenRequests: 1,
};

export function registerDevOpsCommands(program: Command): void {
  // Initialize agents on registration
  loadDevOpsAgents();

  const cmd = program.command('devops').description('DevOps agent management and workflow execution');

  // List all agents
  cmd
    .command('agents')
    .description('List all DevOps agents')
    .option('--category <name>', 'Filter by category (pipeline, security, observability, release)')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action((opts: { category?: string; style?: string }) => {
      const style = opts.style || getDefaultStyle();
      const agents = opts.category ? getAgentsByCategory(opts.category) : listDevOpsAgents();

      if (agents.length === 0) {
        output.warning('No agents found');
        return;
      }

      if (style === 'json') {
        console.log(JSON.stringify(agents, null, 2));
        return;
      }

      output.info(bold(`DevOps Agents (${agents.length})`));
      output.info('');

      // Group by category
      const byCategory = new Map<string, typeof agents>();
      for (const agent of agents) {
        const cat = agent.category;
        if (!byCategory.has(cat)) byCategory.set(cat, []);
        byCategory.get(cat)!.push(agent);
      }

      for (const [category, categoryAgents] of byCategory) {
        output.info(cyan(`[${category}]`));
        for (const agent of categoryAgents) {
          const id = green(agent.id.padEnd(22));
          const role = gray(agent.role);
          output.info(`  ${id} ${role}`);
        }
        output.info('');
      }
    });

  // Show agent details
  cmd
    .command('agent <id>')
    .description('Show details for a specific DevOps agent')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action((id: string, opts: { style?: string }) => {
      const style = opts.style || getDefaultStyle();
      const agent = getDevOpsAgent(id);

      if (!agent) {
        output.error(`Agent not found: ${id}`);
        return;
      }

      if (style === 'json') {
        console.log(JSON.stringify(agent, null, 2));
        return;
      }

      output.info(bold(agent.role));
      output.info('');
      output.info(`${gray('ID:')}       ${green(agent.id)}`);
      output.info(`${gray('Category:')} ${cyan(agent.category)}`);
      output.info(`${gray('Goal:')}     ${agent.goal}`);
      output.info('');
      output.info(`${gray('Aliases:')}  ${agent.aliases.join(', ') || 'none'}`);
      output.info(`${gray('Tools:')}    ${agent.tools.join(', ')}`);
      output.info(`${gray('Inputs:')}   ${agent.inputs.join(', ')}`);
      output.info(`${gray('Outputs:')}  ${agent.outputs.join(', ')}`);
    });

  // List workflows
  cmd
    .command('workflows')
    .description('List available DevOps workflows')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action((opts: { style?: string }) => {
      const style = opts.style || getDefaultStyle();

      if (style === 'json') {
        console.log(
          JSON.stringify(
            Object.entries(WORKFLOWS).map(([name, wf]) => ({
              name,
              description: wf.description,
              steps: wf.steps.length,
            })),
            null,
            2
          )
        );
        return;
      }

      output.info(bold('Available DevOps Workflows'));
      output.info('');

      for (const [name, wf] of Object.entries(WORKFLOWS)) {
        output.info(`${green(name.padEnd(20))} ${gray(wf.description)}`);
        output.info(`  ${gray('Steps:')} ${wf.steps.map((s) => cyan(s.agentId)).join(' → ')}`);
        output.info('');
      }
    });

  // Show workflow details
  cmd
    .command('workflow <name>')
    .description('Show details for a specific workflow')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action((name: string, opts: { style?: string }) => {
      const style = opts.style || getDefaultStyle();
      const wf = WORKFLOWS[name];

      if (!wf) {
        output.error(`Workflow not found: ${name}`);
        output.info(`Available: ${Object.keys(WORKFLOWS).join(', ')}`);
        return;
      }

      if (style === 'json') {
        console.log(JSON.stringify(wf, null, 2));
        return;
      }

      output.info(bold(wf.name));
      output.info(gray(wf.description));
      output.info('');
      output.info(bold('Steps:'));

      for (let i = 0; i < wf.steps.length; i++) {
        const step = wf.steps[i];
        const num = gray(`${i + 1}.`);
        const id = cyan(step.id.padEnd(20));
        const agent = green(step.agentId);
        const action = yellow(step.action);
        output.info(`  ${num} ${id} ${agent} → ${action}`);

        if (step.dependsOn?.length) {
          output.info(`     ${gray('depends on:')} ${step.dependsOn.join(', ')}`);
        }
        if (step.onError) {
          output.info(`     ${gray('on error:')} ${step.onError}`);
        }
      }
    });

  // Run a workflow
  cmd
    .command('run <workflow>')
    .description('Execute a DevOps workflow')
    .option('--dry-run', 'Show what would be executed without running')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action(async (workflowName: string, opts: { dryRun?: boolean; style?: string }) => {
      const style = opts.style || getDefaultStyle();
      const wf = WORKFLOWS[workflowName];

      if (!wf) {
        output.error(`Workflow not found: ${workflowName}`);
        output.info(`Available: ${Object.keys(WORKFLOWS).join(', ')}`);
        return;
      }

      if (opts.dryRun) {
        output.info(bold(`[DRY RUN] Would execute: ${wf.name}`));
        output.info('');
        for (const step of wf.steps) {
          output.info(`  ${gray('→')} ${cyan(step.id)}: ${green(step.agentId)} ${yellow(step.action)}`);
        }
        return;
      }

      output.info(bold(`Executing workflow: ${wf.name}`));
      output.info(gray(wf.description));
      output.info('');

      const startTime = Date.now();

      try {
        const result = await executeDevOpsWorkflow(wf, DEFAULT_CB_CONFIG);

        if (style === 'json') {
          console.log(
            JSON.stringify(
              {
                ...result,
                stepResults: Object.fromEntries(result.stepResults),
              },
              null,
              2
            )
          );
          return;
        }

        output.info('');
        output.info(bold('Results:'));

        for (const [stepId, stepResult] of result.stepResults) {
          const status = stepResult.success ? green('✓') : red('✗');
          const latency = stepResult.latency ? gray(`(${stepResult.latency}ms)`) : '';
          const error = stepResult.error ? red(stepResult.error) : '';
          output.info(`  ${status} ${cyan(stepId)} ${latency} ${error}`);
        }

        output.info('');
        const totalTime = Date.now() - startTime;
        if (result.success) {
          output.success(`Workflow completed in ${totalTime}ms`);
        } else {
          output.error(`Workflow failed. Failed steps: ${result.failedSteps.join(', ')}`);
        }
      } catch (error) {
        output.error(`Workflow execution error: ${error instanceof Error ? error.message : String(error)}`);
      }
    });

  // Quick commands for common workflows
  cmd
    .command('ci')
    .description('Run CI/CD pipeline (shortcut for: devops run ci-cd-pipeline)')
    .option('--dry-run', 'Show what would be executed without running')
    .action(async (opts: { dryRun?: boolean }) => {
      const args = ['run', 'ci-cd-pipeline'];
      if (opts.dryRun) args.push('--dry-run');
      await cmd.parseAsync(args, { from: 'user' });
    });

  cmd
    .command('release')
    .description('Run secure release (shortcut for: devops run secure-release)')
    .option('--dry-run', 'Show what would be executed without running')
    .action(async (opts: { dryRun?: boolean }) => {
      const args = ['run', 'secure-release'];
      if (opts.dryRun) args.push('--dry-run');
      await cmd.parseAsync(args, { from: 'user' });
    });

  cmd
    .command('incident')
    .description('Run incident response (shortcut for: devops run incident-response)')
    .option('--dry-run', 'Show what would be executed without running')
    .action(async (opts: { dryRun?: boolean }) => {
      const args = ['run', 'incident-response'];
      if (opts.dryRun) args.push('--dry-run');
      await cmd.parseAsync(args, { from: 'user' });
    });
}
