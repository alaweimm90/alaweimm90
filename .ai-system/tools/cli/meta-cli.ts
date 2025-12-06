#!/usr/bin/env npx tsx

/**
 * Meta CLI - Unified command-line interface for the meta-governance repository
 *
 * This consolidates 66+ npm scripts into a single, discoverable CLI with subcommands.
 */

import { Command } from 'commander';
import { spawn } from 'child_process';
import chalk from 'chalk';

const program = new Command();

// CLI Configuration
program.name('meta').description('Meta-governance repository CLI').version('2.0.0');

// ============================================================================
// AI Commands - Consolidate 38 ai:* scripts
// ============================================================================

const ai = program
  .command('ai')
  .description('AI orchestration, monitoring, and compliance tools')
  .action(() => {
    ai.outputHelp();
  });

// Orchestration commands
ai.command('start <task>')
  .description('Start AI orchestration task')
  .action((task) => runCommand('tsx', ['tools/ai/orchestrator.ts', 'start', task]));

ai.command('complete')
  .description('Complete AI task')
  .action(() => runCommand('tsx', ['tools/ai/orchestrator.ts', 'complete']));

ai.command('context <type>')
  .description('Get AI context for task type')
  .action((type) => runCommand('tsx', ['tools/ai/orchestrator.ts', 'context', type]));

ai.command('metrics')
  .description('View AI metrics')
  .action(() => runCommand('tsx', ['tools/ai/orchestrator.ts', 'metrics']));

// Cache management
const cache = ai.command('cache').description('AI cache management');

cache
  .command('stats')
  .description('View cache statistics')
  .action(() => runCommand('tsx', ['tools/ai/cache.ts', 'stats']));

cache
  .command('clear')
  .description('Clear AI cache')
  .action(() => runCommand('tsx', ['tools/ai/cache.ts', 'clear']));

// Monitor commands
const monitor = ai.command('monitor').description('AI monitoring tools');

monitor
  .command('status')
  .description('Check monitor status')
  .action(() => runCommand('tsx', ['tools/ai/monitor.ts', 'status']));

monitor
  .command('check')
  .description('Run monitor checks')
  .action(() => runCommand('tsx', ['tools/ai/monitor.ts', 'check']));

// Compliance commands
const compliance = ai.command('compliance').description('AI compliance tools');

compliance
  .command('check')
  .description('Run compliance checks')
  .action(() => runCommand('tsx', ['tools/ai/cli/compliance-cli.ts', 'check']));

compliance
  .command('score')
  .description('Get compliance score')
  .action(() => runCommand('tsx', ['tools/ai/cli/compliance-cli.ts', 'score']));

// Security commands
const security = ai.command('security').description('AI security scanning');

security
  .command('scan')
  .description('Run security scan')
  .action(() => runCommand('tsx', ['tools/ai/cli/security-cli.ts', 'scan']));

security
  .command('secrets')
  .description('Scan for secrets')
  .action(() => runCommand('tsx', ['tools/ai/cli/security-cli.ts', 'secrets']));

security
  .command('vulns')
  .description('Check vulnerabilities')
  .action(() => runCommand('tsx', ['tools/ai/cli/security-cli.ts', 'vulns']));

// ============================================================================
// ORCHEX Commands - Research and orchestration
// ============================================================================

const ORCHEX = program
  .command('ORCHEX')
  .description('ORCHEX research orchestration platform')
  .action(() => runCommand('tsx', ['tools/ORCHEX/cli/index.ts']));

ORCHEX
  .command('api')
  .description('Start ORCHEX API server')
  .action(() => runCommand('tsx', ['tools/ORCHEX/api/cli.ts']));

ORCHEX
  .command('migrate')
  .description('Run storage migrations')
  .action(() => runCommand('tsx', ['tools/ORCHEX/storage/migrate.ts']));

// ============================================================================
// DevOps Commands - Template and generation tools
// ============================================================================

const devops = program
  .command('devops')
  .description('DevOps template and generation tools')
  .action(() => runCommand('tsx', ['tools/cli/devops.ts']));

devops
  .command('init')
  .description('Initialize DevOps environment')
  .action(() => runCommand('tsx', ['tools/cli/devops.ts', 'init']));

devops
  .command('setup')
  .description('Setup DevOps tools')
  .action(() => runCommand('tsx', ['tools/cli/devops.ts', 'setup']));

const template = devops.command('template').description('Template management');

template
  .command('list')
  .description('List available templates')
  .action(() => runCommand('tsx', ['tools/cli/devops.ts', 'template', 'list']));

template
  .command('apply <name>')
  .description('Apply a template')
  .action((name) => runCommand('tsx', ['tools/cli/devops.ts', 'template', 'apply', name]));

devops
  .command('generate <type>')
  .description('Generate DevOps resources')
  .option('--dry-run', 'Preview without creating files')
  .action((type, options) => {
    const args = ['tools/cli/devops.ts', 'generate', type];
    if (options.dryRun) args.push('--dry-run');
    runCommand('tsx', args);
  });

// ============================================================================
// Automation Commands
// ============================================================================

const automation = program
  .command('automation')
  .alias('auto')
  .description('Workflow automation tools')
  .action(() => runCommand('tsx', ['automation/cli/index.ts']));

automation
  .command('list')
  .description('List automation workflows')
  .action(() => runCommand('tsx', ['automation/cli/index.ts', 'list']));

automation
  .command('execute <workflow>')
  .description('Execute a workflow')
  .action((workflow) => runCommand('tsx', ['automation/cli/index.ts', 'execute', workflow]));

automation
  .command('route <task>')
  .description('Route task to appropriate handler')
  .action((task) => runCommand('tsx', ['automation/cli/index.ts', 'route', task]));

// ============================================================================
// Development Commands - Standard tooling
// ============================================================================

const dev = program.command('dev').description('Development tools (lint, test, format)');

dev
  .command('lint')
  .description('Run ESLint')
  .option('--fix', 'Auto-fix issues')
  .action((options) => {
    const args = ['.'];
    if (options.fix) args.push('--fix');
    runCommand('npx', ['eslint', ...args]);
  });

dev
  .command('format')
  .description('Format code with Prettier')
  .option('--check', 'Check formatting without fixing')
  .action((options) => {
    const args = options.check ? ['--check', '.'] : ['--write', '.'];
    runCommand('npx', ['prettier', ...args]);
  });

dev
  .command('test')
  .description('Run tests')
  .option('--coverage', 'Generate coverage report')
  .option('--watch', 'Watch mode')
  .action((options) => {
    const args = [];
    if (options.coverage) args.push('run', '--coverage');
    else if (options.watch) args.push('watch');
    else args.push('run');
    runCommand('npx', ['vitest', ...args]);
  });

dev
  .command('type-check')
  .alias('tsc')
  .description('Check TypeScript types')
  .action(() => runCommand('npx', ['tsc', '--noEmit']));

// ============================================================================
// Python Tools Commands
// ============================================================================

program
  .command('governance')
  .description('Governance tools')
  .action(() => runCommand('python', ['tools/cli/governance.py']));

program
  .command('orchestrate')
  .description('Orchestration tools')
  .action(() => runCommand('python', ['tools/cli/orchestrate.py']));

program
  .command('mcp')
  .description('MCP server tools')
  .action(() => runCommand('python', ['tools/cli/mcp.py']));

// ============================================================================
// Helper Commands
// ============================================================================

program
  .command('clean')
  .description('Clean build artifacts and caches')
  .action(() => {
    console.log(chalk.yellow('🧹 Cleaning build artifacts...'));
    runCommand('rm', ['-rf', 'dist', '.cache', '.mypy_cache', 'coverage']);
  });

program
  .command('setup')
  .description('Initial repository setup')
  .action(async () => {
    console.log(chalk.cyan('🚀 Setting up repository...'));
    await runCommand('npm', ['install']);
    await runCommand('npx', ['husky', 'install']);
    console.log(chalk.green('✅ Setup complete!'));
  });

// ============================================================================
// Utilities
// ============================================================================

function runCommand(command: string, args: string[] = []): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
      shell: true,
      cwd: process.cwd(),
    });

    child.on('error', (error) => {
      console.error(chalk.red(`❌ Error: ${error.message}`));
      reject(error);
    });

    child.on('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Command failed with exit code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

// Show help if no command provided
if (process.argv.length === 2) {
  program.outputHelp();
}

// Parse command-line arguments
program.parse(process.argv);
