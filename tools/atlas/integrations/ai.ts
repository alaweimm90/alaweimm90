/**
 * ATLAS CLI - AI Tools Integration
 * Integrates AI orchestration, compliance, and security tools with ATLAS CLI
 */

import { Command } from 'commander';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

// ============================================================================
// Utilities
// ============================================================================

function runAiCommand(cmd: string): string {
  try {
    return execSync(`npm run ${cmd}`, {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (error) {
    if (error instanceof Error) {
      const execError = error as { stdout?: string; stderr?: string };
      return execError.stdout || execError.stderr || error.message;
    }
    return String(error);
  }
}

function readJsonFile<T>(filePath: string): T | null {
  if (fs.existsSync(filePath)) {
    try {
      return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
    } catch {
      return null;
    }
  }
  return null;
}

function formatTable(headers: string[], rows: string[][]): string {
  const colWidths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => (r[i] || '').length))
  );

  const headerRow = headers.map((h, i) => h.padEnd(colWidths[i])).join(' | ');
  const separator = colWidths.map((w) => '-'.repeat(w)).join('-+-');
  const dataRows = rows.map((row) =>
    row.map((cell, i) => (cell || '').padEnd(colWidths[i])).join(' | ')
  );

  return [headerRow, separator, ...dataRows].join('\n');
}

// ============================================================================
// Command Handlers
// ============================================================================

async function handleDashboard(): Promise<void> {
  console.log(runAiCommand('ai:dashboard'));
}

async function handleCompliance(options: { check?: boolean; files?: string }): Promise<void> {
  if (options.check) {
    const files = options.files ? options.files.split(',') : [];
    console.log(runAiCommand(`ai:compliance:check ${files.join(' ')}`));
  } else {
    // Show quick score
    const report = readJsonFile<{
      overallScore: number;
      grade: string;
      summary: { passed: number; failed: number; warnings: number; critical: number };
    }>(path.join(AI_DIR, 'compliance-report.json'));

    if (report) {
      console.log('\n📋 Compliance Status\n');
      console.log(`  Score: ${report.overallScore}/100 (Grade: ${report.grade})`);
      console.log(`  Passed: ${report.summary.passed} | Failed: ${report.summary.failed}`);
      console.log(`  Warnings: ${report.summary.warnings} | Critical: ${report.summary.critical}\n`);
    } else {
      console.log('No compliance report found. Run: atlas ai compliance --check');
    }
  }
}

async function handleSecurity(options: { scan?: boolean; type?: string }): Promise<void> {
  if (options.scan) {
    const scanType = options.type || 'scan';
    console.log(runAiCommand(`ai:security:${scanType}`));
  } else {
    const report = readJsonFile<{
      riskLevel: string;
      summary: { secrets: number; sensitiveFiles: number; vulnerabilities: number; licenseIssues: number };
    }>(path.join(AI_DIR, 'security-report.json'));

    if (report) {
      console.log('\n🔒 Security Status\n');
      console.log(`  Risk Level: ${report.riskLevel.toUpperCase()}`);
      console.log(`  Secrets: ${report.summary.secrets}`);
      console.log(`  Sensitive Files: ${report.summary.sensitiveFiles}`);
      console.log(`  Vulnerabilities: ${report.summary.vulnerabilities}`);
      console.log(`  License Issues: ${report.summary.licenseIssues}\n`);
    } else {
      console.log('No security report found. Run: atlas ai security --scan');
    }
  }
}

async function handleErrors(options: { severity?: string; stats?: boolean }): Promise<void> {
  if (options.stats) {
    console.log(runAiCommand('ai:errors:stats'));
  } else {
    const errorLog = readJsonFile<{
      errors: Array<{ id: string; code: string; message: string; severity: string; resolved: boolean }>;
    }>(path.join(AI_DIR, 'error-log.json'));

    if (errorLog && errorLog.errors.length > 0) {
      let errors = errorLog.errors.filter((e) => !e.resolved);
      if (options.severity) {
        errors = errors.filter((e) => e.severity === options.severity);
      }

      console.log('\n❌ Unresolved Errors\n');
      const rows = errors.slice(0, 10).map((e) => [e.code, e.severity, e.message.substring(0, 50)]);
      console.log(formatTable(['Code', 'Severity', 'Message'], rows));
      if (errors.length > 10) {
        console.log(`\n  ... and ${errors.length - 10} more\n`);
      }
    } else {
      console.log('\n✅ No unresolved errors\n');
    }
  }
}

async function handleIssues(options: { category?: string; priority?: string; stats?: boolean }): Promise<void> {
  if (options.stats) {
    console.log(runAiCommand('ai:issues:stats'));
  } else {
    const issuesData = readJsonFile<{
      issues: Array<{ id: string; title: string; category: string; priority: string; status: string }>;
    }>(path.join(AI_DIR, 'issues.json'));

    if (issuesData && issuesData.issues.length > 0) {
      let issues = issuesData.issues.filter((i) => i.status === 'open');
      if (options.category) {
        issues = issues.filter((i) => i.category === options.category);
      }
      if (options.priority) {
        issues = issues.filter((i) => i.priority === options.priority);
      }

      console.log('\n📌 Open Issues\n');
      const rows = issues.slice(0, 10).map((i) => [i.priority, i.category, i.title.substring(0, 40)]);
      console.log(formatTable(['Priority', 'Category', 'Title'], rows));
      if (issues.length > 10) {
        console.log(`\n  ... and ${issues.length - 10} more\n`);
      }
    } else {
      console.log('\n✅ No open issues\n');
    }
  }
}

async function handleCache(options: { stats?: boolean; clear?: boolean; layer?: string }): Promise<void> {
  if (options.clear) {
    const layer = options.layer || '';
    console.log(runAiCommand(`ai:cache:clear ${layer}`));
  } else if (options.stats) {
    console.log(runAiCommand('ai:cache:stats'));
  } else {
    console.log(runAiCommand('ai:cache'));
  }
}

async function handleMonitor(options: { status?: boolean; check?: boolean }): Promise<void> {
  if (options.check) {
    console.log(runAiCommand('ai:monitor:check'));
  } else {
    console.log(runAiCommand('ai:monitor:status'));
  }
}

async function handleSync(): Promise<void> {
  console.log(runAiCommand('ai:sync'));
}

async function handleMetrics(): Promise<void> {
  const metrics = readJsonFile<{
    effectiveness: { totalTasks: number; successRate: number };
    efficiency: { avgDuration: number };
  }>(path.join(AI_DIR, 'metrics.json'));

  if (metrics) {
    console.log('\n📊 AI Metrics\n');
    console.log(`  Total Tasks: ${metrics.effectiveness?.totalTasks || 0}`);
    console.log(`  Success Rate: ${(metrics.effectiveness?.successRate || 0) * 100}%`);
    console.log(`  Avg Duration: ${metrics.efficiency?.avgDuration || 0}ms\n`);
  } else {
    console.log('No metrics found. Complete some tasks first.');
  }
}

async function handleStart(type: string, description: string, options: { scope?: string }): Promise<void> {
  const scope = options.scope || '';
  console.log(runAiCommand(`ai:start ${type} ${scope} "${description}"`));
}

async function handleComplete(success: string, options: { files?: string; notes?: string }): Promise<void> {
  const files = options.files || '';
  const notes = options.notes || '';
  console.log(runAiCommand(`ai:complete ${success} "${files}" 0 0 0 "${notes}"`));
}

async function handleServe(options: { mcp?: boolean; api?: boolean; port?: string }): Promise<void> {
  if (options.mcp) {
    const port = options.port || '3100';
    process.env.MCP_PORT = port;
    console.log(`Starting MCP server on port ${port}...`);
    execSync('npm run ai:mcp:start', { cwd: ROOT, stdio: 'inherit' });
  } else if (options.api) {
    const port = options.port || '3200';
    process.env.AI_API_PORT = port;
    console.log(`Starting REST API server on port ${port}...`);
    execSync('npm run ai:api:start', { cwd: ROOT, stdio: 'inherit' });
  } else {
    console.log('Specify --mcp or --api to start a server');
  }
}

// ============================================================================
// Command Registration
// ============================================================================

export function registerAiCommands(program: Command): void {
  const ai = program
    .command('ai')
    .description('AI orchestration, compliance, and security tools');

  // Dashboard
  ai.command('dashboard')
    .description('Show AI metrics dashboard')
    .action(handleDashboard);

  // Compliance
  ai.command('compliance')
    .description('Check compliance status')
    .option('-c, --check', 'Run full compliance check')
    .option('-f, --files <files>', 'Comma-separated files to check')
    .action(handleCompliance);

  // Security
  ai.command('security')
    .description('Check security status')
    .option('-s, --scan', 'Run security scan')
    .option('-t, --type <type>', 'Scan type (scan, secrets, vulns)', 'scan')
    .action(handleSecurity);

  // Errors
  ai.command('errors')
    .description('View AI errors')
    .option('-s, --severity <level>', 'Filter by severity (low, medium, high, critical)')
    .option('--stats', 'Show error statistics')
    .action(handleErrors);

  // Issues
  ai.command('issues')
    .description('View tracked issues')
    .option('-c, --category <cat>', 'Filter by category')
    .option('-p, --priority <pri>', 'Filter by priority')
    .option('--stats', 'Show issue statistics')
    .action(handleIssues);

  // Cache
  ai.command('cache')
    .description('Manage AI cache')
    .option('--stats', 'Show cache statistics')
    .option('--clear', 'Clear cache')
    .option('-l, --layer <layer>', 'Cache layer (semantic, template, result, analysis)')
    .action(handleCache);

  // Monitor
  ai.command('monitor')
    .description('Monitor status and circuit breakers')
    .option('--status', 'Show monitor status', true)
    .option('--check', 'Check for changes')
    .action(handleMonitor);

  // Sync
  ai.command('sync')
    .description('Synchronize AI context')
    .action(handleSync);

  // Metrics
  ai.command('metrics')
    .description('View AI effectiveness metrics')
    .action(handleMetrics);

  // Task management
  ai.command('start <type> <description>')
    .description('Start tracking a task')
    .option('-s, --scope <scope>', 'Task scope areas')
    .action(handleStart);

  ai.command('complete <success>')
    .description('Complete current task')
    .option('-f, --files <files>', 'Changed files')
    .option('-n, --notes <notes>', 'Completion notes')
    .action(handleComplete);

  // Server management
  ai.command('serve')
    .description('Start AI tools server')
    .option('--mcp', 'Start MCP server')
    .option('--api', 'Start REST API server')
    .option('-p, --port <port>', 'Server port')
    .action(handleServe);
}
