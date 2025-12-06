/**
 * VS Code Extension Integration for AI Tools
 * Provides interfaces and utilities for VS Code extension integration
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types for VS Code Integration
// ============================================================================

export interface VSCodeCommand {
  command: string;
  title: string;
  category: string;
  handler: () => Promise<CommandResult>;
}

export interface CommandResult {
  success: boolean;
  message: string;
  data?: unknown;
}

export interface StatusBarItem {
  id: string;
  text: string;
  tooltip: string;
  priority: number;
  command?: string;
}

export interface TreeViewItem {
  id: string;
  label: string;
  description?: string;
  icon?: string;
  contextValue: string;
  children?: TreeViewItem[];
}

export interface DiagnosticItem {
  file: string;
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info' | 'hint';
  source: string;
  code?: string;
}

export interface WebviewMessage {
  type: string;
  payload: unknown;
}

// ============================================================================
// Configuration
// ============================================================================

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

// ============================================================================
// Command Handlers
// ============================================================================

function runNpmCommand(cmd: string): string {
  try {
    return execSync(`npm run ${cmd}`, {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (error) {
    throw new Error(error instanceof Error ? error.message : String(error));
  }
}

function readJsonFile<T>(filePath: string): T | null {
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
  }
  return null;
}

// ============================================================================
// VS Code Commands
// ============================================================================

export const commands: VSCodeCommand[] = [
  {
    command: 'aiTools.showDashboard',
    title: 'Show AI Dashboard',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      try {
        const output = runNpmCommand('ai:dashboard');
        return { success: true, message: 'Dashboard loaded', data: output };
      } catch (error) {
        return {
          success: false,
          message: error instanceof Error ? error.message : 'Failed to load dashboard',
        };
      }
    },
  },
  {
    command: 'aiTools.runComplianceCheck',
    title: 'Run Compliance Check',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      try {
        runNpmCommand('ai:compliance:check');
        const report = readJsonFile(path.join(AI_DIR, 'compliance-report.json'));
        return { success: true, message: 'Compliance check complete', data: report };
      } catch (error) {
        return {
          success: false,
          message: error instanceof Error ? error.message : 'Compliance check failed',
        };
      }
    },
  },
  {
    command: 'aiTools.runSecurityScan',
    title: 'Run Security Scan',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      try {
        runNpmCommand('ai:security:scan');
        const report = readJsonFile(path.join(AI_DIR, 'security-report.json'));
        return { success: true, message: 'Security scan complete', data: report };
      } catch (error) {
        return {
          success: false,
          message: error instanceof Error ? error.message : 'Security scan failed',
        };
      }
    },
  },
  {
    command: 'aiTools.syncContext',
    title: 'Sync AI Context',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      try {
        runNpmCommand('ai:sync');
        return { success: true, message: 'Context synchronized' };
      } catch (error) {
        return { success: false, message: error instanceof Error ? error.message : 'Sync failed' };
      }
    },
  },
  {
    command: 'aiTools.showMetrics',
    title: 'Show AI Metrics',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      const metrics = readJsonFile(path.join(AI_DIR, 'metrics.json'));
      if (metrics) {
        return { success: true, message: 'Metrics loaded', data: metrics };
      }
      return { success: false, message: 'No metrics found' };
    },
  },
  {
    command: 'aiTools.showErrors',
    title: 'Show AI Errors',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      const errorLog = readJsonFile<{ errors: unknown[] }>(path.join(AI_DIR, 'error-log.json'));
      if (errorLog) {
        return { success: true, message: 'Errors loaded', data: errorLog.errors };
      }
      return { success: true, message: 'No errors', data: [] };
    },
  },
  {
    command: 'aiTools.showIssues',
    title: 'Show AI Issues',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      const issues = readJsonFile<{ issues: unknown[] }>(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        return { success: true, message: 'Issues loaded', data: issues.issues };
      }
      return { success: true, message: 'No issues', data: [] };
    },
  },
  {
    command: 'aiTools.clearCache',
    title: 'Clear AI Cache',
    category: 'AI Tools',
    handler: async (): Promise<CommandResult> => {
      try {
        runNpmCommand('ai:cache:clear');
        return { success: true, message: 'Cache cleared' };
      } catch (error) {
        return {
          success: false,
          message: error instanceof Error ? error.message : 'Failed to clear cache',
        };
      }
    },
  },
];

// ============================================================================
// Status Bar Items
// ============================================================================

export function getStatusBarItems(): StatusBarItem[] {
  const items: StatusBarItem[] = [];

  // Compliance score
  const complianceReport = readJsonFile<{ grade: string; overallScore: number }>(
    path.join(AI_DIR, 'compliance-report.json')
  );
  if (complianceReport) {
    items.push({
      id: 'aiTools.compliance',
      text: `$(shield) ${complianceReport.grade} (${complianceReport.overallScore}%)`,
      tooltip: 'AI Compliance Score - Click to run check',
      priority: 100,
      command: 'aiTools.runComplianceCheck',
    });
  }

  // Error count
  const errorLog = readJsonFile<{ stats: { unresolvedCount: number } }>(
    path.join(AI_DIR, 'error-log.json')
  );
  if (errorLog && errorLog.stats.unresolvedCount > 0) {
    items.push({
      id: 'aiTools.errors',
      text: `$(error) ${errorLog.stats.unresolvedCount}`,
      tooltip: `${errorLog.stats.unresolvedCount} unresolved errors`,
      priority: 99,
      command: 'aiTools.showErrors',
    });
  }

  // Issues count
  const issues = readJsonFile<{ stats: { open: number } }>(path.join(AI_DIR, 'issues.json'));
  if (issues && issues.stats?.open > 0) {
    items.push({
      id: 'aiTools.issues',
      text: `$(issues) ${issues.stats.open}`,
      tooltip: `${issues.stats.open} open issues`,
      priority: 98,
      command: 'aiTools.showIssues',
    });
  }

  return items;
}

// ============================================================================
// Tree View Data
// ============================================================================

export function getTreeViewData(): TreeViewItem[] {
  const items: TreeViewItem[] = [];

  // Compliance section
  const complianceReport = readJsonFile<{
    grade: string;
    overallScore: number;
    violations: Array<{ ruleId: string; ruleName: string; message: string }>;
  }>(path.join(AI_DIR, 'compliance-report.json'));

  if (complianceReport) {
    const complianceItem: TreeViewItem = {
      id: 'compliance',
      label: 'Compliance',
      description: `Grade: ${complianceReport.grade}`,
      icon: 'shield',
      contextValue: 'complianceRoot',
      children: complianceReport.violations.map((v, i) => ({
        id: `violation-${i}`,
        label: v.ruleId,
        description: v.message,
        icon: 'warning',
        contextValue: 'violation',
      })),
    };
    items.push(complianceItem);
  }

  // Security section
  const securityReport = readJsonFile<{
    riskLevel: string;
    findings: Array<{ type: string; description: string; severity: string }>;
  }>(path.join(AI_DIR, 'security-report.json'));

  if (securityReport) {
    const securityItem: TreeViewItem = {
      id: 'security',
      label: 'Security',
      description: `Risk: ${securityReport.riskLevel}`,
      icon: 'lock',
      contextValue: 'securityRoot',
      children: securityReport.findings.slice(0, 10).map((f, i) => ({
        id: `finding-${i}`,
        label: f.type,
        description: f.description,
        icon: f.severity === 'critical' ? 'error' : 'warning',
        contextValue: 'finding',
      })),
    };
    items.push(securityItem);
  }

  // Issues section
  const issuesData = readJsonFile<{
    issues: Array<{ id: string; title: string; priority: string; status: string }>;
  }>(path.join(AI_DIR, 'issues.json'));

  if (issuesData) {
    const openIssues = issuesData.issues.filter((i) => i.status === 'open');
    const issuesItem: TreeViewItem = {
      id: 'issues',
      label: 'Issues',
      description: `${openIssues.length} open`,
      icon: 'issues',
      contextValue: 'issuesRoot',
      children: openIssues.slice(0, 10).map((issue) => ({
        id: issue.id,
        label: issue.title,
        description: issue.priority,
        icon: issue.priority === 'critical' ? 'error' : 'issue-opened',
        contextValue: 'issue',
      })),
    };
    items.push(issuesItem);
  }

  // Errors section
  const errorLog = readJsonFile<{
    errors: Array<{
      id: string;
      code: string;
      message: string;
      severity: string;
      resolved: boolean;
    }>;
  }>(path.join(AI_DIR, 'error-log.json'));

  if (errorLog) {
    const unresolvedErrors = errorLog.errors.filter((e) => !e.resolved);
    if (unresolvedErrors.length > 0) {
      const errorsItem: TreeViewItem = {
        id: 'errors',
        label: 'Errors',
        description: `${unresolvedErrors.length} unresolved`,
        icon: 'error',
        contextValue: 'errorsRoot',
        children: unresolvedErrors.slice(0, 10).map((err) => ({
          id: err.id,
          label: err.code,
          description: err.message,
          icon: err.severity === 'critical' ? 'error' : 'warning',
          contextValue: 'error',
        })),
      };
      items.push(errorsItem);
    }
  }

  return items;
}

// ============================================================================
// Diagnostics
// ============================================================================

export function getDiagnostics(): DiagnosticItem[] {
  const diagnostics: DiagnosticItem[] = [];

  // From compliance report
  const complianceReport = readJsonFile<{
    violations: Array<{
      ruleId: string;
      message: string;
      severity: string;
      details?: string[];
    }>;
  }>(path.join(AI_DIR, 'compliance-report.json'));

  if (complianceReport) {
    for (const violation of complianceReport.violations) {
      diagnostics.push({
        file: '',
        line: 1,
        column: 1,
        message: violation.message,
        severity: violation.severity === 'critical' ? 'error' : 'warning',
        source: 'AI Compliance',
        code: violation.ruleId,
      });
    }
  }

  // From security report
  const securityReport = readJsonFile<{
    findings: Array<{
      type: string;
      description: string;
      severity: string;
      file?: string;
      line?: number;
    }>;
  }>(path.join(AI_DIR, 'security-report.json'));

  if (securityReport) {
    for (const finding of securityReport.findings) {
      diagnostics.push({
        file: finding.file || '',
        line: finding.line || 1,
        column: 1,
        message: finding.description,
        severity:
          finding.severity === 'critical'
            ? 'error'
            : finding.severity === 'high'
              ? 'warning'
              : 'info',
        source: 'AI Security',
        code: finding.type,
      });
    }
  }

  return diagnostics;
}

// ============================================================================
// Webview Messages
// ============================================================================

export async function handleWebviewMessage(message: WebviewMessage): Promise<WebviewMessage> {
  switch (message.type) {
    case 'getCompliance':
      return {
        type: 'complianceData',
        payload: readJsonFile(path.join(AI_DIR, 'compliance-report.json')),
      };

    case 'getSecurity':
      return {
        type: 'securityData',
        payload: readJsonFile(path.join(AI_DIR, 'security-report.json')),
      };

    case 'getMetrics':
      return {
        type: 'metricsData',
        payload: readJsonFile(path.join(AI_DIR, 'metrics.json')),
      };

    case 'getErrors':
      return {
        type: 'errorsData',
        payload: readJsonFile(path.join(AI_DIR, 'error-log.json')),
      };

    case 'getIssues':
      return {
        type: 'issuesData',
        payload: readJsonFile(path.join(AI_DIR, 'issues.json')),
      };

    case 'runCommand': {
      const cmd = commands.find((c) => c.command === message.payload);
      if (cmd) {
        const result = await cmd.handler();
        return { type: 'commandResult', payload: result };
      }
      return { type: 'error', payload: 'Command not found' };
    }

    default:
      return { type: 'error', payload: 'Unknown message type' };
  }
}

// ============================================================================
// Extension Activation Helper
// ============================================================================

export interface ExtensionContext {
  commands: VSCodeCommand[];
  statusBarItems: StatusBarItem[];
  treeViewData: TreeViewItem[];
  diagnostics: DiagnosticItem[];
}

export function getExtensionContext(): ExtensionContext {
  return {
    commands,
    statusBarItems: getStatusBarItems(),
    treeViewData: getTreeViewData(),
    diagnostics: getDiagnostics(),
  };
}

// ============================================================================
// CLI
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'commands':
      console.log('\n📋 Available VS Code Commands:\n');
      for (const cmd of commands) {
        console.log(`  ${cmd.command}`);
        console.log(`    ${cmd.title} (${cmd.category})\n`);
      }
      break;

    case 'status':
      console.log('\n📊 Status Bar Items:\n');
      for (const item of getStatusBarItems()) {
        console.log(`  ${item.text}`);
        console.log(`    ${item.tooltip}\n`);
      }
      break;

    case 'tree':
      {
        console.log('\n🌳 Tree View Data:\n');
        const tree = getTreeViewData();
        for (const item of tree) {
          console.log(`  ${item.label} - ${item.description}`);
          if (item.children) {
            for (const child of item.children.slice(0, 3)) {
              console.log(`    └─ ${child.label}: ${child.description}`);
            }
            if (item.children.length > 3) {
              console.log(`    └─ ... and ${item.children.length - 3} more`);
            }
          }
          console.log();
        }
      }
      break;

    case 'diagnostics':
      console.log('\n🔍 Diagnostics:\n');
      for (const diag of getDiagnostics().slice(0, 10)) {
        const icon = diag.severity === 'error' ? '❌' : diag.severity === 'warning' ? '⚠️' : 'ℹ️';
        console.log(`  ${icon} [${diag.source}] ${diag.message}`);
      }
      break;

    default:
      console.log(`
VS Code Integration for AI Tools

Commands:
  commands      List available VS Code commands
  status        Show status bar items
  tree          Show tree view data
  diagnostics   Show diagnostic items

This module provides interfaces for VS Code extension integration.
Import and use in your VS Code extension:

  import { commands, getStatusBarItems, getTreeViewData } from './integration';
      `);
  }
}

main();
