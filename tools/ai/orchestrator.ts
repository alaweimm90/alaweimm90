#!/usr/bin/env npx tsx
/**
 * AI Orchestrator
 * Manages context injection, task tracking, and feedback collection
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

interface Task {
  id: string;
  timestamp: string;
  type: 'feature' | 'bug_fix' | 'refactoring' | 'documentation';
  scope: string[];
  description: string;
  ai_tool: string;
  outcome: 'success' | 'failure' | 'in_progress';
  files_changed: string[];
  metrics: {
    duration_minutes: number;
    lines_added: number;
    lines_removed: number;
    governance_pass: boolean;
  };
  feedback?: {
    developer_rating: number;
    notes: string;
  };
}

interface Context {
  version: string;
  documentation: Record<string, string>;
  policies: Record<string, string>;
  templates: { root: string; categories: string[] };
  tools: Record<string, string>;
  tracking: Record<string, string>;
  routing: Record<string, string>;
}

// Load context configuration
function loadContext(): Context | null {
  const contextPath = path.join(AI_DIR, 'context.yaml');
  if (!fs.existsSync(contextPath)) {
    console.error('❌ Context file not found:', contextPath);
    return null;
  }

  try {
    const content = fs.readFileSync(contextPath, 'utf8');
    return yaml.parse(content);
  } catch (error) {
    console.error('❌ Failed to parse context.yaml:', error);
    return null;
  }
}

// Load task history
function loadTaskHistory(): { tasks: Task[]; patterns: Record<string, string[]> } {
  const historyPath = path.join(AI_DIR, 'task-history.json');
  if (!fs.existsSync(historyPath)) {
    return { tasks: [], patterns: {} };
  }

  try {
    return JSON.parse(fs.readFileSync(historyPath, 'utf8'));
  } catch {
    return { tasks: [], patterns: {} };
  }
}

// Save task history
function saveTaskHistory(history: { tasks: Task[]; patterns: Record<string, string[]> }): void {
  const historyPath = path.join(AI_DIR, 'task-history.json');
  fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));
}

// Generate task ID
function generateTaskId(): string {
  const history = loadTaskHistory();
  const nextNum = history.tasks.length + 1;
  return `task-${String(nextNum).padStart(3, '0')}`;
}

// Start a new task
function startTask(type: Task['type'], scope: string[], description: string): Task {
  const task: Task = {
    id: generateTaskId(),
    timestamp: new Date().toISOString(),
    type,
    scope,
    description,
    ai_tool: 'claude', // Default, can be overridden
    outcome: 'in_progress',
    files_changed: [],
    metrics: {
      duration_minutes: 0,
      lines_added: 0,
      lines_removed: 0,
      governance_pass: false,
    },
  };

  // Save current task
  const currentTaskPath = path.join(AI_DIR, 'current-task.json');
  fs.writeFileSync(currentTaskPath, JSON.stringify(task, null, 2));

  console.log(`🚀 Task started: ${task.id}`);
  console.log(`   Type: ${type}`);
  console.log(`   Scope: ${scope.join(', ')}`);
  console.log(`   Description: ${description}`);

  return task;
}

// Complete a task
function completeTask(
  success: boolean,
  filesChanged: string[],
  linesAdded: number,
  linesRemoved: number,
  rating?: number,
  notes?: string
): void {
  const currentTaskPath = path.join(AI_DIR, 'current-task.json');
  if (!fs.existsSync(currentTaskPath)) {
    console.error('❌ No current task found');
    return;
  }

  const task: Task = JSON.parse(fs.readFileSync(currentTaskPath, 'utf8'));
  const startTime = new Date(task.timestamp);
  const duration = Math.round((Date.now() - startTime.getTime()) / 60000);

  task.outcome = success ? 'success' : 'failure';
  task.files_changed = filesChanged;
  task.metrics = {
    duration_minutes: duration,
    lines_added: linesAdded,
    lines_removed: linesRemoved,
    governance_pass: success,
  };

  if (rating !== undefined) {
    task.feedback = { developer_rating: rating, notes: notes || '' };
  }

  // Add to history
  const history = loadTaskHistory();
  history.tasks.push(task);
  saveTaskHistory(history);

  // Remove current task file
  fs.unlinkSync(currentTaskPath);

  console.log(`✅ Task completed: ${task.id}`);
  console.log(`   Outcome: ${task.outcome}`);
  console.log(`   Duration: ${duration} minutes`);
  console.log(`   Files changed: ${filesChanged.length}`);

  // Update metrics
  updateMetrics();
}

// Update metrics based on task history
function updateMetrics(): void {
  const history = loadTaskHistory();
  const metricsPath = path.join(AI_DIR, 'metrics.json');

  const successful = history.tasks.filter((t) => t.outcome === 'success').length;
  const failed = history.tasks.filter((t) => t.outcome === 'failure').length;
  const total = successful + failed;

  const avgDuration =
    total > 0
      ? history.tasks
          .filter((t) => t.outcome !== 'in_progress')
          .reduce((sum, t) => sum + t.metrics.duration_minutes, 0) / total
      : 0;

  const totalLinesAdded = history.tasks.reduce((sum, t) => sum + t.metrics.lines_added, 0);
  const totalLinesRemoved = history.tasks.reduce((sum, t) => sum + t.metrics.lines_removed, 0);

  const metrics = {
    version: '1.0',
    period: {
      start: history.tasks[0]?.timestamp.split('T')[0] || new Date().toISOString().split('T')[0],
      end: new Date().toISOString().split('T')[0],
    },
    summary: {
      total_tasks: total,
      successful_tasks: successful,
      failed_tasks: failed,
      success_rate: total > 0 ? Math.round((successful / total) * 100) : 0,
      avg_duration_minutes: Math.round(avgDuration),
      total_lines_added: totalLinesAdded,
      total_lines_removed: totalLinesRemoved,
      governance_compliance: total > 0 ? Math.round((successful / total) * 100) : 0,
    },
    updated_at: new Date().toISOString(),
  };

  fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
  console.log('📊 Metrics updated');
}

// Get context for a specific task type
function getContext(taskType: Task['type'], scope: string[]): string {
  const context = loadContext();
  if (!context) return '';

  const lines: string[] = ['# AI Context for Current Task\n'];

  // Always include codemap
  lines.push('## Codemap');
  lines.push(`See: ${context.documentation.codemap}\n`);

  // Include relevant docs based on task type
  if (taskType === 'feature') {
    lines.push('## Architecture');
    lines.push(`See: ${context.documentation.architecture}\n`);
  }

  // Include policies
  lines.push('## Policies');
  lines.push(`Protected files: ${context.policies.protected_files}`);
  lines.push(`AI instructions: ${context.policies.ai_instructions}\n`);

  // Include relevant templates based on scope
  if (scope.some((s) => context.templates.categories.includes(s))) {
    lines.push('## Relevant Templates');
    lines.push(`Root: ${context.templates.root}`);
    for (const s of scope) {
      if (context.templates.categories.includes(s)) {
        lines.push(`- ${s}/`);
      }
    }
    lines.push('');
  }

  // Include similar past tasks
  const history = loadTaskHistory();
  const similarTasks = history.tasks.filter(
    (t) => t.type === taskType && t.scope.some((s) => scope.includes(s)) && t.outcome === 'success'
  );

  if (similarTasks.length > 0) {
    lines.push('## Similar Past Tasks');
    for (const task of similarTasks.slice(-3)) {
      lines.push(`- ${task.id}: ${task.description} (${task.outcome})`);
    }
    lines.push('');
  }

  // Include patterns
  if (history.patterns?.successful_approaches?.length > 0) {
    lines.push('## Successful Approaches');
    for (const approach of history.patterns.successful_approaches) {
      lines.push(`- ${approach}`);
    }
  }

  return lines.join('\n');
}

// CLI
function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'start': {
      const type = (args[1] as Task['type']) || 'feature';
      const scope = args[2]?.split(',') || ['general'];
      const description = args.slice(3).join(' ') || 'Untitled task';
      startTask(type, scope, description);
      break;
    }

    case 'complete': {
      const success = args[1] !== 'false';
      const filesChanged = args[2]?.split(',') || [];
      const linesAdded = parseInt(args[3]) || 0;
      const linesRemoved = parseInt(args[4]) || 0;
      const rating = args[5] ? parseInt(args[5]) : undefined;
      const notes = args.slice(6).join(' ');
      completeTask(success, filesChanged, linesAdded, linesRemoved, rating, notes);
      break;
    }

    case 'context': {
      const type = (args[1] as Task['type']) || 'feature';
      const scope = args[2]?.split(',') || ['general'];
      console.log(getContext(type, scope));
      break;
    }

    case 'metrics': {
      updateMetrics();
      const metricsPath = path.join(AI_DIR, 'metrics.json');
      console.log(fs.readFileSync(metricsPath, 'utf8'));
      break;
    }

    case 'history': {
      const history = loadTaskHistory();
      console.log(JSON.stringify(history.tasks.slice(-10), null, 2));
      break;
    }

    default:
      console.log(`
AI Orchestrator - Manage AI context and task tracking

Commands:
  start <type> <scope> <description>   Start a new task
  complete <success> <files> <+lines> <-lines> [rating] [notes]   Complete current task
  context <type> <scope>               Get context for a task
  metrics                              Show current metrics
  history                              Show recent task history

Types: feature, bug_fix, refactoring, documentation
Scope: comma-separated list (e.g., auth,api)

Examples:
  npm run ai:start feature auth,api Add OAuth authentication
  npm run ai:complete true "src/auth.ts,src/api.ts" 150 20 5 "Clean implementation"
  npm run ai:context feature auth
      `);
  }
}

main();
