#!/usr/bin/env npx tsx
/**
 * AI Metrics Dashboard
 * Displays AI orchestration effectiveness metrics
 */

import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

interface Metrics {
  version: string;
  period: { start: string; end: string };
  summary: {
    total_tasks: number;
    successful_tasks: number;
    failed_tasks: number;
    success_rate: number;
    avg_duration_minutes: number;
    total_lines_added: number;
    total_lines_removed: number;
    governance_compliance: number;
  };
  updated_at?: string;
}

interface TaskHistory {
  tasks: Array<{
    id: string;
    timestamp: string;
    type: string;
    outcome: string;
    metrics: {
      duration_minutes: number;
      lines_added: number;
      lines_removed: number;
    };
  }>;
}

function loadMetrics(): Metrics | null {
  const metricsPath = path.join(AI_DIR, 'metrics.json');
  if (!fs.existsSync(metricsPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
  } catch {
    return null;
  }
}

function loadTaskHistory(): TaskHistory | null {
  const historyPath = path.join(AI_DIR, 'task-history.json');
  if (!fs.existsSync(historyPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(historyPath, 'utf8'));
  } catch {
    return null;
  }
}

function progressBar(value: number, max: number = 100, width: number = 20): string {
  const filled = Math.round((value / max) * width);
  const empty = width - filled;
  return `[${'█'.repeat(filled)}${'░'.repeat(empty)}] ${value}%`;
}

function displayDashboard(): void {
  const metrics = loadMetrics();
  const history = loadTaskHistory();

  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║           🤖 AI ORCHESTRATION METRICS DASHBOARD              ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');

  if (!metrics) {
    console.log('║  No metrics data available. Run some tasks first!           ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    return;
  }

  const s = metrics.summary;

  // Overview Section
  console.log('║                                                              ║');
  console.log('║  📊 OVERVIEW                                                 ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  Period: ${metrics.period.start} to ${metrics.period.end}`.padEnd(65) + '║');
  console.log(`║  Total Tasks: ${s.total_tasks}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Success Metrics
  console.log('║  📈 SUCCESS METRICS                                          ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  Success Rate:     ${progressBar(s.success_rate)}`.padEnd(65) + '║');
  console.log(`║  Governance:       ${progressBar(s.governance_compliance)}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Task Breakdown
  console.log('║  📋 TASK BREAKDOWN                                           ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  ✅ Successful: ${s.successful_tasks}`.padEnd(65) + '║');
  console.log(`║  ❌ Failed:     ${s.failed_tasks}`.padEnd(65) + '║');
  console.log(`║  ⏱️  Avg Time:   ${s.avg_duration_minutes} minutes`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Code Impact
  console.log('║  💻 CODE IMPACT                                              ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  Lines Added:   +${s.total_lines_added}`.padEnd(65) + '║');
  console.log(`║  Lines Removed: -${s.total_lines_removed}`.padEnd(65) + '║');
  console.log(
    `║  Net Change:    ${s.total_lines_added - s.total_lines_removed > 0 ? '+' : ''}${s.total_lines_added - s.total_lines_removed}`.padEnd(
      65
    ) + '║'
  );
  console.log('║                                                              ║');

  // Recent Tasks
  if (history && history.tasks.length > 0) {
    console.log('║  📝 RECENT TASKS                                             ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');

    const recentTasks = history.tasks.slice(-5).reverse();
    for (const task of recentTasks) {
      const icon = task.outcome === 'success' ? '✅' : task.outcome === 'failure' ? '❌' : '🔄';
      const line = `  ${icon} ${task.id}: ${task.type} (${task.metrics.duration_minutes}m)`;
      console.log(`║${line.padEnd(64)}║`);
    }
    console.log('║                                                              ║');
  }

  // Recommendations
  console.log('║  💡 RECOMMENDATIONS                                          ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');

  if (s.success_rate >= 90) {
    console.log('║  • Excellent success rate! Keep up the good work.            ║');
  } else if (s.success_rate >= 70) {
    console.log('║  • Good progress. Review failed tasks for patterns.          ║');
  } else {
    console.log('║  • Consider reviewing AI context and task complexity.        ║');
  }

  if (s.avg_duration_minutes > 30) {
    console.log('║  • Tasks taking longer than expected. Break into subtasks?   ║');
  }

  if (s.governance_compliance < 100) {
    console.log('║  • Some governance violations. Check protected files policy. ║');
  }

  console.log('║                                                              ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('\n');
}

function displayJson(): void {
  const metrics = loadMetrics();
  console.log(JSON.stringify(metrics, null, 2));
}

// CLI
const format = process.argv[2];

if (format === '--json') {
  displayJson();
} else {
  displayDashboard();
}
