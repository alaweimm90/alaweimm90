/**
 * Execute GovernanceDashboard to collect health metrics
 */

import { createGovernanceDashboard } from '../automation/agents/governance/index.js';
import * as fs from 'fs';
import * as path from 'path';

async function runDashboard() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║              GOVERNANCE DASHBOARD - HEALTH METRICS           ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  const dashboard = createGovernanceDashboard({
    workspacePath: process.cwd(),
    historyLength: 10,
  });

  console.log('📊 Collecting health metrics...\n');

  try {
    const metrics = await dashboard.collectMetrics();

    console.log('┌─────────────────────────────────────────────────────────────┐');
    console.log('│                    HEALTH SCORE SUMMARY                     │');
    console.log('├─────────────────────────────────────────────────────────────┤');
    console.log(
      `│  Overall Score: ${metrics.overallScore.toFixed(0).padStart(3)}/100                                     │`
    );
    console.log(
      `│  Status: ${getStatusIcon(metrics.overallScore)} ${getStatusText(metrics.overallScore).padEnd(42)}│`
    );
    console.log('└─────────────────────────────────────────────────────────────┘\n');

    console.log('📋 Category Breakdown:\n');

    const categories = [
      { name: 'TypeScript', data: metrics.categories.typescript, icon: '🔷' },
      { name: 'ESLint', data: metrics.categories.eslint, icon: '📏' },
      { name: 'Tests', data: metrics.categories.tests, icon: '🧪' },
      { name: 'Security', data: metrics.categories.security, icon: '🔒' },
      { name: 'Structure', data: metrics.categories.structure, icon: '📁' },
    ];

    for (const cat of categories) {
      const bar = getProgressBar(cat.data.score);
      console.log(
        `${cat.icon} ${cat.name.padEnd(12)} ${bar} ${cat.data.score.toFixed(0).padStart(3)}/100`
      );
      console.log(`   └─ Status: ${cat.data.status} | ${cat.data.details}`);
      console.log('');
    }

    // Generate and display report
    console.log('┌─────────────────────────────────────────────────────────────┐');
    console.log('│                    DETAILED REPORT                          │');
    console.log('└─────────────────────────────────────────────────────────────┘\n');

    const report = dashboard.generateReport();
    console.log(report);

    // Save metrics
    const reportDir = path.join(process.cwd(), '.archive/reports/governance');
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const metricsPath = path.join(
      reportDir,
      `health-metrics-${new Date().toISOString().split('T')[0]}.json`
    );
    fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
    console.log(`\n💾 Metrics saved to: ${metricsPath}`);
  } catch (error) {
    console.error('❌ Error collecting metrics:', error);
  }
}

function getProgressBar(score: number): string {
  const filled = Math.round(score / 5);
  const empty = 20 - filled;
  return `[${'█'.repeat(filled)}${'░'.repeat(empty)}]`;
}

function getStatusIcon(score: number): string {
  if (score >= 80) return '✅';
  if (score >= 60) return '⚠️';
  if (score >= 40) return '🔶';
  return '❌';
}

function getStatusText(score: number): string {
  if (score >= 80) return 'Healthy';
  if (score >= 60) return 'Needs Attention';
  if (score >= 40) return 'Degraded';
  return 'Critical';
}

runDashboard().catch(console.error);
