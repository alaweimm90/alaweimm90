#!/usr/bin/env tsx
/**
 * Unified telemetry CLI
 * Usage: npm run telemetry <command>
 */
import { readFileSync, existsSync, mkdirSync, appendFileSync } from 'fs';
import { join } from 'path';

const LOGS_DIR = '.logs';
const METRICS_FILE = join(LOGS_DIR, 'metrics.jsonl');

interface MetricEntry {
  timestamp: string;
  type: string;
  data: Record<string, unknown>;
}

function ensureLogsDir() {
  if (!existsSync(LOGS_DIR)) mkdirSync(LOGS_DIR, { recursive: true });
}

function logMetric(type: string, data: Record<string, unknown>) {
  ensureLogsDir();
  const entry: MetricEntry = {
    timestamp: new Date().toISOString(),
    type,
    data
  };
  appendFileSync(METRICS_FILE, JSON.stringify(entry) + '\n');
  console.log(`📊 Logged: ${type}`);
}

function showDashboard() {
  console.log('\n📊 TELEMETRY DASHBOARD\n');
  console.log('='.repeat(50));

  // Token usage
  const tokenLog = '.config/ai/logs/token-usage.jsonl';
  if (existsSync(tokenLog)) {
    const lines = readFileSync(tokenLog, 'utf-8').trim().split('\n');
    const total = lines.reduce((sum, line) => {
      try { return sum + JSON.parse(line).tokens; } catch { return sum; }
    }, 0);
    console.log(`🤖 AI Tokens Used: ${total.toLocaleString()}`);
  }

  // Build metrics
  if (existsSync(METRICS_FILE)) {
    const lines = readFileSync(METRICS_FILE, 'utf-8').trim().split('\n');
    console.log(`📈 Total Metrics Logged: ${lines.length}`);
  }

  console.log('='.repeat(50));
}

const [,, cmd, ...args] = process.argv;
switch (cmd) {
  case 'log':
    logMetric(args[0] || 'generic', { value: args[1] });
    break;
  case 'dashboard':
  case 'show':
    showDashboard();
    break;
  default:
    console.log('Usage: npm run telemetry <log|dashboard>');
}
