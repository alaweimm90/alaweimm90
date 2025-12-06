#!/usr/bin/env tsx
/**
 * Token Usage Tracker & Tier Selection CLI
 * Integrates with the model-tiering system for token optimization
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';
import chalk from 'chalk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..', '..');

// Paths
const TOKEN_LOG = join(ROOT_DIR, '.config', 'ai', 'logs', 'token-usage.jsonl');
const METRICS_FILE = join(ROOT_DIR, '.config', 'ai', 'logs', 'token-metrics.json');

interface TokenEntry {
  timestamp: string;
  tier: string;
  task_type: string;
  input_tokens: number;
  output_tokens: number;
  estimated_cost: number;
  model?: string;
  task_id?: string;
}

interface TierStats {
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  avg_tokens_per_request: number;
}

interface TokenMetrics {
  updated_at: string;
  session: { tokens: number; cost: number; requests: number };
  daily: { tokens: number; cost: number; requests: number };
  by_tier: Record<string, TierStats>;
  savings: { estimated_without_tiering: number; actual: number; saved_percentage: number };
}

// Ensure log directory exists
function ensureLogDir(): void {
  const logDir = dirname(TOKEN_LOG);
  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true });
  }
}

// Load current metrics
function loadMetrics(): TokenMetrics {
  if (existsSync(METRICS_FILE)) {
    try {
      return JSON.parse(readFileSync(METRICS_FILE, 'utf-8'));
    } catch {
      // Return default if parse fails
    }
  }
  return {
    updated_at: new Date().toISOString(),
    session: { tokens: 0, cost: 0, requests: 0 },
    daily: { tokens: 0, cost: 0, requests: 0 },
    by_tier: {
      lightweight: { total_requests: 0, total_tokens: 0, total_cost: 0, avg_tokens_per_request: 0 },
      standard: { total_requests: 0, total_tokens: 0, total_cost: 0, avg_tokens_per_request: 0 },
      heavyweight: { total_requests: 0, total_tokens: 0, total_cost: 0, avg_tokens_per_request: 0 },
    },
    savings: { estimated_without_tiering: 0, actual: 0, saved_percentage: 0 },
  };
}

// Save metrics
function saveMetrics(metrics: TokenMetrics): void {
  metrics.updated_at = new Date().toISOString();
  writeFileSync(METRICS_FILE, JSON.stringify(metrics, null, 2));
}

// Record token usage
function recordUsage(entry: TokenEntry): void {
  ensureLogDir();
  const line = JSON.stringify({ ...entry, timestamp: new Date().toISOString() }) + '\n';
  writeFileSync(TOKEN_LOG, line, { flag: 'a' });

  // Update metrics
  const metrics = loadMetrics();
  const totalTokens = entry.input_tokens + entry.output_tokens;

  metrics.session.tokens += totalTokens;
  metrics.session.cost += entry.estimated_cost;
  metrics.session.requests += 1;

  metrics.daily.tokens += totalTokens;
  metrics.daily.cost += entry.estimated_cost;
  metrics.daily.requests += 1;

  const tier = entry.tier.toLowerCase();
  if (metrics.by_tier[tier]) {
    metrics.by_tier[tier].total_requests += 1;
    metrics.by_tier[tier].total_tokens += totalTokens;
    metrics.by_tier[tier].total_cost += entry.estimated_cost;
    metrics.by_tier[tier].avg_tokens_per_request =
      metrics.by_tier[tier].total_tokens / metrics.by_tier[tier].total_requests;
  }

  // Calculate savings (assume heavyweight cost if no tiering)
  const heavyweightCostPer1k = 0.015;
  const estimatedWithoutTiering = (totalTokens / 1000) * heavyweightCostPer1k;
  metrics.savings.estimated_without_tiering += estimatedWithoutTiering;
  metrics.savings.actual += entry.estimated_cost;
  metrics.savings.saved_percentage =
    metrics.savings.estimated_without_tiering > 0
      ? Math.round(
          ((metrics.savings.estimated_without_tiering - metrics.savings.actual) /
            metrics.savings.estimated_without_tiering) *
            100
        )
      : 0;

  saveMetrics(metrics);
}

// Select tier using Python engine
function selectTier(query: string): void {
  try {
    const result = execSync(
      `python .config/ai/prompt-engine/engine.py select "${query.replace(/"/g, '\\"')}"`,
      { cwd: ROOT_DIR, encoding: 'utf-8' }
    );
    console.log(result);
  } catch (error) {
    console.error(chalk.red('Error selecting tier:'), error);
  }
}

// Show current stats
function showStats(): void {
  const metrics = loadMetrics();

  console.log(chalk.cyan('\n╔════════════════════════════════════════════════════════════╗'));
  console.log(chalk.cyan('║           TOKEN USAGE DASHBOARD                            ║'));
  console.log(chalk.cyan('╚════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.yellow('📊 Session Stats:'));
  console.log(`   Tokens: ${metrics.session.tokens.toLocaleString()}`);
  console.log(`   Cost: $${metrics.session.cost.toFixed(4)}`);
  console.log(`   Requests: ${metrics.session.requests}`);

  console.log(chalk.yellow('\n📈 Daily Stats:'));
  console.log(`   Tokens: ${metrics.daily.tokens.toLocaleString()}`);
  console.log(`   Cost: $${metrics.daily.cost.toFixed(4)}`);
  console.log(`   Requests: ${metrics.daily.requests}`);

  console.log(chalk.yellow('\n🎯 By Tier:'));
  for (const [tierName, stats] of Object.entries(metrics.by_tier)) {
    const emoji = tierName === 'lightweight' ? '🪶' : tierName === 'standard' ? '⚖️' : '🏋️';
    console.log(
      `   ${emoji} ${tierName.toUpperCase()}: ${stats.total_requests} requests, ${stats.total_tokens.toLocaleString()} tokens, $${stats.total_cost.toFixed(4)}`
    );
  }

  console.log(chalk.green('\n💰 Savings:'));
  console.log(`   Without tiering: $${metrics.savings.estimated_without_tiering.toFixed(4)}`);
  console.log(`   With tiering: $${metrics.savings.actual.toFixed(4)}`);
  console.log(`   Saved: ${chalk.green(metrics.savings.saved_percentage + '%')}`);
}

// Reset session stats
function resetSession(): void {
  const metrics = loadMetrics();
  metrics.session = { tokens: 0, cost: 0, requests: 0 };
  saveMetrics(metrics);
  console.log(chalk.green('✅ Session stats reset'));
}

// Reset daily stats
function resetDaily(): void {
  const metrics = loadMetrics();
  metrics.daily = { tokens: 0, cost: 0, requests: 0 };
  saveMetrics(metrics);
  console.log(chalk.green('✅ Daily stats reset'));
}

// Export for integration
export { recordUsage, loadMetrics, saveMetrics, selectTier, showStats, resetSession, resetDaily };
export type { TokenEntry, TokenMetrics, TierStats };

// CLI
const command = process.argv[2];
const arg = process.argv.slice(3).join(' ');

switch (command) {
  case 'tier':
  case 'select':
    if (!arg) {
      console.log(chalk.red('Usage: npm run ai:tier "your task description"'));
      process.exit(1);
    }
    selectTier(arg);
    break;

  case 'stats':
  case 'dashboard':
    showStats();
    break;

  case 'record': {
    // Record usage: npm run ai:tokens record <tier> <input> <output> <cost>
    const [tier, input, output, cost] = arg.split(' ');
    if (!tier || !input || !output) {
      console.log(
        chalk.red('Usage: npm run ai:tokens record <tier> <input_tokens> <output_tokens> [cost]')
      );
      process.exit(1);
    }
    recordUsage({
      timestamp: new Date().toISOString(),
      tier,
      task_type: 'manual',
      input_tokens: parseInt(input, 10),
      output_tokens: parseInt(output, 10),
      estimated_cost: cost ? parseFloat(cost) : 0,
    });
    console.log(chalk.green('✅ Usage recorded'));
    break;
  }

  case 'reset':
    if (arg === 'session') {
      resetSession();
    } else if (arg === 'daily') {
      resetDaily();
    } else {
      console.log(chalk.red('Usage: npm run ai:tokens reset <session|daily>'));
    }
    break;

  case 'help':
  default:
    console.log(chalk.cyan('\n🎯 Token Usage CLI\n'));
    console.log('Commands:');
    console.log('  tier <query>     - Select optimal tier for a task');
    console.log('  stats            - Show token usage dashboard');
    console.log('  record           - Record token usage manually');
    console.log('  reset <scope>    - Reset session or daily stats');
    console.log('\nExamples:');
    console.log('  npm run ai:tier "fix typo in README"');
    console.log('  npm run ai:tokens stats');
    console.log('  npm run ai:tokens reset session');
    break;
}
