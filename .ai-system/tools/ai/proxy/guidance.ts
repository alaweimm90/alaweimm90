#!/usr/bin/env tsx
/**
 * Behavioral Guidance System
 * Pre-task routing and post-session estimation for non-interceptable AI tools
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { selectTier, estimateTokens, logUsage } from './core.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..', '..', '..');
const SESSION_FILE = join(ROOT_DIR, '.config', 'ai', 'logs', 'session.json');

interface Session {
  started_at: string;
  tool: string;
  task_description: string;
  recommended_tier: string;
  actual_tier?: string;
  estimated_tokens?: number;
  notes?: string;
}

// Tool recommendations based on tier
const TOOL_RECOMMENDATIONS: Record<string, string[]> = {
  lightweight: ['Copilot autocomplete', 'Blackbox quick fix', 'Windsurf inline'],
  standard: ['Kilo Code chat', 'Cursor Composer', 'Continue.dev (via proxy)'],
  heavyweight: ['Claude Code (Augment)', 'Cursor Agent', 'Full IDE chat'],
};

// Start a new session
function startSession(tool: string, taskDescription: string): void {
  const messages = [{ role: 'user', content: taskDescription }];
  const tier = selectTier(messages);
  const tokens = estimateTokens(taskDescription);

  const session: Session = {
    started_at: new Date().toISOString(),
    tool,
    task_description: taskDescription,
    recommended_tier: tier,
    estimated_tokens: tokens,
  };

  const logDir = dirname(SESSION_FILE);
  if (!existsSync(logDir)) mkdirSync(logDir, { recursive: true });
  writeFileSync(SESSION_FILE, JSON.stringify(session, null, 2));

  console.log(chalk.cyan('\n╔════════════════════════════════════════════════════════════╗'));
  console.log(chalk.cyan('║          🎯 PRE-TASK GUIDANCE                              ║'));
  console.log(chalk.cyan('╚════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.yellow('Task:'), taskDescription);
  console.log(chalk.yellow('Tool:'), tool);
  console.log(chalk.yellow('Estimated Tokens:'), tokens);

  const tierColor =
    tier === 'lightweight' ? chalk.green : tier === 'standard' ? chalk.yellow : chalk.red;
  console.log(chalk.yellow('Recommended Tier:'), tierColor(tier.toUpperCase()));

  console.log(chalk.cyan('\n📋 Recommended Tools for this tier:'));
  TOOL_RECOMMENDATIONS[tier]?.forEach((t) => console.log(`   • ${t}`));

  if (tier !== 'heavyweight' && tool.toLowerCase().includes('claude')) {
    console.log(chalk.yellow('\n⚠️  Consider using a lighter tool for this task to save tokens!'));
  }

  console.log(chalk.gray('\n💡 When done, run: npm run ai:guide end <actual_tokens>\n'));
}

// End session and log usage
function endSession(actualTokens: number, _notes?: string): void {
  if (!existsSync(SESSION_FILE)) {
    console.log(
      chalk.red('No active session found. Start one with: npm run ai:guide start <tool> "<task>"')
    );
    return;
  }

  const session: Session = JSON.parse(readFileSync(SESSION_FILE, 'utf-8'));
  const duration = (Date.now() - new Date(session.started_at).getTime()) / 1000 / 60;

  // Estimate cost based on tier
  const costPerK =
    session.recommended_tier === 'lightweight'
      ? 0.0005
      : session.recommended_tier === 'standard'
        ? 0.01
        : 0.03;
  const estimatedCost = (actualTokens / 1000) * costPerK;

  // Log the usage
  logUsage(
    session.recommended_tier,
    session.tool,
    Math.floor(actualTokens * 0.7),
    Math.floor(actualTokens * 0.3),
    estimatedCost
  );

  console.log(chalk.cyan('\n╔════════════════════════════════════════════════════════════╗'));
  console.log(chalk.cyan('║          📊 SESSION SUMMARY                                ║'));
  console.log(chalk.cyan('╚════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.yellow('Tool Used:'), session.tool);
  console.log(chalk.yellow('Duration:'), `${duration.toFixed(1)} minutes`);
  console.log(chalk.yellow('Recommended Tier:'), session.recommended_tier);
  console.log(chalk.yellow('Actual Tokens:'), actualTokens);
  console.log(chalk.yellow('Estimated Cost:'), `$${estimatedCost.toFixed(4)}`);

  // Check tier alignment
  const estimatedTier =
    actualTokens < 2000 ? 'lightweight' : actualTokens < 12000 ? 'standard' : 'heavyweight';

  if (estimatedTier !== session.recommended_tier) {
    console.log(chalk.yellow('\n📈 Tier Alignment:'), chalk.red('MISMATCH'));
    console.log(
      `   Recommended: ${session.recommended_tier}, Actual usage suggests: ${estimatedTier}`
    );
  } else {
    console.log(chalk.green('\n✓ Tier selection was appropriate for this task'));
  }

  // Remove session file
  writeFileSync(SESSION_FILE, '');
  console.log(chalk.gray('\n✅ Session logged. View stats with: npm run ai:tokens stats\n'));
}

// Quick tier recommendation without session
function quickRecommend(taskDescription: string): void {
  const messages = [{ role: 'user', content: taskDescription }];
  const tier = selectTier(messages);
  const tokens = estimateTokens(taskDescription);

  const tierEmoji = tier === 'lightweight' ? '🪶' : tier === 'standard' ? '⚖️' : '🏋️';
  const tierColor =
    tier === 'lightweight' ? chalk.green : tier === 'standard' ? chalk.yellow : chalk.red;

  console.log(`\n${tierEmoji} ${tierColor(tier.toUpperCase())} (~${tokens} tokens)`);
  console.log(chalk.gray('Suggested tools:'), TOOL_RECOMMENDATIONS[tier]?.join(', '));
}

// CLI
const command = process.argv[2];
const args = process.argv.slice(3);

switch (command) {
  case 'start':
    if (args.length < 2) {
      console.log(chalk.red('Usage: npm run ai:guide start <tool> "<task description>"'));
      process.exit(1);
    }
    startSession(args[0], args.slice(1).join(' '));
    break;

  case 'end':
    if (args.length < 1) {
      console.log(chalk.red('Usage: npm run ai:guide end <actual_tokens> [notes]'));
      process.exit(1);
    }
    endSession(parseInt(args[0], 10), args.slice(1).join(' '));
    break;

  case 'quick':
    if (args.length < 1) {
      console.log(chalk.red('Usage: npm run ai:guide quick "<task description>"'));
      process.exit(1);
    }
    quickRecommend(args.join(' '));
    break;

  default:
    console.log(`
${chalk.cyan('🎯 Behavioral Guidance System')}

For tools that can't be intercepted (Copilot, Windsurf, Blackbox), this provides:
• Pre-task tier recommendations
• Post-session usage logging
• Tool suggestions by tier

Commands:
  start <tool> "<task>"  - Start tracking a session
  end <tokens> [notes]   - End session and log usage
  quick "<task>"         - Quick tier recommendation

Examples:
  npm run ai:guide start copilot "fix typo in README"
  npm run ai:guide end 500 "completed quickly"
  npm run ai:guide quick "architect payment system"
    `);
}
