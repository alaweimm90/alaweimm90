#!/usr/bin/env npx tsx
/**
 * AI Tools - Unified Entry Point
 * Enterprise-grade AI orchestration, monitoring, and compliance
 */

// Re-export all modules
export { cache } from './cache.ts';
export { compliance } from './compliance.ts';
export { errorHandler, ErrorCodes, AIOperationError } from './errors.ts';
export { issueManager } from './issues.ts';
export { monitor, circuitBreaker } from './monitor.ts';
export { telemetry } from './telemetry.ts';

// ============================================================================
// Quick Reference
// ============================================================================

const TOOLS = {
  orchestrator: {
    description: 'Task management and context injection',
    commands: ['ai:start', 'ai:complete', 'ai:context', 'ai:metrics', 'ai:history'],
  },
  sync: {
    description: 'Context synchronization',
    commands: ['ai:sync'],
  },
  dashboard: {
    description: 'ASCII metrics dashboard',
    commands: ['ai:dashboard'],
  },
  cache: {
    description: 'Multi-layer caching with semantic similarity',
    commands: ['ai:cache', 'ai:cache:stats', 'ai:cache:clear'],
  },
  monitor: {
    description: 'Continuous monitoring with circuit breakers',
    commands: ['ai:monitor', 'ai:monitor:status', 'ai:monitor:check'],
  },
  compliance: {
    description: 'Policy-based validation and scoring',
    commands: ['ai:compliance', 'ai:compliance:check', 'ai:compliance:score'],
  },
  telemetry: {
    description: 'Observability and alerting',
    commands: ['ai:telemetry', 'ai:telemetry:status', 'ai:telemetry:alerts'],
  },
  errors: {
    description: 'Structured error handling with recovery',
    commands: ['ai:errors', 'ai:errors:list', 'ai:errors:stats'],
  },
  security: {
    description: 'Security scanning (secrets, vulns, licenses)',
    commands: ['ai:security', 'ai:security:scan', 'ai:security:secrets', 'ai:security:vulns'],
  },
  issues: {
    description: 'Automated issue management',
    commands: ['ai:issues', 'ai:issues:list', 'ai:issues:critical', 'ai:issues:stats'],
  },
};

// ============================================================================
// CLI
// ============================================================================

function main(): void {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║            🤖 AI TOOLS - UNIFIED INTERFACE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  A comprehensive suite for AI-assisted development with:     ║
║  • Task orchestration and context management                 ║
║  • Multi-layer intelligent caching                           ║
║  • Continuous monitoring with circuit breakers               ║
║  • Compliance scoring and policy enforcement                 ║
║  • Security scanning and vulnerability detection             ║
║  • Automated issue tracking and remediation                  ║
║  • Full observability and telemetry                          ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  AVAILABLE TOOLS                                             ║
╠══════════════════════════════════════════════════════════════╣`);

  for (const [name, info] of Object.entries(TOOLS)) {
    console.log(`║                                                              ║`);
    console.log(`║  📦 ${name.toUpperCase().padEnd(56)}║`);
    console.log(`║     ${info.description.padEnd(55)}║`);
    console.log(`║     Commands: ${info.commands.slice(0, 3).join(', ')}`.padEnd(65) + '║');
  }

  console.log(`║                                                              ║`);
  console.log(`╠══════════════════════════════════════════════════════════════╣`);
  console.log(`║  QUICK START                                                 ║`);
  console.log(`║  ─────────────────────────────────────────────────────────── ║`);
  console.log(`║  npm run ai:dashboard      View metrics dashboard            ║`);
  console.log(`║  npm run ai:compliance:score   Check compliance score        ║`);
  console.log(`║  npm run ai:security:scan  Run security scan                 ║`);
  console.log(`║  npm run ai:sync           Sync context from git             ║`);
  console.log(`║                                                              ║`);
  console.log(`╚══════════════════════════════════════════════════════════════╝
`);
}

main();
