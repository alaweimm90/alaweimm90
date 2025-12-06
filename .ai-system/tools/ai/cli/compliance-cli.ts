#!/usr/bin/env npx tsx
/**
 * Compliance CLI
 * Command-line interface for the compliance engine
 */

import {
  compliance,
  ComplianceReport,
  CheckContext,
  COMPLIANCE_REPORT_PATH,
} from '@ai/compliance.js';
import { saveJson } from '@ai/utils/file-persistence.js';

function displayReport(report: ComplianceReport): void {
  const gradeColors: Record<string, string> = {
    A: '\x1b[32m', // Green
    B: '\x1b[32m', // Green
    C: '\x1b[33m', // Yellow
    D: '\x1b[33m', // Yellow
    F: '\x1b[31m', // Red
  };
  const reset = '\x1b[0m';

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║            📋 AI COMPLIANCE REPORT                           ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log(
    `║  Overall Score: ${gradeColors[report.grade]}${report.overallScore}/100${reset} (Grade: ${gradeColors[report.grade]}${report.grade}${reset})`.padEnd(
      75
    ) + '║'
  );
  console.log(`║  Timestamp: ${report.timestamp}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Summary
  console.log('║  📊 SUMMARY                                                  ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  ✅ Passed: ${report.summary.passed}`.padEnd(65) + '║');
  console.log(`║  ❌ Failed: ${report.summary.failed}`.padEnd(65) + '║');
  console.log(`║  ⚠️  Warnings: ${report.summary.warnings}`.padEnd(65) + '║');
  console.log(`║  🚨 Critical: ${report.summary.critical}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Category scores
  console.log('║  📁 BY CATEGORY                                              ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  for (const [category, score] of Object.entries(report.byCategory)) {
    const bar =
      '█'.repeat(Math.floor(score.score / 10)) + '░'.repeat(10 - Math.floor(score.score / 10));
    console.log(`║  ${category.padEnd(15)} [${bar}] ${score.score}%`.padEnd(65) + '║');
  }
  console.log('║                                                              ║');

  // Violations
  if (report.violations.length > 0) {
    console.log('║  🚫 VIOLATIONS                                               ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');
    for (const v of report.violations.slice(0, 5)) {
      const icon = v.severity === 'critical' ? '🚨' : v.severity === 'high' ? '❌' : '⚠️';
      console.log(`║  ${icon} [${v.ruleId}] ${v.ruleName}`.padEnd(65) + '║');
      console.log(`║     ${v.message}`.padEnd(65) + '║');
    }
    if (report.violations.length > 5) {
      console.log(`║  ... and ${report.violations.length - 5} more`.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  // Recommendations
  if (report.recommendations.length > 0) {
    console.log('║  💡 RECOMMENDATIONS                                          ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');
    for (const rec of report.recommendations.slice(0, 3)) {
      console.log(`║  • ${rec}`.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'check': {
      const files = args.slice(1);
      const context: CheckContext = {
        files: [],
        changedFiles: files.length > 0 ? files : [],
      };
      const report = compliance.evaluate(context);
      displayReport(report);

      // Save report
      saveJson(COMPLIANCE_REPORT_PATH, report);
      break;
    }

    case 'rules': {
      console.log('\n📋 Compliance Rules\n');
      const rules = compliance.listRules();
      for (const rule of rules) {
        const icon =
          rule.severity === 'critical'
            ? '🚨'
            : rule.severity === 'high'
              ? '❌'
              : rule.severity === 'medium'
                ? '⚠️'
                : 'ℹ️';
        console.log(`${icon} [${rule.id}] ${rule.name} (${rule.category})`);
      }
      break;
    }

    case 'score': {
      const context: CheckContext = { files: [], changedFiles: [] };
      const report = compliance.evaluate(context);
      console.log(`\nCompliance Score: ${report.overallScore}/100 (Grade: ${report.grade})\n`);
      break;
    }

    default:
      console.log(`
AI Compliance - Policy-based validation with scoring

Commands:
  check [files...]   Run compliance check on files
  rules              List all compliance rules
  score              Quick score check

Examples:
  npm run ai:compliance check tools/ai/cache.ts
  npm run ai:compliance rules
  npm run ai:compliance score
      `);
  }
}

main();
