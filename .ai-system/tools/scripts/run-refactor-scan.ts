/**
 * Execute AutoRefactorPipeline to scan for refactoring opportunities
 */

import { AutoRefactorPipeline } from '../automation/workflows/auto-refactor.js';
import * as fs from 'fs';
import * as path from 'path';

async function runRefactorScan() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║           AUTO-REFACTOR PIPELINE - OPPORTUNITY SCAN          ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  const pipeline = new AutoRefactorPipeline({
    workspacePath: process.cwd(),
    dryRun: true,
    maxFileSize: 300,
    maxFunctionComplexity: 10,
    maxFunctionLength: 50,
    excludePatterns: ['node_modules', 'dist', '.archive', '.git'],
  });

  console.log('🔍 Scanning for refactoring opportunities...\n');

  try {
    const opportunities = await pipeline.scan();

    console.log(`📊 Found ${opportunities.length} refactoring opportunities\n`);

    // Group by type
    const byType: Record<string, typeof opportunities> = {};
    for (const opp of opportunities) {
      if (!byType[opp.type]) byType[opp.type] = [];
      byType[opp.type].push(opp);
    }

    const typeIcons: Record<string, string> = {
      'large-file': '📄',
      'complex-function': '🔀',
      'missing-types': '📝',
      'long-function': '📏',
    };

    for (const [type, items] of Object.entries(byType)) {
      const icon = typeIcons[type] || '⚙️';
      console.log(`${icon} ${type.replace('-', ' ').toUpperCase()} (${items.length})`);
      console.log('─'.repeat(50));

      for (const opp of items.slice(0, 5)) {
        console.log(`   📍 ${opp.file}`);
        if (opp.location) {
          console.log(`      Line ${opp.location.startLine}-${opp.location.endLine}`);
        }
        console.log(`      💡 ${opp.suggestion}`);
        console.log('');
      }

      if (items.length > 5) {
        console.log(`   ... and ${items.length - 5} more ${type} opportunities\n`);
      }
    }

    // Summary
    console.log('┌─────────────────────────────────────────────────────────────┐');
    console.log('│                         SUMMARY                             │');
    console.log('└─────────────────────────────────────────────────────────────┘\n');

    const summary = {
      total: opportunities.length,
      byType: Object.fromEntries(Object.entries(byType).map(([k, v]) => [k, v.length])),
      topFiles: [...new Set(opportunities.map((o) => o.file))].slice(0, 5),
    };

    console.log(`   Total Opportunities: ${summary.total}`);
    for (const [type, count] of Object.entries(summary.byType)) {
      console.log(`   - ${type}: ${count}`);
    }
    console.log(`\n   Top Files to Refactor:`);
    for (const file of summary.topFiles) {
      const count = opportunities.filter((o) => o.file === file).length;
      console.log(`   - ${file} (${count} opportunities)`);
    }

    // Save report
    const reportDir = path.join(process.cwd(), '.archive/reports/governance');
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const reportPath = path.join(
      reportDir,
      `refactor-scan-${new Date().toISOString().split('T')[0]}.json`
    );
    fs.writeFileSync(reportPath, JSON.stringify({ summary, opportunities }, null, 2));
    console.log(`\n💾 Report saved to: ${reportPath}`);
  } catch (error) {
    console.error('❌ Error scanning:', error);
  }
}

runRefactorScan().catch(console.error);
