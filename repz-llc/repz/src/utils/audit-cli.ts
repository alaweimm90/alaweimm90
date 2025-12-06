#!/usr/bin/env node

/**
 * REPZ CODEBASE AUDIT CLI
 * Command-line interface for running systematic codebase audits
 * 
 * Usage:
 *   npm run audit              # Full audit with all reports
 *   npm run audit:quick        # Quick audit, JSON only
 *   npm run audit:pre-commit   # Lightweight pre-commit check
 *   npm run audit:schedule     # Set up scheduled audits
 */

import { auditRunner, auditScheduler, AuditRunOptions } from './audit-runner';

interface CLIArgs {
  command: 'full' | 'quick' | 'pre-commit' | 'schedule' | 'help';
  options: AuditRunOptions;
}

async function main() {
  const args = parseArgs();
  
  console.log('🔍 REPZ Codebase Audit System');
  console.log('================================\n');

  try {
    switch (args.command) {
      case 'full':
        await runFullAudit(args.options);
        break;
      case 'quick':
        await runQuickAudit(args.options);
        break;
      case 'pre-commit':
        await runPreCommitAudit();
        break;
      case 'schedule':
        setupScheduledAudits();
        break;
      case 'help':
      default:
        showHelp();
        break;
    }
  } catch (error) {
    console.error('❌ Audit failed:', error);
    process.exit(1);
  }
}

async function runFullAudit(options: AuditRunOptions): Promise<void> {
  console.log('🚀 Running comprehensive codebase audit...\n');
  
  const report = await auditRunner.runAudit({
    generateMarkdown: true,
    generateJson: true,
    generateHtml: true,
    generateCsv: true,
    ...options
  });

  // Exit with error code if critical issues found
  const hasCriticalIssues = report.criticalIssues.some(issue => 
    issue.type === 'error'
  );

  if (hasCriticalIssues) {
    console.log('\n❌ Audit completed with critical issues');
    process.exit(1);
  } else {
    console.log('\n✅ Audit completed successfully');
    process.exit(0);
  }
}

async function runQuickAudit(options: AuditRunOptions): Promise<void> {
  console.log('⚡ Running quick audit (JSON only)...\n');
  
  await auditRunner.runAudit({
    generateMarkdown: false,
    generateHtml: false,
    generateCsv: false,
    generateJson: true,
    ...options
  });

  console.log('\n✅ Quick audit completed');
}

async function runPreCommitAudit(): Promise<void> {
  console.log('🔒 Running pre-commit audit...\n');
  
  const passed = await auditScheduler.runPreCommitAudit();
  
  if (!passed) {
    console.log('\n❌ Pre-commit audit failed - commit blocked');
    process.exit(1);
  } else {
    console.log('\n✅ Pre-commit audit passed - commit allowed');
    process.exit(0);
  }
}

function setupScheduledAudits(): void {
  console.log('📅 Setting up scheduled audits...\n');
  
  auditScheduler.scheduleWeeklyAudit();
  
  console.log('✅ Scheduled audits configured:');
  console.log('   - Weekly comprehensive audit: Mondays 9 AM');
  console.log('   - Pre-commit hooks: On every commit');
  console.log('   - PR audits: On pull request creation');
  console.log('\nTo enable, add to your CI/CD pipeline or crontab.');
}

function parseArgs(): CLIArgs {
  const args = process.argv.slice(2);
  
  // Default command
  if (args.length === 0) {
    return { command: 'full', options: {} };
  }

  const command = args[0] as CLIArgs['command'];
  const options: AuditRunOptions = {};

  // Parse additional flags
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--no-markdown':
        options.generateMarkdown = false;
        break;
      case '--no-html':
        options.generateHtml = false;
        break;
      case '--no-json':
        options.generateJson = false;
        break;
      case '--no-csv':
        options.generateCsv = false;
        break;
      case '--strict':
        options.strictMode = true;
        break;
      case '--output':
        options.outputDir = args[++i];
        break;
    }
  }

  return { command, options };
}

function showHelp(): void {
  console.log(`
🔍 REPZ Codebase Audit CLI

COMMANDS:
  full        Run comprehensive audit with all reports (default)
  quick       Run quick audit, JSON output only
  pre-commit  Run lightweight audit for pre-commit hooks
  schedule    Set up scheduled audit tasks
  help        Show this help message

OPTIONS:
  --no-markdown    Skip Markdown report generation
  --no-html        Skip HTML report generation  
  --no-json        Skip JSON report generation
  --no-csv         Skip CSV report generation
  --strict         Use strict mode (more thorough checks)
  --output <dir>   Custom output directory for reports

EXAMPLES:
  npm run audit                    # Full audit with all reports
  npm run audit quick              # Quick JSON-only audit
  npm run audit full --no-html     # Full audit without HTML
  npm run audit pre-commit         # Pre-commit check
  npm run audit schedule           # Set up automated audits

INTEGRATION:
  Add to package.json scripts:
  {
    "scripts": {
      "audit": "node src/utils/audit-cli.js full",
      "audit:quick": "node src/utils/audit-cli.js quick", 
      "audit:pre-commit": "node src/utils/audit-cli.js pre-commit"
    }
  }

  Add to .github/workflows/audit.yml:
  - name: Run Codebase Audit
    run: npm run audit

  Add to .husky/pre-commit:
  npm run audit:pre-commit

REPORTS:
  All reports are saved to ./audit-reports/ by default
  - audit-YYYY-MM-DD.md     # Human-readable Markdown
  - audit-YYYY-MM-DD.json   # Machine-readable JSON
  - audit-YYYY-MM-DD.html   # Interactive HTML dashboard  
  - audit-YYYY-MM-DD.csv    # Spreadsheet-compatible data

For more information, see ARCHITECTURE.md
`);
}

// Execute if run directly
if (require.main === module) {
  main().catch(console.error);
}

export { main as runAuditCLI };