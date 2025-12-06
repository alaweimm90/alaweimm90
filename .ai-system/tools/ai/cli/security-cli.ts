#!/usr/bin/env npx tsx
/**
 * Security Scanner CLI
 * Command-line interface for the security scanner
 */

import * as fs from 'fs';
import { SecurityScanner, SecurityReport, SECURITY_REPORT_FILE } from '@ai/security.js';

function displayReport(report: SecurityReport): void {
  const gradeColors: Record<string, string> = {
    A: '\x1b[32m',
    B: '\x1b[32m',
    C: '\x1b[33m',
    D: '\x1b[33m',
    F: '\x1b[31m',
  };
  const reset = '\x1b[0m';

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║            🔒 SECURITY SCAN REPORT                           ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log(
    `║  Security Score: ${gradeColors[report.grade]}${report.score}/100${reset} (Grade: ${gradeColors[report.grade]}${report.grade}${reset})`.padEnd(
      75
    ) + '║'
  );
  console.log(`║  Timestamp: ${report.timestamp}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Summary
  console.log('║  📊 SUMMARY                                                  ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  🔴 Critical: ${report.summary.critical}`.padEnd(65) + '║');
  console.log(`║  🟠 High: ${report.summary.high}`.padEnd(65) + '║');
  console.log(`║  🟡 Medium: ${report.summary.medium}`.padEnd(65) + '║');
  console.log(`║  🔵 Low: ${report.summary.low}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Breakdown
  console.log('║  📁 BY CATEGORY                                              ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  🔑 Secrets: ${report.summary.secrets}`.padEnd(65) + '║');
  console.log(`║  🐛 Vulnerabilities: ${report.summary.vulnerabilities}`.padEnd(65) + '║');
  console.log(`║  📜 License Issues: ${report.summary.licenseIssues}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Top Findings
  if (report.findings.length > 0) {
    console.log('║  🚨 TOP FINDINGS                                             ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');

    const topFindings = report.findings
      .filter((f) => f.severity === 'critical' || f.severity === 'high')
      .slice(0, 5);

    for (const finding of topFindings) {
      const icon = finding.severity === 'critical' ? '🔴' : '🟠';
      const shortDesc = finding.description.substring(0, 40);
      console.log(`║  ${icon} ${shortDesc}...`.padEnd(65) + '║');
      if (finding.file) {
        console.log(`║     📄 ${finding.file}:${finding.line || ''}`.padEnd(65) + '║');
      }
    }

    if (report.findings.length > 5) {
      console.log(`║  ... and ${report.findings.length - 5} more findings`.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  // Recommendations
  if (report.summary.totalFindings > 0) {
    console.log('║  💡 RECOMMENDATIONS                                          ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');
    if (report.summary.secrets > 0) {
      console.log('║  • Remove secrets from code, use env vars'.padEnd(65) + '║');
    }
    if (report.summary.vulnerabilities > 0) {
      console.log('║  • Run npm audit fix to patch vulnerabilities'.padEnd(65) + '║');
    }
    if (report.summary.licenseIssues > 0) {
      console.log('║  • Review license compatibility'.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];
  const scanner = new SecurityScanner();

  switch (command) {
    case 'scan':
    case 'full': {
      const paths = args.slice(1);
      const report = await scanner.fullScan(paths.length > 0 ? paths : ['.']);
      displayReport(report);
      break;
    }

    case 'secrets': {
      const paths = args.slice(1);
      console.log('\n🔍 Scanning for secrets...\n');
      const findings = await scanner.scanSecrets(paths.length > 0 ? paths : ['.']);
      console.log(`Found ${findings.length} potential secrets\n`);
      for (const finding of findings.slice(0, 10)) {
        console.log(`  ${finding.severity.toUpperCase()}: ${finding.description}`);
        console.log(`    File: ${finding.file}:${finding.line}`);
      }
      break;
    }

    case 'vulns':
    case 'vulnerabilities': {
      console.log('\n📦 Scanning npm vulnerabilities...\n');
      scanner.scanVulnerabilities();
      const report = scanner.generateReport();
      console.log(`Found ${report.summary.vulnerabilities} vulnerabilities\n`);
      break;
    }

    case 'licenses': {
      console.log('\n📜 Scanning licenses...\n');
      const licenses = scanner.scanLicenses();
      const issues = licenses.filter((l) => !l.compatible);
      console.log(`Found ${issues.length} license issues\n`);
      for (const issue of issues.slice(0, 10)) {
        console.log(`  ${issue.package}: ${issue.license}`);
      }
      break;
    }

    case 'report': {
      if (fs.existsSync(SECURITY_REPORT_FILE)) {
        const report = JSON.parse(fs.readFileSync(SECURITY_REPORT_FILE, 'utf8'));
        displayReport(report);
      } else {
        console.log('\n❌ No security report found. Run "npm run ai:security scan" first.\n');
      }
      break;
    }

    default:
      console.log(`
AI Security Scanner - Comprehensive security analysis

Commands:
  scan [paths...]     Run full security scan
  secrets [paths...]  Scan for secrets only
  vulns               Scan npm vulnerabilities
  licenses            Scan license compliance
  report              Display last scan report

Examples:
  npm run ai:security scan
  npm run ai:security secrets tools/
  npm run ai:security vulns
      `);
  }
}

main().catch(console.error);
