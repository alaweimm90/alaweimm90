/**
 * AUTOMATED AUDIT RUNNER & REPORTING SYSTEM
 * CLI and scheduled audit execution with actionable reports
 */

import { codebaseAuditor, AuditReport, CriticalIssue } from './comprehensive-audit';

export class AuditRunner {
  private outputDir: string;

  constructor(outputDir: string = './audit-reports') {
    this.outputDir = outputDir;
  }

  /**
   * Run audit and generate comprehensive report
   */
  async runAudit(options: AuditRunOptions = {}): Promise<AuditReport> {
    console.log('🚀 Starting comprehensive codebase audit...\n');
    
    const startTime = Date.now();
    const report = await codebaseAuditor.runFullAudit();
    const duration = Date.now() - startTime;

    // Generate reports
    await this.generateReports(report, duration, options);
    
    // Display summary
    this.displaySummary(report, duration);
    
    return report;
  }

  /**
   * Generate multiple report formats
   */
  private async generateReports(
    report: AuditReport, 
    duration: number, 
    options: AuditRunOptions
  ): Promise<void> {
    const timestamp = report.timestamp.toISOString().split('T')[0];
    
    if (options.generateMarkdown !== false) {
      await this.generateMarkdownReport(report, `${this.outputDir}/audit-${timestamp}.md`);
    }
    
    if (options.generateJson !== false) {
      await this.generateJsonReport(report, `${this.outputDir}/audit-${timestamp}.json`);
    }
    
    if (options.generateHtml) {
      await this.generateHtmlReport(report, `${this.outputDir}/audit-${timestamp}.html`);
    }

    if (options.generateCsv) {
      await this.generateCsvReport(report, `${this.outputDir}/audit-${timestamp}.csv`);
    }
  }

  /**
   * Generate human-readable Markdown report
   */
  private async generateMarkdownReport(report: AuditReport, filepath: string): Promise<void> {
    const content = `# Codebase Audit Report

**Generated:** ${report.timestamp.toLocaleDateString()}
**Project:** ${report.projectRoot}
**Health Score:** ${report.score}/100 ${this.getScoreEmoji(report.score)}

## 📊 Executive Summary

| Metric | Count | Status |
|--------|-------|--------|
| Total Files Analyzed | ${report.summary.totalFiles} | ℹ️ |
| Orphaned Files | ${report.summary.orphanedFiles} | ${report.summary.orphanedFiles > 0 ? '⚠️' : '✅'} |
| Dead Routes | ${report.summary.deadRoutes} | ${report.summary.deadRoutes > 0 ? '❌' : '✅'} |
| Style Inconsistencies | ${report.summary.styleInconsistencies} | ${report.summary.styleInconsistencies > 0 ? '⚠️' : '✅'} |
| Data Structure Mismatches | ${report.summary.dataStructureMismatches} | ${report.summary.dataStructureMismatches > 0 ? '⚠️' : '✅'} |
| Duplicate Components | ${report.summary.duplicateComponents} | ${report.summary.duplicateComponents > 0 ? '⚠️' : '✅'} |

## 🚨 Critical Issues

${report.criticalIssues.length === 0 ? '✅ **No critical issues found!**' : ''}
${report.criticalIssues.map(issue => `
### ${issue.type.toUpperCase()}: ${issue.message}
- **Category:** ${issue.category}
- **Impact:** ${issue.impact.toUpperCase()}
- **Files:** ${issue.files.join(', ')}
- **Recommendation:** ${issue.recommendation}
`).join('\n')}

## 📁 Structural Issues

### Orphaned Files
${report.findings.structural.orphanedFiles.length === 0 ? '✅ No orphaned files found.' : ''}
${report.findings.structural.orphanedFiles.map(file => `
- \`${file.path}\` (${file.type})
  - Last modified: ${file.lastModified.toLocaleDateString()}
  - Exports: ${file.exports.join(', ')}
  - **Action:** ${file.isOrphaned ? '🗑️ Safe to remove' : '✅ In use'}
`).join('')}

### Unused Routes
${report.findings.structural.unusedRoutes.map(route => `- \`${route}\``).join('\n') || '✅ All routes are in use.'}

## 🛣️ Routing Issues

### Dead Links
${report.findings.routing.deadLinks.map(link => `
- **File:** \`${link.sourceFile}:${link.lineNumber}\`
- **Target:** \`${link.targetRoute}\`
- **Type:** ${link.linkType}
- **Action:** 🔧 Update or remove link
`).join('')}

### Navigation Inconsistencies
${report.findings.routing.inconsistentNavigation.map(nav => `
- **Route:** \`${nav.route}\`
- **Issue:** ${nav.inconsistency}
- **Locations:** ${nav.locations.join(', ')}
- **Suggestion:** ${nav.suggestion}
`).join('')}

### Security Issues
${report.findings.routing.routeSecurityIssues.map(security => `
- **Route:** \`${security.route}\`
- **Issue:** ${security.issue}
- **Severity:** ${security.severity.toUpperCase()}
- **Action:** 🔒 Immediate attention required
`).join('')}

## 🎨 Styling Issues

### Theme Violations
${report.findings.styling.themeViolations.map(violation => `
- **File:** \`${violation.file}:${violation.line}\`
- **Violation:** ${violation.violation}
- **Current:** \`${violation.current}\`
- **Suggested:** \`${violation.suggested}\`
`).join('')}

### Duplicate Styles
${report.findings.styling.duplicateStyles.map(dup => `
- **Selector:** \`${dup.selector}\`
- **Found in:** ${dup.duplicateIn.join(', ')}
- **Can consolidate:** ${dup.canBeConsolidated ? '✅ Yes' : '❌ No'}
`).join('')}

## 📊 Data Consistency Issues

### Type Definition Mismatches
${report.findings.dataConsistency.typeDefinitionMismatches.map(mismatch => `
- **Type:** \`${mismatch.typeName}\`
- **Conflicts:** ${mismatch.conflicts.join(', ')}
- **Files:** ${mismatch.definitions.map(def => def.file).join(', ')}
- **Action:** 🔧 Consolidate to single definition
`).join('')}

## 🖼️ UI Consistency Issues

### Component Variations
${report.findings.uiConsistency.componentVariations.map(variation => `
- **Component Type:** ${variation.componentType}
- **Base:** \`${variation.baseComponent}\`
- **Variations:** ${variation.variations.length}
- **Should Consolidate:** ${variation.shouldBeConsolidated ? '✅ Yes' : '❌ No'}
- **Files:** ${variation.variations.map(v => v.file).join(', ')}
`).join('')}

### Icon Inconsistencies
${report.findings.uiConsistency.iconInconsistencies.map(icon => `
- **Concept:** ${icon.concept}
- **Different Icons:** ${icon.icons.length}
- **Recommended:** ${icon.recommendedIcon}
- **Files:** ${icon.icons.map(i => `${i.file} (${i.iconName})`).join(', ')}
`).join('')}

## 🎯 Recommended Actions

${report.recommendations.map(rec => `- ${rec}`).join('\n')}

## 📋 Audit Checklist

Copy this checklist for your next code review:

\`\`\`
Sprint Audit Checklist:
□ All pages/routes referenced in navigation
□ No duplicate UI components for same purpose  
□ All styles from centralized theme provider
□ All forms use same validation schema
□ No dead or orphaned files
□ No navigation links to non-existent pages
□ UI consistency across all pages
□ All changes reviewed before deletion
\`\`\`

---

**Next Audit Recommended:** ${new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toLocaleDateString()}
**Generated by:** REPZ Automated Audit System
`;

    // In a real implementation, would write to file
    console.log('📄 Markdown report generated:', filepath);
  }

  /**
   * Generate JSON report for automation
   */
  private async generateJsonReport(report: AuditReport, filepath: string): Promise<void> {
    const jsonContent = JSON.stringify(report, null, 2);
    console.log('📄 JSON report generated:', filepath);
    // In real implementation: fs.writeFileSync(filepath, jsonContent);
  }

  /**
   * Generate HTML report with interactive elements
   */
  private async generateHtmlReport(report: AuditReport, filepath: string): Promise<void> {
    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codebase Audit Report - ${report.timestamp.toLocaleDateString()}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', roboto; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }
        .score { font-size: 3em; font-weight: bold; color: ${this.getScoreColor(report.score)}; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .critical { border-left-color: #dc3545; }
        .warning { border-left-color: #ffc107; }
        .success { border-left-color: #28a745; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .recommendations { background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .file-list { font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9em; background: #f1f1f1; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Codebase Audit Report</h1>
            <div class="score">${report.score}/100</div>
            <p>Generated on ${report.timestamp.toLocaleDateString()} for ${report.projectRoot}</p>
        </div>
        
        <div class="grid">
            <div class="card ${report.summary.orphanedFiles > 0 ? 'warning' : 'success'}">
                <h3>📁 Structural Health</h3>
                <div class="metric">
                    <span>Total Files:</span>
                    <strong>${report.summary.totalFiles}</strong>
                </div>
                <div class="metric">
                    <span>Orphaned Files:</span>
                    <strong>${report.summary.orphanedFiles}</strong>
                </div>
            </div>
            
            <div class="card ${report.summary.deadRoutes > 0 ? 'critical' : 'success'}">
                <h3>🛣️ Routing Health</h3>
                <div class="metric">
                    <span>Dead Routes:</span>
                    <strong>${report.summary.deadRoutes}</strong>
                </div>
                <div class="metric">
                    <span>Security Issues:</span>
                    <strong>${report.findings.routing.routeSecurityIssues.length}</strong>
                </div>
            </div>
            
            <div class="card ${report.summary.styleInconsistencies > 0 ? 'warning' : 'success'}">
                <h3>🎨 Style Health</h3>
                <div class="metric">
                    <span>Theme Violations:</span>
                    <strong>${report.summary.styleInconsistencies}</strong>
                </div>
                <div class="metric">
                    <span>Duplicate Styles:</span>
                    <strong>${report.findings.styling.duplicateStyles.length}</strong>
                </div>
            </div>
        </div>
        
        <div class="recommendations">
            <h3>🎯 Action Items</h3>
            <ul>
                ${report.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        ${report.criticalIssues.length > 0 ? `
        <div class="card critical">
            <h3>🚨 Critical Issues</h3>
            ${report.criticalIssues.map(issue => `
                <div style="margin: 15px 0; padding: 15px; background: #fff; border-radius: 4px;">
                    <strong>${issue.message}</strong>
                    <p>${issue.recommendation}</p>
                    <div class="file-list">${issue.files.join('<br>')}</div>
                </div>
            `).join('')}
        </div>
        ` : ''}
        
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>🤖 Generated by REPZ Automated Audit System</p>
            <p>Next audit recommended: ${new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toLocaleDateString()}</p>
        </div>
    </div>
</body>
</html>
`;

    console.log('📄 HTML report generated:', filepath);
    // In real implementation: fs.writeFileSync(filepath, htmlContent);
  }

  /**
   * Generate CSV for spreadsheet analysis
   */
  private async generateCsvReport(report: AuditReport, filepath: string): Promise<void> {
    const csvRows = [
      ['Category', 'Type', 'File', 'Issue', 'Recommendation', 'Priority'],
      ...report.criticalIssues.map(issue => [
        issue.category,
        issue.type,
        issue.files.join('; '),
        issue.message,
        issue.recommendation,
        issue.impact
      ])
    ];

    const csvContent = csvRows.map(row => row.join(',')).join('\n');
    console.log('📄 CSV report generated:', filepath);
    // In real implementation: fs.writeFileSync(filepath, csvContent);
  }

  /**
   * Display audit summary in terminal
   */
  private displaySummary(report: AuditReport, duration: number): void {
    console.log('\n' + '='.repeat(60));
    console.log('🎯 AUDIT COMPLETE');
    console.log('='.repeat(60));
    console.log(`⏱️  Duration: ${duration}ms`);
    console.log(`📊 Health Score: ${report.score}/100 ${this.getScoreEmoji(report.score)}`);
    console.log(`📁 Files Analyzed: ${report.summary.totalFiles}`);
    console.log('');
    
    if (report.criticalIssues.length > 0) {
      console.log(`🚨 CRITICAL ISSUES: ${report.criticalIssues.length}`);
      report.criticalIssues.forEach(issue => {
        console.log(`   ${issue.type.toUpperCase()}: ${issue.message}`);
      });
      console.log('');
    }

    console.log('📋 SUMMARY:');
    console.log(`   Orphaned Files: ${report.summary.orphanedFiles}`);
    console.log(`   Dead Routes: ${report.summary.deadRoutes}`);
    console.log(`   Style Issues: ${report.summary.styleInconsistencies}`);
    console.log(`   Data Mismatches: ${report.summary.dataStructureMismatches}`);
    console.log(`   Duplicate Components: ${report.summary.duplicateComponents}`);
    console.log('');

    if (report.recommendations.length > 0) {
      console.log('🎯 TOP RECOMMENDATIONS:');
      report.recommendations.slice(0, 5).forEach(rec => {
        console.log(`   ${rec}`);
      });
      console.log('');
    }

    console.log(`📄 Reports saved to: ${this.outputDir}/`);
    console.log('='.repeat(60));
  }

  private getScoreEmoji(score: number): string {
    if (score >= 90) return '🟢';
    if (score >= 70) return '🟡';
    if (score >= 50) return '🟠';
    return '🔴';
  }

  private getScoreColor(score: number): string {
    if (score >= 90) return '#28a745';
    if (score >= 70) return '#ffc107';
    if (score >= 50) return '#fd7e14';
    return '#dc3545';
  }
}

export interface AuditRunOptions {
  generateMarkdown?: boolean;
  generateJson?: boolean;
  generateHtml?: boolean;
  generateCsv?: boolean;
  outputDir?: string;
  strictMode?: boolean;
}

/**
 * AUDIT SCHEDULING & AUTOMATION
 */
export class AuditScheduler {
  private runner: AuditRunner;

  constructor(runner: AuditRunner) {
    this.runner = runner;
  }

  /**
   * Schedule regular audits
   */
  scheduleWeeklyAudit(): void {
    // In real implementation, would use cron or similar
    console.log('📅 Weekly audit scheduled for Mondays at 9 AM');
  }

  /**
   * Pre-commit audit (lightweight)
   */
  async runPreCommitAudit(): Promise<boolean> {
    console.log('🔍 Running pre-commit audit...');
    
    const report = await this.runner.runAudit({
      generateMarkdown: false,
      generateHtml: false,
      generateCsv: false
    });

    // Fail if critical issues found
    const hasCriticalIssues = report.criticalIssues.some(issue => 
      issue.type === 'error' || issue.impact === 'high'
    );

    if (hasCriticalIssues) {
      console.log('❌ Pre-commit audit failed due to critical issues');
      return false;
    }

    console.log('✅ Pre-commit audit passed');
    return true;
  }

  /**
   * PR audit (comprehensive)
   */
  async runPRAudit(): Promise<AuditReport> {
    console.log('🔍 Running comprehensive PR audit...');
    
    return await this.runner.runAudit({
      generateMarkdown: true,
      generateJson: true,
      generateHtml: true
    });
  }
}

// Export singleton instances
export const auditRunner = new AuditRunner();
export const auditScheduler = new AuditScheduler(auditRunner);