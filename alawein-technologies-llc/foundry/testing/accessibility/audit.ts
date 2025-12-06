import { chromium, Browser, Page } from 'playwright';
import * as axe from 'axe-core';
import * as fs from 'fs';
import * as path from 'path';

interface AuditResult {
  url: string;
  timestamp: string;
  violations: any[];
  passes: any[];
  incomplete: any[];
  inapplicable: any[];
  score: number;
  wcagCompliance: {
    level: string;
    passed: boolean;
    issues: string[];
  };
}

interface AuditConfig {
  urls: string[];
  viewports: Array<{ width: number; height: number; name: string }>;
  outputPath: string;
  generateReport: boolean;
}

class AccessibilityAuditor {
  private browser: Browser | null = null;
  private results: AuditResult[] = [];
  private config: AuditConfig;

  constructor(config: AuditConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    this.browser = await chromium.launch({
      headless: true,
      args: ['--force-prefers-reduced-motion'],
    });
  }

  async auditPage(url: string, viewport: { width: number; height: number; name: string }): Promise<AuditResult> {
    if (!this.browser) throw new Error('Browser not initialized');

    const page = await this.browser.newPage();
    await page.setViewportSize(viewport);

    try {
      // Navigate to the page
      await page.goto(url, { waitUntil: 'networkidle' });

      // Inject axe-core
      await page.addScriptTag({
        content: fs.readFileSync(require.resolve('axe-core'), 'utf8'),
      });

      // Run accessibility tests
      const results = await page.evaluate(() => {
        return (window as any).axe.run({
          rules: {
            'color-contrast': { enabled: true },
            'image-alt': { enabled: true },
            'label': { enabled: true },
            'button-name': { enabled: true },
            'link-name': { enabled: true },
            'aria-allowed-attr': { enabled: true },
            'aria-required-attr': { enabled: true },
            'aria-valid-attr': { enabled: true },
            'aria-valid-attr-value': { enabled: true },
            'aria-roles': { enabled: true },
          },
        });
      });

      // Additional custom checks
      const customChecks = await this.runCustomChecks(page);

      // Calculate compliance score
      const score = this.calculateScore(results);
      const wcagCompliance = this.checkWCAGCompliance(results);

      const auditResult: AuditResult = {
        url,
        timestamp: new Date().toISOString(),
        violations: results.violations,
        passes: results.passes,
        incomplete: results.incomplete,
        inapplicable: results.inapplicable,
        score,
        wcagCompliance,
      };

      // Take screenshot of issues
      if (results.violations.length > 0) {
        await this.captureViolations(page, results.violations, url, viewport.name);
      }

      return auditResult;
    } finally {
      await page.close();
    }
  }

  async runCustomChecks(page: Page): Promise<any> {
    return await page.evaluate(() => {
      const checks = {
        focusManagement: true,
        keyboardNavigation: true,
        touchTargets: true,
        animationControl: true,
      };

      // Check focus management
      const focusableElements = document.querySelectorAll(
        'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      checks.focusManagement = Array.from(focusableElements).every((el) => {
        const styles = window.getComputedStyle(el as HTMLElement);
        return styles.outline !== 'none' || styles.boxShadow !== 'none';
      });

      // Check keyboard navigation
      const interactiveElements = document.querySelectorAll(
        'a, button, input, select, textarea, [onclick]'
      );
      checks.keyboardNavigation = Array.from(interactiveElements).every((el) => {
        return (
          el.getAttribute('tabindex') !== '-1' &&
          (el as HTMLElement).offsetParent !== null
        );
      });

      // Check touch targets (minimum 44x44 pixels)
      const touchTargets = document.querySelectorAll('button, a, input');
      checks.touchTargets = Array.from(touchTargets).every((el) => {
        const rect = el.getBoundingClientRect();
        return rect.width >= 44 && rect.height >= 44;
      });

      // Check for animation control
      const animatedElements = document.querySelectorAll('[class*="animate"]');
      checks.animationControl =
        animatedElements.length === 0 ||
        document.querySelector('[data-testid="pause-animations"]') !== null;

      return checks;
    });
  }

  calculateScore(results: any): number {
    const totalChecks = results.passes.length + results.violations.length;
    if (totalChecks === 0) return 100;

    const passRate = results.passes.length / totalChecks;
    return Math.round(passRate * 100);
  }

  checkWCAGCompliance(results: any): any {
    const wcagIssues: string[] = [];
    let level = 'AAA';

    // Check for critical WCAG 2.1 Level AA violations
    const criticalViolations = [
      'color-contrast',
      'image-alt',
      'label',
      'button-name',
      'link-name',
      'aria-required-attr',
      'document-title',
      'html-has-lang',
    ];

    results.violations.forEach((violation: any) => {
      if (criticalViolations.includes(violation.id)) {
        wcagIssues.push(`${violation.id}: ${violation.description}`);
        if (level === 'AAA') level = 'AA';
      }
    });

    // Check for Level A violations
    const levelAViolations = ['image-alt', 'document-title', 'html-has-lang'];
    const hasLevelAViolations = results.violations.some((v: any) =>
      levelAViolations.includes(v.id)
    );

    if (hasLevelAViolations) {
      level = 'A';
    }

    return {
      level,
      passed: wcagIssues.length === 0,
      issues: wcagIssues,
    };
  }

  async captureViolations(
    page: Page,
    violations: any[],
    url: string,
    viewportName: string
  ): Promise<void> {
    for (const violation of violations) {
      for (const node of violation.nodes) {
        try {
          const element = await page.$(node.target[0]);
          if (element) {
            const screenshotPath = path.join(
              this.config.outputPath,
              'screenshots',
              `${this.sanitizeFilename(url)}-${viewportName}-${violation.id}.png`
            );

            await element.screenshot({ path: screenshotPath });

            // Highlight the element
            await page.evaluate((selector) => {
              const el = document.querySelector(selector);
              if (el) {
                (el as HTMLElement).style.outline = '3px solid red';
              }
            }, node.target[0]);
          }
        } catch (error) {
          console.error(`Failed to capture violation screenshot: ${error}`);
        }
      }
    }
  }

  sanitizeFilename(url: string): string {
    return url.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  }

  async runFullAudit(): Promise<void> {
    await this.initialize();

    try {
      for (const url of this.config.urls) {
        console.log(`Auditing ${url}...`);

        for (const viewport of this.config.viewports) {
          console.log(`  Testing ${viewport.name} viewport...`);
          const result = await this.auditPage(url, viewport);
          this.results.push(result);
        }
      }

      if (this.config.generateReport) {
        await this.generateReport();
      }
    } finally {
      if (this.browser) {
        await this.browser.close();
      }
    }
  }

  async generateReport(): Promise<void> {
    const report = {
      summary: {
        totalPages: this.results.length,
        averageScore: this.calculateAverageScore(),
        wcagCompliance: this.getOverallCompliance(),
        timestamp: new Date().toISOString(),
      },
      results: this.results,
      recommendations: this.generateRecommendations(),
    };

    // Generate HTML report
    const html = this.generateHTMLReport(report);
    fs.writeFileSync(path.join(this.config.outputPath, 'accessibility-report.html'), html);

    // Generate JSON report
    fs.writeFileSync(
      path.join(this.config.outputPath, 'accessibility-report.json'),
      JSON.stringify(report, null, 2)
    );

    console.log(`Reports generated in ${this.config.outputPath}`);
  }

  calculateAverageScore(): number {
    if (this.results.length === 0) return 0;
    const sum = this.results.reduce((acc, r) => acc + r.score, 0);
    return Math.round(sum / this.results.length);
  }

  getOverallCompliance(): string {
    const levels = this.results.map((r) => r.wcagCompliance.level);
    if (levels.includes('A')) return 'Partial A';
    if (levels.includes('AA')) return 'AA';
    return 'AAA';
  }

  generateRecommendations(): string[] {
    const recommendations: string[] = [];
    const violationCounts: Record<string, number> = {};

    // Count violations
    this.results.forEach((result) => {
      result.violations.forEach((violation: any) => {
        violationCounts[violation.id] = (violationCounts[violation.id] || 0) + 1;
      });
    });

    // Generate recommendations based on common issues
    Object.entries(violationCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .forEach(([violationId, count]) => {
        recommendations.push(this.getRecommendation(violationId, count));
      });

    return recommendations;
  }

  getRecommendation(violationId: string, count: number): string {
    const recommendations: Record<string, string> = {
      'color-contrast': `Fix color contrast issues in ${count} instances. Ensure text has a contrast ratio of at least 4.5:1 for normal text and 3:1 for large text.`,
      'image-alt': `Add alt text to ${count} images. Every image should have descriptive alt text unless it's purely decorative.`,
      'label': `Add labels to ${count} form controls. All form inputs need associated labels for screen reader users.`,
      'button-name': `Add accessible names to ${count} buttons. Use text content or aria-label to provide button purposes.`,
      'link-name': `Add descriptive text to ${count} links. Avoid generic text like "click here" or "read more".`,
      'aria-required-attr': `Add required ARIA attributes to ${count} elements. Elements with ARIA roles need their required attributes.`,
    };

    return recommendations[violationId] || `Fix ${violationId} issues found in ${count} instances.`;
  }

  generateHTMLReport(report: any): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Accessibility Audit Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }
    h1, h2, h3 { color: #333; }
    .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .score { font-size: 48px; font-weight: bold; }
    .score.good { color: #10b981; }
    .score.medium { color: #f59e0b; }
    .score.poor { color: #ef4444; }
    .violation { background: #fee; padding: 10px; margin: 10px 0; border-left: 4px solid #f44; }
    .pass { background: #efe; padding: 10px; margin: 10px 0; border-left: 4px solid #4f4; }
    .recommendation { background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f5f5f5; font-weight: 600; }
    .badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge.critical { background: #ef4444; color: white; }
    .badge.serious { background: #f59e0b; color: white; }
    .badge.moderate { background: #3b82f6; color: white; }
    .badge.minor { background: #6b7280; color: white; }
  </style>
</head>
<body>
  <h1>Accessibility Audit Report</h1>

  <div class="summary">
    <h2>Summary</h2>
    <p>Generated: ${report.summary.timestamp}</p>
    <p>Pages Tested: ${report.summary.totalPages}</p>
    <p>WCAG Compliance: ${report.summary.wcagCompliance}</p>
    <div class="score ${
      report.summary.averageScore >= 90
        ? 'good'
        : report.summary.averageScore >= 70
        ? 'medium'
        : 'poor'
    }">
      Score: ${report.summary.averageScore}%
    </div>
  </div>

  <h2>Recommendations</h2>
  ${report.recommendations
    .map((rec: string) => `<div class="recommendation">${rec}</div>`)
    .join('')}

  <h2>Detailed Results</h2>
  ${report.results
    .map(
      (result: AuditResult) => `
    <h3>${result.url}</h3>
    <p>Score: ${result.score}% | WCAG Level: ${result.wcagCompliance.level}</p>

    ${
      result.violations.length > 0
        ? `
      <h4>Violations (${result.violations.length})</h4>
      <table>
        <thead>
          <tr>
            <th>Issue</th>
            <th>Impact</th>
            <th>Elements</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          ${result.violations
            .map(
              (v: any) => `
            <tr>
              <td>${v.id}</td>
              <td><span class="badge ${v.impact}">${v.impact}</span></td>
              <td>${v.nodes.length}</td>
              <td>${v.description}</td>
            </tr>
          `
            )
            .join('')}
        </tbody>
      </table>
    `
        : '<p class="pass">No violations found!</p>'
    }
  `
    )
    .join('')}

</body>
</html>`;
  }
}

// Export for use in tests
export { AccessibilityAuditor, AuditConfig, AuditResult };

// Run audit if called directly
if (require.main === module) {
  const config: AuditConfig = {
    urls: [
      'http://localhost:3001',
      'http://localhost:3001/projects',
      'http://localhost:3002',
      'http://localhost:3002/discover',
      'http://localhost:3003',
      'http://localhost:3003/generator',
    ],
    viewports: [
      { width: 1920, height: 1080, name: 'desktop' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 375, height: 667, name: 'mobile' },
    ],
    outputPath: './accessibility-reports',
    generateReport: true,
  };

  const auditor = new AccessibilityAuditor(config);
  auditor.runFullAudit().catch(console.error);
}