import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface SecurityIssue {
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: string;
  description: string;
  file?: string;
  line?: number;
  recommendation: string;
}

interface SecurityAuditResult {
  app: string;
  timestamp: string;
  issues: SecurityIssue[];
  score: number;
  passed: boolean;
  summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

class SecurityAuditor {
  private results: SecurityAuditResult[] = [];
  private apps = ['ghost-researcher', 'scientific-tinder', 'chaos-engine'];

  async runFullAudit(): Promise<void> {
    console.log('Starting comprehensive security audit...\n');

    for (const app of this.apps) {
      console.log(`Auditing ${app}...`);
      const result = await this.auditApp(app);
      this.results.push(result);
    }

    await this.generateReport();
  }

  async auditApp(app: string): Promise<SecurityAuditResult> {
    const issues: SecurityIssue[] = [];
    const appPath = path.join('/home/user/CrazyIdeas/frontend', app);

    // 1. Dependency Vulnerabilities
    console.log('  Checking dependencies...');
    const depIssues = await this.checkDependencies(appPath);
    issues.push(...depIssues);

    // 2. Code Security Issues
    console.log('  Analyzing code security...');
    const codeIssues = await this.checkCodeSecurity(appPath);
    issues.push(...codeIssues);

    // 3. Authentication & Authorization
    console.log('  Checking authentication...');
    const authIssues = await this.checkAuthentication(appPath);
    issues.push(...authIssues);

    // 4. Input Validation
    console.log('  Checking input validation...');
    const inputIssues = await this.checkInputValidation(appPath);
    issues.push(...inputIssues);

    // 5. XSS Prevention
    console.log('  Checking XSS prevention...');
    const xssIssues = await this.checkXSSPrevention(appPath);
    issues.push(...xssIssues);

    // 6. API Security
    console.log('  Checking API security...');
    const apiIssues = await this.checkAPISecurity(appPath);
    issues.push(...apiIssues);

    // 7. File Upload Security
    console.log('  Checking file upload security...');
    const uploadIssues = await this.checkFileUploadSecurity(appPath);
    issues.push(...uploadIssues);

    // 8. Environment Variables
    console.log('  Checking environment security...');
    const envIssues = await this.checkEnvironmentSecurity(appPath);
    issues.push(...envIssues);

    // Calculate summary
    const summary = {
      critical: issues.filter(i => i.severity === 'critical').length,
      high: issues.filter(i => i.severity === 'high').length,
      medium: issues.filter(i => i.severity === 'medium').length,
      low: issues.filter(i => i.severity === 'low').length,
    };

    // Calculate score (0-100)
    const score = this.calculateSecurityScore(summary);
    const passed = summary.critical === 0 && summary.high === 0;

    return {
      app,
      timestamp: new Date().toISOString(),
      issues,
      score,
      passed,
      summary,
    };
  }

  async checkDependencies(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];

    try {
      // Run npm audit
      const { stdout } = await execAsync('npm audit --json', { cwd: appPath });
      const audit = JSON.parse(stdout);

      if (audit.vulnerabilities) {
        Object.entries(audit.vulnerabilities).forEach(([pkg, vuln]: [string, any]) => {
          issues.push({
            severity: this.mapNpmSeverity(vuln.severity),
            type: 'Dependency Vulnerability',
            description: `${pkg}: ${vuln.title || vuln.name}`,
            recommendation: `Update ${pkg} to version ${vuln.fixAvailable?.version || 'latest'}`,
          });
        });
      }
    } catch (error) {
      // npm audit returns non-zero exit code if vulnerabilities found
      console.log('    Dependencies check completed with issues');
    }

    // Check for outdated packages
    try {
      const { stdout } = await execAsync('npm outdated --json', { cwd: appPath });
      const outdated = JSON.parse(stdout);

      Object.keys(outdated).forEach(pkg => {
        if (this.isSecurityCriticalPackage(pkg)) {
          issues.push({
            severity: 'medium',
            type: 'Outdated Security Package',
            description: `${pkg} is outdated`,
            recommendation: `Update ${pkg} from ${outdated[pkg].current} to ${outdated[pkg].latest}`,
          });
        }
      });
    } catch {
      // npm outdated returns non-zero if packages are outdated
    }

    return issues;
  }

  async checkCodeSecurity(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    // Scan for common security anti-patterns
    const patterns = [
      {
        pattern: /eval\(/g,
        severity: 'critical' as const,
        type: 'Dangerous Function',
        message: 'Use of eval() detected',
        recommendation: 'Remove eval() and use safer alternatives',
      },
      {
        pattern: /innerHTML\s*=/g,
        severity: 'high' as const,
        type: 'XSS Risk',
        message: 'Direct innerHTML assignment detected',
        recommendation: 'Use textContent or sanitized HTML',
      },
      {
        pattern: /document\.write/g,
        severity: 'high' as const,
        type: 'XSS Risk',
        message: 'document.write() detected',
        recommendation: 'Use DOM methods instead of document.write',
      },
      {
        pattern: /localStorage\.setItem.*password/gi,
        severity: 'critical' as const,
        type: 'Sensitive Data Storage',
        message: 'Storing passwords in localStorage',
        recommendation: 'Never store passwords in localStorage',
      },
      {
        pattern: /api[kK]ey.*=.*['"][^'"]+['"]/g,
        severity: 'critical' as const,
        type: 'Hardcoded Secret',
        message: 'Hardcoded API key detected',
        recommendation: 'Use environment variables for API keys',
      },
      {
        pattern: /password.*=.*['"][^'"]+['"]/gi,
        severity: 'critical' as const,
        type: 'Hardcoded Secret',
        message: 'Hardcoded password detected',
        recommendation: 'Remove hardcoded passwords',
      },
      {
        pattern: /http:\/\//g,
        severity: 'medium' as const,
        type: 'Insecure Protocol',
        message: 'HTTP protocol used instead of HTTPS',
        recommendation: 'Always use HTTPS for external resources',
      },
    ];

    // Scan all TypeScript/JavaScript files
    const files = this.getAllFiles(srcPath, ['.ts', '.tsx', '.js', '.jsx']);

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const lines = content.split('\n');

      patterns.forEach(({ pattern, severity, type, message, recommendation }) => {
        lines.forEach((line, index) => {
          if (pattern.test(line)) {
            issues.push({
              severity,
              type,
              description: message,
              file: path.relative(appPath, file),
              line: index + 1,
              recommendation,
            });
          }
        });
      });
    }

    return issues;
  }

  async checkAuthentication(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    // Check for authentication best practices
    const authPatterns = [
      {
        check: 'JWT_EXPIRY',
        pattern: /expiresIn.*:.*['"]?(\d+)['"]?/g,
        validate: (match: string) => {
          const time = parseInt(match);
          return time > 86400; // More than 24 hours
        },
        issue: {
          severity: 'medium' as const,
          type: 'Authentication',
          description: 'JWT token expiry time is too long',
          recommendation: 'Set JWT expiry to 1 hour or less for better security',
        },
      },
      {
        check: 'MISSING_CSRF',
        pattern: /fetch.*POST|PUT|DELETE/g,
        validate: (match: string, content: string) => {
          return !content.includes('csrf-token') && !content.includes('X-CSRF');
        },
        issue: {
          severity: 'high' as const,
          type: 'CSRF Protection',
          description: 'Missing CSRF token in state-changing requests',
          recommendation: 'Implement CSRF protection for all state-changing operations',
        },
      },
    ];

    // Check auth implementation
    const authFiles = this.getAllFiles(srcPath, ['.ts', '.tsx']).filter(f =>
      f.toLowerCase().includes('auth')
    );

    for (const file of authFiles) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for secure password requirements
      if (content.includes('password') && !content.includes('minLength')) {
        issues.push({
          severity: 'high',
          type: 'Weak Password Policy',
          description: 'No password strength requirements found',
          file: path.relative(appPath, file),
          recommendation: 'Implement minimum password requirements (length, complexity)',
        });
      }

      // Check for rate limiting
      if (content.includes('login') && !content.includes('rateLimit')) {
        issues.push({
          severity: 'high',
          type: 'Missing Rate Limiting',
          description: 'No rate limiting on authentication endpoints',
          file: path.relative(appPath, file),
          recommendation: 'Implement rate limiting to prevent brute force attacks',
        });
      }

      // Check for secure session management
      if (content.includes('session') && !content.includes('httpOnly')) {
        issues.push({
          severity: 'high',
          type: 'Insecure Session',
          description: 'Session cookies not marked as httpOnly',
          file: path.relative(appPath, file),
          recommendation: 'Set httpOnly, secure, and sameSite flags on session cookies',
        });
      }
    }

    return issues;
  }

  async checkInputValidation(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    const formFiles = this.getAllFiles(srcPath, ['.tsx']).filter(f =>
      f.toLowerCase().includes('form')
    );

    for (const file of formFiles) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for input validation
      if (content.includes('<input') && !content.includes('validate')) {
        issues.push({
          severity: 'medium',
          type: 'Input Validation',
          description: 'Form inputs without validation',
          file: path.relative(appPath, file),
          recommendation: 'Add client and server-side validation for all user inputs',
        });
      }

      // Check for SQL injection prevention
      if (content.includes('query') && content.includes('`${')) {
        issues.push({
          severity: 'critical',
          type: 'SQL Injection Risk',
          description: 'Potential SQL injection vulnerability with string interpolation',
          file: path.relative(appPath, file),
          recommendation: 'Use parameterized queries or prepared statements',
        });
      }

      // Check for file upload restrictions
      if (content.includes('type="file"') && !content.includes('accept=')) {
        issues.push({
          severity: 'medium',
          type: 'File Upload',
          description: 'File upload without type restrictions',
          file: path.relative(appPath, file),
          recommendation: 'Restrict file types and validate file content on server',
        });
      }
    }

    return issues;
  }

  async checkXSSPrevention(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    const reactFiles = this.getAllFiles(srcPath, ['.tsx']);

    for (const file of reactFiles) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for dangerouslySetInnerHTML
      if (content.includes('dangerouslySetInnerHTML')) {
        issues.push({
          severity: 'high',
          type: 'XSS Risk',
          description: 'Use of dangerouslySetInnerHTML detected',
          file: path.relative(appPath, file),
          recommendation: 'Sanitize HTML content with DOMPurify or avoid dangerouslySetInnerHTML',
        });
      }

      // Check for user input in URLs
      if (content.includes('window.location') && content.includes('user')) {
        issues.push({
          severity: 'medium',
          type: 'Open Redirect',
          description: 'Potential open redirect vulnerability',
          file: path.relative(appPath, file),
          recommendation: 'Validate and whitelist redirect URLs',
        });
      }

      // Check for Content Security Policy
      const indexHtml = path.join(appPath, 'public', 'index.html');
      if (fs.existsSync(indexHtml)) {
        const htmlContent = fs.readFileSync(indexHtml, 'utf8');
        if (!htmlContent.includes('Content-Security-Policy')) {
          issues.push({
            severity: 'medium',
            type: 'Missing CSP',
            description: 'No Content Security Policy header found',
            file: 'public/index.html',
            recommendation: 'Implement Content Security Policy to prevent XSS attacks',
          });
        }
      }
    }

    return issues;
  }

  async checkAPISecurity(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    const apiFiles = this.getAllFiles(srcPath, ['.ts', '.tsx']).filter(f =>
      f.toLowerCase().includes('api')
    );

    for (const file of apiFiles) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for API key exposure
      if (content.includes('headers') && !content.includes('Authorization')) {
        issues.push({
          severity: 'medium',
          type: 'API Security',
          description: 'API calls without authentication headers',
          file: path.relative(appPath, file),
          recommendation: 'Add proper authentication headers to API requests',
        });
      }

      // Check for CORS configuration
      if (content.includes('Access-Control-Allow-Origin: *')) {
        issues.push({
          severity: 'high',
          type: 'CORS Misconfiguration',
          description: 'Overly permissive CORS configuration',
          file: path.relative(appPath, file),
          recommendation: 'Restrict CORS to specific trusted domains',
        });
      }

      // Check for rate limiting headers
      if (!content.includes('X-RateLimit')) {
        issues.push({
          severity: 'medium',
          type: 'Missing Rate Limiting',
          description: 'No rate limiting headers detected',
          file: path.relative(appPath, file),
          recommendation: 'Implement rate limiting on API endpoints',
        });
      }
    }

    return issues;
  }

  async checkFileUploadSecurity(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];
    const srcPath = path.join(appPath, 'src');

    const uploadFiles = this.getAllFiles(srcPath, ['.ts', '.tsx']).filter(f =>
      f.toLowerCase().includes('upload')
    );

    for (const file of uploadFiles) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for file size limits
      if (!content.includes('maxSize') && !content.includes('sizeLimit')) {
        issues.push({
          severity: 'medium',
          type: 'File Upload',
          description: 'No file size limit enforced',
          file: path.relative(appPath, file),
          recommendation: 'Implement file size limits to prevent DoS attacks',
        });
      }

      // Check for file type validation
      if (!content.includes('mimetype') && !content.includes('fileType')) {
        issues.push({
          severity: 'high',
          type: 'File Upload',
          description: 'No file type validation',
          file: path.relative(appPath, file),
          recommendation: 'Validate file types and scan for malware',
        });
      }

      // Check for path traversal prevention
      if (content.includes('../') || content.includes('..\\\\')) {
        issues.push({
          severity: 'critical',
          type: 'Path Traversal',
          description: 'Potential path traversal vulnerability',
          file: path.relative(appPath, file),
          recommendation: 'Sanitize file paths and use safe file storage',
        });
      }
    }

    return issues;
  }

  async checkEnvironmentSecurity(appPath: string): Promise<SecurityIssue[]> {
    const issues: SecurityIssue[] = [];

    // Check .env files
    const envFile = path.join(appPath, '.env');
    const envExampleFile = path.join(appPath, '.env.example');

    if (fs.existsSync(envFile)) {
      const content = fs.readFileSync(envFile, 'utf8');

      // Check for exposed secrets
      if (content.includes('SECRET') || content.includes('KEY')) {
        issues.push({
          severity: 'critical',
          type: 'Environment Security',
          description: '.env file contains sensitive data',
          file: '.env',
          recommendation: 'Never commit .env files with real secrets to version control',
        });
      }
    }

    if (!fs.existsSync(envExampleFile)) {
      issues.push({
        severity: 'low',
        type: 'Environment Security',
        description: 'Missing .env.example file',
        recommendation: 'Create .env.example with placeholder values for documentation',
      });
    }

    // Check gitignore
    const gitignore = path.join(appPath, '.gitignore');
    if (fs.existsSync(gitignore)) {
      const content = fs.readFileSync(gitignore, 'utf8');
      if (!content.includes('.env')) {
        issues.push({
          severity: 'critical',
          type: 'Version Control',
          description: '.env not in .gitignore',
          file: '.gitignore',
          recommendation: 'Add .env to .gitignore to prevent secret exposure',
        });
      }
    }

    return issues;
  }

  private getAllFiles(dirPath: string, extensions: string[]): string[] {
    const files: string[] = [];

    if (!fs.existsSync(dirPath)) return files;

    const items = fs.readdirSync(dirPath);

    for (const item of items) {
      const fullPath = path.join(dirPath, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory() && !item.includes('node_modules')) {
        files.push(...this.getAllFiles(fullPath, extensions));
      } else if (stat.isFile()) {
        const ext = path.extname(fullPath);
        if (extensions.includes(ext)) {
          files.push(fullPath);
        }
      }
    }

    return files;
  }

  private mapNpmSeverity(npmSeverity: string): SecurityIssue['severity'] {
    switch (npmSeverity) {
      case 'critical':
        return 'critical';
      case 'high':
        return 'high';
      case 'moderate':
        return 'medium';
      case 'low':
        return 'low';
      default:
        return 'low';
    }
  }

  private isSecurityCriticalPackage(pkg: string): boolean {
    const criticalPackages = [
      'jsonwebtoken',
      'bcrypt',
      'helmet',
      'cors',
      'express-rate-limit',
      'express-validator',
      'crypto-js',
      'axios',
    ];
    return criticalPackages.some(cp => pkg.includes(cp));
  }

  private calculateSecurityScore(summary: any): number {
    const weights = {
      critical: -30,
      high: -20,
      medium: -10,
      low: -5,
    };

    let score = 100;
    score += summary.critical * weights.critical;
    score += summary.high * weights.high;
    score += summary.medium * weights.medium;
    score += summary.low * weights.low;

    return Math.max(0, Math.min(100, score));
  }

  async generateReport(): Promise<void> {
    const reportDir = '/home/user/CrazyIdeas/testing/security/reports';
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    // Generate JSON report
    const jsonReport = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: {
        totalApps: this.results.length,
        passedApps: this.results.filter(r => r.passed).length,
        totalIssues: this.results.reduce((sum, r) => sum + r.issues.length, 0),
        criticalIssues: this.results.reduce((sum, r) => sum + r.summary.critical, 0),
        highIssues: this.results.reduce((sum, r) => sum + r.summary.high, 0),
        averageScore: Math.round(
          this.results.reduce((sum, r) => sum + r.score, 0) / this.results.length
        ),
      },
    };

    fs.writeFileSync(
      path.join(reportDir, 'security-audit.json'),
      JSON.stringify(jsonReport, null, 2)
    );

    // Generate HTML report
    const html = this.generateHTMLReport(jsonReport);
    fs.writeFileSync(path.join(reportDir, 'security-audit.html'), html);

    // Generate Markdown report
    const markdown = this.generateMarkdownReport(jsonReport);
    fs.writeFileSync(path.join(reportDir, 'security-audit.md'), markdown);

    console.log(`\nSecurity audit complete!`);
    console.log(`Reports generated in: ${reportDir}`);
    console.log(`Average security score: ${jsonReport.summary.averageScore}/100`);
    console.log(`Critical issues found: ${jsonReport.summary.criticalIssues}`);
  }

  private generateHTMLReport(report: any): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Security Audit Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
    h1 { color: #1a1a1a; margin-bottom: 10px; }
    h2 { color: #333; margin: 30px 0 15px; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
    h3 { color: #555; margin: 20px 0 10px; }
    .timestamp { color: #666; margin-bottom: 30px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
    .metric { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
    .metric-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
    .metric-label { color: #666; font-size: 14px; }
    .score { display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; }
    .score.good { background: #10b981; color: white; }
    .score.medium { background: #f59e0b; color: white; }
    .score.poor { background: #ef4444; color: white; }
    .severity { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; }
    .severity.critical { background: #ef4444; color: white; }
    .severity.high { background: #f59e0b; color: white; }
    .severity.medium { background: #3b82f6; color: white; }
    .severity.low { background: #6b7280; color: white; }
    .issue { background: #fff; border: 1px solid #e0e0e0; border-left: 4px solid; padding: 15px; margin: 10px 0; border-radius: 4px; }
    .issue.critical { border-left-color: #ef4444; }
    .issue.high { border-left-color: #f59e0b; }
    .issue.medium { border-left-color: #3b82f6; }
    .issue.low { border-left-color: #6b7280; }
    .issue-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .issue-type { font-weight: 600; }
    .issue-file { color: #666; font-size: 12px; margin-top: 5px; }
    .recommendation { background: #f0f9ff; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 14px; }
    .app-section { margin: 40px 0; padding: 20px; background: #fafafa; border-radius: 8px; }
    .pass-badge { background: #10b981; color: white; padding: 5px 10px; border-radius: 4px; font-weight: 600; }
    .fail-badge { background: #ef4444; color: white; padding: 5px 10px; border-radius: 4px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🔒 Security Audit Report</h1>
    <div class="timestamp">Generated: ${report.timestamp}</div>

    <h2>Executive Summary</h2>
    <div class="summary-grid">
      <div class="metric">
        <div class="metric-label">Average Score</div>
        <div class="metric-value">${report.summary.averageScore}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Total Issues</div>
        <div class="metric-value">${report.summary.totalIssues}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Critical Issues</div>
        <div class="metric-value" style="color: #ef4444;">${report.summary.criticalIssues}</div>
      </div>
      <div class="metric">
        <div class="metric-label">High Issues</div>
        <div class="metric-value" style="color: #f59e0b;">${report.summary.highIssues}</div>
      </div>
    </div>

    ${report.results
      .map(
        (result: SecurityAuditResult) => `
      <div class="app-section">
        <h2>${result.app}</h2>
        <div style="margin: 15px 0;">
          <span class="score ${
            result.score >= 80 ? 'good' : result.score >= 60 ? 'medium' : 'poor'
          }">Score: ${result.score}/100</span>
          ${result.passed ? '<span class="pass-badge">PASSED</span>' : '<span class="fail-badge">FAILED</span>'}
          <span style="margin-left: 20px;">
            Critical: ${result.summary.critical} |
            High: ${result.summary.high} |
            Medium: ${result.summary.medium} |
            Low: ${result.summary.low}
          </span>
        </div>

        ${
          result.issues.length > 0
            ? `
          <h3>Issues Found</h3>
          ${result.issues
            .sort((a, b) => {
              const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
              return severityOrder[a.severity] - severityOrder[b.severity];
            })
            .map(
              (issue) => `
            <div class="issue ${issue.severity}">
              <div class="issue-header">
                <span class="issue-type">${issue.type}</span>
                <span class="severity ${issue.severity}">${issue.severity}</span>
              </div>
              <div>${issue.description}</div>
              ${issue.file ? `<div class="issue-file">📁 ${issue.file}${issue.line ? `:${issue.line}` : ''}</div>` : ''}
              <div class="recommendation">
                <strong>Recommendation:</strong> ${issue.recommendation}
              </div>
            </div>
          `
            )
            .join('')}
        `
            : '<p style="color: #10b981; font-weight: 600;">✅ No security issues found!</p>'
        }
      </div>
    `
      )
      .join('')}

    <h2>Next Steps</h2>
    <ol style="margin: 20px 0; padding-left: 20px;">
      <li>Address all critical and high severity issues immediately</li>
      <li>Implement recommended security headers and CSP</li>
      <li>Set up automated security scanning in CI/CD pipeline</li>
      <li>Conduct regular security audits and penetration testing</li>
      <li>Keep all dependencies up to date</li>
      <li>Implement security training for development team</li>
    </ol>
  </div>
</body>
</html>`;
  }

  private generateMarkdownReport(report: any): string {
    let markdown = `# Security Audit Report

Generated: ${report.timestamp}

## Executive Summary

- **Average Security Score**: ${report.summary.averageScore}/100
- **Total Issues Found**: ${report.summary.totalIssues}
- **Critical Issues**: ${report.summary.criticalIssues}
- **High Issues**: ${report.summary.highIssues}
- **Apps Passed**: ${report.summary.passedApps}/${report.summary.totalApps}

---
`;

    for (const result of report.results) {
      markdown += `
## ${result.app}

**Score**: ${result.score}/100 | **Status**: ${result.passed ? '✅ PASSED' : '❌ FAILED'}

### Issue Summary
- Critical: ${result.summary.critical}
- High: ${result.summary.high}
- Medium: ${result.summary.medium}
- Low: ${result.summary.low}

`;

      if (result.issues.length > 0) {
        markdown += '### Issues\n\n';
        result.issues
          .sort((a: SecurityIssue, b: SecurityIssue) => {
            const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
            return severityOrder[a.severity] - severityOrder[b.severity];
          })
          .forEach((issue: SecurityIssue) => {
            markdown += `#### [${issue.severity.toUpperCase()}] ${issue.type}

**Description**: ${issue.description}
${issue.file ? `**File**: \`${issue.file}\`${issue.line ? ` (line ${issue.line})` : ''}` : ''}
**Recommendation**: ${issue.recommendation}

---
`;
          });
      } else {
        markdown += '✅ No security issues found!\n\n';
      }
    }

    markdown += `
## Recommendations

1. **Immediate Actions**:
   - Fix all critical severity issues
   - Update vulnerable dependencies
   - Remove hardcoded secrets

2. **Short-term Improvements**:
   - Implement CSP headers
   - Add input validation
   - Set up rate limiting

3. **Long-term Security Strategy**:
   - Regular security audits
   - Automated security scanning
   - Security training for developers
   - Penetration testing
`;

    return markdown;
  }
}

// Export for use in CI/CD
export { SecurityAuditor, SecurityIssue, SecurityAuditResult };

// Run if called directly
if (require.main === module) {
  const auditor = new SecurityAuditor();
  auditor.runFullAudit().catch(console.error);
}