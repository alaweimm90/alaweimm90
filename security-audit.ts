#!/usr/bin/env tsx
/**
 * Security Audit Runner
 *
 * This script performs comprehensive security audits across the repository:
 * - Dependency vulnerability scanning
 * - Secret detection
 * - Code security analysis
 * - Configuration security checks
 *
 * Usage: npm run security:audit
 */

import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface AuditResult {
  category: string;
  status: 'pass' | 'warn' | 'fail';
  message: string;
  details?: any;
}

class SecurityAuditor {
  private results: AuditResult[] = [];

  constructor(private rootDir: string = process.cwd()) {}

  async runFullAudit(): Promise<void> {
    console.log('🔒 Starting Security Audit...\n');

    await this.checkDependencies();
    await this.checkSecrets();
    await this.checkConfigurations();
    await this.checkCodeSecurity();
    await this.checkGitSecurity();
    await this.checkCI_CDSecurity();

    this.generateReport();
  }

  private async checkDependencies(): Promise<void> {
    console.log('📦 Checking dependencies for vulnerabilities...');

    try {
      // Check npm dependencies
      if (existsSync(join(this.rootDir, 'package.json'))) {
        const npmAudit = execSync('npm audit --json', {
          encoding: 'utf8',
          stdio: 'pipe'
        });
        const auditResult = JSON.parse(npmAudit);

        const vulnCount = auditResult.metadata?.vulnerabilities?.total || 0;
        if (vulnCount === 0) {
          this.results.push({
            category: 'Dependencies',
            status: 'pass',
            message: 'No npm vulnerabilities found'
          });
        } else {
          this.results.push({
            category: 'Dependencies',
            status: vulnCount > 5 ? 'fail' : 'warn',
            message: `${vulnCount} npm vulnerabilities found`,
            details: auditResult.vulnerabilities
          });
        }
      }

      // Check Python dependencies in key directories
      const pythonDirs = ['automation', 'alawein-technologies-llc/librex', 'research/maglogic'];
      for (const dir of pythonDirs) {
        if (existsSync(join(this.rootDir, dir, 'requirements.txt'))) {
          try {
            execSync(`pip-audit --requirement ${dir}/requirements.txt`, {
              encoding: 'utf8',
              stdio: 'pipe'
            });
            this.results.push({
              category: 'Dependencies',
              status: 'pass',
              message: `No Python vulnerabilities in ${dir}`
            });
          } catch (error: any) {
            this.results.push({
              category: 'Dependencies',
              status: 'warn',
              message: `Python vulnerabilities found in ${dir}`,
              details: error.stdout
            });
          }
        }
      }
    } catch (error: any) {
      this.results.push({
        category: 'Dependencies',
        status: 'fail',
        message: 'Failed to check dependencies',
        details: error.message
      });
    }
  }

  private async checkSecrets(): Promise<void> {
    console.log('🔍 Checking for leaked secrets...');

    try {
      // Run detect-secrets if available
      if (existsSync(join(this.rootDir, '.secrets.baseline'))) {
        const secretScan = execSync('detect-secrets scan --baseline .secrets.baseline --all-files', {
          encoding: 'utf8',
          stdio: 'pipe'
        });

        if (secretScan.includes('No secrets detected')) {
          this.results.push({
            category: 'Secrets',
            status: 'pass',
            message: 'No secrets detected in codebase'
          });
        } else {
          this.results.push({
            category: 'Secrets',
            status: 'warn',
            message: 'Potential secrets detected',
            details: secretScan
          });
        }
      } else {
        this.results.push({
          category: 'Secrets',
          status: 'warn',
          message: 'No secrets baseline configured'
        });
      }
    } catch (error: any) {
      this.results.push({
        category: 'Secrets',
        status: 'fail',
        message: 'Failed to scan for secrets',
        details: error.message
      });
    }
  }

  private async checkConfigurations(): Promise<void> {
    console.log('⚙️  Checking security configurations...');

    // Check for security headers in web apps
    const webApps = [
      'repz-llc/repz/vite.config.ts',
      'live-it-iconic-llc/liveiticonic/vite.config.ts'
    ];

    for (const app of webApps) {
      const configPath = join(this.rootDir, app);
      if (existsSync(configPath)) {
        const config = readFileSync(configPath, 'utf8');
        const hasSecurityHeaders = config.includes('X-Frame-Options') &&
                                  config.includes('Content-Security-Policy');

        this.results.push({
          category: 'Configuration',
          status: hasSecurityHeaders ? 'pass' : 'warn',
          message: `${app}: ${hasSecurityHeaders ? 'Security headers configured' : 'Missing security headers'}`
        });
      }
    }

    // Check for .github security configurations
    const securityConfigs = [
      '.github/dependabot.yml',
      '.github/secret_scanning.yml',
      'SECURITY.md'
    ];

    for (const config of securityConfigs) {
      const configPath = join(this.rootDir, config);
      this.results.push({
        category: 'Configuration',
        status: existsSync(configPath) ? 'pass' : 'warn',
        message: `${config}: ${existsSync(configPath) ? 'Present' : 'Missing'}`
      });
    }
  }

  private async checkCodeSecurity(): Promise<void> {
    console.log('🛡️  Checking code security practices...');

    // Check for ESLint security rules
    const eslintConfig = join(this.rootDir, 'eslint.config.enhanced.js');
    if (existsSync(eslintConfig)) {
      const config = readFileSync(eslintConfig, 'utf8');
      const hasSecurityRules = config.includes('@typescript-eslint/security') ||
                               config.includes('security') ||
                               config.includes('no-eval') ||
                               config.includes('no-implied-eval');

      this.results.push({
        category: 'Code Security',
        status: hasSecurityRules ? 'pass' : 'warn',
        message: `ESLint security rules: ${hasSecurityRules ? 'Configured' : 'Not configured'}`
      });
    }

    // Check for pre-commit security hooks
    const precommitConfig = join(this.rootDir, '.pre-commit-config.yaml');
    if (existsSync(precommitConfig)) {
      const config = readFileSync(precommitConfig, 'utf8');
      const hasSecurityHooks = config.includes('detect-secrets') ||
                               config.includes('bandit') ||
                               config.includes('security-audit');

      this.results.push({
        category: 'Code Security',
        status: hasSecurityHooks ? 'pass' : 'warn',
        message: `Pre-commit security hooks: ${hasSecurityHooks ? 'Configured' : 'Not configured'}`
      });
    }
  }

  private async checkGitSecurity(): Promise<void> {
    console.log('🔐 Checking Git security settings...');

    try {
      // Check for .gitignore presence
      const gitignorePath = join(this.rootDir, '.gitignore');
      const hasGitignore = existsSync(gitignorePath);

      this.results.push({
        category: 'Git Security',
        status: hasGitignore ? 'pass' : 'warn',
        message: `.gitignore: ${hasGitignore ? 'Present' : 'Missing'}`
      });

      // Check for sensitive files in git history (basic check)
      const sensitiveFiles = ['.env', '.env.local', 'id_rsa', 'private.key'];
      for (const file of sensitiveFiles) {
        try {
          execSync(`git log --all --full-history -- ${file}`, { stdio: 'pipe' });
          this.results.push({
            category: 'Git Security',
            status: 'fail',
            message: `Sensitive file ${file} found in git history`
          });
        } catch {
          // File not found in history - good
        }
      }
    } catch (error: any) {
      this.results.push({
        category: 'Git Security',
        status: 'warn',
        message: 'Failed to check Git security',
        details: error.message
      });
    }
  }

  private async checkCI_CDSecurity(): Promise<void> {
    console.log('🚀 Checking CI/CD security...');

    const ciConfigPath = join(this.rootDir, '.github/workflows/ci-cd-pipeline.yml');
    if (existsSync(ciConfigPath)) {
      const config = readFileSync(ciConfigPath, 'utf8');

      const hasSecurityScanning = config.includes('Snyk') ||
                                  config.includes('Trivy') ||
                                  config.includes('CodeQL');

      const hasPermissionControls = config.includes('permissions:') ||
                                    config.includes('GITHUB_TOKEN');

      this.results.push({
        category: 'CI/CD Security',
        status: hasSecurityScanning ? 'pass' : 'warn',
        message: `Security scanning: ${hasSecurityScanning ? 'Configured' : 'Not configured'}`
      });

      this.results.push({
        category: 'CI/CD Security',
        status: hasPermissionControls ? 'pass' : 'warn',
        message: `Permission controls: ${hasPermissionControls ? 'Configured' : 'Not configured'}`
      });
    }
  }

  private generateReport(): void {
    console.log('\n📊 SECURITY AUDIT REPORT\n');
    console.log('='.repeat(50));

    const passed = this.results.filter(r => r.status === 'pass').length;
    const warned = this.results.filter(r => r.status === 'warn').length;
    const failed = this.results.filter(r => r.status === 'fail').length;

    console.log(`✅ Passed: ${passed}`);
    console.log(`⚠️  Warnings: ${warned}`);
    console.log(`❌ Failed: ${failed}\n`);

    // Group results by category
    const grouped = this.results.reduce((acc, result) => {
      if (!acc[result.category]) acc[result.category] = [];
      acc[result.category].push(result);
      return acc;
    }, {} as Record<string, AuditResult[]>);

    Object.entries(grouped).forEach(([category, results]) => {
      console.log(`\n📂 ${category}:`);
      results.forEach(result => {
        const icon = result.status === 'pass' ? '✅' : result.status === 'warn' ? '⚠️' : '❌';
        console.log(`   ${icon} ${result.message}`);
      });
    });

    // Overall assessment
    console.log('\n' + '='.repeat(50));
    if (failed === 0 && warned <= 2) {
      console.log('🎉 SECURITY AUDIT: EXCELLENT');
      process.exit(0);
    } else if (failed === 0) {
      console.log('👍 SECURITY AUDIT: GOOD');
      process.exit(0);
    } else if (failed <= 2) {
      console.log('⚠️  SECURITY AUDIT: NEEDS ATTENTION');
      process.exit(1);
    } else {
      console.log('🚨 SECURITY AUDIT: CRITICAL ISSUES');
      process.exit(2);
    }
  }
}

// Run the audit if this script is executed directly
if (require.main === module) {
  const auditor = new SecurityAuditor();
  auditor.runFullAudit().catch(error => {
    console.error('Audit failed:', error);
    process.exit(3);
  });
}

export { SecurityAuditor };
