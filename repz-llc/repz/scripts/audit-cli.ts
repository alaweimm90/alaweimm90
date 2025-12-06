#!/usr/bin/env node

/**
 * 🧠 Enterprise Codebase Audit CLI
 * Automated health monitoring and cleanup recommendations
 */

import { promises as fs } from 'fs';
import { join, extname } from 'path';
import { execSync } from 'child_process';

interface AuditResult {
  category: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  issue: string;
  file?: string;
  line?: number;
  recommendation: string;
}

interface HealthScore {
  overall: number;
  categories: {
    orphanedFiles: number;
    themeCompliance: number;
    componentDuplication: number;
    routeConsistency: number;
    businessLogic: number;
  };
}

class CodebaseAuditor {
  private results: AuditResult[] = [];
  private basePath: string;

  constructor(basePath: string = process.cwd()) {
    this.basePath = basePath;
  }

  async runFullAudit(): Promise<{ results: AuditResult[]; score: HealthScore }> {
    console.log('🔍 Starting comprehensive codebase audit...\n');

    await this.auditOrphanedFiles();
    await this.auditThemeCompliance();
    await this.auditComponentDuplication();
    await this.auditRouteConsistency();
    await this.auditBusinessLogic();

    const score = this.calculateHealthScore();
    
    console.log('\n📊 Audit Complete!');
    console.log(`Overall Health Score: ${score.overall}/100`);
    
    return { results: this.results, score };
  }

  private async auditOrphanedFiles(): Promise<void> {
    console.log('📁 Auditing orphaned files...');

    try {
      // Check for files not referenced in main app
      const srcFiles = await this.getAllFiles('src', ['.tsx', '.ts', '.jsx', '.js']);
      const appContent = await fs.readFile(join(this.basePath, 'src/App.tsx'), 'utf-8');
      
      const referencedFiles = new Set<string>();
      const importRegex = /import.*from\s+['"](.*)['"]/g;
      let match;
      
      while ((match = importRegex.exec(appContent)) !== null) {
        referencedFiles.add(match[1]);
      }

      // Check demo pages (should be archived)
      const demoFiles = await this.getAllFiles('demo-pages', ['.tsx', '.ts']);
      for (const file of demoFiles) {
        this.addResult({
          category: 'Orphaned Files',
          severity: 'MEDIUM',
          issue: 'Demo page should be archived',
          file,
          recommendation: `Move ${file} to archive/demo-pages/`
        });
      }

      // Check for orphaned pricing files
      const pricingFiles = await this.getAllFiles('src/pages/pricing', ['.tsx', '.ts']);
      for (const file of pricingFiles) {
        this.addResult({
          category: 'Orphaned Files',
          severity: 'HIGH',
          issue: 'Orphaned pricing page found',
          file,
          recommendation: `Integrate ${file} into unified pricing system or archive`
        });
      }

    } catch (error) {
      console.warn('Warning: Could not complete orphaned files audit:', error);
    }
  }

  private async auditThemeCompliance(): Promise<void> {
    console.log('🎨 Auditing theme compliance...');

    const violations = [
      'text-white',
      'bg-white', 
      'text-black',
      'bg-black',
      'text-red-',
      'bg-red-',
      'text-blue-',
      'bg-blue-'
    ];

    try {
      const componentFiles = await this.getAllFiles('src/components', ['.tsx', '.ts']);
      const pageFiles = await this.getAllFiles('src/pages', ['.tsx', '.ts']);
      const allFiles = [...componentFiles, ...pageFiles];

      for (const file of allFiles) {
        const content = await fs.readFile(file, 'utf-8');
        
        for (const violation of violations) {
          const regex = new RegExp(`className=.*${violation}`, 'g');
          const matches = content.match(regex);
          
          if (matches) {
            this.addResult({
              category: 'Theme Compliance',
              severity: 'MEDIUM',
              issue: `Direct color usage: ${violation}`,
              file,
              recommendation: 'Use semantic tokens from theme provider instead'
            });
          }
        }
      }
    } catch (error) {
      console.warn('Warning: Could not complete theme audit:', error);
    }
  }

  private async auditComponentDuplication(): Promise<void> {
    console.log('🔄 Auditing component duplication...');

    const suspiciousPatterns = [
      'PricingCard',
      'TierCard',
      'AuthModal',
      'LoginForm',
      'SignupForm'
    ];

    try {
      const componentFiles = await this.getAllFiles('src/components', ['.tsx', '.ts']);
      const componentsByPattern: Record<string, string[]> = {};

      for (const pattern of suspiciousPatterns) {
        componentsByPattern[pattern] = [];
        
        for (const file of componentFiles) {
          const content = await fs.readFile(file, 'utf-8');
          if (content.includes(pattern)) {
            componentsByPattern[pattern].push(file);
          }
        }

        if (componentsByPattern[pattern].length > 1) {
          this.addResult({
            category: 'Component Duplication',
            severity: 'HIGH',
            issue: `Multiple ${pattern} implementations found`,
            recommendation: `Consolidate ${pattern} components into single reusable component`
          });
        }
      }
    } catch (error) {
      console.warn('Warning: Could not complete duplication audit:', error);
    }
  }

  private async auditRouteConsistency(): Promise<void> {
    console.log('🗺️ Auditing route consistency...');

    try {
      const appContent = await fs.readFile(join(this.basePath, 'src/App.tsx'), 'utf-8');
      
      // Count legacy redirects
      const legacyRedirects = (appContent.match(/Navigate to=".*" replace/g) || []).length;
      
      if (legacyRedirects > 5) {
        this.addResult({
          category: 'Route Consistency',
          severity: 'MEDIUM',
          issue: `${legacyRedirects} legacy redirects found`,
          file: 'src/App.tsx',
          recommendation: 'Replace with proper 301 redirects in hosting config'
        });
      }

      // Check for hardcoded routes
      const componentFiles = await this.getAllFiles('src/components', ['.tsx', '.ts']);
      for (const file of componentFiles) {
        const content = await fs.readFile(file, 'utf-8');
        const hardcodedRoutes = content.match(/["'](\/\w+)+["']/g);
        
        if (hardcodedRoutes && hardcodedRoutes.length > 2) {
          this.addResult({
            category: 'Route Consistency',
            severity: 'LOW',
            issue: 'Hardcoded routes found',
            file,
            recommendation: 'Use centralized route constants'
          });
        }
      }
    } catch (error) {
      console.warn('Warning: Could not complete route audit:', error);
    }
  }

  private async auditBusinessLogic(): Promise<void> {
    console.log('💼 Auditing business logic centralization...');

    const businessLogicPatterns = [
      'subscription_tier',
      'tier_features',
      'pricing',
      'auth\\.uid',
      'stripe'
    ];

    try {
      const allFiles = await this.getAllFiles('src', ['.tsx', '.ts']);
      const logicDistribution: Record<string, string[]> = {};

      for (const pattern of businessLogicPatterns) {
        logicDistribution[pattern] = [];
        const regex = new RegExp(pattern, 'gi');

        for (const file of allFiles) {
          const content = await fs.readFile(file, 'utf-8');
          if (regex.test(content)) {
            logicDistribution[pattern].push(file);
          }
        }

        if (logicDistribution[pattern].length > 5) {
          this.addResult({
            category: 'Business Logic',
            severity: 'MEDIUM',
            issue: `${pattern} logic scattered across ${logicDistribution[pattern].length} files`,
            recommendation: `Centralize ${pattern} logic into dedicated service/hook`
          });
        }
      }
    } catch (error) {
      console.warn('Warning: Could not complete business logic audit:', error);
    }
  }

  private async getAllFiles(dir: string, extensions: string[]): Promise<string[]> {
    const fullDir = join(this.basePath, dir);
    
    try {
      await fs.access(fullDir);
    } catch {
      return []; // Directory doesn't exist
    }

    const files: string[] = [];
    const items = await fs.readdir(fullDir, { withFileTypes: true });

    for (const item of items) {
      const fullPath = join(fullDir, item.name);
      
      if (item.isDirectory()) {
        const subFiles = await this.getAllFiles(join(dir, item.name), extensions);
        files.push(...subFiles);
      } else if (extensions.includes(extname(item.name))) {
        files.push(fullPath);
      }
    }

    return files;
  }

  private addResult(result: AuditResult): void {
    this.results.push(result);
  }

  private calculateHealthScore(): HealthScore {
    const severityWeights = { LOW: 1, MEDIUM: 3, HIGH: 5, CRITICAL: 10 };
    const maxDeductions = { orphanedFiles: 20, themeCompliance: 25, componentDuplication: 20, routeConsistency: 15, businessLogic: 20 };
    
    const categoryDeductions = {
      orphanedFiles: 0,
      themeCompliance: 0,
      componentDuplication: 0,
      routeConsistency: 0,
      businessLogic: 0
    };

    for (const result of this.results) {
      const deduction = severityWeights[result.severity];
      
      switch (result.category) {
        case 'Orphaned Files':
          categoryDeductions.orphanedFiles += deduction;
          break;
        case 'Theme Compliance':
          categoryDeductions.themeCompliance += deduction;
          break;
        case 'Component Duplication':
          categoryDeductions.componentDuplication += deduction;
          break;
        case 'Route Consistency':
          categoryDeductions.routeConsistency += deduction;
          break;
        case 'Business Logic':
          categoryDeductions.businessLogic += deduction;
          break;
      }
    }

    const categories = {
      orphanedFiles: Math.max(0, 100 - Math.min(categoryDeductions.orphanedFiles, maxDeductions.orphanedFiles)),
      themeCompliance: Math.max(0, 100 - Math.min(categoryDeductions.themeCompliance, maxDeductions.themeCompliance)),
      componentDuplication: Math.max(0, 100 - Math.min(categoryDeductions.componentDuplication, maxDeductions.componentDuplication)),
      routeConsistency: Math.max(0, 100 - Math.min(categoryDeductions.routeConsistency, maxDeductions.routeConsistency)),
      businessLogic: Math.max(0, 100 - Math.min(categoryDeductions.businessLogic, maxDeductions.businessLogic))
    };

    const overall = Math.round(Object.values(categories).reduce((sum, score) => sum + score, 0) / 5);

    return { overall, categories };
  }

  async generateReport(format: 'json' | 'markdown' | 'html' = 'markdown'): Promise<string> {
    const { results, score } = await this.runFullAudit();

    switch (format) {
      case 'json':
        return JSON.stringify({ results, score }, null, 2);
      
      case 'html':
        return this.generateHTMLReport(results, score);
      
      default:
        return this.generateMarkdownReport(results, score);
    }
  }

  private generateMarkdownReport(results: AuditResult[], score: HealthScore): string {
    const critical = results.filter(r => r.severity === 'CRITICAL');
    const high = results.filter(r => r.severity === 'HIGH');
    const medium = results.filter(r => r.severity === 'MEDIUM');
    const low = results.filter(r => r.severity === 'LOW');

    return `# Codebase Audit Report

## Health Score: ${score.overall}/100

### Category Breakdown:
- Orphaned Files: ${score.categories.orphanedFiles}/100
- Theme Compliance: ${score.categories.themeCompliance}/100  
- Component Duplication: ${score.categories.componentDuplication}/100
- Route Consistency: ${score.categories.routeConsistency}/100
- Business Logic: ${score.categories.businessLogic}/100

## Issues Found

### Critical (${critical.length})
${critical.map(r => `- **${r.issue}** in ${r.file || 'multiple files'}\n  *${r.recommendation}*`).join('\n')}

### High Priority (${high.length})
${high.map(r => `- **${r.issue}** in ${r.file || 'multiple files'}\n  *${r.recommendation}*`).join('\n')}

### Medium Priority (${medium.length})
${medium.map(r => `- **${r.issue}** in ${r.file || 'multiple files'}\n  *${r.recommendation}*`).join('\n')}

### Low Priority (${low.length})
${low.map(r => `- **${r.issue}** in ${r.file || 'multiple files'}\n  *${r.recommendation}*`).join('\n')}

---
*Generated on ${new Date().toISOString()}*
`;
  }

  private generateHTMLReport(results: AuditResult[], score: HealthScore): string {
    return `<!DOCTYPE html>
<html>
<head>
    <title>Codebase Audit Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; }
        .score { font-size: 2rem; font-weight: bold; color: ${score.overall >= 90 ? '#22c55e' : score.overall >= 70 ? '#f59e0b' : '#ef4444'}; }
        .category { margin: 1rem 0; padding: 1rem; border-radius: 8px; background: #f8fafc; }
        .issue { margin: 0.5rem 0; padding: 0.5rem; border-left: 4px solid #e2e8f0; }
        .critical { border-left-color: #ef4444; }
        .high { border-left-color: #f59e0b; }
        .medium { border-left-color: #3b82f6; }
        .low { border-left-color: #6b7280; }
    </style>
</head>
<body>
    <h1>Codebase Audit Report</h1>
    <div class="score">Health Score: ${score.overall}/100</div>
    
    <h2>Category Breakdown</h2>
    <div class="category">
        <h3>Orphaned Files: ${score.categories.orphanedFiles}/100</h3>
        <h3>Theme Compliance: ${score.categories.themeCompliance}/100</h3>
        <h3>Component Duplication: ${score.categories.componentDuplication}/100</h3>
        <h3>Route Consistency: ${score.categories.routeConsistency}/100</h3>
        <h3>Business Logic: ${score.categories.businessLogic}/100</h3>
    </div>

    <h2>Issues Found (${results.length})</h2>
    ${results.map(r => `
        <div class="issue ${r.severity.toLowerCase()}">
            <strong>${r.severity}:</strong> ${r.issue}
            ${r.file ? `<br><em>File: ${r.file}</em>` : ''}
            <br><small>${r.recommendation}</small>
        </div>
    `).join('')}
    
    <footer>
        <p><em>Generated on ${new Date().toISOString()}</em></p>
    </footer>
</body>
</html>`;
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';

  const auditor = new CodebaseAuditor();

  switch (command) {
    case 'full':
      console.log('Running full audit...');
      const report = await auditor.generateReport('markdown');
      await fs.writeFile('audit-report.md', report);
      console.log('Report saved to audit-report.md');
      break;

    case 'json':
      const jsonReport = await auditor.generateReport('json');
      await fs.writeFile('audit-report.json', jsonReport);
      console.log('JSON report saved to audit-report.json');
      break;

    case 'html':
      const htmlReport = await auditor.generateReport('html');
      await fs.writeFile('audit-report.html', htmlReport);
      console.log('HTML report saved to audit-report.html');
      break;

    case 'score':
      const { score } = await auditor.runFullAudit();
      console.log(`Health Score: ${score.overall}/100`);
      process.exit(score.overall >= 80 ? 0 : 1); // Exit code for CI
      break;

    case 'precommit':
      const { results } = await auditor.runFullAudit();
      const blocking = results.filter(r => r.severity === 'CRITICAL');
      if (blocking.length > 0) {
        console.error(`❌ ${blocking.length} critical issues found. Commit blocked.`);
        blocking.forEach(issue => console.error(`- ${issue.issue}`));
        process.exit(1);
      }
      console.log('✅ Pre-commit audit passed');
      break;

    default:
      console.log(`
🧠 Codebase Audit CLI

Usage:
  npm run audit:full      # Generate full markdown report
  npm run audit:json      # Generate JSON report  
  npm run audit:html      # Generate HTML dashboard
  npm run audit:score     # Get health score (CI-friendly)
  npm run audit:precommit # Pre-commit hook validation

Examples:
  npx ts-node scripts/audit-cli.ts full
  npx ts-node scripts/audit-cli.ts score
`);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { CodebaseAuditor };