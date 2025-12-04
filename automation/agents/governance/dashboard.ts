/**
 * Governance Dashboard
 * Real-time metrics and health scores for continuous governance monitoring
 */

import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface HealthMetrics {
  timestamp: Date;
  overallScore: number;
  categories: {
    typescript: CategoryScore;
    eslint: CategoryScore;
    tests: CategoryScore;
    security: CategoryScore;
    structure: CategoryScore;
  };
  trends: TrendData[];
}

export interface CategoryScore {
  score: number;
  status: 'pass' | 'warn' | 'fail';
  details: string;
  lastCheck: Date;
}

export interface TrendData {
  date: string;
  score: number;
  issues: number;
}

export interface DashboardConfig {
  workspacePath: string;
  historyPath: string;
  maxHistoryDays: number;
}

const DEFAULT_CONFIG: DashboardConfig = {
  workspacePath: process.cwd(),
  historyPath: '.archive/reports/governance',
  maxHistoryDays: 30,
};

/**
 * Governance Dashboard for continuous monitoring
 */
export class GovernanceDashboard {
  private config: DashboardConfig;
  private metrics: HealthMetrics | null = null;

  constructor(config: Partial<DashboardConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Collect all health metrics
   */
  public async collectMetrics(): Promise<HealthMetrics> {
    const [typescript, eslint, tests, security, structure] = await Promise.all([
      this.checkTypeScript(),
      this.checkESLint(),
      this.checkTests(),
      this.checkSecurity(),
      this.checkStructure(),
    ]);

    const scores = [typescript.score, eslint.score, tests.score, security.score, structure.score];
    const overallScore = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);

    this.metrics = {
      timestamp: new Date(),
      overallScore,
      categories: { typescript, eslint, tests, security, structure },
      trends: await this.loadTrends(),
    };

    await this.saveMetrics();
    return this.metrics;
  }

  private async checkTypeScript(): Promise<CategoryScore> {
    try {
      await execAsync('npx tsc --noEmit', { cwd: this.config.workspacePath });
      return { score: 100, status: 'pass', details: '0 TypeScript errors', lastCheck: new Date() };
    } catch (error) {
      const errorCount = (String(error).match(/error TS/g) || []).length;
      return {
        score: Math.max(0, 100 - errorCount * 10),
        status: errorCount > 0 ? 'fail' : 'pass',
        details: `${errorCount} TypeScript errors`,
        lastCheck: new Date(),
      };
    }
  }

  private async checkESLint(): Promise<CategoryScore> {
    try {
      const { stdout } = await execAsync('npm run lint 2>&1', { cwd: this.config.workspacePath });
      const errorMatch = stdout.match(/(\d+) errors?/);
      const warnMatch = stdout.match(/(\d+) warnings?/);
      const errors = errorMatch ? parseInt(errorMatch[1]) : 0;
      const warnings = warnMatch ? parseInt(warnMatch[1]) : 0;

      return {
        score: Math.max(0, 100 - errors * 10 - warnings * 2),
        status: errors > 0 ? 'fail' : warnings > 10 ? 'warn' : 'pass',
        details: `${errors} errors, ${warnings} warnings`,
        lastCheck: new Date(),
      };
    } catch {
      return { score: 100, status: 'pass', details: '0 errors, 0 warnings', lastCheck: new Date() };
    }
  }

  private async checkTests(): Promise<CategoryScore> {
    try {
      const { stdout } = await execAsync('npm run test:run 2>&1', {
        cwd: this.config.workspacePath,
      });
      const passMatch = stdout.match(/(\d+) passed/);
      const failMatch = stdout.match(/(\d+) failed/);
      const passed = passMatch ? parseInt(passMatch[1]) : 0;
      const failed = failMatch ? parseInt(failMatch[1]) : 0;
      const total = passed + failed;
      const passRate = total > 0 ? (passed / total) * 100 : 100;

      return {
        score: Math.round(passRate),
        status: failed > 0 ? 'fail' : 'pass',
        details: `${passed}/${total} tests passing`,
        lastCheck: new Date(),
      };
    } catch {
      return { score: 0, status: 'fail', details: 'Tests failed to run', lastCheck: new Date() };
    }
  }

  private async checkSecurity(): Promise<CategoryScore> {
    try {
      const { stdout } = await execAsync('npm audit --json 2>&1', {
        cwd: this.config.workspacePath,
      });
      const audit = JSON.parse(stdout);
      const vulns = audit.metadata?.vulnerabilities || {};
      const critical = vulns.critical || 0;
      const high = vulns.high || 0;
      const moderate = vulns.moderate || 0;

      const score = Math.max(0, 100 - critical * 30 - high * 20 - moderate * 5);
      return {
        score,
        status: critical > 0 || high > 0 ? 'fail' : moderate > 0 ? 'warn' : 'pass',
        details: `${critical} critical, ${high} high, ${moderate} moderate`,
        lastCheck: new Date(),
      };
    } catch {
      return { score: 100, status: 'pass', details: '0 vulnerabilities', lastCheck: new Date() };
    }
  }

  private async checkStructure(): Promise<CategoryScore> {
    const policyPath = path.join(
      this.config.workspacePath,
      '.metaHub/policies/root-structure.yaml'
    );
    if (!fs.existsSync(policyPath)) {
      return {
        score: 50,
        status: 'warn',
        details: 'No structure policy found',
        lastCheck: new Date(),
      };
    }

    // Check for files at root that shouldn't be there
    const rootFiles = fs.readdirSync(this.config.workspacePath);
    const allowedPatterns = [
      /^\./, // Hidden files/folders (e.g., .gitignore, .github/)
      /^package.*\.json$/, // package.json, package-lock.json
      /^tsconfig.*\.json$/, // tsconfig.json and variants
      /^README/i, // README.md and variants
      /^LICENSE/i, // LICENSE file
      /^CHANGELOG/i, // CHANGELOG.md
      /^CONTRIBUTING/i, // CONTRIBUTING.md
      /^SECURITY/i, // SECURITY.md
      /^GOVERNANCE/i, // GOVERNANCE.md
      /^CODEOWNERS$/i, // GitHub CODEOWNERS
      /^CLAUDE\.md$/i, // Claude AI instructions
      /^CODE_OF_CONDUCT/i, // Code of conduct
      /\.config\.(js|ts|mjs|cjs)$/, // Config files (eslint.config.js, vitest.config.ts)
      /^vitest\.config/, // Vitest configuration
      /^eslint/, // ESLint config and output
      /^mkdocs\.ya?ml$/, // MkDocs configuration
      /^node_modules$/, // Dependencies folder
      /^DEBUG_REPORT\.md$/, // Debug reports (temporary)
      /^CONSOLIDATION.*\.md$/i, // Consolidation documentation
    ];

    const violations = rootFiles.filter(
      (f) => !allowedPatterns.some((p) => p.test(f)) && !fs.statSync(f).isDirectory()
    );

    return {
      score: Math.max(0, 100 - violations.length * 5),
      status: violations.length > 5 ? 'fail' : violations.length > 0 ? 'warn' : 'pass',
      details: `${violations.length} structure violations`,
      lastCheck: new Date(),
    };
  }

  private async loadTrends(): Promise<TrendData[]> {
    const historyDir = path.join(this.config.workspacePath, this.config.historyPath);
    if (!fs.existsSync(historyDir)) return [];

    const files = fs
      .readdirSync(historyDir)
      .filter((f) => f.startsWith('governance-') && f.endsWith('.json'));

    const trends: TrendData[] = [];
    for (const file of files.slice(-this.config.maxHistoryDays)) {
      try {
        const content = JSON.parse(fs.readFileSync(path.join(historyDir, file), 'utf8'));
        const date = file.replace('governance-', '').replace('.json', '');
        trends.push({
          date,
          score: content.overallScore || 0,
          issues: content.findings?.length || 0,
        });
      } catch {
        // Skip invalid files
      }
    }
    return trends;
  }

  private async saveMetrics(): Promise<void> {
    if (!this.metrics) return;

    const historyDir = path.join(this.config.workspacePath, this.config.historyPath);
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    const filename = `dashboard-${new Date().toISOString().split('T')[0]}.json`;
    fs.writeFileSync(path.join(historyDir, filename), JSON.stringify(this.metrics, null, 2));
  }

  /**
   * Generate a text-based dashboard report
   */
  public generateReport(): string {
    if (!this.metrics) return 'No metrics collected. Run collectMetrics() first.';

    const m = this.metrics;
    const statusIcon = (s: string): string => (s === 'pass' ? '✅' : s === 'warn' ? '⚠️' : '❌');

    return `
╔══════════════════════════════════════════════════════════════╗
║              GOVERNANCE DASHBOARD                            ║
║              ${new Date().toISOString().split('T')[0]}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  OVERALL HEALTH SCORE: ${m.overallScore}/100                              ║
╠══════════════════════════════════════════════════════════════╣
║  Category        │ Score │ Status │ Details                  ║
╠──────────────────┼───────┼────────┼──────────────────────────╣
║  TypeScript      │ ${String(m.categories.typescript.score).padStart(3)}   │ ${statusIcon(m.categories.typescript.status)}     │ ${m.categories.typescript.details.padEnd(24)} ║
║  ESLint          │ ${String(m.categories.eslint.score).padStart(3)}   │ ${statusIcon(m.categories.eslint.status)}     │ ${m.categories.eslint.details.padEnd(24)} ║
║  Tests           │ ${String(m.categories.tests.score).padStart(3)}   │ ${statusIcon(m.categories.tests.status)}     │ ${m.categories.tests.details.padEnd(24)} ║
║  Security        │ ${String(m.categories.security.score).padStart(3)}   │ ${statusIcon(m.categories.security.status)}     │ ${m.categories.security.details.padEnd(24)} ║
║  Structure       │ ${String(m.categories.structure.score).padStart(3)}   │ ${statusIcon(m.categories.structure.status)}     │ ${m.categories.structure.details.padEnd(24)} ║
╚══════════════════════════════════════════════════════════════╝
`.trim();
  }

  /**
   * Get current metrics
   */
  public getMetrics(): HealthMetrics | null {
    return this.metrics;
  }
}

export function createGovernanceDashboard(config?: Partial<DashboardConfig>): GovernanceDashboard {
  return new GovernanceDashboard(config);
}
