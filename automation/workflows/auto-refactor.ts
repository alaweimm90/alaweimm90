/**
 * Automated Refactoring Pipeline
 * Identifies and applies safe refactoring operations with validation
 */

import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface RefactoringOpportunity {
  id: string;
  type: RefactoringType;
  file: string;
  line?: number;
  severity: 'high' | 'medium' | 'low';
  description: string;
  estimatedImpact: string;
  autoFixable: boolean;
}

export type RefactoringType =
  | 'large-file'
  | 'complex-function'
  | 'duplicate-code'
  | 'missing-types'
  | 'dead-code'
  | 'long-function';

export interface RefactoringResult {
  opportunityId: string;
  success: boolean;
  changesMade: string[];
  validationPassed: boolean;
  error?: string;
}

export interface PipelineConfig {
  workspacePath: string;
  maxFileLines: number;
  maxFunctionComplexity: number;
  maxFunctionLines: number;
  dryRun: boolean;
  validateAfterEach: boolean;
}

const DEFAULT_CONFIG: PipelineConfig = {
  workspacePath: process.cwd(),
  maxFileLines: 500,
  maxFunctionComplexity: 10,
  maxFunctionLines: 50,
  dryRun: true,
  validateAfterEach: true,
};

/**
 * Automated Refactoring Pipeline
 */
export class AutoRefactorPipeline {
  private config: PipelineConfig;
  private opportunities: RefactoringOpportunity[] = [];
  private results: RefactoringResult[] = [];

  constructor(config: Partial<PipelineConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Scan for refactoring opportunities
   */
  public async scan(): Promise<RefactoringOpportunity[]> {
    this.opportunities = [];

    // Scan for large files
    await this.scanLargeFiles();

    // Scan for complex functions (using ESLint complexity)
    await this.scanComplexFunctions();

    // Scan for missing types
    await this.scanMissingTypes();

    // Scan for long functions
    await this.scanLongFunctions();

    return this.opportunities;
  }

  private async scanLargeFiles(): Promise<void> {
    const toolsDir = path.join(this.config.workspacePath, 'tools');
    const automationDir = path.join(this.config.workspacePath, 'automation');

    for (const dir of [toolsDir, automationDir]) {
      if (!fs.existsSync(dir)) continue;

      const files = this.getFilesRecursive(dir, ['.ts', '.js', '.py']);
      for (const file of files) {
        const lines = fs.readFileSync(file, 'utf8').split('\n').length;
        if (lines > this.config.maxFileLines) {
          this.opportunities.push({
            id: `large-file-${this.opportunities.length}`,
            type: 'large-file',
            file,
            severity: lines > 800 ? 'high' : lines > 600 ? 'medium' : 'low',
            description: `File has ${lines} lines (max: ${this.config.maxFileLines})`,
            estimatedImpact: 'Reduced cognitive load, improved maintainability',
            autoFixable: false,
          });
        }
      }
    }
  }

  private async scanComplexFunctions(): Promise<void> {
    try {
      const { stdout } = await execAsync(
        'npx eslint . --format json --rule "complexity: [warn, 10]" 2>/dev/null || true',
        { cwd: this.config.workspacePath }
      );

      if (stdout) {
        try {
          const results = JSON.parse(stdout);
          for (const result of results) {
            const complexityMessages = result.messages?.filter(
              (m: { ruleId: string }) => m.ruleId === 'complexity'
            );
            for (const msg of complexityMessages || []) {
              this.opportunities.push({
                id: `complex-fn-${this.opportunities.length}`,
                type: 'complex-function',
                file: result.filePath,
                line: msg.line,
                severity: 'medium',
                description: msg.message,
                estimatedImpact: 'Improved testability and readability',
                autoFixable: false,
              });
            }
          }
        } catch {
          // JSON parse error, skip
        }
      }
    } catch {
      // ESLint not available or failed
    }
  }

  private async scanMissingTypes(): Promise<void> {
    try {
      const { stdout } = await execAsync('npm run lint 2>&1 | grep "Missing return type"', {
        cwd: this.config.workspacePath,
      });

      const lines = stdout.split('\n').filter((l) => l.trim());
      for (const line of lines) {
        const match = line.match(/(.+):(\d+):/);
        if (match) {
          this.opportunities.push({
            id: `missing-type-${this.opportunities.length}`,
            type: 'missing-types',
            file: match[1],
            line: parseInt(match[2]),
            severity: 'low',
            description: 'Missing return type annotation',
            estimatedImpact: 'Better type safety and documentation',
            autoFixable: true,
          });
        }
      }
    } catch {
      // Grep returned no matches or lint failed
    }
  }

  private async scanLongFunctions(): Promise<void> {
    const toolsDir = path.join(this.config.workspacePath, 'tools');
    if (!fs.existsSync(toolsDir)) return;

    const files = this.getFilesRecursive(toolsDir, ['.ts']);
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const functionMatches = content.matchAll(
        /(async\s+)?function\s+(\w+)|(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*=>/g
      );

      for (const match of functionMatches) {
        // Simplified: count lines until next function or end
        const startIndex = match.index ?? 0;
        const nextMatch = content
          .slice(startIndex + 10)
          .search(/\n(async\s+)?function\s+|\n\w+\s*=\s*/);
        const endIndex = nextMatch > 0 ? startIndex + 10 + nextMatch : content.length;
        const functionLines = content.slice(startIndex, endIndex).split('\n').length;

        if (functionLines > this.config.maxFunctionLines) {
          this.opportunities.push({
            id: `long-fn-${this.opportunities.length}`,
            type: 'long-function',
            file,
            line: content.slice(0, startIndex).split('\n').length,
            severity: functionLines > 100 ? 'high' : 'medium',
            description: `Function has ${functionLines} lines (max: ${this.config.maxFunctionLines})`,
            estimatedImpact: 'Improved readability and testability',
            autoFixable: false,
          });
        }
      }
    }
  }

  private getFilesRecursive(dir: string, extensions: string[]): string[] {
    const files: string[] = [];

    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;

      if (entry.isDirectory()) {
        files.push(...this.getFilesRecursive(fullPath, extensions));
      } else if (extensions.some((ext) => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }

    return files;
  }

  /**
   * Run validation checks
   */
  public async validate(): Promise<{ passed: boolean; message: string }> {
    try {
      // TypeScript compilation
      await execAsync('npx tsc --noEmit', { cwd: this.config.workspacePath });

      // ESLint
      await execAsync('npm run lint', { cwd: this.config.workspacePath });

      // Tests
      await execAsync('npm run test:run', { cwd: this.config.workspacePath });

      return { passed: true, message: 'All validation checks passed' };
    } catch (error) {
      return { passed: false, message: `Validation failed: ${error}` };
    }
  }

  /**
   * Generate summary report
   */
  public generateReport(): {
    totalOpportunities: number;
    byType: Record<RefactoringType, number>;
    bySeverity: Record<string, number>;
    autoFixable: number;
  } {
    const byType: Record<string, number> = {};
    const bySeverity: Record<string, number> = {};

    for (const opp of this.opportunities) {
      byType[opp.type] = (byType[opp.type] || 0) + 1;
      bySeverity[opp.severity] = (bySeverity[opp.severity] || 0) + 1;
    }

    return {
      totalOpportunities: this.opportunities.length,
      byType: byType as Record<RefactoringType, number>,
      bySeverity,
      autoFixable: this.opportunities.filter((o) => o.autoFixable).length,
    };
  }

  /**
   * Get all opportunities
   */
  public getOpportunities(): RefactoringOpportunity[] {
    return this.opportunities;
  }

  /**
   * Get results
   */
  public getResults(): RefactoringResult[] {
    return this.results;
  }
}

export function createAutoRefactorPipeline(config?: Partial<PipelineConfig>): AutoRefactorPipeline {
  return new AutoRefactorPipeline(config);
}
