/**
 * AI-Powered Test Generator
 *
 * Automatically generates comprehensive test suites with intelligent
 * coverage analysis and optimization using ATLAS orchestration.
 */

import { glob } from 'glob';
import * as fs from 'fs/promises';
import * as path from 'path';
import { ATLASIntegration } from '../../core/atlas-integration.js';
import { ToolResult } from '../../core/types.js';
import {
  TestGenerationOptions,
  TestSuite,
  TestFile,
  TestAnalysisResult,
  CoverageGap,
  TestRecommendation
} from './types.js';

export class TestGenerator {
  private atlas: ATLASIntegration;
  private config: Partial<TestGenerationOptions>;

  constructor(options: { atlas: ATLASIntegration } & Partial<TestGenerationOptions>) {
    this.atlas = options.atlas;
    this.config = options;
  }

  /**
   * Generate comprehensive test suite for a codebase
   */
  async generate(options: TestGenerationOptions): Promise<ToolResult<TestSuite>> {
    const startTime = Date.now();

    try {
      // Analyze existing codebase
      const analysis = await this.analyze(options);

      // Generate test files
      const testFiles = await this.generateTestFiles(analysis, options);

      // Create test suite
      const testSuite: TestSuite = {
        id: `test-suite-${Date.now()}`,
        name: `${options.framework} Test Suite`,
        framework: options.framework,
        language: options.language,
        files: testFiles,
        coverage: await this.calculateCoverage(testFiles),
        metadata: {
          generatedAt: new Date(),
          generator: 'ai-tools-test-generator',
          version: '1.0.0'
        }
      };

      // Write test files
      await this.writeTestFiles(testSuite, options);

      const duration = Date.now() - startTime;

      // Record telemetry
      await this.atlas.recordTelemetry({
        eventId: `test-gen-${testSuite.id}`,
        eventType: 'test_generation',
        timestamp: new Date(),
        tool: 'test-generator',
        action: 'generate',
        metadata: {
          framework: options.framework,
          language: options.language,
          filesGenerated: testFiles.length,
          estimatedCoverage: testSuite.coverage.overall
        },
        metrics: {
          duration_ms: duration,
          quality_score: testSuite.coverage.overall
        }
      });

      return {
        success: true,
        data: testSuite,
        metadata: {
          duration_ms: duration,
          agent_id: 'test-generator'
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  /**
   * Analyze codebase to identify testing needs
   */
  async analyze(options: TestGenerationOptions): Promise<TestAnalysisResult> {
    const sourceFiles = await this.findSourceFiles(options);
    const existingTests = await this.findExistingTests(options);
    const coverageGaps = await this.identifyCoverageGaps(sourceFiles, existingTests, options);
    const recommendations = this.generateRecommendations(coverageGaps, options);

    return {
      path: options.output?.directory || './tests',
      existingTests,
      coverageGaps,
      recommendations,
      estimatedCoverage: this.estimateCoverage(coverageGaps)
    };
  }

  /**
   * Generate test files based on analysis
   */
  private async generateTestFiles(
    analysis: TestAnalysisResult,
    options: TestGenerationOptions
  ): Promise<TestFile[]> {
    const testFiles: TestFile[] = [];

    // Generate tests for coverage gaps
    for (const gap of analysis.coverageGaps) {
      if (gap.priority === 'high' || gap.priority === 'critical') {
        const testFile = await this.generateTestForGap(gap, options);
        if (testFile) {
          testFiles.push(testFile);
        }
      }
    }

    // Generate additional tests based on recommendations
    for (const rec of analysis.recommendations.slice(0, 10)) { // Limit to top 10
      if (rec.type.startsWith('add-')) {
        const testFile = await this.generateTestForRecommendation(rec, options);
        if (testFile) {
          testFiles.push(testFile);
        }
      }
    }

    return testFiles;
  }

  /**
   * Generate test for a specific coverage gap
   */
  private async generateTestForGap(
    gap: CoverageGap,
    options: TestGenerationOptions
  ): Promise<TestFile | null> {
    // Use ATLAS to generate test code
    const task = await this.atlas.submitTask(
      'code_generation',
      `Generate comprehensive unit tests for ${gap.function} in ${gap.file} using ${options.framework}`,
      {
        repository: process.cwd(),
        files: [gap.file],
        language: options.language
      },
      {
        requiredCapabilities: ['code_generation', 'testing'],
        priority: gap.priority === 'critical' ? 'high' : 'medium'
      }
    );

    const result = await this.atlas.waitForTask(task.task_id);

    if (!result.success) {
      return null;
    }

    const testPath = this.generateTestFilePath(gap.file, options);

    return {
      path: testPath,
      sourcePath: gap.file,
      content: result.result.code,
      tests: this.parseTestCases(result.result.code, options.framework),
      coverage: { statements: 0, branches: 0, functions: 0, lines: 0, overall: 0 } // Will be calculated later
    };
  }

  /**
   * Generate test for a recommendation
   */
  private async generateTestForRecommendation(
    rec: TestRecommendation,
    options: TestGenerationOptions
  ): Promise<TestFile | null> {
    const testType = rec.type.replace('add-', '') as 'unit' | 'integration' | 'e2e';

    const task = await this.atlas.submitTask(
      'code_generation',
      `Generate ${testType} tests for ${rec.file}: ${rec.description}`,
      {
        repository: process.cwd(),
        files: [rec.file],
        language: options.language
      },
      {
        requiredCapabilities: ['code_generation', 'testing'],
        priority: rec.priority === 'critical' ? 'high' : 'medium'
      }
    );

    const result = await this.atlas.waitForTask(task.task_id);

    if (!result.success) {
      return null;
    }

    const testPath = this.generateTestFilePath(rec.file, options, testType);

    return {
      path: testPath,
      sourcePath: rec.file,
      content: result.result.code,
      tests: this.parseTestCases(result.result.code, options.framework),
      coverage: { statements: 0, branches: 0, functions: 0, lines: 0, overall: 0 }
    };
  }

  /**
   * Write generated test files to disk
   */
  private async writeTestFiles(
    testSuite: TestSuite,
    options: TestGenerationOptions
  ): Promise<void> {
    const outputDir = options.output?.directory || './tests';

    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    for (const testFile of testSuite.files) {
      const fullPath = path.join(outputDir, testFile.path);

      // Create subdirectories if needed
      await fs.mkdir(path.dirname(fullPath), { recursive: true });

      // Write file (only if overwrite is true or file doesn't exist)
      if (options.output?.overwrite || !await this.fileExists(fullPath)) {
        await fs.writeFile(fullPath, testFile.content, 'utf-8');
      }
    }
  }

  /**
   * Find source files to test
   */
  private async findSourceFiles(options: TestGenerationOptions): Promise<string[]> {
    const patterns = options.patterns?.include || [
      `**/*.{${this.getExtensionsForLanguage(options.language)}}`
    ];

    const excludePatterns = options.patterns?.exclude || [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/*.test.*',
      '**/*.spec.*'
    ];

    const files: string[] = [];
    for (const pattern of patterns) {
      const matches = await glob(pattern, { ignore: excludePatterns });
      files.push(...matches);
    }

    return [...new Set(files)]; // Remove duplicates
  }

  /**
   * Find existing test files
   */
  private async findExistingTests(options: TestGenerationOptions): Promise<TestFile[]> {
    const testPatterns = this.getTestPatterns(options.framework);
    const testFiles: TestFile[] = [];

    for (const pattern of testPatterns) {
      const matches = await glob(pattern);
      for (const match of matches) {
        try {
          const content = await fs.readFile(match, 'utf-8');
          testFiles.push({
            path: match,
            sourcePath: this.inferSourcePath(match, options),
            content,
            tests: this.parseTestCases(content, options.framework),
            coverage: { statements: 0, branches: 0, functions: 0, lines: 0, overall: 0 }
          });
        } catch (error) {
          // Skip files that can't be read
        }
      }
    }

    return testFiles;
  }

  /**
   * Identify coverage gaps
   */
  private async identifyCoverageGaps(
    sourceFiles: string[],
    existingTests: TestFile[],
    options: TestGenerationOptions
  ): Promise<CoverageGap[]> {
    const gaps: CoverageGap[] = [];

    for (const sourceFile of sourceFiles) {
      try {
        const content = await fs.readFile(sourceFile, 'utf-8');
        const functions = this.extractFunctions(content, options.language);

        for (const func of functions) {
          const hasTest = existingTests.some(test =>
            test.tests.some(t => t.dependencies?.includes(func))
          );

          if (!hasTest) {
            gaps.push({
              file: sourceFile,
              function: func,
              type: 'uncovered',
              priority: this.calculatePriority(func, content),
              reason: `Function ${func} has no test coverage`
            });
          }
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }

    return gaps;
  }

  /**
   * Generate test recommendations
   */
  private generateRecommendations(
    gaps: CoverageGap[],
    options: TestGenerationOptions
  ): TestRecommendation[] {
    const recommendations: TestRecommendation[] = [];

    for (const gap of gaps) {
      recommendations.push({
        type: 'add-unit-test',
        file: gap.file,
        description: `Add unit test for ${gap.function}`,
        priority: gap.priority,
        estimatedEffort: gap.priority === 'critical' ? 2 : gap.priority === 'high' ? 1 : 0.5,
        expectedCoverage: 80
      });
    }

    // Sort by priority
    recommendations.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });

    return recommendations;
  }

  // Helper methods
  private getExtensionsForLanguage(language: string): string {
    const extensions = {
      javascript: 'js',
      typescript: 'ts,tsx',
      python: 'py',
      java: 'java',
      csharp: 'cs'
    };
    return extensions[language as keyof typeof extensions] || 'js';
  }

  private getTestPatterns(framework: string): string[] {
    const patterns = {
      jest: ['**/*.test.js', '**/*.test.ts', '**/*.spec.js', '**/*.spec.ts'],
      vitest: ['**/*.test.js', '**/*.test.ts', '**/*.spec.js', '**/*.spec.ts'],
      mocha: ['**/test/**/*.js', '**/test/**/*.ts', '**/*.test.js', '**/*.test.ts'],
      jasmine: ['**/spec/**/*.js', '**/*.spec.js'],
      pytest: ['**/test_*.py', '**/*_test.py', '**/tests/**/*.py'],
      unittest: ['**/test_*.py', '**/*_test.py', '**/tests/**/*.py'],
      junit: ['**/Test*.java', '**/*Test.java'],
      nunit: ['**/Test*.cs', '**/*Test.cs']
    };
    return patterns[framework as keyof typeof patterns] || ['**/*.test.*'];
  }

  private generateTestFilePath(sourceFile: string, options: TestGenerationOptions, type?: string): string {
    const ext = this.getTestExtension(options.framework);
    const baseName = path.basename(sourceFile, path.extname(sourceFile));
    const typePrefix = type ? `${type}-` : '';
    return `${typePrefix}${baseName}.test.${ext}`;
  }

  private getTestExtension(framework: string): string {
    const extensions = {
      jest: 'js',
      vitest: 'ts',
      mocha: 'js',
      jasmine: 'js',
      pytest: 'py',
      unittest: 'py',
      junit: 'java',
      nunit: 'cs'
    };
    return extensions[framework as keyof typeof extensions] || 'js';
  }

  private parseTestCases(content: string, framework: string): any[] {
    // Basic parsing - would need framework-specific parsers for production
    return []; // Placeholder
  }

  private async calculateCoverage(testFiles: TestFile[]): Promise<any> {
    // Would integrate with coverage tools
    return { statements: 0, branches: 0, functions: 0, lines: 0, overall: 0 };
  }

  private estimateCoverage(gaps: CoverageGap[]): any {
    // Estimate coverage based on gaps
    return { statements: 0, branches: 0, functions: 0, lines: 0, overall: 0 };
  }

  private inferSourcePath(testFile: string, options: TestGenerationOptions): string {
    // Basic inference - would need more sophisticated logic
    return testFile.replace('.test.', '.');
  }

  private extractFunctions(content: string, language: string): string[] {
    // Basic function extraction - would need language-specific parsers
    return [];
  }

  private calculatePriority(func: string, content: string): 'low' | 'medium' | 'high' | 'critical' {
    // Basic priority calculation
    return 'medium';
  }

  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}
