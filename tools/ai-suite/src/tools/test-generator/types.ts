/**
 * Test Generator Types
 */

export interface TestGenerationOptions {
  framework: 'jest' | 'vitest' | 'mocha' | 'jasmine' | 'pytest' | 'unittest' | 'junit' | 'nunit';
  language: 'javascript' | 'typescript' | 'python' | 'java' | 'csharp';
  coverage?: {
    target: number; // 0-100
    exclude?: string[];
  };
  types?: ('unit' | 'integration' | 'e2e')[];
  patterns?: {
    include?: string[];
    exclude?: string[];
  };
  output?: {
    directory: string;
    overwrite?: boolean;
  };
}

export interface TestSuite {
  id: string;
  name: string;
  framework: string;
  language: string;
  files: TestFile[];
  coverage: CoverageMetrics;
  metadata: {
    generatedAt: Date;
    generator: string;
    version: string;
  };
}

export interface TestFile {
  path: string;
  sourcePath: string;
  content: string;
  tests: TestCase[];
  coverage: CoverageMetrics;
}

export interface TestCase {
  id: string;
  name: string;
  type: 'unit' | 'integration' | 'e2e';
  description?: string;
  code: string;
  assertions: Assertion[];
  dependencies?: string[];
}

export interface Assertion {
  type: 'equal' | 'deepEqual' | 'throws' | 'doesNotThrow' | 'contains' | 'matches';
  expected: any;
  actual?: string;
  message?: string;
}

export interface CoverageMetrics {
  statements: number;
  branches: number;
  functions: number;
  lines: number;
  overall: number;
}

export interface TestAnalysisResult {
  path: string;
  existingTests: TestFile[];
  coverageGaps: CoverageGap[];
  recommendations: TestRecommendation[];
  estimatedCoverage: CoverageMetrics;
}

export interface CoverageGap {
  file: string;
  function: string;
  type: 'uncovered' | 'partially-covered' | 'high-risk';
  priority: 'low' | 'medium' | 'high' | 'critical';
  reason: string;
}

export interface TestRecommendation {
  type: 'add-unit-test' | 'add-integration-test' | 'add-e2e-test' | 'improve-coverage' | 'refactor-test';
  file: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedEffort: number; // hours
  expectedCoverage: number;
}
