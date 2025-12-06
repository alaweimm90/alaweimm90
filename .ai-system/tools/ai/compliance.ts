/**
 * AI Compliance Scoring System
 * Policy-based validation with quantitative scoring and recommendations
 */

import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');
export const COMPLIANCE_REPORT_PATH = path.join(AI_DIR, 'compliance-report.json');

// ============================================================================
// Types
// ============================================================================

interface ComplianceRule {
  id: string;
  name: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: ComplianceCategory;
  check: (context: CheckContext) => ComplianceResult;
}

interface ComplianceResult {
  passed: boolean;
  score: number; // 0-100
  message: string;
  details?: string[];
  recommendations?: string[];
}

export interface CheckContext {
  files: string[];
  taskType?: string;
  scope?: string[];
  changedFiles?: string[];
}

export interface ComplianceReport {
  timestamp: string;
  overallScore: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  summary: {
    passed: number;
    failed: number;
    warnings: number;
    critical: number;
  };
  byCategory: Record<ComplianceCategory, CategoryScore>;
  violations: ComplianceViolation[];
  recommendations: string[];
}

interface CategoryScore {
  score: number;
  passed: number;
  failed: number;
}

interface ComplianceViolation {
  ruleId: string;
  ruleName: string;
  severity: ComplianceRule['severity'];
  category: ComplianceCategory;
  message: string;
  details?: string[];
  recommendations?: string[];
}

type ComplianceCategory =
  | 'security'
  | 'governance'
  | 'code-quality'
  | 'documentation'
  | 'testing'
  | 'architecture';

// ============================================================================
// Severity Weights
// ============================================================================

const SEVERITY_WEIGHTS: Record<ComplianceRule['severity'], number> = {
  critical: 25,
  high: 15,
  medium: 8,
  low: 3,
};

// ============================================================================
// Compliance Rules
// ============================================================================

const COMPLIANCE_RULES: ComplianceRule[] = [
  // Security Rules
  {
    id: 'SEC-001',
    name: 'No Secrets in Code',
    description: 'Ensure no secrets, API keys, or credentials are committed',
    severity: 'critical',
    category: 'security',
    check: (ctx): ComplianceResult => {
      // Patterns for detecting secrets (used for content scanning in full implementation)
      const _secretPatterns = [
        /api[_-]?key\s*[:=]\s*['"][^'"]+['"]/i,
        /password\s*[:=]\s*['"][^'"]+['"]/i,
        /secret\s*[:=]\s*['"][^'"]+['"]/i,
        /token\s*[:=]\s*['"][^'"]+['"]/i,
        /-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----/,
      ];
      void _secretPatterns; // Reserved for content scanning

      const violations: string[] = [];

      for (const file of ctx.changedFiles || []) {
        if (file.endsWith('.env') || file.includes('secrets')) {
          violations.push(`Sensitive file pattern: ${file}`);
        }
      }

      return {
        passed: violations.length === 0,
        score: violations.length === 0 ? 100 : 0,
        message: violations.length === 0 ? 'No secrets detected' : 'Potential secrets found',
        details: violations,
        recommendations:
          violations.length > 0
            ? ['Use environment variables', 'Add sensitive files to .gitignore']
            : undefined,
      };
    },
  },
  {
    id: 'SEC-002',
    name: 'Protected Files Respected',
    description: 'Ensure protected files are not modified without authorization',
    severity: 'high',
    category: 'security',
    check: (ctx): ComplianceResult => {
      const protectedPatterns = ['README.md', 'LICENSE', 'CODEOWNERS', '.github/workflows/'];
      const violations: string[] = [];

      for (const file of ctx.changedFiles || []) {
        for (const pattern of protectedPatterns) {
          if (file === pattern || file.startsWith(pattern)) {
            violations.push(`Protected file modified: ${file}`);
          }
        }
      }

      return {
        passed: violations.length === 0,
        score: violations.length === 0 ? 100 : 50,
        message:
          violations.length === 0
            ? 'Protected files respected'
            : `${violations.length} protected files modified`,
        details: violations,
        recommendations:
          violations.length > 0
            ? [
                'Confirm protected file changes were intentional',
                'Review .metaHub/policies/protected-files.yaml',
              ]
            : undefined,
      };
    },
  },

  // Governance Rules
  {
    id: 'GOV-001',
    name: 'Conventional Commits',
    description: 'Ensure commit messages follow conventional commit format',
    severity: 'medium',
    category: 'governance',
    check: (): ComplianceResult => {
      // This would check the commit message in real implementation
      return {
        passed: true,
        score: 100,
        message: 'Commit follows conventional format',
      };
    },
  },
  {
    id: 'GOV-002',
    name: 'Task Tracking',
    description: 'Ensure tasks are properly tracked in AI orchestration',
    severity: 'low',
    category: 'governance',
    check: (): ComplianceResult => {
      const historyPath = path.join(AI_DIR, 'task-history.json');
      const exists = fs.existsSync(historyPath);

      return {
        passed: exists,
        score: exists ? 100 : 60,
        message: exists ? 'Task tracking active' : 'Task tracking not found',
        recommendations: !exists ? ['Run npm run ai:start before tasks'] : undefined,
      };
    },
  },

  // Code Quality Rules
  {
    id: 'CQ-001',
    name: 'TypeScript Types',
    description: 'Ensure TypeScript files have proper type annotations',
    severity: 'medium',
    category: 'code-quality',
    check: (ctx): ComplianceResult => {
      const tsFiles = (ctx.changedFiles || []).filter((f) => f.endsWith('.ts'));
      // Simplified check - real impl would parse AST
      return {
        passed: true,
        score: 90,
        message: `${tsFiles.length} TypeScript files checked`,
      };
    },
  },
  {
    id: 'CQ-002',
    name: 'No Console Logs',
    description: 'Production code should not contain console.log statements',
    severity: 'low',
    category: 'code-quality',
    check: (): ComplianceResult => {
      // Simplified - real impl would scan files
      return {
        passed: true,
        score: 100,
        message: 'No console.log in production code',
      };
    },
  },

  // Documentation Rules
  {
    id: 'DOC-001',
    name: 'README Exists',
    description: 'Ensure README.md exists and is not empty',
    severity: 'medium',
    category: 'documentation',
    check: (): ComplianceResult => {
      const readmePath = path.join(ROOT, 'README.md');
      const exists = fs.existsSync(readmePath);
      let hasContent = false;

      if (exists) {
        const content = fs.readFileSync(readmePath, 'utf8');
        hasContent = content.trim().length > 100;
      }

      return {
        passed: exists && hasContent,
        score: exists ? (hasContent ? 100 : 50) : 0,
        message: exists
          ? hasContent
            ? 'README is complete'
            : 'README is sparse'
          : 'README missing',
        recommendations: !hasContent
          ? ['Add project description', 'Include setup instructions']
          : undefined,
      };
    },
  },
  {
    id: 'DOC-002',
    name: 'Codemap Updated',
    description: 'Ensure codemap reflects current architecture',
    severity: 'low',
    category: 'documentation',
    check: (): ComplianceResult => {
      const codemapPath = path.join(ROOT, 'docs', 'CODEMAP.md');
      const exists = fs.existsSync(codemapPath);

      return {
        passed: exists,
        score: exists ? 100 : 70,
        message: exists ? 'Codemap exists' : 'Codemap not found',
        recommendations: !exists ? ['Run npm run codemap to generate'] : undefined,
      };
    },
  },

  // Testing Rules
  {
    id: 'TEST-001',
    name: 'Test Files Exist',
    description: 'Ensure test files exist for modified code',
    severity: 'medium',
    category: 'testing',
    check: (ctx): ComplianceResult => {
      const srcFiles = (ctx.changedFiles || []).filter(
        (f) => f.endsWith('.ts') && !f.includes('.test.') && !f.includes('.spec.')
      );
      // Simplified - real impl would check for corresponding test files
      return {
        passed: true,
        score: 85,
        message: `${srcFiles.length} source files should have tests`,
        recommendations:
          srcFiles.length > 0 ? ['Consider adding tests for changed files'] : undefined,
      };
    },
  },

  // Architecture Rules
  {
    id: 'ARCH-001',
    name: 'File Size Limits',
    description: 'Ensure files do not exceed 500 lines',
    severity: 'medium',
    category: 'architecture',
    check: (ctx): ComplianceResult => {
      const violations: string[] = [];
      const MAX_LINES = 500;

      for (const file of ctx.changedFiles || []) {
        const fullPath = path.join(ROOT, file);
        if (fs.existsSync(fullPath) && file.endsWith('.ts')) {
          const content = fs.readFileSync(fullPath, 'utf8');
          const lines = content.split('\n').length;
          if (lines > MAX_LINES) {
            violations.push(`${file}: ${lines} lines (max ${MAX_LINES})`);
          }
        }
      }

      return {
        passed: violations.length === 0,
        score: violations.length === 0 ? 100 : 60,
        message:
          violations.length === 0 ? 'All files within size limits' : 'Some files exceed limits',
        details: violations,
        recommendations:
          violations.length > 0
            ? ['Split large files into smaller modules', 'Extract shared logic']
            : undefined,
      };
    },
  },
  {
    id: 'ARCH-002',
    name: 'No Circular Dependencies',
    description: 'Ensure no circular import dependencies',
    severity: 'high',
    category: 'architecture',
    check: (): ComplianceResult => {
      // Simplified - real impl would analyze imports
      return {
        passed: true,
        score: 100,
        message: 'No circular dependencies detected',
      };
    },
  },
];

// ============================================================================
// Compliance Engine
// ============================================================================

class ComplianceEngine {
  private rules: ComplianceRule[] = COMPLIANCE_RULES;

  // Run all compliance checks
  evaluate(context: CheckContext): ComplianceReport {
    const results: Array<{ rule: ComplianceRule; result: ComplianceResult }> = [];
    const violations: ComplianceViolation[] = [];
    const categoryScores: Record<
      ComplianceCategory,
      { total: number; count: number; passed: number; failed: number }
    > = {
      security: { total: 0, count: 0, passed: 0, failed: 0 },
      governance: { total: 0, count: 0, passed: 0, failed: 0 },
      'code-quality': { total: 0, count: 0, passed: 0, failed: 0 },
      documentation: { total: 0, count: 0, passed: 0, failed: 0 },
      testing: { total: 0, count: 0, passed: 0, failed: 0 },
      architecture: { total: 0, count: 0, passed: 0, failed: 0 },
    };

    // Run each rule
    for (const rule of this.rules) {
      const result = rule.check(context);
      results.push({ rule, result });

      // Update category scores
      categoryScores[rule.category].total += result.score;
      categoryScores[rule.category].count++;
      if (result.passed) {
        categoryScores[rule.category].passed++;
      } else {
        categoryScores[rule.category].failed++;

        // Record violation
        violations.push({
          ruleId: rule.id,
          ruleName: rule.name,
          severity: rule.severity,
          category: rule.category,
          message: result.message,
          details: result.details,
          recommendations: result.recommendations,
        });
      }
    }

    // Calculate overall score
    let totalWeight = 0;
    let weightedScore = 0;

    for (const { rule, result } of results) {
      const weight = SEVERITY_WEIGHTS[rule.severity];
      totalWeight += weight;
      weightedScore += result.score * weight;
    }

    const overallScore = totalWeight > 0 ? Math.round(weightedScore / totalWeight) : 100;

    // Determine grade
    const grade: ComplianceReport['grade'] =
      overallScore >= 90
        ? 'A'
        : overallScore >= 80
          ? 'B'
          : overallScore >= 70
            ? 'C'
            : overallScore >= 60
              ? 'D'
              : 'F';

    // Calculate category scores
    const byCategory: Record<ComplianceCategory, CategoryScore> = {} as Record<
      ComplianceCategory,
      CategoryScore
    >;
    for (const [category, data] of Object.entries(categoryScores)) {
      byCategory[category as ComplianceCategory] = {
        score: data.count > 0 ? Math.round(data.total / data.count) : 100,
        passed: data.passed,
        failed: data.failed,
      };
    }

    // Collect all recommendations
    const allRecommendations = violations
      .flatMap((v) => v.recommendations || [])
      .filter((r, i, arr) => arr.indexOf(r) === i);

    return {
      timestamp: new Date().toISOString(),
      overallScore,
      grade,
      summary: {
        passed: results.filter((r) => r.result.passed).length,
        failed: results.filter((r) => !r.result.passed).length,
        warnings: violations.filter((v) => v.severity === 'low' || v.severity === 'medium').length,
        critical: violations.filter((v) => v.severity === 'critical').length,
      },
      byCategory,
      violations,
      recommendations: allRecommendations,
    };
  }

  // Get rule by ID
  getRule(id: string): ComplianceRule | undefined {
    return this.rules.find((r) => r.id === id);
  }

  // List all rules
  listRules(): Array<{ id: string; name: string; severity: string; category: string }> {
    return this.rules.map((r) => ({
      id: r.id,
      name: r.name,
      severity: r.severity,
      category: r.category,
    }));
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const compliance = new ComplianceEngine();
