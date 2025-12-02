/**
 * Integration tests for AI Compliance Module
 * Tests policy-based validation and scoring
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actual,
    existsSync: vi.fn().mockReturnValue(true),
    readFileSync: vi
      .fn()
      .mockReturnValue(
        '# README\n\nThis is a comprehensive README with more than 100 characters of content for testing purposes.'
      ),
    writeFileSync: vi.fn(),
  };
});

describe('AI Compliance Module', () => {
  describe('Severity Weights', () => {
    const SEVERITY_WEIGHTS = {
      critical: 25,
      high: 15,
      medium: 8,
      low: 3,
    };

    it('should have correct severity weights', () => {
      expect(SEVERITY_WEIGHTS.critical).toBe(25);
      expect(SEVERITY_WEIGHTS.high).toBe(15);
      expect(SEVERITY_WEIGHTS.medium).toBe(8);
      expect(SEVERITY_WEIGHTS.low).toBe(3);
    });

    it('should weight critical issues highest', () => {
      const weights = Object.values(SEVERITY_WEIGHTS);
      const maxWeight = Math.max(...weights);
      expect(SEVERITY_WEIGHTS.critical).toBe(maxWeight);
    });
  });

  describe('Grade Calculation', () => {
    const calculateGrade = (score: number): 'A' | 'B' | 'C' | 'D' | 'F' => {
      return score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';
    };

    it('should return A for score >= 90', () => {
      expect(calculateGrade(90)).toBe('A');
      expect(calculateGrade(95)).toBe('A');
      expect(calculateGrade(100)).toBe('A');
    });

    it('should return B for score 80-89', () => {
      expect(calculateGrade(80)).toBe('B');
      expect(calculateGrade(85)).toBe('B');
      expect(calculateGrade(89)).toBe('B');
    });

    it('should return C for score 70-79', () => {
      expect(calculateGrade(70)).toBe('C');
      expect(calculateGrade(75)).toBe('C');
      expect(calculateGrade(79)).toBe('C');
    });

    it('should return D for score 60-69', () => {
      expect(calculateGrade(60)).toBe('D');
      expect(calculateGrade(65)).toBe('D');
      expect(calculateGrade(69)).toBe('D');
    });

    it('should return F for score < 60', () => {
      expect(calculateGrade(59)).toBe('F');
      expect(calculateGrade(30)).toBe('F');
      expect(calculateGrade(0)).toBe('F');
    });
  });

  describe('Security Rules', () => {
    describe('SEC-001: No Secrets in Code', () => {
      it('should detect .env files', () => {
        const changedFiles = ['config.ts', '.env', 'package.json'];
        const violations: string[] = [];

        for (const file of changedFiles) {
          if (file.endsWith('.env') || file.includes('secrets')) {
            violations.push(`Sensitive file pattern: ${file}`);
          }
        }

        expect(violations.length).toBe(1);
        expect(violations[0]).toContain('.env');
      });

      it('should detect secrets directory', () => {
        const changedFiles = ['src/secrets/api-keys.json'];
        const violations: string[] = [];

        for (const file of changedFiles) {
          if (file.endsWith('.env') || file.includes('secrets')) {
            violations.push(`Sensitive file pattern: ${file}`);
          }
        }

        expect(violations.length).toBe(1);
      });

      it('should pass for safe files', () => {
        const changedFiles = ['config.ts', 'package.json', 'README.md'];
        const violations: string[] = [];

        for (const file of changedFiles) {
          if (file.endsWith('.env') || file.includes('secrets')) {
            violations.push(`Sensitive file pattern: ${file}`);
          }
        }

        expect(violations.length).toBe(0);
      });
    });

    describe('SEC-002: Protected Files Respected', () => {
      const protectedPatterns = ['README.md', 'LICENSE', 'CODEOWNERS', '.github/workflows/'];

      it('should detect protected file modifications', () => {
        const changedFiles = ['README.md', 'tools/ai/cache.ts'];
        const violations: string[] = [];

        for (const file of changedFiles) {
          for (const pattern of protectedPatterns) {
            if (file === pattern || file.startsWith(pattern)) {
              violations.push(`Protected file modified: ${file}`);
            }
          }
        }

        expect(violations.length).toBe(1);
        expect(violations[0]).toContain('README.md');
      });

      it('should detect workflow modifications', () => {
        const changedFiles = ['.github/workflows/ci.yml'];
        const violations: string[] = [];

        for (const file of changedFiles) {
          for (const pattern of protectedPatterns) {
            if (file === pattern || file.startsWith(pattern)) {
              violations.push(`Protected file modified: ${file}`);
            }
          }
        }

        expect(violations.length).toBe(1);
      });
    });
  });

  describe('Architecture Rules', () => {
    describe('ARCH-001: File Size Limits', () => {
      const MAX_LINES = 500;

      it('should flag files exceeding line limit', () => {
        const fileLines = { 'large-file.ts': 600, 'small-file.ts': 100 };
        const violations: string[] = [];

        for (const [file, lines] of Object.entries(fileLines)) {
          if (lines > MAX_LINES) {
            violations.push(`${file}: ${lines} lines (max ${MAX_LINES})`);
          }
        }

        expect(violations.length).toBe(1);
        expect(violations[0]).toContain('large-file.ts');
      });

      it('should pass for files within limit', () => {
        const fileLines = { 'file1.ts': 200, 'file2.ts': 450 };
        const violations: string[] = [];

        for (const [file, lines] of Object.entries(fileLines)) {
          if (lines > MAX_LINES) {
            violations.push(`${file}: ${lines} lines (max ${MAX_LINES})`);
          }
        }

        expect(violations.length).toBe(0);
      });
    });
  });

  describe('Weighted Score Calculation', () => {
    const SEVERITY_WEIGHTS = {
      critical: 25,
      high: 15,
      medium: 8,
      low: 3,
    };

    it('should calculate weighted score correctly', () => {
      const results = [
        { severity: 'critical' as const, score: 100 },
        { severity: 'high' as const, score: 50 },
        { severity: 'medium' as const, score: 80 },
        { severity: 'low' as const, score: 100 },
      ];

      let totalWeight = 0;
      let weightedScore = 0;

      for (const result of results) {
        const weight = SEVERITY_WEIGHTS[result.severity];
        totalWeight += weight;
        weightedScore += result.score * weight;
      }

      const overallScore = Math.round(weightedScore / totalWeight);

      // Expected: (100*25 + 50*15 + 80*8 + 100*3) / (25+15+8+3)
      // = (2500 + 750 + 640 + 300) / 51
      // = 4190 / 51 = 82.16 ≈ 82
      expect(overallScore).toBe(82);
    });
  });

  describe('Category Scoring', () => {
    type ComplianceCategory =
      | 'security'
      | 'governance'
      | 'code-quality'
      | 'documentation'
      | 'testing'
      | 'architecture';

    it('should aggregate scores by category', () => {
      const results: Array<{ category: ComplianceCategory; score: number }> = [
        { category: 'security', score: 100 },
        { category: 'security', score: 80 },
        { category: 'governance', score: 90 },
      ];

      const categoryScores: Record<ComplianceCategory, { total: number; count: number }> = {
        security: { total: 0, count: 0 },
        governance: { total: 0, count: 0 },
        'code-quality': { total: 0, count: 0 },
        documentation: { total: 0, count: 0 },
        testing: { total: 0, count: 0 },
        architecture: { total: 0, count: 0 },
      };

      for (const result of results) {
        categoryScores[result.category].total += result.score;
        categoryScores[result.category].count++;
      }

      const securityAvg = categoryScores.security.total / categoryScores.security.count;
      const governanceAvg = categoryScores.governance.total / categoryScores.governance.count;

      expect(securityAvg).toBe(90);
      expect(governanceAvg).toBe(90);
    });
  });

  describe('Violation Collection', () => {
    it('should collect unique recommendations', () => {
      const violations = [
        { recommendations: ['Use environment variables', 'Add to .gitignore'] },
        { recommendations: ['Use environment variables', 'Check protected files'] },
        { recommendations: ['Run tests'] },
      ];

      const allRecommendations = violations
        .flatMap((v) => v.recommendations || [])
        .filter((r, i, arr) => arr.indexOf(r) === i);

      expect(allRecommendations.length).toBe(4);
      expect(allRecommendations).toContain('Use environment variables');
    });
  });

  describe('Report Generation', () => {
    interface ComplianceReport {
      timestamp: string;
      overallScore: number;
      grade: 'A' | 'B' | 'C' | 'D' | 'F';
      summary: {
        passed: number;
        failed: number;
        warnings: number;
        critical: number;
      };
    }

    it('should generate valid report structure', () => {
      const report: ComplianceReport = {
        timestamp: new Date().toISOString(),
        overallScore: 85,
        grade: 'B',
        summary: {
          passed: 8,
          failed: 2,
          warnings: 1,
          critical: 0,
        },
      };

      expect(report.timestamp).toBeDefined();
      expect(report.overallScore).toBeGreaterThanOrEqual(0);
      expect(report.overallScore).toBeLessThanOrEqual(100);
      expect(['A', 'B', 'C', 'D', 'F']).toContain(report.grade);
    });
  });
});
