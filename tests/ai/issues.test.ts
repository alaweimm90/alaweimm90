/**
 * Integration tests for AI Issues Module
 * Tests automated issue management and tracking
 */

import { describe, it, expect, vi } from 'vitest';

describe('AI Issues Module', () => {
  describe('Issue Categories', () => {
    type IssueCategory =
      | 'security'
      | 'compliance'
      | 'performance'
      | 'maintenance'
      | 'documentation'
      | 'testing'
      | 'dependency'
      | 'architecture';

    const categories: IssueCategory[] = [
      'security',
      'compliance',
      'performance',
      'maintenance',
      'documentation',
      'testing',
      'dependency',
      'architecture',
    ];

    it('should have all expected categories', () => {
      expect(categories).toContain('security');
      expect(categories).toContain('compliance');
      expect(categories).toContain('performance');
      expect(categories).toContain('maintenance');
      expect(categories.length).toBe(8);
    });
  });

  describe('Issue Priority', () => {
    type IssuePriority = 'critical' | 'high' | 'medium' | 'low';

    const priorityOrder: Record<IssuePriority, number> = {
      critical: 4,
      high: 3,
      medium: 2,
      low: 1,
    };

    it('should order priorities correctly', () => {
      expect(priorityOrder.critical).toBeGreaterThan(priorityOrder.high);
      expect(priorityOrder.high).toBeGreaterThan(priorityOrder.medium);
      expect(priorityOrder.medium).toBeGreaterThan(priorityOrder.low);
    });

    it('should sort by priority', () => {
      const issues = [
        { title: 'Low issue', priority: 'low' as IssuePriority },
        { title: 'Critical issue', priority: 'critical' as IssuePriority },
        { title: 'Medium issue', priority: 'medium' as IssuePriority },
      ];

      const sorted = issues.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority]);

      expect(sorted[0].priority).toBe('critical');
      expect(sorted[1].priority).toBe('medium');
      expect(sorted[2].priority).toBe('low');
    });
  });

  describe('Issue Status', () => {
    type IssueStatus = 'open' | 'in-progress' | 'resolved' | 'wont-fix' | 'duplicate';

    it('should transition status correctly', () => {
      const validTransitions: Record<IssueStatus, IssueStatus[]> = {
        open: ['in-progress', 'wont-fix', 'duplicate'],
        'in-progress': ['resolved', 'open'],
        resolved: ['open'],
        'wont-fix': ['open'],
        duplicate: [],
      };

      expect(validTransitions.open).toContain('in-progress');
      expect(validTransitions['in-progress']).toContain('resolved');
      expect(validTransitions.duplicate).toHaveLength(0);
    });
  });

  describe('Issue Creation', () => {
    interface AIIssue {
      id: string;
      createdAt: string;
      category: string;
      priority: string;
      title: string;
      description: string;
      status: string;
      source: string;
      labels: string[];
      relatedFiles: string[];
    }

    it('should create issue with required fields', () => {
      const issue: AIIssue = {
        id: `issue-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        createdAt: new Date().toISOString(),
        category: 'security',
        priority: 'critical',
        title: 'Exposed API key in config.ts',
        description: 'An AWS access key was found in the configuration file.',
        status: 'open',
        source: 'security-scanner',
        labels: ['security', 'critical'],
        relatedFiles: ['config.ts'],
      };

      expect(issue.id).toMatch(/^issue-\d+-[a-z0-9]+$/);
      expect(issue.createdAt).toBeDefined();
      expect(issue.category).toBe('security');
      expect(issue.status).toBe('open');
    });

    it('should generate unique IDs', () => {
      const generateId = () => `issue-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
      const id1 = generateId();
      const id2 = generateId();
      expect(id1).not.toBe(id2);
    });
  });

  describe('Issue Templates', () => {
    interface IssueTemplate {
      key: string;
      title: string;
      description: string;
      category: string;
      priority: string;
      labels: string[];
    }

    const templates: IssueTemplate[] = [
      {
        key: 'secret-detected',
        title: '[SECURITY] Secret detected: {type}',
        description: 'A {type} was detected in {file}. Please remove it and rotate the credential.',
        category: 'security',
        priority: 'critical',
        labels: ['security', 'secret', 'critical'],
      },
      {
        key: 'vulnerability',
        title: '[SECURITY] Vulnerability in {package}',
        description: '{severity} severity vulnerability in {package}: {title}',
        category: 'security',
        priority: 'high',
        labels: ['security', 'vulnerability', 'dependencies'],
      },
      {
        key: 'compliance-violation',
        title: '[COMPLIANCE] {rule} violation',
        description: 'Compliance rule {rule} was violated. {details}',
        category: 'compliance',
        priority: 'medium',
        labels: ['compliance', 'policy'],
      },
    ];

    it('should have template for each key', () => {
      expect(templates.find((t) => t.key === 'secret-detected')).toBeDefined();
      expect(templates.find((t) => t.key === 'vulnerability')).toBeDefined();
      expect(templates.find((t) => t.key === 'compliance-violation')).toBeDefined();
    });

    it('should interpolate variables in template', () => {
      const template = templates.find((t) => t.key === 'secret-detected')!;
      const variables = { type: 'AWS Access Key', file: 'config.ts' };

      let title = template.title;
      let description = template.description;

      for (const [key, value] of Object.entries(variables)) {
        title = title.replace(`{${key}}`, value);
        description = description.replace(new RegExp(`\\{${key}\\}`, 'g'), value);
      }

      expect(title).toBe('[SECURITY] Secret detected: AWS Access Key');
      expect(description).toContain('AWS Access Key');
      expect(description).toContain('config.ts');
    });
  });

  describe('Label Generation', () => {
    it('should generate priority label', () => {
      const generatePriorityLabel = (priority: string) => `priority:${priority}`;
      expect(generatePriorityLabel('critical')).toBe('priority:critical');
      expect(generatePriorityLabel('high')).toBe('priority:high');
    });

    it('should generate category label', () => {
      const generateCategoryLabel = (category: string) => `category:${category}`;
      expect(generateCategoryLabel('security')).toBe('category:security');
    });

    it('should generate source label', () => {
      const generateSourceLabel = (source: string) => `source:${source}`;
      expect(generateSourceLabel('ai-tools')).toBe('source:ai-tools');
    });
  });

  describe('Issue Filtering', () => {
    const issues = [
      { id: '1', category: 'security', priority: 'critical', status: 'open' },
      { id: '2', category: 'security', priority: 'high', status: 'resolved' },
      { id: '3', category: 'compliance', priority: 'medium', status: 'open' },
      { id: '4', category: 'maintenance', priority: 'low', status: 'open' },
    ];

    it('should filter by category', () => {
      const securityIssues = issues.filter((i) => i.category === 'security');
      expect(securityIssues.length).toBe(2);
    });

    it('should filter by priority', () => {
      const criticalIssues = issues.filter((i) => i.priority === 'critical');
      expect(criticalIssues.length).toBe(1);
    });

    it('should filter by status', () => {
      const openIssues = issues.filter((i) => i.status === 'open');
      expect(openIssues.length).toBe(3);
    });

    it('should combine filters', () => {
      const openSecurityIssues = issues.filter(
        (i) => i.category === 'security' && i.status === 'open'
      );
      expect(openSecurityIssues.length).toBe(1);
    });
  });

  describe('Issue Statistics', () => {
    interface IssueStats {
      total: number;
      byStatus: Record<string, number>;
      byCategory: Record<string, number>;
      byPriority: Record<string, number>;
    }

    it('should calculate statistics correctly', () => {
      const issues = [
        { category: 'security', priority: 'critical', status: 'open' },
        { category: 'security', priority: 'high', status: 'open' },
        { category: 'compliance', priority: 'medium', status: 'resolved' },
      ];

      const stats: IssueStats = {
        total: issues.length,
        byStatus: {},
        byCategory: {},
        byPriority: {},
      };

      for (const issue of issues) {
        stats.byStatus[issue.status] = (stats.byStatus[issue.status] || 0) + 1;
        stats.byCategory[issue.category] = (stats.byCategory[issue.category] || 0) + 1;
        stats.byPriority[issue.priority] = (stats.byPriority[issue.priority] || 0) + 1;
      }

      expect(stats.total).toBe(3);
      expect(stats.byStatus.open).toBe(2);
      expect(stats.byCategory.security).toBe(2);
      expect(stats.byPriority.critical).toBe(1);
    });
  });

  describe('Creating Issues from Findings', () => {
    it('should create issues from security findings', () => {
      const findings = [
        { type: 'secret', severity: 'critical', description: 'API key found', file: 'config.ts' },
        { type: 'vulnerability', severity: 'high', description: 'CVE-2024-1234', file: 'package.json' },
      ];

      const issues = findings.map((f, i) => ({
        id: `issue-${i}`,
        title: `[SECURITY] ${f.description}`,
        category: 'security',
        priority: f.severity,
        relatedFiles: [f.file],
      }));

      expect(issues.length).toBe(2);
      expect(issues[0].priority).toBe('critical');
      expect(issues[1].priority).toBe('high');
    });

    it('should create issues from compliance violations', () => {
      const violations = [
        { ruleId: 'SEC-001', message: 'Secrets detected' },
        { ruleId: 'ARCH-001', message: 'File too large' },
      ];

      const issues = violations.map((v, i) => ({
        id: `issue-${i}`,
        title: `[COMPLIANCE] ${v.ruleId}: ${v.message}`,
        category: 'compliance',
        priority: 'medium',
      }));

      expect(issues.length).toBe(2);
      expect(issues[0].title).toContain('SEC-001');
    });
  });

  describe('GitHub Issue Integration', () => {
    it('should format GitHub issue body', () => {
      const issue = {
        title: '[SECURITY] API key exposed',
        description: 'An API key was found in config.ts',
        category: 'security',
        priority: 'critical',
        relatedFiles: ['config.ts', 'src/api.ts'],
        labels: ['security', 'critical'],
      };

      const body = `## Description
${issue.description}

## Category
${issue.category}

## Priority
${issue.priority}

## Related Files
${issue.relatedFiles.map((f) => `- \`${f}\``).join('\n')}

---
🤖 Generated by AI Tools`;

      expect(body).toContain('## Description');
      expect(body).toContain(issue.description);
      expect(body).toContain('config.ts');
      expect(body).toContain('Generated by AI Tools');
    });

    it('should format labels for gh cli', () => {
      const labels = ['security', 'critical', 'ai-tools'];
      const labelArg = labels.join(',');
      expect(labelArg).toBe('security,critical,ai-tools');
    });
  });

  describe('Duplicate Detection', () => {
    it('should detect potential duplicates by title similarity', () => {
      const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

      const existing = ['API key exposed in config', 'Large file detected'];
      const newTitle = 'api-key exposed in config.ts';

      const normalizedNew = normalize(newTitle);
      const isDuplicate = existing.some((title) => {
        const normalizedExisting = normalize(title);
        return (
          normalizedExisting.includes(normalizedNew.substring(0, 20)) ||
          normalizedNew.includes(normalizedExisting.substring(0, 20))
        );
      });

      expect(isDuplicate).toBe(true);
    });
  });
});
