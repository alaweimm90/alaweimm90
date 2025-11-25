/**
 * Workflow Manager - Manages workflow templates
 */

import * as fs from 'fs';
import * as path from 'path';
import { Workflow } from '@monorepo/agent-core';
import { WorkflowTemplate, WorkflowLibrary, WORKFLOW_CATEGORIES } from './types';

export class WorkflowManager {
  private templates: Map<string, WorkflowTemplate> = new Map();

  private templateDir: string;

  constructor(templateDir: string = path.join(__dirname, '../templates')) {
    this.templateDir = templateDir;
    this.loadTemplates();
  }

  /**
   * Load templates from directory
   */
  private loadTemplates(): void {
    if (!fs.existsSync(this.templateDir)) {
      return;
    }

    const files = fs.readdirSync(this.templateDir);
    for (const file of files) {
      if (file.endsWith('.json')) {
        try {
          const filePath = path.join(this.templateDir, file);
          const data = fs.readFileSync(filePath, 'utf-8');
          const template = JSON.parse(data) as WorkflowTemplate;
          this.templates.set(template.id, template);
        } catch (error) {
          console.warn(`Failed to load workflow template ${file}:`, error);
        }
      }
    }
  }

  /**
   * Get a template by ID
   * @param id
   */
  public getTemplate(id: string): WorkflowTemplate | undefined {
    return this.templates.get(id);
  }

  /**
   * Get all templates
   */
  public getAllTemplates(): WorkflowTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Get templates by category
   * @param category
   */
  public getTemplatesByCategory(category: string): WorkflowTemplate[] {
    return Array.from(this.templates.values()).filter(t => t.category === category);
  }

  /**
   * Get templates by tags
   * @param tag
   */
  public getTemplatesByTag(tag: string): WorkflowTemplate[] {
    return Array.from(this.templates.values()).filter(t => t.tags?.includes(tag));
  }

  /**
   * Register a new template
   * @param template
   */
  public registerTemplate(template: WorkflowTemplate): void {
    this.templates.set(template.id, template);
  }

  /**
   * Create a workflow from template
   * @param templateId
   * @param overrides
   */
  public createWorkflow(templateId: string, overrides: Partial<WorkflowTemplate>): Workflow {
    const template = this.getTemplate(templateId);
    if (!template) {
      throw new Error(`Workflow template not found: ${templateId}`);
    }

    return {
      ...template,
      ...overrides,
      id: overrides.id || `workflow-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  /**
   * Export templates as library
   */
  public exportLibrary(): WorkflowLibrary {
    return {
      workflows: this.getAllTemplates(),
      version: '1.0.0',
      lastUpdated: new Date().toISOString(),
    };
  }

  /**
   * Save library to file
   * @param outputPath
   */
  public saveLibrary(outputPath: string): void {
    const library = this.exportLibrary();
    const dir = path.dirname(outputPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(outputPath, JSON.stringify(library, null, 2), 'utf-8');
  }
}

/**
 * Create default workflow templates
 */
export function createDefaultTemplates(): WorkflowTemplate[] {
  return [
    {
      id: 'code-review-workflow',
      name: 'Code Review Workflow',
      description: 'Standard code review process',
      version: '1.0.0',
      category: WORKFLOW_CATEGORIES.DEVELOPMENT,
      tags: ['review', 'quality', 'development'],
      difficulty: 'intermediate',
      estimatedTime: '30 minutes',
      enabled: true,
      steps: [
        {
          id: 'step-1',
          name: 'Lint Check',
          type: 'task',
          action: 'lint',
          agentId: 'code-agent',
        },
        {
          id: 'step-2',
          name: 'Type Check',
          type: 'task',
          action: 'type-check',
          agentId: 'code-agent',
        },
        {
          id: 'step-3',
          name: 'Test Execution',
          type: 'task',
          action: 'test',
          agentId: 'code-agent',
        },
        {
          id: 'step-4',
          name: 'Security Scan',
          type: 'task',
          action: 'security-scan',
          agentId: 'analysis-agent',
        },
      ],
    },
    {
      id: 'bug-fix-workflow',
      name: 'Bug Fix Workflow',
      description: 'Process for identifying and fixing bugs',
      version: '1.0.0',
      category: WORKFLOW_CATEGORIES.DEVELOPMENT,
      tags: ['bug', 'fix', 'development'],
      difficulty: 'beginner',
      estimatedTime: '1 hour',
      enabled: true,
      steps: [
        {
          id: 'step-1',
          name: 'Create Issue',
          type: 'task',
          action: 'create-issue',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-2',
          name: 'Reproduce Bug',
          type: 'task',
          action: 'test',
          agentId: 'code-agent',
        },
        {
          id: 'step-3',
          name: 'Implement Fix',
          type: 'task',
          action: 'code-fix',
          agentId: 'code-agent',
        },
        {
          id: 'step-4',
          name: 'Verify Fix',
          type: 'task',
          action: 'test',
          agentId: 'code-agent',
        },
      ],
    },
    {
      id: 'feature-development-workflow',
      name: 'Feature Development Workflow',
      description: 'Complete process for developing new features',
      version: '1.0.0',
      category: WORKFLOW_CATEGORIES.DEVELOPMENT,
      tags: ['feature', 'development'],
      difficulty: 'advanced',
      estimatedTime: '2-3 days',
      enabled: true,
      steps: [
        {
          id: 'step-1',
          name: 'Create Feature Issue',
          type: 'task',
          action: 'create-issue',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-2',
          name: 'Design Review',
          type: 'task',
          action: 'design-review',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-3',
          name: 'Implement Feature',
          type: 'task',
          action: 'code-feature',
          agentId: 'code-agent',
        },
        {
          id: 'step-4',
          name: 'Write Tests',
          type: 'task',
          action: 'test',
          agentId: 'code-agent',
        },
        {
          id: 'step-5',
          name: 'Code Review',
          type: 'task',
          action: 'review',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-6',
          name: 'Documentation',
          type: 'task',
          action: 'document',
          agentId: 'code-agent',
        },
      ],
    },
    {
      id: 'security-audit-workflow',
      name: 'Security Audit Workflow',
      description: 'Comprehensive security audit process',
      version: '1.0.0',
      category: WORKFLOW_CATEGORIES.SECURITY,
      tags: ['security', 'audit', 'compliance'],
      difficulty: 'advanced',
      estimatedTime: '2 hours',
      enabled: true,
      steps: [
        {
          id: 'step-1',
          name: 'Dependency Scan',
          type: 'task',
          action: 'scan-dependencies',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-2',
          name: 'Code Security Scan',
          type: 'task',
          action: 'security-scan',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-3',
          name: 'Configuration Review',
          type: 'task',
          action: 'config-review',
          agentId: 'analysis-agent',
        },
        {
          id: 'step-4',
          name: 'Generate Report',
          type: 'task',
          action: 'generate-report',
          agentId: 'analysis-agent',
        },
      ],
    },
  ];
}
