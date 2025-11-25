/**
 * Issue Manager - Manages issue templates and creation
 */

import * as fs from 'fs';
import * as path from 'path';
import { IssueTemplate, Issue, IssueType, IssuePriority, FieldType } from './types';

export class IssueManager {
  private templates: Map<string, IssueTemplate> = new Map();

  private issues: Map<string, Issue> = new Map();

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
          const template = JSON.parse(data) as IssueTemplate;
          this.templates.set(template.id, template);
        } catch (error) {
          console.warn(`Failed to load template ${file}:`, error);
        }
      }
    }
  }

  /**
   * Get a template by ID
   * @param id
   */
  public getTemplate(id: string): IssueTemplate | undefined {
    return this.templates.get(id);
  }

  /**
   * Get all templates
   */
  public getAllTemplates(): IssueTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Get templates by type
   * @param type
   */
  public getTemplatesByType(type: IssueType): IssueTemplate[] {
    return Array.from(this.templates.values()).filter(t => t.type === type);
  }

  /**
   * Register a new template
   * @param template
   */
  public registerTemplate(template: IssueTemplate): void {
    this.templates.set(template.id, template);
  }

  /**
   * Create an issue from template
   * @param templateId
   * @param data
   */
  public createIssue(templateId: string, data: Record<string, unknown>): Issue {
    const template = this.getTemplate(templateId);
    if (!template) {
      throw new Error(`Template not found: ${templateId}`);
    }

    // Validate required fields
    for (const field of template.fields) {
      if (field.required && !(field.name in data)) {
        throw new Error(`Required field missing: ${field.name}`);
      }
    }

    const issue: Issue = {
      id: this.generateIssueId(),
      templateId,
      title: data.title as string,
      description: data.description as string,
      type: template.type,
      priority: (data.priority as IssuePriority) || IssuePriority.MEDIUM,
      fields: data,
      labels: template.labels,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.issues.set(issue.id!, issue);
    return issue;
  }

  /**
   * Get an issue by ID
   * @param id
   */
  public getIssue(id: string): Issue | undefined {
    return this.issues.get(id);
  }

  /**
   * Get all issues
   */
  public getAllIssues(): Issue[] {
    return Array.from(this.issues.values());
  }

  /**
   * Update an issue
   * @param id
   * @param updates
   */
  public updateIssue(id: string, updates: Partial<Issue>): Issue {
    const issue = this.getIssue(id);
    if (!issue) {
      throw new Error(`Issue not found: ${id}`);
    }

    const updated = { ...issue, ...updates, updatedAt: new Date() };
    this.issues.set(id, updated);
    return updated;
  }

  /**
   * Delete an issue
   * @param id
   */
  public deleteIssue(id: string): void {
    this.issues.delete(id);
  }

  private generateIssueId(): string {
    return `issue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Create default issue templates
 */
export function createDefaultTemplates(): IssueTemplate[] {
  return [
    {
      id: 'bug-report',
      name: 'Bug Report',
      description: 'Report a bug in the project',
      type: IssueType.BUG,
      priority: IssuePriority.HIGH,
      fields: [
        {
          name: 'title',
          description: 'Short summary of the bug',
          type: FieldType.TEXT,
          required: true,
        },
        {
          name: 'description',
          description: 'Detailed description of the bug',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'steps',
          description: 'Steps to reproduce the bug',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'expected',
          description: 'Expected behavior',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'actual',
          description: 'Actual behavior',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'environment',
          description: 'Environment details',
          type: FieldType.TEXTAREA,
          required: false,
        },
      ],
      labels: ['bug'],
    },
    {
      id: 'feature-request',
      name: 'Feature Request',
      description: 'Request a new feature',
      type: IssueType.FEATURE,
      priority: IssuePriority.MEDIUM,
      fields: [
        {
          name: 'title',
          description: 'Feature title',
          type: FieldType.TEXT,
          required: true,
        },
        {
          name: 'description',
          description: 'Feature description and use case',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'motivation',
          description: 'Why is this feature needed?',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'acceptance',
          description: 'Acceptance criteria',
          type: FieldType.TEXTAREA,
          required: false,
        },
      ],
      labels: ['enhancement'],
    },
    {
      id: 'refactor',
      name: 'Refactoring',
      description: 'Code refactoring task',
      type: IssueType.REFACTOR,
      priority: IssuePriority.LOW,
      fields: [
        {
          name: 'title',
          description: 'Refactoring title',
          type: FieldType.TEXT,
          required: true,
        },
        {
          name: 'description',
          description: 'What needs to be refactored',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'rationale',
          description: 'Why is this refactoring needed?',
          type: FieldType.TEXTAREA,
          required: true,
        },
        {
          name: 'scope',
          description: 'Scope of the refactoring',
          type: FieldType.TEXTAREA,
          required: false,
        },
      ],
      labels: ['refactor'],
    },
  ];
}
