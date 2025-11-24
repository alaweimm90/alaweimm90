/**
 * Issue library types
 */

export interface IssueTemplate {
  id: string;
  name: string;
  description: string;
  type: IssueType;
  fields: IssueField[];
  labels?: string[];
  priority?: IssuePriority;
}

export enum IssueType {
  BUG = 'bug',
  FEATURE = 'feature',
  REFACTOR = 'refactor',
  DOCUMENTATION = 'documentation',
  CHORE = 'chore',
  PERFORMANCE = 'performance',
  SECURITY = 'security',
  TESTING = 'testing',
}

export enum IssuePriority {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
}

export interface IssueField {
  name: string;
  description: string;
  type: FieldType;
  required: boolean;
  default?: string;
  options?: string[];
}

export enum FieldType {
  TEXT = 'text',
  TEXTAREA = 'textarea',
  SELECT = 'select',
  CHECKBOX = 'checkbox',
  DATE = 'date',
  MULTI_SELECT = 'multi_select',
}

export interface Issue {
  id?: string;
  templateId: string;
  title: string;
  description: string;
  type: IssueType;
  priority: IssuePriority;
  fields: Record<string, unknown>;
  labels?: string[];
  createdAt?: Date;
  updatedAt?: Date;
}
