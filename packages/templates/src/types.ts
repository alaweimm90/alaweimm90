/**
 * Workflow template types
 */

import { Workflow, WorkflowStep } from '@monorepo/agent-core';

export interface WorkflowTemplate extends Workflow {
  tags?: string[];
  category?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime?: string;
}

export interface WorkflowLibrary {
  workflows: WorkflowTemplate[];
  version: string;
  lastUpdated: string;
}

export interface StepTemplate {
  id: string;
  name: string;
  description: string;
  template: Omit<WorkflowStep, 'id' | 'name'>;
}

export const WORKFLOW_CATEGORIES = {
  DEVELOPMENT: 'development',
  SECURITY: 'security',
  DEPLOYMENT: 'deployment',
  TESTING: 'testing',
  MAINTENANCE: 'maintenance',
  DOCUMENTATION: 'documentation',
} as const;
