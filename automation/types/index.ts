// Unified TypeScript interfaces for consolidated automation system

// Export new unified types
// Export enums as values (they need to be available at runtime)
export { DeploymentTarget } from './DeploymentTarget';
export { ExecutionStatus } from './ExecutionContext';

// Export interfaces and types as type-only exports
export type { DeploymentConfig, ResourceRequirements, ScalingConfig, NetworkConfig, SecurityConfig } from './DeploymentTarget';
export type { PromptConfig, PromptCollection } from './PromptConfig';
export type { ExecutionContext, Checkpoint, TelemetryData } from './ExecutionContext';

// Extended AgentTemplate interface (extends existing Agent from TypeScript system)
export interface AgentTemplate {
  role: string;
  goal: string;
  backstory: string;
  tools?: string[];
  llm_config?: LLMConfig;
  verbose?: boolean;
  allow_delegation?: boolean;
  // Extended fields for unified system
  template_name?: string;
  template_version?: string;
  description?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  deployment_targets?: string[];
  validation_rules?: string[];
}

// Extended ValidationRule interface (extends existing ValidationResult)
export interface ValidationRule {
  name: string;
  description?: string;
  severity: ValidationSeverity;
  condition: string;
  error_message: string;
  suggestion?: string;
  category?: string;
  applies_to?: string[]; // ['agent', 'workflow', 'prompt', 'deployment']
}

export enum ValidationSeverity {
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info'
}

// Existing TypeScript system types (imported and re-exported for compatibility)
export interface Agent {
  role: string;
  goal: string;
  backstory: string;
  tools?: string[];
  llm_config?: LLMConfig;
  verbose?: boolean;
  allow_delegation?: boolean;
}

export interface LLMConfig {
  model: string;
  temperature?: number;
  max_tokens?: number;
}

export interface WorkflowStage {
  name: string;
  agent?: string;
  action: string;
  inputs?: string[];
  outputs?: string[];
  condition?: string;
  parallel?: boolean;
  loop?: boolean;
  depends_on?: string[];
  gate?: string;
}

export interface Workflow {
  name: string;
  description: string;
  pattern: string;
  source?: string;
  stages: WorkflowStage[];
  orchestrator?: string;
  workers?: string[];
  agents?: string[];
  success_criteria?: string[];
  timeout_minutes?: number;
  termination?: Array<{ condition: string; value?: number }>;
}

export interface Prompt {
  path: string;
  name: string;
  category: 'system' | 'project' | 'tasks';
  size: number;
}

export interface OrchestrationPattern {
  name: string;
  description: string;
  use_cases: string[];
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  type: string;
  message: string;
  path?: string;
}

export interface ValidationWarning {
  type: string;
  message: string;
  path?: string;
}

export interface RouteResult {
  task_type: string;
  confidence: number;
  recommended_tools: string[];
  suggested_agents: string[];
}

export interface DeploymentProject {
  name: string;
  path: string;
  type: string;
  organization: string;
  technologies: string[];
  status: string;
}

export interface DeploymentTemplate {
  name: string;
  description: string;
  platform: string;
  files: string[];
}

export interface ExecutionResult {
  success: boolean;
  outputs: Map<string, unknown>;
  duration_ms: number;
  stages_completed: string[];
  error?: string;
}
