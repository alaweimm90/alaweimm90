// Core Types for Automation System

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

export interface ExecutionContext {
  workflow: Workflow;
  agents: Map<string, Agent>;
  variables: Map<string, unknown>;
  checkpoints: string[];
}

export interface ExecutionResult {
  success: boolean;
  outputs: Map<string, unknown>;
  duration_ms: number;
  stages_completed: string[];
  error?: string;
}

export interface CrewMember {
  name: string;
  agent_ref: string;
  role_in_crew: string;
  responsibilities: string[];
  delegation_authority?: boolean;
  specialization?: Record<string, string[]>;
}

export interface CrewTask {
  name: string;
  description: string;
  assigned_to: string;
  depends_on?: string[];
  expected_output: string;
  priority: number;
}

export interface Crew {
  name: string;
  description: string;
  version: string;
  agents: CrewMember[];
  tasks: CrewTask[];
  process: {
    type: string;
    manager?: string;
    workflow: Array<{
      phase: string;
      tasks: string[];
      parallel: boolean;
    }>;
  };
  quality_gates?: Array<{
    name: string;
    condition: string;
    blocking: boolean;
  }>;
}
