/**
 * Core types for Agent framework
 */

export interface AgentConfig {
  id: string;
  name: string;
  description: string;
  version: string;
  type: AgentType;
  capabilities: string[];
  requiredMcps?: string[];
  enabled: boolean;
  author?: string;
}

export enum AgentType {
  BASE = 'base',
  CODE = 'code',
  ANALYSIS = 'analysis',
  ORCHESTRATOR = 'orchestrator',
  CUSTOM = 'custom',
}

export interface AgentContext {
  workspaceRoot: string;
  projectName: string;
  environment: NodeJS.ProcessEnv;
  mcpServers: string[];
  metadata?: Record<string, unknown>;
}

export interface AgentTask {
  id: string;
  name: string;
  description: string;
  type: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  status: TaskStatus;
  agentId?: string;
}

export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface AgentResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: Error;
  duration: number;
}

export interface WorkflowStep {
  id: string;
  name: string;
  type: 'task' | 'conditional' | 'parallel' | 'sequential';
  agentId?: string;
  action?: string;
  inputs?: Record<string, unknown>;
  outputs?: string[];
  dependencies?: string[];
  condition?: string;
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  version: string;
  steps: WorkflowStep[];
  enabled: boolean;
}

export interface OrchestrationRule {
  id: string;
  name: string;
  description: string;
  trigger: string;
  actions: OrchestrationAction[];
  enabled: boolean;
}

export interface OrchestrationAction {
  type: 'execute' | 'schedule' | 'webhook';
  target: string;
  parameters?: Record<string, unknown>;
}
