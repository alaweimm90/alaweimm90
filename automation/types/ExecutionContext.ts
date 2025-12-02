/**
 * Unified context for workflow execution across deployment targets
 */
export interface ExecutionContext {
  workflowId: string;
  stageId?: string;
  inputs: Record<string, unknown>;
  outputs: Map<string, unknown>;
  variables: Map<string, unknown>;
  checkpoints: Checkpoint[];
  startTime: Date;
  endTime?: Date;
  status: ExecutionStatus;
  metadata?: Record<string, unknown>;
  telemetry?: TelemetryData;
}

/**
 * Execution status enumeration
 */
export enum ExecutionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

/**
 * Checkpoint for resuming execution
 */
export interface Checkpoint {
  stageId: string;
  timestamp: Date;
  state: Record<string, unknown>;
  description?: string;
}

/**
 * Telemetry data for execution monitoring
 */
export interface TelemetryData {
  executionTime: number;
  stagesExecuted: number;
  errors: string[];
  warnings: string[];
  customMetrics?: Record<string, number>;
}
