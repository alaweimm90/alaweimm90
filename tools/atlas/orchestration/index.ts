/**
 * ATLAS Task Orchestrator
 * Core orchestration engine with routing, execution, and governance
 */

// Router - Intelligent task routing
export { TaskRouter, createRouter } from './router.js';

// Executor - Task execution engine
export { executeWorkflow } from './executor.js';

// Workflows - Workflow planning and execution
export { loadWorkflow, planWorkflow } from './workflows.js';
export type { WorkflowStep, WorkflowDef, PlannedStep, WorkflowPlan } from './workflows.js';

// Circuit Breaker - Failover and resilience
export { allowRequest, recordSuccess, recordFailure } from './circuit-breaker.js';
export type { CircuitBreakerConfig, CircuitBreakerState } from '@atlas/types/index.js';

// Fallback - Fallback strategies
export { FallbackManager, fallbackManager } from './fallback.js';

// Governance - Policy enforcement
export {
  governance,
  preTaskCheck,
  postTaskCheck,
  checkFilePath,
  checkDirectoryPath,
  type GovernanceCheck,
  type GovernanceViolation,
} from './governance.js';

// DevOps Agents - 20 essential DevOps automation agents
export {
  devOpsOrchestration,
  loadDevOpsAgents,
  getDevOpsAgent,
  listDevOpsAgents,
  getAgentsByCategory,
  executeDevOpsWorkflow,
  routeToDevOpsAgent,
  CI_CD_PIPELINE,
  SECURE_RELEASE,
  INCIDENT_RESPONSE,
  type DevOpsAgent,
  type DevOpsAgentId,
  type DevOpsWorkflow,
  type DevOpsWorkflowStep,
  type DevOpsExecutionResult,
} from './devops-agents.js';

// Re-export all types
export type {
  Task,
  TaskType,
  RoutingDecision,
  RoutingStrategy,
  OrchestrationConfig,
} from '@atlas/types/index.js';
