/**
 * ATLAS Task Orchestrator
 * Core orchestration engine with routing, execution, and governance
 */

// Router - Intelligent task routing
export { router, route, getRoutingStrategies } from './router.js';

// Executor - Task execution engine
export { executor, executeTask, type ExecutionResult } from './executor.js';

// Workflows - Workflow planning and execution
export { workflowPlanner, planWorkflow, executeWorkflow } from './workflows.js';

// Circuit Breaker - Failover and resilience
export { circuitBreaker, CircuitState } from './circuit-breaker.js';

// Fallback - Fallback strategies
export { FallbackManager, type FallbackResult } from './fallback.js';

// Adapters - LLM provider adapters
export { adapters, type LLMAdapter, type LLMResponse } from './adapters.js';

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
} from '../types/index.js';
