/**
 * ATLAS DevOps Agent Integration
 * Integrates the 20 essential DevOps agents with the ATLAS orchestration system
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { Task, TaskType, RoutingDecision, TaskResult } from '@atlas/types/index.js';
import { allowRequest, recordFailure, recordSuccess } from './circuit-breaker.js';

// ============================================================================
// DevOps Agent Types
// ============================================================================

export type DevOpsAgentId =
  | 'pipeline-orchestrator'
  | 'build'
  | 'test-runner'
  | 'artifact-repo'
  | 'container-build'
  | 'image-scan'
  | 'secrets'
  | 'infra-provisioner'
  | 'config-manager'
  | 'k8s-deploy'
  | 'progressive-delivery'
  | 'rollback'
  | 'metrics'
  | 'log-shipper'
  | 'alert-router'
  | 'triage'
  | 'release-manager'
  | 'feature-flags'
  | 'cost-monitor'
  | 'compliance-audit';

export interface DevOpsAgent {
  id: DevOpsAgentId;
  role: string;
  goal: string;
  aliases: string[];
  tools: string[];
  inputs: string[];
  outputs: string[];
  category: string;
}

export interface DevOpsWorkflowStep {
  id: string;
  agentId: DevOpsAgentId;
  action: string;
  inputs?: Record<string, unknown>;
  dependsOn?: string[];
  onError?: 'continue' | 'abort' | 'retry';
  maxRetries?: number;
  timeout?: number;
}

export interface DevOpsWorkflow {
  name: string;
  description: string;
  steps: DevOpsWorkflowStep[];
}

export interface DevOpsExecutionResult {
  workflowName: string;
  success: boolean;
  stepResults: Map<string, TaskResult>;
  totalDuration: number;
  failedSteps: string[];
}

// ============================================================================
// Agent Registry
// ============================================================================

const devOpsAgentRegistry: Map<DevOpsAgentId, DevOpsAgent> = new Map();
const aliasMap: Map<string, DevOpsAgentId> = new Map();

/**
 * Load DevOps agents from YAML configuration
 */
export function loadDevOpsAgents(configPath?: string): void {
  const defaultPath = path.resolve(process.cwd(), 'automation/agents/config/devops-agents.yaml');
  const yamlPath = configPath || defaultPath;

  if (!fs.existsSync(yamlPath)) {
    console.warn(`DevOps agents config not found at ${yamlPath}`);
    return;
  }

  const content = fs.readFileSync(yamlPath, 'utf8');
  const config = yaml.load(content) as {
    agents: Record<string, Partial<DevOpsAgent>>;
    categories: Record<string, { agents: string[] }>;
  };

  // Build category lookup
  const categoryLookup: Map<string, string> = new Map();
  for (const [category, data] of Object.entries(config.categories || {})) {
    for (const agentName of data.agents || []) {
      categoryLookup.set(agentName, category);
    }
  }

  // Load agents
  for (const [name, agentData] of Object.entries(config.agents || {})) {
    const id = (agentData.id || name) as DevOpsAgentId;
    const agent: DevOpsAgent = {
      id,
      role: agentData.role || name,
      goal: agentData.goal || '',
      aliases: agentData.aliases || [],
      tools: agentData.tools || [],
      inputs: agentData.inputs || [],
      outputs: agentData.outputs || [],
      category: categoryLookup.get(name) || 'pipeline',
    };

    devOpsAgentRegistry.set(id, agent);

    // Register aliases
    for (const alias of agent.aliases) {
      aliasMap.set(alias, id);
    }
  }
}

/**
 * Get a DevOps agent by ID or alias
 */
export function getDevOpsAgent(idOrAlias: string): DevOpsAgent | undefined {
  // Try direct ID lookup
  const agent = devOpsAgentRegistry.get(idOrAlias as DevOpsAgentId);
  if (agent) return agent;

  // Try alias lookup
  const resolvedId = aliasMap.get(idOrAlias);
  if (resolvedId) {
    return devOpsAgentRegistry.get(resolvedId);
  }

  return undefined;
}

/**
 * List all DevOps agents
 */
export function listDevOpsAgents(): DevOpsAgent[] {
  return Array.from(devOpsAgentRegistry.values());
}

/**
 * Get agents by category
 */
export function getAgentsByCategory(category: string): DevOpsAgent[] {
  return Array.from(devOpsAgentRegistry.values()).filter((a) => a.category === category);
}

// ============================================================================
// Task Type Mapping
// ============================================================================

/**
 * Map DevOps agent IDs to ATLAS task types
 */
const DEVOPS_TO_TASK_TYPE: Record<DevOpsAgentId, TaskType> = {
  'pipeline-orchestrator': 'code_generation',
  build: 'code_generation',
  'test-runner': 'testing',
  'artifact-repo': 'code_generation',
  'container-build': 'code_generation',
  'image-scan': 'analysis',
  secrets: 'code_generation',
  'infra-provisioner': 'code_generation',
  'config-manager': 'code_generation',
  'k8s-deploy': 'code_generation',
  'progressive-delivery': 'code_generation',
  rollback: 'code_generation',
  metrics: 'analysis',
  'log-shipper': 'code_generation',
  'alert-router': 'code_generation',
  triage: 'analysis',
  'release-manager': 'documentation',
  'feature-flags': 'code_generation',
  'cost-monitor': 'analysis',
  'compliance-audit': 'analysis',
};

/**
 * Create an ATLAS task from DevOps workflow step
 */
export function createTaskFromStep(step: DevOpsWorkflowStep): Task {
  return {
    id: step.id,
    type: DEVOPS_TO_TASK_TYPE[step.agentId] || 'code_generation',
    description: `Execute ${step.agentId}: ${step.action}`,
    context: {
      additionalContext: JSON.stringify(step.inputs || {}),
    },
    priority: 'medium' as const,
    status: 'pending' as const,
    createdAt: new Date().toISOString(),
    metadata: {
      devOpsAgentId: step.agentId,
      action: step.action,
      dependsOn: step.dependsOn,
      inputs: step.inputs || {},
    },
  };
}

// ============================================================================
// Workflow Execution
// ============================================================================

/**
 * Execute a DevOps workflow using ATLAS orchestration
 */
export async function executeDevOpsWorkflow(
  workflow: DevOpsWorkflow,
  circuitBreakerConfig: {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    halfOpenRequests: number;
  }
): Promise<DevOpsExecutionResult> {
  const startTime = Date.now();
  const stepResults = new Map<string, TaskResult>();
  const completed = new Set<string>();
  const failed: string[] = [];
  const pending = new Map(workflow.steps.map((s) => [s.id, s]));

  while (pending.size > 0) {
    // Find steps with all dependencies met
    const ready = Array.from(pending.values()).filter((step) => {
      const deps = step.dependsOn || [];
      return deps.every((d) => completed.has(d));
    });

    if (ready.length === 0 && pending.size > 0) {
      // Deadlock or all remaining steps have failed dependencies
      break;
    }

    // Execute ready steps (could be parallelized)
    for (const step of ready) {
      const result = await executeDevOpsStep(step, circuitBreakerConfig);
      stepResults.set(step.id, result);
      pending.delete(step.id);

      if (result.success) {
        completed.add(step.id);
      } else {
        failed.push(step.id);
        if (step.onError === 'abort') {
          // Clear remaining pending steps
          pending.clear();
          break;
        }
      }
    }
  }

  return {
    workflowName: workflow.name,
    success: failed.length === 0,
    stepResults,
    totalDuration: Date.now() - startTime,
    failedSteps: failed,
  };
}

/**
 * Execute a single DevOps workflow step
 */
async function executeDevOpsStep(
  step: DevOpsWorkflowStep,
  cbConfig: {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    halfOpenRequests: number;
  }
): Promise<TaskResult> {
  const agentId = step.agentId;
  const maxRetries = step.maxRetries || 1;
  let attempt = 0;
  let lastResult: TaskResult = {
    success: false,
    error: 'not_run',
    agentId,
  };

  while (attempt < maxRetries) {
    attempt++;

    // Check circuit breaker
    if (!allowRequest(agentId, cbConfig)) {
      return {
        success: false,
        error: 'circuit_open',
        agentId,
      };
    }

    const startTime = Date.now();

    try {
      // Execute the agent action
      const output = await executeAgentAction(step);
      const latency = Date.now() - startTime;

      recordSuccess(agentId, cbConfig);

      lastResult = {
        success: true,
        output: typeof output === 'string' ? output : JSON.stringify(output),
        latency,
        agentId,
      };
      break;
    } catch (error) {
      const latency = Date.now() - startTime;
      recordFailure(agentId, cbConfig);

      lastResult = {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        latency,
        agentId,
      };

      if (step.onError !== 'retry') {
        break;
      }
    }
  }

  return lastResult;
}

/**
 * Execute a specific agent action
 */
async function executeAgentAction(step: DevOpsWorkflowStep): Promise<unknown> {
  const agent = getDevOpsAgent(step.agentId);
  if (!agent) {
    throw new Error(`DevOps agent not found: ${step.agentId}`);
  }

  // This is a simulation - in a real implementation, this would:
  // 1. Route to the appropriate executor
  // 2. Pass inputs and collect outputs
  // 3. Handle timeouts and retries

  console.log(`Executing ${agent.role}: ${step.action}`);

  // Simulate execution time
  await new Promise((resolve) => setTimeout(resolve, 100));

  return {
    status: 'completed',
    agentId: step.agentId,
    action: step.action,
    timestamp: new Date().toISOString(),
  };
}

// ============================================================================
// Pre-built Workflows
// ============================================================================

/**
 * Standard CI/CD pipeline workflow
 */
export const CI_CD_PIPELINE: DevOpsWorkflow = {
  name: 'ci-cd-pipeline',
  description: 'Full CI/CD pipeline: build, test, scan, deploy',
  steps: [
    {
      id: 'fetch-secrets',
      agentId: 'secrets',
      action: 'fetch',
      inputs: { secretsRef: 'kv://ci/prod' },
    },
    {
      id: 'build',
      agentId: 'build',
      action: 'compile',
      dependsOn: ['fetch-secrets'],
      inputs: { sourcePath: '.', buildSpec: 'buildspec.yml' },
    },
    {
      id: 'test',
      agentId: 'test-runner',
      action: 'run',
      dependsOn: ['build'],
      inputs: { testSpec: 'tests.yml', coverage: true },
    },
    {
      id: 'container-build',
      agentId: 'container-build',
      action: 'build',
      dependsOn: ['test'],
      inputs: { dockerfile: 'Dockerfile', context: '.' },
    },
    {
      id: 'image-scan',
      agentId: 'image-scan',
      action: 'scan',
      dependsOn: ['container-build'],
      inputs: { policy: { maxSeverity: 'high' } },
      onError: 'abort',
    },
    {
      id: 'deploy',
      agentId: 'k8s-deploy',
      action: 'apply',
      dependsOn: ['image-scan'],
      inputs: { clusterRef: 'prod', namespace: 'app' },
    },
    {
      id: 'canary',
      agentId: 'progressive-delivery',
      action: 'rollout',
      dependsOn: ['deploy'],
      inputs: { strategy: 'canary', weights: [10, 30, 60, 100] },
    },
  ],
};

/**
 * Security-focused release workflow
 */
export const SECURE_RELEASE: DevOpsWorkflow = {
  name: 'secure-release',
  description: 'Security-focused release pipeline with compliance checks',
  steps: [
    {
      id: 'secrets',
      agentId: 'secrets',
      action: 'fetch',
      inputs: { secretsRef: 'kv://release/prod', ttl: 900 },
    },
    {
      id: 'build',
      agentId: 'build',
      action: 'compile',
      dependsOn: ['secrets'],
    },
    {
      id: 'container',
      agentId: 'container-build',
      action: 'build',
      dependsOn: ['build'],
    },
    {
      id: 'vuln-scan',
      agentId: 'image-scan',
      action: 'scan',
      dependsOn: ['container'],
      inputs: { policy: { maxSeverity: 'medium', failOnFix: true } },
      onError: 'abort',
    },
    {
      id: 'compliance',
      agentId: 'compliance-audit',
      action: 'audit',
      dependsOn: ['vuln-scan'],
      inputs: { policiesDir: 'policies', scope: 'prod' },
      onError: 'abort',
    },
    {
      id: 'release',
      agentId: 'release-manager',
      action: 'create',
      dependsOn: ['compliance'],
      inputs: { publish: true },
    },
  ],
};

/**
 * Incident response workflow
 */
export const INCIDENT_RESPONSE: DevOpsWorkflow = {
  name: 'incident-response',
  description: 'Automated incident detection and response',
  steps: [
    {
      id: 'collect-metrics',
      agentId: 'metrics',
      action: 'collect',
      inputs: { targetsFile: 'scrape.yml' },
    },
    {
      id: 'triage',
      agentId: 'triage',
      action: 'classify',
      dependsOn: ['collect-metrics'],
      inputs: { runbooksDir: 'runbooks' },
    },
    {
      id: 'alert',
      agentId: 'alert-router',
      action: 'notify',
      dependsOn: ['triage'],
      inputs: { rulesFile: 'alerts/routes.yml' },
    },
    {
      id: 'evaluate-rollback',
      agentId: 'rollback',
      action: 'evaluate',
      dependsOn: ['triage'],
      inputs: { rollbackPlan: 'rollback.yml' },
    },
  ],
};

// ============================================================================
// Route DevOps Tasks
// ============================================================================

/**
 * Route a task to the appropriate DevOps agent
 */
export function routeToDevOpsAgent(task: Task): RoutingDecision | null {
  // Check if this is a DevOps task
  const metadata = task.metadata as { devOpsAgentId?: string } | undefined;
  if (!metadata?.devOpsAgentId) {
    return null;
  }

  const agent = getDevOpsAgent(metadata.devOpsAgentId);
  if (!agent) {
    return null;
  }

  return {
    agentId: agent.id,
    confidence: 0.95,
    reasoning: `Routing to DevOps agent: ${agent.role}`,
    estimatedCost: 0,
    estimatedTime: 5000,
  };
}

// ============================================================================
// Exports
// ============================================================================

export const devOpsOrchestration = {
  loadAgents: loadDevOpsAgents,
  getAgent: getDevOpsAgent,
  listAgents: listDevOpsAgents,
  getByCategory: getAgentsByCategory,
  executeWorkflow: executeDevOpsWorkflow,
  routeTask: routeToDevOpsAgent,
  workflows: {
    ciCdPipeline: CI_CD_PIPELINE,
    secureRelease: SECURE_RELEASE,
    incidentResponse: INCIDENT_RESPONSE,
  },
};

export default devOpsOrchestration;
