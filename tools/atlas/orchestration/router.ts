/**
 * ATLAS Task Router
 * Intelligent task routing based on capability matching, performance, and cost
 */

import {
  Task,
  TaskType,
  RoutingDecision,
  RoutingStrategy,
  OrchestrationConfig,
  Agent,
  AgentCapability,
} from '../types/index.js';
import { agentRegistry } from '../agents/registry.js';
import { routeToDevOpsAgent, loadDevOpsAgents } from './devops-agents.js';

// ============================================================================
// Capability to Task Type Mapping
// ============================================================================

const TASK_TO_CAPABILITY: Record<TaskType, AgentCapability> = {
  code_generation: 'code_generation',
  code_review: 'code_review',
  refactoring: 'refactoring',
  documentation: 'documentation',
  testing: 'testing',
  debugging: 'debugging',
  analysis: 'analysis',
  explanation: 'explanation',
  chat: 'chat',
};

// ============================================================================
// Routing Strategies
// ============================================================================

/**
 * Capability-based routing: Select agent with best capability match
 */
const capabilityStrategy: RoutingStrategy = {
  name: 'capability',
  description: 'Select agent based on capability match and performance',
  selectAgent: (task: Task, agents: Agent[]): RoutingDecision | null => {
    const requiredCapability = TASK_TO_CAPABILITY[task.type];
    const capable = agents.filter(
      (a) => a.status === 'available' && a.capabilities.includes(requiredCapability)
    );

    if (capable.length === 0) return null;

    // Sort by success rate, then by latency
    const sorted = capable.sort((a, b) => {
      const successDiff =
        b.metrics.successfulRequests / Math.max(b.metrics.totalRequests, 1) -
        a.metrics.successfulRequests / Math.max(a.metrics.totalRequests, 1);
      if (Math.abs(successDiff) > 0.1) return successDiff;
      return a.metrics.avgLatency - b.metrics.avgLatency;
    });

    const selected = sorted[0];
    const alternatives = sorted.slice(1, 3).map((a) => ({
      agentId: a.id,
      confidence: calculateConfidence(a, requiredCapability),
      reason: `${a.name} also supports ${requiredCapability}`,
    }));

    return {
      agentId: selected.id,
      confidence: calculateConfidence(selected, requiredCapability),
      reasoning: `Selected ${selected.name} for ${task.type} based on capability match and performance`,
      estimatedCost: estimateCost(selected, task),
      estimatedTime: estimateTime(selected),
      alternatives,
    };
  },
};

/**
 * Load-balanced routing: Distribute tasks across available agents
 */
const loadBalanceStrategy: RoutingStrategy = {
  name: 'load_balance',
  description: 'Distribute tasks evenly across available agents',
  selectAgent: (task: Task, agents: Agent[]): RoutingDecision | null => {
    const requiredCapability = TASK_TO_CAPABILITY[task.type];
    const capable = agents.filter(
      (a) => a.status === 'available' && a.capabilities.includes(requiredCapability)
    );

    if (capable.length === 0) return null;

    // Sort by last used (least recently used first)
    const sorted = capable.sort((a, b) => {
      const aLastUsed = a.lastUsed ? new Date(a.lastUsed).getTime() : 0;
      const bLastUsed = b.lastUsed ? new Date(b.lastUsed).getTime() : 0;
      return aLastUsed - bLastUsed;
    });

    const selected = sorted[0];

    return {
      agentId: selected.id,
      confidence: calculateConfidence(selected, requiredCapability),
      reasoning: `Selected ${selected.name} for load balancing (least recently used)`,
      estimatedCost: estimateCost(selected, task),
      estimatedTime: estimateTime(selected),
    };
  },
};

/**
 * Cost-optimized routing: Select cheapest capable agent
 */
const costStrategy: RoutingStrategy = {
  name: 'cost',
  description: 'Select the cheapest agent that can handle the task',
  selectAgent: (task: Task, agents: Agent[]): RoutingDecision | null => {
    const requiredCapability = TASK_TO_CAPABILITY[task.type];
    const capable = agents.filter(
      (a) => a.status === 'available' && a.capabilities.includes(requiredCapability)
    );

    if (capable.length === 0) return null;

    // Sort by cost per token
    const sorted = capable.sort((a, b) => {
      const costA = a.config.costPerToken ?? Infinity;
      const costB = b.config.costPerToken ?? Infinity;
      return costA - costB;
    });

    const selected = sorted[0];

    return {
      agentId: selected.id,
      confidence: calculateConfidence(selected, requiredCapability),
      reasoning: `Selected ${selected.name} as lowest cost option`,
      estimatedCost: estimateCost(selected, task),
      estimatedTime: estimateTime(selected),
    };
  },
};

/**
 * Latency-optimized routing: Select fastest capable agent
 */
const latencyStrategy: RoutingStrategy = {
  name: 'latency',
  description: 'Select the fastest agent based on historical latency',
  selectAgent: (task: Task, agents: Agent[]): RoutingDecision | null => {
    const requiredCapability = TASK_TO_CAPABILITY[task.type];
    const capable = agents.filter(
      (a) => a.status === 'available' && a.capabilities.includes(requiredCapability)
    );

    if (capable.length === 0) return null;

    // Sort by average latency (new agents get benefit of doubt)
    const sorted = capable.sort((a, b) => {
      const latA = a.metrics.totalRequests > 0 ? a.metrics.avgLatency : 1000;
      const latB = b.metrics.totalRequests > 0 ? b.metrics.avgLatency : 1000;
      return latA - latB;
    });

    const selected = sorted[0];

    return {
      agentId: selected.id,
      confidence: calculateConfidence(selected, requiredCapability),
      reasoning: `Selected ${selected.name} as fastest option (avg: ${selected.metrics.avgLatency.toFixed(0)}ms)`,
      estimatedCost: estimateCost(selected, task),
      estimatedTime: estimateTime(selected),
    };
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

function calculateConfidence(agent: Agent, capability: AgentCapability): number {
  let confidence = 0.5; // Base confidence

  // Has the capability
  if (agent.capabilities.includes(capability)) {
    confidence += 0.2;
  }

  // Good success rate
  if (agent.metrics.totalRequests > 0) {
    const successRate = agent.metrics.successfulRequests / agent.metrics.totalRequests;
    confidence += successRate * 0.2;
  } else {
    confidence += 0.1; // New agents get some confidence
  }

  // API key configured
  if (process.env[agent.config.apiKeyEnv || '']) {
    confidence += 0.1;
  }

  return Math.min(confidence, 1.0);
}

function estimateCost(agent: Agent, _task: Task): number {
  const baseTokens = 1000; // Estimate for a typical task
  const costPerToken = agent.config.costPerToken ?? 0;
  return baseTokens * costPerToken;
}

function estimateTime(agent: Agent): number {
  // Use historical average or default
  return agent.metrics.avgLatency > 0 ? agent.metrics.avgLatency : 5000;
}

// ============================================================================
// Task Router Implementation
// ============================================================================

const STRATEGIES: Record<string, RoutingStrategy> = {
  capability: capabilityStrategy,
  load_balance: loadBalanceStrategy,
  cost: costStrategy,
  latency: latencyStrategy,
};

export class TaskRouter {
  private config: OrchestrationConfig;
  private strategy: RoutingStrategy;

  constructor(config: OrchestrationConfig) {
    this.config = config;
    this.strategy = STRATEGIES[config.routing.strategy] || capabilityStrategy;
  }

  /**
   * Route a task to the best available agent
   */
  route(task: Task): RoutingDecision | null {
    // First, check if this is a DevOps task
    const devOpsRouting = routeToDevOpsAgent(task);
    if (devOpsRouting) {
      return devOpsRouting;
    }

    const agents = agentRegistry.getAll();

    // Check for task-specific preference
    if (this.config.routing.preferences?.[task.type]) {
      const preferredId = this.config.routing.preferences[task.type];
      const preferred = agents.find((a) => a.id === preferredId && a.status === 'available');
      if (preferred) {
        return {
          agentId: preferred.id,
          confidence: 0.95,
          reasoning: `Using preferred agent for ${task.type}`,
          estimatedCost: estimateCost(preferred, task),
          estimatedTime: estimateTime(preferred),
        };
      }
    }

    // Use configured strategy
    return this.strategy.selectAgent(task, agents);
  }

  /**
   * Route with fallback chain
   */
  routeWithFallback(task: Task): RoutingDecision | null {
    // Try primary routing
    const primary = this.route(task);
    if (primary) return primary;

    // Try fallback chain
    for (const agentId of this.config.fallbackChain) {
      const agent = agentRegistry.get(agentId);
      if (agent && agent.status === 'available') {
        const capability = TASK_TO_CAPABILITY[task.type];
        if (agent.capabilities.includes(capability)) {
          return {
            agentId: agent.id,
            confidence: 0.5,
            reasoning: `Fallback to ${agent.name} from fallback chain`,
            estimatedCost: estimateCost(agent, task),
            estimatedTime: estimateTime(agent),
          };
        }
      }
    }

    return null;
  }

  /**
   * Get routing explanation for a task
   */
  explain(task: Task): string {
    const decision = this.route(task);
    if (!decision) {
      return `No available agent can handle task type: ${task.type}`;
    }

    const agent = agentRegistry.get(decision.agentId);
    if (!agent) {
      return `Agent ${decision.agentId} not found in registry`;
    }

    let explanation = `Task: ${task.type}\n`;
    explanation += `Strategy: ${this.strategy.name}\n`;
    explanation += `Selected: ${agent.name} (${agent.provider})\n`;
    explanation += `Confidence: ${(decision.confidence * 100).toFixed(0)}%\n`;
    explanation += `Reasoning: ${decision.reasoning}\n`;
    explanation += `Estimated Cost: $${decision.estimatedCost.toFixed(4)}\n`;
    explanation += `Estimated Time: ${decision.estimatedTime}ms\n`;

    if (decision.alternatives?.length) {
      explanation += `\nAlternatives:\n`;
      for (const alt of decision.alternatives) {
        explanation += `  - ${alt.agentId}: ${alt.reason}\n`;
      }
    }

    return explanation;
  }

  /**
   * Change routing strategy
   */
  setStrategy(strategyName: string): boolean {
    const strategy = STRATEGIES[strategyName];
    if (!strategy) return false;

    this.strategy = strategy;
    return true;
  }

  /**
   * Get available strategies
   */
  getStrategies(): string[] {
    return Object.keys(STRATEGIES);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createRouter(config?: Partial<OrchestrationConfig>): TaskRouter {
  const defaultConfig: OrchestrationConfig = {
    fallbackChain: ['claude-sonnet', 'claude-opus', 'gpt-4'],
    circuitBreaker: {
      failureThreshold: 5,
      successThreshold: 3,
      timeout: 60000,
      halfOpenRequests: 1,
    },
    routing: {
      strategy: 'capability',
    },
    telemetry: {
      enabled: true,
      metricsPath: '.atlas/metrics.json',
    },
    ...config,
  };

  // Initialize DevOps agents
  loadDevOpsAgents();

  return new TaskRouter(defaultConfig);
}
