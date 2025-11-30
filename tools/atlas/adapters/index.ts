/**
 * ATLAS Adapter Registry
 * Unified executor that connects agents to their adapters
 */

import { Agent, Task, TaskResult, AgentProvider } from '../types/index.js';
import { agentRegistry } from '../agents/registry.js';
import { createRouter, TaskRouter } from '../orchestration/router.js';
import { fallbackManager } from '../orchestration/fallback.js';
import { LLMAdapter } from './base.js';
import { createAnthropicAdapter } from './anthropic.js';
import { createOpenAIAdapter } from './openai.js';
import { createGoogleAdapter } from './google.js';

// ============================================================================
// Adapter Factory
// ============================================================================

type AdapterFactory = (agent: Agent) => LLMAdapter;

const ADAPTER_FACTORIES: Record<AgentProvider, AdapterFactory> = {
  anthropic: createAnthropicAdapter,
  openai: createOpenAIAdapter,
  google: createGoogleAdapter,
  local: () => {
    throw new Error('Local adapter not implemented');
  },
  custom: () => {
    throw new Error('Custom adapter requires explicit configuration');
  },
};

/**
 * Create an adapter for an agent
 */
export function createAdapter(agent: Agent): LLMAdapter {
  const factory = ADAPTER_FACTORIES[agent.provider];
  if (!factory) {
    throw new Error(`No adapter factory for provider: ${agent.provider}`);
  }
  return factory(agent);
}

// ============================================================================
// Adapter Cache
// ============================================================================

const adapterCache = new Map<string, LLMAdapter>();

/**
 * Get or create an adapter for an agent
 */
function getAdapter(agent: Agent): LLMAdapter {
  let adapter = adapterCache.get(agent.id);
  if (!adapter) {
    adapter = createAdapter(agent);
    adapterCache.set(agent.id, adapter);
  }
  return adapter;
}

/**
 * Clear adapter cache
 */
export function clearAdapterCache(): void {
  adapterCache.clear();
}

// ============================================================================
// Task Executor
// ============================================================================

export interface ExecutorConfig {
  router?: TaskRouter;
  maxRetries?: number;
  useFallback?: boolean;
}

export interface ExecutionResult {
  success: boolean;
  result?: TaskResult;
  agentUsed?: string;
  attempts: number;
  errors: string[];
}

/**
 * Execute a task with automatic agent selection and fallback
 */
export async function executeTask(
  task: Task,
  config: ExecutorConfig = {}
): Promise<ExecutionResult> {
  const { router = createRouter(), maxRetries = 3, useFallback = true } = config;

  const errors: string[] = [];
  let attempts = 0;

  // Route to best agent
  const decision = useFallback ? router.routeWithFallback(task) : router.route(task);

  if (!decision) {
    return {
      success: false,
      attempts: 0,
      errors: ['No agent available for this task type'],
    };
  }

  // Try with fallback support
  const tryExecution = async (agentId: string): Promise<TaskResult | null> => {
    const agent = agentRegistry.get(agentId);
    if (!agent) {
      errors.push(`Agent ${agentId} not found`);
      return null;
    }

    const adapter = getAdapter(agent);
    if (!adapter.isConfigured()) {
      errors.push(`Agent ${agentId} not configured (missing API key)`);
      return null;
    }

    attempts++;

    try {
      const result = await adapter.executeTask(task);

      if (result.success) {
        // Record success
        fallbackManager.recordSuccess(agentId, result.latency || 0, result.tokensUsed || 0);
        return result;
      } else {
        errors.push(`Agent ${agentId}: ${result.error || 'Unknown error'}`);
        fallbackManager.recordFailure(agentId);
        return null;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.push(`Agent ${agentId}: ${message}`);
      fallbackManager.recordFailure(agentId);
      return null;
    }
  };

  // Try primary agent
  let result = await tryExecution(decision.agentId);
  if (result) {
    return {
      success: true,
      result,
      agentUsed: decision.agentId,
      attempts,
      errors,
    };
  }

  // Try alternatives if available
  if (useFallback && decision.alternatives) {
    for (const alt of decision.alternatives) {
      if (attempts >= maxRetries) break;

      result = await tryExecution(alt.agentId);
      if (result) {
        return {
          success: true,
          result,
          agentUsed: alt.agentId,
          attempts,
          errors,
        };
      }
    }
  }

  // Try fallback chain
  if (useFallback) {
    const fallbackResult = await fallbackManager.executeWithFallback(
      task,
      async (agent) => {
        const adapter = getAdapter(agent);
        if (!adapter.isConfigured()) {
          throw new Error('Not configured');
        }
        attempts++;
        const taskResult = await adapter.executeTask(task);
        if (!taskResult.success) {
          throw new Error(taskResult.error || 'Task failed');
        }
        return taskResult;
      },
      decision.agentId
    );

    if (fallbackResult) {
      return {
        success: true,
        result: fallbackResult.result,
        agentUsed: fallbackResult.agentUsed,
        attempts,
        errors,
      };
    }
  }

  return {
    success: false,
    attempts,
    errors,
  };
}

// ============================================================================
// Quick Execution Helpers
// ============================================================================

/**
 * Quick code generation
 */
export async function generateCode(
  description: string,
  language?: string,
  context?: string
): Promise<ExecutionResult> {
  const task: Task = {
    id: `gen-${Date.now()}`,
    type: 'code_generation',
    description,
    context: {
      language,
      additionalContext: context,
    },
    priority: 'medium',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  return executeTask(task);
}

/**
 * Quick code review
 */
export async function reviewCode(
  code: string,
  language?: string,
  context?: string
): Promise<ExecutionResult> {
  const task: Task = {
    id: `review-${Date.now()}`,
    type: 'code_review',
    description: 'Review this code for bugs, security issues, and improvements',
    context: {
      codeSnippet: code,
      language,
      additionalContext: context,
    },
    priority: 'medium',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  return executeTask(task);
}

/**
 * Quick explanation
 */
export async function explainCode(code: string, language?: string): Promise<ExecutionResult> {
  const task: Task = {
    id: `explain-${Date.now()}`,
    type: 'explanation',
    description: 'Explain what this code does',
    context: {
      codeSnippet: code,
      language,
    },
    priority: 'low',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  return executeTask(task);
}

/**
 * Quick chat
 */
export async function chat(message: string): Promise<ExecutionResult> {
  const task: Task = {
    id: `chat-${Date.now()}`,
    type: 'chat',
    description: message,
    context: {},
    priority: 'low',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  return executeTask(task);
}

// ============================================================================
// Status and Diagnostics
// ============================================================================

export interface AdapterStatus {
  agentId: string;
  provider: AgentProvider;
  configured: boolean;
  rateLimitStatus: {
    remaining: number;
    limit: number;
    isLimited: boolean;
  };
}

/**
 * Get status of all adapters
 */
export function getAdapterStatuses(): AdapterStatus[] {
  const agents = agentRegistry.getAll();

  return agents.map((agent) => {
    try {
      const adapter = getAdapter(agent);
      return {
        agentId: agent.id,
        provider: agent.provider,
        configured: adapter.isConfigured(),
        rateLimitStatus: adapter.getRateLimitStatus(),
      };
    } catch {
      return {
        agentId: agent.id,
        provider: agent.provider,
        configured: false,
        rateLimitStatus: {
          remaining: 0,
          limit: 0,
          isLimited: true,
        },
      };
    }
  });
}

// ============================================================================
// Re-exports
// ============================================================================

export { LLMAdapter, BaseAdapter, CompletionRequest, CompletionResponse } from './base.js';
export { AnthropicAdapter, createAnthropicAdapter } from './anthropic.js';
export { OpenAIAdapter, createOpenAIAdapter } from './openai.js';
export { GoogleAdapter, createGoogleAdapter } from './google.js';
