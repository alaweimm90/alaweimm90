/**
 * ATLAS API Router
 * Route handlers for all API endpoints
 */

import { APIRequest, APIResponse } from './server.js';
import { agentRegistry } from '../agents/registry.js';
import { createRouter } from '../orchestration/router.js';
import { fallbackManager } from '../orchestration/fallback.js';
import {
  executeTask,
  generateCode,
  reviewCode,
  explainCode,
  chat,
  getAdapterStatuses,
} from '../adapters/index.js';
import { Task, TaskType, TaskPriority } from '../types/index.js';

// ============================================================================
// Route Definitions
// ============================================================================

type RouteHandler = (req: APIRequest) => Promise<APIResponse>;

interface Route {
  method: string;
  pattern: RegExp;
  handler: RouteHandler;
}

const routes: Route[] = [];

function addRoute(method: string, path: string, handler: RouteHandler): void {
  // Convert path pattern to regex (supports :param syntax)
  const pattern = new RegExp(
    '^' + path.replace(/:[a-zA-Z]+/g, '([^/]+)').replace(/\//g, '\\/') + '$'
  );
  routes.push({ method, pattern, handler });
}

// ============================================================================
// Health & Status Endpoints
// ============================================================================

addRoute('GET', '/', async () => ({
  status: 200,
  body: {
    name: 'ATLAS API',
    version: '0.1.0-alpha',
    status: 'running',
    endpoints: {
      health: 'GET /health',
      agents: 'GET /agents',
      execute: 'POST /execute',
      generate: 'POST /generate',
      review: 'POST /review',
      explain: 'POST /explain',
      chat: 'POST /chat',
    },
  },
}));

addRoute('GET', '/health', async () => {
  const agents = agentRegistry.getAll();
  const adapters = getAdapterStatuses();
  const health = fallbackManager.getHealth();

  return {
    status: 200,
    body: {
      status: health.unhealthy === 0 ? 'healthy' : health.healthy > 0 ? 'degraded' : 'unhealthy',
      timestamp: new Date().toISOString(),
      agents: {
        total: agents.length,
        available: agents.filter((a) => a.status === 'available').length,
        configured: adapters.filter((a) => a.configured).length,
      },
      circuits: {
        healthy: health.healthy,
        degraded: health.degraded,
        unhealthy: health.unhealthy,
      },
    },
  };
});

addRoute('GET', '/status', async () => {
  const agents = agentRegistry.getAll();
  const adapters = getAdapterStatuses();
  const circuits = fallbackManager.getStatus();

  return {
    status: 200,
    body: {
      agents: agents.map((agent) => {
        const adapter = adapters.find((a) => a.agentId === agent.id);
        const circuit = circuits[agent.id];

        return {
          id: agent.id,
          name: agent.name,
          provider: agent.provider,
          model: agent.model,
          status: agent.status,
          configured: adapter?.configured || false,
          circuitState: circuit?.state || 'unknown',
          metrics: {
            totalRequests: agent.metrics.totalRequests,
            successRate:
              agent.metrics.totalRequests > 0
                ? (agent.metrics.successfulRequests / agent.metrics.totalRequests) * 100
                : 0,
            avgLatency: agent.metrics.avgLatency,
          },
        };
      }),
      routing: {
        strategies: ['capability', 'load_balance', 'cost', 'latency'],
        defaultStrategy: 'capability',
      },
    },
  };
});

// ============================================================================
// Agent Management Endpoints
// ============================================================================

addRoute('GET', '/agents', async () => {
  const agents = agentRegistry.getAll();
  const adapters = getAdapterStatuses();

  return {
    status: 200,
    body: {
      agents: agents.map((agent) => {
        const adapter = adapters.find((a) => a.agentId === agent.id);
        return {
          id: agent.id,
          name: agent.name,
          provider: agent.provider,
          model: agent.model,
          capabilities: agent.capabilities,
          status: agent.status,
          configured: adapter?.configured || false,
          registeredAt: agent.registeredAt,
          lastUsed: agent.lastUsed,
        };
      }),
    },
  };
});

addRoute('GET', '/agents/:id', async (req) => {
  const match = req.path.match(/\/agents\/([^/]+)/);
  const id = match?.[1];

  if (!id) {
    return { status: 400, body: { error: 'Agent ID required' } };
  }

  const agent = agentRegistry.get(id);
  if (!agent) {
    return { status: 404, body: { error: `Agent '${id}' not found` } };
  }

  const adapters = getAdapterStatuses();
  const adapter = adapters.find((a) => a.agentId === id);
  const circuits = fallbackManager.getStatus();

  return {
    status: 200,
    body: {
      ...agent,
      configured: adapter?.configured || false,
      rateLimitStatus: adapter?.rateLimitStatus,
      circuitState: circuits[id]?.state || 'unknown',
    },
  };
});

addRoute('POST', '/agents/:id/reset-circuit', async (req) => {
  const match = req.path.match(/\/agents\/([^/]+)/);
  const id = match?.[1];

  if (!id) {
    return { status: 400, body: { error: 'Agent ID required' } };
  }

  const success = fallbackManager.resetCircuit(id);
  if (!success) {
    return { status: 404, body: { error: `Agent '${id}' not found or no circuit exists` } };
  }

  return {
    status: 200,
    body: { message: `Circuit breaker reset for agent '${id}'` },
  };
});

// ============================================================================
// Task Execution Endpoints
// ============================================================================

addRoute('POST', '/execute', async (req) => {
  const body = req.body as {
    type?: TaskType;
    description?: string;
    context?: {
      files?: string[];
      codeSnippet?: string;
      language?: string;
      framework?: string;
      additionalContext?: string;
    };
    priority?: TaskPriority;
    preferredAgent?: string;
  };

  if (!body.type || !body.description) {
    return {
      status: 400,
      body: { error: 'Missing required fields: type, description' },
    };
  }

  const task: Task = {
    id: `api-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: body.type,
    description: body.description,
    context: body.context || {},
    priority: body.priority || 'medium',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  const result = await executeTask(task);

  return {
    status: result.success ? 200 : 500,
    body: {
      taskId: task.id,
      success: result.success,
      agentUsed: result.agentUsed,
      attempts: result.attempts,
      result: result.result
        ? {
            output: result.result.output,
            tokensUsed: result.result.tokensUsed,
            latency: result.result.latency,
          }
        : undefined,
      errors: result.errors.length > 0 ? result.errors : undefined,
    },
  };
});

addRoute('POST', '/generate', async (req) => {
  const body = req.body as {
    description: string;
    language?: string;
    context?: string;
  };

  if (!body.description) {
    return { status: 400, body: { error: 'Missing required field: description' } };
  }

  const result = await generateCode(body.description, body.language, body.context);

  return {
    status: result.success ? 200 : 500,
    body: {
      success: result.success,
      agentUsed: result.agentUsed,
      code: result.result?.output,
      tokensUsed: result.result?.tokensUsed,
      latency: result.result?.latency,
      errors: result.errors.length > 0 ? result.errors : undefined,
    },
  };
});

addRoute('POST', '/review', async (req) => {
  const body = req.body as {
    code: string;
    language?: string;
    context?: string;
  };

  if (!body.code) {
    return { status: 400, body: { error: 'Missing required field: code' } };
  }

  const result = await reviewCode(body.code, body.language, body.context);

  return {
    status: result.success ? 200 : 500,
    body: {
      success: result.success,
      agentUsed: result.agentUsed,
      review: result.result?.output,
      tokensUsed: result.result?.tokensUsed,
      latency: result.result?.latency,
      errors: result.errors.length > 0 ? result.errors : undefined,
    },
  };
});

addRoute('POST', '/explain', async (req) => {
  const body = req.body as {
    code: string;
    language?: string;
  };

  if (!body.code) {
    return { status: 400, body: { error: 'Missing required field: code' } };
  }

  const result = await explainCode(body.code, body.language);

  return {
    status: result.success ? 200 : 500,
    body: {
      success: result.success,
      agentUsed: result.agentUsed,
      explanation: result.result?.output,
      tokensUsed: result.result?.tokensUsed,
      latency: result.result?.latency,
      errors: result.errors.length > 0 ? result.errors : undefined,
    },
  };
});

addRoute('POST', '/chat', async (req) => {
  const body = req.body as {
    message: string;
  };

  if (!body.message) {
    return { status: 400, body: { error: 'Missing required field: message' } };
  }

  const result = await chat(body.message);

  return {
    status: result.success ? 200 : 500,
    body: {
      success: result.success,
      agentUsed: result.agentUsed,
      response: result.result?.output,
      tokensUsed: result.result?.tokensUsed,
      latency: result.result?.latency,
      errors: result.errors.length > 0 ? result.errors : undefined,
    },
  };
});

// ============================================================================
// Routing Endpoints
// ============================================================================

addRoute('POST', '/route', async (req) => {
  const body = req.body as {
    type: TaskType;
    strategy?: 'capability' | 'load_balance' | 'cost' | 'latency';
  };

  if (!body.type) {
    return { status: 400, body: { error: 'Missing required field: type' } };
  }

  const router = createRouter({
    routing: { strategy: body.strategy || 'capability' },
  });

  const task: Task = {
    id: 'routing-check',
    type: body.type,
    description: 'Routing check',
    context: {},
    priority: 'medium',
    status: 'pending',
    createdAt: new Date().toISOString(),
  };

  const decision = router.routeWithFallback(task);

  if (!decision) {
    return {
      status: 404,
      body: { error: `No agent available for task type: ${body.type}` },
    };
  }

  const agent = agentRegistry.get(decision.agentId);

  return {
    status: 200,
    body: {
      decision: {
        agentId: decision.agentId,
        agentName: agent?.name,
        confidence: decision.confidence,
        reasoning: decision.reasoning,
        estimatedCost: decision.estimatedCost,
        estimatedTime: decision.estimatedTime,
      },
      alternatives: decision.alternatives?.map((alt) => ({
        agentId: alt.agentId,
        confidence: alt.confidence,
        reason: alt.reason,
      })),
    },
  };
});

// ============================================================================
// Router Export
// ============================================================================

export async function router(req: APIRequest): Promise<APIResponse> {
  // Find matching route
  for (const route of routes) {
    if (route.method === req.method && route.pattern.test(req.path)) {
      return route.handler(req);
    }
  }

  // 404 for unmatched routes
  return {
    status: 404,
    body: {
      error: 'Not found',
      path: req.path,
      method: req.method,
    },
  };
}
