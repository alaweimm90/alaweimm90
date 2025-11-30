/**
 * ATLAS Multi-Agent Orchestration Platform
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import { createRouter, agentRegistry, executeTask } from '@atlas/cli';
 *
 * // Execute a task with automatic agent selection
 * const result = await executeTask({
 *   id: 'task-1',
 *   type: 'code_generation',
 *   description: 'Generate a hello world function',
 *   context: { language: 'typescript' },
 *   priority: 'medium',
 *   status: 'pending',
 *   createdAt: new Date().toISOString(),
 * });
 * ```
 */

// Types
export * from './types/index.js';

// Agent Registry
export { agentRegistry, AgentRegistry } from './agents/registry.js';
export type { Agent } from './agents/registry.js';

// Orchestration
export { createRouter, TaskRouter } from './orchestration/router.js';
export { fallbackManager, FallbackManager, CircuitBreaker } from './orchestration/fallback.js';

// Adapters
export {
  executeTask,
  generateCode,
  reviewCode,
  explainCode,
  chat,
  getAdapter,
  getAdapterStatuses,
} from './adapters/index.js';
export { BaseAdapter } from './adapters/base.js';
export { AnthropicAdapter } from './adapters/anthropic.js';
export { OpenAIAdapter } from './adapters/openai.js';
export { GoogleAdapter } from './adapters/google.js';

// Storage
export {
  createStorage,
  getStorage,
  initializeStorage,
  closeStorage,
  agentStorage,
  circuitStorage,
  metricsStorage,
  taskStorage,
  cacheStorage,
  JsonStorageBackend,
  SqliteStorageBackend,
} from './storage/index.js';
export type { StorageBackend, StorageConfig, QueryResult, QueryOptions } from './storage/index.js';

// API
export { createServer, startServer } from './api/server.js';
export { router } from './api/router.js';
export type { APIRequest, APIResponse, ServerConfig } from './api/server.js';
