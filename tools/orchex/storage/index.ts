/**
 * ORCHEX Storage Module
 * Pluggable storage system with support for multiple backends
 */

import {
  StorageBackend,
  StorageConfig,
  DEFAULT_STORAGE_CONFIG,
  COLLECTIONS,
  QueryResult,
} from './types.js';
import { JsonStorageBackend } from './json-backend.js';
import { SqliteStorageBackend } from './sqlite-backend.js';
import { PostgresStorageBackend } from './postgres-backend.js';

export * from './types.js';
export { JsonStorageBackend } from './json-backend.js';
export { SqliteStorageBackend } from './sqlite-backend.js';
export { PostgresStorageBackend } from './postgres-backend.js';

// ============================================================================
// Storage Factory
// ============================================================================

/**
 * Create a storage backend based on configuration
 */
export function createStorage(config: Partial<StorageConfig> = {}): StorageBackend {
  const fullConfig = { ...DEFAULT_STORAGE_CONFIG, ...config };

  switch (fullConfig.backend) {
    case 'json':
      return new JsonStorageBackend(fullConfig.basePath);

    case 'sqlite': {
      const dbPath = fullConfig.connectionString || '.ORCHEX/ORCHEX.db';
      return new SqliteStorageBackend(dbPath);
    }

    case 'postgres': {
      const connString = fullConfig.connectionString || 'postgresql://localhost:5432/ORCHEX';
      return new PostgresStorageBackend(connString);
    }

    default:
      throw new Error(`Unknown storage backend: ${fullConfig.backend}`);
  }
}

// ============================================================================
// Storage Manager Singleton
// ============================================================================

let storageInstance: StorageBackend | null = null;
let initPromise: Promise<void> | null = null;

/**
 * Get the global storage instance
 * Initializes with JSON backend if not already configured
 */
export async function getStorage(): Promise<StorageBackend> {
  if (!storageInstance) {
    storageInstance = createStorage();
  }

  if (!storageInstance.isReady()) {
    if (!initPromise) {
      initPromise = storageInstance.initialize();
    }
    await initPromise;
    initPromise = null;
  }

  return storageInstance;
}

/**
 * Configure and initialize storage with custom settings
 */
export async function initializeStorage(
  config: Partial<StorageConfig> = {}
): Promise<StorageBackend> {
  // Close existing instance if any
  if (storageInstance) {
    await storageInstance.close();
  }

  storageInstance = createStorage(config);
  await storageInstance.initialize();
  return storageInstance;
}

/**
 * Close the global storage instance
 */
export async function closeStorage(): Promise<void> {
  if (storageInstance) {
    await storageInstance.close();
    storageInstance = null;
    initPromise = null;
  }
}

// ============================================================================
// Convenience Types
// ============================================================================

interface AgentStorageAccessor {
  get: <T>(id: string) => Promise<T | null>;
  set: <T>(id: string, agent: T) => Promise<void>;
  delete: (id: string) => Promise<boolean>;
  getAll: <T>() => Promise<QueryResult<T>>;
  count: () => Promise<number>;
}

interface CircuitStorageAccessor {
  get: <T>(agentId: string) => Promise<T | null>;
  set: <T>(agentId: string, state: T) => Promise<void>;
  delete: (agentId: string) => Promise<boolean>;
  getAll: <T>() => Promise<QueryResult<T>>;
  clear: () => Promise<void>;
}

interface MetricsStorageAccessor {
  get: <T>(key: string) => Promise<T | null>;
  set: <T>(key: string, metrics: T) => Promise<void>;
  getAll: <T>() => Promise<QueryResult<T>>;
  query: <T>(predicate: (item: T) => boolean) => Promise<QueryResult<T>>;
}

interface TaskStorageAccessor {
  get: <T>(taskId: string) => Promise<T | null>;
  set: <T>(taskId: string, task: T) => Promise<void>;
  delete: (taskId: string) => Promise<boolean>;
  getAll: <T>() => Promise<QueryResult<T>>;
  query: <T>(predicate: (item: T) => boolean) => Promise<QueryResult<T>>;
  count: () => Promise<number>;
}

interface CacheStorageAccessor {
  get: <T>(key: string) => Promise<T | null>;
  set: <T>(key: string, entry: T) => Promise<void>;
  delete: (key: string) => Promise<boolean>;
  exists: (key: string) => Promise<boolean>;
  clear: () => Promise<void>;
  count: () => Promise<number>;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Typed storage access for agents
 */
export async function agentStorage(): Promise<AgentStorageAccessor> {
  const storage = await getStorage();
  return {
    get: <T>(id: string): Promise<T | null> => storage.get<T>(COLLECTIONS.AGENTS, id),
    set: <T>(id: string, agent: T): Promise<void> => storage.set(COLLECTIONS.AGENTS, id, agent),
    delete: (id: string): Promise<boolean> => storage.delete(COLLECTIONS.AGENTS, id),
    getAll: <T>(): Promise<QueryResult<T>> => storage.getAll<T>(COLLECTIONS.AGENTS),
    count: (): Promise<number> => storage.count(COLLECTIONS.AGENTS),
  };
}

/**
 * Typed storage access for circuit breaker states
 */
export async function circuitStorage(): Promise<CircuitStorageAccessor> {
  const storage = await getStorage();
  return {
    get: <T>(agentId: string): Promise<T | null> => storage.get<T>(COLLECTIONS.CIRCUITS, agentId),
    set: <T>(agentId: string, state: T): Promise<void> =>
      storage.set(COLLECTIONS.CIRCUITS, agentId, state),
    delete: (agentId: string): Promise<boolean> => storage.delete(COLLECTIONS.CIRCUITS, agentId),
    getAll: <T>(): Promise<QueryResult<T>> => storage.getAll<T>(COLLECTIONS.CIRCUITS),
    clear: (): Promise<void> => storage.clear(COLLECTIONS.CIRCUITS),
  };
}

/**
 * Typed storage access for metrics
 */
export async function metricsStorage(): Promise<MetricsStorageAccessor> {
  const storage = await getStorage();
  return {
    get: <T>(key: string): Promise<T | null> => storage.get<T>(COLLECTIONS.METRICS, key),
    set: <T>(key: string, metrics: T): Promise<void> =>
      storage.set(COLLECTIONS.METRICS, key, metrics),
    getAll: <T>(): Promise<QueryResult<T>> => storage.getAll<T>(COLLECTIONS.METRICS),
    query: <T>(predicate: (item: T) => boolean): Promise<QueryResult<T>> =>
      storage.query(COLLECTIONS.METRICS, predicate),
  };
}

/**
 * Typed storage access for tasks
 */
export async function taskStorage(): Promise<TaskStorageAccessor> {
  const storage = await getStorage();
  return {
    get: <T>(taskId: string): Promise<T | null> => storage.get<T>(COLLECTIONS.TASKS, taskId),
    set: <T>(taskId: string, task: T): Promise<void> =>
      storage.set(COLLECTIONS.TASKS, taskId, task),
    delete: (taskId: string): Promise<boolean> => storage.delete(COLLECTIONS.TASKS, taskId),
    getAll: <T>(): Promise<QueryResult<T>> => storage.getAll<T>(COLLECTIONS.TASKS),
    query: <T>(predicate: (item: T) => boolean): Promise<QueryResult<T>> =>
      storage.query(COLLECTIONS.TASKS, predicate),
    count: (): Promise<number> => storage.count(COLLECTIONS.TASKS),
  };
}

/**
 * Typed storage access for cache entries
 */
export async function cacheStorage(): Promise<CacheStorageAccessor> {
  const storage = await getStorage();
  return {
    get: <T>(key: string): Promise<T | null> => storage.get<T>(COLLECTIONS.CACHE, key),
    set: <T>(key: string, entry: T): Promise<void> => storage.set(COLLECTIONS.CACHE, key, entry),
    delete: (key: string): Promise<boolean> => storage.delete(COLLECTIONS.CACHE, key),
    exists: (key: string): Promise<boolean> => storage.exists(COLLECTIONS.CACHE, key),
    clear: (): Promise<void> => storage.clear(COLLECTIONS.CACHE),
    count: (): Promise<number> => storage.count(COLLECTIONS.CACHE),
  };
}
