/**
 * ORCHEX Storage Types
 * Abstract storage interface for pluggable backends
 */

// ============================================================================
// Core Storage Types
// ============================================================================

export interface StorageRecord {
  id: string;
  createdAt: string;
  updatedAt: string;
}

export interface QueryOptions {
  limit?: number;
  offset?: number;
  orderBy?: string;
  orderDir?: 'asc' | 'desc';
}

export interface QueryResult<T> {
  items: T[];
  total: number;
  hasMore: boolean;
}

// ============================================================================
// Storage Backend Interface
// ============================================================================

export interface StorageBackend {
  readonly name: string;
  readonly type: 'json' | 'sqlite' | 'postgres';

  // Lifecycle
  initialize(): Promise<void>;
  close(): Promise<void>;
  isReady(): boolean;

  // Key-Value Operations
  get<T>(collection: string, key: string): Promise<T | null>;
  set<T>(collection: string, key: string, value: T): Promise<void>;
  delete(collection: string, key: string): Promise<boolean>;
  exists(collection: string, key: string): Promise<boolean>;

  // Collection Operations
  getAll<T>(collection: string, options?: QueryOptions): Promise<QueryResult<T>>;
  query<T>(
    collection: string,
    predicate: (item: T) => boolean,
    options?: QueryOptions
  ): Promise<QueryResult<T>>;
  count(collection: string): Promise<number>;
  clear(collection: string): Promise<void>;

  // Bulk Operations
  setMany<T>(collection: string, items: Array<{ key: string; value: T }>): Promise<void>;
  deleteMany(collection: string, keys: string[]): Promise<number>;

  // Transaction Support (optional)
  beginTransaction?(): Promise<void>;
  commit?(): Promise<void>;
  rollback?(): Promise<void>;
}

// ============================================================================
// Storage Configuration
// ============================================================================

export interface StorageConfig {
  backend: 'json' | 'sqlite' | 'postgres';
  basePath?: string; // For JSON backend
  connectionString?: string; // For database backends
  options?: Record<string, unknown>;
}

export const DEFAULT_STORAGE_CONFIG: StorageConfig = {
  backend: 'json',
  basePath: '.ORCHEX/data',
};

// ============================================================================
// Collection Names
// ============================================================================

export const COLLECTIONS = {
  AGENTS: 'agents',
  CIRCUITS: 'circuits',
  METRICS: 'metrics',
  TASKS: 'tasks',
  SESSIONS: 'sessions',
  CACHE: 'cache',
} as const;

export type CollectionName = (typeof COLLECTIONS)[keyof typeof COLLECTIONS];
