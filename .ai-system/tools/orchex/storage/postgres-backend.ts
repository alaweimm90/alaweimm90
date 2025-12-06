/**
 * ORCHEX PostgreSQL Storage Backend
 * Production-grade PostgreSQL-based storage implementation
 */

import { StorageBackend, QueryOptions, QueryResult } from './types.js';

// ============================================================================
// PostgreSQL Backend Implementation
// ============================================================================

/**
 * PostgreSQL connection pool interface
 * Uses pg library when available
 */
interface PgPool {
  query(text: string, params?: unknown[]): Promise<{ rows: unknown[]; rowCount: number }>;
  connect(): Promise<PgClient>;
  end(): Promise<void>;
}

interface PgClient {
  query(text: string, params?: unknown[]): Promise<{ rows: unknown[]; rowCount: number }>;
  release(): void;
}

export class PostgresStorageBackend implements StorageBackend {
  readonly name = 'postgres';
  readonly type = 'postgres' as const;

  private connectionString: string;
  private pool: PgPool | null = null;
  private ready = false;
  private client: PgClient | null = null;
  private inTransaction = false;

  constructor(connectionString: string = 'postgresql://localhost:5432/ORCHEX') {
    this.connectionString = connectionString;
  }

  // ============================================================================
  // Lifecycle
  // ============================================================================

  async initialize(): Promise<void> {
    try {
      // Dynamic import of pg library
      const pg = await import('pg');
      const { Pool } = pg.default || pg;

      this.pool = new Pool({
        connectionString: this.connectionString,
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });

      // Test connection
      const client = await this.pool.connect();
      client.release();

      // Create collections table if not exists
      await this.pool.query(`
        CREATE TABLE IF NOT EXISTS collections (
          collection TEXT NOT NULL,
          key TEXT NOT NULL,
          value JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          PRIMARY KEY (collection, key)
        );

        CREATE INDEX IF NOT EXISTS idx_collections_collection ON collections(collection);
        CREATE INDEX IF NOT EXISTS idx_collections_updated_at ON collections(updated_at);
        CREATE INDEX IF NOT EXISTS idx_collections_value ON collections USING GIN(value);
      `);

      this.ready = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to initialize PostgreSQL storage: ${message}`);
    }
  }

  async close(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
    }
    this.ready = false;
  }

  isReady(): boolean {
    return this.ready && this.pool !== null;
  }

  // ============================================================================
  // Key-Value Operations
  // ============================================================================

  async get<T>(collection: string, key: string): Promise<T | null> {
    this.ensureReady();

    const result = await this.pool!.query(
      'SELECT value FROM collections WHERE collection = $1 AND key = $2',
      [collection, key]
    );

    if (result.rows.length === 0) return null;

    const row = result.rows[0] as { value: T };
    return row.value;
  }

  async set<T>(collection: string, key: string, value: T): Promise<void> {
    this.ensureReady();

    await this.pool!.query(
      `
      INSERT INTO collections (collection, key, value, created_at, updated_at)
      VALUES ($1, $2, $3, NOW(), NOW())
      ON CONFLICT (collection, key) DO UPDATE SET
        value = EXCLUDED.value,
        updated_at = NOW()
      `,
      [collection, key, JSON.stringify(value)]
    );
  }

  async delete(collection: string, key: string): Promise<boolean> {
    this.ensureReady();

    const result = await this.pool!.query(
      'DELETE FROM collections WHERE collection = $1 AND key = $2',
      [collection, key]
    );

    return (result.rowCount ?? 0) > 0;
  }

  async exists(collection: string, key: string): Promise<boolean> {
    this.ensureReady();

    const result = await this.pool!.query(
      'SELECT 1 FROM collections WHERE collection = $1 AND key = $2 LIMIT 1',
      [collection, key]
    );

    return result.rows.length > 0;
  }

  // ============================================================================
  // Collection Operations
  // ============================================================================

  async getAll<T>(collection: string, options?: QueryOptions): Promise<QueryResult<T>> {
    this.ensureReady();

    // Get total count
    const countResult = await this.pool!.query(
      'SELECT COUNT(*) as count FROM collections WHERE collection = $1',
      [collection]
    );
    const total = parseInt((countResult.rows[0] as { count: string }).count, 10);

    // Build query with ordering and pagination
    let query = 'SELECT key, value FROM collections WHERE collection = $1';
    const params: (string | number)[] = [collection];
    let paramIndex = 2;

    if (options?.orderBy) {
      const dir = options.orderDir === 'desc' ? 'DESC' : 'ASC';
      query += ` ORDER BY value->>'${options.orderBy}' ${dir}`;
    } else {
      query += ' ORDER BY updated_at DESC';
    }

    if (options?.limit) {
      query += ` LIMIT $${paramIndex}`;
      params.push(options.limit);
      paramIndex++;
    }

    if (options?.offset) {
      query += ` OFFSET $${paramIndex}`;
      params.push(options.offset);
    }

    const result = await this.pool!.query(query, params);
    const items = result.rows.map((row) => (row as { value: T }).value);

    const offset = options?.offset || 0;

    return {
      items,
      total,
      hasMore: offset + items.length < total,
    };
  }

  async query<T>(
    collection: string,
    predicate: (item: T) => boolean,
    options?: QueryOptions
  ): Promise<QueryResult<T>> {
    // PostgreSQL doesn't support arbitrary JS predicates
    // Fetch all and filter in memory
    const all = await this.getAll<T>(collection);
    let filtered = all.items.filter(predicate);
    const total = filtered.length;

    // Apply ordering
    if (options?.orderBy) {
      filtered = this.sortItems(filtered, options.orderBy, options.orderDir || 'asc');
    }

    // Apply pagination
    const offset = options?.offset || 0;
    const limit = options?.limit || filtered.length;
    filtered = filtered.slice(offset, offset + limit);

    return {
      items: filtered,
      total,
      hasMore: offset + filtered.length < total,
    };
  }

  async count(collection: string): Promise<number> {
    this.ensureReady();

    const result = await this.pool!.query(
      'SELECT COUNT(*) as count FROM collections WHERE collection = $1',
      [collection]
    );

    return parseInt((result.rows[0] as { count: string }).count, 10);
  }

  async clear(collection: string): Promise<void> {
    this.ensureReady();

    await this.pool!.query('DELETE FROM collections WHERE collection = $1', [collection]);
  }

  // ============================================================================
  // Bulk Operations
  // ============================================================================

  async setMany<T>(collection: string, items: Array<{ key: string; value: T }>): Promise<void> {
    this.ensureReady();

    if (items.length === 0) return;

    // Use a transaction for bulk inserts
    const client = await this.pool!.connect();
    try {
      await client.query('BEGIN');

      for (const { key, value } of items) {
        await client.query(
          `
          INSERT INTO collections (collection, key, value, created_at, updated_at)
          VALUES ($1, $2, $3, NOW(), NOW())
          ON CONFLICT (collection, key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = NOW()
          `,
          [collection, key, JSON.stringify(value)]
        );
      }

      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async deleteMany(collection: string, keys: string[]): Promise<number> {
    this.ensureReady();

    if (keys.length === 0) return 0;

    const placeholders = keys.map((_, i) => `$${i + 2}`).join(',');
    const result = await this.pool!.query(
      `DELETE FROM collections WHERE collection = $1 AND key IN (${placeholders})`,
      [collection, ...keys]
    );

    return result.rowCount ?? 0;
  }

  // ============================================================================
  // Transaction Support
  // ============================================================================

  async beginTransaction(): Promise<void> {
    this.ensureReady();

    if (this.inTransaction) {
      throw new Error('Transaction already in progress');
    }

    this.client = await this.pool!.connect();
    await this.client.query('BEGIN');
    this.inTransaction = true;
  }

  async commit(): Promise<void> {
    if (!this.inTransaction || !this.client) {
      throw new Error('No transaction in progress');
    }

    await this.client.query('COMMIT');
    this.client.release();
    this.client = null;
    this.inTransaction = false;
  }

  async rollback(): Promise<void> {
    if (!this.inTransaction || !this.client) {
      throw new Error('No transaction in progress');
    }

    await this.client.query('ROLLBACK');
    this.client.release();
    this.client = null;
    this.inTransaction = false;
  }

  // ============================================================================
  // PostgreSQL-Specific Methods
  // ============================================================================

  /**
   * Execute raw SQL query (for advanced use cases)
   */
  async rawQuery<T>(sql: string, params: unknown[] = []): Promise<T[]> {
    this.ensureReady();
    const result = await this.pool!.query(sql, params);
    return result.rows as T[];
  }

  /**
   * Execute raw SQL statement (for advanced use cases)
   */
  async rawExecute(sql: string, params: unknown[] = []): Promise<number> {
    this.ensureReady();
    const result = await this.pool!.query(sql, params);
    return result.rowCount ?? 0;
  }

  /**
   * Get connection string (masked)
   */
  getConnectionString(): string {
    // Mask password in connection string
    return this.connectionString.replace(/:[^:@]+@/, ':****@');
  }

  /**
   * Get database statistics
   */
  async getStats(): Promise<{ collections: number; totalRecords: number; dbSize: string }> {
    this.ensureReady();

    const collectionsResult = await this.pool!.query(
      'SELECT COUNT(DISTINCT collection) as count FROM collections'
    );
    const recordsResult = await this.pool!.query('SELECT COUNT(*) as count FROM collections');
    const sizeResult = await this.pool!.query(
      'SELECT pg_size_pretty(pg_database_size(current_database())) as size'
    );

    return {
      collections: parseInt((collectionsResult.rows[0] as { count: string }).count, 10),
      totalRecords: parseInt((recordsResult.rows[0] as { count: string }).count, 10),
      dbSize: (sizeResult.rows[0] as { size: string }).size,
    };
  }

  /**
   * Vacuum and analyze tables for performance
   */
  async vacuum(): Promise<void> {
    this.ensureReady();
    await this.pool!.query('VACUUM ANALYZE collections');
  }

  /**
   * Query using JSONB operators for efficient filtering
   */
  async queryJsonb<T>(
    collection: string,
    jsonPath: string,
    value: unknown,
    options?: QueryOptions
  ): Promise<QueryResult<T>> {
    this.ensureReady();

    // Count matching records
    const countResult = await this.pool!.query(
      `SELECT COUNT(*) as count FROM collections
       WHERE collection = $1 AND value @> $2::jsonb`,
      [collection, JSON.stringify({ [jsonPath]: value })]
    );
    const total = parseInt((countResult.rows[0] as { count: string }).count, 10);

    // Build query
    let query = `SELECT key, value FROM collections
                 WHERE collection = $1 AND value @> $2::jsonb`;
    const params: (string | number | object)[] = [
      collection,
      JSON.stringify({ [jsonPath]: value }),
    ];
    let paramIndex = 3;

    if (options?.orderBy) {
      const dir = options.orderDir === 'desc' ? 'DESC' : 'ASC';
      query += ` ORDER BY value->>'${options.orderBy}' ${dir}`;
    }

    if (options?.limit) {
      query += ` LIMIT $${paramIndex}`;
      params.push(options.limit);
      paramIndex++;
    }

    if (options?.offset) {
      query += ` OFFSET $${paramIndex}`;
      params.push(options.offset);
    }

    const result = await this.pool!.query(query, params);
    const items = result.rows.map((row) => (row as { value: T }).value);

    const offset = options?.offset || 0;

    return {
      items,
      total,
      hasMore: offset + items.length < total,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private ensureReady(): void {
    if (!this.ready || !this.pool) {
      throw new Error('PostgreSQL storage not initialized. Call initialize() first.');
    }
  }

  private sortItems<T>(items: T[], orderBy: string, orderDir: 'asc' | 'desc'): T[] {
    return [...items].sort((a, b) => {
      const aVal = (a as Record<string, unknown>)[orderBy];
      const bVal = (b as Record<string, unknown>)[orderBy];

      let comparison = 0;
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        comparison = aVal.localeCompare(bVal);
      } else if (typeof aVal === 'number' && typeof bVal === 'number') {
        comparison = aVal - bVal;
      } else {
        comparison = String(aVal).localeCompare(String(bVal));
      }

      return orderDir === 'desc' ? -comparison : comparison;
    });
  }
}
