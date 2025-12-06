/**
 * ORCHEX SQLite Storage Backend
 * High-performance SQLite-based storage implementation
 */

import * as fs from 'fs';
import * as path from 'path';
import Database from 'better-sqlite3';
import type { Database as DatabaseType } from 'better-sqlite3';
import { StorageBackend, QueryOptions, QueryResult } from './types.js';

// ============================================================================
// SQLite Backend Implementation
// ============================================================================

export class SqliteStorageBackend implements StorageBackend {
  readonly name = 'sqlite';
  readonly type = 'sqlite' as const;

  private dbPath: string;
  private db: DatabaseType | null = null;
  private ready = false;

  constructor(dbPath: string = '.ORCHEX/ORCHEX.db') {
    this.dbPath = path.isAbsolute(dbPath) ? dbPath : path.join(process.cwd(), dbPath);
  }

  // ============================================================================
  // Lifecycle
  // ============================================================================

  async initialize(): Promise<void> {
    // Ensure directory exists
    const dir = path.dirname(this.dbPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Open database
    this.db = new Database(this.dbPath);

    // Enable WAL mode for better concurrent performance
    this.db.pragma('journal_mode = WAL');

    // Create collections table if not exists
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS collections (
        collection TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (collection, key)
      );

      CREATE INDEX IF NOT EXISTS idx_collection ON collections(collection);
      CREATE INDEX IF NOT EXISTS idx_updated_at ON collections(updated_at);
    `);

    this.ready = true;
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    this.ready = false;
  }

  isReady(): boolean {
    return this.ready && this.db !== null;
  }

  // ============================================================================
  // Key-Value Operations
  // ============================================================================

  async get<T>(collection: string, key: string): Promise<T | null> {
    this.ensureReady();

    const stmt = this.db!.prepare('SELECT value FROM collections WHERE collection = ? AND key = ?');
    const row = stmt.get(collection, key) as { value: string } | undefined;

    if (!row) return null;

    try {
      return JSON.parse(row.value) as T;
    } catch {
      return null;
    }
  }

  async set<T>(collection: string, key: string, value: T): Promise<void> {
    this.ensureReady();

    const stmt = this.db!.prepare(`
      INSERT INTO collections (collection, key, value, created_at, updated_at)
      VALUES (?, ?, ?, datetime('now'), datetime('now'))
      ON CONFLICT(collection, key) DO UPDATE SET
        value = excluded.value,
        updated_at = datetime('now')
    `);

    stmt.run(collection, key, JSON.stringify(value));
  }

  async delete(collection: string, key: string): Promise<boolean> {
    this.ensureReady();

    const stmt = this.db!.prepare('DELETE FROM collections WHERE collection = ? AND key = ?');
    const result = stmt.run(collection, key);

    return result.changes > 0;
  }

  async exists(collection: string, key: string): Promise<boolean> {
    this.ensureReady();

    const stmt = this.db!.prepare(
      'SELECT 1 FROM collections WHERE collection = ? AND key = ? LIMIT 1'
    );
    const row = stmt.get(collection, key);

    return row !== undefined;
  }

  // ============================================================================
  // Collection Operations
  // ============================================================================

  async getAll<T>(collection: string, options?: QueryOptions): Promise<QueryResult<T>> {
    this.ensureReady();

    // Get total count
    const countStmt = this.db!.prepare(
      'SELECT COUNT(*) as count FROM collections WHERE collection = ?'
    );
    const countRow = countStmt.get(collection) as { count: number };
    const total = countRow.count;

    // Build query with ordering and pagination
    let query = 'SELECT key, value FROM collections WHERE collection = ?';
    const params: (string | number)[] = [collection];

    if (options?.orderBy) {
      // For JSON ordering, we extract the field from the value
      const dir = options.orderDir === 'desc' ? 'DESC' : 'ASC';
      query += ` ORDER BY json_extract(value, '$.${options.orderBy}') ${dir}`;
    } else {
      query += ' ORDER BY updated_at DESC';
    }

    if (options?.limit) {
      query += ' LIMIT ?';
      params.push(options.limit);
    }

    if (options?.offset) {
      query += ' OFFSET ?';
      params.push(options.offset);
    }

    const stmt = this.db!.prepare(query);
    const rows = stmt.all(...params) as Array<{ key: string; value: string }>;

    const items = rows
      .map((row) => {
        try {
          return JSON.parse(row.value) as T;
        } catch {
          return null;
        }
      })
      .filter((item): item is T => item !== null);

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
    // SQLite doesn't support arbitrary JS predicates, so we fetch all and filter
    // For better performance, consider using SQL WHERE clauses for common queries
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

    const stmt = this.db!.prepare('SELECT COUNT(*) as count FROM collections WHERE collection = ?');
    const row = stmt.get(collection) as { count: number };

    return row.count;
  }

  async clear(collection: string): Promise<void> {
    this.ensureReady();

    const stmt = this.db!.prepare('DELETE FROM collections WHERE collection = ?');
    stmt.run(collection);
  }

  // ============================================================================
  // Bulk Operations
  // ============================================================================

  async setMany<T>(collection: string, items: Array<{ key: string; value: T }>): Promise<void> {
    this.ensureReady();

    const stmt = this.db!.prepare(`
      INSERT INTO collections (collection, key, value, created_at, updated_at)
      VALUES (?, ?, ?, datetime('now'), datetime('now'))
      ON CONFLICT(collection, key) DO UPDATE SET
        value = excluded.value,
        updated_at = datetime('now')
    `);

    const insertMany = this.db!.transaction((items: Array<{ key: string; value: T }>) => {
      for (const { key, value } of items) {
        stmt.run(collection, key, JSON.stringify(value));
      }
    });

    insertMany(items);
  }

  async deleteMany(collection: string, keys: string[]): Promise<number> {
    this.ensureReady();

    if (keys.length === 0) return 0;

    const placeholders = keys.map(() => '?').join(',');
    const stmt = this.db!.prepare(
      `DELETE FROM collections WHERE collection = ? AND key IN (${placeholders})`
    );
    const result = stmt.run(collection, ...keys);

    return result.changes;
  }

  // ============================================================================
  // Transaction Support
  // ============================================================================

  async beginTransaction(): Promise<void> {
    this.ensureReady();
    this.db!.exec('BEGIN TRANSACTION');
  }

  async commit(): Promise<void> {
    this.ensureReady();
    this.db!.exec('COMMIT');
  }

  async rollback(): Promise<void> {
    this.ensureReady();
    this.db!.exec('ROLLBACK');
  }

  // ============================================================================
  // SQLite-Specific Methods
  // ============================================================================

  /**
   * Execute raw SQL query (for advanced use cases)
   */
  rawQuery<T>(sql: string, params: unknown[] = []): T[] {
    this.ensureReady();
    const stmt = this.db!.prepare(sql);
    return stmt.all(...params) as T[];
  }

  /**
   * Execute raw SQL statement (for advanced use cases)
   */
  rawExecute(sql: string, params: unknown[] = []): void {
    this.ensureReady();
    const stmt = this.db!.prepare(sql);
    stmt.run(...params);
  }

  /**
   * Get database file path
   */
  getDbPath(): string {
    return this.dbPath;
  }

  /**
   * Get database statistics
   */
  getStats(): { size: number; collections: number; totalRecords: number } {
    this.ensureReady();

    const stats = fs.statSync(this.dbPath);
    const collectionsStmt = this.db!.prepare(
      'SELECT COUNT(DISTINCT collection) as count FROM collections'
    );
    const recordsStmt = this.db!.prepare('SELECT COUNT(*) as count FROM collections');

    const collections = (collectionsStmt.get() as { count: number }).count;
    const totalRecords = (recordsStmt.get() as { count: number }).count;

    return {
      size: stats.size,
      collections,
      totalRecords,
    };
  }

  /**
   * Vacuum database to reclaim space
   */
  vacuum(): void {
    this.ensureReady();
    this.db!.exec('VACUUM');
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private ensureReady(): void {
    if (!this.ready || !this.db) {
      throw new Error('SQLite storage not initialized. Call initialize() first.');
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
