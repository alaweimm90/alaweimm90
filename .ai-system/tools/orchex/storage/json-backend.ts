/**
 * ORCHEX JSON Storage Backend
 * File-based JSON storage implementation
 */

import * as fs from 'fs';
import * as path from 'path';
import { StorageBackend, QueryOptions, QueryResult } from './types.js';

// ============================================================================
// JSON Backend Implementation
// ============================================================================

export class JsonStorageBackend implements StorageBackend {
  readonly name = 'json-file';
  readonly type = 'json' as const;

  private basePath: string;
  private ready = false;
  private cache: Map<string, Map<string, unknown>> = new Map();
  private dirty: Set<string> = new Set();
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(basePath: string = '.ORCHEX/data') {
    this.basePath = path.isAbsolute(basePath) ? basePath : path.join(process.cwd(), basePath);
  }

  // ============================================================================
  // Lifecycle
  // ============================================================================

  async initialize(): Promise<void> {
    if (!fs.existsSync(this.basePath)) {
      fs.mkdirSync(this.basePath, { recursive: true });
    }

    // Load all existing collections into cache
    const files = fs.readdirSync(this.basePath).filter((f) => f.endsWith('.json'));
    for (const file of files) {
      const collection = file.replace('.json', '');
      await this.loadCollection(collection);
    }

    this.ready = true;
  }

  async close(): Promise<void> {
    // Flush any pending writes
    await this.flush();

    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    this.cache.clear();
    this.dirty.clear();
    this.ready = false;
  }

  isReady(): boolean {
    return this.ready;
  }

  // ============================================================================
  // Key-Value Operations
  // ============================================================================

  async get<T>(collection: string, key: string): Promise<T | null> {
    const col = await this.getCollection(collection);
    return (col.get(key) as T) ?? null;
  }

  async set<T>(collection: string, key: string, value: T): Promise<void> {
    const col = await this.getCollection(collection);
    col.set(key, value);
    this.markDirty(collection);
  }

  async delete(collection: string, key: string): Promise<boolean> {
    const col = await this.getCollection(collection);
    const existed = col.has(key);
    col.delete(key);
    if (existed) {
      this.markDirty(collection);
    }
    return existed;
  }

  async exists(collection: string, key: string): Promise<boolean> {
    const col = await this.getCollection(collection);
    return col.has(key);
  }

  // ============================================================================
  // Collection Operations
  // ============================================================================

  async getAll<T>(collection: string, options?: QueryOptions): Promise<QueryResult<T>> {
    const col = await this.getCollection(collection);
    let items = Array.from(col.values()) as T[];
    const total = items.length;

    // Apply ordering if specified
    if (options?.orderBy) {
      items = this.sortItems(items, options.orderBy, options.orderDir || 'asc');
    }

    // Apply pagination
    const offset = options?.offset || 0;
    const limit = options?.limit || items.length;
    items = items.slice(offset, offset + limit);

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
    const col = await this.getCollection(collection);
    let items = Array.from(col.values()).filter((item) => predicate(item as T)) as T[];
    const total = items.length;

    // Apply ordering
    if (options?.orderBy) {
      items = this.sortItems(items, options.orderBy, options.orderDir || 'asc');
    }

    // Apply pagination
    const offset = options?.offset || 0;
    const limit = options?.limit || items.length;
    items = items.slice(offset, offset + limit);

    return {
      items,
      total,
      hasMore: offset + items.length < total,
    };
  }

  async count(collection: string): Promise<number> {
    const col = await this.getCollection(collection);
    return col.size;
  }

  async clear(collection: string): Promise<void> {
    this.cache.set(collection, new Map());
    this.markDirty(collection);
  }

  // ============================================================================
  // Bulk Operations
  // ============================================================================

  async setMany<T>(collection: string, items: Array<{ key: string; value: T }>): Promise<void> {
    const col = await this.getCollection(collection);
    for (const { key, value } of items) {
      col.set(key, value);
    }
    this.markDirty(collection);
  }

  async deleteMany(collection: string, keys: string[]): Promise<number> {
    const col = await this.getCollection(collection);
    let deleted = 0;
    for (const key of keys) {
      if (col.delete(key)) {
        deleted++;
      }
    }
    if (deleted > 0) {
      this.markDirty(collection);
    }
    return deleted;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private async getCollection(name: string): Promise<Map<string, unknown>> {
    if (!this.cache.has(name)) {
      await this.loadCollection(name);
    }
    return this.cache.get(name) || new Map();
  }

  private async loadCollection(name: string): Promise<void> {
    const filePath = this.getFilePath(name);
    const collection = new Map<string, unknown>();

    if (fs.existsSync(filePath)) {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const data = JSON.parse(content) as Record<string, unknown>;
        for (const [key, value] of Object.entries(data)) {
          collection.set(key, value);
        }
      } catch {
        // If file is corrupted, start with empty collection
        console.warn(`Warning: Could not parse ${filePath}, starting with empty collection`);
      }
    }

    this.cache.set(name, collection);
  }

  private markDirty(collection: string): void {
    this.dirty.add(collection);
    this.scheduleFlush();
  }

  private scheduleFlush(): void {
    if (this.flushTimer) return;

    // Debounce writes - flush after 100ms of no changes
    this.flushTimer = setTimeout(() => {
      this.flush().catch(console.error);
      this.flushTimer = null;
    }, 100);
  }

  private async flush(): Promise<void> {
    for (const collection of this.dirty) {
      const col = this.cache.get(collection);
      if (col) {
        const data: Record<string, unknown> = {};
        for (const [key, value] of col) {
          data[key] = value;
        }
        const filePath = this.getFilePath(collection);
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
      }
    }
    this.dirty.clear();
  }

  private getFilePath(collection: string): string {
    return path.join(this.basePath, `${collection}.json`);
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
