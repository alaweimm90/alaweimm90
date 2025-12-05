/**
 * Database Layer for AI Tools
 *
 * Provides SQLite database operations to replace file-based persistence.
 * Supports migration from JSON files and maintains backward compatibility.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as crypto from 'crypto';
import { loadJson, ensureDir } from './file-persistence.js';

// Database file path
const AI_DIR = path.join(process.cwd(), '.ai');
const DB_PATH = path.join(AI_DIR, 'ai-tools.db');
ensureDir(AI_DIR);

// ============================================================================
// Database Schema
// ============================================================================

const SCHEMA = {
  // Cache entries for multi-layer caching system
  cache_entries: `
    CREATE TABLE IF NOT EXISTS cache_entries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      key_hash TEXT NOT NULL UNIQUE,
      key TEXT NOT NULL,
      data TEXT NOT NULL,
      metadata TEXT NOT NULL,
      layer TEXT NOT NULL,
      tags TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      expires_at DATETIME,
      last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
      access_count INTEGER DEFAULT 0,
      semantic_key TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_cache_key_hash ON cache_entries(key_hash);
    CREATE INDEX IF NOT EXISTS idx_cache_layer ON cache_entries(layer);
    CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
    CREATE INDEX IF NOT EXISTS idx_cache_semantic ON cache_entries(semantic_key);
  `,

  // Error tracking and recovery
  errors: `
    CREATE TABLE IF NOT EXISTS errors (
      id TEXT PRIMARY KEY,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      category TEXT NOT NULL,
      severity TEXT NOT NULL,
      code TEXT NOT NULL,
      message TEXT NOT NULL,
      stack TEXT,
      context TEXT,
      recoverable INTEGER DEFAULT 1,
      recovery_attempts INTEGER DEFAULT 0,
      resolved INTEGER DEFAULT 0,
      resolved_at DATETIME,
      resolution TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(timestamp);
    CREATE INDEX IF NOT EXISTS idx_errors_category ON errors(category);
    CREATE INDEX IF NOT EXISTS idx_errors_severity ON errors(severity);
    CREATE INDEX IF NOT EXISTS idx_errors_resolved ON errors(resolved);
  `,

  // Telemetry events, metrics, and alerts
  telemetry_events: `
    CREATE TABLE IF NOT EXISTS telemetry_events (
      id TEXT PRIMARY KEY,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      type TEXT NOT NULL,
      source TEXT NOT NULL,
      duration REAL,
      success INTEGER,
      metadata TEXT,
      tags TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_telemetry_type ON telemetry_events(type);
    CREATE INDEX IF NOT EXISTS idx_telemetry_source ON telemetry_events(source);
  `,

  telemetry_metrics: `
    CREATE TABLE IF NOT EXISTS telemetry_metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      value REAL NOT NULL,
      unit TEXT NOT NULL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      tags TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_metrics_name ON telemetry_metrics(name);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON telemetry_metrics(timestamp);
  `,

  telemetry_alerts: `
    CREATE TABLE IF NOT EXISTS telemetry_alerts (
      id TEXT PRIMARY KEY,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      severity TEXT NOT NULL,
      source TEXT NOT NULL,
      message TEXT NOT NULL,
      resolved INTEGER DEFAULT 0,
      resolved_at DATETIME
    );
    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON telemetry_alerts(timestamp);
    CREATE INDEX IF NOT EXISTS idx_alerts_severity ON telemetry_alerts(severity);
    CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON telemetry_alerts(resolved);
  `,

  // Issue tracking
  issues: `
    CREATE TABLE IF NOT EXISTS issues (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      issue_id TEXT UNIQUE,
      title TEXT NOT NULL,
      description TEXT,
      category TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'open',
      priority TEXT DEFAULT 'medium',
      assigned_to TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      closed_at DATETIME,
      tags TEXT,
      metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
    CREATE INDEX IF NOT EXISTS idx_issues_category ON issues(category);
    CREATE INDEX IF NOT EXISTS idx_issues_priority ON issues(priority);
  `,

  // Compliance scan results
  compliance_reports: `
    CREATE TABLE IF NOT EXISTS compliance_reports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      report_id TEXT UNIQUE,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      overall_score REAL,
      grade TEXT,
      file_count INTEGER DEFAULT 0,
      violation_count INTEGER DEFAULT 0,
      report_data TEXT NOT NULL,
      metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_compliance_timestamp ON compliance_reports(timestamp);
  `,

  // Security scan results
  security_reports: `
    CREATE TABLE IF NOT EXISTS security_reports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      report_id TEXT UNIQUE,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      scan_type TEXT,
      vulnerability_count INTEGER DEFAULT 0,
      high_severity INTEGER DEFAULT 0,
      medium_severity INTEGER DEFAULT 0,
      low_severity INTEGER DEFAULT 0,
      report_data TEXT NOT NULL,
      metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_reports(timestamp);
  `,

  // Metrics and statistics computed tables (views)
  telemetry_stats: `
    CREATE VIEW IF NOT EXISTS telemetry_stats AS
    SELECT
      COUNT(*) as events_total,
      json_group_object(type, COUNT(*)) as events_by_type,
      AVG(CASE WHEN success = 1 THEN duration ELSE NULL END) as avg_success_duration,
      AVG(CASE WHEN success = 0 THEN duration ELSE NULL END) as avg_failure_duration,
      (COUNT(CASE WHEN type = 'error' THEN 1 WHEN type LIKE '%.fail' THEN 1 WHEN success = 0 THEN 1 END) * 100.0 /
       NULLIF(COUNT(*), 0)) as error_rate_pct
    FROM telemetry_events
    WHERE timestamp > datetime('now', '-1 hour');
  `,

  error_stats: `
    CREATE VIEW IF NOT EXISTS error_stats AS
    SELECT
      COUNT(*) as total,
      COUNT(CASE WHEN resolved = 1 THEN 1 END) as recovered,
      COUNT(CASE WHEN resolved = 0 THEN 1 END) as unresolved,
      json_group_object(category, COUNT(*)) as by_category,
      json_group_object(severity, COUNT(*)) as by_severity
    FROM errors;
  `,

  cache_stats: `
    CREATE VIEW IF NOT EXISTS cache_stats AS
    SELECT
      COUNT(*) as total_entries,
      AVG(access_count) as avg_access_count
    FROM cache_entries;
  `,
};

// ============================================================================
// Database Manager
// ============================================================================

class DatabaseManager {
  private db: Database.Database;
  private initialized = false;

  constructor(dbPath: string = DB_PATH) {
    this.db = new Database(dbPath);
    this.init();
  }

  private init(): void {
    if (this.initialized) return;

    // Use WAL mode for better performance
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = 1000000'); // 1GB cache
    this.db.pragma('foreign_keys = ON');

    // Create all tables
    for (const [name, sql] of Object.entries(SCHEMA)) {
      try {
        this.db.exec(sql);
      } catch (error) {
        console.warn(`Failed to create ${name}:`, error);
      }
    }

    this.initialized = true;
  }

  // Get database connection
  getConnection(): Database.Database {
    return this.db;
  }

  // Execute a query with parameters
  execute(sql: string, params: unknown[] = []): Database.RunResult {
    const stmt = this.db.prepare(sql);
    return stmt.run(params);
  }

  // Query multiple rows
  query<T>(sql: string, params: unknown[] = []): T[] {
    const stmt = this.db.prepare(sql);
    return stmt.all(params) as T[];
  }

  // Query single row
  queryOne<T>(sql: string, params: unknown[] = []): T | undefined {
    const stmt = this.db.prepare(sql);
    return stmt.get(params) as T | undefined;
  }

  // Query scalar value
  queryScalar<T>(sql: string, params: unknown[] = []): T | null {
    const stmt = this.db.prepare(sql);
    return stmt.pluck().get(params) as T | null;
  }

  // Transaction wrapper
  transaction<T>(callback: () => T): T {
    return this.db.transaction(callback)();
  }

  // Close database
  close(): void {
    this.db.close();
  }

  // Migrate data from JSON files
  migrateFromJson(): { migrated: number; failed: number } {
    let migrated = 0;
    let failed = 0;

    // Migrate cache entries
    try {
      const cacheFile = path.join(AI_DIR, 'cache', 'cache.json');
      const cacheData = loadJson<{
        entries: Record<string, any>;
        stats: any;
      }>(cacheFile);

      if (cacheData?.entries) {
        this.migrateCacheEntries(Object.values(cacheData.entries));
        migrated++;
      }
    } catch {
      failed++;
    }

    // Migrate error log
    try {
      const errorFile = path.join(AI_DIR, 'error-log.json');
      const errorData = loadJson<{ errors: any[] }>(errorFile);

      if (errorData?.errors) {
        this.migrateErrors(errorData.errors);
        migrated++;
      }
    } catch {
      failed++;
    }

    // Migrate telemetry
    try {
      const telemetryFile = path.join(AI_DIR, 'telemetry.json');
      const telemetryData = loadJson<{
        events: any[];
        metrics: any[];
        alerts: any[];
      }>(telemetryFile);

      if (telemetryData) {
        this.migrateTelemetry(telemetryData);
        migrated++;
      }
    } catch {
      failed++;
    }

    return { migrated, failed };
  }

  private migrateCacheEntries(entries: any[]): void {
    const insertCacheEntry = this.db.prepare(`
      INSERT OR REPLACE INTO cache_entries
      (key_hash, key, data, metadata, layer, tags, expires_at, last_accessed, access_count, semantic_key)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const entry of entries) {
      const metadata = entry.metadata || {};
      insertCacheEntry.run([
        entry.hash,
        entry.key,
        JSON.stringify(entry.data),
        JSON.stringify(metadata),
        metadata.layer || 'result',
        JSON.stringify(metadata.tags || []),
        metadata.expiresAt,
        metadata.lastAccessedAt || metadata.createdAt,
        metadata.accessCount || 0,
        metadata.semanticKey,
      ]);
    }
  }

  private migrateErrors(errors: any[]): void {
    const insertError = this.db.prepare(`
      INSERT OR REPLACE INTO errors
      (id, timestamp, category, severity, code, message, stack, context, recoverable, recovery_attempts, resolved, resolved_at, resolution)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const error of errors) {
      insertError.run([
        error.id,
        error.timestamp,
        error.category,
        error.severity,
        error.code,
        error.message,
        error.stack,
        JSON.stringify(error.context || {}),
        error.recoverable ? 1 : 0,
        error.recoveryAttempts || 0,
        error.resolved ? 1 : 0,
        error.resolvedAt,
        error.resolution,
      ]);
    }
  }

  private migrateTelemetry(data: { events?: any[]; metrics?: any[]; alerts?: any[] }): void {
    // Migrate events
    if (data.events) {
      const insertEvent = this.db.prepare(`
        INSERT OR REPLACE INTO telemetry_events
        (id, timestamp, type, source, duration, success, metadata, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `);

      for (const event of data.events) {
        insertEvent.run([
          event.id,
          event.timestamp,
          event.type,
          event.source,
          event.duration,
          event.success !== undefined ? (event.success ? 1 : 0) : null,
          JSON.stringify(event.metadata || {}),
          JSON.stringify(event.tags || []),
        ]);
      }
    }

    // Migrate metrics
    if (data.metrics) {
      const insertMetric = this.db.prepare(`
        INSERT OR REPLACE INTO telemetry_metrics
        (name, value, unit, timestamp, tags)
        VALUES (?, ?, ?, ?, ?)
      `);

      for (const metric of data.metrics) {
        insertMetric.run([
          metric.name,
          metric.value,
          metric.unit,
          metric.timestamp,
          JSON.stringify(metric.tags || []),
        ]);
      }
    }

    // Migrate alerts
    if (data.alerts) {
      const insertAlert = this.db.prepare(`
        INSERT OR REPLACE INTO telemetry_alerts
        (id, timestamp, severity, source, message, resolved, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `);

      for (const alert of data.alerts) {
        insertAlert.run([
          alert.id,
          alert.timestamp,
          alert.severity,
          alert.source,
          alert.message,
          alert.resolved ? 1 : 0,
          alert.resolvedAt,
        ]);
      }
    }
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const db = new DatabaseManager();

// ============================================================================
// Database-backed Cache Interface
// ============================================================================

export interface DBCacheEntry extends Record<string, unknown> {
  id?: number;
  key_hash: string;
  key: string;
  data: string;
  metadata: string;
  layer: string;
  tags: string;
  created_at: string;
  expires_at?: string;
  last_accessed: string;
  access_count: number;
  semantic_key?: string;
}

export class DatabaseCache {
  private db: DatabaseManager;

  constructor(dbManager: DatabaseManager = db) {
    this.db = dbManager;
  }

  // Store an entry
  set<T>(
    key: string,
    data: T,
    layer: string,
    ttlMs?: number,
    metadata: Record<string, unknown> = {}
  ): void {
    const insert = this.db.getConnection().prepare(`
      INSERT OR REPLACE INTO cache_entries
      (key_hash, key, data, metadata, layer, tags, expires_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    const keyHash = crypto.createHash('sha256').update(key).digest('hex').substring(0, 16);
    const expiresAt = ttlMs ? new Date(Date.now() + ttlMs).toISOString() : null;

    insert.run([
      keyHash,
      key,
      JSON.stringify(data),
      JSON.stringify(metadata),
      layer,
      JSON.stringify([]),
      expiresAt,
    ]);
  }

  // Retrieve an entry
  get<T>(key: string): T | null {
    const select = this.db.getConnection().prepare(`
      SELECT data, last_accessed, access_count
      FROM cache_entries
      WHERE key_hash = ? AND (expires_at IS NULL OR expires_at > ?)
    `);

    const keyHash = crypto.createHash('sha256').update(key).digest('hex').substring(0, 16);
    const result = select.get([keyHash, new Date().toISOString()]) as
      | {
          data: string;
          last_accessed: string;
          access_count: number;
        }
      | undefined;

    if (result) {
      // Update access metadata
      const update = this.db.getConnection().prepare(`
        UPDATE cache_entries
        SET last_accessed = ?, access_count = access_count + 1
        WHERE key_hash = ?
      `);
      update.run([new Date().toISOString(), keyHash]);

      return JSON.parse(result.data) as T;
    }

    return null;
  }

  // Clear expired entries
  clearExpired(): number {
    const deleteExpired = this.db.getConnection().prepare(`
      DELETE FROM cache_entries
      WHERE expires_at < ?
    `);
    const result = deleteExpired.run([new Date().toISOString()]);
    return result.changes;
  }

  // Get cache statistics
  getStats(): {
    totalEntries: number;
    totalSize: number;
    hitRate: number;
    byLayer: Record<string, number>;
  } {
    const totalStats = this.db.queryOne<{ totalEntries: number; totalSize: number }>(
      'SELECT COUNT(*) as totalEntries, SUM(LENGTH(data)) as totalSize FROM cache_entries'
    );

    // Get layer distribution
    const layerStats = this.db.query<{ layer: string; count: number }>(
      'SELECT layer, COUNT(*) as count FROM cache_entries GROUP BY layer'
    );

    const byLayer: Record<string, number> = {};
    layerStats.forEach((ls) => {
      byLayer[ls.layer] = ls.count;
    });

    // Calculate rough hit rate (recent hits vs misses)
    const recent = this.db.query<{ type: string }>(
      `SELECT type FROM telemetry_events WHERE type IN ('cache.hit', 'cache.miss') AND timestamp > datetime('now', '-1 hour') LIMIT 1000`
    );

    const hits = recent.filter((r) => r.type === 'cache.hit').length;
    const misses = recent.filter((r) => r.type === 'cache.miss').length;
    const hitRate = hits + misses > 0 ? Math.round((hits / (hits + misses)) * 100) : 0;

    return {
      totalEntries: totalStats?.totalEntries || 0,
      totalSize: totalStats?.totalSize || 0,
      hitRate,
      byLayer,
    };
  }

  // Clear by layer
  clearLayer(layer: string): number {
    const deleteLayer = this.db.getConnection().prepare(`
      DELETE FROM cache_entries WHERE layer = ?
    `);
    const result = deleteLayer.run([layer]);
    return result.changes;
  }
}

// ============================================================================
// CLI Interface
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'init': {
      console.log('Database initialized at:', DB_PATH);
      const cache = new DatabaseCache();
      const stats = cache.getStats();
      console.log('Initial stats:', stats);
      break;
    }

    case 'migrate': {
      console.log('Migrating data from JSON files...');
      const result = db.migrateFromJson();
      console.log(`Migration complete: ${result.migrated} migrated, ${result.failed} failed`);
      break;
    }

    case 'stats': {
      const cache = new DatabaseCache();
      const stats = cache.getStats();
      console.log('Database Cache Stats:', stats);
      break;
    }

    case 'clear-expired': {
      const cache = new DatabaseCache();
      const cleared = cache.clearExpired();
      console.log(`Cleared ${cleared} expired entries`);
      break;
    }

    case 'dump': {
      const what = args[1];
      let query = '';

      switch (what) {
        case 'cache':
          query = 'SELECT * FROM cache_entries LIMIT 10';
          break;
        case 'errors':
          query = 'SELECT * FROM errors ORDER BY timestamp DESC LIMIT 10';
          break;
        case 'events':
          query = 'SELECT * FROM telemetry_events ORDER BY timestamp DESC LIMIT 10';
          break;
        case 'metrics':
          query = 'SELECT * FROM telemetry_metrics ORDER BY timestamp DESC LIMIT 10';
          break;
        case 'alerts':
          query = 'SELECT * FROM telemetry_alerts ORDER BY timestamp DESC LIMIT 10';
          break;
        default:
          console.log('Usage: database dump <cache|errors|events|metrics|alerts>');
          return;
      }

      const results = db.query(query);
      console.log(JSON.stringify(results, null, 2));
      break;
    }

    default:
      console.log(`
Database Manager - SQLite database operations

Commands:
  init              Initialize/recreate database
  migrate           Migrate data from JSON files
  stats             Show cache statistics
  clear-expired     Clear expired cache entries
  dump <table>      Dump recent records from a table

Tables: cache, errors, events, metrics, alerts

Examples:
  npm run db:init
  npm run db:migrate
  npm run db:stats
  npm run db:dump errors
      `);
  }
}

main();
