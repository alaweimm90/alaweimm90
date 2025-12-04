#!/usr/bin/env npx tsx
/**
 * Compliance Check Cache
 * Caches compliance results by file hash, invalidates on changes
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const ROOT = process.cwd();
const CACHE_PATH = path.join(ROOT, '.ai/cache/compliance-cache.json');
const CACHE_TTL_HOURS = 24;

export interface ComplianceResult {
  passed: boolean;
  score: number;
  issues: { type: string; message: string; line?: number }[];
  timestamp: string;
}

export interface CacheEntry {
  fileHash: string;
  filePath: string;
  result: ComplianceResult;
  cachedAt: string;
  expiresAt: string;
}

export interface ComplianceCache {
  entries: Record<string, CacheEntry>;
  stats: { hits: number; misses: number; invalidations: number };
  lastCleanup: string;
}

class ComplianceCacheManager {
  private cache: ComplianceCache;

  constructor() {
    this.ensureDirectory();
    this.cache = this.loadCache();
    this.cleanupExpired();
  }

  private ensureDirectory(): void {
    const dir = path.dirname(CACHE_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

  private loadCache(): ComplianceCache {
    if (fs.existsSync(CACHE_PATH)) {
      return JSON.parse(fs.readFileSync(CACHE_PATH, 'utf8'));
    }
    return {
      entries: {},
      stats: { hits: 0, misses: 0, invalidations: 0 },
      lastCleanup: new Date().toISOString(),
    };
  }

  private saveCache(): void {
    fs.writeFileSync(CACHE_PATH, JSON.stringify(this.cache, null, 2));
  }

  private computeFileHash(filePath: string): string | null {
    const fullPath = path.isAbsolute(filePath) ? filePath : path.join(ROOT, filePath);
    if (!fs.existsSync(fullPath)) return null;
    const content = fs.readFileSync(fullPath);
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  private cleanupExpired(): void {
    const now = new Date();
    let cleaned = 0;
    for (const [key, entry] of Object.entries(this.cache.entries)) {
      if (new Date(entry.expiresAt) < now) {
        delete this.cache.entries[key];
        cleaned++;
      }
    }
    if (cleaned > 0) {
      this.cache.lastCleanup = now.toISOString();
      this.saveCache();
    }
  }

  get(filePath: string): ComplianceResult | null {
    const hash = this.computeFileHash(filePath);
    if (!hash) return null;

    const entry = this.cache.entries[filePath];
    if (!entry) {
      this.cache.stats.misses++;
      this.saveCache();
      return null;
    }

    // Check if file has changed
    if (entry.fileHash !== hash) {
      delete this.cache.entries[filePath];
      this.cache.stats.invalidations++;
      this.cache.stats.misses++;
      this.saveCache();
      return null;
    }

    // Check if expired
    if (new Date(entry.expiresAt) < new Date()) {
      delete this.cache.entries[filePath];
      this.cache.stats.misses++;
      this.saveCache();
      return null;
    }

    this.cache.stats.hits++;
    this.saveCache();
    return entry.result;
  }

  set(filePath: string, result: ComplianceResult): void {
    const hash = this.computeFileHash(filePath);
    if (!hash) return;

    const now = new Date();
    const expiresAt = new Date(now.getTime() + CACHE_TTL_HOURS * 60 * 60 * 1000);

    this.cache.entries[filePath] = {
      fileHash: hash,
      filePath,
      result,
      cachedAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
    };
    this.saveCache();
  }

  invalidate(filePath: string): boolean {
    if (this.cache.entries[filePath]) {
      delete this.cache.entries[filePath];
      this.cache.stats.invalidations++;
      this.saveCache();
      return true;
    }
    return false;
  }

  invalidateAll(): number {
    const count = Object.keys(this.cache.entries).length;
    this.cache.entries = {};
    this.cache.stats.invalidations += count;
    this.saveCache();
    return count;
  }

  getStats(): {
    hits: number;
    misses: number;
    invalidations: number;
    size: number;
    hitRate: string;
  } {
    const total = this.cache.stats.hits + this.cache.stats.misses;
    const hitRate = total > 0 ? ((this.cache.stats.hits / total) * 100).toFixed(1) + '%' : 'N/A';
    return { ...this.cache.stats, size: Object.keys(this.cache.entries).length, hitRate };
  }
}

export const complianceCache = new ComplianceCacheManager();
export default ComplianceCacheManager;

// CLI
if (require.main === module || process.argv[1]?.includes('compliance-cache')) {
  const args = process.argv.slice(2);
  const cmd = args[0];

  switch (cmd) {
    case 'get': {
      const result = complianceCache.get(args[1]);
      if (result) console.log(JSON.stringify(result, null, 2));
      else console.log('❌ Not cached or file changed');
      break;
    }
    case 'set': {
      const [, filePath, passed, score, issuesJson] = args;
      if (!filePath) {
        console.log('Usage: compliance-cache set <file> <passed> <score> [issuesJson]');
        process.exit(1);
      }
      const result: ComplianceResult = {
        passed: passed === 'true',
        score: parseInt(score) || 100,
        issues: issuesJson ? JSON.parse(issuesJson) : [],
        timestamp: new Date().toISOString(),
      };
      complianceCache.set(filePath, result);
      console.log(`✅ Cached compliance result for ${filePath}`);
      break;
    }
    case 'invalidate':
      if (args[1] === '--all') {
        const count = complianceCache.invalidateAll();
        console.log(`✅ Invalidated ${count} entries`);
      } else {
        const success = complianceCache.invalidate(args[1]);
        console.log(success ? `✅ Invalidated ${args[1]}` : '❌ Not in cache');
      }
      break;
    case 'stats':
      console.log(JSON.stringify(complianceCache.getStats(), null, 2));
      break;
    default:
      console.log(`
Compliance Cache CLI

Commands:
  get <file>                              Get cached result for file
  set <file> <passed> <score> [issues]    Set cached result
  invalidate <file>                       Invalidate single file
  invalidate --all                        Invalidate all entries
  stats                                   Show cache statistics
`);
  }
}
