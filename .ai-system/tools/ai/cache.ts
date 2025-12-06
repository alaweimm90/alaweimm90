#!/usr/bin/env npx tsx
/**
 * Multi-Layer AI Cache
 * Enterprise-grade caching with semantic similarity, TTL management, and LRU eviction
 */

import * as path from 'path';
import * as crypto from 'crypto';
import { loadJson, saveJson, ensureDir } from './utils/file-persistence.js';

const AI_DIR = path.join(process.cwd(), '.ai');
const CACHE_DIR = path.join(AI_DIR, 'cache');
const CACHE_FILE = path.join(CACHE_DIR, 'cache.json');

// Ensure cache directory exists
ensureDir(CACHE_DIR);

// ============================================================================
// Types
// ============================================================================

interface CacheEntry<T> {
  key: string;
  hash: string;
  data: T;
  metadata: {
    createdAt: string;
    expiresAt: string;
    accessCount: number;
    lastAccessedAt: string;
    ttlMs: number;
    layer: CacheLayer;
    tags: string[];
    semanticKey?: string;
  };
}

interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  totalEntries: number;
  totalSizeBytes: number;
  hitRate: number;
  byLayer: Record<CacheLayer, { hits: number; misses: number; entries: number }>;
}

type CacheLayer = 'semantic' | 'template' | 'result' | 'analysis';

interface CacheConfig {
  maxEntries: number;
  maxSizeBytes: number;
  defaultTtlMs: number;
  ttlByLayer: Record<CacheLayer, number>;
  enableSemanticSimilarity: boolean;
  similarityThreshold: number;
  persistToDisk: boolean;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: CacheConfig = {
  maxEntries: 1000,
  maxSizeBytes: 50 * 1024 * 1024, // 50MB
  defaultTtlMs: 3600000, // 1 hour
  ttlByLayer: {
    semantic: 86400000, // 24 hours - semantic cache lasts longer
    template: 604800000, // 7 days - templates rarely change
    result: 3600000, // 1 hour - results expire quickly
    analysis: 21600000, // 6 hours - analysis moderately cached
  },
  enableSemanticSimilarity: true,
  similarityThreshold: 0.85,
  persistToDisk: true,
};

// ============================================================================
// Cache Implementation
// ============================================================================

class AICache {
  private memoryCache: Map<string, CacheEntry<unknown>> = new Map();
  private stats: CacheStats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    totalEntries: 0,
    totalSizeBytes: 0,
    hitRate: 0,
    byLayer: {
      semantic: { hits: 0, misses: 0, entries: 0 },
      template: { hits: 0, misses: 0, entries: 0 },
      result: { hits: 0, misses: 0, entries: 0 },
      analysis: { hits: 0, misses: 0, entries: 0 },
    },
  };
  private config: CacheConfig;

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.loadFromDisk();
  }

  // Generate cache key hash
  private generateHash(key: string): string {
    return crypto.createHash('sha256').update(key).digest('hex').substring(0, 16);
  }

  // Generate semantic key for similarity matching
  private generateSemanticKey(text: string): string {
    // Normalize and tokenize for semantic comparison
    const normalized = text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    // Extract key terms (simple TF approach)
    const words = normalized.split(' ').filter((w) => w.length > 3);
    const wordFreq = new Map<string, number>();
    words.forEach((w) => wordFreq.set(w, (wordFreq.get(w) || 0) + 1));

    // Get top 10 terms by frequency
    const topTerms = [...wordFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word)
      .sort()
      .join('|');

    return this.generateHash(topTerms);
  }

  // Calculate similarity between two semantic keys
  private calculateSimilarity(key1: string, key2: string): number {
    if (key1 === key2) return 1.0;

    // Jaccard similarity on character bigrams
    const getBigrams = (s: string): Set<string> => {
      const bigrams = new Set<string>();
      for (let i = 0; i < s.length - 1; i++) {
        bigrams.add(s.substring(i, i + 2));
      }
      return bigrams;
    };

    const bigrams1 = getBigrams(key1);
    const bigrams2 = getBigrams(key2);

    const intersection = new Set([...bigrams1].filter((x) => bigrams2.has(x)));
    const union = new Set([...bigrams1, ...bigrams2]);

    return intersection.size / union.size;
  }

  // Find semantically similar cache entry
  private findSimilar<T>(semanticKey: string, layer: CacheLayer): CacheEntry<T> | null {
    if (!this.config.enableSemanticSimilarity) return null;

    let bestMatch: CacheEntry<T> | null = null;
    let bestSimilarity = 0;

    for (const entry of this.memoryCache.values()) {
      if (entry.metadata.layer !== layer) continue;
      if (!entry.metadata.semanticKey) continue;

      const similarity = this.calculateSimilarity(semanticKey, entry.metadata.semanticKey);
      if (similarity > this.config.similarityThreshold && similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = entry as CacheEntry<T>;
      }
    }

    return bestMatch;
  }

  // Check if entry is expired
  private isExpired(entry: CacheEntry<unknown>): boolean {
    return new Date(entry.metadata.expiresAt) < new Date();
  }

  // Evict LRU entries when cache is full
  private evictIfNeeded(): void {
    while (this.memoryCache.size >= this.config.maxEntries) {
      // Find LRU entry
      let lruKey: string | null = null;
      let lruTime = Infinity;

      for (const [key, entry] of this.memoryCache.entries()) {
        const accessTime = new Date(entry.metadata.lastAccessedAt).getTime();
        if (accessTime < lruTime) {
          lruTime = accessTime;
          lruKey = key;
        }
      }

      if (lruKey) {
        const entry = this.memoryCache.get(lruKey);
        if (entry) {
          this.stats.byLayer[entry.metadata.layer].entries--;
        }
        this.memoryCache.delete(lruKey);
        this.stats.evictions++;
      }
    }
  }

  // Get entry from cache
  get<T>(key: string, layer: CacheLayer, enableSemantic = true): T | null {
    const hash = this.generateHash(key);
    const entry = this.memoryCache.get(hash) as CacheEntry<T> | undefined;

    if (entry && !this.isExpired(entry)) {
      // Update access metadata
      entry.metadata.accessCount++;
      entry.metadata.lastAccessedAt = new Date().toISOString();

      this.stats.hits++;
      this.stats.byLayer[layer].hits++;
      this.updateHitRate();

      return entry.data;
    }

    // Try semantic similarity match
    if (enableSemantic && this.config.enableSemanticSimilarity) {
      const semanticKey = this.generateSemanticKey(key);
      const similar = this.findSimilar<T>(semanticKey, layer);

      if (similar) {
        similar.metadata.accessCount++;
        similar.metadata.lastAccessedAt = new Date().toISOString();

        this.stats.hits++;
        this.stats.byLayer[layer].hits++;
        this.updateHitRate();

        return similar.data;
      }
    }

    this.stats.misses++;
    this.stats.byLayer[layer].misses++;
    this.updateHitRate();

    return null;
  }

  // Set entry in cache
  set<T>(key: string, data: T, layer: CacheLayer, tags: string[] = []): void {
    this.evictIfNeeded();

    const hash = this.generateHash(key);
    const ttl = this.config.ttlByLayer[layer];
    const now = new Date();

    const entry: CacheEntry<T> = {
      key,
      hash,
      data,
      metadata: {
        createdAt: now.toISOString(),
        expiresAt: new Date(now.getTime() + ttl).toISOString(),
        accessCount: 1,
        lastAccessedAt: now.toISOString(),
        ttlMs: ttl,
        layer,
        tags,
        semanticKey: this.config.enableSemanticSimilarity
          ? this.generateSemanticKey(key)
          : undefined,
      },
    };

    // Check if updating existing entry
    if (!this.memoryCache.has(hash)) {
      this.stats.byLayer[layer].entries++;
    }

    this.memoryCache.set(hash, entry);
    this.stats.totalEntries = this.memoryCache.size;

    if (this.config.persistToDisk) {
      this.saveToDisk();
    }
  }

  // Invalidate by key
  invalidate(key: string): boolean {
    const hash = this.generateHash(key);
    const entry = this.memoryCache.get(hash);

    if (entry) {
      this.stats.byLayer[entry.metadata.layer].entries--;
      this.memoryCache.delete(hash);
      return true;
    }
    return false;
  }

  // Invalidate by tag
  invalidateByTag(tag: string): number {
    let count = 0;
    for (const [hash, entry] of this.memoryCache.entries()) {
      if (entry.metadata.tags.includes(tag)) {
        this.stats.byLayer[entry.metadata.layer].entries--;
        this.memoryCache.delete(hash);
        count++;
      }
    }
    return count;
  }

  // Invalidate entire layer
  invalidateLayer(layer: CacheLayer): number {
    let count = 0;
    for (const [hash, entry] of this.memoryCache.entries()) {
      if (entry.metadata.layer === layer) {
        this.memoryCache.delete(hash);
        count++;
      }
    }
    this.stats.byLayer[layer].entries = 0;
    return count;
  }

  // Clear all expired entries
  clearExpired(): number {
    let count = 0;
    for (const [hash, entry] of this.memoryCache.entries()) {
      if (this.isExpired(entry)) {
        this.stats.byLayer[entry.metadata.layer].entries--;
        this.memoryCache.delete(hash);
        count++;
      }
    }
    return count;
  }

  // Get cache statistics
  getStats(): CacheStats {
    return { ...this.stats };
  }

  // Update hit rate calculation
  private updateHitRate(): void {
    const total = this.stats.hits + this.stats.misses;
    this.stats.hitRate = total > 0 ? Math.round((this.stats.hits / total) * 100) : 0;
  }

  // Save cache to disk
  private saveToDisk(): void {
    if (!this.config.persistToDisk) return;

    const data = {
      entries: Object.fromEntries(this.memoryCache),
      stats: this.stats,
      savedAt: new Date().toISOString(),
    };

    saveJson(CACHE_FILE, data);
  }

  // Load cache from disk
  private loadFromDisk(): void {
    if (!this.config.persistToDisk) return;

    const data = loadJson<{
      entries: Record<string, CacheEntry<unknown>>;
      stats: CacheStats;
    }>(CACHE_FILE);

    if (data) {
      this.memoryCache = new Map(Object.entries(data.entries || {}));
      this.stats = data.stats || this.stats;
      // Clear expired entries on load
      this.clearExpired();
    }
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const cache = new AICache();

// ============================================================================
// CLI Interface
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'stats': {
      const stats = cache.getStats();
      console.log('\n📊 AI Cache Statistics\n');
      console.log(`Total Entries: ${stats.totalEntries}`);
      console.log(`Hit Rate: ${stats.hitRate}%`);
      console.log(`Hits: ${stats.hits} | Misses: ${stats.misses}`);
      console.log(`Evictions: ${stats.evictions}`);
      console.log('\nBy Layer:');
      for (const [layer, data] of Object.entries(stats.byLayer)) {
        console.log(
          `  ${layer}: ${data.entries} entries, ${data.hits}/${data.hits + data.misses} hits`
        );
      }
      break;
    }

    case 'clear': {
      const layer = args[1] as CacheLayer | undefined;
      if (layer) {
        const count = cache.invalidateLayer(layer);
        console.log(`Cleared ${count} entries from ${layer} layer`);
      } else {
        const expired = cache.clearExpired();
        console.log(`Cleared ${expired} expired entries`);
      }
      break;
    }

    case 'set': {
      const key = args[1];
      const value = args[2];
      const layer = (args[3] as CacheLayer) || 'result';
      if (key && value) {
        cache.set(key, value, layer);
        console.log(`Cached: ${key} → ${layer} layer`);
      }
      break;
    }

    case 'get': {
      const key = args[1];
      const layer = (args[2] as CacheLayer) || 'result';
      if (key) {
        const value = cache.get(key, layer);
        console.log(value ? `Hit: ${JSON.stringify(value)}` : 'Miss');
      }
      break;
    }

    default:
      console.log(`
AI Cache - Multi-layer caching with semantic similarity

Commands:
  stats              Show cache statistics
  clear [layer]      Clear expired entries or specific layer
  set <key> <value> [layer]   Set a cache entry
  get <key> [layer]           Get a cache entry

Layers: semantic, template, result, analysis
      `);
  }
}

main();
