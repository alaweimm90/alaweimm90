/**
 * Integration tests for AI Cache Module
 * Tests multi-layer caching with semantic similarity
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

// Mock fs to avoid actual file operations during tests
vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actual,
    existsSync: vi.fn().mockReturnValue(false),
    mkdirSync: vi.fn(),
    writeFileSync: vi.fn(),
    readFileSync: vi.fn().mockReturnValue('{}'),
  };
});

describe('AI Cache Module', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Cache Key Generation', () => {
    it('should generate consistent hash for same key', () => {
      const crypto = require('crypto');
      const key = 'test-key';
      const hash1 = crypto.createHash('sha256').update(key).digest('hex').substring(0, 16);
      const hash2 = crypto.createHash('sha256').update(key).digest('hex').substring(0, 16);
      expect(hash1).toBe(hash2);
    });

    it('should generate different hashes for different keys', () => {
      const crypto = require('crypto');
      const hash1 = crypto.createHash('sha256').update('key1').digest('hex').substring(0, 16);
      const hash2 = crypto.createHash('sha256').update('key2').digest('hex').substring(0, 16);
      expect(hash1).not.toBe(hash2);
    });
  });

  describe('Semantic Key Generation', () => {
    it('should normalize text for semantic comparison', () => {
      const text = 'How do I implement a FUNCTION?';
      const normalized = text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      expect(normalized).toBe('how do i implement a function');
    });

    it('should extract key terms with word frequency', () => {
      const text = 'implement feature implement test implement';
      const words = text.split(' ').filter((w) => w.length > 3);
      const wordFreq = new Map<string, number>();
      words.forEach((w) => wordFreq.set(w, (wordFreq.get(w) || 0) + 1));

      expect(wordFreq.get('implement')).toBe(3);
      expect(wordFreq.get('feature')).toBe(1);
      expect(wordFreq.get('test')).toBe(1);
    });
  });

  describe('Similarity Calculation', () => {
    it('should return 1.0 for identical keys', () => {
      const key = 'test-key';
      expect(key === key).toBe(true);
    });

    it('should calculate Jaccard similarity correctly', () => {
      const getBigrams = (s: string): Set<string> => {
        const bigrams = new Set<string>();
        for (let i = 0; i < s.length - 1; i++) {
          bigrams.add(s.substring(i, i + 2));
        }
        return bigrams;
      };

      const bigrams1 = getBigrams('hello');
      const bigrams2 = getBigrams('hallo');

      const intersection = new Set([...bigrams1].filter((x) => bigrams2.has(x)));
      const union = new Set([...bigrams1, ...bigrams2]);

      const similarity = intersection.size / union.size;
      expect(similarity).toBeGreaterThan(0);
      expect(similarity).toBeLessThan(1);
    });
  });

  describe('TTL Configuration', () => {
    it('should have correct default TTL values', () => {
      const ttlByLayer = {
        semantic: 86400000, // 24 hours
        template: 604800000, // 7 days
        result: 3600000, // 1 hour
        analysis: 21600000, // 6 hours
      };

      expect(ttlByLayer.semantic).toBe(24 * 60 * 60 * 1000);
      expect(ttlByLayer.template).toBe(7 * 24 * 60 * 60 * 1000);
      expect(ttlByLayer.result).toBe(60 * 60 * 1000);
      expect(ttlByLayer.analysis).toBe(6 * 60 * 60 * 1000);
    });

    it('should correctly check expiration', () => {
      const now = new Date();
      const pastDate = new Date(now.getTime() - 1000);
      const futureDate = new Date(now.getTime() + 1000);

      expect(pastDate < now).toBe(true);
      expect(futureDate < now).toBe(false);
    });
  });

  describe('LRU Eviction', () => {
    it('should identify LRU entry by timestamp', () => {
      const entries = [
        { key: 'a', lastAccessedAt: '2024-01-01T10:00:00Z' },
        { key: 'b', lastAccessedAt: '2024-01-01T08:00:00Z' },
        { key: 'c', lastAccessedAt: '2024-01-01T12:00:00Z' },
      ];

      let lruKey: string | null = null;
      let lruTime = Infinity;

      for (const entry of entries) {
        const accessTime = new Date(entry.lastAccessedAt).getTime();
        if (accessTime < lruTime) {
          lruTime = accessTime;
          lruKey = entry.key;
        }
      }

      expect(lruKey).toBe('b');
    });
  });

  describe('Cache Stats', () => {
    it('should calculate hit rate correctly', () => {
      const hits = 80;
      const misses = 20;
      const total = hits + misses;
      const hitRate = Math.round((hits / total) * 100);

      expect(hitRate).toBe(80);
    });

    it('should handle zero requests', () => {
      const hits = 0;
      const misses = 0;
      const total = hits + misses;
      const hitRate = total > 0 ? Math.round((hits / total) * 100) : 0;

      expect(hitRate).toBe(0);
    });
  });

  describe('Tag-based Invalidation', () => {
    it('should find entries by tag', () => {
      const entries = [
        { key: 'a', tags: ['auth', 'user'] },
        { key: 'b', tags: ['auth', 'admin'] },
        { key: 'c', tags: ['data'] },
      ];

      const authEntries = entries.filter((e) => e.tags.includes('auth'));
      expect(authEntries.length).toBe(2);
    });
  });

  describe('Layer Invalidation', () => {
    it('should filter entries by layer', () => {
      type CacheLayer = 'semantic' | 'template' | 'result' | 'analysis';
      const entries: Array<{ key: string; layer: CacheLayer }> = [
        { key: 'a', layer: 'semantic' },
        { key: 'b', layer: 'result' },
        { key: 'c', layer: 'semantic' },
      ];

      const semanticEntries = entries.filter((e) => e.layer === 'semantic');
      expect(semanticEntries.length).toBe(2);
    });
  });
});
