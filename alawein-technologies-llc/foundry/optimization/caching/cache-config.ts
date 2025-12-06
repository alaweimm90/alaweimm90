/**
 * Comprehensive Caching Strategy Configuration
 * Multi-layer caching for optimal performance
 */

import { QueryClient, QueryClientConfig } from '@tanstack/react-query';
import { AxiosRequestConfig } from 'axios';

/**
 * Cache Layer Types
 */
export enum CacheLayer {
  BROWSER = 'browser',
  MEMORY = 'memory',
  SERVICE_WORKER = 'service_worker',
  CDN = 'cdn',
  API = 'api',
  DATABASE = 'database'
}

/**
 * Cache Strategy Types
 */
export enum CacheStrategy {
  CACHE_FIRST = 'cache-first',
  NETWORK_FIRST = 'network-first',
  STALE_WHILE_REVALIDATE = 'stale-while-revalidate',
  NETWORK_ONLY = 'network-only',
  CACHE_ONLY = 'cache-only'
}

/**
 * Cache Configuration Interface
 */
export interface CacheConfig {
  layer: CacheLayer;
  strategy: CacheStrategy;
  ttl: number; // Time to live in milliseconds
  maxAge?: number;
  staleTime?: number;
  gcTime?: number;
  maxEntries?: number;
  sizeLimit?: number; // In bytes
  invalidateOn?: string[];
  tags?: string[];
}

/**
 * React Query Cache Configuration
 */
export const reactQueryConfig: QueryClientConfig = {
  defaultOptions: {
    queries: {
      // Data considered fresh for 5 minutes
      staleTime: 5 * 60 * 1000,

      // Cache garbage collected after 10 minutes
      gcTime: 10 * 60 * 1000,

      // Retry failed requests twice
      retry: 2,

      // Retry delay exponential backoff
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

      // Don't refetch on window focus in production
      refetchOnWindowFocus: process.env.NODE_ENV === 'development',

      // Don't refetch on reconnect automatically
      refetchOnReconnect: 'always',

      // Refetch data every 30 minutes
      refetchInterval: 30 * 60 * 1000,

      // Pause refetch when window is not visible
      refetchIntervalInBackground: false,
    },
    mutations: {
      // Retry mutations once
      retry: 1,

      // Retry delay for mutations
      retryDelay: 1000,
    },
  },
};

/**
 * Create configured Query Client
 */
export function createQueryClient(): QueryClient {
  return new QueryClient(reactQueryConfig);
}

/**
 * Cache Configuration by Resource Type
 */
export const cacheConfigurations: Record<string, CacheConfig> = {
  // Static Assets
  'static-assets': {
    layer: CacheLayer.CDN,
    strategy: CacheStrategy.CACHE_FIRST,
    ttl: 365 * 24 * 60 * 60 * 1000, // 1 year
    maxAge: 31536000,
  },

  // Images
  'images': {
    layer: CacheLayer.BROWSER,
    strategy: CacheStrategy.CACHE_FIRST,
    ttl: 30 * 24 * 60 * 60 * 1000, // 30 days
    maxAge: 2592000,
    maxEntries: 100,
    sizeLimit: 50 * 1024 * 1024, // 50MB
  },

  // API - User Data
  'user-data': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.STALE_WHILE_REVALIDATE,
    ttl: 5 * 60 * 1000, // 5 minutes
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    invalidateOn: ['user-update', 'logout'],
    tags: ['user', 'profile'],
  },

  // API - Research Papers
  'research-papers': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.CACHE_FIRST,
    ttl: 24 * 60 * 60 * 1000, // 24 hours
    staleTime: 60 * 60 * 1000, // 1 hour
    gcTime: 7 * 24 * 60 * 60 * 1000, // 7 days
    maxEntries: 500,
    tags: ['research', 'papers'],
  },

  // API - Matches
  'matches': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.NETWORK_FIRST,
    ttl: 10 * 60 * 1000, // 10 minutes
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
    invalidateOn: ['new-match', 'match-update'],
    tags: ['matches', 'connections'],
  },

  // API - Ideas
  'ideas': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.STALE_WHILE_REVALIDATE,
    ttl: 15 * 60 * 1000, // 15 minutes
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 60 * 60 * 1000, // 1 hour
    maxEntries: 200,
    invalidateOn: ['idea-create', 'idea-update', 'idea-delete'],
    tags: ['ideas', 'innovation'],
  },

  // API - Analytics
  'analytics': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.STALE_WHILE_REVALIDATE,
    ttl: 30 * 60 * 1000, // 30 minutes
    staleTime: 15 * 60 * 1000, // 15 minutes
    gcTime: 2 * 60 * 60 * 1000, // 2 hours
    tags: ['analytics', 'metrics'],
  },

  // Fonts
  'fonts': {
    layer: CacheLayer.BROWSER,
    strategy: CacheStrategy.CACHE_FIRST,
    ttl: 365 * 24 * 60 * 60 * 1000, // 1 year
    maxAge: 31536000,
  },

  // HTML Pages
  'html-pages': {
    layer: CacheLayer.SERVICE_WORKER,
    strategy: CacheStrategy.NETWORK_FIRST,
    ttl: 60 * 60 * 1000, // 1 hour
    maxAge: 3600,
  },

  // Real-time Data
  'realtime': {
    layer: CacheLayer.MEMORY,
    strategy: CacheStrategy.NETWORK_ONLY,
    ttl: 0, // No caching
    tags: ['realtime', 'live'],
  },
};

/**
 * Cache Invalidation Manager
 */
export class CacheInvalidationManager {
  private invalidationRules: Map<string, Set<string>> = new Map();
  private queryClient: QueryClient | null = null;

  constructor(queryClient?: QueryClient) {
    this.queryClient = queryClient || null;
    this.setupInvalidationRules();
  }

  /**
   * Setup invalidation rules
   */
  private setupInvalidationRules() {
    // User actions
    this.addRule('user-login', ['user-data', 'preferences', 'session']);
    this.addRule('user-logout', ['user-data', 'preferences', 'session', 'matches', 'ideas']);
    this.addRule('user-update', ['user-data', 'profile']);

    // Match actions
    this.addRule('new-match', ['matches', 'notifications']);
    this.addRule('match-accept', ['matches', 'connections']);
    this.addRule('match-reject', ['matches']);

    // Idea actions
    this.addRule('idea-create', ['ideas', 'user-stats']);
    this.addRule('idea-update', ['ideas']);
    this.addRule('idea-delete', ['ideas', 'user-stats']);

    // Research actions
    this.addRule('paper-upload', ['research-papers', 'user-stats']);
    this.addRule('paper-delete', ['research-papers', 'user-stats']);
  }

  /**
   * Add invalidation rule
   */
  addRule(action: string, tags: string[]) {
    if (!this.invalidationRules.has(action)) {
      this.invalidationRules.set(action, new Set());
    }
    tags.forEach(tag => this.invalidationRules.get(action)!.add(tag));
  }

  /**
   * Invalidate caches based on action
   */
  async invalidate(action: string) {
    const tags = this.invalidationRules.get(action);
    if (!tags) return;

    const promises: Promise<void>[] = [];

    // Invalidate React Query cache
    if (this.queryClient) {
      tags.forEach(tag => {
        promises.push(
          this.queryClient!.invalidateQueries({
            predicate: (query) => {
              const queryKey = query.queryKey as string[];
              return queryKey.some(key =>
                typeof key === 'string' && key.includes(tag)
              );
            }
          })
        );
      });
    }

    // Invalidate Service Worker cache
    if ('caches' in window) {
      tags.forEach(tag => {
        promises.push(this.invalidateServiceWorkerCache(tag));
      });
    }

    // Invalidate browser cache (using cache busting)
    this.invalidateBrowserCache(tags);

    await Promise.all(promises);
  }

  /**
   * Invalidate Service Worker cache
   */
  private async invalidateServiceWorkerCache(tag: string) {
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      const promises = cacheNames
        .filter(name => name.includes(tag))
        .map(name => caches.delete(name));
      await Promise.all(promises);
    }
  }

  /**
   * Invalidate browser cache using cache busting
   */
  private invalidateBrowserCache(tags: Set<string>) {
    // Update cache version in localStorage
    const cacheVersion = localStorage.getItem('cache-version') || '1';
    const newVersion = String(parseInt(cacheVersion) + 1);
    localStorage.setItem('cache-version', newVersion);

    // Store invalidated tags
    const invalidatedTags = JSON.stringify(Array.from(tags));
    localStorage.setItem('invalidated-tags', invalidatedTags);
    localStorage.setItem('invalidation-timestamp', String(Date.now()));
  }

  /**
   * Clear all caches
   */
  async clearAll() {
    // Clear React Query cache
    if (this.queryClient) {
      this.queryClient.clear();
    }

    // Clear Service Worker caches
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map(name => caches.delete(name)));
    }

    // Clear localStorage cache data
    localStorage.removeItem('cache-version');
    localStorage.removeItem('invalidated-tags');
    localStorage.removeItem('invalidation-timestamp');

    // Clear sessionStorage
    sessionStorage.clear();
  }

  /**
   * Get cache statistics
   */
  async getCacheStats() {
    const stats = {
      reactQuery: {
        queriesCount: 0,
        mutationsCount: 0,
      },
      serviceWorker: {
        caches: [] as { name: string; count: number; size: number }[],
      },
      localStorage: {
        used: 0,
        available: 0,
      },
      sessionStorage: {
        used: 0,
        available: 0,
      },
    };

    // React Query stats
    if (this.queryClient) {
      const cache = this.queryClient.getQueryCache();
      stats.reactQuery.queriesCount = cache.getAll().length;

      const mutationCache = this.queryClient.getMutationCache();
      stats.reactQuery.mutationsCount = mutationCache.getAll().length;
    }

    // Service Worker cache stats
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const requests = await cache.keys();
        let totalSize = 0;

        for (const request of requests) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.blob();
            totalSize += blob.size;
          }
        }

        stats.serviceWorker.caches.push({
          name,
          count: requests.length,
          size: totalSize,
        });
      }
    }

    // Storage stats
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      stats.localStorage.used = estimate.usage || 0;
      stats.localStorage.available = estimate.quota || 0;
    }

    return stats;
  }
}

/**
 * Axios Request Interceptor for Caching
 */
export function setupAxiosCaching(axiosInstance: any) {
  // Request interceptor - add cache headers
  axiosInstance.interceptors.request.use((config: AxiosRequestConfig) => {
    const cacheConfig = getCacheConfigForEndpoint(config.url || '');

    if (cacheConfig) {
      config.headers = {
        ...config.headers,
        'Cache-Control': getCacheControlHeader(cacheConfig),
        'X-Cache-Strategy': cacheConfig.strategy,
      };
    }

    return config;
  });

  // Response interceptor - handle cache headers
  axiosInstance.interceptors.response.use(
    (response: any) => {
      // Store cache metadata
      if (response.config.url) {
        const cacheKey = `cache-meta-${response.config.url}`;
        const metadata = {
          timestamp: Date.now(),
          etag: response.headers.etag,
          lastModified: response.headers['last-modified'],
        };
        sessionStorage.setItem(cacheKey, JSON.stringify(metadata));
      }

      return response;
    },
    (error: any) => {
      // Try to serve from cache if network fails
      if (error.code === 'ECONNABORTED' || !navigator.onLine) {
        return serveFromCache(error.config);
      }
      return Promise.reject(error);
    }
  );
}

/**
 * Get cache configuration for endpoint
 */
function getCacheConfigForEndpoint(url: string): CacheConfig | null {
  // Match URL patterns to cache configurations
  if (url.includes('/api/user')) return cacheConfigurations['user-data'];
  if (url.includes('/api/papers')) return cacheConfigurations['research-papers'];
  if (url.includes('/api/matches')) return cacheConfigurations['matches'];
  if (url.includes('/api/ideas')) return cacheConfigurations['ideas'];
  if (url.includes('/api/analytics')) return cacheConfigurations['analytics'];
  if (url.includes('/api/realtime')) return cacheConfigurations['realtime'];

  return null;
}

/**
 * Get Cache-Control header from configuration
 */
function getCacheControlHeader(config: CacheConfig): string {
  const directives: string[] = [];

  if (config.strategy === CacheStrategy.CACHE_FIRST) {
    directives.push('public');
    if (config.maxAge) {
      directives.push(`max-age=${config.maxAge}`);
    }
  } else if (config.strategy === CacheStrategy.NETWORK_FIRST) {
    directives.push('no-cache');
  } else if (config.strategy === CacheStrategy.STALE_WHILE_REVALIDATE) {
    directives.push('public');
    if (config.maxAge) {
      directives.push(`max-age=${config.maxAge}`);
    }
    if (config.staleTime) {
      directives.push(`stale-while-revalidate=${config.staleTime / 1000}`);
    }
  } else if (config.strategy === CacheStrategy.NETWORK_ONLY) {
    directives.push('no-store');
  }

  return directives.join(', ');
}

/**
 * Serve response from cache when offline
 */
async function serveFromCache(config: AxiosRequestConfig): Promise<any> {
  if (!config.url) {
    return Promise.reject(new Error('No URL provided'));
  }

  // Try to get from React Query cache first
  const queryClient = (window as any).__queryClient;
  if (queryClient) {
    const queryKey = [config.url, config.params];
    const data = queryClient.getQueryData(queryKey);
    if (data) {
      return { data, cached: true, status: 200 };
    }
  }

  // Try Service Worker cache
  if ('caches' in window) {
    const cache = await caches.open('api-cache');
    const request = new Request(config.url);
    const response = await cache.match(request);

    if (response) {
      const data = await response.json();
      return { data, cached: true, status: 200 };
    }
  }

  return Promise.reject(new Error('No cached data available'));
}

// Export singleton instance
export const cacheInvalidationManager = new CacheInvalidationManager();