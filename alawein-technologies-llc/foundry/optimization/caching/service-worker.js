/**
 * Service Worker for Advanced Caching
 * Implements multiple caching strategies for optimal performance
 */

const CACHE_VERSION = 'v1.0.0';
const CACHE_NAMES = {
  STATIC: `static-${CACHE_VERSION}`,
  DYNAMIC: `dynamic-${CACHE_VERSION}`,
  API: `api-${CACHE_VERSION}`,
  IMAGES: `images-${CACHE_VERSION}`,
  FONTS: `fonts-${CACHE_VERSION}`,
};

// Assets to precache
const PRECACHE_ASSETS = [
  '/',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/_next/static/css/*.css',
  '/_next/static/js/*.js',
];

// Cache strategies
const CACHE_STRATEGIES = {
  CACHE_FIRST: 'cache-first',
  NETWORK_FIRST: 'network-first',
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate',
  NETWORK_ONLY: 'network-only',
  CACHE_ONLY: 'cache-only',
};

// Route to strategy mapping
const ROUTE_STRATEGIES = new Map([
  // Static assets - Cache First
  [/\.(js|css|woff2?|ttf|otf|eot)$/, CACHE_STRATEGIES.CACHE_FIRST],

  // Images - Cache First with fallback
  [/\.(png|jpg|jpeg|gif|svg|webp|avif|ico)$/, CACHE_STRATEGIES.CACHE_FIRST],

  // API calls - Stale While Revalidate
  [/^\/api\/(?!realtime)/, CACHE_STRATEGIES.STALE_WHILE_REVALIDATE],

  // Real-time API - Network Only
  [/^\/api\/realtime/, CACHE_STRATEGIES.NETWORK_ONLY],

  // HTML pages - Network First
  [/\.html?$/, CACHE_STRATEGIES.NETWORK_FIRST],

  // Default - Network First
  [/.*/, CACHE_STRATEGIES.NETWORK_FIRST],
]);

// Install event - Precache essential assets
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');

  event.waitUntil(
    caches
      .open(CACHE_NAMES.STATIC)
      .then((cache) => {
        console.log('[ServiceWorker] Precaching assets');
        return cache.addAll(PRECACHE_ASSETS.filter(asset => !asset.includes('*')));
      })
      .then(() => {
        console.log('[ServiceWorker] Installation complete');
        return self.skipWaiting();
      })
  );
});

// Activate event - Clean old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');

  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((cacheName) => {
              return !Object.values(CACHE_NAMES).includes(cacheName);
            })
            .map((cacheName) => {
              console.log('[ServiceWorker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      })
      .then(() => {
        console.log('[ServiceWorker] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - Implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-HTTP(S) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // Skip cross-origin requests (except for CDN)
  if (url.origin !== location.origin && !url.hostname.includes('cdn')) {
    return;
  }

  // Determine strategy based on request
  const strategy = getStrategy(request);

  // Handle request based on strategy
  switch (strategy) {
    case CACHE_STRATEGIES.CACHE_FIRST:
      event.respondWith(cacheFirst(request));
      break;
    case CACHE_STRATEGIES.NETWORK_FIRST:
      event.respondWith(networkFirst(request));
      break;
    case CACHE_STRATEGIES.STALE_WHILE_REVALIDATE:
      event.respondWith(staleWhileRevalidate(request));
      break;
    case CACHE_STRATEGIES.NETWORK_ONLY:
      event.respondWith(networkOnly(request));
      break;
    case CACHE_STRATEGIES.CACHE_ONLY:
      event.respondWith(cacheOnly(request));
      break;
    default:
      event.respondWith(networkFirst(request));
  }
});

// Cache First Strategy
async function cacheFirst(request) {
  try {
    const cached = await caches.match(request);
    if (cached) {
      console.log('[ServiceWorker] Cache hit:', request.url);
      return cached;
    }

    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(getCacheName(request));
      await cache.put(request, response.clone());
      console.log('[ServiceWorker] Cached:', request.url);
    }
    return response;
  } catch (error) {
    console.error('[ServiceWorker] Cache first failed:', error);
    return createErrorResponse('Resource not available');
  }
}

// Network First Strategy
async function networkFirst(request) {
  try {
    const response = await fetchWithTimeout(request, 5000);
    if (response.ok) {
      const cache = await caches.open(getCacheName(request));
      await cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log('[ServiceWorker] Network failed, trying cache:', request.url);
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }

    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const offlinePage = await caches.match('/offline.html');
      if (offlinePage) {
        return offlinePage;
      }
    }

    return createErrorResponse('Network error');
  }
}

// Stale While Revalidate Strategy
async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        const cache = caches.open(getCacheName(request));
        cache.then((c) => c.put(request, response.clone()));
      }
      return response;
    })
    .catch((error) => {
      console.error('[ServiceWorker] Background fetch failed:', error);
      return cached || createErrorResponse('Resource not available');
    });

  return cached || fetchPromise;
}

// Network Only Strategy
async function networkOnly(request) {
  try {
    return await fetch(request);
  } catch (error) {
    console.error('[ServiceWorker] Network only failed:', error);
    return createErrorResponse('Network error');
  }
}

// Cache Only Strategy
async function cacheOnly(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  return createErrorResponse('Resource not in cache');
}

// Helper Functions

// Get strategy for request
function getStrategy(request) {
  const url = new URL(request.url);

  for (const [pattern, strategy] of ROUTE_STRATEGIES) {
    if (pattern.test(url.pathname) || pattern.test(url.href)) {
      return strategy;
    }
  }

  return CACHE_STRATEGIES.NETWORK_FIRST;
}

// Get appropriate cache name
function getCacheName(request) {
  const url = new URL(request.url);

  if (/\.(js|css)$/.test(url.pathname)) {
    return CACHE_NAMES.STATIC;
  }
  if (/\.(png|jpg|jpeg|gif|svg|webp|avif|ico)$/.test(url.pathname)) {
    return CACHE_NAMES.IMAGES;
  }
  if (/\.(woff2?|ttf|otf|eot)$/.test(url.pathname)) {
    return CACHE_NAMES.FONTS;
  }
  if (url.pathname.startsWith('/api/')) {
    return CACHE_NAMES.API;
  }

  return CACHE_NAMES.DYNAMIC;
}

// Fetch with timeout
function fetchWithTimeout(request, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('Request timeout'));
    }, timeout);

    fetch(request)
      .then((response) => {
        clearTimeout(timer);
        resolve(response);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

// Create error response
function createErrorResponse(message) {
  return new Response(
    JSON.stringify({ error: message }),
    {
      status: 503,
      statusText: 'Service Unavailable',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
}

// Message handling
self.addEventListener('message', (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CLEAR_CACHE':
      event.waitUntil(clearAllCaches());
      break;

    case 'CLEAR_SPECIFIC_CACHE':
      event.waitUntil(clearSpecificCache(payload.cacheName));
      break;

    case 'CACHE_URLS':
      event.waitUntil(cacheUrls(payload.urls));
      break;

    case 'GET_CACHE_SIZE':
      event.waitUntil(getCacheSize().then(size => {
        event.ports[0].postMessage({ type: 'CACHE_SIZE', size });
      }));
      break;
  }
});

// Clear all caches
async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(cacheNames.map(name => caches.delete(name)));
  console.log('[ServiceWorker] All caches cleared');
}

// Clear specific cache
async function clearSpecificCache(cacheName) {
  await caches.delete(cacheName);
  console.log('[ServiceWorker] Cache cleared:', cacheName);
}

// Cache specific URLs
async function cacheUrls(urls) {
  const cache = await caches.open(CACHE_NAMES.DYNAMIC);
  await cache.addAll(urls);
  console.log('[ServiceWorker] URLs cached:', urls);
}

// Get cache size
async function getCacheSize() {
  if (!navigator.storage || !navigator.storage.estimate) {
    return { usage: 0, quota: 0 };
  }

  const estimate = await navigator.storage.estimate();
  return {
    usage: estimate.usage || 0,
    quota: estimate.quota || 0,
    percentage: estimate.usage && estimate.quota
      ? Math.round((estimate.usage / estimate.quota) * 100)
      : 0,
  };
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[ServiceWorker] Background sync:', event.tag);

  switch (event.tag) {
    case 'sync-metrics':
      event.waitUntil(syncMetrics());
      break;
    case 'sync-data':
      event.waitUntil(syncOfflineData());
      break;
  }
});

// Sync metrics to server
async function syncMetrics() {
  try {
    const db = await openDB();
    const metrics = await getAllMetrics(db);

    if (metrics.length > 0) {
      await fetch('/api/metrics/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metrics),
      });

      await clearMetrics(db);
      console.log('[ServiceWorker] Metrics synced:', metrics.length);
    }
  } catch (error) {
    console.error('[ServiceWorker] Failed to sync metrics:', error);
  }
}

// Sync offline data
async function syncOfflineData() {
  try {
    const db = await openDB();
    const requests = await getAllOfflineRequests(db);

    for (const request of requests) {
      try {
        await fetch(request.url, {
          method: request.method,
          headers: request.headers,
          body: request.body,
        });

        await removeOfflineRequest(db, request.id);
        console.log('[ServiceWorker] Synced offline request:', request.url);
      } catch (error) {
        console.error('[ServiceWorker] Failed to sync request:', request.url, error);
      }
    }
  } catch (error) {
    console.error('[ServiceWorker] Failed to sync offline data:', error);
  }
}

// IndexedDB helpers (simplified)
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ServiceWorkerDB', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('metrics')) {
        db.createObjectStore('metrics', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('offline')) {
        db.createObjectStore('offline', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

function getAllMetrics(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['metrics'], 'readonly');
    const store = transaction.objectStore('metrics');
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

function clearMetrics(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['metrics'], 'readwrite');
    const store = transaction.objectStore('metrics');
    const request = store.clear();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

function getAllOfflineRequests(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['offline'], 'readonly');
    const store = transaction.objectStore('offline');
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

function removeOfflineRequest(db, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['offline'], 'readwrite');
    const store = transaction.objectStore('offline');
    const request = store.delete(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}