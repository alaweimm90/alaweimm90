// TalAI Progressive Web App Service Worker
// Enables offline research capabilities and advanced caching

const CACHE_VERSION = 'talai-v2.0.0';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const RESEARCH_CACHE = `${CACHE_VERSION}-research`;
const API_CACHE = `${CACHE_VERSION}-api`;

// Assets to pre-cache for offline functionality
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/css/app.css',
  '/js/app.js',
  '/js/wasm/validator.wasm',
  '/js/wasm/processor.wasm',
  '/icons/icon-72x72.png',
  '/icons/icon-96x96.png',
  '/icons/icon-128x128.png',
  '/icons/icon-144x144.png',
  '/icons/icon-152x152.png',
  '/icons/icon-192x192.png',
  '/icons/icon-384x384.png',
  '/icons/icon-512x512.png',
  '/offline.html'
];

// API endpoints that can be cached
const CACHEABLE_API_PATTERNS = [
  /\/api\/research\/papers\//,
  /\/api\/research\/citations\//,
  /\/api\/abstracts\//,
  /\/api\/literature\//,
  /\/api\/prompts\/marketplace\//
];

// Install event - Pre-cache static assets
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');

  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('[ServiceWorker] Pre-caching static assets');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => {
      console.log('[ServiceWorker] Install complete');
      return self.skipWaiting(); // Activate immediately
    })
  );
});

// Activate event - Clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((cacheName) => {
            return cacheName.startsWith('talai-') &&
                   !cacheName.startsWith(CACHE_VERSION);
          })
          .map((cacheName) => {
            console.log('[ServiceWorker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          })
      );
    }).then(() => {
      console.log('[ServiceWorker] Activation complete');
      return self.clients.claim(); // Take control immediately
    })
  );
});

// Fetch event - Implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    event.respondWith(fetch(request));
    return;
  }

  // Determine caching strategy based on request type
  if (url.origin === location.origin) {
    // Same-origin requests
    if (url.pathname.startsWith('/api/')) {
      // API requests - Network first, cache fallback
      event.respondWith(networkFirstStrategy(request));
    } else if (url.pathname.match(/\.(png|jpg|jpeg|svg|gif|webp)$/)) {
      // Images - Cache first, network fallback
      event.respondWith(cacheFirstStrategy(request));
    } else if (url.pathname.match(/\.(js|css|wasm)$/)) {
      // Scripts and styles - Stale while revalidate
      event.respondWith(staleWhileRevalidateStrategy(request));
    } else {
      // HTML and other - Network first
      event.respondWith(networkFirstStrategy(request));
    }
  } else {
    // Cross-origin requests (CDN, external APIs)
    if (url.hostname.includes('anthropic.com') || url.hostname.includes('openai.com')) {
      // AI API requests - Network only with offline queue
      event.respondWith(networkOnlyWithQueueStrategy(request));
    } else {
      // Other external resources - Network first
      event.respondWith(networkFirstStrategy(request));
    }
  }
});

// Cache-first strategy
async function cacheFirstStrategy(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);

  if (cached) {
    console.log('[ServiceWorker] Cache hit:', request.url);
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[ServiceWorker] Fetch failed:', error);
    return caches.match('/offline.html');
  }
}

// Network-first strategy
async function networkFirstStrategy(request) {
  try {
    const response = await fetch(request);

    if (response.ok && shouldCache(request)) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    console.log('[ServiceWorker] Network failed, checking cache:', request.url);
    const cached = await caches.match(request);

    if (cached) {
      return cached;
    }

    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/offline.html');
    }

    // Return error response for API requests
    return new Response(JSON.stringify({
      error: 'Offline',
      message: 'This feature requires an internet connection'
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Stale-while-revalidate strategy
async function staleWhileRevalidateStrategy(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  }).catch(() => cached);

  return cached || fetchPromise;
}

// Network-only with offline queue
async function networkOnlyWithQueueStrategy(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    console.log('[ServiceWorker] Queueing request for later:', request.url);

    // Store request in IndexedDB for later retry
    await queueRequest(request);

    return new Response(JSON.stringify({
      queued: true,
      message: 'Request queued for processing when online'
    }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Background sync for queued requests
self.addEventListener('sync', async (event) => {
  if (event.tag === 'sync-queue') {
    console.log('[ServiceWorker] Processing queued requests');
    event.waitUntil(processQueuedRequests());
  }
});

// Process queued requests when back online
async function processQueuedRequests() {
  const db = await openDB();
  const tx = db.transaction('requests', 'readwrite');
  const store = tx.objectStore('requests');
  const requests = await store.getAll();

  for (const queuedRequest of requests) {
    try {
      const request = new Request(queuedRequest.url, {
        method: queuedRequest.method,
        headers: queuedRequest.headers,
        body: queuedRequest.body
      });

      const response = await fetch(request);

      if (response.ok) {
        // Remove from queue on success
        await store.delete(queuedRequest.id);

        // Notify client of successful sync
        const clients = await self.clients.matchAll();
        clients.forEach(client => {
          client.postMessage({
            type: 'sync-complete',
            requestId: queuedRequest.id,
            response: await response.json()
          });
        });
      }
    } catch (error) {
      console.error('[ServiceWorker] Failed to sync request:', error);
    }
  }
}

// IndexedDB helpers
async function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TalAIOfflineQueue', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('requests')) {
        db.createObjectStore('requests', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

async function queueRequest(request) {
  const db = await openDB();
  const tx = db.transaction('requests', 'readwrite');
  const store = tx.objectStore('requests');

  const body = await request.text();
  const headers = {};
  request.headers.forEach((value, key) => {
    headers[key] = value;
  });

  await store.add({
    url: request.url,
    method: request.method,
    headers,
    body,
    timestamp: Date.now()
  });

  // Register for background sync
  if ('sync' in self.registration) {
    await self.registration.sync.register('sync-queue');
  }
}

// Helper function to determine if a request should be cached
function shouldCache(request) {
  const url = new URL(request.url);

  // Check if it matches cacheable patterns
  return CACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname));
}

// Push notifications for research updates
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};

  const options = {
    title: data.title || 'TalAI Research Update',
    body: data.body || 'New research results available',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: data.id,
      url: data.url || '/'
    },
    actions: [
      {
        action: 'view',
        title: 'View Results'
      },
      {
        action: 'dismiss',
        title: 'Dismiss'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(options.title, options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url)
    );
  }
});

// Periodic background sync for research updates
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'research-update') {
    event.waitUntil(checkForResearchUpdates());
  }
});

async function checkForResearchUpdates() {
  try {
    const response = await fetch('/api/research/updates');
    const updates = await response.json();

    if (updates.hasNew) {
      // Cache new research data
      const cache = await caches.open(RESEARCH_CACHE);
      for (const update of updates.items) {
        const updateResponse = await fetch(update.url);
        await cache.put(update.url, updateResponse);
      }

      // Notify user
      self.registration.showNotification('New Research Available', {
        body: `${updates.count} new research items available`,
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png'
      });
    }
  } catch (error) {
    console.error('[ServiceWorker] Failed to check for updates:', error);
  }
}

// Message handling from clients
self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  } else if (event.data.type === 'CACHE_RESEARCH') {
    cacheResearchData(event.data.urls);
  } else if (event.data.type === 'CLEAR_CACHE') {
    clearCache(event.data.cacheName);
  }
});

async function cacheResearchData(urls) {
  const cache = await caches.open(RESEARCH_CACHE);
  const promises = urls.map(url => {
    return fetch(url).then(response => {
      if (response.ok) {
        return cache.put(url, response);
      }
    }).catch(error => {
      console.error(`[ServiceWorker] Failed to cache ${url}:`, error);
    });
  });

  await Promise.all(promises);
}

async function clearCache(cacheName) {
  if (cacheName) {
    await caches.delete(cacheName);
  } else {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames
        .filter(name => name.startsWith('talai-'))
        .map(name => caches.delete(name))
    );
  }
}

console.log('[ServiceWorker] Loaded successfully');