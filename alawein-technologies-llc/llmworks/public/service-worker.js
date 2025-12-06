/**
 * Service Worker for LLM Works
 * Provides offline support, caching, and background sync
 */

const CACHE_NAME = 'llm-works-v1';
const STATIC_CACHE_NAME = 'llm-works-static-v1';
const DYNAMIC_CACHE_NAME = 'llm-works-dynamic-v1';
const API_CACHE_NAME = 'llm-works-api-v1';

// Resources to cache on install
const STATIC_RESOURCES = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.ico',
  '/favicon.svg',
  '/apple-touch-icon.png'
];

// API endpoints to cache
const CACHEABLE_API_PATTERNS = [
  /\/api\/models/,
  /\/api\/benchmarks/,
  /\/api\/leaderboard/
];

// Cache strategies
const CACHE_STRATEGIES = {
  CACHE_FIRST: 'cache-first',
  NETWORK_FIRST: 'network-first',
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate',
  NETWORK_ONLY: 'network-only',
  CACHE_ONLY: 'cache-only'
};

// Install event - cache static resources
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install');
  
  event.waitUntil(
    caches.open(STATIC_CACHE_NAME)
      .then((cache) => {
        console.log('[ServiceWorker] Caching static resources');
        return cache.addAll(STATIC_RESOURCES);
      })
      .then(() => {
        console.log('[ServiceWorker] Skip waiting');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((cacheName) => {
            return cacheName.startsWith('llm-works-') &&
              cacheName !== CACHE_NAME &&
              cacheName !== STATIC_CACHE_NAME &&
              cacheName !== DYNAMIC_CACHE_NAME &&
              cacheName !== API_CACHE_NAME;
          })
          .map((cacheName) => {
            console.log('[ServiceWorker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          })
      );
    }).then(() => {
      console.log('[ServiceWorker] Claiming clients');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http(s) requests
  if (!request.url.startsWith('http')) {
    return;
  }

  // Determine cache strategy based on request type
  let strategy = CACHE_STRATEGIES.NETWORK_FIRST;

  // Static resources - cache first
  if (isStaticResource(url)) {
    strategy = CACHE_STRATEGIES.CACHE_FIRST;
  }
  // API requests - stale while revalidate
  else if (isApiRequest(url)) {
    strategy = CACHE_STRATEGIES.STALE_WHILE_REVALIDATE;
  }
  // Dynamic content - network first
  else {
    strategy = CACHE_STRATEGIES.NETWORK_FIRST;
  }

  event.respondWith(
    handleFetch(request, strategy)
  );
});

// Handle fetch based on strategy
async function handleFetch(request, strategy) {
  switch (strategy) {
    case CACHE_STRATEGIES.CACHE_FIRST:
      return cacheFirst(request);
    
    case CACHE_STRATEGIES.NETWORK_FIRST:
      return networkFirst(request);
    
    case CACHE_STRATEGIES.STALE_WHILE_REVALIDATE:
      return staleWhileRevalidate(request);
    
    case CACHE_STRATEGIES.NETWORK_ONLY:
      return fetch(request);
    
    case CACHE_STRATEGIES.CACHE_ONLY:
      return caches.match(request) || new Response('Not found', { status: 404 });
    
    default:
      return networkFirst(request);
  }
}

// Cache first strategy
async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('[ServiceWorker] Fetch failed:', error);
    return new Response('Offline', { status: 503 });
  }
}

// Network first strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response('Offline', { status: 503 });
  }
}

// Stale while revalidate strategy
async function staleWhileRevalidate(request) {
  const cachedResponse = await caches.match(request);
  
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse.ok) {
      const cache = caches.open(API_CACHE_NAME);
      cache.then((c) => c.put(request, networkResponse.clone()));
    }
    return networkResponse;
  }).catch(() => {
    // Silently fail - we'll use cached version
  });

  return cachedResponse || fetchPromise;
}

// Check if URL is a static resource
function isStaticResource(url) {
  const staticExtensions = [
    '.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.woff', '.woff2', '.ttf', '.eot', '.ico'
  ];
  
  return staticExtensions.some(ext => url.pathname.endsWith(ext));
}

// Check if URL is an API request
function isApiRequest(url) {
  return CACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname));
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[ServiceWorker] Background sync:', event.tag);
  
  if (event.tag === 'sync-evaluations') {
    event.waitUntil(syncEvaluations());
  }
});

// Sync offline evaluations
async function syncEvaluations() {
  try {
    // Get pending evaluations from IndexedDB
    const pendingEvaluations = await getPendingEvaluations();
    
    if (pendingEvaluations.length === 0) {
      return;
    }
    
    // Send each evaluation to the server
    for (const evaluation of pendingEvaluations) {
      try {
        const response = await fetch('/api/evaluations', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(evaluation)
        });
        
        if (response.ok) {
          await removePendingEvaluation(evaluation.id);
        }
      } catch (error) {
        console.error('[ServiceWorker] Failed to sync evaluation:', error);
      }
    }
  } catch (error) {
    console.error('[ServiceWorker] Sync failed:', error);
  }
}

// Placeholder functions for IndexedDB operations
async function getPendingEvaluations() {
  // Implementation would interact with IndexedDB
  return [];
}

async function removePendingEvaluation(id) {
  // Implementation would interact with IndexedDB
  return true;
}

// Push notifications
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New update available',
    icon: '/favicon.svg',
    badge: '/favicon.svg',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Update',
        icon: '/favicon.svg'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/favicon.svg'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('LLM Works', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handler for client communication
self.addEventListener('message', (event) => {
  console.log('[ServiceWorker] Message received:', event.data);
  
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            return caches.delete(cacheName);
          })
        );
      }).then(() => {
        event.ports[0].postMessage({ success: true });
      })
    );
  }
});