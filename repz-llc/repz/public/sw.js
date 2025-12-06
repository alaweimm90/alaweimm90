// REPZ Service Worker - Advanced Caching and Performance
// Implements caching strategies for optimal performance and offline functionality

const CACHE_NAME = 'repz-v1.2.0';
const STATIC_CACHE = 'repz-static-v1.2.0';
const DYNAMIC_CACHE = 'repz-dynamic-v1.2.0';
const API_CACHE = 'repz-api-v1.2.0';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html', // Create this as fallback page
  // Core CSS and JS will be added dynamically
];

// API endpoints to cache
const CACHEABLE_APIS = [
  '/api/tiers',
  '/api/features',
  '/api/public-data'
];

// Cache strategies
const CACHE_STRATEGIES = {
  // Static assets: Cache first, fallback to network
  CACHE_FIRST: 'cache-first',
  // API data: Network first, fallback to cache
  NETWORK_FIRST: 'network-first',
  // User content: Network only (no cache)
  NETWORK_ONLY: 'network-only',
  // Images: Stale while revalidate
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate'
};

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');
  
  event.waitUntil(
    Promise.all([
      // Cache static assets
      caches.open(STATIC_CACHE).then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      }),
      
      // Skip waiting to activate immediately
      self.skipWaiting()
    ])
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((cacheName) => {
              return cacheName !== STATIC_CACHE && 
                     cacheName !== DYNAMIC_CACHE && 
                     cacheName !== API_CACHE;
            })
            .map((cacheName) => {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      }),
      
      // Take control of all pages
      self.clients.claim()
    ])
  );
});

// Fetch event - handle requests with caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Handle different types of requests
  if (isStaticAsset(url)) {
    event.respondWith(handleStaticAsset(request));
  } else if (isAPIRequest(url)) {
    event.respondWith(handleAPIRequest(request));
  } else if (isImageRequest(url)) {
    event.respondWith(handleImageRequest(request));
  } else if (isNavigationRequest(request)) {
    event.respondWith(handleNavigationRequest(request));
  } else {
    event.respondWith(handleDynamicRequest(request));
  }
});

// Check if request is for static assets
function isStaticAsset(url) {
  return url.pathname.includes('/assets/') ||
         url.pathname.endsWith('.js') ||
         url.pathname.endsWith('.css') ||
         url.pathname.endsWith('.woff2') ||
         url.pathname.endsWith('.woff');
}

// Check if request is for API
function isAPIRequest(url) {
  return url.pathname.startsWith('/api/') ||
         url.pathname.startsWith('/_functions/') ||
         url.hostname.includes('supabase');
}

// Check if request is for images
function isImageRequest(url) {
  return /\.(jpg|jpeg|png|gif|webp|svg|ico)$/i.test(url.pathname);
}

// Check if request is navigation (page request)
function isNavigationRequest(request) {
  return request.mode === 'navigate';
}

// Handle static assets - Cache First strategy
async function handleStaticAsset(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Fetch from network and cache
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[SW] Static asset failed:', error);
    // Return cached version if available
    return caches.match(request);
  }
}

// Handle API requests - Network First strategy
async function handleAPIRequest(request) {
  const url = new URL(request.url);
  
  // Don't cache sensitive API calls
  if (isSensitiveAPI(url)) {
    return fetch(request);
  }
  
  try {
    // Try network first
    const networkResponse = await fetch(request, {
      // Add timeout for API requests
      signal: AbortSignal.timeout(10000)
    });
    
    if (networkResponse.ok && isCacheableAPI(url)) {
      // Cache successful responses
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[SW] API request failed, trying cache:', error);
    
    // Fallback to cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline response for API failures
    return new Response(
      JSON.stringify({
        error: 'offline',
        message: 'Request failed and no cached version available'
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Handle images - Stale While Revalidate strategy
async function handleImageRequest(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  // Fetch from network in background
  const networkFetch = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  }).catch(() => null);
  
  // Return cached version immediately if available
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Wait for network if no cache
  return networkFetch || new Response('Image not available offline', { status: 404 });
}

// Handle navigation requests - Network First with offline fallback
async function handleNavigationRequest(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful page loads
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }
    
    throw new Error('Network response not ok');
  } catch (error) {
    console.log('[SW] Navigation failed, trying cache:', error);
    
    // Try cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Fallback to offline page
    return caches.match('/offline.html') || 
           new Response('You are offline', { status: 503 });
  }
}

// Handle other dynamic requests
async function handleDynamicRequest(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    return cachedResponse || new Response('Resource not available offline', { status: 404 });
  }
}

// Check if API endpoint is sensitive (shouldn't be cached)
function isSensitiveAPI(url) {
  const sensitivePaths = [
    '/auth/',
    '/user/',
    '/payment/',
    '/admin/',
    '/private/',
    'password',
    'token',
    'session'
  ];
  
  return sensitivePaths.some(path => 
    url.pathname.toLowerCase().includes(path.toLowerCase())
  );
}

// Check if API endpoint is cacheable
function isCacheableAPI(url) {
  return CACHEABLE_APIS.some(api => url.pathname.startsWith(api));
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(handleBackgroundSync());
  }
});

async function handleBackgroundSync() {
  // Handle queued offline actions when back online
  const offlineActions = await getOfflineActions();
  
  for (const action of offlineActions) {
    try {
      await processOfflineAction(action);
      await removeOfflineAction(action.id);
    } catch (error) {
      console.log('[SW] Failed to sync offline action:', error);
    }
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  if (!event.data) return;
  
  const data = event.data.json();
  const options = {
    body: data.body,
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: data.data,
    actions: data.actions || []
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  const urlToOpen = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      // Focus existing window if available
      for (const client of clientList) {
        if (client.url === urlToOpen && 'focus' in client) {
          return client.focus();
        }
      }
      
      // Open new window
      if (clients.openWindow) {
        return clients.openWindow(urlToOpen);
      }
    })
  );
});

// Periodic background sync for analytics
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'analytics-sync') {
    event.waitUntil(syncAnalytics());
  }
});

async function syncAnalytics() {
  // Sync cached analytics data when connection is available
  try {
    const cache = await caches.open('analytics-cache');
    const requests = await cache.keys();
    
    for (const request of requests) {
      const response = await cache.match(request);
      const data = await response.json();
      
      // Send to analytics service
      await fetch('/api/analytics', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json' }
      });
      
      // Remove from cache after successful sync
      await cache.delete(request);
    }
  } catch (error) {
    console.log('[SW] Analytics sync failed:', error);
  }
}

// Helper functions for offline action queue
async function getOfflineActions() {
  // In a real implementation, this would read from IndexedDB
  return [];
}

async function processOfflineAction(action) {
  // Process queued offline actions
  return fetch(action.url, action.options);
}

async function removeOfflineAction(actionId) {
  // Remove processed action from queue
  return true;
}

// Cache management - periodic cleanup
setInterval(async () => {
  const cacheNames = await caches.keys();
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    
    // Remove old entries (older than 30 days)
    const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    
    for (const request of requests) {
      const response = await cache.match(request);
      const dateHeader = response.headers.get('date');
      
      if (dateHeader && new Date(dateHeader).getTime() < thirtyDaysAgo) {
        await cache.delete(request);
      }
    }
  }
}, 24 * 60 * 60 * 1000); // Run daily