// LLM Works Service Worker
// Provides caching and offline functionality for the evaluation platform

const CACHE_NAME = 'llm-works-v1';
const CACHE_VERSION = '1.0.0';

// Static assets to cache immediately
const STATIC_CACHE = [
  '/',
  '/index.html',
  '/favicon.ico',
  '/favicon.svg',
  '/robots.txt',
  '/sitemap.xml',
];

// API routes to cache with network-first strategy
const API_CACHE_PATTERNS = [
  /^https:\/\/api\./,
  /\/api\//,
];

// Assets to cache with cache-first strategy
const ASSET_CACHE_PATTERNS = [
  /\.(?:js|css|png|jpg|jpeg|svg|gif|webp|woff|woff2|ttf|eot)$/,
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log(`[SW] Installing service worker ${CACHE_VERSION}`);
  
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_CACHE);
    }).then(() => {
      // Skip waiting to activate immediately
      return self.skipWaiting();
    })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log(`[SW] Activating service worker ${CACHE_VERSION}`);
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log(`[SW] Deleting old cache: ${cacheName}`);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all clients immediately
      return self.clients.claim();
    })
  );
});

// Fetch event - handle requests with appropriate caching strategy
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip chrome-extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // API requests - Network first, then cache
  if (API_CACHE_PATTERNS.some(pattern => pattern.test(request.url))) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }
  
  // Static assets - Cache first, then network
  if (ASSET_CACHE_PATTERNS.some(pattern => pattern.test(request.url))) {
    event.respondWith(cacheFirstStrategy(request));
    return;
  }
  
  // Navigation requests - Network first with offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(navigationStrategy(request));
    return;
  }
  
  // Default - Network first
  event.respondWith(networkFirstStrategy(request));
});

// Network-first strategy (good for API calls and dynamic content)
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log(`[SW] Network failed for ${request.url}, trying cache`);
    
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return a fallback response for failed requests
    if (request.mode === 'navigate') {
      return new Response('Offline - Please check your connection', {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'text/plain' }
      });
    }
    
    throw error;
  }
}

// Cache-first strategy (good for static assets)
async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    // Update cache in background
    fetch(request).then(response => {
      if (response.ok) {
        caches.open(CACHE_NAME).then(cache => {
          cache.put(request, response);
        });
      }
    }).catch(() => {
      // Ignore background update failures
    });
    
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log(`[SW] Failed to fetch ${request.url}`);
    throw error;
  }
}

// Navigation strategy (for page requests)
async function navigationStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log(`[SW] Navigation failed for ${request.url}, trying cache`);
    
    // Try to serve cached version
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Try to serve cached index.html for SPA routes
    const indexResponse = await caches.match('/index.html');
    if (indexResponse) {
      return indexResponse;
    }
    
    // Fallback offline page
    return new Response(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>LLM Works - Offline</title>
          <style>
            body { font-family: system-ui, sans-serif; text-align: center; padding: 2rem; }
            .offline { color: #666; }
          </style>
        </head>
        <body>
          <h1>LLM Works</h1>
          <div class="offline">
            <h2>You're offline</h2>
            <p>Please check your internet connection and try again.</p>
            <button onclick="window.location.reload()">Retry</button>
          </div>
        </body>
      </html>
    `, {
      headers: { 'Content-Type': 'text/html' }
    });
  }
}

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});