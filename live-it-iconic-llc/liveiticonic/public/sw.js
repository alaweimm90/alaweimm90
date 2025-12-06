// Service Worker for Live It Iconic E-Commerce
const CACHE_NAME = 'liveiticonic-v1';
const RUNTIME_CACHE = 'liveiticonic-runtime';

// Assets to cache on install
const PRECACHE_ASSETS = [
  '/',
  '/shop',
  '/index.html',
  '/favicon.ico',
  '/icon-192.png',
  '/icon-512.png',
];

// Install event - cache assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(PRECACHE_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(cacheName => {
            return cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE;
          })
          .map(cacheName => {
            return caches.delete(cacheName);
          })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip API calls
  if (event.request.url.includes('/api/')) {
    return;
  }

  // Cache images and static assets
  if (
    event.request.destination === 'image' ||
    event.request.url.includes('/assets/') ||
    event.request.url.includes('/mockups/')
  ) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }

        return fetch(event.request).then(response => {
          // Don't cache if not ok
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }

          const responseToCache = response.clone();
          caches.open(RUNTIME_CACHE).then(cache => {
            cache.put(event.request, responseToCache);
          });

          return response;
        });
      })
    );
    return;
  }

  // For HTML pages, try network first, fallback to cache
  if (event.request.destination === 'document') {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          const responseToCache = response.clone();
          caches.open(RUNTIME_CACHE).then(cache => {
            cache.put(event.request, responseToCache);
          });
          return response;
        })
        .catch(() => {
          return caches.match(event.request);
        })
    );
    return;
  }
});
