// Service Worker for SimCore Physics Platform
// Provides offline functionality, caching, and background sync

const CACHE_NAME = 'simcore-v1.0.0';
const OFFLINE_CACHE = 'simcore-offline-v1';
const DYNAMIC_CACHE = 'simcore-dynamic-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/static/js/bundle.js',
  '/static/css/main.css',
  // Add your built assets here
];

// Physics modules to cache for offline use
const PHYSICS_MODULES = [
  '/tdse-solver',
  '/llg-dynamics', 
  '/quantum-tunneling',
  '/bloch-sphere',
  '/graphene-band-structure',
  '/crystal-visualizer',
  '/mos2-valley-physics',
  '/phonon-band-structure',
  '/quantum-field-theory',
  '/laplace-eigenmodes',
  '/pinn-schrodinger',
  '/bz-folding'
];

// Network-first strategy for API calls
const API_ROUTES = [
  '/api/',
  '/data/',
  '/calculations/'
];

// Cache-first strategy for static assets
const CACHE_FIRST_ROUTES = [
  '/static/',
  '/assets/',
  '/icons/',
  '/images/'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing SimCore Service Worker');
  
  event.waitUntil(
    Promise.all([
      // Cache static assets
      caches.open(CACHE_NAME).then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      }),
      
      // Cache physics modules
      caches.open(OFFLINE_CACHE).then((cache) => {
        console.log('[SW] Caching physics modules');
        return Promise.allSettled(
          PHYSICS_MODULES.map(url => 
            cache.add(url).catch(err => 
              console.warn(`[SW] Failed to cache ${url}:`, err)
            )
          )
        );
      })
    ]).then(() => {
      console.log('[SW] Installation complete');
      return self.skipWaiting();
    })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating SimCore Service Worker');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter(cacheName => 
              cacheName.startsWith('simcore-') && 
              cacheName !== CACHE_NAME &&
              cacheName !== OFFLINE_CACHE &&
              cacheName !== DYNAMIC_CACHE
            )
            .map(cacheName => {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      }),
      
      // Take control of all clients
      self.clients.claim()
    ]).then(() => {
      console.log('[SW] Activation complete');
    })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip cross-origin requests (except for known APIs)
  if (url.origin !== location.origin && !isAllowedOrigin(url.origin)) {
    return;
  }
  
  event.respondWith(handleFetch(request));
});

// Handle different types of requests with appropriate strategies
async function handleFetch(request) {
  const url = new URL(request.url);
  const pathname = url.pathname;
  
  try {
    // API routes - Network first with cache fallback
    if (API_ROUTES.some(route => pathname.startsWith(route))) {
      return await networkFirstStrategy(request);
    }
    
    // Static assets - Cache first
    if (CACHE_FIRST_ROUTES.some(route => pathname.startsWith(route))) {
      return await cacheFirstStrategy(request);
    }
    
    // Physics modules - Stale while revalidate
    if (PHYSICS_MODULES.some(module => pathname.startsWith(module))) {
      return await staleWhileRevalidateStrategy(request);
    }
    
    // HTML pages - Network first with offline fallback
    if (request.headers.get('accept')?.includes('text/html')) {
      return await htmlNetworkFirstStrategy(request);
    }
    
    // Default - Network first
    return await networkFirstStrategy(request);
    
  } catch (error) {
    console.error('[SW] Fetch error:', error);
    return await offlineResponse(request);
  }
}

// Cache strategies
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  const networkResponse = await fetch(request);
  
  if (networkResponse.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, networkResponse.clone());
  }
  
  return networkResponse;
}

async function staleWhileRevalidateStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  const networkPromise = fetch(request).then(async (networkResponse) => {
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  }).catch(() => {
    // Network failed, return cached version if available
    return cachedResponse;
  });
  
  return cachedResponse || await networkPromise;
}

async function htmlNetworkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page
    const offlinePage = await caches.match('/');
    if (offlinePage) {
      return offlinePage;
    }
    
    throw error;
  }
}

// Offline response for failed requests
async function offlineResponse(request) {
  if (request.headers.get('accept')?.includes('text/html')) {
    const offlinePage = await caches.match('/');
    if (offlinePage) {
      return offlinePage;
    }
  }
  
  return new Response('Offline - Content not available', {
    status: 503,
    statusText: 'Service Unavailable',
    headers: {
      'Content-Type': 'text/plain'
    }
  });
}

// Check if origin is allowed for cross-origin caching
function isAllowedOrigin(origin) {
  const allowedOrigins = [
    'https://cdnjs.cloudflare.com',
    'https://unpkg.com',
    'https://cdn.jsdelivr.net'
  ];
  
  return allowedOrigins.includes(origin);
}

// Background sync for when connection is restored
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync triggered:', event.tag);
  
  if (event.tag === 'background-sync-physics-data') {
    event.waitUntil(syncPhysicsData());
  }
});

async function syncPhysicsData() {
  try {
    // Sync any pending physics calculations or data
    console.log('[SW] Syncing physics data...');
    
    // Get pending data from IndexedDB
    const pendingData = await getPendingData();
    
    for (const data of pendingData) {
      try {
        await uploadPhysicsData(data);
        await markDataAsSynced(data.id);
      } catch (error) {
        console.error('[SW] Failed to sync data:', error);
      }
    }
    
    console.log('[SW] Physics data sync complete');
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// IndexedDB helpers for offline data storage
async function getPendingData() {
  // This would integrate with your IndexedDB implementation
  return [];
}

async function uploadPhysicsData(data) {
  // Upload simulation results or user data
  return fetch('/api/physics-data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
}

async function markDataAsSynced(id) {
  // Mark data as synced in IndexedDB
  console.log('[SW] Marked data as synced:', id);
}

// Push notifications for physics simulation completion
self.addEventListener('push', (event) => {
  if (!event.data) return;
  
  const data = event.data.json();
  
  const options = {
    body: data.body || 'Your physics simulation has completed',
    icon: '/icon-192.png',
    badge: '/badge-72.png',
    tag: 'physics-simulation',
    actions: [
      {
        action: 'view',
        title: 'View Results',
        icon: '/action-view.png'
      },
      {
        action: 'download',
        title: 'Download Data',
        icon: '/action-download.png'
      }
    ],
    data: {
      url: data.url || '/',
      simulationType: data.simulationType
    }
  };
  
  event.waitUntil(
    self.registration.showNotification(
      data.title || 'SimCore',
      options
    )
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  const { action, data } = event;
  
  let url = data?.url || '/';
  
  if (action === 'view') {
    url = data?.url || '/';
  } else if (action === 'download') {
    url = `/download/${data?.simulationType || 'results'}`;
  }
  
  event.waitUntil(
    clients.openWindow(url)
  );
});

// Message handling for communication with main thread
self.addEventListener('message', (event) => {
  const { type, payload } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_VERSION':
      event.ports[0].postMessage({ version: CACHE_NAME });
      break;
      
    case 'CLEAR_CACHE':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    case 'CACHE_PHYSICS_DATA':
      cachePhysicsData(payload).then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    default:
      console.warn('[SW] Unknown message type:', type);
  }
});

async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(
    cacheNames
      .filter(name => name.startsWith('simcore-'))
      .map(name => caches.delete(name))
  );
}

async function cachePhysicsData(data) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const response = new Response(JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' }
  });
  await cache.put(`/cached-data/${data.id}`, response);
}

console.log('[SW] SimCore Service Worker loaded');