// QMLab Service Worker - Advanced PWA Capabilities
// Version 1.0.0

const CACHE_NAME = 'qmlab-v1.0.0';
const STATIC_CACHE = 'qmlab-static-v1.0.0';
const DYNAMIC_CACHE = 'qmlab-dynamic-v1.0.0';
const QUANTUM_CACHE = 'qmlab-quantum-v1.0.0';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/manifest.webmanifest',
  '/favicon.svg',
  '/favicon.ico',
  // Core CSS and JS will be added dynamically
];

// Quantum simulation results cache (for expensive computations)
const QUANTUM_SIMULATION_CACHE = 'qmlab-simulations-v1.0.0';

// Routes that should work offline
const OFFLINE_ROUTES = [
  '/',
  '/circuit-builder',
  '/bloch-sphere',
  '/training-dashboard'
];

// Analytics queue for offline events
let offlineAnalyticsQueue = [];

// Install event - cache critical assets
self.addEventListener('install', event => {
  console.log('🔧 QMLab Service Worker installing...');
  
  event.waitUntil(
    (async () => {
      try {
        // Cache static assets
        const staticCache = await caches.open(STATIC_CACHE);
        await staticCache.addAll(STATIC_ASSETS);
        
        // Cache quantum simulation results storage
        await caches.open(QUANTUM_SIMULATION_CACHE);
        
        console.log('✅ QMLab Service Worker installed successfully');
        
        // Skip waiting to activate immediately
        await self.skipWaiting();
      } catch (error) {
        console.error('❌ Service Worker installation failed:', error);
      }
    })()
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🚀 QMLab Service Worker activating...');
  
  event.waitUntil(
    (async () => {
      try {
        // Take control of all clients immediately
        await self.clients.claim();
        
        // Clean up old caches
        const cacheNames = await caches.keys();
        const cachesToDelete = cacheNames.filter(name => 
          name.startsWith('qmlab-') && 
          !name.includes('v1.0.0')
        );
        
        await Promise.all(
          cachesToDelete.map(name => caches.delete(name))
        );
        
        console.log('✅ QMLab Service Worker activated');
        
        // Notify all clients about SW update
        const clients = await self.clients.matchAll();
        clients.forEach(client => {
          client.postMessage({
            type: 'SW_ACTIVATED',
            version: '1.0.0'
          });
        });
      } catch (error) {
        console.error('❌ Service Worker activation failed:', error);
      }
    })()
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-HTTP requests
  if (!request.url.startsWith('http')) return;
  
  // Different strategies for different types of requests
  if (url.pathname.startsWith('/api/quantum-simulation')) {
    // Quantum simulations - cache first with network fallback
    event.respondWith(handleQuantumSimulation(request));
  } else if (url.pathname.match(/\.(js|css|woff2?|svg|ico)$/)) {
    // Static assets - cache first
    event.respondWith(handleStaticAsset(request));
  } else if (url.pathname.startsWith('/api/')) {
    // API calls - network first with cache fallback
    event.respondWith(handleApiRequest(request));
  } else if (OFFLINE_ROUTES.some(route => url.pathname.startsWith(route))) {
    // App routes - stale while revalidate
    event.respondWith(handleAppRoute(request));
  } else {
    // Everything else - network first
    event.respondWith(handleNetworkFirst(request));
  }
});

// Quantum simulation caching (expensive computations)
async function handleQuantumSimulation(request) {
  const cache = await caches.open(QUANTUM_SIMULATION_CACHE);
  
  try {
    // Check cache first for quantum simulations
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      console.log('🎯 Quantum simulation served from cache');
      
      // Track cache hit
      trackOfflineEvent('quantum_simulation_cache_hit', {
        url: request.url,
        method: request.method
      });
      
      return cachedResponse;
    }
    
    // Fetch from network and cache result
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const responseClone = networkResponse.clone();
      await cache.put(request, responseClone);
      
      trackOfflineEvent('quantum_simulation_cached', {
        url: request.url,
        method: request.method
      });
    }
    
    return networkResponse;
  } catch (error) {
    console.error('❌ Quantum simulation request failed:', error);
    
    // Return cached version if available
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline quantum simulation placeholder
    return new Response(JSON.stringify({
      error: 'Quantum simulation unavailable offline',
      offline: true,
      cached: false
    }), {
      headers: { 'Content-Type': 'application/json' },
      status: 503
    });
  }
}

// Static asset caching (cache first)
async function handleStaticAsset(request) {
  const cache = await caches.open(STATIC_CACHE);
  
  try {
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const responseClone = networkResponse.clone();
      await cache.put(request, responseClone);
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

// API request handling (network first)
async function handleApiRequest(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok && request.method === 'GET') {
      const responseClone = networkResponse.clone();
      await cache.put(request, responseClone);
    }
    
    return networkResponse;
  } catch (error) {
    // Return cached version for GET requests
    if (request.method === 'GET') {
      const cachedResponse = await cache.match(request);
      if (cachedResponse) {
        trackOfflineEvent('api_served_from_cache', {
          url: request.url,
          method: request.method
        });
        return cachedResponse;
      }
    }
    
    // Queue POST/PUT requests for background sync
    if (['POST', 'PUT', 'PATCH'].includes(request.method)) {
      await queueOfflineRequest(request);
    }
    
    throw error;
  }
}

// App route handling (stale while revalidate)
async function handleAppRoute(request) {
  const cache = await caches.open(STATIC_CACHE);
  
  try {
    // Serve from cache immediately
    const cachedResponse = await cache.match(request);
    
    // Fetch from network in background
    const networkResponse = fetch(request).then(response => {
      if (response.ok) {
        const responseClone = response.clone();
        cache.put(request, responseClone);
      }
      return response;
    }).catch(() => null);
    
    // Return cached version or wait for network
    return cachedResponse || await networkResponse || await cache.match('/');
  } catch (error) {
    // Return app shell
    return await cache.match('/') || new Response('QMLab offline', {
      headers: { 'Content-Type': 'text/html' }
    });
  }
}

// Network first fallback
async function handleNetworkFirst(request) {
  try {
    return await fetch(request);
  } catch (error) {
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    throw error;
  }
}

// Queue offline requests for background sync
async function queueOfflineRequest(request) {
  try {
    const requestData = {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: request.method !== 'GET' ? await request.text() : null,
      timestamp: Date.now()
    };
    
    // Store in IndexedDB for persistence
    await storeOfflineRequest(requestData);
    
    // Register background sync
    if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
      await self.registration.sync.register('offline-requests');
    }
    
    trackOfflineEvent('request_queued_offline', {
      url: request.url,
      method: request.method
    });
  } catch (error) {
    console.error('Failed to queue offline request:', error);
  }
}

// Background sync event
self.addEventListener('sync', event => {
  if (event.tag === 'offline-requests') {
    event.waitUntil(processOfflineRequests());
  } else if (event.tag === 'analytics-sync') {
    event.waitUntil(syncOfflineAnalytics());
  }
});

// Process queued offline requests
async function processOfflineRequests() {
  try {
    const requests = await getOfflineRequests();
    
    for (const requestData of requests) {
      try {
        const response = await fetch(requestData.url, {
          method: requestData.method,
          headers: requestData.headers,
          body: requestData.body
        });
        
        if (response.ok) {
          await removeOfflineRequest(requestData.timestamp);
          
          trackOfflineEvent('offline_request_synced', {
            url: requestData.url,
            method: requestData.method,
            delay: Date.now() - requestData.timestamp
          });
        }
      } catch (error) {
        console.error('Failed to sync offline request:', error);
      }
    }
  } catch (error) {
    console.error('Failed to process offline requests:', error);
  }
}

// Sync offline analytics events
async function syncOfflineAnalytics() {
  try {
    if (offlineAnalyticsQueue.length === 0) return;
    
    // Send batched analytics events
    const events = [...offlineAnalyticsQueue];
    offlineAnalyticsQueue = [];
    
    await fetch('/api/analytics/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events })
    });
    
    console.log(`📊 Synced ${events.length} offline analytics events`);
  } catch (error) {
    console.error('Failed to sync analytics:', error);
    // Re-queue events on failure
    offlineAnalyticsQueue = [...offlineAnalyticsQueue, ...events];
  }
}

// Track offline events for later sync
function trackOfflineEvent(eventName, parameters = {}) {
  offlineAnalyticsQueue.push({
    event: eventName,
    parameters: {
      ...parameters,
      offline_queued: true,
      timestamp: Date.now()
    }
  });
  
  // Try to sync immediately if online
  if (navigator.onLine) {
    self.registration.sync.register('analytics-sync');
  }
}

// Message handling for client communication
self.addEventListener('message', event => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_VERSION':
      event.ports[0].postMessage({ version: '1.0.0' });
      break;
      
    case 'CLEAR_CACHE':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    case 'CACHE_QUANTUM_RESULT':
      cacheQuantumResult(data).then(() => {
        event.ports[0].postMessage({ cached: true });
      });
      break;
  }
});

// Clear all caches
async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(
    cacheNames
      .filter(name => name.startsWith('qmlab-'))
      .map(name => caches.delete(name))
  );
}

// Cache quantum computation results
async function cacheQuantumResult(data) {
  const cache = await caches.open(QUANTUM_SIMULATION_CACHE);
  const response = new Response(JSON.stringify(data.result), {
    headers: { 'Content-Type': 'application/json' }
  });
  await cache.put(data.key, response);
}

// IndexedDB utilities for offline requests (simplified)
async function storeOfflineRequest(requestData) {
  // Implementation would use IndexedDB to store request data
  console.log('Storing offline request:', requestData);
}

async function getOfflineRequests() {
  // Implementation would retrieve from IndexedDB
  return [];
}

async function removeOfflineRequest(timestamp) {
  // Implementation would remove from IndexedDB
  console.log('Removing offline request:', timestamp);
}

console.log('🔧 QMLab Service Worker loaded and ready');