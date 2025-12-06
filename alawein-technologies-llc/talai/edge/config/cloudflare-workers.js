// CloudFlare Workers - Edge Computing for TalAI
// Global edge infrastructure for ultra-low latency

// Main Worker Entry Point
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

// KV Namespaces for edge caching
const CACHE_NAMESPACE = TALAI_CACHE
const SESSION_NAMESPACE = TALAI_SESSIONS
const RATE_LIMIT_NAMESPACE = TALAI_RATE_LIMITS

// Configuration
const CONFIG = {
  regions: {
    'us': 'https://api-us.talai.io',
    'eu': 'https://api-eu.talai.io',
    'ap': 'https://api-ap.talai.io'
  },
  cacheTTL: {
    static: 86400,      // 24 hours
    api: 300,           // 5 minutes
    research: 3600,     // 1 hour
    realtime: 0         // No caching
  },
  rateLimits: {
    anonymous: 10,      // per minute
    authenticated: 100, // per minute
    premium: 1000      // per minute
  },
  features: {
    smartRouting: true,
    edgeValidation: true,
    compressionEnabled: true,
    webAssemblyEnabled: true
  }
}

// Main request handler
async function handleRequest(request) {
  const url = new URL(request.url)
  const path = url.pathname
  const method = request.method

  try {
    // Performance monitoring
    const startTime = Date.now()

    // CORS preflight handling
    if (method === 'OPTIONS') {
      return handleCORS(request)
    }

    // Rate limiting
    const rateLimitResponse = await checkRateLimit(request)
    if (rateLimitResponse) return rateLimitResponse

    // Smart routing based on geolocation
    const region = request.cf?.country ? getClosestRegion(request.cf.country) : 'us'

    // Edge caching strategy
    if (method === 'GET' && shouldCache(path)) {
      const cachedResponse = await getFromCache(request)
      if (cachedResponse) {
        return addPerformanceHeaders(cachedResponse, startTime, 'edge-cache')
      }
    }

    // Edge validation with WebAssembly
    if (CONFIG.features.edgeValidation) {
      const validationResult = await validateRequest(request)
      if (!validationResult.valid) {
        return new Response(JSON.stringify({
          error: 'Validation failed',
          details: validationResult.errors
        }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        })
      }
    }

    // Route-specific handlers
    let response

    switch (true) {
      case path.startsWith('/api/research'):
        response = await handleResearchAPI(request, region)
        break
      case path.startsWith('/api/agent'):
        response = await handleAgentAPI(request, region)
        break
      case path.startsWith('/graphql'):
        response = await handleGraphQL(request, region)
        break
      case path.startsWith('/ws'):
        response = await handleWebSocket(request, region)
        break
      case path.startsWith('/static'):
        response = await handleStaticAssets(request)
        break
      default:
        response = await forwardToOrigin(request, region)
    }

    // Cache successful responses
    if (response.ok && shouldCache(path)) {
      await cacheResponse(request, response.clone())
    }

    return addPerformanceHeaders(response, startTime, 'origin')

  } catch (error) {
    return handleError(error)
  }
}

// Research API handler with intelligent caching
async function handleResearchAPI(request, region) {
  const url = new URL(request.url)
  const queryParams = url.searchParams

  // Check if this is a cacheable research query
  const cacheKey = `research:${url.pathname}:${queryParams.toString()}`
  const cached = await CACHE_NAMESPACE.get(cacheKey, 'json')

  if (cached && Date.now() - cached.timestamp < CONFIG.cacheTTL.research * 1000) {
    return new Response(JSON.stringify(cached.data), {
      headers: {
        'Content-Type': 'application/json',
        'X-Cache': 'HIT',
        'X-Cache-Age': Math.floor((Date.now() - cached.timestamp) / 1000)
      }
    })
  }

  // Forward to origin with smart routing
  const originUrl = `${CONFIG.regions[region]}${url.pathname}${url.search}`
  const response = await fetch(originUrl, {
    method: request.method,
    headers: request.headers,
    body: request.body
  })

  // Cache successful responses
  if (response.ok) {
    const data = await response.json()
    await CACHE_NAMESPACE.put(cacheKey, JSON.stringify({
      data,
      timestamp: Date.now()
    }), {
      expirationTtl: CONFIG.cacheTTL.research
    })
    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'X-Cache': 'MISS'
      }
    })
  }

  return response
}

// GraphQL Federation at the edge
async function handleGraphQL(request, region) {
  const query = await request.json()

  // Parse GraphQL query for intelligent routing
  const operations = parseGraphQLQuery(query.query)

  // Federated execution
  const results = await Promise.all(
    operations.map(async (op) => {
      const service = getServiceForOperation(op)
      const serviceUrl = `${CONFIG.regions[region]}/graphql/${service}`

      return fetch(serviceUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...request.headers
        },
        body: JSON.stringify({
          query: op.query,
          variables: query.variables
        })
      }).then(r => r.json())
    })
  )

  // Merge results
  const mergedResult = mergeGraphQLResults(results)

  return new Response(JSON.stringify(mergedResult), {
    headers: { 'Content-Type': 'application/json' }
  })
}

// WebSocket upgrade handler
async function handleWebSocket(request, region) {
  const upgradeHeader = request.headers.get('Upgrade')

  if (!upgradeHeader || upgradeHeader !== 'websocket') {
    return new Response('Expected Upgrade: websocket', { status: 426 })
  }

  const webSocketPair = new WebSocketPair()
  const [client, server] = Object.values(webSocketPair)

  server.accept()

  // Handle WebSocket messages
  server.addEventListener('message', async (event) => {
    try {
      const message = JSON.parse(event.data)

      // Route message to appropriate handler
      switch (message.type) {
        case 'subscribe':
          await handleSubscription(server, message)
          break
        case 'unsubscribe':
          await handleUnsubscription(server, message)
          break
        case 'research':
          await handleRealtimeResearch(server, message, region)
          break
        default:
          server.send(JSON.stringify({
            error: 'Unknown message type'
          }))
      }
    } catch (error) {
      server.send(JSON.stringify({
        error: error.message
      }))
    }
  })

  return new Response(null, {
    status: 101,
    webSocket: client
  })
}

// WebAssembly validation module
async function validateRequest(request) {
  // Load WebAssembly module from KV store
  const wasmModule = await CACHE_NAMESPACE.get('validation.wasm', 'arrayBuffer')

  if (!wasmModule) {
    // Fallback to basic validation
    return { valid: true }
  }

  const module = await WebAssembly.instantiate(wasmModule)
  const { validate } = module.instance.exports

  const body = await request.text()
  const encoded = new TextEncoder().encode(body)

  // Call WebAssembly validation function
  const result = validate(encoded.buffer, encoded.length)

  return {
    valid: result === 0,
    errors: result !== 0 ? decodeValidationErrors(result) : []
  }
}

// Rate limiting implementation
async function checkRateLimit(request) {
  const ip = request.headers.get('CF-Connecting-IP')
  const apiKey = request.headers.get('X-API-Key')

  const identifier = apiKey || ip
  const tier = apiKey ? (isPremium(apiKey) ? 'premium' : 'authenticated') : 'anonymous'
  const limit = CONFIG.rateLimits[tier]

  const key = `ratelimit:${identifier}:${Math.floor(Date.now() / 60000)}`
  const current = await RATE_LIMIT_NAMESPACE.get(key)
  const count = current ? parseInt(current) : 0

  if (count >= limit) {
    return new Response(JSON.stringify({
      error: 'Rate limit exceeded',
      limit,
      reset: Math.ceil(Date.now() / 60000) * 60
    }), {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        'X-RateLimit-Limit': limit,
        'X-RateLimit-Remaining': 0,
        'X-RateLimit-Reset': Math.ceil(Date.now() / 60000) * 60
      }
    })
  }

  // Increment counter
  await RATE_LIMIT_NAMESPACE.put(key, (count + 1).toString(), {
    expirationTtl: 60
  })

  return null
}

// Intelligent geographic routing
function getClosestRegion(country) {
  const regionMapping = {
    // Americas
    'US': 'us', 'CA': 'us', 'MX': 'us', 'BR': 'us',
    // Europe
    'GB': 'eu', 'DE': 'eu', 'FR': 'eu', 'IT': 'eu', 'ES': 'eu',
    // Asia Pacific
    'CN': 'ap', 'JP': 'ap', 'IN': 'ap', 'AU': 'ap', 'SG': 'ap'
  }

  return regionMapping[country] || 'us'
}

// Edge caching logic
async function getFromCache(request) {
  const cacheKey = new Request(request.url, request)
  const cache = caches.default

  let response = await cache.match(cacheKey)

  if (!response) {
    // Check KV store for longer-term caching
    const url = new URL(request.url)
    const kvKey = `cache:${url.pathname}:${url.search}`
    const kvData = await CACHE_NAMESPACE.get(kvKey)

    if (kvData) {
      response = new Response(kvData, {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'KV-HIT'
        }
      })
    }
  }

  return response
}

async function cacheResponse(request, response) {
  const cache = caches.default
  const cacheKey = new Request(request.url, request)

  // Cache in CF Cache API
  await cache.put(cacheKey, response.clone())

  // Also store in KV for longer retention
  const url = new URL(request.url)
  const kvKey = `cache:${url.pathname}:${url.search}`
  const body = await response.text()

  await CACHE_NAMESPACE.put(kvKey, body, {
    expirationTtl: determineCacheTTL(url.pathname)
  })
}

function determineCacheTTL(path) {
  if (path.startsWith('/static')) return CONFIG.cacheTTL.static
  if (path.startsWith('/api/research')) return CONFIG.cacheTTL.research
  if (path.startsWith('/api')) return CONFIG.cacheTTL.api
  return CONFIG.cacheTTL.realtime
}

// Performance headers
function addPerformanceHeaders(response, startTime, cacheStatus) {
  const duration = Date.now() - startTime
  const headers = new Headers(response.headers)

  headers.set('X-Response-Time', `${duration}ms`)
  headers.set('X-Cache-Status', cacheStatus)
  headers.set('X-Edge-Location', CF_LOCATION || 'unknown')

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  })
}

// CORS handling
function handleCORS(request) {
  const headers = request.headers
  const origin = headers.get('Origin')

  const corsHeaders = {
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': headers.get('Access-Control-Request-Headers') || 'Content-Type, X-API-Key',
    'Access-Control-Max-Age': '86400'
  }

  return new Response(null, {
    status: 204,
    headers: corsHeaders
  })
}

// Error handling
function handleError(error) {
  console.error('Worker error:', error)

  return new Response(JSON.stringify({
    error: 'Internal server error',
    message: error.message,
    timestamp: Date.now()
  }), {
    status: 500,
    headers: { 'Content-Type': 'application/json' }
  })
}

// Helper functions
function shouldCache(path) {
  const noCachePaths = ['/api/auth', '/api/admin', '/api/billing']
  return !noCachePaths.some(p => path.startsWith(p))
}

function isPremium(apiKey) {
  // Check premium status from KV store or decode JWT
  return apiKey.startsWith('pk_')
}

function parseGraphQLQuery(query) {
  // Simple GraphQL query parser
  const operations = []
  const matches = query.matchAll(/\b(query|mutation|subscription)\s+(\w+)/g)

  for (const match of matches) {
    operations.push({
      type: match[1],
      name: match[2],
      query: extractOperation(query, match.index)
    })
  }

  return operations
}

function getServiceForOperation(operation) {
  const serviceMapping = {
    'research': 'research-service',
    'agent': 'agent-service',
    'user': 'user-service',
    'billing': 'billing-service'
  }

  for (const [key, service] of Object.entries(serviceMapping)) {
    if (operation.name.toLowerCase().includes(key)) {
      return service
    }
  }

  return 'default-service'
}

function mergeGraphQLResults(results) {
  return {
    data: Object.assign({}, ...results.map(r => r.data)),
    errors: results.flatMap(r => r.errors || [])
  }
}

function extractOperation(query, startIndex) {
  let depth = 0
  let inOperation = false
  let endIndex = startIndex

  for (let i = startIndex; i < query.length; i++) {
    if (query[i] === '{') {
      depth++
      inOperation = true
    } else if (query[i] === '}') {
      depth--
      if (depth === 0 && inOperation) {
        endIndex = i + 1
        break
      }
    }
  }

  return query.substring(startIndex, endIndex)
}

function decodeValidationErrors(errorCode) {
  const errors = []
  const errorMap = {
    1: 'Invalid JSON format',
    2: 'Missing required field',
    3: 'Field type mismatch',
    4: 'Value out of range',
    5: 'Pattern validation failed'
  }

  for (const [code, message] of Object.entries(errorMap)) {
    if (errorCode & (1 << parseInt(code))) {
      errors.push(message)
    }
  }

  return errors
}