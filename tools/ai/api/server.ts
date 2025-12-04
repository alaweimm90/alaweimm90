#!/usr/bin/env npx tsx
/**
 * REST API Server for AI Tools
 * Exposes AI tools via HTTP REST endpoints with rate limiting
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { URL } from 'url';

import {
  routes,
  RouteHandler,
  parseBody,
  error,
  jsonResponse,
  incrementRequestCount,
} from './routes.js';

const PORT = parseInt(process.env.AI_API_PORT || '3200', 10);
const RATE_LIMIT_WINDOW_MS = 60000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = parseInt(process.env.RATE_LIMIT_MAX || '100', 10);

// Rate limiter store
interface RateLimitEntry {
  count: number;
  resetAt: number;
}
const rateLimitStore = new Map<string, RateLimitEntry>();

function getClientIP(req: IncomingMessage): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) return (Array.isArray(forwarded) ? forwarded[0] : forwarded).split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}

function checkRateLimit(clientIP: string): {
  allowed: boolean;
  remaining: number;
  resetAt: number;
} {
  const now = Date.now();
  let entry = rateLimitStore.get(clientIP);

  if (!entry || entry.resetAt < now) {
    entry = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    rateLimitStore.set(clientIP, entry);
  }

  entry.count++;
  const remaining = Math.max(0, RATE_LIMIT_MAX_REQUESTS - entry.count);
  return { allowed: entry.count <= RATE_LIMIT_MAX_REQUESTS, remaining, resetAt: entry.resetAt };
}

// Clean up expired entries periodically
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitStore.entries()) {
    if (entry.resetAt < now) rateLimitStore.delete(ip);
  }
}, RATE_LIMIT_WINDOW_MS);

function findRoute(method: string, pathname: string): RouteHandler | undefined {
  return routes.find((r) => r.method === method && r.path === pathname);
}

// ============================================================================
// Server
// ============================================================================

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Rate limiting
  const clientIP = getClientIP(req);
  const rateLimit = checkRateLimit(clientIP);
  res.setHeader('X-RateLimit-Limit', RATE_LIMIT_MAX_REQUESTS.toString());
  res.setHeader('X-RateLimit-Remaining', rateLimit.remaining.toString());
  res.setHeader('X-RateLimit-Reset', Math.ceil(rateLimit.resetAt / 1000).toString());

  if (!rateLimit.allowed) {
    jsonResponse(res, 429, {
      success: false,
      error: 'Too many requests. Please try again later.',
      timestamp: new Date().toISOString(),
    });
    return;
  }

  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const pathname = url.pathname.replace(/^\/api/, ''); // Strip /api prefix if present
  const params = url.searchParams;

  const route = findRoute(req.method || 'GET', pathname);

  if (!route) {
    error(res, 404, `Not found: ${req.method} ${pathname}`);
    return;
  }

  try {
    incrementRequestCount();
    const body = ['POST', 'PUT'].includes(req.method || '') ? await parseBody(req) : undefined;
    await route.handler(req, res, params, body);
  } catch (err) {
    error(res, 500, err instanceof Error ? err.message : 'Internal server error');
  }
});

// ============================================================================
// CLI
// ============================================================================

function printRoutes(): void {
  console.log('\n📍 Available Endpoints:\n');
  const grouped: Record<string, RouteHandler[]> = {};

  for (const route of routes) {
    const category = route.path.split('/')[1] || 'root';
    if (!grouped[category]) grouped[category] = [];
    grouped[category].push(route);
  }

  for (const [category, categoryRoutes] of Object.entries(grouped)) {
    console.log(`  ${category.toUpperCase()}`);
    for (const route of categoryRoutes) {
      console.log(`    ${route.method.padEnd(6)} ${route.path}`);
    }
    console.log();
  }
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'start':
      server.listen(PORT, () => {
        console.log(`
╔══════════════════════════════════════════════════════════════╗
║            🌐 REST API SERVER - AI TOOLS                     ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at http://localhost:${PORT.toString().padEnd(24)}║
║                                                              ║
║  Endpoints: ${routes.length.toString().padEnd(49)}║
║                                                              ║
║  Features:                                                   ║
║    • Full CORS support                                       ║
║    • Rate limiting (${RATE_LIMIT_MAX_REQUESTS}/min per IP)                            ║
║    • JSON request/response                                   ║
║    • Query parameter filtering                               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Quick Start:                                                ║
║    GET  /health           - Health check                     ║
║    GET  /compliance/score - Get compliance score             ║
║    POST /security/scan    - Run security scan                ║
║    GET  /metrics          - Get AI metrics                   ║
║    GET  /dashboard        - Get ASCII dashboard              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        `);
      });
      break;

    case 'routes':
      printRoutes();
      break;

    default:
      console.log(`
REST API Server for AI Tools

Commands:
  start       Start the API server
  routes      List all available routes

Environment Variables:
  AI_API_PORT        Server port (default: 3200)
  RATE_LIMIT_MAX     Max requests per minute per IP (default: 100)

Example:
  npm run ai:api:start
  AI_API_PORT=3300 npm run ai:api:start

API Usage:
  curl http://localhost:3200/health
  curl http://localhost:3200/compliance/score
  curl -X POST http://localhost:3200/security/scan
      `);
  }
}

main();
