#!/usr/bin/env npx tsx
/**
 * REST API Server for AI Tools
 * Exposes AI tools via HTTP REST endpoints
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { URL } from 'url';

import { routes, RouteHandler, parseBody, error } from './routes.js';

const PORT = parseInt(process.env.AI_API_PORT || '3200', 10);

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

  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const pathname = url.pathname.replace(/^\/api/, ''); // Strip /api prefix if present
  const params = url.searchParams;

  const route = findRoute(req.method || 'GET', pathname);

  if (!route) {
    error(res, 404, `Not found: ${req.method} ${pathname}`);
    return;
  }

  try {
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
║    • JSON request/response                                   ║
║    • Query parameter filtering                               ║
║    • Comprehensive error handling                            ║
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
  AI_API_PORT    Server port (default: 3200)

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
