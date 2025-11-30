#!/usr/bin/env npx tsx
/**
 * REST API Server for AI Tools
 * Exposes AI tools via HTTP REST endpoints
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { URL } from 'url';

// ============================================================================
// Configuration
// ============================================================================

const PORT = parseInt(process.env.AI_API_PORT || '3200', 10);
const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

// ============================================================================
// Types
// ============================================================================

interface RouteHandler {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  handler: (req: IncomingMessage, res: ServerResponse, params: URLSearchParams, body?: unknown) => Promise<void>;
}

interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// ============================================================================
// Utilities
// ============================================================================

function jsonResponse<T>(res: ServerResponse, status: number, data: ApiResponse<T>): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data, null, 2));
}

function success<T>(res: ServerResponse, data: T): void {
  jsonResponse(res, 200, { success: true, data, timestamp: new Date().toISOString() });
}

function error(res: ServerResponse, status: number, message: string): void {
  jsonResponse(res, status, { success: false, error: message, timestamp: new Date().toISOString() });
}

async function parseBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve({});
      }
    });
    req.on('error', reject);
  });
}

function readJsonFile(filePath: string): unknown {
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }
  return null;
}

function runCommand(cmd: string): string {
  try {
    return execSync(cmd, { cwd: ROOT, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
  } catch (err) {
    throw new Error(err instanceof Error ? err.message : String(err));
  }
}

// ============================================================================
// Route Handlers
// ============================================================================

const routes: RouteHandler[] = [
  // Health check
  {
    method: 'GET',
    path: '/health',
    handler: async (_req, res) => {
      success(res, { status: 'healthy', uptime: process.uptime() });
    },
  },

  // Compliance endpoints
  {
    method: 'GET',
    path: '/compliance/score',
    handler: async (_req, res) => {
      const report = readJsonFile(path.join(AI_DIR, 'compliance-report.json'));
      if (report) {
        success(res, report);
      } else {
        error(res, 404, 'No compliance report found. Run npm run ai:compliance:check first.');
      }
    },
  },
  {
    method: 'POST',
    path: '/compliance/check',
    handler: async (_req, res, _params, body) => {
      try {
        const files = (body as { files?: string[] })?.files || [];
        runCommand(`npm run ai:compliance:check ${files.join(' ')}`);
        const report = readJsonFile(path.join(AI_DIR, 'compliance-report.json'));
        success(res, report);
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Compliance check failed');
      }
    },
  },
  {
    method: 'GET',
    path: '/compliance/rules',
    handler: async (_req, res) => {
      try {
        const output = runCommand('npm run ai:compliance rules 2>&1');
        success(res, { rules: output });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to list rules');
      }
    },
  },

  // Security endpoints
  {
    method: 'GET',
    path: '/security/report',
    handler: async (_req, res) => {
      const report = readJsonFile(path.join(AI_DIR, 'security-report.json'));
      if (report) {
        success(res, report);
      } else {
        error(res, 404, 'No security report found. Run npm run ai:security:scan first.');
      }
    },
  },
  {
    method: 'POST',
    path: '/security/scan',
    handler: async (_req, res, _params, body) => {
      try {
        const scanType = (body as { type?: string })?.type || 'scan';
        runCommand(`npm run ai:security:${scanType}`);
        const report = readJsonFile(path.join(AI_DIR, 'security-report.json'));
        success(res, report);
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Security scan failed');
      }
    },
  },

  // Cache endpoints
  {
    method: 'GET',
    path: '/cache/stats',
    handler: async (_req, res) => {
      try {
        const output = runCommand('npm run ai:cache:stats 2>&1');
        success(res, { stats: output });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get cache stats');
      }
    },
  },
  {
    method: 'DELETE',
    path: '/cache',
    handler: async (_req, res, params) => {
      try {
        const layer = params.get('layer') || '';
        runCommand(`npm run ai:cache:clear ${layer}`);
        success(res, { message: layer ? `Cleared ${layer} layer` : 'Cleared expired entries' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to clear cache');
      }
    },
  },

  // Monitor endpoints
  {
    method: 'GET',
    path: '/monitor/status',
    handler: async (_req, res) => {
      const state = readJsonFile(path.join(AI_DIR, 'monitor-state.json'));
      if (state) {
        success(res, state);
      } else {
        success(res, { message: 'Monitor not initialized' });
      }
    },
  },
  {
    method: 'POST',
    path: '/monitor/check',
    handler: async (_req, res) => {
      try {
        runCommand('npm run ai:monitor:check');
        const state = readJsonFile(path.join(AI_DIR, 'monitor-state.json'));
        success(res, state);
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Monitor check failed');
      }
    },
  },

  // Errors endpoints
  {
    method: 'GET',
    path: '/errors',
    handler: async (_req, res, params) => {
      const errorLog = readJsonFile(path.join(AI_DIR, 'error-log.json'));
      if (errorLog) {
        const severity = params.get('severity');
        let errors = (errorLog as { errors: unknown[] }).errors || [];
        if (severity) {
          errors = errors.filter((e: unknown) => (e as { severity: string }).severity === severity);
        }
        success(res, { errors });
      } else {
        success(res, { errors: [] });
      }
    },
  },
  {
    method: 'GET',
    path: '/errors/stats',
    handler: async (_req, res) => {
      const errorLog = readJsonFile(path.join(AI_DIR, 'error-log.json'));
      if (errorLog) {
        success(res, { stats: (errorLog as { stats: unknown }).stats });
      } else {
        success(res, { stats: { total: 0 } });
      }
    },
  },

  // Issues endpoints
  {
    method: 'GET',
    path: '/issues',
    handler: async (_req, res, params) => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        let list = (issues as { issues: unknown[] }).issues || [];
        const category = params.get('category');
        const status = params.get('status');
        const priority = params.get('priority');
        if (category) list = list.filter((i: unknown) => (i as { category: string }).category === category);
        if (status) list = list.filter((i: unknown) => (i as { status: string }).status === status);
        if (priority) list = list.filter((i: unknown) => (i as { priority: string }).priority === priority);
        success(res, { issues: list });
      } else {
        success(res, { issues: [] });
      }
    },
  },
  {
    method: 'GET',
    path: '/issues/critical',
    handler: async (_req, res) => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        const critical = ((issues as { issues: unknown[] }).issues || []).filter(
          (i: unknown) => (i as { priority: string }).priority === 'critical' && (i as { status: string }).status === 'open'
        );
        success(res, { issues: critical });
      } else {
        success(res, { issues: [] });
      }
    },
  },
  {
    method: 'GET',
    path: '/issues/stats',
    handler: async (_req, res) => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        success(res, { stats: (issues as { stats: unknown }).stats });
      } else {
        success(res, { stats: { total: 0 } });
      }
    },
  },

  // Metrics endpoints
  {
    method: 'GET',
    path: '/metrics',
    handler: async (_req, res) => {
      const metrics = readJsonFile(path.join(AI_DIR, 'metrics.json'));
      if (metrics) {
        success(res, metrics);
      } else {
        error(res, 404, 'No metrics found');
      }
    },
  },

  // Context endpoints
  {
    method: 'GET',
    path: '/context',
    handler: async (_req, res) => {
      const contextPath = path.join(AI_DIR, 'context.yaml');
      if (fs.existsSync(contextPath)) {
        const content = fs.readFileSync(contextPath, 'utf8');
        success(res, { context: content });
      } else {
        error(res, 404, 'No context file found');
      }
    },
  },
  {
    method: 'POST',
    path: '/sync',
    handler: async (_req, res) => {
      try {
        runCommand('npm run ai:sync');
        success(res, { message: 'Sync completed' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Sync failed');
      }
    },
  },

  // Task endpoints
  {
    method: 'GET',
    path: '/tasks/history',
    handler: async (_req, res) => {
      const history = readJsonFile(path.join(AI_DIR, 'task-history.json'));
      if (history) {
        success(res, history);
      } else {
        success(res, { tasks: [] });
      }
    },
  },
  {
    method: 'POST',
    path: '/tasks/start',
    handler: async (_req, res, _params, body) => {
      try {
        const { type, scope, description } = body as { type: string; scope?: string; description: string };
        runCommand(`npm run ai:start ${type} ${scope || ''} "${description}"`);
        success(res, { message: 'Task started' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to start task');
      }
    },
  },
  {
    method: 'POST',
    path: '/tasks/complete',
    handler: async (_req, res, _params, body) => {
      try {
        const { success: taskSuccess, filesChanged, notes } = body as { success: boolean; filesChanged?: string; notes?: string };
        runCommand(`npm run ai:complete ${taskSuccess} "${filesChanged || ''}" 0 0 0 "${notes || ''}"`);
        success(res, { message: 'Task completed' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to complete task');
      }
    },
  },

  // Telemetry endpoints
  {
    method: 'GET',
    path: '/telemetry/status',
    handler: async (_req, res) => {
      try {
        const output = runCommand('npm run ai:telemetry:status 2>&1');
        success(res, { status: output });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get telemetry status');
      }
    },
  },
  {
    method: 'GET',
    path: '/telemetry/alerts',
    handler: async (_req, res) => {
      try {
        const output = runCommand('npm run ai:telemetry:alerts 2>&1');
        success(res, { alerts: output });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get alerts');
      }
    },
  },

  // Dashboard endpoint
  {
    method: 'GET',
    path: '/dashboard',
    handler: async (_req, res) => {
      try {
        const output = runCommand('npm run ai:dashboard 2>&1');
        success(res, { dashboard: output });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get dashboard');
      }
    },
  },
];

// ============================================================================
// Router
// ============================================================================

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
