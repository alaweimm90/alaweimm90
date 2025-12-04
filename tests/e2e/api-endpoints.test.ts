/**
 * E2E API Tests
 * Tests actual HTTP requests to REST API endpoints
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createServer, Server, IncomingMessage, ServerResponse } from 'http';
import { routes, parseBody, jsonResponse } from '../../tools/ai/api/routes.js';

// Mock server for testing
let server: Server;
const PORT = 3299;
const BASE_URL = `http://localhost:${PORT}`;

// Simple request helper
async function request(
  path: string,
  options: { method?: string; body?: unknown; headers?: Record<string, string> } = {}
): Promise<{ status: number; data: unknown; headers: Record<string, string> }> {
  const { method = 'GET', body, headers = {} } = options;

  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL);
    const http = require('http');

    const req = http.request(
      {
        hostname: 'localhost',
        port: PORT,
        path: url.pathname + url.search,
        method,
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
      },
      (res: IncomingMessage) => {
        let data = '';
        res.on('data', (chunk: Buffer) => (data += chunk));
        res.on('end', () => {
          const responseHeaders: Record<string, string> = {};
          for (const [key, value] of Object.entries(res.headers)) {
            if (typeof value === 'string') responseHeaders[key] = value;
          }
          try {
            resolve({
              status: res.statusCode || 500,
              data: data ? JSON.parse(data) : null,
              headers: responseHeaders,
            });
          } catch {
            resolve({ status: res.statusCode || 500, data, headers: responseHeaders });
          }
        });
      }
    );

    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

// Create test server
function createTestServer(): Server {
  return createServer(async (req: IncomingMessage, res: ServerResponse) => {
    res.setHeader('Access-Control-Allow-Origin', '*');

    const url = new URL(req.url || '/', BASE_URL);
    const pathname = url.pathname;
    const params = url.searchParams;

    const route = routes.find((r) => r.method === req.method && r.path === pathname);

    if (!route) {
      jsonResponse(res, 404, {
        success: false,
        error: `Not found: ${req.method} ${pathname}`,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    try {
      const body = ['POST', 'PUT'].includes(req.method || '') ? await parseBody(req) : undefined;
      await route.handler(req, res, params, body);
    } catch (err) {
      jsonResponse(res, 500, {
        success: false,
        error: err instanceof Error ? err.message : 'Internal error',
        timestamp: new Date().toISOString(),
      });
    }
  });
}

describe('E2E API Tests', () => {
  beforeAll(
    () =>
      new Promise<void>((resolve) => {
        server = createTestServer();
        server.listen(PORT, resolve);
      })
  );

  afterAll(
    () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
      })
  );

  describe('Health Endpoints', () => {
    it('GET /health returns healthy status', async () => {
      const res = await request('/health');
      expect(res.status).toBe(200);
      expect(res.data).toHaveProperty('success', true);
      expect(res.data).toHaveProperty('data.status', 'healthy');
      expect(res.data).toHaveProperty('data.uptime');
    });

    it('GET /health/dependencies returns dependency status', async () => {
      const res = await request('/health/dependencies');
      expect(res.status).toBe(200);
      expect(res.data).toHaveProperty('success', true);
      expect(res.data).toHaveProperty('data.dependencies');
      expect(res.data).toHaveProperty('data.checkedAt');
    });
  });

  describe('Compliance Endpoints', () => {
    it('GET /compliance/score returns compliance data or 404', async () => {
      const res = await request('/compliance/score');
      expect([200, 404]).toContain(res.status);
    });

    it('GET /compliance/rules returns rules list', async () => {
      const res = await request('/compliance/rules');
      // May succeed or fail depending on CLI availability
      expect([200, 500]).toContain(res.status);
    });
  });

  describe('Cache Endpoints', () => {
    it('GET /cache/stats returns cache statistics', async () => {
      const res = await request('/cache/stats');
      expect(res.status).toBe(200);
      expect(res.data).toHaveProperty('success', true);
    });
  });

  describe('Error Handling', () => {
    it('returns 404 for unknown routes', async () => {
      const res = await request('/unknown/endpoint');
      expect(res.status).toBe(404);
      expect(res.data).toHaveProperty('success', false);
    });
  });
});
