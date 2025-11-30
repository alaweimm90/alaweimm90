/**
 * ATLAS REST API Server
 * Lightweight HTTP server for multi-agent orchestration
 */

import * as http from 'http';
import { URL } from 'url';
import { router } from './router.js';

// ============================================================================
// Types
// ============================================================================

export interface APIRequest {
  method: string;
  path: string;
  query: Record<string, string>;
  body: unknown;
  headers: http.IncomingHttpHeaders;
}

export interface APIResponse {
  status: number;
  body: unknown;
  headers?: Record<string, string>;
}

export interface ServerConfig {
  port: number;
  host: string;
  apiKey?: string;
}

// ============================================================================
// Request Parsing
// ============================================================================

async function parseBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];

    req.on('data', (chunk: Buffer) => {
      chunks.push(chunk);
    });

    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8');
      if (!body) {
        resolve(null);
        return;
      }

      try {
        resolve(JSON.parse(body));
      } catch {
        resolve(body);
      }
    });

    req.on('error', reject);
  });
}

function parseQuery(url: URL): Record<string, string> {
  const query: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    query[key] = value;
  });
  return query;
}

// ============================================================================
// Response Helpers
// ============================================================================

function sendJSON(res: http.ServerResponse, status: number, data: unknown): void {
  const body = JSON.stringify(data, null, 2);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
  });
  res.end(body);
}

function sendError(res: http.ServerResponse, status: number, message: string): void {
  sendJSON(res, status, { error: message, status });
}

// ============================================================================
// Authentication
// ============================================================================

function authenticate(
  req: http.IncomingMessage,
  config: ServerConfig
): { valid: boolean; error?: string } {
  if (!config.apiKey) {
    return { valid: true }; // No auth required
  }

  const authHeader = req.headers['authorization'];
  const apiKeyHeader = req.headers['x-api-key'];

  // Check Bearer token
  if (authHeader?.startsWith('Bearer ')) {
    const token = authHeader.slice(7);
    if (token === config.apiKey) {
      return { valid: true };
    }
  }

  // Check X-API-Key header
  if (apiKeyHeader === config.apiKey) {
    return { valid: true };
  }

  return { valid: false, error: 'Invalid or missing API key' };
}

// ============================================================================
// Request Handler
// ============================================================================

async function handleRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  config: ServerConfig
): Promise<void> {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
      'Access-Control-Max-Age': '86400',
    });
    res.end();
    return;
  }

  // Parse request
  const url = new URL(req.url || '/', `http://${req.headers.host}`);
  const path = url.pathname;

  // Skip auth for health check
  if (path !== '/health' && path !== '/') {
    const auth = authenticate(req, config);
    if (!auth.valid) {
      sendError(res, 401, auth.error || 'Unauthorized');
      return;
    }
  }

  try {
    const body = await parseBody(req);
    const apiRequest: APIRequest = {
      method: req.method || 'GET',
      path,
      query: parseQuery(url),
      body,
      headers: req.headers,
    };

    const response = await router(apiRequest);

    if (response.headers) {
      Object.entries(response.headers).forEach(([key, value]) => {
        res.setHeader(key, value);
      });
    }

    sendJSON(res, response.status, response.body);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Internal server error';
    console.error('API Error:', error);
    sendError(res, 500, message);
  }
}

// ============================================================================
// Server Creation
// ============================================================================

export function createServer(config: Partial<ServerConfig> = {}): http.Server {
  const fullConfig: ServerConfig = {
    port: parseInt(process.env.ATLAS_API_PORT || '3200', 10),
    host: process.env.ATLAS_API_HOST || '127.0.0.1',
    apiKey: process.env.ATLAS_API_KEY,
    ...config,
  };

  const server = http.createServer((req, res) => {
    handleRequest(req, res, fullConfig).catch((error) => {
      console.error('Unhandled error:', error);
      sendError(res, 500, 'Internal server error');
    });
  });

  return server;
}

export function startServer(config: Partial<ServerConfig> = {}): Promise<http.Server> {
  return new Promise((resolve, reject) => {
    const fullConfig: ServerConfig = {
      port: parseInt(process.env.ATLAS_API_PORT || '3200', 10),
      host: process.env.ATLAS_API_HOST || '127.0.0.1',
      apiKey: process.env.ATLAS_API_KEY,
      ...config,
    };

    const server = createServer(fullConfig);

    server.on('error', reject);

    server.listen(fullConfig.port, fullConfig.host, () => {
      console.log(`ATLAS API server running at http://${fullConfig.host}:${fullConfig.port}`);
      if (fullConfig.apiKey) {
        console.log('API key authentication enabled');
      } else {
        console.log('WARNING: No API key configured - authentication disabled');
      }
      resolve(server);
    });
  });
}
