/**
 * ATLAS REST API Server
 * Lightweight HTTP server with JWT auth and RBAC
 */

import * as http from 'http';
import { URL } from 'url';
import { router } from './router.js';
import {
  authenticate as authAuthenticate,
  authorize,
  pathToResource,
  methodToAction,
  logAuditEntry,
  getAuthConfig,
  AuthConfig,
  User,
} from './auth.js';

// ============================================================================
// Types
// ============================================================================

export interface APIRequest {
  method: string;
  path: string;
  query: Record<string, string>;
  body: unknown;
  headers: http.IncomingHttpHeaders;
  user?: User;
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
  jwtSecret?: string;
  jwtExpiry?: number;
  enableRBAC?: boolean;
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
// Authentication & Authorization
// ============================================================================

function authenticateRequest(
  req: http.IncomingMessage,
  _config: ServerConfig,
  authConfig: AuthConfig
): { valid: boolean; user?: User; error?: string } {
  // Build headers object
  const headers: Record<string, string | string[] | undefined> = {};
  Object.entries(req.headers).forEach(([key, value]) => {
    headers[key] = value;
  });

  return authAuthenticate(headers, authConfig);
}

function authorizeRequest(
  user: User | undefined,
  path: string,
  method: string,
  enableRBAC: boolean
): boolean {
  if (!enableRBAC) {
    return true; // RBAC disabled
  }

  const resource = pathToResource(path);
  const action = methodToAction(method);

  return authorize(user, resource, action);
}

// ============================================================================
// Request Handler
// ============================================================================

async function handleRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  config: ServerConfig,
  authConfig: AuthConfig
): Promise<void> {
  const ip = req.socket.remoteAddress;

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
  const method = req.method || 'GET';

  // Skip auth for health check and root
  const isPublicPath = path === '/health' || path === '/' || path === '/auth/login';
  let user: User | undefined;

  if (!isPublicPath) {
    const auth = authenticateRequest(req, config, authConfig);

    if (!auth.valid) {
      logAuditEntry({
        action: 'authenticate',
        resource: pathToResource(path),
        method,
        path,
        ip,
        success: false,
        error: auth.error,
      });
      sendError(res, 401, auth.error || 'Unauthorized');
      return;
    }

    user = auth.user;

    // Check authorization (RBAC)
    if (!authorizeRequest(user, path, method, config.enableRBAC ?? false)) {
      logAuditEntry({
        userId: user?.id,
        username: user?.username,
        action: 'authorize',
        resource: pathToResource(path),
        method,
        path,
        ip,
        success: false,
        error: 'Forbidden',
      });
      sendError(res, 403, 'Forbidden - insufficient permissions');
      return;
    }
  }

  try {
    const body = await parseBody(req);
    const apiRequest: APIRequest = {
      method,
      path,
      query: parseQuery(url),
      body,
      headers: req.headers,
      user,
    };

    const response = await router(apiRequest);

    // Log successful request
    logAuditEntry({
      userId: user?.id,
      username: user?.username,
      action: methodToAction(method),
      resource: pathToResource(path),
      method,
      path,
      ip,
      success: true,
    });

    if (response.headers) {
      Object.entries(response.headers).forEach(([key, value]) => {
        res.setHeader(key, value);
      });
    }

    sendJSON(res, response.status, response.body);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Internal server error';
    console.error('API Error:', error);

    logAuditEntry({
      userId: user?.id,
      username: user?.username,
      action: methodToAction(method),
      resource: pathToResource(path),
      method,
      path,
      ip,
      success: false,
      error: message,
    });

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
    jwtSecret: process.env.ATLAS_JWT_SECRET,
    jwtExpiry: parseInt(process.env.ATLAS_JWT_EXPIRY || '86400', 10),
    enableRBAC: process.env.ATLAS_ENABLE_RBAC === 'true',
    ...config,
  };

  const authConfig: AuthConfig = {
    ...getAuthConfig(),
    apiKey: fullConfig.apiKey,
    jwtSecret: fullConfig.jwtSecret || getAuthConfig().jwtSecret,
    jwtExpiry: fullConfig.jwtExpiry || 86400,
  };

  const server = http.createServer((req, res) => {
    handleRequest(req, res, fullConfig, authConfig).catch((error) => {
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
      jwtSecret: process.env.ATLAS_JWT_SECRET,
      jwtExpiry: parseInt(process.env.ATLAS_JWT_EXPIRY || '86400', 10),
      enableRBAC: process.env.ATLAS_ENABLE_RBAC === 'true',
      ...config,
    };

    const server = createServer(fullConfig);

    server.on('error', reject);

    server.listen(fullConfig.port, fullConfig.host, () => {
      console.log(`ATLAS API server running at http://${fullConfig.host}:${fullConfig.port}`);
      if (fullConfig.apiKey || fullConfig.jwtSecret) {
        console.log('Authentication enabled (API key + JWT)');
      } else {
        console.log('WARNING: No authentication configured');
      }
      if (fullConfig.enableRBAC) {
        console.log('RBAC authorization enabled');
      }
      resolve(server);
    });
  });
}
