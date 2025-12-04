/**
 * REST API Route Handlers
 * Extracted from server.ts for maintainability
 */

import { IncomingMessage, ServerResponse } from 'http';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

// Types
export interface RouteHandler {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  handler: (
    req: IncomingMessage,
    res: ServerResponse,
    params: URLSearchParams,
    body?: unknown
  ) => Promise<void>;
}

interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// Utilities
export function jsonResponse<T>(res: ServerResponse, status: number, data: ApiResponse<T>): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data, null, 2));
}

export function success<T>(res: ServerResponse, data: T): void {
  jsonResponse(res, 200, { success: true, data, timestamp: new Date().toISOString() });
}

export function error(res: ServerResponse, status: number, message: string): void {
  jsonResponse(res, status, {
    success: false,
    error: message,
    timestamp: new Date().toISOString(),
  });
}

export async function parseBody(req: IncomingMessage): Promise<unknown> {
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

export function readJsonFile(filePath: string): unknown {
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }
  return null;
}

export function runCommand(cmd: string): string {
  try {
    return execSync(cmd, { cwd: ROOT, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
  } catch (err) {
    throw new Error(err instanceof Error ? err.message : String(err));
  }
}

// Health Routes
export const healthRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/health',
    handler: async (_req, res): Promise<void> => {
      success(res, { status: 'healthy', uptime: process.uptime() });
    },
  },
];

// Compliance Routes
export const complianceRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/compliance/score',
    handler: async (_req, res): Promise<void> => {
      const report = readJsonFile(path.join(AI_DIR, 'compliance-report.json'));
      if (report) success(res, report);
      else error(res, 404, 'No compliance report found. Run npm run ai:compliance:check first.');
    },
  },
  {
    method: 'POST',
    path: '/compliance/check',
    handler: async (_req, res, _params, body): Promise<void> => {
      try {
        const files = (body as { files?: string[] })?.files || [];
        runCommand(`npm run ai:compliance:check ${files.join(' ')}`);
        success(res, readJsonFile(path.join(AI_DIR, 'compliance-report.json')));
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Compliance check failed');
      }
    },
  },
  {
    method: 'GET',
    path: '/compliance/rules',
    handler: async (_req, res): Promise<void> => {
      try {
        success(res, { rules: runCommand('npm run ai:compliance rules 2>&1') });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to list rules');
      }
    },
  },
];

// Security Routes
export const securityRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/security/report',
    handler: async (_req, res): Promise<void> => {
      const report = readJsonFile(path.join(AI_DIR, 'security-report.json'));
      if (report) success(res, report);
      else error(res, 404, 'No security report found. Run npm run ai:security:scan first.');
    },
  },
  {
    method: 'POST',
    path: '/security/scan',
    handler: async (_req, res, _params, body): Promise<void> => {
      try {
        const scanType = (body as { type?: string })?.type || 'scan';
        runCommand(`npm run ai:security:${scanType}`);
        success(res, readJsonFile(path.join(AI_DIR, 'security-report.json')));
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Security scan failed');
      }
    },
  },
];

// Cache Routes
export const cacheRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/cache/stats',
    handler: async (_req, res): Promise<void> => {
      try {
        success(res, { stats: runCommand('npm run ai:cache:stats 2>&1') });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get cache stats');
      }
    },
  },
  {
    method: 'DELETE',
    path: '/cache',
    handler: async (_req, res, params): Promise<void> => {
      try {
        const layer = params.get('layer') || '';
        runCommand(`npm run ai:cache:clear ${layer}`);
        success(res, { message: layer ? `Cleared ${layer} layer` : 'Cleared expired entries' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to clear cache');
      }
    },
  },
];

// Monitor Routes
export const monitorRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/monitor/status',
    handler: async (_req, res): Promise<void> => {
      const state = readJsonFile(path.join(AI_DIR, 'monitor-state.json'));
      success(res, state || { message: 'Monitor not initialized' });
    },
  },
  {
    method: 'POST',
    path: '/monitor/check',
    handler: async (_req, res): Promise<void> => {
      try {
        runCommand('npm run ai:monitor:check');
        success(res, readJsonFile(path.join(AI_DIR, 'monitor-state.json')));
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Monitor check failed');
      }
    },
  },
];

// Errors & Issues Routes
export const errorsRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/errors',
    handler: async (_req, res, params): Promise<void> => {
      const errorLog = readJsonFile(path.join(AI_DIR, 'error-log.json'));
      if (errorLog) {
        const severity = params.get('severity');
        let errors = (errorLog as { errors: unknown[] }).errors || [];
        if (severity)
          errors = errors.filter((e: unknown) => (e as { severity: string }).severity === severity);
        success(res, { errors });
      } else success(res, { errors: [] });
    },
  },
  {
    method: 'GET',
    path: '/errors/stats',
    handler: async (_req, res): Promise<void> => {
      const errorLog = readJsonFile(path.join(AI_DIR, 'error-log.json'));
      success(res, { stats: errorLog ? (errorLog as { stats: unknown }).stats : { total: 0 } });
    },
  },
];

export const issuesRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/issues',
    handler: async (_req, res, params): Promise<void> => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        let list = (issues as { issues: unknown[] }).issues || [];
        const category = params.get('category'),
          status = params.get('status'),
          priority = params.get('priority');
        if (category)
          list = list.filter((i: unknown) => (i as { category: string }).category === category);
        if (status) list = list.filter((i: unknown) => (i as { status: string }).status === status);
        if (priority)
          list = list.filter((i: unknown) => (i as { priority: string }).priority === priority);
        success(res, { issues: list });
      } else success(res, { issues: [] });
    },
  },
  {
    method: 'GET',
    path: '/issues/critical',
    handler: async (_req, res): Promise<void> => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      if (issues) {
        const critical = ((issues as { issues: unknown[] }).issues || []).filter(
          (i: unknown) =>
            (i as { priority: string }).priority === 'critical' &&
            (i as { status: string }).status === 'open'
        );
        success(res, { issues: critical });
      } else success(res, { issues: [] });
    },
  },
  {
    method: 'GET',
    path: '/issues/stats',
    handler: async (_req, res): Promise<void> => {
      const issues = readJsonFile(path.join(AI_DIR, 'issues.json'));
      success(res, { stats: issues ? (issues as { stats: unknown }).stats : { total: 0 } });
    },
  },
];

// Metrics, Context, Tasks, Telemetry, Dashboard Routes
export const miscRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/metrics',
    handler: async (_req, res) => {
      const metrics = readJsonFile(path.join(AI_DIR, 'metrics.json'));
      if (metrics) success(res, metrics);
      else error(res, 404, 'No metrics found');
    },
  },
  {
    method: 'GET',
    path: '/context',
    handler: async (_req, res) => {
      const contextPath = path.join(AI_DIR, 'context.yaml');
      if (fs.existsSync(contextPath))
        success(res, { context: fs.readFileSync(contextPath, 'utf8') });
      else error(res, 404, 'No context file found');
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
  {
    method: 'GET',
    path: '/tasks/history',
    handler: async (_req, res) => {
      const history = readJsonFile(path.join(AI_DIR, 'task-history.json'));
      success(res, history || { tasks: [] });
    },
  },
  {
    method: 'POST',
    path: '/tasks/start',
    handler: async (_req, res, _params, body) => {
      try {
        const { type, scope, description } = body as {
          type: string;
          scope?: string;
          description: string;
        };
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
        const {
          success: s,
          filesChanged,
          notes,
        } = body as { success: boolean; filesChanged?: string; notes?: string };
        runCommand(`npm run ai:complete ${s} "${filesChanged || ''}" 0 0 0 "${notes || ''}"`);
        success(res, { message: 'Task completed' });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to complete task');
      }
    },
  },
  {
    method: 'GET',
    path: '/telemetry/status',
    handler: async (_req, res) => {
      try {
        success(res, { status: runCommand('npm run ai:telemetry:status 2>&1') });
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
        success(res, { alerts: runCommand('npm run ai:telemetry:alerts 2>&1') });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get alerts');
      }
    },
  },
  {
    method: 'GET',
    path: '/dashboard',
    handler: async (_req, res) => {
      try {
        success(res, { dashboard: runCommand('npm run ai:dashboard 2>&1') });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get dashboard');
      }
    },
  },
];

// All routes combined
export const routes: RouteHandler[] = [
  ...healthRoutes,
  ...complianceRoutes,
  ...securityRoutes,
  ...cacheRoutes,
  ...monitorRoutes,
  ...errorsRoutes,
  ...issuesRoutes,
  ...miscRoutes,
];
