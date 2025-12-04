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

// JSON file structure interfaces
interface TaskHistory {
  tasks?: Array<{ type?: string; [key: string]: unknown }>;
}

interface ErrorTracking {
  errors?: unknown[];
}

interface IssueTracking {
  issues?: unknown[];
}

interface MetricsData {
  healthScore?: number;
  complianceScore?: number;
  [key: string]: unknown;
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
  {
    method: 'GET',
    path: '/health/dependencies',
    handler: async (_req, res): Promise<void> => {
      const dependencies: Record<string, { status: string; latency?: number; error?: string }> = {};

      // Check MCP servers config
      const mcpStart = Date.now();
      try {
        const mcpPath = path.join(AI_DIR, 'mcp/mcp-servers.json');
        if (fs.existsSync(mcpPath)) {
          const config = JSON.parse(fs.readFileSync(mcpPath, 'utf8'));
          const serverCount = Object.keys(config.mcpServers || {}).length;
          dependencies.mcp = { status: 'healthy', latency: Date.now() - mcpStart };
          dependencies.mcp_servers = { status: `${serverCount} configured`, latency: 0 };
        } else {
          dependencies.mcp = { status: 'not_configured', latency: Date.now() - mcpStart };
        }
      } catch (err) {
        dependencies.mcp = {
          status: 'error',
          error: err instanceof Error ? err.message : 'Unknown',
        };
      }

      // Check cache
      const cacheStart = Date.now();
      try {
        const cachePath = path.join(AI_DIR, 'cache');
        dependencies.cache = {
          status: fs.existsSync(cachePath) ? 'healthy' : 'not_initialized',
          latency: Date.now() - cacheStart,
        };
      } catch (err) {
        dependencies.cache = {
          status: 'error',
          error: err instanceof Error ? err.message : 'Unknown',
        };
      }

      // Check filesystem
      const fsStart = Date.now();
      try {
        fs.accessSync(ROOT, fs.constants.R_OK | fs.constants.W_OK);
        dependencies.filesystem = { status: 'healthy', latency: Date.now() - fsStart };
      } catch (err) {
        dependencies.filesystem = {
          status: 'error',
          error: err instanceof Error ? err.message : 'Unknown',
        };
      }

      // Check AI context
      const ctxStart = Date.now();
      try {
        const ctxPath = path.join(AI_DIR, 'context.yaml');
        dependencies.ai_context = {
          status: fs.existsSync(ctxPath) ? 'healthy' : 'missing',
          latency: Date.now() - ctxStart,
        };
      } catch (err) {
        dependencies.ai_context = {
          status: 'error',
          error: err instanceof Error ? err.message : 'Unknown',
        };
      }

      const allHealthy = Object.values(dependencies).every(
        (d) => d.status === 'healthy' || d.status.includes('configured')
      );

      success(res, {
        status: allHealthy ? 'healthy' : 'degraded',
        uptime: process.uptime(),
        dependencies,
        checkedAt: new Date().toISOString(),
      });
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
    handler: async (_req, res): Promise<void> => {
      const metrics = readJsonFile(path.join(AI_DIR, 'metrics.json'));
      if (metrics) success(res, metrics);
      else error(res, 404, 'No metrics found');
    },
  },
  {
    method: 'GET',
    path: '/context',
    handler: async (_req, res): Promise<void> => {
      const contextPath = path.join(AI_DIR, 'context.yaml');
      if (fs.existsSync(contextPath))
        success(res, { context: fs.readFileSync(contextPath, 'utf8') });
      else error(res, 404, 'No context file found');
    },
  },
  {
    method: 'POST',
    path: '/sync',
    handler: async (_req, res): Promise<void> => {
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
    handler: async (_req, res): Promise<void> => {
      const history = readJsonFile(path.join(AI_DIR, 'task-history.json'));
      success(res, history || { tasks: [] });
    },
  },
  {
    method: 'POST',
    path: '/tasks/start',
    handler: async (_req, res, _params, body): Promise<void> => {
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
    handler: async (_req, res, _params, body): Promise<void> => {
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
    handler: async (_req, res): Promise<void> => {
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
    handler: async (_req, res): Promise<void> => {
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
    handler: async (_req, res): Promise<void> => {
      try {
        success(res, { dashboard: runCommand('npm run ai:dashboard 2>&1') });
      } catch (err) {
        error(res, 500, err instanceof Error ? err.message : 'Failed to get dashboard');
      }
    },
  },
];

// Visual Dashboard endpoint
export const dashboardRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/dashboard/ui',
    handler: async (_req, res): Promise<void> => {
      const dashboardPath = path.join(ROOT, 'tools/ai/dashboard/index.html');
      if (fs.existsSync(dashboardPath)) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(fs.readFileSync(dashboardPath, 'utf8'));
      } else {
        error(res, 404, 'Dashboard not found');
      }
    },
  },
];

// API Documentation (Swagger UI)
export const docsRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/docs',
    handler: async (_req, res): Promise<void> => {
      const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Meta-Orchestration API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: '/docs/openapi.yaml',
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: 'BaseLayout'
    });
  </script>
</body>
</html>`;
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    },
  },
  {
    method: 'GET',
    path: '/docs/openapi.yaml',
    handler: async (_req, res): Promise<void> => {
      const specPath = path.join(ROOT, 'tools/ai/api/openapi.yaml');
      if (fs.existsSync(specPath)) {
        res.writeHead(200, { 'Content-Type': 'text/yaml' });
        res.end(fs.readFileSync(specPath, 'utf8'));
      } else {
        error(res, 404, 'OpenAPI spec not found');
      }
    },
  },
];

// Prometheus metrics endpoint
export const prometheusRoutes: RouteHandler[] = [
  {
    method: 'GET',
    path: '/metrics/prometheus',
    handler: async (_req, res): Promise<void> => {
      const metrics = (readJsonFile(path.join(AI_DIR, 'metrics.json')) || {}) as MetricsData;
      const history = (readJsonFile(path.join(AI_DIR, 'task-history.json')) || {
        tasks: [],
      }) as TaskHistory;
      const errors = (readJsonFile(path.join(ROOT, '.metaHub/reports/error-tracking.json')) || {
        errors: [],
      }) as ErrorTracking;
      const issues = (readJsonFile(path.join(ROOT, '.metaHub/reports/issues.json')) || {
        issues: [],
      }) as IssueTracking;

      const promMetrics: string[] = [
        '# HELP ai_tasks_total Total number of AI tasks',
        '# TYPE ai_tasks_total counter',
        `ai_tasks_total ${history.tasks?.length || 0}`,
        '',
        '# HELP ai_tasks_by_type Tasks grouped by type',
        '# TYPE ai_tasks_by_type gauge',
      ];

      // Count tasks by type
      const taskTypes: Record<string, number> = {};
      for (const task of history.tasks || []) {
        const type = task.type || 'unknown';
        taskTypes[type] = (taskTypes[type] || 0) + 1;
      }
      for (const [type, count] of Object.entries(taskTypes)) {
        promMetrics.push(`ai_tasks_by_type{type="${type}"} ${count}`);
      }

      promMetrics.push('');
      promMetrics.push('# HELP ai_errors_total Total number of errors');
      promMetrics.push('# TYPE ai_errors_total counter');
      promMetrics.push(`ai_errors_total ${errors.errors?.length || 0}`);
      promMetrics.push('');
      promMetrics.push('# HELP ai_issues_total Total number of issues');
      promMetrics.push('# TYPE ai_issues_total counter');
      promMetrics.push(`ai_issues_total ${issues.issues?.length || 0}`);
      promMetrics.push('');
      promMetrics.push('# HELP ai_health_score Current health score');
      promMetrics.push('# TYPE ai_health_score gauge');
      promMetrics.push(`ai_health_score ${metrics.healthScore || 100}`);
      promMetrics.push('');
      promMetrics.push('# HELP ai_compliance_score Current compliance score');
      promMetrics.push('# TYPE ai_compliance_score gauge');
      promMetrics.push(`ai_compliance_score ${metrics.complianceScore || 100}`);
      promMetrics.push('');
      promMetrics.push('# HELP ai_api_requests_total Total API requests (since restart)');
      promMetrics.push('# TYPE ai_api_requests_total counter');
      promMetrics.push(`ai_api_requests_total ${globalRequestCount}`);
      promMetrics.push('');
      promMetrics.push(`# Generated at ${new Date().toISOString()}`);

      res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4' });
      res.end(promMetrics.join('\n'));
    },
  },
];

// Request counter for Prometheus
let globalRequestCount = 0;
export function incrementRequestCount(): void {
  globalRequestCount++;
}

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
  ...dashboardRoutes,
  ...docsRoutes,
  ...prometheusRoutes,
];
