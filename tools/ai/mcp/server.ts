#!/usr/bin/env npx tsx
/**
 * MCP Server for AI Tools
 * Exposes AI tools via Model Context Protocol for integration with AI assistants
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { promisify } from 'util';
import { exec } from 'child_process';
const execAsync = promisify(exec);

// ============================================================================
// Performance Optimizations
// ============================================================================

// Simple in-memory cache for expensive operations
interface CacheEntry {
  value: unknown;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
}

class SimpleCache {
  private cache = new Map<string, CacheEntry>();

  get(key: string): unknown | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.value;
  }

  set(key: string, value: unknown, ttlMs = 300000): void {
    // Default 5 minutes
    this.cache.set(key, {
      value,
      timestamp: Date.now(),
      ttl: ttlMs,
    });
  }

  clear(pattern?: string): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
      }
    }
  }

  size(): number {
    return this.cache.size;
  }
}

const toolCache = new SimpleCache();
const REQUEST_TIMEOUT = 30000; // 30 seconds timeout for tool execution

// Async tool execution with timeout
async function executeToolAsync(name: string, params: Record<string, unknown>): Promise<unknown> {
  const startTime = Date.now();

  try {
    // Create cache key
    const cacheKey = `${name}:${JSON.stringify(params)}`;
    const cached = toolCache.get(cacheKey);
    if (cached) return cached;

    let result: unknown;

    switch (name) {
      case 'ai_compliance_check': {
        const files = params.files ? String(params.files).split(',') : [];
        const { stdout } = await execAsync(`npm run ai:compliance check ${files.join(' ')}`, {
          cwd: ROOT,
          timeout: REQUEST_TIMEOUT,
        });
        result = { success: true, output: stdout };
        break;
      }

      case 'ai_compliance_score': {
        const reportPath = path.join(AI_DIR, 'compliance-report.json');
        if (fs.existsSync(reportPath)) {
          const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
          result = { success: true, score: report.overallScore, grade: report.grade };
        } else {
          result = { success: false, message: 'No compliance report found' };
        }
        break;
      }

      case 'ai_security_scan': {
        const scanType = params.type || 'full';
        const cmd = scanType === 'full' ? 'ai:security:scan' : `ai:security:${scanType}`;
        const { stdout } = await execAsync(`npm run ${cmd}`, {
          cwd: ROOT,
          timeout: REQUEST_TIMEOUT,
        });
        result = { success: true, output: stdout };
        break;
      }

      default:
        // Fallback to sync execution for simpler tools
        result = executeTool(name, params);
    }

    // Cache if it's an expensive operation and took decent time
    const executionTime = Date.now() - startTime;
    if (executionTime > 1000) {
      // Only cache operations that took > 1 second
      toolCache.set(cacheKey, result, 300000); // 5 minute cache
    }

    return result;
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
      executionTime: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Types
// ============================================================================

interface MCPTool {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, { type: string; description: string; enum?: string[] }>;
    required: string[];
  };
}

interface MCPRequest {
  jsonrpc: '2.0';
  id: number | string;
  method: string;
  params?: Record<string, unknown>;
}

interface MCPResponse {
  jsonrpc: '2.0';
  id: number | string;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

interface MCPResource {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
}

// ============================================================================
// Tool Definitions
// ============================================================================

const TOOLS: MCPTool[] = [
  {
    name: 'ai_compliance_check',
    description: 'Run compliance check on specified files and return compliance score',
    parameters: {
      type: 'object',
      properties: {
        files: {
          type: 'string',
          description: 'Comma-separated list of files to check',
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_compliance_score',
    description: 'Get current compliance score without file-specific check',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_security_scan',
    description: 'Run security scan for secrets, vulnerabilities, and license issues',
    parameters: {
      type: 'object',
      properties: {
        paths: {
          type: 'string',
          description: 'Comma-separated paths to scan (defaults to current directory)',
        },
        type: {
          type: 'string',
          description: 'Type of scan to run',
          enum: ['full', 'secrets', 'vulns', 'licenses'],
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_cache_stats',
    description: 'Get cache statistics including hit rate and entries by layer',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_cache_clear',
    description: 'Clear cache entries, optionally by layer',
    parameters: {
      type: 'object',
      properties: {
        layer: {
          type: 'string',
          description: 'Cache layer to clear',
          enum: ['semantic', 'template', 'result', 'analysis'],
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_monitor_status',
    description: 'Get monitor status including circuit breaker states',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_errors_list',
    description: 'List recent errors with optional severity filter',
    parameters: {
      type: 'object',
      properties: {
        severity: {
          type: 'string',
          description: 'Filter by severity level',
          enum: ['low', 'medium', 'high', 'critical'],
        },
        limit: {
          type: 'string',
          description: 'Maximum number of errors to return',
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_errors_stats',
    description: 'Get error statistics by category and severity',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_issues_list',
    description: 'List tracked issues with optional filters',
    parameters: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          description: 'Filter by category',
          enum: [
            'security',
            'compliance',
            'performance',
            'maintenance',
            'documentation',
            'testing',
            'dependency',
            'architecture',
          ],
        },
        status: {
          type: 'string',
          description: 'Filter by status',
          enum: ['open', 'in-progress', 'resolved', 'wont-fix', 'duplicate'],
        },
        priority: {
          type: 'string',
          description: 'Filter by priority',
          enum: ['critical', 'high', 'medium', 'low'],
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_issues_critical',
    description: 'List critical issues that need immediate attention',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_sync',
    description: 'Synchronize AI context from git and other sources',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'ai_task_start',
    description: 'Start tracking a new task',
    parameters: {
      type: 'object',
      properties: {
        type: {
          type: 'string',
          description: 'Type of task',
          enum: ['feature', 'bugfix', 'refactor', 'docs', 'test', 'devops'],
        },
        scope: {
          type: 'string',
          description: 'Comma-separated scope areas',
        },
        description: {
          type: 'string',
          description: 'Task description',
        },
      },
      required: ['type', 'description'],
    },
  },
  {
    name: 'ai_task_complete',
    description: 'Mark current task as complete',
    parameters: {
      type: 'object',
      properties: {
        success: {
          type: 'string',
          description: 'Whether task was successful',
          enum: ['true', 'false'],
        },
        filesChanged: {
          type: 'string',
          description: 'Comma-separated list of changed files',
        },
        notes: {
          type: 'string',
          description: 'Completion notes',
        },
      },
      required: ['success'],
    },
  },
  {
    name: 'ai_context',
    description: 'Get AI context for a specific task type',
    parameters: {
      type: 'object',
      properties: {
        taskType: {
          type: 'string',
          description: 'Type of task',
          enum: ['feature', 'bugfix', 'refactor', 'docs', 'test', 'devops'],
        },
        scope: {
          type: 'string',
          description: 'Scope area',
        },
      },
      required: [],
    },
  },
  {
    name: 'ai_metrics',
    description: 'Get AI effectiveness metrics',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
];

// ============================================================================
// Resources
// ============================================================================

const RESOURCES: MCPResource[] = [
  {
    uri: 'ai://context/current',
    name: 'Current AI Context',
    description: 'Current AI orchestration context and configuration',
    mimeType: 'application/json',
  },
  {
    uri: 'ai://metrics/dashboard',
    name: 'Metrics Dashboard',
    description: 'AI effectiveness metrics and statistics',
    mimeType: 'application/json',
  },
  {
    uri: 'ai://compliance/report',
    name: 'Compliance Report',
    description: 'Latest compliance scan report',
    mimeType: 'application/json',
  },
  {
    uri: 'ai://security/report',
    name: 'Security Report',
    description: 'Latest security scan report',
    mimeType: 'application/json',
  },
];

// ============================================================================
// Tool Execution
// ============================================================================

import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

function executeTool(name: string, params: Record<string, unknown>): unknown {
  try {
    switch (name) {
      case 'ai_compliance_check': {
        const files = params.files ? String(params.files).split(',') : [];
        const output = execSync(`npm run ai:compliance check ${files.join(' ')}`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_compliance_score': {
        const reportPath = path.join(AI_DIR, 'compliance-report.json');
        if (fs.existsSync(reportPath)) {
          const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
          return { success: true, score: report.overallScore, grade: report.grade };
        }
        return { success: false, message: 'No compliance report found' };
      }

      case 'ai_security_scan': {
        const scanType = params.type || 'full';
        const cmd = scanType === 'full' ? 'ai:security:scan' : `ai:security:${scanType}`;
        const output = execSync(`npm run ${cmd}`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_cache_stats': {
        const output = execSync('npm run ai:cache:stats', {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_cache_clear': {
        const layer = params.layer ? String(params.layer) : '';
        const output = execSync(`npm run ai:cache:clear ${layer}`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_monitor_status': {
        const output = execSync('npm run ai:monitor:status', {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_errors_list': {
        const severity = params.severity ? String(params.severity) : '';
        const output = execSync(`npm run ai:errors:list ${severity}`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_errors_stats': {
        const output = execSync('npm run ai:errors:stats', {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_issues_list': {
        const category = params.category ? String(params.category) : '';
        const status = params.status ? String(params.status) : '';
        const output = execSync(`npm run ai:issues:list ${category} ${status}`.trim(), {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_issues_critical': {
        const output = execSync('npm run ai:issues:critical', {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_sync': {
        const output = execSync('npm run ai:sync', {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_task_start': {
        const type = String(params.type);
        const scope = params.scope || '';
        const desc = String(params.description);
        const output = execSync(`npm run ai:start ${type} ${scope} "${desc}"`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_task_complete': {
        const success = params.success === 'true';
        const files = params.filesChanged || '';
        const notes = params.notes || '';
        const output = execSync(`npm run ai:complete ${success} "${files}" 0 0 0 "${notes}"`, {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_context': {
        const taskType = params.taskType || 'feature';
        const scope = params.scope || '';
        const output = execSync(`npm run ai:context ${taskType} ${scope}`.trim(), {
          cwd: ROOT,
          encoding: 'utf8',
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        return { success: true, output };
      }

      case 'ai_metrics': {
        const metricsPath = path.join(AI_DIR, 'metrics.json');
        if (fs.existsSync(metricsPath)) {
          const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
          return { success: true, metrics };
        }
        return { success: false, message: 'No metrics found' };
      }

      default:
        return { success: false, error: `Unknown tool: ${name}` };
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// ============================================================================
// Resource Reading
// ============================================================================

function readResource(uri: string): { content: string; mimeType: string } | null {
  try {
    switch (uri) {
      case 'ai://context/current': {
        const contextPath = path.join(AI_DIR, 'context.yaml');
        if (fs.existsSync(contextPath)) {
          return {
            content: fs.readFileSync(contextPath, 'utf8'),
            mimeType: 'text/yaml',
          };
        }
        break;
      }

      case 'ai://metrics/dashboard': {
        const metricsPath = path.join(AI_DIR, 'metrics.json');
        if (fs.existsSync(metricsPath)) {
          return {
            content: fs.readFileSync(metricsPath, 'utf8'),
            mimeType: 'application/json',
          };
        }
        break;
      }

      case 'ai://compliance/report': {
        const reportPath = path.join(AI_DIR, 'compliance-report.json');
        if (fs.existsSync(reportPath)) {
          return {
            content: fs.readFileSync(reportPath, 'utf8'),
            mimeType: 'application/json',
          };
        }
        break;
      }

      case 'ai://security/report': {
        const reportPath = path.join(AI_DIR, 'security-report.json');
        if (fs.existsSync(reportPath)) {
          return {
            content: fs.readFileSync(reportPath, 'utf8'),
            mimeType: 'application/json',
          };
        }
        break;
      }
    }
  } catch {
    // Fall through
  }

  return null;
}

// ============================================================================
// MCP Request Handler
// ============================================================================

async function handleRequest(request: MCPRequest): Promise<MCPResponse> {
  const { id, method, params } = request;

  switch (method) {
    case 'initialize':
      return {
        jsonrpc: '2.0',
        id,
        result: {
          protocolVersion: '2024-11-05',
          serverInfo: {
            name: 'ai-tools-mcp',
            version: '1.0.0',
          },
          capabilities: {
            tools: {},
            resources: {},
          },
        },
      };

    case 'tools/list':
      return {
        jsonrpc: '2.0',
        id,
        result: {
          tools: TOOLS.map((t) => ({
            name: t.name,
            description: t.description,
            inputSchema: t.parameters,
          })),
        },
      };

    case 'tools/call': {
      const toolName = (params as { name: string })?.name;
      const toolParams = (params as { arguments?: Record<string, unknown> })?.arguments || {};

      // Use async execution for expensive operations
      let result: unknown;

      if (['ai_compliance_check', 'ai_security_scan', 'ai_cache_clear'].includes(toolName)) {
        result = await executeToolAsync(toolName, toolParams);
      } else if (toolName === 'ai_cache_stats') {
        // Special handling for cache statistics
        result = {
          success: true,
          cacheSize: toolCache.size(),
          timestamp: new Date().toISOString(),
        };
      } else {
        result = executeTool(toolName, toolParams);
      }

      return {
        jsonrpc: '2.0',
        id,
        result: {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        },
      };
    }

    case 'resources/list':
      return {
        jsonrpc: '2.0',
        id,
        result: {
          resources: RESOURCES,
        },
      };

    case 'resources/read': {
      const uri = (params as { uri: string })?.uri;
      const resource = readResource(uri);
      if (resource) {
        return {
          jsonrpc: '2.0',
          id,
          result: {
            contents: [
              {
                uri,
                mimeType: resource.mimeType,
                text: resource.content,
              },
            ],
          },
        };
      }
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32602,
          message: `Resource not found: ${uri}`,
        },
      };
    }

    default:
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32601,
          message: `Method not found: ${method}`,
        },
      };
  }
}

// ============================================================================
// HTTP Server
// ============================================================================

const PORT = parseInt(process.env.MCP_PORT || '3100', 10);

function parseBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Method not allowed' }));
    return;
  }

  try {
    const body = await parseBody(req);
    const request: MCPRequest = JSON.parse(body);
    const response = await handleRequest(request);

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(response));
  } catch (error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: 'Parse error',
          data: error instanceof Error ? error.message : String(error),
        },
      })
    );
  }
});

// ============================================================================
// CLI
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'start':
      server.listen(PORT, () => {
        console.log(`
╔══════════════════════════════════════════════════════════════╗
║            🔌 MCP SERVER - AI TOOLS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at http://localhost:${PORT}                    ║
║                                                              ║
║  Available Tools: ${TOOLS.length.toString().padEnd(43)}║
║  Available Resources: ${RESOURCES.length.toString().padEnd(39)}║
║                                                              ║
║  Endpoints:                                                  ║
║    POST /  - JSON-RPC 2.0 endpoint                           ║
║                                                              ║
║  Methods:                                                    ║
║    initialize    - Initialize connection                     ║
║    tools/list    - List available tools                      ║
║    tools/call    - Execute a tool                            ║
║    resources/list - List available resources                 ║
║    resources/read - Read a resource                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        `);
      });
      break;

    case 'tools':
      console.log('\n📦 Available MCP Tools:\n');
      for (const tool of TOOLS) {
        console.log(`  ${tool.name}`);
        console.log(`    ${tool.description}\n`);
      }
      break;

    case 'resources':
      console.log('\n📁 Available MCP Resources:\n');
      for (const resource of RESOURCES) {
        console.log(`  ${resource.uri}`);
        console.log(`    ${resource.description}\n`);
      }
      break;

    default:
      console.log(`
MCP Server for AI Tools

Commands:
  start       Start the MCP server
  tools       List available tools
  resources   List available resources

Environment Variables:
  MCP_PORT    Server port (default: 3100)

Example:
  npm run ai:mcp start
  MCP_PORT=3200 npm run ai:mcp start
      `);
  }
}

main();
