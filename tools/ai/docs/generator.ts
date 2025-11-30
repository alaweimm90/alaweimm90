#!/usr/bin/env npx tsx
/**
 * AI Tools Documentation Generator
 * Generates comprehensive markdown documentation for all AI tools
 */

import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, 'docs', 'ai-tools');

// ============================================================================
// Tool Metadata
// ============================================================================

interface ToolDoc {
  name: string;
  description: string;
  category: 'core' | 'infrastructure' | 'governance';
  commands: CommandDoc[];
  exports?: string[];
  configuration?: ConfigDoc[];
  examples?: ExampleDoc[];
}

interface CommandDoc {
  name: string;
  script: string;
  description: string;
  args?: ArgDoc[];
}

interface ArgDoc {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default?: string;
}

interface ConfigDoc {
  name: string;
  type: string;
  description: string;
  default: string;
}

interface ExampleDoc {
  title: string;
  code: string;
  description?: string;
}

// ============================================================================
// Tool Definitions
// ============================================================================

const TOOLS: ToolDoc[] = [
  {
    name: 'Orchestrator',
    description: 'Task management and context injection for AI-assisted development',
    category: 'core',
    commands: [
      {
        name: 'start',
        script: 'npm run ai:start',
        description: 'Start tracking a new task',
        args: [
          { name: 'type', type: 'string', description: 'Task type (feature, bugfix, refactor, docs, test, devops)', required: true },
          { name: 'scope', type: 'string', description: 'Comma-separated scope areas', required: false },
          { name: 'description', type: 'string', description: 'Task description', required: true },
        ],
      },
      {
        name: 'complete',
        script: 'npm run ai:complete',
        description: 'Mark current task as complete',
        args: [
          { name: 'success', type: 'boolean', description: 'Whether task was successful', required: true },
          { name: 'filesChanged', type: 'string', description: 'Comma-separated list of changed files', required: false },
          { name: 'linesAdded', type: 'number', description: 'Lines of code added', required: false, default: '0' },
          { name: 'linesRemoved', type: 'number', description: 'Lines of code removed', required: false, default: '0' },
          { name: 'testsAdded', type: 'number', description: 'Number of tests added', required: false, default: '0' },
          { name: 'notes', type: 'string', description: 'Completion notes', required: false },
        ],
      },
      { name: 'context', script: 'npm run ai:context', description: 'Get AI context for a task type' },
      { name: 'metrics', script: 'npm run ai:metrics', description: 'View or update AI effectiveness metrics' },
      { name: 'history', script: 'npm run ai:history', description: 'View task history' },
    ],
    examples: [
      {
        title: 'Start a feature task',
        code: 'npm run ai:start feature auth,api "Add OAuth authentication"',
      },
      {
        title: 'Complete a task',
        code: 'npm run ai:complete true "src/auth.ts,src/api.ts" 150 20 5 "Added OAuth flow"',
      },
    ],
  },
  {
    name: 'Sync',
    description: 'Context synchronization from git and other sources',
    category: 'core',
    commands: [
      { name: 'sync', script: 'npm run ai:sync', description: 'Synchronize all context sources' },
    ],
    examples: [
      { title: 'Run sync', code: 'npm run ai:sync' },
    ],
  },
  {
    name: 'Dashboard',
    description: 'ASCII metrics dashboard for AI effectiveness visualization',
    category: 'core',
    commands: [
      { name: 'dashboard', script: 'npm run ai:dashboard', description: 'Display ASCII metrics dashboard' },
    ],
    examples: [
      { title: 'View dashboard', code: 'npm run ai:dashboard' },
    ],
  },
  {
    name: 'Cache',
    description: 'Multi-layer caching with semantic similarity for AI operations',
    category: 'infrastructure',
    commands: [
      { name: 'cache', script: 'npm run ai:cache', description: 'Cache management CLI' },
      { name: 'stats', script: 'npm run ai:cache:stats', description: 'Show cache statistics' },
      { name: 'clear', script: 'npm run ai:cache:clear', description: 'Clear cache entries',
        args: [{ name: 'layer', type: 'string', description: 'Cache layer to clear (semantic, template, result, analysis)', required: false }] },
    ],
    exports: ['cache'],
    configuration: [
      { name: 'maxEntries', type: 'number', description: 'Maximum cache entries', default: '1000' },
      { name: 'maxSizeBytes', type: 'number', description: 'Maximum cache size in bytes', default: '52428800' },
      { name: 'defaultTtlMs', type: 'number', description: 'Default TTL in milliseconds', default: '3600000' },
      { name: 'enableSemanticSimilarity', type: 'boolean', description: 'Enable semantic matching', default: 'true' },
      { name: 'similarityThreshold', type: 'number', description: 'Similarity threshold (0-1)', default: '0.85' },
    ],
    examples: [
      { title: 'View cache stats', code: 'npm run ai:cache:stats' },
      { title: 'Clear semantic layer', code: 'npm run ai:cache:clear semantic' },
    ],
  },
  {
    name: 'Monitor',
    description: 'Continuous monitoring with circuit breakers for fault tolerance',
    category: 'infrastructure',
    commands: [
      { name: 'monitor', script: 'npm run ai:monitor', description: 'Monitor CLI' },
      { name: 'status', script: 'npm run ai:monitor:status', description: 'Show monitor status and circuit breaker states' },
      { name: 'check', script: 'npm run ai:monitor:check', description: 'Check for changes and trigger actions' },
    ],
    exports: ['monitor', 'circuitBreaker'],
    configuration: [
      { name: 'debounceMs', type: 'number', description: 'Debounce time for changes', default: '2000' },
      { name: 'maxFrequencyMs', type: 'number', description: 'Minimum time between triggers', default: '30000' },
      { name: 'failureThreshold', type: 'number', description: 'Circuit breaker failure threshold', default: '3' },
      { name: 'resetTimeoutMs', type: 'number', description: 'Circuit breaker reset timeout', default: '60000' },
    ],
    examples: [
      { title: 'Check monitor status', code: 'npm run ai:monitor:status' },
    ],
  },
  {
    name: 'Compliance',
    description: 'Policy-based validation with quantitative scoring and recommendations',
    category: 'governance',
    commands: [
      { name: 'compliance', script: 'npm run ai:compliance', description: 'Compliance CLI' },
      { name: 'check', script: 'npm run ai:compliance:check', description: 'Run compliance check on files',
        args: [{ name: 'files', type: 'string[]', description: 'Files to check', required: false }] },
      { name: 'score', script: 'npm run ai:compliance:score', description: 'Quick compliance score check' },
    ],
    exports: ['compliance'],
    examples: [
      { title: 'Run compliance check', code: 'npm run ai:compliance:check src/api.ts' },
      { title: 'Quick score', code: 'npm run ai:compliance:score' },
    ],
  },
  {
    name: 'Telemetry',
    description: 'Observability and alerting for AI operations',
    category: 'infrastructure',
    commands: [
      { name: 'telemetry', script: 'npm run ai:telemetry', description: 'Telemetry CLI' },
      { name: 'status', script: 'npm run ai:telemetry:status', description: 'Show telemetry status' },
      { name: 'alerts', script: 'npm run ai:telemetry:alerts', description: 'Show active alerts' },
    ],
    exports: ['telemetry'],
    examples: [
      { title: 'Check telemetry status', code: 'npm run ai:telemetry:status' },
    ],
  },
  {
    name: 'Errors',
    description: 'Structured error handling with automatic recovery strategies',
    category: 'infrastructure',
    commands: [
      { name: 'errors', script: 'npm run ai:errors', description: 'Error handler CLI' },
      { name: 'list', script: 'npm run ai:errors:list', description: 'List recent errors',
        args: [{ name: 'severity', type: 'string', description: 'Filter by severity (low, medium, high, critical)', required: false }] },
      { name: 'stats', script: 'npm run ai:errors:stats', description: 'Show error statistics' },
    ],
    exports: ['errorHandler', 'ErrorCodes', 'AIOperationError'],
    examples: [
      { title: 'List critical errors', code: 'npm run ai:errors:list critical' },
      { title: 'View error stats', code: 'npm run ai:errors:stats' },
    ],
  },
  {
    name: 'Security',
    description: 'Security scanning for secrets, vulnerabilities, and license compliance',
    category: 'governance',
    commands: [
      { name: 'security', script: 'npm run ai:security', description: 'Security scanner CLI' },
      { name: 'scan', script: 'npm run ai:security:scan', description: 'Run full security scan' },
      { name: 'secrets', script: 'npm run ai:security:secrets', description: 'Scan for exposed secrets' },
      { name: 'vulns', script: 'npm run ai:security:vulns', description: 'Scan for vulnerabilities' },
    ],
    exports: ['securityScanner'],
    examples: [
      { title: 'Run full scan', code: 'npm run ai:security:scan' },
      { title: 'Check for secrets', code: 'npm run ai:security:secrets src/' },
    ],
  },
  {
    name: 'Issues',
    description: 'Automated issue management and tracking',
    category: 'governance',
    commands: [
      { name: 'issues', script: 'npm run ai:issues', description: 'Issue manager CLI' },
      { name: 'list', script: 'npm run ai:issues:list', description: 'List tracked issues' },
      { name: 'critical', script: 'npm run ai:issues:critical', description: 'List critical issues' },
      { name: 'stats', script: 'npm run ai:issues:stats', description: 'Show issue statistics' },
    ],
    exports: ['issueManager'],
    examples: [
      { title: 'List critical issues', code: 'npm run ai:issues:critical' },
      { title: 'View issue stats', code: 'npm run ai:issues:stats' },
    ],
  },
];

// ============================================================================
// Document Generators
// ============================================================================

function generateToolDoc(tool: ToolDoc): string {
  let doc = `# ${tool.name}\n\n`;
  doc += `> ${tool.description}\n\n`;
  doc += `**Category:** ${tool.category}\n\n`;

  // Commands section
  doc += `## Commands\n\n`;
  for (const cmd of tool.commands) {
    doc += `### \`${cmd.script}\`\n\n`;
    doc += `${cmd.description}\n\n`;
    if (cmd.args && cmd.args.length > 0) {
      doc += `**Arguments:**\n\n`;
      doc += `| Name | Type | Required | Description | Default |\n`;
      doc += `|------|------|----------|-------------|---------|\n`;
      for (const arg of cmd.args) {
        doc += `| ${arg.name} | ${arg.type} | ${arg.required ? 'Yes' : 'No'} | ${arg.description} | ${arg.default || '-'} |\n`;
      }
      doc += '\n';
    }
  }

  // Configuration section
  if (tool.configuration && tool.configuration.length > 0) {
    doc += `## Configuration\n\n`;
    doc += `| Option | Type | Description | Default |\n`;
    doc += `|--------|------|-------------|---------|\n`;
    for (const config of tool.configuration) {
      doc += `| ${config.name} | ${config.type} | ${config.description} | ${config.default} |\n`;
    }
    doc += '\n';
  }

  // Exports section
  if (tool.exports && tool.exports.length > 0) {
    doc += `## Exports\n\n`;
    doc += `\`\`\`typescript\n`;
    doc += `import { ${tool.exports.join(', ')} } from 'tools/ai/${tool.name.toLowerCase()}';\n`;
    doc += `\`\`\`\n\n`;
  }

  // Examples section
  if (tool.examples && tool.examples.length > 0) {
    doc += `## Examples\n\n`;
    for (const example of tool.examples) {
      doc += `### ${example.title}\n\n`;
      if (example.description) {
        doc += `${example.description}\n\n`;
      }
      doc += `\`\`\`bash\n${example.code}\n\`\`\`\n\n`;
    }
  }

  return doc;
}

function generateIndexDoc(): string {
  let doc = `# AI Tools Documentation\n\n`;
  doc += `> Enterprise-grade AI orchestration, monitoring, and compliance tools\n\n`;

  doc += `## Overview\n\n`;
  doc += `The AI Tools suite provides comprehensive capabilities for AI-assisted development:\n\n`;
  doc += `- **Task Orchestration**: Track and manage AI-assisted tasks\n`;
  doc += `- **Multi-layer Caching**: Semantic similarity-based caching\n`;
  doc += `- **Circuit Breakers**: Fault tolerance for AI operations\n`;
  doc += `- **Compliance Scoring**: Policy-based validation\n`;
  doc += `- **Security Scanning**: Secrets, vulnerabilities, licenses\n`;
  doc += `- **Issue Management**: Automated issue tracking\n`;
  doc += `- **Full Observability**: Telemetry and alerting\n\n`;

  doc += `## Quick Start\n\n`;
  doc += `\`\`\`bash\n`;
  doc += `# View all available tools\n`;
  doc += `npm run ai\n\n`;
  doc += `# View metrics dashboard\n`;
  doc += `npm run ai:dashboard\n\n`;
  doc += `# Run compliance check\n`;
  doc += `npm run ai:compliance:score\n\n`;
  doc += `# Run security scan\n`;
  doc += `npm run ai:security:scan\n\n`;
  doc += `# Sync context\n`;
  doc += `npm run ai:sync\n`;
  doc += `\`\`\`\n\n`;

  doc += `## Tools by Category\n\n`;

  const categories = {
    core: 'Core Tools',
    infrastructure: 'Infrastructure Tools',
    governance: 'Governance Tools',
  };

  for (const [cat, title] of Object.entries(categories)) {
    const catTools = TOOLS.filter((t) => t.category === cat);
    doc += `### ${title}\n\n`;
    doc += `| Tool | Description |\n`;
    doc += `|------|-------------|\n`;
    for (const tool of catTools) {
      doc += `| [${tool.name}](./${tool.name.toLowerCase()}.md) | ${tool.description} |\n`;
    }
    doc += '\n';
  }

  doc += `## Integration\n\n`;
  doc += `### MCP Server\n\n`;
  doc += `Start the MCP server for AI assistant integration:\n\n`;
  doc += `\`\`\`bash\nnpm run ai:mcp:start\n\`\`\`\n\n`;

  doc += `### REST API\n\n`;
  doc += `Start the REST API server:\n\n`;
  doc += `\`\`\`bash\nnpm run ai:api:start\n\`\`\`\n\n`;

  doc += `### VS Code Extension\n\n`;
  doc += `Import the VS Code integration module in your extension:\n\n`;
  doc += `\`\`\`typescript\n`;
  doc += `import { commands, getStatusBarItems, getTreeViewData } from 'tools/ai/vscode/integration';\n`;
  doc += `\`\`\`\n\n`;

  return doc;
}

function generateApiDoc(): string {
  let doc = `# REST API Reference\n\n`;
  doc += `> HTTP REST API for AI Tools\n\n`;

  doc += `## Base URL\n\n`;
  doc += `\`\`\`\nhttp://localhost:3200\n\`\`\`\n\n`;

  const endpoints = [
    { method: 'GET', path: '/health', desc: 'Health check' },
    { method: 'GET', path: '/compliance/score', desc: 'Get compliance report' },
    { method: 'POST', path: '/compliance/check', desc: 'Run compliance check' },
    { method: 'GET', path: '/security/report', desc: 'Get security report' },
    { method: 'POST', path: '/security/scan', desc: 'Run security scan' },
    { method: 'GET', path: '/cache/stats', desc: 'Get cache statistics' },
    { method: 'DELETE', path: '/cache', desc: 'Clear cache' },
    { method: 'GET', path: '/monitor/status', desc: 'Get monitor status' },
    { method: 'GET', path: '/errors', desc: 'List errors' },
    { method: 'GET', path: '/issues', desc: 'List issues' },
    { method: 'GET', path: '/metrics', desc: 'Get AI metrics' },
    { method: 'POST', path: '/sync', desc: 'Sync context' },
    { method: 'GET', path: '/dashboard', desc: 'Get ASCII dashboard' },
  ];

  doc += `## Endpoints\n\n`;
  doc += `| Method | Path | Description |\n`;
  doc += `|--------|------|-------------|\n`;
  for (const ep of endpoints) {
    doc += `| ${ep.method} | ${ep.path} | ${ep.desc} |\n`;
  }
  doc += '\n';

  doc += `## Examples\n\n`;
  doc += `### Get Compliance Score\n\n`;
  doc += `\`\`\`bash\ncurl http://localhost:3200/compliance/score\n\`\`\`\n\n`;

  doc += `### Run Security Scan\n\n`;
  doc += `\`\`\`bash\ncurl -X POST http://localhost:3200/security/scan\n\`\`\`\n\n`;

  doc += `### List Critical Issues\n\n`;
  doc += `\`\`\`bash\ncurl http://localhost:3200/issues/critical\n\`\`\`\n\n`;

  return doc;
}

// ============================================================================
// Main Generator
// ============================================================================

function generateDocs(): void {
  // Ensure docs directory exists
  if (!fs.existsSync(DOCS_DIR)) {
    fs.mkdirSync(DOCS_DIR, { recursive: true });
  }

  // Generate index
  fs.writeFileSync(path.join(DOCS_DIR, 'README.md'), generateIndexDoc());
  console.log('✅ Generated: README.md');

  // Generate individual tool docs
  for (const tool of TOOLS) {
    const filename = `${tool.name.toLowerCase()}.md`;
    fs.writeFileSync(path.join(DOCS_DIR, filename), generateToolDoc(tool));
    console.log(`✅ Generated: ${filename}`);
  }

  // Generate API reference
  fs.writeFileSync(path.join(DOCS_DIR, 'api.md'), generateApiDoc());
  console.log('✅ Generated: api.md');

  console.log(`\n📚 Documentation generated in ${DOCS_DIR}`);
}

// ============================================================================
// CLI
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'generate':
      generateDocs();
      break;

    case 'list':
      console.log('\n📚 Available Tools:\n');
      for (const tool of TOOLS) {
        console.log(`  ${tool.name} (${tool.category})`);
        console.log(`    ${tool.description}\n`);
      }
      break;

    default:
      console.log(`
AI Tools Documentation Generator

Commands:
  generate    Generate all documentation
  list        List all documented tools

Output:
  Documentation is generated to docs/ai-tools/

Files Generated:
  - README.md       - Main index
  - <tool>.md       - Individual tool docs
  - api.md          - REST API reference
      `);
  }
}

main();
