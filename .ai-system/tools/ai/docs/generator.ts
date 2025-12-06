#!/usr/bin/env npx tsx
/**
 * AI Tools Documentation Generator
 * Generates comprehensive markdown documentation for all AI tools
 */

import * as fs from 'fs';
import * as path from 'path';

import { TOOLS, ToolDoc } from './tools-metadata.js';

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, 'docs', 'ai-tools');

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
