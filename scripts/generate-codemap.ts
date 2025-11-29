#!/usr/bin/env npx tsx
/**
 * Codemap Generator
 * Auto-generates Mermaid diagrams from codebase structure
 */

import * as fs from 'fs';
import * as path from 'path';

const ROOT = process.cwd();

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
}

interface TemplateManifest {
  name: string;
  category: string;
  version: string;
  description?: string;
}

// Scan directory structure
function scanDirectory(dirPath: string, depth = 2): FileNode | null {
  if (!fs.existsSync(dirPath)) return null;

  const stats = fs.statSync(dirPath);
  const name = path.basename(dirPath);

  if (stats.isFile()) {
    return { name, path: dirPath, type: 'file' };
  }

  if (depth <= 0) {
    return { name, path: dirPath, type: 'directory' };
  }

  const children = fs
    .readdirSync(dirPath)
    .filter((f) => !f.startsWith('.') && f !== 'node_modules')
    .map((f) => scanDirectory(path.join(dirPath, f), depth - 1))
    .filter((n): n is FileNode => n !== null);

  return { name, path: dirPath, type: 'directory', children };
}

// Discover templates
function discoverTemplates(): TemplateManifest[] {
  const templatesDir = path.join(ROOT, 'templates', 'devops');
  if (!fs.existsSync(templatesDir)) return [];

  const templates: TemplateManifest[] = [];

  function walk(dir: string) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.name === 'template.json') {
        try {
          const manifest = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
          templates.push(manifest);
        } catch {
          // Skip invalid manifests
        }
      }
    }
  }

  walk(templatesDir);
  return templates;
}

// Count workflows
function countWorkflows(): number {
  const workflowsDir = path.join(ROOT, '.github', 'workflows');
  if (!fs.existsSync(workflowsDir)) return 0;
  return fs.readdirSync(workflowsDir).filter((f) => f.endsWith('.yml')).length;
}

// Generate template tree mermaid
function generateTemplateTree(templates: TemplateManifest[]): string {
  const byCategory = templates.reduce(
    (acc, t) => {
      acc[t.category] = acc[t.category] || [];
      acc[t.category].push(t);
      return acc;
    },
    {} as Record<string, TemplateManifest[]>
  );

  let mermaid = 'flowchart TD\n';
  mermaid += '    templates[templates/devops/]\n\n';

  for (const [category, categoryTemplates] of Object.entries(byCategory)) {
    const catId = category.replace(/[^a-z0-9]/gi, '');
    mermaid += `    templates --> ${catId}[${category}/]\n`;

    for (const tmpl of categoryTemplates) {
      const tmplId = tmpl.name.replace(/[^a-z0-9]/gi, '');
      mermaid += `    ${catId} --> ${tmplId}[${tmpl.name}]\n`;
    }
  }

  return mermaid;
}

// Generate stats
function generateStats(templates: TemplateManifest[], workflowCount: number): string {
  return `
## Codebase Statistics

| Metric | Count |
|--------|-------|
| Templates | ${templates.length} |
| Workflows | ${workflowCount} |
| Template Categories | ${new Set(templates.map((t) => t.category)).size} |

*Auto-generated on ${new Date().toISOString().split('T')[0]}*
`;
}

// Main
function main() {
  console.log('🗺️  Generating codemap...');

  const templates = discoverTemplates();
  const workflowCount = countWorkflows();

  console.log(`   Found ${templates.length} templates`);
  console.log(`   Found ${workflowCount} workflows`);

  // Scan directory structure for stats
  const toolsTree = scanDirectory(path.join(ROOT, 'tools'), 1);
  const toolsCount = toolsTree?.children?.length ?? 0;
  console.log(`   Found ${toolsCount} tool directories`);

  // Read existing codemap
  const codemapPath = path.join(ROOT, 'docs', 'CODEMAP.md');
  let content = fs.existsSync(codemapPath) ? fs.readFileSync(codemapPath, 'utf8') : '';

  // Generate template tree mermaid
  const templateTree = generateTemplateTree(templates);
  console.log(`   Generated template tree (${templateTree.split('\n').length} lines)`);

  // Update stats section
  const stats = generateStats(templates, workflowCount);
  if (content.includes('## Codebase Statistics')) {
    content = content.replace(
      /## Codebase Statistics[\s\S]*?\*Auto-generated[^*]+\*/,
      stats.trim()
    );
  } else {
    content += '\n' + stats;
  }

  fs.writeFileSync(codemapPath, content);
  console.log('✅ Codemap updated: docs/CODEMAP.md');
}

main();
