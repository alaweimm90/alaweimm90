#!/usr/bin/env npx tsx
/**
 * AI Sync Tool
 * Synchronizes context between AI tools and governance systems
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

interface SyncResult {
  action: string;
  status: 'success' | 'skipped' | 'error';
  details?: string;
}

// Sync recent git changes to context
function syncRecentChanges(): SyncResult {
  try {
    const result = execSync(
      'git log --oneline -20 --pretty=format:\'{"hash":"%h","message":"%s","author":"%an","date":"%ci"}\'',
      {
        encoding: 'utf8',
        cwd: ROOT,
      }
    );

    const changes = result
      .trim()
      .split('\n')
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    const outputPath = path.join(AI_DIR, 'recent-changes.json');
    fs.writeFileSync(
      outputPath,
      JSON.stringify(
        {
          updated_at: new Date().toISOString(),
          changes,
        },
        null,
        2
      )
    );

    return {
      action: 'sync-recent-changes',
      status: 'success',
      details: `Synced ${changes.length} recent commits`,
    };
  } catch (error) {
    return {
      action: 'sync-recent-changes',
      status: 'error',
      details: String(error),
    };
  }
}

// Sync template inventory
function syncTemplates(): SyncResult {
  try {
    const templatesDir = path.join(ROOT, 'templates', 'devops');
    if (!fs.existsSync(templatesDir)) {
      return {
        action: 'sync-templates',
        status: 'skipped',
        details: 'Templates directory not found',
      };
    }

    interface Template {
      name: string;
      category: string;
      version: string;
      path: string;
    }

    const templates: Template[] = [];

    function walk(dir: string): void {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(fullPath);
        } else if (entry.name === 'template.json') {
          try {
            const manifest = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
            templates.push({
              name: manifest.name,
              category: manifest.category,
              version: manifest.version,
              path: path.relative(ROOT, fullPath),
            });
          } catch {
            // Skip invalid manifests
          }
        }
      }
    }

    walk(templatesDir);

    const outputPath = path.join(AI_DIR, 'template-inventory.json');
    fs.writeFileSync(
      outputPath,
      JSON.stringify(
        {
          updated_at: new Date().toISOString(),
          count: templates.length,
          templates,
        },
        null,
        2
      )
    );

    return {
      action: 'sync-templates',
      status: 'success',
      details: `Synced ${templates.length} templates`,
    };
  } catch (error) {
    return {
      action: 'sync-templates',
      status: 'error',
      details: String(error),
    };
  }
}

// Sync workflow inventory
function syncWorkflows(): SyncResult {
  try {
    const workflowsDir = path.join(ROOT, '.github', 'workflows');
    if (!fs.existsSync(workflowsDir)) {
      return {
        action: 'sync-workflows',
        status: 'skipped',
        details: 'Workflows directory not found',
      };
    }

    const workflows = fs
      .readdirSync(workflowsDir)
      .filter((f) => f.endsWith('.yml'))
      .map((f) => ({
        name: f.replace('.yml', ''),
        path: `.github/workflows/${f}`,
      }));

    const outputPath = path.join(AI_DIR, 'workflow-inventory.json');
    fs.writeFileSync(
      outputPath,
      JSON.stringify(
        {
          updated_at: new Date().toISOString(),
          count: workflows.length,
          workflows,
        },
        null,
        2
      )
    );

    return {
      action: 'sync-workflows',
      status: 'success',
      details: `Synced ${workflows.length} workflows`,
    };
  } catch (error) {
    return {
      action: 'sync-workflows',
      status: 'error',
      details: String(error),
    };
  }
}

// Update codemap if needed
function updateCodemap(): SyncResult {
  try {
    execSync('npm run codemap', { cwd: ROOT, stdio: 'pipe' });
    return {
      action: 'update-codemap',
      status: 'success',
      details: 'Codemap updated',
    };
  } catch (error) {
    return {
      action: 'update-codemap',
      status: 'error',
      details: String(error),
    };
  }
}

// Sync governance status
function syncGovernance(): SyncResult {
  try {
    const catalogPath = path.join(ROOT, '.metaHub', 'catalog', 'catalog.json');
    if (!fs.existsSync(catalogPath)) {
      return {
        action: 'sync-governance',
        status: 'skipped',
        details: 'Catalog not found',
      };
    }

    const catalog = JSON.parse(fs.readFileSync(catalogPath, 'utf8'));

    const summary = {
      updated_at: new Date().toISOString(),
      catalog_version: catalog.version,
      catalog_generated: catalog.generated_at,
      organizations: catalog.organizations?.length || 0,
      total_repos:
        catalog.organizations?.reduce(
          (sum: number, org: { repos?: unknown[] }) => sum + (org.repos?.length || 0),
          0
        ) || 0,
    };

    const outputPath = path.join(AI_DIR, 'governance-status.json');
    fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2));

    return {
      action: 'sync-governance',
      status: 'success',
      details: `Synced governance for ${summary.total_repos} repos`,
    };
  } catch (error) {
    return {
      action: 'sync-governance',
      status: 'error',
      details: String(error),
    };
  }
}

// Full sync
function fullSync(): void {
  console.log('🔄 Starting AI context sync...\n');

  const results: SyncResult[] = [
    syncRecentChanges(),
    syncTemplates(),
    syncWorkflows(),
    syncGovernance(),
    updateCodemap(),
  ];

  console.log('Sync Results:');
  console.log('─'.repeat(60));

  for (const result of results) {
    const icon = result.status === 'success' ? '✅' : result.status === 'skipped' ? '⏭️' : '❌';
    console.log(`${icon} ${result.action}: ${result.details || result.status}`);
  }

  console.log('─'.repeat(60));

  const successful = results.filter((r) => r.status === 'success').length;
  const total = results.length;
  console.log(`\n✨ Sync complete: ${successful}/${total} actions successful`);
}

// CLI
const command = process.argv[2];

switch (command) {
  case 'changes':
    console.log(JSON.stringify(syncRecentChanges(), null, 2));
    break;
  case 'templates':
    console.log(JSON.stringify(syncTemplates(), null, 2));
    break;
  case 'workflows':
    console.log(JSON.stringify(syncWorkflows(), null, 2));
    break;
  case 'governance':
    console.log(JSON.stringify(syncGovernance(), null, 2));
    break;
  case 'codemap':
    console.log(JSON.stringify(updateCodemap(), null, 2));
    break;
  case 'all':
  default:
    fullSync();
}
