#!/usr/bin/env npx tsx
/**
 * AI Sync Tool
 * Synchronizes context between AI tools and governance systems
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { loadJson, saveJson } from './utils/file-persistence.js';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');

// Output file paths
const RECENT_CHANGES_PATH = path.join(AI_DIR, 'recent-changes.json');
const TEMPLATE_INVENTORY_PATH = path.join(AI_DIR, 'template-inventory.json');
const WORKFLOW_INVENTORY_PATH = path.join(AI_DIR, 'workflow-inventory.json');
const GOVERNANCE_STATUS_PATH = path.join(AI_DIR, 'governance-status.json');

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

    saveJson(RECENT_CHANGES_PATH, {
      updated_at: new Date().toISOString(),
      changes,
    });

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
    const templatesDir = path.join(ROOT, 'infrastructure', 'templates', 'devops');
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
          const manifest = loadJson<{ name: string; category: string; version: string }>(fullPath);
          if (manifest) {
            templates.push({
              name: manifest.name,
              category: manifest.category,
              version: manifest.version,
              path: path.relative(ROOT, fullPath),
            });
          }
        }
      }
    }

    walk(templatesDir);

    saveJson(TEMPLATE_INVENTORY_PATH, {
      updated_at: new Date().toISOString(),
      count: templates.length,
      templates,
    });

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

    saveJson(WORKFLOW_INVENTORY_PATH, {
      updated_at: new Date().toISOString(),
      count: workflows.length,
      workflows,
    });

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

// Update codemap if needed (currently disabled - codemap script removed)
function updateCodemap(): SyncResult {
  return {
    action: 'update-codemap',
    status: 'skipped',
    details: 'Codemap generation disabled',
  };
}

// Sync governance status
function syncGovernance(): SyncResult {
  try {
    const catalogPath = path.join(ROOT, '.metaHub', 'catalog', 'catalog.json');
    const catalog = loadJson<{
      version: string;
      generated_at: string;
      organizations?: { repos?: unknown[] }[];
    }>(catalogPath);

    if (!catalog) {
      return {
        action: 'sync-governance',
        status: 'skipped',
        details: 'Catalog not found',
      };
    }

    const summary = {
      updated_at: new Date().toISOString(),
      catalog_version: catalog.version,
      catalog_generated: catalog.generated_at,
      organizations: catalog.organizations?.length || 0,
      total_repos:
        catalog.organizations?.reduce((sum: number, org) => sum + (org.repos?.length || 0), 0) || 0,
    };

    saveJson(GOVERNANCE_STATUS_PATH, summary);

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
