#!/usr/bin/env tsx

/**
 * Simple script to migrate relative imports to path aliases
 * Uses only built-in Node.js modules
 */

import * as fs from 'fs';
import * as path from 'path';

// Mapping of directory patterns to their aliases
const ALIAS_MAP: Record<string, string> = {
  'tools/atlas': '@atlas',
  'tools/ai': '@ai',
  'tools/lib': '@lib',
  'tools/devops': '@devops',
  'tools/cli': '@cli',
  automation: '@automation',
  types: '@types',
  config: '@config',
  tests: '@test',
  '.metaHub': '@metaHub',
};

// List of files to migrate (from our grep results)
const FILES_TO_MIGRATE = [
  'tools/cli/devops.ts',
  'tools/atlas/cli/workflow-run.ts',
  'tools/atlas/orchestration/workflows.ts',
  'tools/atlas/cli/workflow-plan.ts',
  'tools/atlas/orchestration/router.ts',
  'tools/atlas/cli/utils.ts',
  'tools/atlas/cli/commands.ts',
  'tools/atlas/orchestration/index.ts',
  'tools/atlas/orchestration/fallback.ts',
  'tools/atlas/orchestration/executor.ts',
  'tools/atlas/adapters/openai.ts',
  'tools/atlas/adapters/google.ts',
  'tools/atlas/adapters/anthropic.ts',
  'tools/atlas/adapters/base.ts',
  'tools/atlas/orchestration/devops-agents.ts',
  'tools/atlas/api/router.ts',
  'tools/ai/cli/security-cli.ts',
  'tools/ai/cli/issues-cli.ts',
  'tools/ai/cli/compliance-cli.ts',
  'tools/atlas/services/index.ts',
  'tools/atlas/services/optimizer.ts',
  'tools/atlas/services/monitor.ts',
  'tools/atlas/refactoring/engine.ts',
  'tools/atlas/services/dashboard.ts',
];

function getAliasForPath(fromFile: string, importPath: string): string | null {
  // Handle the import path
  const fileDir = path.dirname(fromFile);

  // Count how many ../ we have
  const upCount = (importPath.match(/\.\.\//g) || []).length;

  // Get the base path after removing ../
  const basePath = importPath.replace(/^(\.\.\/)+/, '');

  // Navigate up from current file directory
  let currentDir = fileDir;
  for (let i = 0; i < upCount; i++) {
    currentDir = path.dirname(currentDir);
  }

  // Construct the full path
  const fullPath = path.join(currentDir, basePath).replace(/\\/g, '/');

  // Find matching alias
  for (const [dir, alias] of Object.entries(ALIAS_MAP)) {
    if (fullPath.includes(dir)) {
      const parts = fullPath.split(dir);
      const subPath = parts[1] || '';
      return `${alias}${subPath}`;
    }
  }

  return null;
}

function migrateFile(filePath: string): number {
  const fullPath = path.join(process.cwd(), filePath);

  if (!fs.existsSync(fullPath)) {
    console.log(`⚠️  Skipping ${filePath} (file not found)`);
    return 0;
  }

  const content = fs.readFileSync(fullPath, 'utf-8');
  let updatedContent = content;
  let replacementCount = 0;

  // Match import statements with relative paths
  const importRegex = /from\s+['"](\.\.[^'"]+)['"]/g;

  const replacements: Array<{ original: string; replacement: string }> = [];

  let match;
  while ((match = importRegex.exec(content)) !== null) {
    const originalImport = match[1];
    const aliasPath = getAliasForPath(filePath, originalImport);

    if (aliasPath) {
      replacements.push({
        original: `from '${originalImport}'`,
        replacement: `from '${aliasPath}'`,
      });
    }
  }

  // Apply replacements
  replacements.forEach(({ original, replacement }) => {
    if (updatedContent.includes(original)) {
      updatedContent = updatedContent.replace(original, replacement);
      replacementCount++;
      console.log(`  ✓ ${original} → ${replacement}`);
    }
  });

  if (replacementCount > 0) {
    fs.writeFileSync(fullPath, updatedContent, 'utf-8');
    console.log(`✅ Updated ${filePath} (${replacementCount} imports)\n`);
  }

  return replacementCount;
}

function main(): void {
  console.log('🔄 Starting import migration to path aliases...\n');
  console.log(`Processing ${FILES_TO_MIGRATE.length} files...\n`);

  let totalReplacements = 0;
  let filesUpdated = 0;

  FILES_TO_MIGRATE.forEach((file) => {
    const count = migrateFile(file);
    if (count > 0) {
      totalReplacements += count;
      filesUpdated++;
    }
  });

  console.log('\n' + '='.repeat(50));
  console.log(`✅ Migration complete!`);
  console.log(`📊 Statistics:`);
  console.log(`   - Files updated: ${filesUpdated}`);
  console.log(`   - Imports migrated: ${totalReplacements}`);
  console.log('\n💡 Next steps:');
  console.log('  1. Run `npm run type-check` to verify no TypeScript errors');
  console.log('  2. Run `npm test` to ensure all tests pass');
  console.log('  3. Commit the changes');
}

// Run the migration
try {
  main();
} catch (error) {
  console.error('❌ Migration failed:', error);
  process.exit(1);
}
