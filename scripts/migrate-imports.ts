#!/usr/bin/env tsx

/**
 * Script to migrate relative imports to path aliases
 *
 * This script will:
 * 1. Find all TypeScript files with relative imports
 * 2. Analyze the import paths to determine the correct alias
 * 3. Replace the relative imports with path aliases
 */

import * as fs from 'fs';
import * as path from 'path';
import { glob } from 'glob';

// Mapping of directory patterns to their aliases
const ALIAS_MAP: Record<string, string> = {
  'tools/atlas': '@atlas',
  'tools/ai': '@ai',
  'tools/lib': '@lib',
  'tools/devops': '@devops',
  'tools/cli': '@cli',
  'automation': '@automation',
  'types': '@types',
  'config': '@config',
  'tests': '@test',
  '.metaHub': '@metaHub'
};

interface ImportReplacement {
  file: string;
  original: string;
  replacement: string;
  line: number;
}

function getAliasForPath(fromFile: string, importPath: string): string | null {
  // Resolve the absolute path of the import
  const fileDir = path.dirname(fromFile);
  const resolvedPath = path.resolve(fileDir, importPath);
  const relativePath = path.relative(process.cwd(), resolvedPath).replace(/\\/g, '/');

  // Find matching alias
  for (const [dir, alias] of Object.entries(ALIAS_MAP)) {
    if (relativePath.startsWith(dir)) {
      const subPath = relativePath.substring(dir.length);
      return `${alias}${subPath}`;
    }
  }

  return null;
}

function processFile(filePath: string): ImportReplacement[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const replacements: ImportReplacement[] = [];

  const importRegex = /from\s+['"](\.\.[^'"]+)['"]/g;

  lines.forEach((line, index) => {
    const matches = [...line.matchAll(importRegex)];
    matches.forEach(match => {
      const originalImport = match[1];
      const aliasPath = getAliasForPath(filePath, originalImport);

      if (aliasPath) {
        replacements.push({
          file: filePath,
          original: match[0],
          replacement: `from '${aliasPath}'`,
          line: index + 1
        });
      }
    });
  });

  return replacements;
}

function applyReplacements(replacements: ImportReplacement[]): void {
  const fileGroups = new Map<string, ImportReplacement[]>();

  // Group replacements by file
  replacements.forEach(r => {
    if (!fileGroups.has(r.file)) {
      fileGroups.set(r.file, []);
    }
    fileGroups.get(r.file)!.push(r);
  });

  // Apply replacements to each file
  fileGroups.forEach((reps, filePath) => {
    let content = fs.readFileSync(filePath, 'utf-8');

    // Sort replacements by position (reverse order to maintain positions)
    reps.sort((a, b) => b.line - a.line);

    // Apply each replacement
    reps.forEach(rep => {
      content = content.replace(rep.original, rep.replacement);
    });

    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`✅ Updated ${filePath} (${reps.length} imports)`);
  });
}

async function main() {
  console.log('🔄 Starting import migration to path aliases...\n');

  // Find all TypeScript files
  const files = await glob('tools/**/*.ts', {
    ignore: ['**/node_modules/**', '**/dist/**', '**/*.test.ts', '**/*.spec.ts']
  });

  console.log(`Found ${files.length} TypeScript files to analyze\n`);

  const allReplacements: ImportReplacement[] = [];

  // Process each file
  files.forEach(file => {
    const replacements = processFile(file);
    if (replacements.length > 0) {
      allReplacements.push(...replacements);
    }
  });

  if (allReplacements.length === 0) {
    console.log('✨ No relative imports found to migrate!');
    return;
  }

  console.log(`\n📊 Found ${allReplacements.length} imports to migrate in ${new Set(allReplacements.map(r => r.file)).size} files\n`);

  // Show preview
  console.log('Preview of changes:');
  allReplacements.slice(0, 5).forEach(r => {
    console.log(`  ${path.basename(r.file)}:${r.line}`);
    console.log(`    - ${r.original}`);
    console.log(`    + ${r.replacement}`);
  });

  if (allReplacements.length > 5) {
    console.log(`  ... and ${allReplacements.length - 5} more\n`);
  }

  // Apply all replacements
  console.log('\n🚀 Applying replacements...\n');
  applyReplacements(allReplacements);

  console.log(`\n✅ Migration complete! Updated ${new Set(allReplacements.map(r => r.file)).size} files`);
  console.log('\n💡 Next steps:');
  console.log('  1. Run `npm run type-check` to verify no TypeScript errors');
  console.log('  2. Run `npm test` to ensure all tests pass');
  console.log('  3. Commit the changes');
}

// Run the migration
main().catch(error => {
  console.error('❌ Migration failed:', error);
  process.exit(1);
});