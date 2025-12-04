/**
 * Find and list structure violations at root level
 */

import * as fs from 'fs';
import * as path from 'path';

const workspacePath = process.cwd();

console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║              STRUCTURE VIOLATION ANALYSIS                    ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

// Allowed patterns from dashboard.ts
const allowedPatterns = [
  /^\./, // Hidden files/folders
  /^package.*\.json$/, // package.json, package-lock.json
  /^tsconfig.*\.json$/, // tsconfig files
  /^README/i, // README files
  /^LICENSE/i, // License files
  /^CHANGELOG/i, // Changelog
  /\.config\.(js|ts|mjs|cjs)$/, // Config files
  /^vitest/, // Vitest config
  /^eslint/, // ESLint config
];

const rootFiles = fs.readdirSync(workspacePath);

console.log('📂 Scanning root directory...\n');

const violations: string[] = [];
const allowed: string[] = [];
const directories: string[] = [];

for (const file of rootFiles) {
  const fullPath = path.join(workspacePath, file);
  const isDir = fs.statSync(fullPath).isDirectory();

  if (isDir) {
    directories.push(file);
  } else if (allowedPatterns.some((p) => p.test(file))) {
    allowed.push(file);
  } else {
    violations.push(file);
  }
}

console.log('✅ ALLOWED FILES (' + allowed.length + ')');
console.log('─'.repeat(50));
for (const f of allowed) {
  console.log(`   ${f}`);
}

console.log('\n📁 DIRECTORIES (' + directories.length + ') - Not counted as violations');
console.log('─'.repeat(50));
for (const d of directories) {
  console.log(`   ${d}/`);
}

console.log('\n❌ STRUCTURE VIOLATIONS (' + violations.length + ')');
console.log('─'.repeat(50));
for (const v of violations) {
  console.log(`   ${v}`);
}

console.log('\n┌─────────────────────────────────────────────────────────────┐');
console.log('│                    RECOMMENDATIONS                          │');
console.log('└─────────────────────────────────────────────────────────────┘\n');

for (const v of violations) {
  const ext = path.extname(v).toLowerCase();
  let recommendation = '';

  if (ext === '.md') {
    recommendation = `Move to docs/ or add to allowedPatterns`;
  } else if (ext === '.ts' || ext === '.js') {
    recommendation = `Move to scripts/ or automation/`;
  } else if (ext === '.yaml' || ext === '.yml') {
    recommendation = `Move to .metaHub/policies/ or automation/workflows/config/`;
  } else if (ext === '.json') {
    recommendation = `Move to config/ or add pattern to allowedPatterns`;
  } else {
    recommendation = `Move to appropriate directory or add to .gitignore`;
  }

  console.log(`📍 ${v}`);
  console.log(`   └─ ${recommendation}\n`);
}

// Calculate score
const score = Math.max(0, 100 - violations.length * 5);
console.log(`\n📊 Current Structure Score: ${score}/100`);
console.log(`   To reach 100: Remove or relocate ${violations.length} files`);
