#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
const fs = require('fs');
const path = require('path');

const MAX_LINES = 500;
const DIRS_TO_CHECK = ['tools/cli', 'tools/lib'];
const EXTENSIONS = ['.ts', '.js', '.py'];

let violations = 0;
let totalFiles = 0;

console.log('📏 KILO File Size Checker');
console.log(`Max lines per file: ${MAX_LINES}`);
console.log(`Checking directories: ${DIRS_TO_CHECK.join(', ')}\n`);

function checkDirectory(dir) {
  if (!fs.existsSync(dir)) {
    console.log(`⚠️  Directory not found: ${dir}`);
    return;
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      checkDirectory(fullPath);
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name);
      if (EXTENSIONS.includes(ext)) {
        totalFiles++;
        const content = fs.readFileSync(fullPath, 'utf-8');
        const lines = content.split('\n').length;

        if (lines > MAX_LINES) {
          console.error(`❌ ${fullPath}: ${lines} lines (max ${MAX_LINES})`);
          violations++;
        } else {
          console.log(`✅ ${fullPath}: ${lines} lines`);
        }
      }
    }
  }
}

// Check each directory
DIRS_TO_CHECK.forEach((dir) => {
  checkDirectory(dir);
});

console.log(`\n📊 Summary:`);
console.log(`   Files checked: ${totalFiles}`);
console.log(`   Violations: ${violations}`);

if (violations > 0) {
  console.error(`\n❌ Found ${violations} file(s) exceeding ${MAX_LINES} lines`);
  console.error('💡 Refactor large files into smaller, focused modules');
  process.exit(1);
}

console.log('\n✅ All files within size limits');
process.exit(0);
