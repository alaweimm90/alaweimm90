#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');

console.log('🔍 Running simple codebase audit...\n');

const stats = {
  totalFiles: 0,
  componentFiles: 0,
  testFiles: 0,
  storyFiles: 0,
  utilFiles: 0,
  hookFiles: 0,
  pageFiles: 0,
  potentiallyUnused: []
};

// Common imports to check
const commonImports = new Map();
const fileContents = new Map();

// Scan all TypeScript/JavaScript files
function scanDirectory(dir) {
  const items = fs.readdirSync(dir);
  
  items.forEach(item => {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory() && !item.includes('node_modules') && !item.startsWith('.')) {
      scanDirectory(itemPath);
    } else if (stat.isFile() && (item.endsWith('.tsx') || item.endsWith('.ts') || item.endsWith('.jsx') || item.endsWith('.js'))) {
      stats.totalFiles++;
      
      // Categorize files
      if (item.includes('.test.') || item.includes('.spec.')) stats.testFiles++;
      else if (item.includes('.stories.')) stats.storyFiles++;
      else if (dir.includes('components')) stats.componentFiles++;
      else if (dir.includes('utils')) stats.utilFiles++;
      else if (dir.includes('hooks')) stats.hookFiles++;
      else if (dir.includes('pages')) stats.pageFiles++;
      
      // Read file content
      try {
        const content = fs.readFileSync(itemPath, 'utf8');
        fileContents.set(itemPath, content);
        
        // Extract imports
        const importRegex = /import\s+(?:{[^}]*}|\*\s+as\s+\w+|\w+)\s+from\s+['"]([^'"]+)['"]/g;
        let match;
        while ((match = importRegex.exec(content)) !== null) {
          const importPath = match[1];
          if (importPath.startsWith('.') || importPath.startsWith('@/')) {
            commonImports.set(importPath, (commonImports.get(importPath) || 0) + 1);
          }
        }
      } catch (error) {
        // Ignore read errors
      }
    }
  });
}

console.log('📂 Scanning src directory...\n');
scanDirectory(srcDir);

// Find potentially unused files
console.log('🔍 Analyzing file usage...\n');

fileContents.forEach((content, filePath) => {
  const fileName = path.basename(filePath, path.extname(filePath));
  const relativePath = path.relative(rootDir, filePath);
  
  // Skip test files, stories, and main entry points
  if (fileName.includes('.test') || 
      fileName.includes('.spec') || 
      fileName.includes('.stories') ||
      fileName === 'App' ||
      fileName === 'main' ||
      fileName === 'index' ||
      fileName === 'vite-env.d') {
    return;
  }
  
  // Check if this file/component is imported anywhere
  let isImported = false;
  
  fileContents.forEach((otherContent, otherPath) => {
    if (otherPath === filePath) return;
    
    // Check various import patterns
    if (otherContent.includes(fileName) || 
        otherContent.includes(`/${fileName}'`) ||
        otherContent.includes(`/${fileName}"`) ||
        otherContent.includes(`${fileName}.tsx`) ||
        otherContent.includes(`${fileName}.ts`)) {
      isImported = true;
    }
  });
  
  if (!isImported && !relativePath.includes('pages/')) {
    stats.potentiallyUnused.push(relativePath);
  }
});

// Analyze old structure
console.log('📊 Current Structure Analysis:');
console.log('─'.repeat(50));
console.log(`Total source files: ${stats.totalFiles}`);
console.log(`Component files: ${stats.componentFiles}`);
console.log(`Test files: ${stats.testFiles}`);
console.log(`Story files: ${stats.storyFiles}`);
console.log(`Utility files: ${stats.utilFiles}`);
console.log(`Hook files: ${stats.hookFiles}`);
console.log(`Page files: ${stats.pageFiles}`);
console.log('─'.repeat(50));

// Show potentially unused files
if (stats.potentiallyUnused.length > 0) {
  console.log(`\n⚠️  Potentially unused files (${stats.potentiallyUnused.length}):`);
  stats.potentiallyUnused.slice(0, 20).forEach(file => {
    console.log(`  - ${file}`);
  });
  if (stats.potentiallyUnused.length > 20) {
    console.log(`  ... and ${stats.potentiallyUnused.length - 20} more`);
  }
}

// Save results
const auditResults = {
  timestamp: new Date().toISOString(),
  stats: stats,
  unusedFiles: stats.potentiallyUnused,
  mostImported: Array.from(commonImports.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([path, count]) => ({ path, count }))
};

fs.writeFileSync(
  path.join(rootDir, 'audit-results.json'),
  JSON.stringify(auditResults, null, 2)
);

console.log('\n✅ Audit complete! Results saved to audit-results.json');

// Recommendations
console.log('\n💡 Recommendations:');
console.log('1. Review potentially unused files before removing');
console.log('2. Consider moving components to feature-based structure');
console.log('3. Consolidate utilities into feature modules');
console.log('4. Set up import aliases for cleaner imports');

process.exit(0);