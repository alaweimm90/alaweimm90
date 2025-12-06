#!/usr/bin/env node

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🔍 Running comprehensive codebase audit...\n');

// Initialize results
const results = {
  unusedFiles: [],
  unusedExports: [],
  unusedDeps: [],
  timestamp: new Date().toISOString()
};

// 1. Run knip to find unused files
console.log('📋 Finding unused files with knip...');
try {
  // Install knip temporarily if not available
  try {
    execSync('npx knip --version', { stdio: 'ignore' });
  } catch {
    console.log('Installing knip...');
    execSync('npm install -D @knip/cli', { stdio: 'inherit' });
  }

  // Create knip config if doesn't exist
  const knipConfig = {
    entry: ['src/index.tsx', 'src/App.tsx'],
    project: ['src/**/*.{ts,tsx,js,jsx}'],
    ignore: [
      'node_modules/**',
      'dist/**',
      'build/**',
      '.storybook/**',
      '**/*.test.{ts,tsx}',
      '**/*.spec.{ts,tsx}',
      '**/*.stories.{ts,tsx}',
      'vite.config.ts',
      'tailwind.config.ts',
      'postcss.config.js'
    ]
  };

  fs.writeFileSync(
    path.join(rootDir, 'knip.json'),
    JSON.stringify(knipConfig, null, 2)
  );

  const knipOutput = execSync('npx knip --no-progress', { 
    encoding: 'utf8',
    cwd: rootDir 
  });

  // Parse knip output
  const lines = knipOutput.split('\n').filter(line => line.trim());
  lines.forEach(line => {
    if (line.includes('Unused file:') || line.includes('src/')) {
      const match = line.match(/src\/[^\s]+/);
      if (match) {
        results.unusedFiles.push(match[0]);
      }
    }
  });

  console.log(`✅ Found ${results.unusedFiles.length} unused files\n`);
} catch (error) {
  console.log('⚠️  Knip analysis failed, continuing with other tools...\n');
}

// 2. Run ts-prune to find unused exports
console.log('📋 Finding unused exports with ts-prune...');
try {
  // Install ts-prune if needed
  try {
    execSync('npx ts-prune --version', { stdio: 'ignore' });
  } catch {
    console.log('Installing ts-prune...');
    execSync('npm install -D ts-prune', { stdio: 'inherit' });
  }

  const tsPruneOutput = execSync('npx ts-prune --ignore "node_modules|.test.|.spec.|.stories."', {
    encoding: 'utf8',
    cwd: rootDir
  });

  // Parse ts-prune output
  const exportLines = tsPruneOutput.split('\n').filter(line => line.trim());
  exportLines.forEach(line => {
    if (line.includes(':') && !line.includes('node_modules')) {
      results.unusedExports.push(line.trim());
    }
  });

  console.log(`✅ Found ${results.unusedExports.length} unused exports\n`);
} catch (error) {
  console.log('⚠️  ts-prune analysis failed, continuing...\n');
}

// 3. Run depcheck to find unused dependencies
console.log('📋 Finding unused dependencies with depcheck...');
try {
  // Install depcheck if needed
  try {
    execSync('npx depcheck --version', { stdio: 'ignore' });
  } catch {
    console.log('Installing depcheck...');
    execSync('npm install -D depcheck', { stdio: 'inherit' });
  }

  const depcheckOutput = execSync('npx depcheck --json', {
    encoding: 'utf8',
    cwd: rootDir
  });

  const depcheckResult = JSON.parse(depcheckOutput);
  results.unusedDeps = depcheckResult.dependencies || [];

  console.log(`✅ Found ${results.unusedDeps.length} unused dependencies\n`);
} catch (error) {
  console.log('⚠️  depcheck analysis failed, continuing...\n');
}

// 4. Additional custom checks
console.log('📋 Running custom component usage analysis...');

// Find all component files
const componentFiles = [];
function findComponents(dir) {
  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
      findComponents(fullPath);
    } else if (file.endsWith('.tsx') || file.endsWith('.jsx')) {
      componentFiles.push(fullPath);
    }
  });
}

try {
  findComponents(path.join(rootDir, 'src'));
  
  // Check component usage
  componentFiles.forEach(file => {
    const componentName = path.basename(file, path.extname(file));
    const fileContent = fs.readFileSync(file, 'utf8');
    
    // Skip if it's an index file or main entry
    if (componentName === 'index' || componentName === 'App' || componentName === 'main') {
      return;
    }
    
    // Check if component is exported
    if (!fileContent.includes('export')) {
      return;
    }
    
    // Search for imports of this component
    let isUsed = false;
    componentFiles.forEach(otherFile => {
      if (otherFile === file) return;
      
      const otherContent = fs.readFileSync(otherFile, 'utf8');
      if (otherContent.includes(componentName) && 
          (otherContent.includes(`from '${file.replace(rootDir, '').replace(/\\/g, '/')}'`) ||
           otherContent.includes(`from "${file.replace(rootDir, '').replace(/\\/g, '/')}"`) ||
           otherContent.includes(`/${componentName}`))) {
        isUsed = true;
      }
    });
    
    if (!isUsed) {
      const relativePath = path.relative(rootDir, file).replace(/\\/g, '/');
      if (!results.unusedFiles.includes(relativePath)) {
        results.unusedFiles.push(relativePath);
      }
    }
  });
  
  console.log(`✅ Component analysis complete\n`);
} catch (error) {
  console.log('⚠️  Component analysis failed\n');
}

// 5. Save results
console.log('💾 Saving audit results...');

// Save unused files
fs.writeFileSync(
  path.join(rootDir, 'unused-files.json'),
  JSON.stringify(results.unusedFiles, null, 2)
);

// Save unused exports
fs.writeFileSync(
  path.join(rootDir, 'unused-exports.txt'),
  results.unusedExports.join('\n')
);

// Save unused dependencies
fs.writeFileSync(
  path.join(rootDir, 'unused-deps.json'),
  JSON.stringify(results.unusedDeps, null, 2)
);

// Summary report
console.log('\n📊 Audit Summary:');
console.log('─'.repeat(40));
console.log(`Unused files: ${results.unusedFiles.length}`);
console.log(`Unused exports: ${results.unusedExports.length}`);
console.log(`Unused dependencies: ${results.unusedDeps.length}`);
console.log('─'.repeat(40));
console.log('\nResults saved to:');
console.log('  - unused-files.json');
console.log('  - unused-exports.txt');
console.log('  - unused-deps.json');

// Cleanup temp files
try {
  fs.unlinkSync(path.join(rootDir, 'knip.json'));
} catch {}

process.exit(0);