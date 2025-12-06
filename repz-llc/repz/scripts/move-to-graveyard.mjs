#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🪦 Moving unused files to graveyard...\n');

// Create graveyard directory
const graveyardDir = path.join(rootDir, '_graveyard');
if (!fs.existsSync(graveyardDir)) {
  fs.mkdirSync(graveyardDir, { recursive: true });
}

// Create timestamp for this cleanup
const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
const sessionDir = path.join(graveyardDir, `cleanup-${timestamp}`);
fs.mkdirSync(sessionDir, { recursive: true });

// Load unused files list
let unusedFiles = [];
try {
  const unusedFilesPath = path.join(rootDir, 'unused-files.json');
  if (fs.existsSync(unusedFilesPath)) {
    unusedFiles = JSON.parse(fs.readFileSync(unusedFilesPath, 'utf8'));
  } else {
    console.error('❌ unused-files.json not found. Run prune-codebase.mjs first.');
    process.exit(1);
  }
} catch (error) {
  console.error('❌ Error loading unused files:', error.message);
  process.exit(1);
}

// Statistics
const stats = {
  movedFiles: 0,
  skippedFiles: 0,
  errors: 0,
  totalSize: 0
};

// Critical files to never move
const protectedFiles = [
  'App.tsx',
  'main.tsx',
  'index.tsx',
  'vite-env.d.ts',
  'RepzHome.tsx',
  'routes.tsx',
  'supabase/client.ts',
  'tailwind.config.ts',
  'global.css',
  'tokens.json'
];

// Move files to graveyard
console.log(`Found ${unusedFiles.length} files to process...\n`);

unusedFiles.forEach(filePath => {
  try {
    const fullPath = path.join(rootDir, filePath);
    
    // Skip if file doesn't exist
    if (!fs.existsSync(fullPath)) {
      console.log(`⏩ Skipped (not found): ${filePath}`);
      stats.skippedFiles++;
      return;
    }
    
    // Skip protected files
    const fileName = path.basename(filePath);
    if (protectedFiles.some(protected => filePath.includes(protected))) {
      console.log(`🛡️  Protected: ${filePath}`);
      stats.skippedFiles++;
      return;
    }
    
    // Skip test files, stories, and config files
    if (filePath.includes('.test.') || 
        filePath.includes('.spec.') || 
        filePath.includes('.stories.') ||
        filePath.includes('setupTests') ||
        filePath.includes('test-utils')) {
      console.log(`⏩ Skipped (test file): ${filePath}`);
      stats.skippedFiles++;
      return;
    }
    
    // Get file stats
    const fileStats = fs.statSync(fullPath);
    stats.totalSize += fileStats.size;
    
    // Create destination path maintaining structure
    const destPath = path.join(sessionDir, filePath);
    const destDir = path.dirname(destPath);
    
    // Create destination directory
    fs.mkdirSync(destDir, { recursive: true });
    
    // Use git mv if in git repo, otherwise use fs
    try {
      execSync(`git mv "${fullPath}" "${destPath}"`, { 
        stdio: 'pipe',
        cwd: rootDir 
      });
      console.log(`✅ Moved: ${filePath}`);
    } catch {
      // Fallback to regular move if not in git
      fs.renameSync(fullPath, destPath);
      console.log(`✅ Moved (no git): ${filePath}`);
    }
    
    stats.movedFiles++;
  } catch (error) {
    console.error(`❌ Error moving ${filePath}:`, error.message);
    stats.errors++;
  }
});

// Create manifest file
const manifest = {
  timestamp: new Date().toISOString(),
  stats: stats,
  movedFiles: unusedFiles.filter(f => {
    const fullPath = path.join(rootDir, f);
    return fs.existsSync(path.join(sessionDir, f));
  })
};

fs.writeFileSync(
  path.join(sessionDir, 'manifest.json'),
  JSON.stringify(manifest, null, 2)
);

// Create restoration script
const restoreScript = `#!/usr/bin/env node
// Restoration script for cleanup session ${timestamp}

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const manifest = ${JSON.stringify(manifest, null, 2)};

console.log('🔄 Restoring files from graveyard...');

manifest.movedFiles.forEach(file => {
  const graveyardPath = path.join(__dirname, file);
  const originalPath = path.join(__dirname, '../..', file);
  
  try {
    const originalDir = path.dirname(originalPath);
    fs.mkdirSync(originalDir, { recursive: true });
    
    try {
      execSync(\`git mv "\${graveyardPath}" "\${originalPath}"\`, { stdio: 'pipe' });
    } catch {
      fs.renameSync(graveyardPath, originalPath);
    }
    
    console.log(\`✅ Restored: \${file}\`);
  } catch (error) {
    console.error(\`❌ Failed to restore \${file}:\`, error.message);
  }
});

console.log('\\n✅ Restoration complete!');
`;

fs.writeFileSync(
  path.join(sessionDir, 'restore.mjs'),
  restoreScript
);
fs.chmodSync(path.join(sessionDir, 'restore.mjs'), '755');

// Summary
console.log('\n📊 Graveyard Summary:');
console.log('─'.repeat(50));
console.log(`Files moved: ${stats.movedFiles}`);
console.log(`Files skipped: ${stats.skippedFiles}`);
console.log(`Errors: ${stats.errors}`);
console.log(`Total size moved: ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB`);
console.log('─'.repeat(50));
console.log(`\n📁 Files moved to: ${sessionDir}`);
console.log(`📜 Manifest saved: ${path.join(sessionDir, 'manifest.json')}`);
console.log(`🔄 To restore, run: node ${path.join(sessionDir, 'restore.mjs')}`);

// Clean up the JSON files
try {
  fs.unlinkSync(path.join(rootDir, 'unused-files.json'));
  fs.unlinkSync(path.join(rootDir, 'unused-exports.txt'));
  fs.unlinkSync(path.join(rootDir, 'unused-deps.json'));
} catch {}

process.exit(0);