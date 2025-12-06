#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🪦 Safe graveyard process for unused files...\n');

// Load audit results
let auditResults;
try {
  auditResults = JSON.parse(fs.readFileSync(path.join(rootDir, 'audit-results.json'), 'utf8'));
} catch (error) {
  console.error('❌ audit-results.json not found. Run simple-audit.mjs first.');
  process.exit(1);
}

// Create graveyard
const graveyardDir = path.join(rootDir, '_graveyard');
const timestamp = new Date().toISOString().split('T')[0];
const sessionDir = path.join(graveyardDir, `unused-files-${timestamp}`);

if (!fs.existsSync(sessionDir)) {
  fs.mkdirSync(sessionDir, { recursive: true });
}

// Conservative list of files that are definitely safe to move
const safeToMove = auditResults.unusedFiles.filter(file => {
  return file.includes('ContactLinks') ||
         file.includes('LaTeXRenderer') ||
         file.includes('BlogSystem') ||
         file.includes('SimpleCalendlyButton') ||
         file.includes('FeatureFlagDevTools') ||
         file.includes('RealTimeAnalytics') ||
         file.includes('AdvancedAnalytics') ||
         file.includes('CompactAuthForm') ||
         (file.includes('booking/') && !file.includes('main')) ||
         (file.includes('blog/') && !file.includes('main'));
});

const stats = {
  moved: 0,
  skipped: 0,
  errors: 0
};

console.log(`Found ${safeToMove.length} files safe to move to graveyard:\n`);

safeToMove.forEach(filePath => {
  try {
    const fullPath = path.join(rootDir, filePath);
    
    if (!fs.existsSync(fullPath)) {
      console.log(`⏩ Skipped (not found): ${filePath}`);
      stats.skipped++;
      return;
    }
    
    // Create destination path
    const destPath = path.join(sessionDir, filePath);
    const destDir = path.dirname(destPath);
    fs.mkdirSync(destDir, { recursive: true });
    
    // Move file (use regular move, not git mv to avoid issues)
    fs.renameSync(fullPath, destPath);
    console.log(`✅ Moved: ${filePath}`);
    stats.moved++;
    
  } catch (error) {
    console.error(`❌ Error moving ${filePath}:`, error.message);
    stats.errors++;
  }
});

// Create manifest
const manifest = {
  timestamp: new Date().toISOString(),
  totalAuditedFiles: auditResults.stats.totalFiles,
  potentiallyUnused: auditResults.unusedFiles.length,
  actuallyMoved: stats.moved,
  movedFiles: safeToMove.filter(f => fs.existsSync(path.join(sessionDir, f))),
  stats: stats
};

fs.writeFileSync(
  path.join(sessionDir, 'manifest.json'),
  JSON.stringify(manifest, null, 2)
);

// Create restore script
const restoreScript = `#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

const manifest = ${JSON.stringify(manifest, null, 2)};

console.log('🔄 Restoring ${stats.moved} files from graveyard...');

manifest.movedFiles.forEach(file => {
  const graveyardPath = path.join(__dirname, file);
  const originalPath = path.join(rootDir, file);
  
  try {
    const originalDir = path.dirname(originalPath);
    fs.mkdirSync(originalDir, { recursive: true });
    fs.renameSync(graveyardPath, originalPath);
    console.log(\`✅ Restored: \${file}\`);
  } catch (error) {
    console.error(\`❌ Failed to restore \${file}:\`, error.message);
  }
});

console.log('\\n✅ Restoration complete!');
`;

fs.writeFileSync(path.join(sessionDir, 'restore.mjs'), restoreScript);

// Summary
console.log('\n📊 Graveyard Summary:');
console.log('─'.repeat(50));
console.log(`Files moved: ${stats.moved}`);
console.log(`Files skipped: ${stats.skipped}`);
console.log(`Errors: ${stats.errors}`);
console.log('─'.repeat(50));
console.log(`\n📁 Files moved to: ${sessionDir}`);
console.log(`📜 Run 'node ${path.join(sessionDir, 'restore.mjs')}' to restore if needed`);

console.log('\n✅ Safe graveyard process complete!');

process.exit(0);