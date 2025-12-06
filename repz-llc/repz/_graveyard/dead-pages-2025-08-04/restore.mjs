#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

const manifest = {
  "timestamp": "2025-08-04T15:45:05.455Z",
  "type": "dead-pages-cleanup",
  "movedFiles": [
    "src/pages/DesignSystemDocs.tsx",
    "src/pages/RepzHome.tsx",
    "src/pages/__tests__/RepzHome.test.tsx"
  ],
  "totalFiles": 3
};

console.log('🔄 Restoring dead pages...');

manifest.movedFiles.forEach(file => {
  const graveyardPath = path.join(__dirname, file);
  const originalPath = path.join(rootDir, file);
  
  try {
    const originalDir = path.dirname(originalPath);
    fs.mkdirSync(originalDir, { recursive: true });
    fs.renameSync(graveyardPath, originalPath);
    console.log(`✅ Restored: ${file}`);
  } catch (error) {
    console.error(`❌ Failed to restore ${file}:`, error.message);
  }
});

console.log('\n✅ Restoration complete!');
