#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

const manifest = {
  "timestamp": "2025-08-04T14:11:02.852Z",
  "totalAuditedFiles": 464,
  "potentiallyUnused": 86,
  "actuallyMoved": 9,
  "movedFiles": [
    "src/components/ContactLinks.tsx",
    "src/components/advanced/AdvancedAnalytics.tsx",
    "src/components/analytics/RealTimeAnalytics.tsx",
    "src/components/auth/CompactAuthForm.tsx",
    "src/components/blog/BlogSystem.tsx",
    "src/components/blog/LaTeXRenderer.tsx",
    "src/components/booking/BookingSection.tsx",
    "src/components/booking/SimpleCalendlyButton.tsx",
    "src/components/feature-flags/FeatureFlagDevTools.tsx"
  ],
  "stats": {
    "moved": 9,
    "skipped": 0,
    "errors": 0
  }
};

console.log('🔄 Restoring 9 files from graveyard...');

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
