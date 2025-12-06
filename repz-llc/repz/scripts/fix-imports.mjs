#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🔧 Fixing import paths after refactoring...\n');

// Define import replacements
const importReplacements = [
  // UI components moved to atoms/molecules/organisms
  { from: '@/components/ui/toaster', to: '@/ui/molecules/Toaster' },
  { from: '@/components/ui/toast', to: '@/ui/molecules/Toast' },
  { from: '@/components/ui/button', to: '@/ui/atoms/Button' },
  { from: '@/components/ui/input', to: '@/ui/atoms/Input' },
  { from: '@/components/ui/card', to: '@/ui/molecules/Card' },
  { from: '@/components/ui/dialog', to: '@/ui/molecules/Dialog' },
  { from: '@/components/ui/alert', to: '@/ui/molecules/Alert' },
  { from: '@/components/ui/tabs', to: '@/ui/molecules/Tabs' },
  { from: '@/components/ui/select', to: '@/ui/molecules/Select' },
  { from: '@/components/ui/dropdown-menu', to: '@/ui/molecules/DropdownMenu' },
  { from: '@/components/ui/accordion', to: '@/ui/molecules/Accordion' },
  { from: '@/components/ui/badge', to: '@/ui/atoms/Badge' },
  { from: '@/components/ui/avatar', to: '@/ui/atoms/Avatar' },
  { from: '@/components/ui/skeleton', to: '@/ui/atoms/Skeleton' },
  { from: '@/components/ui/switch', to: '@/ui/atoms/Switch' },
  { from: '@/components/ui/checkbox', to: '@/ui/atoms/Checkbox' },
  { from: '@/components/ui/label', to: '@/ui/atoms/Label' },
  { from: '@/components/ui/separator', to: '@/ui/atoms/Separator' },
  { from: '@/components/ui/use-toast', to: '@/ui/molecules/useToast' },
  
  // Components moved to features
  { from: '@/components/ai/AICoachingPanel', to: '@/features/ai/components/AICoachingPanel' },
  { from: '@/components/ai/AIHub', to: '@/features/ai/components/AIHub' },
  { from: '@/components/analytics/AnalyticsDashboard', to: '@/features/analytics/components/AnalyticsDashboard' },
  { from: '@/components/analytics/AnalyticsHub', to: '@/features/analytics/components/AnalyticsHub' },
  { from: '@/components/dashboard/ClientDashboard', to: '@/features/dashboard/components/ClientDashboard' },
  
  // RepzLogo moved to organisms
  { from: '@/components/RepzLogo', to: '@/ui/organisms/RepzLogo' }
];

const stats = {
  filesProcessed: 0,
  importsFixed: 0,
  errors: 0
};

// Function to fix imports in a file
function fixImportsInFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    let newContent = content;
    let changed = false;
    
    importReplacements.forEach(({ from, to }) => {
      const importRegex = new RegExp(`(['"])${from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\1`, 'g');
      if (importRegex.test(newContent)) {
        newContent = newContent.replace(importRegex, `$1${to}$1`);
        changed = true;
        stats.importsFixed++;
      }
    });
    
    if (changed) {
      fs.writeFileSync(filePath, newContent);
      console.log(`✅ Fixed imports in: ${path.relative(rootDir, filePath)}`);
    }
    
    stats.filesProcessed++;
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
    stats.errors++;
  }
}

// Find and process TypeScript/JavaScript files
function processDirectory(dir) {
  const items = fs.readdirSync(dir);
  
  items.forEach(item => {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory() && 
        !item.includes('node_modules') && 
        !item.startsWith('.') && 
        !item.includes('_graveyard')) {
      processDirectory(itemPath);
    } else if (stat.isFile() && 
               (item.endsWith('.tsx') || item.endsWith('.ts') || item.endsWith('.jsx') || item.endsWith('.js')) &&
               !item.includes('.test.') &&
               !item.includes('.spec.')) {
      fixImportsInFile(itemPath);
    }
  });
}

console.log('🔍 Processing src directory...\n');
processDirectory(path.join(rootDir, 'src'));

// Also process key files in root
const keyFiles = [
  'src/App.tsx',
  'src/main.tsx'
];

console.log('\n🔍 Processing key files...\n');
keyFiles.forEach(file => {
  const filePath = path.join(rootDir, file);
  if (fs.existsSync(filePath)) {
    fixImportsInFile(filePath);
  }
});

// Summary
console.log('\n📊 Import Fix Summary:');
console.log('─'.repeat(50));
console.log(`Files processed: ${stats.filesProcessed}`);
console.log(`Imports fixed: ${stats.importsFixed}`);
console.log(`Errors: ${stats.errors}`);
console.log('─'.repeat(50));

console.log('\n✅ Import paths have been updated!');

process.exit(0);