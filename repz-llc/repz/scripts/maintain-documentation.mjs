#!/usr/bin/env node

// Documentation Maintenance Script
// Run this monthly to keep documentation up to date

import fs from 'fs';
import path from 'path';

console.log('📚 Running documentation maintenance...');

// Check for files that haven't been updated in 6 months
const sixMonthsAgo = new Date();
sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

function checkDocumentationFreshness(dir) {
  const items = fs.readdirSync(dir);
  const staleFiles = [];
  
  items.forEach(item => {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory() && !item.includes('node_modules')) {
      staleFiles.push(...checkDocumentationFreshness(itemPath));
    } else if (item.endsWith('.md')) {
      if (stat.mtime < sixMonthsAgo) {
        staleFiles.push({
          file: path.relative(process.cwd(), itemPath),
          lastModified: stat.mtime.toISOString().split('T')[0]
        });
      }
    }
  });
  
  return staleFiles;
}

const staleFiles = checkDocumentationFreshness('.');

if (staleFiles.length > 0) {
  console.log('⚠️  Found stale documentation files:');
  staleFiles.forEach(file => {
    console.log(`  📅 ${file.file} (last modified: ${file.lastModified})`);
  });
  console.log('\n💡 Consider reviewing and updating these files.');
} else {
  console.log('✅ All documentation appears to be up to date!');
}

console.log('\n📚 Documentation maintenance complete!');
