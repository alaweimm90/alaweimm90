#!/usr/bin/env tsx
/**
 * Bundle size analyzer for all web apps
 * Usage: npm run perf:bundle
 */
import { execSync } from 'child_process';
import { readdirSync, statSync } from 'fs';
import { join } from 'path';

const WEB_APPS = [
  'repz-llc/repz',
  'live-it-iconic-llc/liveiticonic',
  'alawein-technologies-llc/attributa'
];

function analyzeBundles() {
  console.log('📦 Bundle Size Analysis\n');

  for (const app of WEB_APPS) {
    const distPath = join(app, 'dist');
    try {
      const files = readdirSync(distPath, { recursive: true }) as string[];
      let totalSize = 0;

      for (const file of files) {
        const filePath = join(distPath, file);
        const stat = statSync(filePath);
        if (stat.isFile()) totalSize += stat.size;
      }

      console.log(`${app}: ${(totalSize / 1024).toFixed(1)} KB`);
    } catch {
      console.log(`${app}: No dist folder`);
    }
  }
}

analyzeBundles();
