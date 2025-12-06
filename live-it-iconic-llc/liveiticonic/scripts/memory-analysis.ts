#!/usr/bin/env tsx
/**
 * Memory Analysis Script
 * Analyzes TypeScript/React codebase for memory-related issues
 */

import * as fs from 'fs';
import * as path from 'path';

interface MemoryIssue {
  file: string;
  line: number;
  type: 'warning' | 'error' | 'info';
  category: string;
  message: string;
  snippet?: string;
}

const issues: MemoryIssue[] = [];

// Patterns to detect
const patterns = {
  eventListeners: /addEventListener|on\w+\s*=/gi,
  intervals: /setInterval|setTimeout/gi,
  largeArrays: /new Array\(\d{3,}\)|Array\.from\(\{[^}]*length:\s*\d{3,}/gi,
  mapCache: /new Map\(\)|new WeakMap\(\)|new Set\(\)|new WeakSet\(\)/gi,
  refs: /useRef|createRef/gi,
  closures: /function.*\{[\s\S]*return\s+function/gi,
  memoization: /useMemo|useCallback|memo\(/gi,
  effectCleanup: /useEffect\(/gi,
};

function analyzeFile(filePath: string): void {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  lines.forEach((line, index) => {
    // Check for event listeners without cleanup
    if (patterns.eventListeners.test(line)) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'warning',
        category: 'Event Listeners',
        message: 'Event listener detected - ensure proper cleanup in useEffect',
        snippet: line.trim()
      });
    }

    // Check for intervals/timeouts
    if (patterns.intervals.test(line)) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'warning',
        category: 'Timers',
        message: 'Timer detected - ensure clearInterval/clearTimeout in cleanup',
        snippet: line.trim()
      });
    }

    // Check for large array allocations
    if (patterns.largeArrays.test(line)) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'error',
        category: 'Large Allocations',
        message: 'Large array allocation detected - consider pagination or lazy loading',
        snippet: line.trim()
      });
    }

    // Check for caches without size limits
    if (patterns.mapCache.test(line)) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'info',
        category: 'Caching',
        message: 'Cache structure detected - ensure size limits and TTL',
        snippet: line.trim()
      });
    }
  });

  // Check for useEffect without cleanup
  const effectMatches = content.match(/useEffect\([^)]*\)/gs) || [];
  effectMatches.forEach((match) => {
    if (!match.includes('return')) {
      const lineNumber = content.substring(0, content.indexOf(match)).split('\n').length;
      issues.push({
        file: filePath,
        line: lineNumber,
        type: 'warning',
        category: 'Effect Cleanup',
        message: 'useEffect without cleanup function - potential memory leak',
        snippet: match.substring(0, 50) + '...'
      });
    }
  });
}

function walkDirectory(dir: string, extensions: string[] = ['.ts', '.tsx']): void {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
      walkDirectory(filePath, extensions);
    } else if (stat.isFile() && extensions.some(ext => file.endsWith(ext))) {
      analyzeFile(filePath);
    }
  });
}

function generateReport(): void {
  console.log('\n=== MEMORY ANALYSIS REPORT ===\n');

  // Group by category
  const categories = issues.reduce((acc, issue) => {
    if (!acc[issue.category]) acc[issue.category] = [];
    acc[issue.category].push(issue);
    return acc;
  }, {} as Record<string, MemoryIssue[]>);

  // Summary
  console.log('SUMMARY:');
  console.log(`Total Issues: ${issues.length}`);
  console.log(`Errors: ${issues.filter(i => i.type === 'error').length}`);
  console.log(`Warnings: ${issues.filter(i => i.type === 'warning').length}`);
  console.log(`Info: ${issues.filter(i => i.type === 'info').length}`);
  console.log('\nISSUES BY CATEGORY:\n');

  Object.entries(categories).forEach(([category, categoryIssues]) => {
    console.log(`\n${category} (${categoryIssues.length}):`);
    console.log('─'.repeat(50));

    // Show top 5 per category
    categoryIssues.slice(0, 5).forEach((issue) => {
      const relativePath = issue.file.replace(process.cwd(), '.');
      console.log(`  [${issue.type.toUpperCase()}] ${relativePath}:${issue.line}`);
      console.log(`  ${issue.message}`);
      if (issue.snippet) {
        console.log(`  > ${issue.snippet}`);
      }
      console.log('');
    });

    if (categoryIssues.length > 5) {
      console.log(`  ... and ${categoryIssues.length - 5} more\n`);
    }
  });

  // Save full report to file
  const reportPath = path.join(process.cwd(), 'reports', 'memory-analysis.json');
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    summary: {
      total: issues.length,
      errors: issues.filter(i => i.type === 'error').length,
      warnings: issues.filter(i => i.type === 'warning').length,
      info: issues.filter(i => i.type === 'info').length,
    },
    issues: issues,
    categories: Object.keys(categories).map(cat => ({
      name: cat,
      count: categories[cat].length
    }))
  }, null, 2));

  console.log(`\nFull report saved to: ${reportPath}`);
}

// Run analysis
const srcDir = path.join(process.cwd(), 'src');
console.log('Analyzing codebase for memory issues...');
console.log(`Source directory: ${srcDir}\n`);

walkDirectory(srcDir);
generateReport();
