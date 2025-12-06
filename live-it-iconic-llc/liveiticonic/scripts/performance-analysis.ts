#!/usr/bin/env tsx
/**
 * Performance Analysis Script
 * Analyzes TypeScript/React codebase for performance issues
 */

import * as fs from 'fs';
import * as path from 'path';

interface PerformanceIssue {
  file: string;
  line?: number;
  type: 'warning' | 'error' | 'info';
  category: string;
  message: string;
  metric?: number;
}

const issues: PerformanceIssue[] = [];

function calculateCyclomaticComplexity(content: string): number {
  const decisionPoints = [
    /\bif\b/g,
    /\belse\b/g,
    /\bfor\b/g,
    /\bwhile\b/g,
    /\bcase\b/g,
    /\bcatch\b/g,
    /&&/g,
    /\|\|/g,
    /\?/g,
  ];

  let complexity = 1;
  decisionPoints.forEach(pattern => {
    const matches = content.match(pattern);
    if (matches) complexity += matches.length;
  });

  return complexity;
}

function analyzeFileSize(filePath: string): void {
  const stats = fs.statSync(filePath);
  const sizeKB = stats.size / 1024;

  if (sizeKB > 500) {
    issues.push({
      file: filePath,
      type: 'error',
      category: 'File Size',
      message: `Large file: ${sizeKB.toFixed(2)} KB - consider splitting`,
      metric: sizeKB
    });
  } else if (sizeKB > 300) {
    issues.push({
      file: filePath,
      type: 'warning',
      category: 'File Size',
      message: `Large file: ${sizeKB.toFixed(2)} KB - may need refactoring`,
      metric: sizeKB
    });
  }
}

function analyzeComplexity(filePath: string, content: string): void {
  const lines = content.split('\n');
  const functions = content.match(/function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>/g) || [];

  // Analyze function complexity
  const functionPattern = /(function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>)\s*\{/g;
  let match;

  while ((match = functionPattern.exec(content)) !== null) {
    const startIndex = match.index;
    const functionContent = extractFunctionBody(content, startIndex);
    const complexity = calculateCyclomaticComplexity(functionContent);

    if (complexity > 20) {
      const lineNumber = content.substring(0, startIndex).split('\n').length;
      issues.push({
        file: filePath,
        line: lineNumber,
        type: 'error',
        category: 'Cyclomatic Complexity',
        message: `High complexity: ${complexity} - refactor recommended`,
        metric: complexity
      });
    } else if (complexity > 10) {
      const lineNumber = content.substring(0, startIndex).split('\n').length;
      issues.push({
        file: filePath,
        line: lineNumber,
        type: 'warning',
        category: 'Cyclomatic Complexity',
        message: `Medium complexity: ${complexity} - consider simplifying`,
        metric: complexity
      });
    }
  }

  // Check for long functions
  lines.forEach((line, index) => {
    if (line.includes('function') || line.match(/const\s+\w+\s*=.*=>/)) {
      const functionBody = extractFunctionFromLine(lines, index);
      if (functionBody.length > 100) {
        issues.push({
          file: filePath,
          line: index + 1,
          type: 'warning',
          category: 'Function Length',
          message: `Long function: ${functionBody.length} lines - consider decomposition`,
          metric: functionBody.length
        });
      }
    }
  });
}

function extractFunctionBody(content: string, startIndex: number): string {
  let braceCount = 0;
  let inFunction = false;
  let body = '';

  for (let i = startIndex; i < content.length; i++) {
    const char = content[i];
    if (char === '{') {
      braceCount++;
      inFunction = true;
    }
    if (inFunction) body += char;
    if (char === '}') {
      braceCount--;
      if (braceCount === 0) break;
    }
  }

  return body;
}

function extractFunctionFromLine(lines: string[], startLine: number): string[] {
  const body: string[] = [];
  let braceCount = 0;
  let started = false;

  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i];
    const openBraces = (line.match(/\{/g) || []).length;
    const closeBraces = (line.match(/\}/g) || []).length;

    if (openBraces > 0) started = true;
    if (started) {
      body.push(line);
      braceCount += openBraces - closeBraces;
      if (braceCount === 0) break;
    }
  }

  return body;
}

function analyzePerformancePatterns(filePath: string, content: string): void {
  const lines = content.split('\n');

  lines.forEach((line, index) => {
    // Check for inline object/array creation in JSX
    if (line.includes('style={{') || line.includes('className={[')) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'warning',
        category: 'React Performance',
        message: 'Inline object/array in JSX - causes re-renders',
      });
    }

    // Check for missing React.memo
    if (line.includes('export const') && line.includes('FC') && !content.includes('memo(')) {
      const componentName = line.match(/const\s+(\w+)/)?.[1];
      if (componentName && !content.includes(`memo(${componentName})`)) {
        issues.push({
          file: filePath,
          line: index + 1,
          type: 'info',
          category: 'React Performance',
          message: `Component ${componentName} not memoized - consider React.memo`,
        });
      }
    }

    // Check for expensive operations in render
    if (line.includes('.map(') && !line.includes('useMemo') && !line.includes('useCallback')) {
      const prevLines = lines.slice(Math.max(0, index - 5), index).join('\n');
      if (!prevLines.includes('useMemo') && !prevLines.includes('useCallback')) {
        issues.push({
          file: filePath,
          line: index + 1,
          type: 'warning',
          category: 'React Performance',
          message: 'Array operation without memoization - consider useMemo',
        });
      }
    }

    // Check for synchronous blocking operations
    if (line.match(/fs\.readFileSync|fs\.writeFileSync/)) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'error',
        category: 'Blocking Operations',
        message: 'Synchronous file operation - use async version',
      });
    }

    // Check for inefficient queries
    if (line.includes('.filter(') && line.includes('.map(')) {
      issues.push({
        file: filePath,
        line: index + 1,
        type: 'info',
        category: 'Algorithm Efficiency',
        message: 'Chained filter+map - consider single reduce for better performance',
      });
    }
  });

  // Check for missing keys in lists
  const listPattern = /\.map\([^)]*\)\s*=>\s*</g;
  let match;
  while ((match = listPattern.exec(content)) !== null) {
    const snippet = content.substring(match.index, match.index + 200);
    if (!snippet.includes('key=')) {
      const lineNumber = content.substring(0, match.index).split('\n').length;
      issues.push({
        file: filePath,
        line: lineNumber,
        type: 'error',
        category: 'React Performance',
        message: 'Missing key prop in list - causes inefficient re-renders',
      });
    }
  }
}

function analyzeFile(filePath: string): void {
  const content = fs.readFileSync(filePath, 'utf-8');

  analyzeFileSize(filePath);
  analyzeComplexity(filePath, content);
  analyzePerformancePatterns(filePath, content);
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
  console.log('\n=== PERFORMANCE ANALYSIS REPORT ===\n');

  // Group by category
  const categories = issues.reduce((acc, issue) => {
    if (!acc[issue.category]) acc[issue.category] = [];
    acc[issue.category].push(issue);
    return acc;
  }, {} as Record<string, PerformanceIssue[]>);

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

    // Show top 10 per category
    categoryIssues.slice(0, 10).forEach((issue) => {
      const relativePath = issue.file.replace(process.cwd(), '.');
      const location = issue.line ? `${relativePath}:${issue.line}` : relativePath;
      console.log(`  [${issue.type.toUpperCase()}] ${location}`);
      console.log(`  ${issue.message}`);
      if (issue.metric) {
        console.log(`  Metric: ${issue.metric}`);
      }
      console.log('');
    });

    if (categoryIssues.length > 10) {
      console.log(`  ... and ${categoryIssues.length - 10} more\n`);
    }
  });

  // Top 10 most complex functions
  const complexityIssues = issues
    .filter(i => i.category === 'Cyclomatic Complexity' && i.metric)
    .sort((a, b) => (b.metric || 0) - (a.metric || 0))
    .slice(0, 10);

  if (complexityIssues.length > 0) {
    console.log('\n\nTOP 10 MOST COMPLEX FUNCTIONS:');
    console.log('─'.repeat(50));
    complexityIssues.forEach((issue, i) => {
      const relativePath = issue.file.replace(process.cwd(), '.');
      console.log(`${i + 1}. ${relativePath}:${issue.line} - Complexity: ${issue.metric}`);
    });
  }

  // Top 10 largest files
  const sizeIssues = issues
    .filter(i => i.category === 'File Size' && i.metric)
    .sort((a, b) => (b.metric || 0) - (a.metric || 0))
    .slice(0, 10);

  if (sizeIssues.length > 0) {
    console.log('\n\nTOP 10 LARGEST FILES:');
    console.log('─'.repeat(50));
    sizeIssues.forEach((issue, i) => {
      const relativePath = issue.file.replace(process.cwd(), '.');
      console.log(`${i + 1}. ${relativePath} - ${issue.metric?.toFixed(2)} KB`);
    });
  }

  // Save full report to file
  const reportPath = path.join(process.cwd(), 'reports', 'performance-analysis.json');
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
    })),
    topComplexFunctions: complexityIssues,
    largestFiles: sizeIssues
  }, null, 2));

  console.log(`\n\nFull report saved to: ${reportPath}`);
}

// Run analysis
const srcDir = path.join(process.cwd(), 'src');
console.log('Analyzing codebase for performance issues...');
console.log(`Source directory: ${srcDir}\n`);

walkDirectory(srcDir);
generateReport();
