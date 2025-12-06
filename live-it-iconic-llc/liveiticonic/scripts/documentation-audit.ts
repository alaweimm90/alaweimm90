#!/usr/bin/env tsx
/**
 * Documentation Audit Script
 * Analyzes documentation coverage and quality
 */

import * as fs from 'fs';
import * as path from 'path';

interface DocStats {
  totalFiles: number;
  filesWithDocstrings: number;
  filesWithoutDocstrings: number;
  totalFunctions: number;
  documentedFunctions: number;
  totalClasses: number;
  documentedClasses: number;
  totalInterfaces: number;
  documentedInterfaces: number;
  markdownFiles: string[];
  readmeFiles: string[];
  missingDocs: string[];
}

const stats: DocStats = {
  totalFiles: 0,
  filesWithDocstrings: 0,
  filesWithoutDocstrings: 0,
  totalFunctions: 0,
  documentedFunctions: 0,
  totalClasses: 0,
  documentedClasses: 0,
  totalInterfaces: 0,
  documentedInterfaces: 0,
  markdownFiles: [],
  readmeFiles: [],
  missingDocs: [],
};

function hasDocstring(lines: string[], startIndex: number): boolean {
  // Check if there's a comment block before the declaration
  for (let i = startIndex - 1; i >= Math.max(0, startIndex - 5); i--) {
    const line = lines[i].trim();
    if (line.startsWith('/**') || line.startsWith('//')) {
      return true;
    }
    if (line && !line.startsWith('*') && !line.startsWith('//')) {
      break;
    }
  }
  return false;
}

function analyzeFile(filePath: string): void {
  stats.totalFiles++;
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  let hasAnyDoc = false;

  // Check for functions
  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Function declarations
    if (trimmed.match(/^(export\s+)?(async\s+)?function\s+\w+/) ||
        trimmed.match(/^(export\s+)?const\s+\w+\s*=\s*(\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)/)) {
      stats.totalFunctions++;
      if (hasDocstring(lines, index)) {
        stats.documentedFunctions++;
        hasAnyDoc = true;
      }
    }

    // Class declarations
    if (trimmed.match(/^(export\s+)?(abstract\s+)?class\s+\w+/)) {
      stats.totalClasses++;
      if (hasDocstring(lines, index)) {
        stats.documentedClasses++;
        hasAnyDoc = true;
      }
    }

    // Interface declarations
    if (trimmed.match(/^(export\s+)?interface\s+\w+/)) {
      stats.totalInterfaces++;
      if (hasDocstring(lines, index)) {
        stats.documentedInterfaces++;
        hasAnyDoc = true;
      }
    }
  });

  if (hasAnyDoc) {
    stats.filesWithDocstrings++;
  } else {
    stats.filesWithoutDocstrings++;
    stats.missingDocs.push(filePath);
  }
}

function findMarkdownFiles(dir: string): void {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
      findMarkdownFiles(filePath);
    } else if (file.endsWith('.md')) {
      stats.markdownFiles.push(filePath);
      if (file.toLowerCase() === 'readme.md') {
        stats.readmeFiles.push(filePath);
      }
    }
  });
}

function analyzeCodeFiles(dir: string, extensions: string[] = ['.ts', '.tsx']): void {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
      analyzeCodeFiles(filePath, extensions);
    } else if (stat.isFile() && extensions.some(ext => file.endsWith(ext))) {
      analyzeFile(filePath);
    }
  });
}

function checkRequiredDocs(): { name: string; exists: boolean; path?: string }[] {
  const requiredDocs = [
    'README.md',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    'LICENSE',
    'CODE_OF_CONDUCT.md',
    'SECURITY.md',
    '.github/PULL_REQUEST_TEMPLATE.md',
    '.github/ISSUE_TEMPLATE.md',
    'docs/API.md',
    'docs/ARCHITECTURE.md',
    'docs/DEPLOYMENT.md',
    'docs/TESTING.md',
  ];

  return requiredDocs.map((doc) => {
    const fullPath = path.join(process.cwd(), doc);
    return {
      name: doc,
      exists: fs.existsSync(fullPath),
      path: fullPath,
    };
  });
}

function generateReport(): void {
  console.log('\n=== DOCUMENTATION AUDIT REPORT ===\n');

  // Code documentation stats
  console.log('CODE DOCUMENTATION COVERAGE:');
  console.log('─'.repeat(50));
  console.log(`Total Files Analyzed: ${stats.totalFiles}`);
  console.log(`Files with Documentation: ${stats.filesWithDocstrings} (${((stats.filesWithDocstrings / stats.totalFiles) * 100).toFixed(1)}%)`);
  console.log(`Files without Documentation: ${stats.filesWithoutDocstrings} (${((stats.filesWithoutDocstrings / stats.totalFiles) * 100).toFixed(1)}%)\n`);

  console.log('FUNCTION DOCUMENTATION:');
  console.log(`Total Functions: ${stats.totalFunctions}`);
  console.log(`Documented Functions: ${stats.documentedFunctions} (${((stats.documentedFunctions / stats.totalFunctions) * 100).toFixed(1)}%)`);
  console.log(`Undocumented Functions: ${stats.totalFunctions - stats.documentedFunctions}\n`);

  console.log('CLASS DOCUMENTATION:');
  console.log(`Total Classes: ${stats.totalClasses}`);
  console.log(`Documented Classes: ${stats.documentedClasses} (${((stats.documentedClasses / stats.totalClasses) * 100).toFixed(1)}%)`);
  console.log(`Undocumented Classes: ${stats.totalClasses - stats.documentedClasses}\n`);

  console.log('INTERFACE DOCUMENTATION:');
  console.log(`Total Interfaces: ${stats.totalInterfaces}`);
  console.log(`Documented Interfaces: ${stats.documentedInterfaces} (${((stats.documentedInterfaces / stats.totalInterfaces) * 100).toFixed(1)}%)`);
  console.log(`Undocumented Interfaces: ${stats.totalInterfaces - stats.documentedInterfaces}\n`);

  // Markdown documentation
  console.log('\nMARKDOWN DOCUMENTATION:');
  console.log('─'.repeat(50));
  console.log(`Total Markdown Files: ${stats.markdownFiles.length}`);
  console.log(`README Files: ${stats.readmeFiles.length}\n`);

  // Required documentation check
  console.log('REQUIRED DOCUMENTATION CHECK:');
  console.log('─'.repeat(50));
  const requiredDocs = checkRequiredDocs();
  requiredDocs.forEach((doc) => {
    const status = doc.exists ? '✅' : '❌';
    console.log(`${status} ${doc.name}`);
  });

  const missingRequired = requiredDocs.filter(d => !d.exists);
  console.log(`\nMissing Required Docs: ${missingRequired.length}/${requiredDocs.length}\n`);

  // Top undocumented files
  if (stats.missingDocs.length > 0) {
    console.log('\nFILES WITHOUT DOCUMENTATION (Top 20):');
    console.log('─'.repeat(50));
    stats.missingDocs.slice(0, 20).forEach((file) => {
      console.log(`  ${file.replace(process.cwd(), '.')}`);
    });
    if (stats.missingDocs.length > 20) {
      console.log(`  ... and ${stats.missingDocs.length - 20} more\n`);
    }
  }

  // Documentation quality score
  const functionDocScore = stats.totalFunctions > 0
    ? (stats.documentedFunctions / stats.totalFunctions) * 100
    : 100;
  const classDocScore = stats.totalClasses > 0
    ? (stats.documentedClasses / stats.totalClasses) * 100
    : 100;
  const fileDocScore = (stats.filesWithDocstrings / stats.totalFiles) * 100;
  const requiredDocScore = (requiredDocs.filter(d => d.exists).length / requiredDocs.length) * 100;

  const overallScore = (functionDocScore + classDocScore + fileDocScore + requiredDocScore) / 4;

  console.log('\nDOCUMENTATION QUALITY SCORE:');
  console.log('─'.repeat(50));
  console.log(`Function Documentation: ${functionDocScore.toFixed(1)}%`);
  console.log(`Class Documentation: ${classDocScore.toFixed(1)}%`);
  console.log(`File Documentation: ${fileDocScore.toFixed(1)}%`);
  console.log(`Required Docs: ${requiredDocScore.toFixed(1)}%`);
  console.log(`\nOVERALL SCORE: ${overallScore.toFixed(1)}%`);

  let grade = 'F';
  if (overallScore >= 90) grade = 'A';
  else if (overallScore >= 80) grade = 'B';
  else if (overallScore >= 70) grade = 'C';
  else if (overallScore >= 60) grade = 'D';

  console.log(`GRADE: ${grade}\n`);

  // Save report
  const reportPath = path.join(process.cwd(), 'reports', 'documentation-audit.json');
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    stats,
    requiredDocs,
    scores: {
      functionDocScore: functionDocScore.toFixed(1),
      classDocScore: classDocScore.toFixed(1),
      fileDocScore: fileDocScore.toFixed(1),
      requiredDocScore: requiredDocScore.toFixed(1),
      overallScore: overallScore.toFixed(1),
      grade,
    }
  }, null, 2));

  console.log(`Full report saved to: ${reportPath}`);
}

// Run analysis
const srcDir = path.join(process.cwd(), 'src');
console.log('Analyzing codebase documentation...');
console.log(`Source directory: ${srcDir}\n`);

findMarkdownFiles(process.cwd());
analyzeCodeFiles(srcDir);
generateReport();
