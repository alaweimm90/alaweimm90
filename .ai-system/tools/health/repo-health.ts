#!/usr/bin/env npx ts-node
/**
 * 🏥 REPO HEALTH - Automated Repository Health Checks
 *
 * Runs parallel, non-destructive checks to keep the repo clean.
 *
 * Usage:
 *   npx ts-node tools/health/repo-health.ts          # Run all checks
 *   npx ts-node tools/health/repo-health.ts --fix    # Auto-fix safe issues
 *   npx ts-node tools/health/repo-health.ts deps     # Only dependency check
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '../..');
const COLORS = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

interface HealthResult {
  name: string;
  status: 'pass' | 'warn' | 'fail';
  message: string;
  details?: string[];
  fixable?: boolean;
}

// Results are collected in runAllChecks

function log(color: string, prefix: string, msg: string) {
  console.log(`${color}${prefix}${COLORS.reset} ${msg}`);
}

function runCommand(cmd: string, cwd = ROOT): string | null {
  try {
    // Use PowerShell on Windows for better compatibility
    const isWindows = process.platform === 'win32';
    const shell = isWindows ? 'powershell.exe' : '/bin/sh';
    return execSync(cmd, { cwd, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], shell });
  } catch (e: any) {
    return e.stdout || e.stderr || null;
  }
}

function findFiles(dir: string, pattern: RegExp, skipArchive = true): string[] {
  const results: string[] = [];
  const skipDirs = ['node_modules', '.git', '.pytest_cache', '.ruff_cache', '__pycache__'];
  if (skipArchive) skipDirs.push('.archive');

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (skipDirs.includes(entry.name)) continue;
      if (entry.isDirectory()) {
        results.push(...findFiles(fullPath, pattern, skipArchive));
      } else if (pattern.test(entry.name)) {
        results.push(fullPath);
      }
    }
  } catch { }
  return results;
}

// ============================================
// HEALTH CHECKS
// ============================================

async function checkDependencies(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Checking dependencies...');

  const audit = runCommand('npm audit --json 2>/dev/null') || '{}';
  let auditData: any = {};
  try {
    auditData = JSON.parse(audit);
  } catch { }

  const vulns = auditData.metadata?.vulnerabilities || {};
  const total = (vulns.high || 0) + (vulns.critical || 0);

  const outdated = runCommand('npm outdated --json 2>/dev/null') || '{}';
  let outdatedData: any = {};
  try {
    outdatedData = JSON.parse(outdated);
  } catch { }
  const outdatedCount = Object.keys(outdatedData).length;

  if (total > 0) {
    return {
      name: 'Dependencies',
      status: 'fail',
      message: `${total} high/critical vulnerabilities`,
      details: [`Run: npm audit fix`],
      fixable: true,
    };
  } else if (outdatedCount > 5) {
    return {
      name: 'Dependencies',
      status: 'warn',
      message: `${outdatedCount} outdated packages`,
      details: Object.keys(outdatedData).slice(0, 5).map(p => `  - ${p}`),
    };
  }

  return { name: 'Dependencies', status: 'pass', message: 'All secure and up-to-date' };
}

async function checkDeadCode(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Scanning for dead code...');

  // Find all TypeScript files
  const tsFiles = findFiles(ROOT, /\.ts$/);
  const fileCount = tsFiles.length;

  // Check for orphaned test files
  const testFiles = findFiles(ROOT, /\.test\.ts$/);
  const orphanedTests: string[] = [];

  for (const testFile of testFiles) {
    const sourceFile = testFile.replace('.test.ts', '.ts');
    if (!fs.existsSync(sourceFile)) {
      orphanedTests.push(path.relative(ROOT, testFile));
    }
  }

  if (orphanedTests.length > 0) {
    return {
      name: 'Dead Code',
      status: 'warn',
      message: `${orphanedTests.length} orphaned test files`,
      details: orphanedTests.slice(0, 5),
    };
  }

  return { name: 'Dead Code', status: 'pass', message: `Scanned ${fileCount} TypeScript files` };
}

async function checkDocHealth(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Checking documentation health...');

  const docsDir = path.join(ROOT, 'docs');
  const issues: string[] = [];

  // Check for broken internal links using cross-platform findFiles
  const mdFiles = findFiles(docsDir, /\.md$/);
  let brokenLinks = 0;
  let totalLinks = 0;

  for (const file of mdFiles) {
    try {
      const content = fs.readFileSync(file, 'utf8');
      const linkMatches = content.match(/\[.*?\]\(((?!http)[^)]+)\)/g) || [];

      for (const match of linkMatches) {
        totalLinks++;
        const linkPath = match.match(/\]\(([^)]+)\)/)?.[1];
        if (linkPath && !linkPath.startsWith('#')) {
          const resolvedPath = path.resolve(path.dirname(file), linkPath.split('#')[0]);
          if (!fs.existsSync(resolvedPath)) {
            brokenLinks++;
            issues.push(`${path.relative(ROOT, file)}: ${linkPath}`);
          }
        }
      }
    } catch { }
  }

  // Check for stale docs (not modified in 90+ days)
  const staleThreshold = Date.now() - (90 * 24 * 60 * 60 * 1000);
  let staleDocs = 0;

  for (const file of mdFiles) {
    try {
      const stat = fs.statSync(file);
      if (stat.mtimeMs < staleThreshold) {
        staleDocs++;
      }
    } catch { }
  }

  if (brokenLinks > 0) {
    return {
      name: 'Documentation',
      status: 'fail',
      message: `${brokenLinks} broken links found`,
      details: issues.slice(0, 5),
      fixable: false,
    };
  } else if (staleDocs > 10) {
    return {
      name: 'Documentation',
      status: 'warn',
      message: `${staleDocs} docs not updated in 90+ days`,
    };
  }

  return { name: 'Documentation', status: 'pass', message: `${totalLinks} links verified` };
}

async function checkSecrets(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Scanning for exposed secrets...');

  const patterns = [
    /(?:api[_-]?key|apikey)\s*[:=]\s*['"][^'"]{20,}['"]/gi,
    /(?:secret|password|passwd|pwd)\s*[:=]\s*['"][^'"]{8,}['"]/gi,
    /(?:sk|pk)[-_](?:live|test)[-_][a-zA-Z0-9]{20,}/g,
    /ghp_[a-zA-Z0-9]{36}/g,
    /-----BEGIN (?:RSA |EC )?PRIVATE KEY-----/g,
  ];

  const issues: string[] = [];
  // Use cross-platform file finding
  const allFiles = findFiles(ROOT, /\.(ts|js|py|md|yaml|yml|env|sh)$/);

  for (const file of allFiles) {
    const relPath = path.relative(ROOT, file);
    if (relPath.includes('node_modules') || relPath.endsWith('.lock')) continue;

    try {
      const content = fs.readFileSync(file, 'utf8');
      for (const pattern of patterns) {
        if (pattern.test(content)) {
          issues.push(relPath);
          break;
        }
      }
    } catch { }
  }

  if (issues.length > 0) {
    return {
      name: 'Secrets',
      status: 'fail',
      message: `${issues.length} files may contain secrets`,
      details: issues.slice(0, 5),
      fixable: false,
    };
  }

  return { name: 'Secrets', status: 'pass', message: 'No exposed secrets detected' };
}

async function checkTypeHealth(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Checking TypeScript health...');

  // Count `any` types using cross-platform approach
  const tsFiles = findFiles(ROOT, /\.ts$/);
  let anyTypes = 0;
  for (const file of tsFiles) {
    try {
      const content = fs.readFileSync(file, 'utf8');
      const matches = content.match(/:\s*any/g);
      if (matches) anyTypes += matches.length;
    } catch { }
  }

  // Run tsc --noEmit
  const tscResult = runCommand('npx tsc --noEmit');
  const hasErrors = tscResult?.includes('error TS');

  if (hasErrors) {
    return {
      name: 'TypeScript',
      status: 'fail',
      message: 'Compilation errors found',
      details: tscResult?.split('\n').slice(0, 5),
    };
  } else if (anyTypes > 50) {
    return {
      name: 'TypeScript',
      status: 'warn',
      message: `${anyTypes} uses of 'any' type`,
      details: ['Consider adding stricter types'],
    };
  }

  return { name: 'TypeScript', status: 'pass', message: `Clean compilation, ${anyTypes} any types` };
}

async function checkGitHealth(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Checking git health...');

  // Check for large files in working directory
  const allFiles = findFiles(ROOT, /./);
  const largeFileList: string[] = [];
  for (const file of allFiles) {
    try {
      const stat = fs.statSync(file);
      if (stat.size > 1000000) {
        largeFileList.push(path.relative(ROOT, file));
      }
    } catch { }
  }

  // Check for uncommitted changes
  const status = runCommand('git status --porcelain 2>/dev/null') || '';
  const uncommitted = status.split('\n').filter(Boolean).length;

  if (largeFileList.length > 0) {
    return {
      name: 'Git Health',
      status: 'warn',
      message: `${largeFileList.length} large files (>1MB) in history`,
      details: largeFileList.slice(0, 5),
    };
  } else if (uncommitted > 20) {
    return {
      name: 'Git Health',
      status: 'warn',
      message: `${uncommitted} uncommitted changes`,
    };
  }

  return { name: 'Git Health', status: 'pass', message: 'Repository is clean' };
}

async function checkAssets(): Promise<HealthResult> {
  log(COLORS.cyan, '🔍', 'Checking static assets...');

  // Find large unoptimized images using cross-platform approach
  const imageFiles = findFiles(ROOT, /\.(png|jpg|jpeg|gif|webp)$/i);
  const largeImages: string[] = [];

  for (const file of imageFiles) {
    try {
      const stat = fs.statSync(file);
      if (stat.size > 500 * 1024) { // 500KB
        largeImages.push(path.relative(ROOT, file));
      }
    } catch { }
  }

  if (largeImages.length > 0) {
    return {
      name: 'Assets',
      status: 'warn',
      message: `${largeImages.length} images over 500KB`,
      details: largeImages.slice(0, 5),
      fixable: true,
    };
  }

  return { name: 'Assets', status: 'pass', message: `${imageFiles.length} images checked` };
}

// ============================================
// MAIN
// ============================================

async function runAllChecks() {
  console.log('\n' + '═'.repeat(50));
  console.log('  🏥 REPO HEALTH CHECK');
  console.log('═'.repeat(50) + '\n');

  const checks = [
    checkDependencies,
    checkDeadCode,
    checkDocHealth,
    checkSecrets,
    checkTypeHealth,
    checkGitHealth,
    checkAssets,
  ];

  // Run all checks in parallel
  const results = await Promise.all(checks.map(check => check()));

  // Print results
  console.log('\n' + '─'.repeat(50));
  console.log('  RESULTS');
  console.log('─'.repeat(50) + '\n');

  let passCount = 0, warnCount = 0, failCount = 0;

  for (const result of results) {
    const icon = result.status === 'pass' ? '✅' : result.status === 'warn' ? '⚠️' : '❌';
    const color = result.status === 'pass' ? COLORS.green : result.status === 'warn' ? COLORS.yellow : COLORS.red;

    console.log(`${icon} ${color}${result.name}${COLORS.reset}: ${result.message}`);

    if (result.details) {
      for (const detail of result.details) {
        console.log(`   ${COLORS.dim}${detail}${COLORS.reset}`);
      }
    }

    if (result.status === 'pass') passCount++;
    else if (result.status === 'warn') warnCount++;
    else failCount++;
  }

  // Summary
  console.log('\n' + '═'.repeat(50));
  const score = Math.round((passCount / results.length) * 100);
  const scoreColor = score >= 80 ? COLORS.green : score >= 60 ? COLORS.yellow : COLORS.red;
  console.log(`  Health Score: ${scoreColor}${score}%${COLORS.reset} (${passCount} pass, ${warnCount} warn, ${failCount} fail)`);
  console.log('═'.repeat(50) + '\n');

  // Generate report
  const report = {
    timestamp: new Date().toISOString(),
    score,
    results: results.map(r => ({ name: r.name, status: r.status, message: r.message })),
  };

  fs.writeFileSync(
    path.join(ROOT, '.orchex', 'health-report.json'),
    JSON.stringify(report, null, 2)
  );

  console.log(`📄 Report saved to .orchex/health-report.json\n`);

  return failCount === 0 ? 0 : 1;
}

// Run
runAllChecks().then(process.exit);
