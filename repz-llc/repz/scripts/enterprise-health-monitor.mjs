#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🏥 Enterprise Health Monitor & System Status Dashboard\n');

const healthMetrics = {
  timestamp: new Date().toISOString(),
  system: {
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    nodeVersion: process.version,
    platform: process.platform
  },
  codebase: {
    files: 0,
    lines: 0,
    components: 0,
    features: 0,
    tests: 0
  },
  dependencies: {
    total: 0,
    outdated: 0,
    vulnerabilities: 0,
    bundleSize: 0
  },
  quality: {
    typeErrors: 0,
    lintErrors: 0,
    testCoverage: 0,
    deadCode: 0
  },
  performance: {
    buildTime: 0,
    bundleSize: 0,
    firstLoadJs: 0
  },
  alerts: [],
  recommendations: []
};

// 1. CODEBASE HEALTH ANALYSIS
console.log('📊 Analyzing codebase health...\n');

function analyzeCodebase() {
  const srcDir = path.join(rootDir, 'src');
  if (!fs.existsSync(srcDir)) return;

  let totalFiles = 0;
  let totalLines = 0;
  let components = 0;
  let features = 0;
  let tests = 0;

  function scanDirectory(dir) {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory() && !item.includes('node_modules')) {
          if (item === 'features') {
            const featureItems = fs.readdirSync(itemPath);
            features = featureItems.filter(f => {
              const featurePath = path.join(itemPath, f);
              return fs.statSync(featurePath).isDirectory();
            }).length;
          }
          scanDirectory(itemPath);
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          totalFiles++;
          
          if (item.includes('.test.') || item.includes('.spec.')) {
            tests++;
          }
          
          if (item.endsWith('.tsx') && !item.includes('.test.')) {
            components++;
          }
          
          try {
            const content = fs.readFileSync(itemPath, 'utf8');
            totalLines += content.split('\n').length;
          } catch (error) {
            // Skip files that can't be read
          }
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanDirectory(srcDir);
  
  healthMetrics.codebase = {
    files: totalFiles,
    lines: totalLines,
    components,
    features,
    tests
  };
  
  console.log(`📁 Files: ${totalFiles}`);
  console.log(`📝 Lines of Code: ${totalLines.toLocaleString()}`);
  console.log(`🧩 Components: ${components}`);
  console.log(`🎯 Features: ${features}`);
  console.log(`🧪 Tests: ${tests}`);
  console.log();
}

analyzeCodebase();

// 2. DEPENDENCY HEALTH CHECK
console.log('📦 Checking dependency health...\n');

function checkDependencies() {
  const packageJsonPath = path.join(rootDir, 'package.json');
  if (!fs.existsSync(packageJsonPath)) return;
  
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const totalDeps = Object.keys({
    ...packageJson.dependencies || {},
    ...packageJson.devDependencies || {}
  }).length;
  
  healthMetrics.dependencies.total = totalDeps;
  
  // Check for outdated packages
  try {
    const outdated = execSync('npm outdated --json 2>/dev/null || echo "{}"', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    const outdatedPackages = JSON.parse(outdated);
    healthMetrics.dependencies.outdated = Object.keys(outdatedPackages).length;
  } catch (error) {
    healthMetrics.dependencies.outdated = 0;
  }
  
  // Check for vulnerabilities
  try {
    const audit = execSync('npm audit --json 2>/dev/null || echo "{\\"vulnerabilities\\":{}}"', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    const auditResult = JSON.parse(audit);
    if (auditResult.vulnerabilities) {
      healthMetrics.dependencies.vulnerabilities = Object.keys(auditResult.vulnerabilities).length;
    }
  } catch (error) {
    healthMetrics.dependencies.vulnerabilities = 0;
  }
  
  console.log(`📦 Total Dependencies: ${totalDeps}`);
  console.log(`📅 Outdated: ${healthMetrics.dependencies.outdated}`);
  console.log(`🔒 Vulnerabilities: ${healthMetrics.dependencies.vulnerabilities}`);
  console.log();
}

checkDependencies();

// 3. CODE QUALITY METRICS
console.log('✨ Analyzing code quality...\n');

function checkCodeQuality() {
  // TypeScript errors
  try {
    execSync('npx tsc --noEmit --skipLibCheck', { 
      stdio: 'ignore',
      cwd: rootDir 
    });
    healthMetrics.quality.typeErrors = 0;
    console.log('✅ TypeScript: No errors');
  } catch (error) {
    // Count TypeScript errors from output
    try {
      const tscOutput = execSync('npx tsc --noEmit --skipLibCheck 2>&1 || true', { 
        encoding: 'utf8',
        cwd: rootDir 
      });
      const errorMatches = tscOutput.match(/error TS\d+:/g);
      healthMetrics.quality.typeErrors = errorMatches ? errorMatches.length : 0;
      console.log(`❌ TypeScript: ${healthMetrics.quality.typeErrors} errors`);
    } catch {
      healthMetrics.quality.typeErrors = 0;
    }
  }
  
  // ESLint errors
  try {
    execSync('npx eslint src --format=json > /dev/null 2>&1', { 
      stdio: 'ignore',
      cwd: rootDir 
    });
    healthMetrics.quality.lintErrors = 0;
    console.log('✅ ESLint: No errors');
  } catch (error) {
    try {
      const lintOutput = execSync('npx eslint src --format=json 2>/dev/null || echo "[]"', { 
        encoding: 'utf8',
        cwd: rootDir 
      });
      const lintResults = JSON.parse(lintOutput);
      const totalErrors = lintResults.reduce((sum, file) => sum + file.errorCount, 0);
      healthMetrics.quality.lintErrors = totalErrors;
      console.log(`❌ ESLint: ${totalErrors} errors`);
    } catch {
      healthMetrics.quality.lintErrors = 0;
    }
  }
  
  // Test coverage (if available)
  try {
    if (fs.existsSync(path.join(rootDir, 'coverage/coverage-summary.json'))) {
      const coverage = JSON.parse(fs.readFileSync(path.join(rootDir, 'coverage/coverage-summary.json'), 'utf8'));
      healthMetrics.quality.testCoverage = Math.round(coverage.total.lines.pct);
      console.log(`🧪 Test Coverage: ${healthMetrics.quality.testCoverage}%`);
    } else {
      console.log('📊 Test Coverage: Not available (run npm run test:coverage)');
    }
  } catch (error) {
    console.log('📊 Test Coverage: Not available');
  }
  
  console.log();
}

checkCodeQuality();

// 4. PERFORMANCE METRICS
console.log('⚡ Measuring performance...\n');

function checkPerformance() {
  // Build time measurement
  try {
    console.log('🔨 Measuring build time...');
    const buildStart = Date.now();
    execSync('npm run build 2>/dev/null', { 
      stdio: 'ignore',
      cwd: rootDir 
    });
    const buildTime = Date.now() - buildStart;
    healthMetrics.performance.buildTime = Math.round(buildTime / 1000);
    console.log(`⏱️  Build Time: ${healthMetrics.performance.buildTime}s`);
    
    // Check bundle size
    const distDir = path.join(rootDir, 'dist');
    if (fs.existsSync(distDir)) {
      let totalSize = 0;
      function calculateSize(dir) {
        const items = fs.readdirSync(dir);
        items.forEach(item => {
          const itemPath = path.join(dir, item);
          const stat = fs.statSync(itemPath);
          if (stat.isDirectory()) {
            calculateSize(itemPath);
          } else {
            totalSize += stat.size;
          }
        });
      }
      calculateSize(distDir);
      healthMetrics.performance.bundleSize = Math.round(totalSize / 1024 / 1024 * 100) / 100;
      console.log(`📦 Bundle Size: ${healthMetrics.performance.bundleSize}MB`);
    }
  } catch (error) {
    console.log('⚠️  Build failed - check build configuration');
    healthMetrics.alerts.push({
      level: 'error',
      message: 'Build process failed',
      action: 'Check build configuration and dependencies'
    });
  }
  
  console.log();
}

checkPerformance();

// 5. GENERATE HEALTH ALERTS
console.log('🚨 Generating health alerts...\n');

function generateAlerts() {
  // Critical alerts
  if (healthMetrics.dependencies.vulnerabilities > 0) {
    healthMetrics.alerts.push({
      level: 'critical',
      message: `${healthMetrics.dependencies.vulnerabilities} security vulnerabilities detected`,
      action: 'Run npm audit fix immediately'
    });
  }
  
  if (healthMetrics.quality.typeErrors > 10) {
    healthMetrics.alerts.push({
      level: 'high',
      message: `${healthMetrics.quality.typeErrors} TypeScript errors detected`,
      action: 'Fix type errors to improve code reliability'
    });
  }
  
  // Warning alerts
  if (healthMetrics.dependencies.outdated > 20) {
    healthMetrics.alerts.push({
      level: 'warning',
      message: `${healthMetrics.dependencies.outdated} outdated dependencies`,
      action: 'Consider updating dependencies for security and features'
    });
  }
  
  if (healthMetrics.performance.buildTime > 60) {
    healthMetrics.alerts.push({
      level: 'warning',
      message: `Build time is ${healthMetrics.performance.buildTime}s (slow)`,
      action: 'Optimize build configuration and dependencies'
    });
  }
  
  if (healthMetrics.performance.bundleSize > 5) {
    healthMetrics.alerts.push({
      level: 'warning',
      message: `Bundle size is ${healthMetrics.performance.bundleSize}MB (large)`,
      action: 'Optimize bundle size with code splitting and tree shaking'
    });
  }
  
  // Info alerts
  if (healthMetrics.codebase.tests < healthMetrics.codebase.components * 0.5) {
    healthMetrics.alerts.push({
      level: 'info',
      message: `Test coverage may be low (${healthMetrics.codebase.tests} tests for ${healthMetrics.codebase.components} components)`,
      action: 'Consider adding more tests for better coverage'
    });
  }
  
  // Display alerts
  if (healthMetrics.alerts.length === 0) {
    console.log('✅ No health alerts - system is healthy!');
  } else {
    healthMetrics.alerts.forEach(alert => {
      const icon = alert.level === 'critical' ? '🔴' : 
                   alert.level === 'high' ? '🟠' :
                   alert.level === 'warning' ? '🟡' : '🔵';
      console.log(`${icon} ${alert.level.toUpperCase()}: ${alert.message}`);
      console.log(`   💡 Action: ${alert.action}\n`);
    });
  }
}

generateAlerts();

// 6. GENERATE RECOMMENDATIONS
function generateRecommendations() {
  const recs = [];
  
  if (healthMetrics.dependencies.vulnerabilities > 0) {
    recs.push('Address security vulnerabilities with npm audit fix');
  }
  
  if (healthMetrics.dependencies.outdated > 10) {
    recs.push('Update outdated dependencies for latest features and security');
  }
  
  if (healthMetrics.quality.typeErrors > 0) {
    recs.push('Fix TypeScript errors to improve code reliability');
  }
  
  if (healthMetrics.quality.lintErrors > 0) {
    recs.push('Fix ESLint errors to maintain code quality standards');
  }
  
  if (healthMetrics.performance.buildTime > 30) {
    recs.push('Optimize build performance with caching and parallelization');
  }
  
  if (healthMetrics.codebase.features > 10 && healthMetrics.codebase.tests < 50) {
    recs.push('Increase test coverage for better reliability');
  }
  
  recs.push('Set up automated health monitoring with alerts');
  recs.push('Implement performance budgets for builds and bundles');
  recs.push('Schedule regular dependency updates and security scans');
  
  healthMetrics.recommendations = recs;
}

generateRecommendations();

// 7. SAVE HEALTH REPORT
const healthReport = {
  ...healthMetrics,
  summary: {
    overallHealth: healthMetrics.alerts.filter(a => a.level === 'critical').length === 0 ? 
                   (healthMetrics.alerts.filter(a => a.level === 'high').length === 0 ? 'healthy' : 'warning') : 'critical',
    totalAlerts: healthMetrics.alerts.length,
    criticalIssues: healthMetrics.alerts.filter(a => a.level === 'critical').length,
    warningIssues: healthMetrics.alerts.filter(a => a.level === 'warning').length
  }
};

fs.writeFileSync(
  path.join(rootDir, 'health-report.json'),
  JSON.stringify(healthReport, null, 2)
);

// 8. DISPLAY HEALTH DASHBOARD
console.log('🏥 Enterprise Health Dashboard');
console.log('═'.repeat(60));
console.log(`📊 Overall Health: ${healthReport.summary.overallHealth.toUpperCase()}`);
console.log(`🚨 Total Alerts: ${healthReport.summary.totalAlerts}`);
console.log(`🔴 Critical: ${healthReport.summary.criticalIssues}`);
console.log(`🟡 Warnings: ${healthReport.summary.warningIssues}`);
console.log('═'.repeat(60));

console.log('\n📈 Key Metrics:');
console.log(`🗂️  Codebase: ${healthMetrics.codebase.files} files, ${healthMetrics.codebase.lines.toLocaleString()} lines`);
console.log(`🧩 Components: ${healthMetrics.codebase.components}`);
console.log(`🎯 Features: ${healthMetrics.codebase.features}`);
console.log(`📦 Dependencies: ${healthMetrics.dependencies.total} (${healthMetrics.dependencies.outdated} outdated)`);
console.log(`⚡ Build Time: ${healthMetrics.performance.buildTime}s`);
console.log(`📊 Bundle Size: ${healthMetrics.performance.bundleSize}MB`);

if (healthMetrics.recommendations.length > 0) {
  console.log('\n💡 Recommendations:');
  healthMetrics.recommendations.forEach((rec, i) => {
    console.log(`  ${i + 1}. ${rec}`);
  });
}

console.log('\n📄 Health report saved to: health-report.json');
console.log('\n🏥 Health monitoring complete!');

process.exit(healthReport.summary.criticalIssues > 0 ? 1 : 0);