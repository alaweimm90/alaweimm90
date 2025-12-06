#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('📦 Enterprise Dependency Analyzer & Cleaner\n');

const analysis = {
  packageJson: null,
  unusedDependencies: [],
  outdatedDependencies: [],
  vulnerabilities: [],
  duplicates: [],
  heavyDependencies: [],
  devDependencyIssues: []
};

// 1. LOAD AND ANALYZE PACKAGE.JSON
function loadPackageAnalysis() {
  const packagePath = path.join(rootDir, 'package.json');
  if (!fs.existsSync(packagePath)) {
    console.error('❌ package.json not found');
    return false;
  }
  
  analysis.packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  console.log('📋 Package.json loaded successfully\n');
  return true;
}

// 2. FIND UNUSED DEPENDENCIES
async function findUnusedDependencies() {
  console.log('🔍 Analyzing unused dependencies...\n');
  
  try {
    // Install depcheck if not available
    try {
      execSync('npx depcheck --version', { stdio: 'ignore' });
    } catch {
      console.log('Installing depcheck...');
      execSync('npm install -D depcheck', { stdio: 'inherit' });
    }
    
    // Run depcheck
    const result = execSync('npx depcheck --json', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    
    const depcheckResult = JSON.parse(result);
    
    analysis.unusedDependencies = depcheckResult.dependencies || [];
    analysis.devDependencyIssues = depcheckResult.devDependencies || [];
    
    console.log(`Found ${analysis.unusedDependencies.length} unused dependencies:`);
    analysis.unusedDependencies.forEach(dep => {
      console.log(`  📦 ${dep}`);
    });
    
    if (analysis.devDependencyIssues.length > 0) {
      console.log(`\nFound ${analysis.devDependencyIssues.length} unused dev dependencies:`);
      analysis.devDependencyIssues.forEach(dep => {
        console.log(`  🛠️  ${dep}`);
      });
    }
    
    console.log();
  } catch (error) {
    console.log('⚠️  Depcheck analysis failed, continuing...\n');
  }
}

// 3. CHECK FOR OUTDATED DEPENDENCIES
async function checkOutdatedDependencies() {
  console.log('📅 Checking for outdated dependencies...\n');
  
  try {
    const result = execSync('npm outdated --json', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    
    const outdated = JSON.parse(result);
    
    Object.entries(outdated).forEach(([name, info]) => {
      analysis.outdatedDependencies.push({
        name,
        current: info.current,
        wanted: info.wanted,
        latest: info.latest,
        type: info.type
      });
    });
    
    console.log(`Found ${analysis.outdatedDependencies.length} outdated dependencies:`);
    analysis.outdatedDependencies.forEach(dep => {
      console.log(`  📦 ${dep.name}: ${dep.current} → ${dep.latest}`);
    });
    
  } catch (error) {
    // npm outdated returns non-zero exit code when outdated packages exist
    if (error.stdout) {
      try {
        const outdated = JSON.parse(error.stdout);
        Object.entries(outdated).forEach(([name, info]) => {
          analysis.outdatedDependencies.push({
            name,
            current: info.current,
            wanted: info.wanted,
            latest: info.latest,
            type: info.type
          });
        });
        
        console.log(`Found ${analysis.outdatedDependencies.length} outdated dependencies:`);
        analysis.outdatedDependencies.forEach(dep => {
          console.log(`  📦 ${dep.name}: ${dep.current} → ${dep.latest}`);
        });
      } catch {
        console.log('⚠️  Could not parse outdated dependencies');
      }
    }
  }
  
  console.log();
}

// 4. SECURITY VULNERABILITY CHECK
async function checkVulnerabilities() {
  console.log('🔒 Checking for security vulnerabilities...\n');
  
  try {
    const result = execSync('npm audit --json', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    
    const auditResult = JSON.parse(result);
    
    if (auditResult.vulnerabilities) {
      Object.entries(auditResult.vulnerabilities).forEach(([name, vuln]) => {
        analysis.vulnerabilities.push({
          name,
          severity: vuln.severity,
          range: vuln.range,
          via: vuln.via
        });
      });
      
      console.log(`Found ${analysis.vulnerabilities.length} vulnerabilities:`);
      analysis.vulnerabilities.forEach(vuln => {
        const severityColor = vuln.severity === 'critical' ? '🔴' :
                            vuln.severity === 'high' ? '🟠' :
                            vuln.severity === 'moderate' ? '🟡' : '🔵';
        console.log(`  ${severityColor} ${vuln.name}: ${vuln.severity}`);
      });
    } else {
      console.log('✅ No vulnerabilities found!');
    }
    
  } catch (error) {
    if (error.stdout) {
      try {
        const auditResult = JSON.parse(error.stdout);
        if (auditResult.vulnerabilities) {
          Object.entries(auditResult.vulnerabilities).forEach(([name, vuln]) => {
            analysis.vulnerabilities.push({
              name,
              severity: vuln.severity,
              range: vuln.range,
              via: vuln.via
            });
          });
          
          console.log(`Found ${analysis.vulnerabilities.length} vulnerabilities:`);
          analysis.vulnerabilities.forEach(vuln => {
            const severityColor = vuln.severity === 'critical' ? '🔴' :
                                 vuln.severity === 'high' ? '🟠' :
                                 vuln.severity === 'moderate' ? '🟡' : '🔵';
            console.log(`  ${severityColor} ${vuln.name}: ${vuln.severity}`);
          });
        }
      } catch {
        console.log('⚠️  Could not parse security audit');
      }
    }
  }
  
  console.log();
}

// 5. ANALYZE BUNDLE SIZE & HEAVY DEPENDENCIES
async function analyzeBundleSize() {
  console.log('📏 Analyzing bundle size and heavy dependencies...\n');
  
  try {
    // Try to use webpack-bundle-analyzer if available
    const nodeModulesPath = path.join(rootDir, 'node_modules');
    if (!fs.existsSync(nodeModulesPath)) {
      console.log('⚠️  node_modules not found, skipping bundle analysis');
      return;
    }
    
    // Analyze node_modules sizes
    const dependencies = Object.keys(analysis.packageJson.dependencies || {});
    
    dependencies.forEach(dep => {
      const depPath = path.join(nodeModulesPath, dep);
      if (fs.existsSync(depPath)) {
        try {
          const size = getDirSize(depPath);
          if (size > 5 * 1024 * 1024) { // > 5MB
            analysis.heavyDependencies.push({
              name: dep,
              size: size,
              sizeFormatted: formatBytes(size)
            });
          }
        } catch (error) {
          // Skip if can't read
        }
      }
    });
    
    analysis.heavyDependencies.sort((a, b) => b.size - a.size);
    
    console.log(`Found ${analysis.heavyDependencies.length} heavy dependencies (>5MB):`);
    analysis.heavyDependencies.forEach(dep => {
      console.log(`  📦 ${dep.name}: ${dep.sizeFormatted}`);
    });
    
  } catch (error) {
    console.log('⚠️  Bundle size analysis failed');
  }
  
  console.log();
}

// 6. FIND DUPLICATE DEPENDENCIES
function findDuplicateDependencies() {
  console.log('🔄 Checking for duplicate dependencies...\n');
  
  const allDeps = {
    ...analysis.packageJson.dependencies || {},
    ...analysis.packageJson.devDependencies || {}
  };
  
  const seen = new Set();
  const duplicates = [];
  
  Object.keys(allDeps).forEach(dep => {
    const baseName = dep.replace(/^@[\w-]+\//, ''); // Remove scope
    if (seen.has(baseName)) {
      duplicates.push(dep);
    } else {
      seen.add(baseName);
    }
  });
  
  analysis.duplicates = duplicates;
  
  if (duplicates.length > 0) {
    console.log(`Found ${duplicates.length} potential duplicate dependencies:`);
    duplicates.forEach(dep => {
      console.log(`  🔄 ${dep}`);
    });
  } else {
    console.log('✅ No duplicate dependencies found!');
  }
  
  console.log();
}

// 7. GENERATE CLEANUP SCRIPT
function generateCleanupScript() {
  console.log('🧹 Generating dependency cleanup script...\n');
  
  const cleanupCommands = [];
  
  // Remove unused dependencies
  if (analysis.unusedDependencies.length > 0) {
    const depsToRemove = analysis.unusedDependencies.join(' ');
    cleanupCommands.push(`# Remove unused dependencies`);
    cleanupCommands.push(`npm uninstall ${depsToRemove}`);
    cleanupCommands.push('');
  }
  
  // Remove unused dev dependencies
  if (analysis.devDependencyIssues.length > 0) {
    const devDepsToRemove = analysis.devDependencyIssues.join(' ');
    cleanupCommands.push(`# Remove unused dev dependencies`);
    cleanupCommands.push(`npm uninstall -D ${devDepsToRemove}`);
    cleanupCommands.push('');
  }
  
  // Update outdated dependencies
  if (analysis.outdatedDependencies.length > 0) {
    cleanupCommands.push(`# Update outdated dependencies`);
    analysis.outdatedDependencies.forEach(dep => {
      cleanupCommands.push(`npm install ${dep.name}@${dep.latest}`);
    });
    cleanupCommands.push('');
  }
  
  // Fix vulnerabilities
  if (analysis.vulnerabilities.length > 0) {
    cleanupCommands.push(`# Fix security vulnerabilities`);
    cleanupCommands.push(`npm audit fix`);
    cleanupCommands.push('');
  }
  
  const cleanupScript = `#!/bin/bash
# Dependency Cleanup Script
# Generated: ${new Date().toISOString()}

echo "🧹 Starting dependency cleanup..."

${cleanupCommands.join('\n')}

echo "✅ Dependency cleanup complete!"
echo "🔍 Run 'npm audit' to verify security status"
echo "📦 Run 'npm outdated' to check for remaining updates"
`;
  
  fs.writeFileSync(
    path.join(rootDir, 'scripts/cleanup-dependencies.sh'),
    cleanupScript
  );
  
  // Make executable
  try {
    fs.chmodSync(path.join(rootDir, 'scripts/cleanup-dependencies.sh'), '755');
  } catch (error) {
    // chmod might not work on all systems
  }
  
  console.log('✅ Cleanup script generated: scripts/cleanup-dependencies.sh');
}

// 8. GENERATE COMPREHENSIVE REPORT
function generateReport() {
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      totalDependencies: Object.keys(analysis.packageJson?.dependencies || {}).length,
      totalDevDependencies: Object.keys(analysis.packageJson?.devDependencies || {}).length,
      unusedDependencies: analysis.unusedDependencies.length,
      outdatedDependencies: analysis.outdatedDependencies.length,
      vulnerabilities: analysis.vulnerabilities.length,
      heavyDependencies: analysis.heavyDependencies.length,
      duplicates: analysis.duplicates.length
    },
    details: {
      unusedDependencies: analysis.unusedDependencies,
      outdatedDependencies: analysis.outdatedDependencies,
      vulnerabilities: analysis.vulnerabilities,
      heavyDependencies: analysis.heavyDependencies,
      duplicates: analysis.duplicates,
      devDependencyIssues: analysis.devDependencyIssues
    },
    recommendations: [
      'Remove unused dependencies to reduce bundle size',
      'Update outdated dependencies for security and features',
      'Fix security vulnerabilities immediately',
      'Consider alternatives for heavy dependencies',
      'Remove duplicate/redundant dependencies',
      'Set up automated dependency monitoring'
    ]
  };
  
  fs.writeFileSync(
    path.join(rootDir, 'dependency-analysis-report.json'),
    JSON.stringify(report, null, 2)
  );
  
  return report;
}

// UTILITY FUNCTIONS
function getDirSize(dirPath) {
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
  
  calculateSize(dirPath);
  return totalSize;
}

function formatBytes(bytes) {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

// MAIN EXECUTION
async function main() {
  console.log('🎯 Starting enterprise dependency analysis...\n');
  
  if (!loadPackageAnalysis()) return;
  
  await findUnusedDependencies();
  await checkOutdatedDependencies();
  await checkVulnerabilities();
  await analyzeBundleSize();
  findDuplicateDependencies();
  
  const report = generateReport();
  generateCleanupScript();
  
  // Display summary
  console.log('📊 Dependency Analysis Summary:');
  console.log('═'.repeat(60));
  console.log(`📦 Total Dependencies: ${report.summary.totalDependencies}`);
  console.log(`🛠️  Dev Dependencies: ${report.summary.totalDevDependencies}`);
  console.log(`🗑️  Unused: ${report.summary.unusedDependencies}`);
  console.log(`📅 Outdated: ${report.summary.outdatedDependencies}`);
  console.log(`🔒 Vulnerabilities: ${report.summary.vulnerabilities}`);
  console.log(`📏 Heavy (>5MB): ${report.summary.heavyDependencies}`);
  console.log(`🔄 Duplicates: ${report.summary.duplicates}`);
  console.log('═'.repeat(60));
  
  console.log('\n💡 Recommendations:');
  report.recommendations.forEach((rec, i) => {
    console.log(`  ${i + 1}. ${rec}`);
  });
  
  console.log('\n📄 Reports generated:');
  console.log('  • dependency-analysis-report.json');
  console.log('  • scripts/cleanup-dependencies.sh');
  
  console.log('\n🎉 Dependency analysis complete!');
}

main().catch(console.error);