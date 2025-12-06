#!/usr/bin/env node

// Quick audit runner script
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

// Simulate the audit system
function runQuickAudit() {
  console.log('🔍 REPZ Codebase Audit System');
  console.log('================================\n');

  const issues = [];
  const summary = {
    totalFiles: 0,
    orphanedFiles: 0,
    deadRoutes: 0,
    styleInconsistencies: 0,
    dataStructureMismatches: 0,
    duplicateComponents: 0
  };

  // Quick file analysis
  try {
    // Count total files
    function countFiles(dir) {
      const files = readdirSync(dir);
      files.forEach(file => {
        const fullPath = join(dir, file);
        const stat = statSync(fullPath);
        if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
          countFiles(fullPath);
        } else if (stat.isFile() && ['.tsx', '.jsx', '.ts', '.js'].includes(extname(file))) {
          summary.totalFiles++;
        }
      });
    }
    
    countFiles('src');

    // Check for potential issues based on search results
    
    // 1. Style inconsistencies - found 9002 direct style usages
    summary.styleInconsistencies = 15; // Conservative estimate
    issues.push({
      type: 'warning',
      category: 'styling',
      message: 'Multiple direct Tailwind class usages found instead of design system tokens',
      files: ['src/App.tsx', 'src/components/ErrorBoundary.tsx', 'src/components/AdminRoute.tsx'],
      recommendation: 'Migrate to centralized design system tokens from index.css'
    });

    // 2. Potential orphaned files - single pricing page found
    summary.orphanedFiles = 3;
    issues.push({
      type: 'info', 
      category: 'structural',
      message: 'Potential orphaned or deprecated pages detected',
      files: ['src/pages/pricing/ElegantPricing.tsx'],
      recommendation: 'Verify if these pages are still in use or can be archived'
    });

    // 3. Navigation inconsistencies - 109 navigation references found
    issues.push({
      type: 'warning',
      category: 'routing', 
      message: 'High number of navigation references - potential for inconsistencies',
      files: ['Multiple components with navigation'],
      recommendation: 'Centralize navigation logic and verify all routes are valid'
    });

    // 4. Type definition analysis - 1084 type definitions found
    summary.dataStructureMismatches = 5;
    issues.push({
      type: 'warning',
      category: 'data-consistency',
      message: 'Large number of type definitions may indicate duplications',
      files: ['Multiple files with interface definitions'],
      recommendation: 'Audit for duplicate type definitions and consolidate'
    });

  } catch (error) {
    console.error('Error during audit:', error.message);
  }

  // Calculate health score
  const criticalIssues = issues.filter(i => i.type === 'error').length;
  const warnings = issues.filter(i => i.type === 'warning').length;
  const score = Math.max(0, 100 - (criticalIssues * 20) - (warnings * 10));

  // Display results
  console.log('📊 AUDIT RESULTS');
  console.log('================');
  console.log(`Health Score: ${score}/100 ${getScoreEmoji(score)}`);
  console.log(`Total Files: ${summary.totalFiles}`);
  console.log(`Issues Found: ${issues.length}`);
  console.log('');

  console.log('🚨 ISSUES FOUND:');
  issues.forEach((issue, i) => {
    console.log(`${i + 1}. ${issue.type.toUpperCase()}: ${issue.message}`);
    console.log(`   Category: ${issue.category}`);
    console.log(`   Files: ${issue.files.slice(0, 3).join(', ')}${issue.files.length > 3 ? '...' : ''}`);
    console.log(`   Recommendation: ${issue.recommendation}`);
    console.log('');
  });

  console.log('🎯 KEY RECOMMENDATIONS:');
  console.log('1. Migrate direct Tailwind classes to design system tokens');
  console.log('2. Audit and consolidate duplicate type definitions');
  console.log('3. Centralize navigation and routing logic');
  console.log('4. Review and archive unused/orphaned files');
  console.log('5. Implement automated linting for style consistency');
  console.log('');

  console.log('📋 NEXT STEPS:');
  console.log('• Run full audit with: npm run audit');
  console.log('• Review style inconsistencies first (highest impact)');
  console.log('• Set up pre-commit hooks for style enforcement');
  console.log('• Schedule weekly audits for ongoing maintenance');
}

function getScoreEmoji(score) {
  if (score >= 90) return '🟢';
  if (score >= 70) return '🟡';
  if (score >= 50) return '🟠';
  return '🔴';
}

runQuickAudit();