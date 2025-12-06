#!/usr/bin/env tsx

import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

interface PackageJson {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}

interface DependencyAnalysis {
  name: string;
  version: string;
  type: 'production' | 'development';
  size?: string;
  description?: string;
  usage?: 'heavy' | 'moderate' | 'light' | 'unused';
}

const HEAVY_PACKAGES = [
  'react', 'react-dom', 'framer-motion', '@tanstack/react-query',
  '@supabase/supabase-js', 'recharts', '@stripe/stripe-js'
];

const MODERATE_PACKAGES = [
  'react-router-dom', 'react-hook-form', 'date-fns', 'lucide-react',
  'zod', '@radix-ui'
];

const LIGHT_PACKAGES = [
  'clsx', 'tailwind-merge', 'class-variance-authority'
];

function analyzeDependencies(): DependencyAnalysis[] {
  const packageJsonPath = resolve(process.cwd(), 'package.json');
  const packageJson: PackageJson = JSON.parse(readFileSync(packageJsonPath, 'utf-8'));
  
  const analysis: DependencyAnalysis[] = [];
  
  // Analyze production dependencies
  if (packageJson.dependencies) {
    for (const [name, version] of Object.entries(packageJson.dependencies)) {
      let usage: DependencyAnalysis['usage'] = 'light';
      
      if (HEAVY_PACKAGES.some(pkg => name.includes(pkg))) {
        usage = 'heavy';
      } else if (MODERATE_PACKAGES.some(pkg => name.includes(pkg))) {
        usage = 'moderate';
      }
      
      analysis.push({
        name,
        version,
        type: 'production',
        usage
      });
    }
  }
  
  // Analyze dev dependencies
  if (packageJson.devDependencies) {
    for (const [name, version] of Object.entries(packageJson.devDependencies)) {
      analysis.push({
        name,
        version,
        type: 'development',
        usage: 'light' // Dev deps don't affect bundle size
      });
    }
  }
  
  return analysis.sort((a, b) => {
    const usageOrder = { heavy: 3, moderate: 2, light: 1, unused: 0 };
    return usageOrder[b.usage || 'light'] - usageOrder[a.usage || 'light'];
  });
}

function generateOptimizationReport(analysis: DependencyAnalysis[]): string {
  const heavy = analysis.filter(d => d.usage === 'heavy' && d.type === 'production');
  const moderate = analysis.filter(d => d.usage === 'moderate' && d.type === 'production');
  const light = analysis.filter(d => d.usage === 'light' && d.type === 'production');
  
  let report = `# Dependency Analysis Report\n\n`;
  report += `Generated: ${new Date().toISOString()}\n\n`;
  
  report += `## Summary\n`;
  report += `- Total Dependencies: ${analysis.filter(d => d.type === 'production').length}\n`;
  report += `- Heavy Dependencies: ${heavy.length}\n`;
  report += `- Moderate Dependencies: ${moderate.length}\n`;
  report += `- Light Dependencies: ${light.length}\n\n`;
  
  report += `## Heavy Dependencies (Bundle Impact: High)\n`;
  heavy.forEach(dep => {
    report += `- **${dep.name}** (${dep.version})\n`;
  });
  report += '\n';
  
  report += `## Optimization Recommendations\n`;
  report += `### 1. Code Splitting Opportunities\n`;
  report += `Consider lazy loading these heavy components:\n`;
  report += `- Recharts components (charts)\n`;
  report += `- Framer Motion animations\n`;
  report += `- Radix UI components not used on initial load\n\n`;
  
  report += `### 2. Bundle Size Optimizations\n`;
  report += `- Use tree shaking for Radix UI components\n`;
  report += `- Consider lighter alternatives for date manipulation\n`;
  report += `- Optimize Lucide React icon imports\n`;
  report += `- Use dynamic imports for rarely used components\n\n`;
  
  report += `### 3. Performance Recommendations\n`;
  report += `- Implement React.memo for expensive components\n`;
  report += `- Use useMemo for complex calculations\n`;
  report += `- Lazy load non-critical routes\n`;
  report += `- Optimize image loading with intersection observer\n\n`;
  
  return report;
}

function main() {
  console.log('🔍 Analyzing dependencies...');
  
  const analysis = analyzeDependencies();
  const report = generateOptimizationReport(analysis);
  
  // Write detailed analysis to file
  const reportPath = resolve(process.cwd(), 'dependency-analysis.md');
  writeFileSync(reportPath, report);
  
  console.log(`📊 Analysis complete! Report saved to: ${reportPath}`);
  
  // Console summary
  console.log('\n📋 Quick Summary:');
  console.log(`Heavy Dependencies: ${analysis.filter(d => d.usage === 'heavy').length}`);
  console.log(`Moderate Dependencies: ${analysis.filter(d => d.usage === 'moderate').length}`);
  console.log(`Light Dependencies: ${analysis.filter(d => d.usage === 'light').length}`);
  
  console.log('\n🎯 Top Optimization Targets:');
  analysis
    .filter(d => d.usage === 'heavy' && d.type === 'production')
    .slice(0, 5)
    .forEach(dep => console.log(`  - ${dep.name}`));
}

// Run the main function
main();