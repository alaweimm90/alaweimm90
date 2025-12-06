#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('⚡ Enterprise Build Optimizer & Bundle Analyzer\n');

const buildReport = {
  timestamp: new Date().toISOString(),
  performance: {
    buildTime: 0,
    bundleSize: 0,
    firstLoadJs: 0,
    chunkCount: 0,
    largestChunk: 0
  },
  optimization: {
    treeshaking: { enabled: false, savings: 0 },
    codesplitting: { enabled: false, chunks: 0 },
    compression: { enabled: false, ratio: 0 },
    minification: { enabled: false, savings: 0 }
  },
  analysis: {
    duplicates: [],
    heavyDependencies: [],
    unusedExports: [],
    circularDependencies: []
  },
  recommendations: [],
  alerts: []
};

// 1. BUILD PERFORMANCE ANALYSIS
console.log('🔨 Analyzing build performance...\n');

function analyzeBuildPerformance() {
  try {
    console.log('📊 Running production build...');
    const buildStart = Date.now();
    
    // Clean previous build
    const distPath = path.join(rootDir, 'dist');
    if (fs.existsSync(distPath)) {
      fs.rmSync(distPath, { recursive: true });
    }
    
    // Run build with timing
    execSync('npm run build', { 
      stdio: 'pipe',
      cwd: rootDir 
    });
    
    const buildTime = Date.now() - buildStart;
    buildReport.performance.buildTime = Math.round(buildTime / 1000);
    
    console.log(`⏱️  Build completed in ${buildReport.performance.buildTime}s`);
    
    // Analyze bundle size
    if (fs.existsSync(distPath)) {
      const bundleAnalysis = analyzeBundleSize(distPath);
      Object.assign(buildReport.performance, bundleAnalysis);
      
      console.log(`📦 Total Bundle Size: ${buildReport.performance.bundleSize}MB`);
      console.log(`🚀 First Load JS: ${buildReport.performance.firstLoadJs}MB`);
      console.log(`📄 Chunk Count: ${buildReport.performance.chunkCount}`);
      console.log(`📈 Largest Chunk: ${buildReport.performance.largestChunk}MB`);
    }
    
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    buildReport.alerts.push({
      level: 'critical',
      message: 'Build process failed',
      action: 'Fix build configuration and dependencies'
    });
  }
  
  console.log();
}

function analyzeBundleSize(distPath) {
  let totalSize = 0;
  let firstLoadJs = 0;
  let chunkCount = 0;
  let largestChunk = 0;
  const chunks = [];
  
  function calculateSize(dir) {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory()) {
        calculateSize(itemPath);
      } else {
        const sizeInMB = stat.size / 1024 / 1024;
        totalSize += sizeInMB;
        
        if (item.endsWith('.js')) {
          chunkCount++;
          chunks.push({ name: item, size: sizeInMB });
          
          if (sizeInMB > largestChunk) {
            largestChunk = sizeInMB;
          }
          
          // Estimate first load JS (main chunks)
          if (item.includes('index') || item.includes('main') || item.includes('app')) {
            firstLoadJs += sizeInMB;
          }
        }
      }
    });
  }
  
  calculateSize(distPath);
  
  return {
    bundleSize: Math.round(totalSize * 100) / 100,
    firstLoadJs: Math.round(firstLoadJs * 100) / 100,
    chunkCount,
    largestChunk: Math.round(largestChunk * 100) / 100,
    chunks: chunks.sort((a, b) => b.size - a.size)
  };
}

analyzeBuildPerformance();

// 2. DEPENDENCY ANALYSIS
console.log('📦 Analyzing dependencies and bundle composition...\n');

function analyzeDependencies() {
  const packageJsonPath = path.join(rootDir, 'package.json');
  if (!fs.existsSync(packageJsonPath)) return;
  
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const dependencies = packageJson.dependencies || {};
  const devDependencies = packageJson.devDependencies || {};
  
  const heavyDependencies = [];
  const nodeModulesPath = path.join(rootDir, 'node_modules');
  
  if (fs.existsSync(nodeModulesPath)) {
    Object.keys(dependencies).forEach(dep => {
      const depPath = path.join(nodeModulesPath, dep);
      if (fs.existsSync(depPath)) {
        try {
          const size = getDirSize(depPath);
          const sizeInMB = size / 1024 / 1024;
          
          if (sizeInMB > 1) { // Dependencies larger than 1MB
            heavyDependencies.push({
              name: dep,
              size: Math.round(sizeInMB * 100) / 100,
              type: 'production'
            });
          }
        } catch (error) {
          // Skip if can't read directory
        }
      }
    });
  }
  
  heavyDependencies.sort((a, b) => b.size - a.size);
  buildReport.analysis.heavyDependencies = heavyDependencies.slice(0, 10);
  
  if (heavyDependencies.length > 0) {
    console.log(`📊 Top ${Math.min(10, heavyDependencies.length)} heaviest dependencies:`);
    heavyDependencies.slice(0, 10).forEach((dep, i) => {
      console.log(`  ${i + 1}. ${dep.name}: ${dep.size}MB`);
    });
  } else {
    console.log('✅ No unusually heavy dependencies detected');
  }
  
  console.log();
}

function getDirSize(dirPath) {
  let totalSize = 0;
  
  function calculateSize(dir) {
    try {
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
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  calculateSize(dirPath);
  return totalSize;
}

analyzeDependencies();

// 3. OPTIMIZATION DETECTION
console.log('🎯 Detecting build optimizations...\n');

function detectOptimizations() {
  const viteConfigPath = path.join(rootDir, 'vite.config.ts');
  const viteConfigJsPath = path.join(rootDir, 'vite.config.js');
  
  let configContent = '';
  if (fs.existsSync(viteConfigPath)) {
    configContent = fs.readFileSync(viteConfigPath, 'utf8');
  } else if (fs.existsSync(viteConfigJsPath)) {
    configContent = fs.readFileSync(viteConfigJsPath, 'utf8');
  }
  
  // Check for various optimizations
  const optimizations = {
    treeshaking: {
      enabled: configContent.includes('treeshake') || !configContent.includes('treeshake: false'),
      description: 'Dead code elimination'
    },
    codesplitting: {
      enabled: configContent.includes('splitChunks') || configContent.includes('manualChunks'),
      description: 'Dynamic imports and chunk splitting'
    },
    compression: {
      enabled: configContent.includes('compression') || configContent.includes('gzip'),
      description: 'Asset compression'
    },
    minification: {
      enabled: configContent.includes('minify') || !configContent.includes('minify: false'),
      description: 'Code minification'
    }
  };
  
  Object.entries(optimizations).forEach(([key, opt]) => {
    buildReport.optimization[key] = { enabled: opt.enabled };
    
    const status = opt.enabled ? '✅' : '❌';
    console.log(`${status} ${opt.description}: ${opt.enabled ? 'Enabled' : 'Disabled'}`);
  });
  
  console.log();
}

detectOptimizations();

// 4. CIRCULAR DEPENDENCY DETECTION
console.log('🔄 Detecting circular dependencies...\n');

function detectCircularDependencies() {
  const circularDeps = [];
  const visited = new Set();
  const visiting = new Set();
  const dependencies = new Map();
  
  // Build dependency graph
  function buildDependencyGraph(filePath, relativePath) {
    if (visiting.has(relativePath)) {
      // Found circular dependency
      circularDeps.push({
        file: relativePath,
        cycle: Array.from(visiting),
        severity: 'warning'
      });
      return;
    }
    
    if (visited.has(relativePath)) return;
    
    visiting.add(relativePath);
    
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const imports = extractImports(content);
      dependencies.set(relativePath, imports);
      
      imports.forEach(importPath => {
        if (importPath.startsWith('./') || importPath.startsWith('../')) {
          const fullImportPath = path.resolve(path.dirname(filePath), importPath);
          const importRelativePath = path.relative(rootDir, fullImportPath);
          
          if (fs.existsSync(fullImportPath + '.ts') || 
              fs.existsSync(fullImportPath + '.tsx') ||
              fs.existsSync(fullImportPath + '.js') ||
              fs.existsSync(fullImportPath + '.jsx')) {
            buildDependencyGraph(fullImportPath + '.ts', importRelativePath + '.ts');
          }
        }
      });
    } catch (error) {
      // Skip files that can't be read
    }
    
    visiting.delete(relativePath);
    visited.add(relativePath);
  }
  
  // Scan all TypeScript/JavaScript files
  function scanForFiles(dir, relativePath = '') {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        if (item === 'node_modules' || item === '.git' || item === 'dist') return;
        
        const itemPath = path.join(dir, item);
        const itemRelativePath = path.join(relativePath, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanForFiles(itemPath, itemRelativePath);
        } else if (item.endsWith('.ts') || item.endsWith('.tsx')) {
          buildDependencyGraph(itemPath, itemRelativePath);
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanForFiles(path.join(rootDir, 'src'));
  
  buildReport.analysis.circularDependencies = circularDeps;
  
  if (circularDeps.length > 0) {
    console.log(`⚠️  Found ${circularDeps.length} circular dependencies:`);
    circularDeps.slice(0, 5).forEach((dep, i) => {
      console.log(`  🔄 ${dep.file}`);
    });
    if (circularDeps.length > 5) {
      console.log(`  ... and ${circularDeps.length - 5} more`);
    }
  } else {
    console.log('✅ No circular dependencies detected');
  }
  
  console.log();
}

function extractImports(content) {
  const imports = [];
  const importRegex = /import\s+.*?\s+from\s+['"]([^'"]+)['"]/g;
  let match;
  
  while ((match = importRegex.exec(content)) !== null) {
    imports.push(match[1]);
  }
  
  return imports;
}

detectCircularDependencies();

// 5. DUPLICATE CODE DETECTION
console.log('🔀 Detecting duplicate code patterns...\n');

function detectDuplicates() {
  const duplicates = [];
  const codeHashes = new Map();
  
  function analyzeFile(filePath, relativePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const functions = extractFunctions(content);
      
      functions.forEach(func => {
        const hash = generateHash(func.code);
        if (codeHashes.has(hash)) {
          const existing = codeHashes.get(hash);
          duplicates.push({
            original: existing.file,
            duplicate: relativePath,
            function: func.name,
            lines: func.lines,
            similarity: 100
          });
        } else {
          codeHashes.set(hash, { file: relativePath, function: func.name });
        }
      });
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  function scanForDuplicates(dir, relativePath = '') {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        if (item === 'node_modules' || item === '.git') return;
        
        const itemPath = path.join(dir, item);
        const itemRelativePath = path.join(relativePath, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanForDuplicates(itemPath, itemRelativePath);
        } else if (item.endsWith('.ts') || item.endsWith('.tsx')) {
          analyzeFile(itemPath, itemRelativePath);
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanForDuplicates(path.join(rootDir, 'src'));
  
  buildReport.analysis.duplicates = duplicates;
  
  if (duplicates.length > 0) {
    console.log(`🔀 Found ${duplicates.length} potential duplicates:`);
    duplicates.slice(0, 5).forEach((dup, i) => {
      console.log(`  📋 ${dup.function} in ${dup.duplicate} (similar to ${dup.original})`);
    });
  } else {
    console.log('✅ No significant code duplication detected');
  }
  
  console.log();
}

function extractFunctions(content) {
  const functions = [];
  const funcRegex = /(function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>|export\s+function\s+\w+)\s*[^{]*{([^{}]*{[^{}]*}[^{}]*)*[^{}]*}/g;
  let match;
  
  while ((match = funcRegex.exec(content)) !== null) {
    const funcCode = match[0];
    const lines = funcCode.split('\n').length;
    
    if (lines > 5) { // Only consider functions with more than 5 lines
      functions.push({
        name: extractFunctionName(match[1]),
        code: funcCode,
        lines: lines
      });
    }
  }
  
  return functions;
}

function extractFunctionName(declaration) {
  const nameMatch = declaration.match(/(?:function\s+|const\s+)(\w+)/);
  return nameMatch ? nameMatch[1] : 'anonymous';
}

function generateHash(code) {
  // Simple hash function for code similarity
  const normalized = code
    .replace(/\s+/g, ' ')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '')
    .trim();
  
  let hash = 0;
  for (let i = 0; i < normalized.length; i++) {
    const char = normalized.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
}

detectDuplicates();

// 6. GENERATE PERFORMANCE ALERTS
console.log('🚨 Generating performance alerts...\n');

function generatePerformanceAlerts() {
  const alerts = [];
  
  // Build time alerts
  if (buildReport.performance.buildTime > 60) {
    alerts.push({
      level: 'warning',
      category: 'build-performance',
      message: `Build time is ${buildReport.performance.buildTime}s (slow)`,
      action: 'Optimize build configuration, enable caching, or parallelize builds'
    });
  }
  
  // Bundle size alerts
  if (buildReport.performance.bundleSize > 5) {
    alerts.push({
      level: 'warning',
      category: 'bundle-size',
      message: `Bundle size is ${buildReport.performance.bundleSize}MB (large)`,
      action: 'Implement code splitting, tree shaking, and remove unused dependencies'
    });
  }
  
  if (buildReport.performance.firstLoadJs > 2) {
    alerts.push({
      level: 'high',
      category: 'performance',
      message: `First load JS is ${buildReport.performance.firstLoadJs}MB (very large)`,
      action: 'Split main bundle and lazy load non-critical features'
    });
  }
  
  // Chunk alerts
  if (buildReport.performance.chunkCount > 50) {
    alerts.push({
      level: 'info',
      category: 'optimization',
      message: `High chunk count (${buildReport.performance.chunkCount})`,
      action: 'Consider consolidating smaller chunks to reduce HTTP requests'
    });
  }
  
  // Optimization alerts
  if (!buildReport.optimization.codesplitting?.enabled) {
    alerts.push({
      level: 'info',
      category: 'optimization',
      message: 'Code splitting not enabled',
      action: 'Enable code splitting to improve initial load performance'
    });
  }
  
  if (!buildReport.optimization.compression?.enabled) {
    alerts.push({
      level: 'info',
      category: 'optimization',
      message: 'Asset compression not enabled',
      action: 'Enable gzip/brotli compression to reduce transfer sizes'
    });
  }
  
  // Dependency alerts
  const heavyDeps = buildReport.analysis.heavyDependencies.filter(d => d.size > 5);
  if (heavyDeps.length > 0) {
    alerts.push({
      level: 'warning',
      category: 'dependencies',
      message: `${heavyDeps.length} very heavy dependencies detected`,
      action: 'Consider lighter alternatives or lazy loading for heavy dependencies'
    });
  }
  
  buildReport.alerts = alerts;
  
  if (alerts.length > 0) {
    alerts.forEach(alert => {
      const levelIcon = alert.level === 'critical' ? '🔴' :
                       alert.level === 'high' ? '🟠' :
                       alert.level === 'warning' ? '🟡' : '🔵';
      console.log(`${levelIcon} ${alert.level.toUpperCase()}: ${alert.message}`);
      console.log(`   💡 Action: ${alert.action}\n`);
    });
  } else {
    console.log('✅ No performance alerts - build is well optimized!');
  }
  
  console.log();
}

generatePerformanceAlerts();

// 7. GENERATE OPTIMIZATION RECOMMENDATIONS
function generateOptimizationRecommendations() {
  const recommendations = [];
  
  // Build performance recommendations
  if (buildReport.performance.buildTime > 30) {
    recommendations.push('Enable build caching and incremental builds');
    recommendations.push('Use parallel processing for builds');
    recommendations.push('Optimize TypeScript compilation settings');
  }
  
  // Bundle optimization recommendations
  if (buildReport.performance.bundleSize > 3) {
    recommendations.push('Implement dynamic imports for route-based code splitting');
    recommendations.push('Enable tree shaking to eliminate dead code');
    recommendations.push('Use bundle analyzer to identify optimization opportunities');
  }
  
  // Dependency recommendations
  if (buildReport.analysis.heavyDependencies.length > 3) {
    recommendations.push('Audit heavy dependencies and consider lighter alternatives');
    recommendations.push('Implement lazy loading for non-critical dependencies');
    recommendations.push('Use CDN for common libraries to leverage browser caching');
  }
  
  // Code quality recommendations
  if (buildReport.analysis.duplicates.length > 0) {
    recommendations.push('Refactor duplicate code into reusable components/utilities');
    recommendations.push('Implement code sharing patterns and utility libraries');
  }
  
  if (buildReport.analysis.circularDependencies.length > 0) {
    recommendations.push('Resolve circular dependencies to improve bundle efficiency');
    recommendations.push('Restructure imports to create cleaner dependency graphs');
  }
  
  // General optimization recommendations
  recommendations.push('Implement performance budgets in CI/CD pipeline');
  recommendations.push('Set up automated bundle analysis on pull requests');
  recommendations.push('Monitor Core Web Vitals in production');
  recommendations.push('Implement service worker for offline caching');
  
  buildReport.recommendations = recommendations;
}

generateOptimizationRecommendations();

// 8. SAVE BUILD REPORT
fs.writeFileSync(
  path.join(rootDir, 'build-optimization-report.json'),
  JSON.stringify(buildReport, null, 2)
);

// 9. DISPLAY BUILD OPTIMIZATION DASHBOARD
console.log('⚡ Enterprise Build Optimization Dashboard');
console.log('═'.repeat(60));
console.log(`⏱️  Build Time: ${buildReport.performance.buildTime}s`);
console.log(`📦 Bundle Size: ${buildReport.performance.bundleSize}MB`);
console.log(`🚀 First Load JS: ${buildReport.performance.firstLoadJs}MB`);
console.log(`📄 Chunks: ${buildReport.performance.chunkCount}`);
console.log(`🚨 Alerts: ${buildReport.alerts.length}`);
console.log('═'.repeat(60));

console.log('\n🎯 Optimization Status:');
Object.entries(buildReport.optimization).forEach(([key, opt]) => {
  const status = opt.enabled ? '✅' : '❌';
  console.log(`${status} ${key.charAt(0).toUpperCase() + key.slice(1)}: ${opt.enabled ? 'Enabled' : 'Disabled'}`);
});

console.log('\n📊 Analysis Results:');
console.log(`🔄 Circular Dependencies: ${buildReport.analysis.circularDependencies.length}`);
console.log(`🔀 Code Duplicates: ${buildReport.analysis.duplicates.length}`);
console.log(`📦 Heavy Dependencies: ${buildReport.analysis.heavyDependencies.length}`);

if (buildReport.recommendations.length > 0) {
  console.log('\n💡 Optimization Recommendations:');
  buildReport.recommendations.forEach((rec, i) => {
    console.log(`  ${i + 1}. ${rec}`);
  });
}

console.log('\n📄 Build optimization report saved to: build-optimization-report.json');
console.log('\n⚡ Build optimization analysis complete!');

// Exit with warning if significant issues found
const criticalAlerts = buildReport.alerts.filter(a => a.level === 'critical' || a.level === 'high').length;
process.exit(criticalAlerts > 0 ? 1 : 0);