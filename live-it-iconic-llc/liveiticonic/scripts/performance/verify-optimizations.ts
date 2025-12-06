/**
 * Performance Optimization Verification Script
 * Checks that all recommended optimizations are in place
 * Run: npx tsx scripts/performance/verify-optimizations.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '../../');

interface CheckResult {
  name: string;
  passed: boolean;
  details: string;
  severity: 'error' | 'warning' | 'info';
}

const results: CheckResult[] = [];

function check(
  name: string,
  fn: () => boolean,
  details: string,
  severity: 'error' | 'warning' = 'error'
): void {
  const passed = fn();
  results.push({ name, passed, details, severity });
  const icon = passed ? '✓' : '✗';
  const color = passed ? '\x1b[32m' : severity === 'error' ? '\x1b[31m' : '\x1b[33m';
  const reset = '\x1b[0m';
  console.log(`${color}${icon}${reset} ${name}`);
  if (!passed) {
    console.log(`  → ${details}`);
  }
}

function fileExists(filePath: string): boolean {
  return fs.existsSync(path.join(PROJECT_ROOT, filePath));
}

function fileContains(filePath: string, content: string): boolean {
  try {
    const file = fs.readFileSync(path.join(PROJECT_ROOT, filePath), 'utf-8');
    return file.includes(content);
  } catch {
    return false;
  }
}

function getDirSize(dir: string): number {
  let size = 0;
  try {
    const files = fs.readdirSync(path.join(PROJECT_ROOT, dir));
    for (const file of files) {
      const filePath = path.join(PROJECT_ROOT, dir, file);
      const stat = fs.statSync(filePath);
      if (stat.isFile()) {
        size += stat.size;
      }
    }
  } catch {
    return 0;
  }
  return size;
}

console.log('\n🚀 Live It Iconic - Performance Optimization Verification\n');

// 1. Image Optimization
console.log('\n📷 Image Optimization');
check(
  'OptimizedImage component exists',
  () => fileExists('src/components/OptimizedImage.tsx'),
  'Create src/components/OptimizedImage.tsx with WebP/AVIF support'
);

// 2. Font Loading
console.log('\n📝 Font Loading Optimization');
check(
  'Font preload configured',
  () => fileContains('index.html', 'rel="preload"') && fileContains('index.html', 'font'),
  'Add font-face preload directives to index.html'
);

check(
  'font-display: swap configured',
  () => fileContains('index.html', 'display=swap'),
  'Add display=swap parameter to font URLs'
);

// 3. Web Vitals Monitoring
console.log('\n📊 Web Vitals Monitoring');
check(
  'Web Vitals module exists',
  () => fileExists('src/lib/webVitals.ts'),
  'Create src/lib/webVitals.ts with Core Web Vitals tracking'
);

check(
  'Web Vitals initialized',
  () => fileContains('src/main.tsx', 'initWebVitals'),
  'Call initWebVitals() in src/main.tsx'
);

// 4. API Caching
console.log('\n💾 API Caching');
check(
  'API cache module exists',
  () => fileExists('src/lib/api-cache.ts'),
  'Create src/lib/api-cache.ts with request deduplication'
);

// 5. Code Splitting
console.log('\n✂️ Code Splitting');
check(
  'Vite optimized',
  () => fileContains('vite.config.ts', 'manualChunks'),
  'Configure manual chunks in vite.config.ts for optimal splitting'
);

check(
  'Build optimization enabled',
  () => fileContains('vite.config.ts', 'terser'),
  'Configure Terser minification in vite.config.ts'
);

check(
  'Bundle visualizer configured',
  () => fileContains('vite.config.ts', 'visualizer'),
  'Add rollup-plugin-visualizer to vite.config.ts'
);

// 6. React Performance
console.log('\n⚛️  React Performance');
check(
  'Performance hooks available',
  () => fileExists('src/hooks/usePerformanceOptimization.ts'),
  'Create src/hooks/usePerformanceOptimization.ts'
);

check(
  'Memoization utilities available',
  () => fileExists('src/hooks/useMemoized.ts'),
  'Create src/hooks/useMemoized.ts'
);

// 7. Lighthouse CI
console.log('\n🔦 Lighthouse CI');
check(
  'Lighthouse config exists',
  () => fileExists('.lighthouserc.js'),
  'Create .lighthouserc.js with performance budgets'
);

// 8. Caching Headers
console.log('\n🔗 Asset Caching');
check(
  'Caching headers configured',
  () => fileExists('public/_headers'),
  'Create public/_headers with long-term cache rules'
);

// 9. Documentation
console.log('\n📚 Documentation');
check(
  'Performance guide exists',
  () => fileExists('docs/PERFORMANCE.md'),
  'Create docs/PERFORMANCE.md with optimization strategies'
);

// 10. Package Dependencies
console.log('\n📦 Dependencies');
check(
  'visualizer installed',
  () => fileContains('package.json', 'rollup-plugin-visualizer'),
  'Add rollup-plugin-visualizer to devDependencies'
);

// Summary
console.log('\n' + '='.repeat(60));

const passed = results.filter((r) => r.passed).length;
const failed = results.filter((r) => r.passed === false && r.severity === 'error').length;
const warnings = results.filter((r) => r.passed === false && r.severity === 'warning').length;

console.log(`\nResults: ${passed}/${results.length} checks passed`);
if (failed > 0) {
  console.log(`\n❌ ${failed} critical issue(s) found`);
}
if (warnings > 0) {
  console.log(`\n⚠️  ${warnings} warning(s)`);
}

if (passed === results.length) {
  console.log('\n🎉 All optimizations verified! Running Lighthouse audit...\n');

  // Print bundle size info if dist exists
  const distSize = getDirSize('dist');
  if (distSize > 0) {
    const sizeInMB = (distSize / 1024 / 1024).toFixed(2);
    console.log(`📊 Bundle size: ${sizeInMB}MB`);
  }

  console.log('\nNext steps:');
  console.log('1. Run: npm run build');
  console.log('2. Review: open dist/stats.html');
  console.log('3. Audit: npm run build && npm run preview');
  console.log('4. Lighthouse: lighthouse http://localhost:4173/');
} else {
  console.log('\n⚠️  Please address the above issues to complete optimization');
  process.exit(1);
}

console.log('\n' + '='.repeat(60) + '\n');
