#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🏢 Enterprise-Level Cleanup Phase 2 - Advanced Analysis\n');

const analysis = {
  ghostRoutes: [],
  deadPages: [],
  redundantFiles: [],
  deprecatedComponents: [],
  unusedDependencies: [],
  documentationIssues: [],
  securityConcerns: [],
  performanceIssues: []
};

// 1. GHOST ROUTES & DEAD PAGES ANALYSIS
console.log('👻 Analyzing ghost routes and dead pages...\n');

function analyzeRoutes() {
  const appTsxPath = path.join(rootDir, 'src/App.tsx');
  if (!fs.existsSync(appTsxPath)) return;
  
  const appContent = fs.readFileSync(appTsxPath, 'utf8');
  
  // Extract all route definitions
  const routeRegex = /<Route\s+path="([^"]+)"\s+element={[^}]+}/g;
  const routes = [];
  let match;
  
  while ((match = routeRegex.exec(appContent)) !== null) {
    routes.push(match[1]);
  }
  
  // Check for referenced but non-existent pages
  const pageImportRegex = /import\s+\w+\s+from\s+["']\.\/pages\/([^"']+)["']/g;
  const importedPages = [];
  
  while ((match = pageImportRegex.exec(appContent)) !== null) {
    importedPages.push(match[1]);
  }
  
  // Find ghost routes (routes without components)
  routes.forEach(route => {
    if (route !== '*' && route !== '/' && !route.includes(':')) {
      const expectedPagePath = route.replace('/', '').replace(/\//g, '/');
      const pageExists = importedPages.some(page => 
        page.toLowerCase().includes(expectedPagePath.toLowerCase())
      );
      
      if (!pageExists) {
        analysis.ghostRoutes.push({
          route,
          issue: 'Route defined but no corresponding page component found',
          severity: 'medium'
        });
      }
    }
  });
  
  // Find dead pages (pages without routes)
  const pagesDir = path.join(rootDir, 'src/pages');
  if (fs.existsSync(pagesDir)) {
    function scanPagesDir(dir, relativePath = '') {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanPagesDir(itemPath, path.join(relativePath, item));
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          const pageName = path.join(relativePath, item).replace(/\.(tsx|ts)$/, '');
          const isReferenced = appContent.includes(pageName) || 
                             appContent.includes(item.replace(/\.(tsx|ts)$/, ''));
          
          if (!isReferenced && !item.includes('index') && !item.includes('NotFound')) {
            analysis.deadPages.push({
              file: path.join('src/pages', relativePath, item),
              issue: 'Page component exists but not referenced in routes',
              severity: 'high'
            });
          }
        }
      });
    }
    
    scanPagesDir(pagesDir);
  }
}

analyzeRoutes();

// 2. REDUNDANT FILES ANALYSIS
console.log('🔄 Analyzing redundant and duplicate files...\n');

function findRedundantFiles() {
  const fileHashes = new Map();
  const duplicates = [];
  
  function scanForDuplicates(dir) {
    if (dir.includes('node_modules') || dir.includes('_graveyard') || dir.includes('.git')) return;
    
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanForDuplicates(itemPath);
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          try {
            const content = fs.readFileSync(itemPath, 'utf8');
            const normalizedContent = content
              .replace(/\s+/g, ' ')
              .replace(/\/\*[\s\S]*?\*\//g, '')
              .replace(/\/\/.*$/gm, '')
              .trim();
            
            if (normalizedContent.length > 100) { // Only check substantial files
              const existingFile = fileHashes.get(normalizedContent);
              if (existingFile) {
                duplicates.push({
                  original: existingFile,
                  duplicate: path.relative(rootDir, itemPath),
                  size: stat.size
                });
              } else {
                fileHashes.set(normalizedContent, path.relative(rootDir, itemPath));
              }
            }
          } catch (error) {
            // Skip files that can't be read
          }
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanForDuplicates(path.join(rootDir, 'src'));
  analysis.redundantFiles = duplicates;
}

findRedundantFiles();

// 3. DEPRECATED COMPONENTS ANALYSIS
console.log('🚫 Analyzing deprecated and legacy components...\n');

function findDeprecatedComponents() {
  const deprecatedPatterns = [
    /\/\*\s*@deprecated/i,
    /\/\/\s*@deprecated/i,
    /\/\*\s*TODO:\s*remove/i,
    /\/\/\s*TODO:\s*remove/i,
    /\/\*\s*LEGACY/i,
    /\/\/\s*LEGACY/i,
    /\/\*\s*FIXME/i,
    /\/\/\s*FIXME/i,
    /console\.log\(/g,
    /console\.warn\(/g,
    /debugger\s*;/g
  ];
  
  function scanForDeprecated(dir) {
    if (dir.includes('node_modules') || dir.includes('_graveyard')) return;
    
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanForDeprecated(itemPath);
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          try {
            const content = fs.readFileSync(itemPath, 'utf8');
            const relativePath = path.relative(rootDir, itemPath);
            
            deprecatedPatterns.forEach((pattern, index) => {
              const matches = content.match(pattern);
              if (matches) {
                const issueType = index < 2 ? 'deprecated' : 
                                index < 4 ? 'todo-remove' :
                                index < 6 ? 'legacy' :
                                index < 8 ? 'fixme' : 'debug-code';
                
                analysis.deprecatedComponents.push({
                  file: relativePath,
                  issue: `Contains ${issueType} markers or code`,
                  matches: matches.length,
                  severity: issueType === 'debug-code' ? 'high' : 'medium'
                });
              }
            });
          } catch (error) {
            // Skip files that can't be read
          }
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanForDeprecated(path.join(rootDir, 'src'));
}

findDeprecatedComponents();

// 4. DOCUMENTATION ISSUES ANALYSIS
console.log('📚 Analyzing documentation quality and issues...\n');

function analyzeDocumentation() {
  const issues = [];
  
  // Check for missing README files in feature directories
  const featuresDir = path.join(rootDir, 'src/features');
  if (fs.existsSync(featuresDir)) {
    const features = fs.readdirSync(featuresDir);
    features.forEach(feature => {
      const featureDir = path.join(featuresDir, feature);
      const readmePath = path.join(featureDir, 'README.md');
      
      if (!fs.existsSync(readmePath) && fs.statSync(featureDir).isDirectory()) {
        issues.push({
          type: 'missing-readme',
          location: `src/features/${feature}/README.md`,
          severity: 'medium',
          issue: 'Feature module missing documentation'
        });
      }
    });
  }
  
  // Check for outdated documentation
  const docsDir = path.join(rootDir, 'docs');
  if (fs.existsSync(docsDir)) {
    function scanDocs(dir) {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanDocs(itemPath);
        } else if (item.endsWith('.md')) {
          const content = fs.readFileSync(itemPath, 'utf8');
          const relativePath = path.relative(rootDir, itemPath);
          
          // Check for outdated dates
          const dateRegex = /(?:updated?|modified|created?):\s*(\d{4})/gi;
          let match;
          while ((match = dateRegex.exec(content)) !== null) {
            const year = parseInt(match[1]);
            const currentYear = new Date().getFullYear();
            
            if (currentYear - year > 1) {
              issues.push({
                type: 'outdated-doc',
                location: relativePath,
                severity: 'low',
                issue: `Documentation may be outdated (last updated: ${year})`
              });
            }
          }
          
          // Check for broken internal links
          const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
          while ((match = linkRegex.exec(content)) !== null) {
            const linkPath = match[2];
            if (linkPath.startsWith('./') || linkPath.startsWith('../')) {
              const fullLinkPath = path.resolve(path.dirname(itemPath), linkPath);
              if (!fs.existsSync(fullLinkPath)) {
                issues.push({
                  type: 'broken-link',
                  location: relativePath,
                  severity: 'medium',
                  issue: `Broken internal link: ${linkPath}`
                });
              }
            }
          }
        }
      });
    }
    
    scanDocs(docsDir);
  }
  
  analysis.documentationIssues = issues;
}

analyzeDocumentation();

// 5. DEPENDENCY ANALYSIS
console.log('📦 Analyzing dependencies and package health...\n');

function analyzeDependencies() {
  const packageJsonPath = path.join(rootDir, 'package.json');
  if (!fs.existsSync(packageJsonPath)) return;
  
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const allDeps = {
    ...packageJson.dependencies || {},
    ...packageJson.devDependencies || {}
  };
  
  const issues = [];
  
  // Check for potential security issues (simplified check)
  Object.keys(allDeps).forEach(dep => {
    const version = allDeps[dep];
    
    // Check for outdated major versions (simplified)
    if (version.includes('^0.') || version.includes('~0.')) {
      issues.push({
        dependency: dep,
        version,
        issue: 'Pre-1.0 version may be unstable',
        severity: 'low'
      });
    }
    
    // Check for exact versions (potential maintenance issue)
    if (!version.includes('^') && !version.includes('~') && !version.includes('>=')) {
      issues.push({
        dependency: dep,
        version,
        issue: 'Exact version pinning may prevent security updates',
        severity: 'medium'
      });
    }
  });
  
  analysis.unusedDependencies = issues;
}

analyzeDependencies();

// GENERATE REPORT
console.log('📊 Generating enterprise cleanup analysis report...\n');

const report = {
  timestamp: new Date().toISOString(),
  summary: {
    ghostRoutes: analysis.ghostRoutes.length,
    deadPages: analysis.deadPages.length,
    redundantFiles: analysis.redundantFiles.length,
    deprecatedComponents: analysis.deprecatedComponents.length,
    documentationIssues: analysis.documentationIssues.length,
    dependencyIssues: analysis.unusedDependencies.length
  },
  details: analysis,
  recommendations: [
    'Remove or fix ghost routes that point to non-existent components',
    'Delete dead page components that are not referenced in routing',
    'Eliminate duplicate/redundant files to reduce maintenance burden',
    'Clean up deprecated components and debug code',
    'Fix broken documentation links and update outdated content',
    'Review dependency versions for security and maintenance',
    'Implement automated checks to prevent regression'
  ]
};

// Save detailed report
fs.writeFileSync(
  path.join(rootDir, 'enterprise-analysis-report.json'),
  JSON.stringify(report, null, 2)
);

// Display summary
console.log('🏢 Enterprise Analysis Summary:');
console.log('═'.repeat(60));
console.log(`👻 Ghost Routes: ${report.summary.ghostRoutes}`);
console.log(`💀 Dead Pages: ${report.summary.deadPages}`);
console.log(`🔄 Redundant Files: ${report.summary.redundantFiles}`);
console.log(`🚫 Deprecated Components: ${report.summary.deprecatedComponents}`);
console.log(`📚 Documentation Issues: ${report.summary.documentationIssues}`);
console.log(`📦 Dependency Issues: ${report.summary.dependencyIssues}`);
console.log('═'.repeat(60));

if (report.summary.ghostRoutes > 0) {
  console.log('\n👻 Ghost Routes Found:');
  analysis.ghostRoutes.forEach(route => {
    console.log(`  ⚠️  ${route.route}: ${route.issue}`);
  });
}

if (report.summary.deadPages > 0) {
  console.log('\n💀 Dead Pages Found:');
  analysis.deadPages.slice(0, 10).forEach(page => {
    console.log(`  🗑️  ${page.file}: ${page.issue}`);
  });
  if (analysis.deadPages.length > 10) {
    console.log(`  ... and ${analysis.deadPages.length - 10} more`);
  }
}

if (report.summary.redundantFiles > 0) {
  console.log('\n🔄 Redundant Files Found:');
  analysis.redundantFiles.slice(0, 5).forEach(dup => {
    console.log(`  📋 ${dup.duplicate} (duplicate of ${dup.original})`);
  });
}

console.log('\n💡 Next Steps:');
report.recommendations.forEach((rec, i) => {
  console.log(`  ${i + 1}. ${rec}`);
});

console.log('\n✅ Enterprise analysis complete!');
console.log(`📄 Detailed report saved to: enterprise-analysis-report.json`);

process.exit(0);