#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🔍 Comprehensive Codebase Health Audit & System Consolidation\n');

const auditReport = {
  timestamp: new Date().toISOString(),
  summary: {
    totalIssues: 0,
    criticalIssues: 0,
    organizationScore: 0,
    functionalityScore: 0
  },
  routes: {
    defined: [],
    broken: [],
    unused: [],
    missing: []
  },
  components: {
    total: 0,
    functional: 0,
    broken: 0,
    unused: 0,
    duplicates: []
  },
  themes: {
    scattered: [],
    inconsistent: [],
    centralized: false
  },
  buttons: {
    total: 0,
    functional: 0,
    broken: [],
    missing: []
  },
  architecture: {
    issues: [],
    improvements: [],
    consolidationOpportunities: []
  },
  actions: []
};

// 1. COMPREHENSIVE ROUTE ANALYSIS
console.log('🗺️ Analyzing routing system...\n');

function analyzeRoutes() {
  const appTsxPath = path.join(rootDir, 'src/App.tsx');
  if (!fs.existsSync(appTsxPath)) {
    auditReport.routes.missing.push('src/App.tsx not found');
    return;
  }

  const appContent = fs.readFileSync(appTsxPath, 'utf8');
  
  // Extract route definitions
  const routeMatches = appContent.matchAll(/<Route[^>]*path=['"](.*?)['"][^>]*>/g);
  const definedRoutes = Array.from(routeMatches).map(match => match[1]);
  
  auditReport.routes.defined = definedRoutes;
  
  console.log(`📍 Found ${definedRoutes.length} defined routes:`);
  definedRoutes.forEach(route => {
    console.log(`  • ${route}`);
  });
  
  // Check for route implementations
  const pagesDir = path.join(rootDir, 'src/pages');
  const featuresDir = path.join(rootDir, 'src/features');
  
  definedRoutes.forEach(route => {
    if (route === '*' || route === '/' || route.includes(':')) return;
    
    const routeName = route.replace(/^\/+/, '').replace(/\/+/g, '-') || 'index';
    // Convert kebab-case to PascalCase for React component naming
    const pascalCaseName = routeName
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join('');
    
    const possibleFiles = [
      path.join(pagesDir, `${routeName}.tsx`),
      path.join(pagesDir, `${pascalCaseName}.tsx`),
      path.join(pagesDir, `${routeName}.ts`),
      path.join(pagesDir, routeName, 'index.tsx'),
      // Check in auth subdirectory
      path.join(pagesDir, 'auth', `${routeName}.tsx`),
      path.join(pagesDir, 'auth', `${pascalCaseName}.tsx`),
      // Check in features
      ...findInFeatures(routeName),
      ...findInFeatures(pascalCaseName)
    ];
    
    const exists = possibleFiles.some(file => fs.existsSync(file));
    if (!exists) {
      auditReport.routes.broken.push({
        route,
        issue: 'No corresponding component found',
        expectedFiles: possibleFiles.slice(0, 3).map(f => path.relative(rootDir, f))
      });
    }
  });
  
  console.log(`❌ Found ${auditReport.routes.broken.length} broken routes`);
  auditReport.routes.broken.forEach(broken => {
    console.log(`  • ${broken.route}: ${broken.issue}`);
  });
  
  console.log();
}

function findInFeatures(routeName) {
  const featuresDir = path.join(rootDir, 'src/features');
  const possibilities = [];
  
  if (fs.existsSync(featuresDir)) {
    const features = fs.readdirSync(featuresDir);
    features.forEach(feature => {
      const featureDir = path.join(featuresDir, feature);
      if (fs.statSync(featureDir).isDirectory()) {
        possibilities.push(
          path.join(featureDir, 'pages', `${routeName}.tsx`),
          path.join(featureDir, 'components', `${routeName}.tsx`)
        );
      }
    });
  }
  
  return possibilities;
}

analyzeRoutes();

// 2. COMPONENT ANALYSIS & ORGANIZATION ASSESSMENT
console.log('🧩 Analyzing component organization...\n');

function analyzeComponents() {
  const componentDirs = [
    path.join(rootDir, 'src/components'),
    path.join(rootDir, 'src/pages'),
    path.join(rootDir, 'src/features'),
    path.join(rootDir, 'src/ui')
  ];
  
  let totalComponents = 0;
  const componentMap = new Map();
  const duplicateNames = new Map();
  
  componentDirs.forEach(dir => {
    if (fs.existsSync(dir)) {
      scanForComponents(dir, '', componentMap, duplicateNames);
    }
  });
  
  totalComponents = componentMap.size;
  auditReport.components.total = totalComponents;
  
  // Find duplicates
  const duplicates = Array.from(duplicateNames.entries())
    .filter(([name, paths]) => paths.length > 1)
    .map(([name, paths]) => ({ name, paths }));
  
  auditReport.components.duplicates = duplicates;
  
  console.log(`📊 Component Analysis:`);
  console.log(`  • Total components: ${totalComponents}`);
  console.log(`  • Duplicate names: ${duplicates.length}`);
  
  if (duplicates.length > 0) {
    console.log(`  🔄 Duplicate components found:`);
    duplicates.forEach(dup => {
      console.log(`    • ${dup.name}: ${dup.paths.length} instances`);
      dup.paths.forEach(p => console.log(`      - ${p}`));
    });
  }
  
  console.log();
}

function scanForComponents(dir, relativePath, componentMap, duplicateNames) {
  try {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const itemRelativePath = path.join(relativePath, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory() && !item.includes('node_modules')) {
        scanForComponents(itemPath, itemRelativePath, componentMap, duplicateNames);
      } else if (item.endsWith('.tsx') && !item.includes('.test.') && !item.includes('.spec.')) {
        const componentName = item.replace('.tsx', '');
        const fullPath = path.relative(rootDir, itemPath);
        
        componentMap.set(fullPath, componentName);
        
        if (!duplicateNames.has(componentName)) {
          duplicateNames.set(componentName, []);
        }
        duplicateNames.get(componentName).push(fullPath);
      }
    });
  } catch (error) {
    // Skip directories that can't be read
  }
}

analyzeComponents();

// 3. THEME & STYLING CONSOLIDATION ANALYSIS
console.log('🎨 Analyzing theme and styling organization...\n');

function analyzeThemes() {
  const stylingFiles = [];
  const themeIssues = [];
  
  // Find all styling-related files
  function findStylingFiles(dir, relativePath = '') {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const itemRelativePath = path.join(relativePath, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory() && item !== 'node_modules') {
          findStylingFiles(itemPath, itemRelativePath);
        } else if (
          item.endsWith('.css') || 
          item.endsWith('.scss') || 
          item.includes('style') || 
          item.includes('theme') ||
          item === 'tailwind.config.js'
        ) {
          stylingFiles.push({
            file: itemRelativePath,
            type: getStyleType(item),
            size: stat.size
          });
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  findStylingFiles(rootDir);
  
  // Analyze theme consolidation
  const themeFiles = stylingFiles.filter(f => f.type === 'theme');
  const scatteredStyles = stylingFiles.filter(f => f.type === 'component-style');
  
  console.log(`🎨 Styling Analysis:`);
  console.log(`  • Total styling files: ${stylingFiles.length}`);
  console.log(`  • Theme files: ${themeFiles.length}`);
  console.log(`  • Scattered styles: ${scatteredStyles.length}`);
  
  // Check for theme centralization
  const hasCentralTheme = themeFiles.some(f => 
    f.file.includes('theme') || f.file.includes('globals') || f.file === 'tailwind.config.js'
  );
  
  auditReport.themes.centralized = hasCentralTheme;
  auditReport.themes.scattered = scatteredStyles.map(f => f.file);
  
  if (!hasCentralTheme) {
    themeIssues.push('No centralized theme system detected');
  }
  
  if (scatteredStyles.length > 10) {
    themeIssues.push(`High number of scattered style files (${scatteredStyles.length})`);
  }
  
  auditReport.themes.inconsistent = themeIssues;
  
  console.log(`  🎯 Theme centralization: ${hasCentralTheme ? '✅ Yes' : '❌ No'}`);
  if (themeIssues.length > 0) {
    console.log(`  ⚠️ Issues found:`);
    themeIssues.forEach(issue => console.log(`    • ${issue}`));
  }
  
  console.log();
}

function getStyleType(filename) {
  if (filename.includes('theme') || filename.includes('globals') || filename === 'tailwind.config.js') {
    return 'theme';
  } else if (filename.includes('component') || filename.includes('module')) {
    return 'component-style';
  } else {
    return 'general';
  }
}

analyzeThemes();

// 4. BUTTON & INTERACTIVE ELEMENT ANALYSIS
console.log('🔘 Analyzing buttons and interactive elements...\n');

function analyzeButtons() {
  const buttonAnalysis = {
    total: 0,
    withHandlers: 0,
    withoutHandlers: 0,
    brokenButtons: []
  };
  
  function analyzeFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const relativePath = path.relative(rootDir, filePath);
      
      // Find button elements
      const buttonMatches = content.matchAll(/<(button|Button)[^>]*>/g);
      const buttons = Array.from(buttonMatches);
      
      buttons.forEach((match, index) => {
        buttonAnalysis.total++;
        const buttonTag = match[0];
        
        // Check for event handlers
        const hasOnClick = buttonTag.includes('onClick');
        const hasOnSubmit = buttonTag.includes('onSubmit');
        const hasHandler = hasOnClick || hasOnSubmit || buttonTag.includes('type="submit"');
        
        if (hasHandler) {
          buttonAnalysis.withHandlers++;
        } else {
          buttonAnalysis.withoutHandlers++;
          buttonAnalysis.brokenButtons.push({
            file: relativePath,
            line: content.substring(0, match.index).split('\n').length,
            button: buttonTag.substring(0, 100) + '...',
            issue: 'No event handler found'
          });
        }
      });
      
      // Find link elements that might be broken
      const linkMatches = content.matchAll(/<Link[^>]*to=['"](.*?)['"][^>]*>/g);
      Array.from(linkMatches).forEach((match, index) => {
        const linkPath = match[1];
        if (!auditReport.routes.defined.includes(linkPath) && !linkPath.startsWith('http') && linkPath !== '#') {
          buttonAnalysis.brokenButtons.push({
            file: relativePath,
            line: content.substring(0, match.index).split('\n').length,
            button: match[0].substring(0, 100) + '...',
            issue: `Link to undefined route: ${linkPath}`
          });
        }
      });
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  // Scan all TSX files
  function scanDirectory(dir) {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory() && item !== 'node_modules') {
          scanDirectory(itemPath);
        } else if (item.endsWith('.tsx')) {
          analyzeFile(itemPath);
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanDirectory(path.join(rootDir, 'src'));
  
  auditReport.buttons = buttonAnalysis;
  
  console.log(`🔘 Button Analysis:`);
  console.log(`  • Total buttons found: ${buttonAnalysis.total}`);
  console.log(`  • With handlers: ${buttonAnalysis.withHandlers}`);
  console.log(`  • Without handlers: ${buttonAnalysis.withoutHandlers}`);
  console.log(`  • Broken buttons/links: ${buttonAnalysis.brokenButtons.length}`);
  
  if (buttonAnalysis.brokenButtons.length > 0) {
    console.log(`  ❌ Issues found:`);
    buttonAnalysis.brokenButtons.slice(0, 10).forEach(broken => {
      console.log(`    • ${broken.file}:${broken.line} - ${broken.issue}`);
    });
    if (buttonAnalysis.brokenButtons.length > 10) {
      console.log(`    ... and ${buttonAnalysis.brokenButtons.length - 10} more issues`);
    }
  }
  
  console.log();
}

analyzeButtons();

// 5. ARCHITECTURE ANALYSIS
console.log('🏗️ Analyzing architecture and consolidation opportunities...\n');

function analyzeArchitecture() {
  const architectureIssues = [];
  const improvements = [];
  const consolidationOpportunities = [];
  
  // Check for consistent directory structure
  const expectedDirs = ['components', 'pages', 'hooks', 'utils', 'types'];
  const srcDir = path.join(rootDir, 'src');
  
  if (fs.existsSync(srcDir)) {
    const srcItems = fs.readdirSync(srcDir);
    
    expectedDirs.forEach(expectedDir => {
      if (!srcItems.includes(expectedDir)) {
        architectureIssues.push(`Missing expected directory: src/${expectedDir}`);
      }
    });
    
    // Check for mixed patterns
    const hasFeatures = srcItems.includes('features');
    const hasComponents = srcItems.includes('components');
    
    if (hasFeatures && hasComponents) {
      consolidationOpportunities.push({
        type: 'directory-structure',
        description: 'Mixed feature-based and component-based organization',
        action: 'Consolidate to consistent pattern (recommend feature-based)'
      });
    }
  }
  
  // Check for consistent import patterns
  const inconsistentImports = findInconsistentImports();
  if (inconsistentImports.length > 0) {
    architectureIssues.push(`Inconsistent import patterns found in ${inconsistentImports.length} files`);
    improvements.push({
      type: 'import-consistency',
      description: 'Standardize import patterns across the codebase',
      files: inconsistentImports.slice(0, 5)
    });
  }
  
  // Check for centralized configuration
  const configFiles = [
    'tsconfig.json',
    'vite.config.ts',
    'tailwind.config.js',
    '.eslintrc.json'
  ];
  
  const missingConfigs = configFiles.filter(config => 
    !fs.existsSync(path.join(rootDir, config))
  );
  
  if (missingConfigs.length > 0) {
    architectureIssues.push(`Missing configuration files: ${missingConfigs.join(', ')}`);
  }
  
  auditReport.architecture = {
    issues: architectureIssues,
    improvements: improvements,
    consolidationOpportunities: consolidationOpportunities
  };
  
  console.log(`🏗️ Architecture Analysis:`);
  console.log(`  • Issues found: ${architectureIssues.length}`);
  console.log(`  • Improvement opportunities: ${improvements.length}`);
  console.log(`  • Consolidation opportunities: ${consolidationOpportunities.length}`);
  
  if (architectureIssues.length > 0) {
    console.log(`  ❌ Architecture issues:`);
    architectureIssues.forEach(issue => console.log(`    • ${issue}`));
  }
  
  if (consolidationOpportunities.length > 0) {
    console.log(`  🔄 Consolidation opportunities:`);
    consolidationOpportunities.forEach(opp => {
      console.log(`    • ${opp.description}`);
      console.log(`      Action: ${opp.action}`);
    });
  }
  
  console.log();
}

function findInconsistentImports() {
  const inconsistentFiles = [];
  
  function analyzeFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      
      let hasRelativeImports = false;
      let hasAbsoluteImports = false;
      
      lines.forEach(line => {
        if (line.trim().startsWith('import') && line.includes('from')) {
          if (line.includes("from '../") || line.includes("from './")) {
            hasRelativeImports = true;
          } else if (line.includes("from '@/")) {
            hasAbsoluteImports = true;
          }
        }
      });
      
      if (hasRelativeImports && hasAbsoluteImports) {
        inconsistentFiles.push(path.relative(rootDir, filePath));
      }
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  function scanDirectory(dir) {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory() && item !== 'node_modules') {
          scanDirectory(itemPath);
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          analyzeFile(itemPath);
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanDirectory(path.join(rootDir, 'src'));
  return inconsistentFiles;
}

analyzeArchitecture();

// 6. GENERATE ACTION PLAN
console.log('📋 Generating comprehensive action plan...\n');

function generateActionPlan() {
  const actions = [];
  
  // Route fixes
  if (auditReport.routes.broken.length > 0) {
    actions.push({
      priority: 'high',
      category: 'routes',
      title: `Fix ${auditReport.routes.broken.length} broken routes`,
      description: 'Create missing page components or remove unused routes',
      estimatedTime: `${auditReport.routes.broken.length * 15} minutes`
    });
  }
  
  // Component organization
  if (auditReport.components.duplicates.length > 0) {
    actions.push({
      priority: 'medium',
      category: 'components',
      title: `Resolve ${auditReport.components.duplicates.length} duplicate components`,
      description: 'Consolidate or rename duplicate components',
      estimatedTime: `${auditReport.components.duplicates.length * 20} minutes`
    });
  }
  
  // Theme centralization
  if (!auditReport.themes.centralized) {
    actions.push({
      priority: 'medium',
      category: 'themes',
      title: 'Centralize theme system',
      description: 'Create unified theme configuration and consolidate scattered styles',
      estimatedTime: '2 hours'
    });
  }
  
  // Button fixes
  if (auditReport.buttons.brokenButtons.length > 0) {
    actions.push({
      priority: 'high',
      category: 'functionality',
      title: `Fix ${auditReport.buttons.brokenButtons.length} broken buttons/links`,
      description: 'Add missing event handlers and fix broken navigation',
      estimatedTime: `${auditReport.buttons.brokenButtons.length * 10} minutes`
    });
  }
  
  // Architecture improvements
  auditReport.architecture.consolidationOpportunities.forEach(opp => {
    actions.push({
      priority: 'low',
      category: 'architecture',
      title: opp.description,
      description: opp.action,
      estimatedTime: '1-3 hours'
    });
  });
  
  auditReport.actions = actions;
  
  console.log(`📋 Action Plan (${actions.length} items):`);
  actions.forEach((action, i) => {
    const priorityIcon = action.priority === 'high' ? '🔴' : 
                        action.priority === 'medium' ? '🟡' : '🔵';
    console.log(`  ${priorityIcon} ${action.title}`);
    console.log(`     ${action.description}`);
    console.log(`     Estimated time: ${action.estimatedTime}`);
    console.log();
  });
}

generateActionPlan();

// 7. CALCULATE SCORES
function calculateScores() {
  // Organization score (0-100)
  let orgScore = 100;
  orgScore -= auditReport.components.duplicates.length * 5;
  orgScore -= auditReport.themes.scattered.length * 2;
  orgScore -= auditReport.architecture.issues.length * 10;
  if (!auditReport.themes.centralized) orgScore -= 15;
  
  // Functionality score (0-100)
  let funcScore = 100;
  funcScore -= auditReport.routes.broken.length * 10;
  funcScore -= auditReport.buttons.brokenButtons.length * 5;
  funcScore -= auditReport.buttons.withoutHandlers * 2;
  
  auditReport.summary.organizationScore = Math.max(0, orgScore);
  auditReport.summary.functionalityScore = Math.max(0, funcScore);
  
  const totalIssues = auditReport.routes.broken.length + 
                     auditReport.components.duplicates.length + 
                     auditReport.buttons.brokenButtons.length + 
                     auditReport.architecture.issues.length;
  
  const criticalIssues = auditReport.routes.broken.length + 
                        auditReport.buttons.brokenButtons.length;
  
  auditReport.summary.totalIssues = totalIssues;
  auditReport.summary.criticalIssues = criticalIssues;
}

calculateScores();

// 8. SAVE REPORT
fs.writeFileSync(
  path.join(rootDir, 'codebase-health-audit.json'),
  JSON.stringify(auditReport, null, 2)
);

// 9. DISPLAY SUMMARY
console.log('🔍 Codebase Health Audit Summary');
console.log('═'.repeat(60));
console.log(`📊 Organization Score: ${auditReport.summary.organizationScore}/100`);
console.log(`⚙️ Functionality Score: ${auditReport.summary.functionalityScore}/100`);
console.log(`🚨 Total Issues: ${auditReport.summary.totalIssues}`);
console.log(`🔴 Critical Issues: ${auditReport.summary.criticalIssues}`);
console.log('═'.repeat(60));

console.log('\n📈 Key Findings:');
console.log(`🗺️ Routes: ${auditReport.routes.defined.length} defined, ${auditReport.routes.broken.length} broken`);
console.log(`🧩 Components: ${auditReport.components.total} total, ${auditReport.components.duplicates.length} duplicates`);
console.log(`🎨 Themes: ${auditReport.themes.centralized ? 'Centralized' : 'Scattered'}`);
console.log(`🔘 Buttons: ${auditReport.buttons.total} total, ${auditReport.buttons.brokenButtons.length} broken`);

console.log('\n📄 Detailed audit report saved to: codebase-health-audit.json');
console.log('\n🔍 Codebase health audit complete!');

process.exit(0);