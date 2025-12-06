#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('👻 Ghost Routes & Dead Pages Eliminator\n');

const cleanup = {
  ghostRoutes: [],
  deadPages: [],
  redirects: [],
  cleaned: 0
};

// 1. ANALYZE APP.TSX FOR ROUTES
function analyzeAppRoutes() {
  const appPath = path.join(rootDir, 'src/App.tsx');
  if (!fs.existsSync(appPath)) {
    console.log('❌ App.tsx not found');
    return;
  }
  
  const content = fs.readFileSync(appPath, 'utf8');
  const lines = content.split('\n');
  
  console.log('🔍 Analyzing routes in App.tsx...\n');
  
  // Find route definitions and their components
  const routePattern = /<Route\s+path="([^"]+)"\s+element={[^}]*<(\w+)[^}]*}[^>]*\/>/g;
  const lazyPattern = /const\s+(\w+)\s+=\s+lazy\(\(\)\s+=>\s+import\("([^"]+)"\)\);/g;
  
  let match;
  const routes = [];
  const lazyComponents = new Map();
  
  // Extract lazy imports
  while ((match = lazyPattern.exec(content)) !== null) {
    lazyComponents.set(match[1], match[2]);
  }
  
  // Extract routes
  const simpleRoutePattern = /<Route\s+path="([^"]+)"/g;
  while ((match = simpleRoutePattern.exec(content)) !== null) {
    const routePath = match[1];
    
    if (routePath !== '*' && routePath !== '/' && !routePath.includes(':')) {
      routes.push(routePath);
    }
  }
  
  console.log(`Found ${routes.length} routes to analyze:`);
  routes.forEach(route => console.log(`  • ${route}`));
  console.log();
  
  return { routes, lazyComponents, content, lines };
}

// 2. FIND DEAD PAGES
function findDeadPages() {
  console.log('💀 Scanning for dead pages...\n');
  
  const pagesDir = path.join(rootDir, 'src/pages');
  if (!fs.existsSync(pagesDir)) return;
  
  const appContent = fs.readFileSync(path.join(rootDir, 'src/App.tsx'), 'utf8');
  const deadPages = [];
  
  function scanPages(dir, relativePath = '') {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory()) {
        scanPages(itemPath, path.join(relativePath, item));
      } else if (item.endsWith('.tsx')) {
        const componentName = item.replace('.tsx', '');
        const fullPath = path.join(relativePath, item);
        
        // Skip essential pages
        if (['NotFound', 'Index', 'App', 'main'].includes(componentName)) {
          return;
        }
        
        // Check if page is referenced in App.tsx
        const isImported = appContent.includes(componentName) || 
                          appContent.includes(fullPath);
        
        if (!isImported) {
          deadPages.push({
            file: path.join('src/pages', fullPath),
            component: componentName,
            size: stat.size
          });
        }
      }
    });
  }
  
  scanPages(pagesDir);
  cleanup.deadPages = deadPages;
  
  if (deadPages.length > 0) {
    console.log(`Found ${deadPages.length} potentially dead pages:`);
    deadPages.forEach(page => {
      console.log(`  🗑️  ${page.file} (${Math.round(page.size/1024)}KB)`);
    });
  } else {
    console.log('✅ No dead pages found!');
  }
  console.log();
}

// 3. FIND GHOST ROUTES (routes without components)
function findGhostRoutes() {
  console.log('👻 Scanning for ghost routes...\n');
  
  const { routes } = analyzeAppRoutes();
  const ghostRoutes = [];
  
  routes.forEach(route => {
    // Skip special routes
    if (route === '/' || route === '*' || route.includes(':')) return;
    
    // Convert route to expected page path
    const expectedPageName = route
      .replace(/^\//, '')
      .split('/')
      .map(part => part.charAt(0).toUpperCase() + part.slice(1))
      .join('');
    
    const possiblePaths = [
      path.join(rootDir, 'src/pages', `${expectedPageName}.tsx`),
      path.join(rootDir, 'src/pages', route.replace(/^\//, ''), 'index.tsx'),
      path.join(rootDir, 'src/pages', `${route.replace(/^\//, '')}.tsx`)
    ];
    
    const pageExists = possiblePaths.some(p => fs.existsSync(p));
    
    if (!pageExists) {
      ghostRoutes.push({
        route,
        expectedComponent: expectedPageName,
        possiblePaths
      });
    }
  });
  
  cleanup.ghostRoutes = ghostRoutes;
  
  if (ghostRoutes.length > 0) {
    console.log(`Found ${ghostRoutes.length} ghost routes:`);
    ghostRoutes.forEach(ghost => {
      console.log(`  👻 ${ghost.route} → Expected: ${ghost.expectedComponent}`);
    });
  } else {
    console.log('✅ No ghost routes found!');
  }
  console.log();
}

// 4. CLEAN UP GHOST ROUTES
function cleanupGhostRoutes() {
  if (cleanup.ghostRoutes.length === 0) return;
  
  console.log('🧹 Cleaning up ghost routes...\n');
  
  const appPath = path.join(rootDir, 'src/App.tsx');
  let content = fs.readFileSync(appPath, 'utf8');
  
  cleanup.ghostRoutes.forEach(ghost => {
    // Remove the route line
    const routeRegex = new RegExp(`\\s*<Route\\s+path="${ghost.route.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"[^>]*>([\\s\\S]*?</Route>|[^>]*/>)`, 'g');
    
    const beforeLength = content.length;
    content = content.replace(routeRegex, '');
    
    if (content.length < beforeLength) {
      console.log(`  ✅ Removed ghost route: ${ghost.route}`);
      cleanup.cleaned++;
    }
  });
  
  // Clean up extra whitespace
  content = content.replace(/\n\s*\n\s*\n/g, '\n\n');
  
  fs.writeFileSync(appPath, content);
}

// 5. MOVE DEAD PAGES TO GRAVEYARD
function movePagesToGraveyard() {
  if (cleanup.deadPages.length === 0) return;
  
  console.log('🪦 Moving dead pages to graveyard...\n');
  
  const timestamp = new Date().toISOString().split('T')[0];
  const graveyardDir = path.join(rootDir, '_graveyard', `dead-pages-${timestamp}`);
  
  if (!fs.existsSync(graveyardDir)) {
    fs.mkdirSync(graveyardDir, { recursive: true });
  }
  
  cleanup.deadPages.forEach(page => {
    const sourcePath = path.join(rootDir, page.file);
    const destPath = path.join(graveyardDir, page.file);
    const destDir = path.dirname(destPath);
    
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    try {
      fs.renameSync(sourcePath, destPath);
      console.log(`  🪦 Moved: ${page.file}`);
      cleanup.cleaned++;
    } catch (error) {
      console.log(`  ❌ Failed to move: ${page.file} - ${error.message}`);
    }
  });
  
  // Create manifest
  const manifest = {
    timestamp: new Date().toISOString(),
    type: 'dead-pages-cleanup',
    movedFiles: cleanup.deadPages.map(p => p.file),
    totalFiles: cleanup.deadPages.length
  };
  
  fs.writeFileSync(
    path.join(graveyardDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  
  // Create restore script
  const restoreScript = `#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');

const manifest = ${JSON.stringify(manifest, null, 2)};

console.log('🔄 Restoring dead pages...');

manifest.movedFiles.forEach(file => {
  const graveyardPath = path.join(__dirname, file);
  const originalPath = path.join(rootDir, file);
  
  try {
    const originalDir = path.dirname(originalPath);
    fs.mkdirSync(originalDir, { recursive: true });
    fs.renameSync(graveyardPath, originalPath);
    console.log(\`✅ Restored: \${file}\`);
  } catch (error) {
    console.error(\`❌ Failed to restore \${file}:\`, error.message);
  }
});

console.log('\\n✅ Restoration complete!');
`;
  
  fs.writeFileSync(path.join(graveyardDir, 'restore.mjs'), restoreScript);
}

// 6. GENERATE CLEANUP REPORT
function generateReport() {
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      ghostRoutesFound: cleanup.ghostRoutes.length,
      deadPagesFound: cleanup.deadPages.length,
      totalCleaned: cleanup.cleaned
    },
    ghostRoutes: cleanup.ghostRoutes,
    deadPages: cleanup.deadPages,
    actions: [
      cleanup.ghostRoutes.length > 0 ? 'Removed ghost routes from App.tsx' : 'No ghost routes to clean',
      cleanup.deadPages.length > 0 ? 'Moved dead pages to graveyard' : 'No dead pages to clean'
    ]
  };
  
  fs.writeFileSync(
    path.join(rootDir, 'ghost-cleanup-report.json'),
    JSON.stringify(report, null, 2)
  );
  
  return report;
}

// MAIN EXECUTION
async function main() {
  console.log('🎯 Starting ghost routes and dead pages cleanup...\n');
  
  // Step 1: Find issues
  findGhostRoutes();
  findDeadPages();
  
  // Step 2: Clean up if issues found
  if (cleanup.ghostRoutes.length > 0 || cleanup.deadPages.length > 0) {
    console.log('🧹 Performing cleanup...\n');
    
    cleanupGhostRoutes();
    movePagesToGraveyard();
    
    console.log('\n📊 Cleanup Summary:');
    console.log('═'.repeat(50));
    console.log(`👻 Ghost routes removed: ${cleanup.ghostRoutes.length}`);
    console.log(`💀 Dead pages moved: ${cleanup.deadPages.length}`);
    console.log(`🧹 Total items cleaned: ${cleanup.cleaned}`);
    console.log('═'.repeat(50));
  } else {
    console.log('✅ No cleanup needed - repository is already clean!');
  }
  
  // Step 3: Generate report
  const report = generateReport();
  console.log('\n📄 Cleanup report saved to: ghost-cleanup-report.json');
  
  console.log('\n🎉 Ghost routes and dead pages cleanup complete!');
}

main().catch(console.error);