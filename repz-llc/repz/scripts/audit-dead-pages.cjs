#!/usr/bin/env node

/**
 * REPZ Dead Page Detection Script
 * Identifies orphaned routes, unused components, and potential dead code
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const SRC_DIR = path.join(PROJECT_ROOT, 'src');
const PAGES_DIR = path.join(SRC_DIR, 'pages');
const COMPONENTS_DIR = path.join(SRC_DIR, 'components');

// ANSI colors for console output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

class DeadPageDetector {
  constructor() {
    this.routes = new Set();
    this.pageFiles = new Set();
    this.componentFiles = new Set();
    this.importReferences = new Map();
    this.routeDefinitions = new Map();
    this.deadRoutes = [];
    this.orphanedPages = [];
    this.unusedComponents = [];
    this.suspiciousPatterns = [];
  }

  // Recursively find all files with specific extensions
  findFiles(dir, extensions = ['.tsx', '.ts', '.jsx', '.js']) {
    const files = [];
    
    if (!fs.existsSync(dir)) return files;
    
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        files.push(...this.findFiles(fullPath, extensions));
      } else if (entry.isFile() && extensions.some(ext => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  // Extract route definitions from App.tsx and other routing files
  extractRoutes() {
    console.log(`${colors.blue}🔍 Extracting route definitions...${colors.reset}`);
    
    const routingFiles = [
      path.join(SRC_DIR, 'App.tsx'),
      ...this.findFiles(SRC_DIR).filter(file => 
        file.includes('Route') || file.includes('router') || file.includes('routing')
      )
    ];

    for (const file of routingFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        // Extract Route path definitions
        const routeRegex = /<Route\s+path="([^"]+)"/g;
        let match;
        
        while ((match = routeRegex.exec(content)) !== null) {
          const routePath = match[1];
          this.routes.add(routePath);
          this.routeDefinitions.set(routePath, file);
        }

        // Extract lazy imports and components
        const lazyImportRegex = /lazy\(\(\)\s*=>\s*import\(["']([^"']+)["']\)/g;
        while ((match = lazyImportRegex.exec(content)) !== null) {
          const importPath = match[1];
          this.importReferences.set(importPath, file);
        }

        // Extract static imports
        const staticImportRegex = /import\s+(?:\w+|\{[^}]+\}|\*\s+as\s+\w+)\s+from\s+["']([^"']+)["']/g;
        while ((match = staticImportRegex.exec(content)) !== null) {
          const importPath = match[1];
          if (importPath.startsWith('./') || importPath.startsWith('../')) {
            this.importReferences.set(importPath, file);
          }
        }
      } catch (error) {
        console.warn(`${colors.yellow}⚠️  Could not read ${file}: ${error.message}${colors.reset}`);
      }
    }

    console.log(`${colors.green}✅ Found ${this.routes.size} route definitions${colors.reset}`);
  }

  // Find all page files
  findPageFiles() {
    console.log(`${colors.blue}🔍 Scanning page files...${colors.reset}`);
    
    const pageFiles = this.findFiles(PAGES_DIR);
    
    for (const file of pageFiles) {
      const relativePath = path.relative(SRC_DIR, file);
      this.pageFiles.add(relativePath);
    }

    console.log(`${colors.green}✅ Found ${this.pageFiles.size} page files${colors.reset}`);
  }

  // Find all component files
  findComponentFiles() {
    console.log(`${colors.blue}🔍 Scanning component files...${colors.reset}`);
    
    const componentFiles = this.findFiles(COMPONENTS_DIR);
    
    for (const file of componentFiles) {
      const relativePath = path.relative(SRC_DIR, file);
      this.componentFiles.add(relativePath);
    }

    console.log(`${colors.green}✅ Found ${this.componentFiles.size} component files${colors.reset}`);
  }

  // Check if a file is referenced anywhere in the codebase
  isFileReferenced(filePath) {
    try {
      // Remove file extension and normalize path for searching
      const baseName = path.basename(filePath, path.extname(filePath));
      const searchPatterns = [
        baseName,
        `./${filePath}`,
        `../${filePath}`,
        filePath.replace(/\.(tsx?|jsx?)$/, ''),
        filePath
      ];

      for (const pattern of searchPatterns) {
        try {
          const result = execSync(
            `grep -r "${pattern}" "${SRC_DIR}" --include="*.tsx" --include="*.ts" --include="*.jsx" --include="*.js" -l`,
            { encoding: 'utf8', stdio: 'pipe' }
          );
          
          if (result.trim()) {
            return true;
          }
        } catch (grepError) {
          // grep returns non-zero exit code when no matches found
          continue;
        }
      }
      
      return false;
    } catch (error) {
      console.warn(`${colors.yellow}⚠️  Could not check references for ${filePath}: ${error.message}${colors.reset}`);
      return true; // Assume referenced if we can't check
    }
  }

  // Detect dead routes (routes that don't lead to existing pages)
  detectDeadRoutes() {
    console.log(`${colors.blue}🔍 Detecting dead routes...${colors.reset}`);

    for (const route of this.routes) {
      if (route === '*' || route === '404' || route.includes(':')) {
        continue; // Skip catch-all and dynamic routes
      }

      // Check if route has corresponding page or component
      let hasCorrespondingFile = false;
      
      // Check if any page file could handle this route
      for (const pageFile of this.pageFiles) {
        const fileName = path.basename(pageFile, path.extname(pageFile));
        const routeName = route.replace(/^\//, '').replace(/\//g, '');
        
        if (fileName.toLowerCase().includes(routeName.toLowerCase()) ||
            routeName.toLowerCase().includes(fileName.toLowerCase())) {
          hasCorrespondingFile = true;
          break;
        }
      }

      if (!hasCorrespondingFile) {
        this.deadRoutes.push({
          route,
          definedIn: this.routeDefinitions.get(route),
          reason: 'No corresponding page file found'
        });
      }
    }

    console.log(`${colors.red}❌ Found ${this.deadRoutes.length} potentially dead routes${colors.reset}`);
  }

  // Detect orphaned pages (pages not referenced by any route)
  detectOrphanedPages() {
    console.log(`${colors.blue}🔍 Detecting orphaned pages...${colors.reset}`);

    for (const pageFile of this.pageFiles) {
      if (!this.isFileReferenced(pageFile)) {
        this.orphanedPages.push({
          file: pageFile,
          reason: 'Page file not referenced in any route or import'
        });
      }
    }

    console.log(`${colors.red}❌ Found ${this.orphanedPages.length} orphaned pages${colors.reset}`);
  }

  // Detect unused components
  detectUnusedComponents() {
    console.log(`${colors.blue}🔍 Detecting unused components...${colors.reset}`);

    // Focus on high-risk directories for unused components
    const riskDirectories = [
      'components/ui',
      'components/shared',
      'features',
      'pages'
    ];

    for (const componentFile of this.componentFiles) {
      // Skip certain directories that are likely to be used
      if (componentFile.includes('ui/') && componentFile.includes('index.ts')) {
        continue; // Skip index files which re-export components
      }

      const shouldCheck = riskDirectories.some(dir => componentFile.includes(dir));
      
      if (shouldCheck && !this.isFileReferenced(componentFile)) {
        this.unusedComponents.push({
          file: componentFile,
          reason: 'Component not referenced in any file'
        });
      }
    }

    console.log(`${colors.red}❌ Found ${this.unusedComponents.length} potentially unused components${colors.reset}`);
  }

  // Detect suspicious patterns that might indicate dead code
  detectSuspiciousPatterns() {
    console.log(`${colors.blue}🔍 Detecting suspicious patterns...${colors.reset}`);

    const allFiles = [
      ...this.findFiles(SRC_DIR)
    ];

    for (const file of allFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        const relativePath = path.relative(PROJECT_ROOT, file);

        // Check for commented out routes
        const commentedRoutes = content.match(/\/\*[\s\S]*?<Route[\s\S]*?\*\//g) || 
                               content.match(/\/\/.*<Route.*$/gm) || [];
        
        if (commentedRoutes.length > 0) {
          this.suspiciousPatterns.push({
            file: relativePath,
            pattern: 'Commented out routes',
            details: `${commentedRoutes.length} commented route(s) found`,
            lines: commentedRoutes
          });
        }

        // Check for TODO/FIXME comments related to routing
        const todoRoutes = content.match(/\/\/\s*(TODO|FIXME|HACK).*route/gi) || [];
        
        if (todoRoutes.length > 0) {
          this.suspiciousPatterns.push({
            file: relativePath,
            pattern: 'TODO/FIXME in routing code',
            details: `${todoRoutes.length} routing-related TODO(s) found`,
            lines: todoRoutes
          });
        }

        // Check for duplicate route paths
        const routePaths = [...content.matchAll(/<Route\s+path="([^"]+)"/g)];
        const pathCounts = {};
        
        for (const match of routePaths) {
          const path = match[1];
          pathCounts[path] = (pathCounts[path] || 0) + 1;
        }

        for (const [routePath, count] of Object.entries(pathCounts)) {
          if (count > 1) {
            this.suspiciousPatterns.push({
              file: relativePath,
              pattern: 'Duplicate route paths',
              details: `Route "${routePath}" defined ${count} times in this file`
            });
          }
        }

      } catch (error) {
        // Skip files that can't be read
        continue;
      }
    }

    console.log(`${colors.yellow}⚠️  Found ${this.suspiciousPatterns.length} suspicious patterns${colors.reset}`);
  }

  // Generate comprehensive report
  generateReport() {
    console.log(`\n${colors.bold}${colors.cyan}📊 REPZ Dead Page Detection Report${colors.reset}\n`);
    console.log(`Generated: ${new Date().toLocaleString()}\n`);

    // Summary
    console.log(`${colors.bold}📈 Summary:${colors.reset}`);
    console.log(`Routes found: ${colors.green}${this.routes.size}${colors.reset}`);
    console.log(`Page files: ${colors.green}${this.pageFiles.size}${colors.reset}`);
    console.log(`Component files: ${colors.green}${this.componentFiles.size}${colors.reset}`);
    
    console.log(`\n${colors.bold}🚨 Issues Found:${colors.reset}`);
    console.log(`Dead routes: ${colors.red}${this.deadRoutes.length}${colors.reset}`);
    console.log(`Orphaned pages: ${colors.red}${this.orphanedPages.length}${colors.reset}`);
    console.log(`Unused components: ${colors.red}${this.unusedComponents.length}${colors.reset}`);
    console.log(`Suspicious patterns: ${colors.yellow}${this.suspiciousPatterns.length}${colors.reset}`);

    // Dead Routes
    if (this.deadRoutes.length > 0) {
      console.log(`\n${colors.bold}${colors.red}💀 Dead Routes:${colors.reset}`);
      for (const deadRoute of this.deadRoutes) {
        console.log(`  ${colors.red}✗${colors.reset} ${deadRoute.route}`);
        console.log(`    ${colors.gray}Defined in: ${deadRoute.definedIn}${colors.reset}`);
        console.log(`    ${colors.gray}Reason: ${deadRoute.reason}${colors.reset}`);
      }
    }

    // Orphaned Pages
    if (this.orphanedPages.length > 0) {
      console.log(`\n${colors.bold}${colors.red}🏚️  Orphaned Pages:${colors.reset}`);
      for (const orphan of this.orphanedPages) {
        console.log(`  ${colors.red}✗${colors.reset} ${orphan.file}`);
        console.log(`    ${colors.gray}Reason: ${orphan.reason}${colors.reset}`);
      }
    }

    // Unused Components (limit to first 10 to avoid spam)
    if (this.unusedComponents.length > 0) {
      console.log(`\n${colors.bold}${colors.red}🗑️  Unused Components (showing first 10):${colors.reset}`);
      for (const component of this.unusedComponents.slice(0, 10)) {
        console.log(`  ${colors.red}✗${colors.reset} ${component.file}`);
        console.log(`    ${colors.gray}Reason: ${component.reason}${colors.reset}`);
      }
      
      if (this.unusedComponents.length > 10) {
        console.log(`  ${colors.gray}... and ${this.unusedComponents.length - 10} more${colors.reset}`);
      }
    }

    // Suspicious Patterns
    if (this.suspiciousPatterns.length > 0) {
      console.log(`\n${colors.bold}${colors.yellow}⚠️  Suspicious Patterns:${colors.reset}`);
      for (const pattern of this.suspiciousPatterns) {
        console.log(`  ${colors.yellow}⚠${colors.reset} ${pattern.file}`);
        console.log(`    ${colors.gray}Pattern: ${pattern.pattern}${colors.reset}`);
        console.log(`    ${colors.gray}Details: ${pattern.details}${colors.reset}`);
      }
    }

    // Recommendations
    console.log(`\n${colors.bold}${colors.cyan}💡 Recommendations:${colors.reset}`);
    
    if (this.deadRoutes.length > 0) {
      console.log(`  ${colors.cyan}•${colors.reset} Remove or fix dead routes in App.tsx`);
    }
    
    if (this.orphanedPages.length > 0) {
      console.log(`  ${colors.cyan}•${colors.reset} Add routes for orphaned pages or remove unused pages`);
    }
    
    if (this.unusedComponents.length > 0) {
      console.log(`  ${colors.cyan}•${colors.reset} Remove unused components to reduce bundle size`);
    }
    
    if (this.suspiciousPatterns.length > 0) {
      console.log(`  ${colors.cyan}•${colors.reset} Review and clean up commented/duplicate routing code`);
    }

    // Risk Assessment
    const totalIssues = this.deadRoutes.length + this.orphanedPages.length + 
                       this.unusedComponents.length + this.suspiciousPatterns.length;
    
    console.log(`\n${colors.bold}🎯 Risk Assessment:${colors.reset}`);
    if (totalIssues === 0) {
      console.log(`  ${colors.green}✅ LOW RISK: No dead pages or unused code detected${colors.reset}`);
    } else if (totalIssues < 5) {
      console.log(`  ${colors.yellow}⚠️  MEDIUM RISK: ${totalIssues} issues found - review and clean up${colors.reset}`);
    } else {
      console.log(`  ${colors.red}🚨 HIGH RISK: ${totalIssues} issues found - immediate cleanup recommended${colors.reset}`);
    }

    return {
      routes: Array.from(this.routes),
      deadRoutes: this.deadRoutes,
      orphanedPages: this.orphanedPages,
      unusedComponents: this.unusedComponents,
      suspiciousPatterns: this.suspiciousPatterns,
      totalIssues
    };
  }

  // Run full audit
  async runAudit() {
    console.log(`${colors.bold}${colors.cyan}🔍 Starting REPZ Dead Page Detection Audit...${colors.reset}\n`);
    
    this.extractRoutes();
    this.findPageFiles();
    this.findComponentFiles();
    this.detectDeadRoutes();
    this.detectOrphanedPages();
    this.detectUnusedComponents();
    this.detectSuspiciousPatterns();
    
    return this.generateReport();
  }
}

// Main execution
if (require.main === module) {
  const detector = new DeadPageDetector();
  
  detector.runAudit()
    .then(report => {
      // Write detailed report to file
      const reportPath = path.join(PROJECT_ROOT, 'DEAD-PAGE-AUDIT-REPORT.json');
      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
      
      console.log(`\n${colors.green}📄 Detailed report saved to: ${reportPath}${colors.reset}`);
      
      // Exit with error code if high risk
      if (report.totalIssues >= 5) {
        process.exit(1);
      }
    })
    .catch(error => {
      console.error(`${colors.red}❌ Audit failed: ${error.message}${colors.reset}`);
      process.exit(1);
    });
}

module.exports = DeadPageDetector;