#!/usr/bin/env node

/**
 * REPZ Quick Audit Script - Focus on critical routing issues
 */

const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const SRC_DIR = path.join(PROJECT_ROOT, 'src');

// ANSI colors
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

class QuickAuditor {
  constructor() {
    this.issues = [];
    this.routes = [];
    this.pageFiles = [];
  }

  extractRoutes() {
    console.log(`${colors.blue}🔍 Extracting routes from App.tsx...${colors.reset}`);
    
    const appFile = path.join(SRC_DIR, 'App.tsx');
    const content = fs.readFileSync(appFile, 'utf8');
    
    // Extract route paths
    const routeRegex = /<Route\s+path="([^"]+)"/g;
    let match;
    
    while ((match = routeRegex.exec(content)) !== null) {
      this.routes.push(match[1]);
    }
    
    console.log(`${colors.green}✅ Found ${this.routes.length} routes${colors.reset}`);
    return this.routes;
  }

  findPageFiles() {
    console.log(`${colors.blue}🔍 Scanning pages directory...${colors.reset}`);
    
    const pagesDir = path.join(SRC_DIR, 'pages');
    const files = this.getAllFiles(pagesDir, ['.tsx', '.ts']);
    
    this.pageFiles = files.map(f => path.relative(SRC_DIR, f));
    console.log(`${colors.green}✅ Found ${this.pageFiles.length} page files${colors.reset}`);
    return this.pageFiles;
  }

  getAllFiles(dir, extensions) {
    const files = [];
    if (!fs.existsSync(dir)) return files;
    
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        files.push(...this.getAllFiles(fullPath, extensions));
      } else if (extensions.some(ext => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  checkRoutingIssues() {
    console.log(`${colors.blue}🔍 Checking for routing issues...${colors.reset}`);
    
    const appFile = path.join(SRC_DIR, 'App.tsx');
    const content = fs.readFileSync(appFile, 'utf8');

    // Check for hardcoded paths
    const hardcodedPaths = content.match(/window\.location\.href\s*=\s*['"`][^'"`]+['"`]/g) || [];
    if (hardcodedPaths.length > 0) {
      this.issues.push({
        type: 'ROUTING',
        severity: 'HIGH',
        message: `Found ${hardcodedPaths.length} hardcoded navigation paths`,
        details: hardcodedPaths,
        file: 'App.tsx'
      });
    }

    // Check for duplicate routes
    const routeCounts = {};
    this.routes.forEach(route => {
      routeCounts[route] = (routeCounts[route] || 0) + 1;
    });

    Object.entries(routeCounts).forEach(([route, count]) => {
      if (count > 1) {
        this.issues.push({
          type: 'ROUTING',
          severity: 'HIGH',
          message: `Duplicate route found: ${route}`,
          details: `Route "${route}" is defined ${count} times`,
          file: 'App.tsx'
        });
      }
    });

    // Check for commented routes
    const commentedRoutes = content.match(/\/\*[\s\S]*?Route[\s\S]*?\*\//g) || 
                           content.match(/\/\/.*Route.*$/gm) || [];
    
    if (commentedRoutes.length > 0) {
      this.issues.push({
        type: 'DEAD_CODE',
        severity: 'MEDIUM',
        message: `Found ${commentedRoutes.length} commented out routes`,
        details: commentedRoutes.slice(0, 3), // Show first 3
        file: 'App.tsx'
      });
    }

    console.log(`${colors.yellow}⚠️  Found ${this.issues.length} routing issues${colors.reset}`);
  }

  checkImportIssues() {
    console.log(`${colors.blue}🔍 Checking for import issues...${colors.reset}`);
    
    const appFile = path.join(SRC_DIR, 'App.tsx');
    const content = fs.readFileSync(appFile, 'utf8');

    // Check for missing lazy imports
    const lazyImports = [...content.matchAll(/lazy\(\(\)\s*=>\s*import\(["']([^"']+)["']\)/g)];
    
    for (const match of lazyImports) {
      const importPath = match[1];
      let resolvedPath;
      
      if (importPath.startsWith('./')) {
        resolvedPath = path.join(SRC_DIR, importPath.substring(2));
      } else if (importPath.startsWith('../')) {
        resolvedPath = path.resolve(SRC_DIR, importPath);
      } else {
        continue; // Skip node_modules imports
      }

      // Try common extensions
      const extensions = ['.tsx', '.ts', '.jsx', '.js'];
      let exists = false;
      
      for (const ext of extensions) {
        if (fs.existsSync(resolvedPath + ext)) {
          exists = true;
          break;
        }
      }
      
      if (!exists && fs.existsSync(resolvedPath + '/index.tsx')) {
        exists = true;
      }

      if (!exists) {
        this.issues.push({
          type: 'MISSING_FILE',
          severity: 'HIGH',
          message: `Lazy import points to missing file: ${importPath}`,
          details: `File not found: ${resolvedPath}`,
          file: 'App.tsx'
        });
      }
    }
  }

  checkDashboardIntegrity() {
    console.log(`${colors.blue}🔍 Checking dashboard integrity...${colors.reset}`);
    
    const dashboardFiles = [
      'pages/Dashboard.tsx',
      'pages/CoachDashboard.tsx',
      'features/dashboard/components/ClientDashboard.tsx',
      'features/analytics/components/AnalyticsDashboard.tsx'
    ];

    for (const dashboardFile of dashboardFiles) {
      const fullPath = path.join(SRC_DIR, dashboardFile);
      
      if (fs.existsSync(fullPath)) {
        const content = fs.readFileSync(fullPath, 'utf8');
        
        // Check for tier-based access control
        if (!content.includes('useTierAccess') && !content.includes('TierGate')) {
          this.issues.push({
            type: 'SECURITY',
            severity: 'MEDIUM',
            message: `Dashboard missing tier access control: ${dashboardFile}`,
            details: 'Dashboard should implement tier-based access control',
            file: dashboardFile
          });
        }

        // Check for error boundaries
        if (!content.includes('ErrorBoundary') && !content.includes('try') && !content.includes('catch')) {
          this.issues.push({
            type: 'RELIABILITY',
            severity: 'MEDIUM',
            message: `Dashboard missing error handling: ${dashboardFile}`,
            details: 'Dashboard should have error boundaries or try/catch blocks',
            file: dashboardFile
          });
        }
      } else {
        this.issues.push({
          type: 'MISSING_FILE',
          severity: 'HIGH',
          message: `Expected dashboard file missing: ${dashboardFile}`,
          details: 'Dashboard file referenced in routes but does not exist',
          file: dashboardFile
        });
      }
    }
  }

  generateReport() {
    console.log(`\n${colors.bold}${colors.cyan}📊 REPZ Quick Audit Report${colors.reset}\n`);
    
    const severityCounts = {
      HIGH: this.issues.filter(i => i.severity === 'HIGH').length,
      MEDIUM: this.issues.filter(i => i.severity === 'MEDIUM').length,
      LOW: this.issues.filter(i => i.severity === 'LOW').length
    };

    console.log(`${colors.bold}📈 Summary:${colors.reset}`);
    console.log(`Total Issues: ${colors.red}${this.issues.length}${colors.reset}`);
    console.log(`High Severity: ${colors.red}${severityCounts.HIGH}${colors.reset}`);
    console.log(`Medium Severity: ${colors.yellow}${severityCounts.MEDIUM}${colors.reset}`);
    console.log(`Low Severity: ${colors.green}${severityCounts.LOW}${colors.reset}`);

    console.log(`\n${colors.bold}🚨 Issues Found:${colors.reset}`);
    
    const groupedIssues = {};
    this.issues.forEach(issue => {
      if (!groupedIssues[issue.type]) {
        groupedIssues[issue.type] = [];
      }
      groupedIssues[issue.type].push(issue);
    });

    Object.entries(groupedIssues).forEach(([type, issues]) => {
      console.log(`\n${colors.bold}${type}:${colors.reset}`);
      
      issues.forEach(issue => {
        const severityColor = issue.severity === 'HIGH' ? colors.red : 
                             issue.severity === 'MEDIUM' ? colors.yellow : colors.green;
        
        console.log(`  ${severityColor}■${colors.reset} ${issue.message}`);
        console.log(`    ${colors.cyan}File: ${issue.file}${colors.reset}`);
        
        if (issue.details) {
          if (Array.isArray(issue.details)) {
            issue.details.slice(0, 2).forEach(detail => {
              console.log(`    ${colors.cyan}Detail: ${detail}${colors.reset}`);
            });
          } else {
            console.log(`    ${colors.cyan}Detail: ${issue.details}${colors.reset}`);
          }
        }
      });
    });

    // Risk assessment
    console.log(`\n${colors.bold}🎯 Risk Assessment:${colors.reset}`);
    if (severityCounts.HIGH === 0 && severityCounts.MEDIUM <= 2) {
      console.log(`  ${colors.green}✅ LOW RISK: Minimal issues found${colors.reset}`);
    } else if (severityCounts.HIGH <= 2 && severityCounts.MEDIUM <= 5) {
      console.log(`  ${colors.yellow}⚠️  MEDIUM RISK: Review and address issues${colors.reset}`);
    } else {
      console.log(`  ${colors.red}🚨 HIGH RISK: Immediate attention required${colors.reset}`);
    }

    return {
      totalIssues: this.issues.length,
      severityCounts,
      issues: this.issues,
      routes: this.routes,
      pageFiles: this.pageFiles
    };
  }

  runQuickAudit() {
    console.log(`${colors.bold}${colors.cyan}🔍 Starting REPZ Quick Audit...${colors.reset}\n`);
    
    this.extractRoutes();
    this.findPageFiles();
    this.checkRoutingIssues();
    this.checkImportIssues();
    this.checkDashboardIntegrity();
    
    return this.generateReport();
  }
}

// Run audit
const auditor = new QuickAuditor();
const report = auditor.runQuickAudit();

// Save report
const reportPath = path.join(PROJECT_ROOT, 'QUICK-AUDIT-REPORT.json');
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

console.log(`\n${colors.green}📄 Report saved to: QUICK-AUDIT-REPORT.json${colors.reset}`);

// Exit with appropriate code
process.exit(report.severityCounts.HIGH > 0 ? 1 : 0);