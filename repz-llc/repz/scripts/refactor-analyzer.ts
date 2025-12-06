#!/usr/bin/env node
/**
 * Enterprise Refactoring Analyzer
 * Provides comprehensive analysis and metrics for the refactoring process
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

interface ComponentAnalysis {
  totalComponents: number;
  duplicateComponents: ComponentDuplicate[];
  unusedComponents: string[];
  largeComponents: ComponentSize[];
  complexComponents: ComponentComplexity[];
}

interface ComponentDuplicate {
  name: string;
  locations: string[];
  similarity: number;
}

interface ComponentSize {
  path: string;
  lines: number;
  size: number;
}

interface ComponentComplexity {
  path: string;
  cyclomaticComplexity: number;
  dependencies: number;
}

interface ThemeAnalysis {
  totalStyles: number;
  hardcodedColors: string[];
  unusedClasses: string[];
  duplicateStyles: string[];
  modernizationOpportunities: string[];
}

interface ArchitectureAnalysis {
  routeComplexity: number;
  stateManagementIssues: string[];
  performanceIssues: string[];
  securityConcerns: string[];
}

class RefactorAnalyzer {
  private srcDir: string;
  
  constructor() {
    this.srcDir = path.join(process.cwd(), 'src');
  }
  
  // Main analysis entry point
  async analyzeProject(): Promise<void> {
    console.log('🔍 Starting comprehensive refactoring analysis...');
    
    const results = {
      timestamp: new Date().toISOString(),
      components: await this.analyzeComponents(),
      theme: await this.analyzeTheme(),
      architecture: await this.analyzeArchitecture(),
      metrics: await this.collectBaselineMetrics()
    };
    
    // Save analysis report
    const reportPath = path.join(process.cwd(), 'REFACTOR_ANALYSIS.md');
    await this.generateReport(results, reportPath);
    
    // Save raw data for scripts
    const dataPath = path.join(process.cwd(), '.rollbacks', 'analysis-data.json');
    if (!fs.existsSync(path.dirname(dataPath))) {
      fs.mkdirSync(path.dirname(dataPath), { recursive: true });
    }
    fs.writeFileSync(dataPath, JSON.stringify(results, null, 2));
    
    console.log(`📊 Analysis complete. Report saved to: ${reportPath}`);
    console.log(`💾 Raw data saved to: ${dataPath}`);
  }
  
  private async analyzeComponents(): Promise<ComponentAnalysis> {
    console.log('  🧩 Analyzing components...');
    
    const componentFiles = this.findComponentFiles();
    const totalComponents = componentFiles.length;
    
    const duplicateComponents = await this.findDuplicateComponents(componentFiles);
    const unusedComponents = await this.findUnusedComponents(componentFiles);
    const largeComponents = await this.findLargeComponents(componentFiles);
    const complexComponents = await this.findComplexComponents(componentFiles);
    
    return {
      totalComponents,
      duplicateComponents,
      unusedComponents,
      largeComponents,
      complexComponents
    };
  }
  
  private async analyzeTheme(): Promise<ThemeAnalysis> {
    console.log('  🎨 Analyzing theme system...');
    
    const styleFiles = this.findStyleFiles();
    const totalStyles = styleFiles.length;
    
    const hardcodedColors = await this.findHardcodedColors();
    const unusedClasses = await this.findUnusedClasses();
    const duplicateStyles = await this.findDuplicateStyles();
    const modernizationOpportunities = await this.findModernizationOpportunities();
    
    return {
      totalStyles,
      hardcodedColors,
      unusedClasses,
      duplicateStyles,
      modernizationOpportunities
    };
  }
  
  private async analyzeArchitecture(): Promise<ArchitectureAnalysis> {
    console.log('  🏗️ Analyzing architecture...');
    
    const routeComplexity = await this.analyzeRouteComplexity();
    const stateManagementIssues = await this.findStateManagementIssues();
    const performanceIssues = await this.findPerformanceIssues();
    const securityConcerns = await this.findSecurityConcerns();
    
    return {
      routeComplexity,
      stateManagementIssues,
      performanceIssues,
      securityConcerns
    };
  }
  
  private findComponentFiles(): string[] {
    const components: string[] = [];
    
    const walkDir = (dir: string) => {
      if (!fs.existsSync(dir)) return;
      
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
          walkDir(fullPath);
        } else if (item.endsWith('.tsx') && !item.endsWith('.test.tsx')) {
          components.push(fullPath);
        }
      }
    };
    
    walkDir(path.join(this.srcDir, 'components'));
    return components;
  }
  
  private findStyleFiles(): string[] {
    const styles: string[] = [];
    
    const walkDir = (dir: string) => {
      if (!fs.existsSync(dir)) return;
      
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
          walkDir(fullPath);
        } else if (item.endsWith('.css') || item.endsWith('.scss') || item.endsWith('.tsx')) {
          styles.push(fullPath);
        }
      }
    };
    
    walkDir(this.srcDir);
    return styles;
  }
  
  private async findDuplicateComponents(componentFiles: string[]): Promise<ComponentDuplicate[]> {
    const duplicates: ComponentDuplicate[] = [];
    const componentMap = new Map<string, string[]>();
    
    // Group components by name
    for (const file of componentFiles) {
      const name = path.basename(file, '.tsx');
      if (!componentMap.has(name)) {
        componentMap.set(name, []);
      }
      componentMap.get(name)!.push(file);
    }
    
    // Find actual duplicates
    for (const [name, locations] of componentMap) {
      if (locations.length > 1) {
        // Calculate similarity (simplified)
        const similarity = await this.calculateSimilarity(locations);
        if (similarity > 0.7) {
          duplicates.push({ name, locations, similarity });
        }
      }
    }
    
    return duplicates;
  }
  
  private async calculateSimilarity(files: string[]): Promise<number> {
    if (files.length < 2) return 0;
    
    try {
      const contents = files.map(file => fs.readFileSync(file, 'utf8'));
      const baseContent = contents[0];
      let totalSimilarity = 0;
      
      for (let i = 1; i < contents.length; i++) {
        const similarity = this.stringSimilarity(baseContent, contents[i]);
        totalSimilarity += similarity;
      }
      
      return totalSimilarity / (contents.length - 1);
    } catch (error) {
      return 0;
    }
  }
  
  private stringSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const editDistance = this.levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }
  
  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }
  
  private async findUnusedComponents(componentFiles: string[]): Promise<string[]> {
    const unused: string[] = [];
    const allFiles = this.getAllTsxFiles();
    
    for (const componentFile of componentFiles) {
      const componentName = path.basename(componentFile, '.tsx');
      let isUsed = false;
      
      for (const file of allFiles) {
        if (file === componentFile) continue;
        
        try {
          const content = fs.readFileSync(file, 'utf8');
          if (content.includes(componentName)) {
            isUsed = true;
            break;
          }
        } catch (error) {
          // Skip files that can't be read
        }
      }
      
      if (!isUsed) {
        unused.push(componentFile);
      }
    }
    
    return unused;
  }
  
  private getAllTsxFiles(): string[] {
    const files: string[] = [];
    
    const walkDir = (dir: string) => {
      if (!fs.existsSync(dir)) return;
      
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
          walkDir(fullPath);
        } else if (item.endsWith('.tsx') || item.endsWith('.ts')) {
          files.push(fullPath);
        }
      }
    };
    
    walkDir(this.srcDir);
    return files;
  }
  
  private async findLargeComponents(componentFiles: string[]): Promise<ComponentSize[]> {
    const large: ComponentSize[] = [];
    
    for (const file of componentFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        const lines = content.split('\n').length;
        const size = fs.statSync(file).size;
        
        if (lines > 200 || size > 10000) {
          large.push({ path: file, lines, size });
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return large.sort((a, b) => b.lines - a.lines);
  }
  
  private async findComplexComponents(componentFiles: string[]): Promise<ComponentComplexity[]> {
    const complex: ComponentComplexity[] = [];
    
    for (const file of componentFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        // Simple complexity metrics
        const cyclomaticComplexity = this.calculateCyclomaticComplexity(content);
        const dependencies = this.countDependencies(content);
        
        if (cyclomaticComplexity > 10 || dependencies > 15) {
          complex.push({ path: file, cyclomaticComplexity, dependencies });
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return complex.sort((a, b) => b.cyclomaticComplexity - a.cyclomaticComplexity);
  }
  
  private calculateCyclomaticComplexity(content: string): number {
    // Simple approximation: count decision points
    const patterns = [
      /\bif\s*\(/g,
      /\belse\s+if\b/g,
      /\bwhile\s*\(/g,
      /\bfor\s*\(/g,
      /\bswitch\s*\(/g,
      /\bcase\s+/g,
      /\bcatch\s*\(/g,
      /\?/g,
      /&&/g,
      /\|\|/g
    ];
    
    let complexity = 1; // Base complexity
    
    for (const pattern of patterns) {
      const matches = content.match(pattern);
      if (matches) {
        complexity += matches.length;
      }
    }
    
    return complexity;
  }
  
  private countDependencies(content: string): number {
    const importMatches = content.match(/^import\s+.*?from\s+['"].*?['"];?$/gm);
    return importMatches ? importMatches.length : 0;
  }
  
  private async findHardcodedColors(): Promise<string[]> {
    const colors: string[] = [];
    const files = this.getAllTsxFiles();
    
    const colorPatterns = [
      /#[0-9A-Fa-f]{3,6}/g, // Hex colors
      /rgb\([^)]+\)/g,      // RGB colors
      /rgba\([^)]+\)/g,     // RGBA colors
      /hsl\([^)]+\)/g,      // HSL colors
      /hsla\([^)]+\)/g      // HSLA colors
    ];
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        for (const pattern of colorPatterns) {
          const matches = content.match(pattern);
          if (matches) {
            colors.push(...matches.map(match => `${file}: ${match}`));
          }
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return Array.from(new Set(colors));
  }
  
  private async findUnusedClasses(): Promise<string[]> {
    // This would require more sophisticated analysis
    // For now, return empty array
    return [];
  }
  
  private async findDuplicateStyles(): Promise<string[]> {
    // This would require CSS parsing
    // For now, return empty array
    return [];
  }
  
  private async findModernizationOpportunities(): Promise<string[]> {
    const opportunities: string[] = [];
    const files = this.getAllTsxFiles();
    
    const patterns = [
      { pattern: /className="[^"]*\s+[^"]*"/g, suggestion: 'Use clsx or cn utility for conditional classes' },
      { pattern: /style={{[^}]+}}/g, suggestion: 'Consider using CSS-in-JS or Tailwind utilities' },
      { pattern: /\bvar\(/g, suggestion: 'Convert to CSS custom properties with semantic tokens' }
    ];
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        for (const { pattern, suggestion } of patterns) {
          const matches = content.match(pattern);
          if (matches) {
            opportunities.push(`${file}: ${suggestion} (${matches.length} occurrences)`);
          }
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return opportunities;
  }
  
  private async analyzeRouteComplexity(): Promise<number> {
    try {
      const appFile = path.join(this.srcDir, 'App.tsx');
      const content = fs.readFileSync(appFile, 'utf8');
      
      // Count route definitions
      const routeMatches = content.match(/<Route/g);
      return routeMatches ? routeMatches.length : 0;
    } catch (error) {
      return 0;
    }
  }
  
  private async findStateManagementIssues(): Promise<string[]> {
    const issues: string[] = [];
    const files = this.getAllTsxFiles();
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        // Look for potential issues
        if (content.includes('useState') && content.split('useState').length > 5) {
          issues.push(`${file}: Too many useState hooks (consider useReducer)`);
        }
        
        if (content.includes('useContext') && content.includes('useState')) {
          issues.push(`${file}: Mixed context and local state (consider consolidation)`);
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return issues;
  }
  
  private async findPerformanceIssues(): Promise<string[]> {
    const issues: string[] = [];
    const files = this.getAllTsxFiles();
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        // Look for performance anti-patterns
        if (content.includes('useEffect') && !content.includes('useMemo')) {
          const effectCount = (content.match(/useEffect/g) || []).length;
          if (effectCount > 3) {
            issues.push(`${file}: Multiple useEffect without memoization`);
          }
        }
        
        if (content.includes('map(') && content.includes('onClick')) {
          issues.push(`${file}: Inline event handlers in map (use useCallback)`);
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return issues;
  }
  
  private async findSecurityConcerns(): Promise<string[]> {
    const concerns: string[] = [];
    const files = this.getAllTsxFiles();
    
    for (const file of files) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        
        // Look for security anti-patterns
        if (content.includes('dangerouslySetInnerHTML')) {
          concerns.push(`${file}: Uses dangerouslySetInnerHTML`);
        }
        
        if (content.includes('eval(') || content.includes('Function(')) {
          concerns.push(`${file}: Uses eval or Function constructor`);
        }
        
        if (content.includes('localStorage') && !content.includes('JSON.parse')) {
          concerns.push(`${file}: Unsafe localStorage usage`);
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    return concerns;
  }
  
  private async collectBaselineMetrics(): Promise<any> {
    const metrics = {
      timestamp: new Date().toISOString(),
      codebase: {
        totalFiles: 0,
        totalLines: 0,
        components: 0,
        hooks: 0,
        contexts: 0
      },
      build: {
        success: false,
        time: 0,
        size: 0,
        warnings: 0,
        errors: 0
      },
      tests: {
        total: 0,
        passing: 0,
        failing: 0,
        coverage: 0
      }
    };
    
    // Collect codebase metrics
    const allFiles = this.getAllTsxFiles();
    metrics.codebase.totalFiles = allFiles.length;
    
    for (const file of allFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        metrics.codebase.totalLines += content.split('\n').length;
        
        if (file.includes('/components/')) metrics.codebase.components++;
        if (file.includes('/hooks/')) metrics.codebase.hooks++;
        if (file.includes('/contexts/')) metrics.codebase.contexts++;
      } catch (error) {
        // Skip files that can't be read
      }
    }
    
    // Try to collect build metrics
    try {
      const startTime = Date.now();
      execSync('npm run build:dev', { stdio: 'pipe' });
      metrics.build.success = true;
      metrics.build.time = Date.now() - startTime;
      
      const distDir = path.join(process.cwd(), 'dist');
      if (fs.existsSync(distDir)) {
        metrics.build.size = this.calculateDirSize(distDir);
      }
    } catch (error) {
      metrics.build.success = false;
    }
    
    // Try to collect test metrics
    try {
      const testResult = execSync('npm run test:run', { stdio: 'pipe' }).toString();
      // Parse test results (simplified)
      const passMatch = testResult.match(/(\d+) passing/);
      const failMatch = testResult.match(/(\d+) failing/);
      
      if (passMatch) metrics.tests.passing = parseInt(passMatch[1]);
      if (failMatch) metrics.tests.failing = parseInt(failMatch[1]);
      metrics.tests.total = metrics.tests.passing + metrics.tests.failing;
    } catch (error) {
      // Tests might not be set up or failing
    }
    
    return metrics;
  }
  
  private calculateDirSize(dir: string): number {
    if (!fs.existsSync(dir)) return 0;
    
    let size = 0;
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        size += this.calculateDirSize(fullPath);
      } else {
        size += stat.size;
      }
    }
    
    return size;
  }
  
  private async generateReport(results: any, reportPath: string): Promise<void> {
    const report = `# Enterprise Refactoring Analysis Report

Generated: ${results.timestamp}

## Executive Summary

### Component Analysis
- **Total Components**: ${results.components.totalComponents}
- **Duplicate Components**: ${results.components.duplicateComponents.length}
- **Unused Components**: ${results.components.unusedComponents.length}
- **Large Components**: ${results.components.largeComponents.length}
- **Complex Components**: ${results.components.complexComponents.length}

### Theme Analysis
- **Hardcoded Colors**: ${results.theme.hardcodedColors.length}
- **Modernization Opportunities**: ${results.theme.modernizationOpportunities.length}

### Architecture Analysis
- **Route Complexity**: ${results.architecture.routeComplexity} routes
- **State Management Issues**: ${results.architecture.stateManagementIssues.length}
- **Performance Issues**: ${results.architecture.performanceIssues.length}
- **Security Concerns**: ${results.architecture.securityConcerns.length}

### Baseline Metrics
- **Total Files**: ${results.metrics.codebase.totalFiles}
- **Total Lines**: ${results.metrics.codebase.totalLines}
- **Build Success**: ${results.metrics.build.success ? '✅' : '❌'}
- **Build Size**: ${Math.round(results.metrics.build.size / 1024)}KB

## Detailed Findings

### Duplicate Components
${results.components.duplicateComponents.map((dup: ComponentDuplicate) => `
**${dup.name}** (${Math.round(dup.similarity * 100)}% similarity)
${dup.locations.map((loc: string) => `- ${loc}`).join('\n')}`).join('\n')}

### Unused Components
${results.components.unusedComponents.map((comp: string) => `- ${comp}`).join('\n')}

### Large Components (>200 lines)
${results.components.largeComponents.map((comp: ComponentSize) => `- ${comp.path} (${comp.lines} lines)`).join('\n')}

### Complex Components (>10 complexity)
${results.components.complexComponents.map((comp: ComponentComplexity) => `- ${comp.path} (complexity: ${comp.cyclomaticComplexity}, deps: ${comp.dependencies})`).join('\n')}

### Hardcoded Colors
${results.theme.hardcodedColors.slice(0, 20).map((color: string) => `- ${color}`).join('\n')}
${results.theme.hardcodedColors.length > 20 ? `\n... and ${results.theme.hardcodedColors.length - 20} more` : ''}

### State Management Issues
${results.architecture.stateManagementIssues.map((issue: string) => `- ${issue}`).join('\n')}

### Performance Issues
${results.architecture.performanceIssues.map((issue: string) => `- ${issue}`).join('\n')}

### Security Concerns
${results.architecture.securityConcerns.map((concern: string) => `- ${concern}`).join('\n')}

## Recommendations

### Phase 2: Component Consolidation
1. Merge duplicate components: ${results.components.duplicateComponents.length} pairs identified
2. Remove unused components: ${results.components.unusedComponents.length} candidates
3. Split large components: ${results.components.largeComponents.length} candidates
4. Refactor complex components: ${results.components.complexComponents.length} candidates

### Phase 3: Theme Migration
1. Replace hardcoded colors with semantic tokens: ${results.theme.hardcodedColors.length} instances
2. Implement modernization opportunities: ${results.theme.modernizationOpportunities.length} identified

### Phase 4: Architecture Optimization
1. Optimize state management: ${results.architecture.stateManagementIssues.length} issues
2. Address performance concerns: ${results.architecture.performanceIssues.length} issues
3. Fix security concerns: ${results.architecture.securityConcerns.length} issues

---

*This report was generated by the Enterprise Refactoring Analyzer*
`;
    
    fs.writeFileSync(reportPath, report);
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const analyzer = new RefactorAnalyzer();
  analyzer.analyzeProject().catch(console.error);
}

export default RefactorAnalyzer;