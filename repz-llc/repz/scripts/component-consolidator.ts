#!/usr/bin/env node

/**
 * 🧩 Component Consolidation Assistant
 * Identifies and helps consolidate duplicate components
 */

import { promises as fs } from 'fs';
import { join, relative, dirname } from 'path';

interface ComponentInfo {
  name: string;
  path: string;
  exports: string[];
  imports: string[];
  props: string[];
  complexity: number;
  size: number;
}

interface DuplicationAnalysis {
  similarComponents: ComponentInfo[][];
  redundantComponents: string[];
  consolidationPlan: ConsolidationPlan[];
}

interface ConsolidationPlan {
  action: 'merge' | 'archive' | 'refactor';
  components: string[];
  targetPath: string;
  reason: string;
  benefits: string[];
  risks: string[];
}

class ComponentConsolidator {
  private components: ComponentInfo[] = [];
  private basePath: string;

  constructor(basePath: string = process.cwd()) {
    this.basePath = basePath;
  }

  async analyzeComponents(): Promise<DuplicationAnalysis> {
    console.log('🧩 Analyzing component structure...\n');

    await this.scanComponents();
    const analysis = await this.findDuplications();
    
    console.log(`📊 Found ${this.components.length} components`);
    console.log(`🔍 Identified ${analysis.similarComponents.length} potential duplications`);
    
    return analysis;
  }

  private async scanComponents(): Promise<void> {
    const componentDirs = [
      'src/components',
      'src/pages', 
      'demo-pages'
    ];

    for (const dir of componentDirs) {
      try {
        await this.scanDirectory(join(this.basePath, dir));
      } catch (error) {
        console.warn(`Warning: Could not scan ${dir}`);
      }
    }
  }

  private async scanDirectory(dirPath: string): Promise<void> {
    try {
      const items = await fs.readdir(dirPath, { withFileTypes: true });

      for (const item of items) {
        const fullPath = join(dirPath, item.name);
        
        if (item.isDirectory() && !item.name.startsWith('.')) {
          await this.scanDirectory(fullPath);
        } else if (item.name.endsWith('.tsx') || item.name.endsWith('.ts')) {
          const componentInfo = await this.analyzeComponent(fullPath);
          if (componentInfo) {
            this.components.push(componentInfo);
          }
        }
      }
    } catch (error) {
      console.warn(`Warning: Could not read directory ${dirPath}`);
    }
  }

  private async analyzeComponent(filePath: string): Promise<ComponentInfo | null> {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const stats = await fs.stat(filePath);
      
      // Extract component info
      const name = this.extractComponentName(content, filePath);
      const exports = this.extractExports(content);
      const imports = this.extractImports(content);
      const props = this.extractProps(content);
      const complexity = this.calculateComplexity(content);

      return {
        name,
        path: relative(this.basePath, filePath),
        exports,
        imports,
        props,
        complexity,
        size: stats.size
      };
    } catch (error) {
      console.warn(`Warning: Could not analyze ${filePath}`);
      return null;
    }
  }

  private extractComponentName(content: string, filePath: string): string {
    // Try to extract from function/class declaration
    const functionMatch = content.match(/export\s+(?:default\s+)?(?:function\s+)?(\w+)/);
    if (functionMatch) return functionMatch[1];
    
    // Fallback to filename
    const filename = filePath.split('/').pop()?.replace(/\.(tsx?|jsx?)$/, '');
    return filename || 'Unknown';
  }

  private extractExports(content: string): string[] {
    const exports: string[] = [];
    
    // Named exports
    const namedExports = content.match(/export\s+(?:const|function|class)\s+(\w+)/g);
    if (namedExports) {
      exports.push(...namedExports.map(exp => exp.match(/(\w+)$/)?.[1] || ''));
    }
    
    // Export { ... }
    const exportBlocks = content.match(/export\s*\{\s*([^}]+)\s*\}/g);
    if (exportBlocks) {
      exportBlocks.forEach(block => {
        const names = block.match(/\w+/g)?.slice(1); // Skip 'export'
        if (names) exports.push(...names);
      });
    }
    
    // Default export
    if (content.includes('export default')) {
      exports.push('default');
    }
    
    return [...new Set(exports.filter(Boolean))];
  }

  private extractImports(content: string): string[] {
    const imports: string[] = [];
    const importRegex = /import\s+.*?\s+from\s+['"]([^'"]+)['"]/g;
    let match;
    
    while ((match = importRegex.exec(content)) !== null) {
      imports.push(match[1]);
    }
    
    return imports;
  }

  private extractProps(content: string): string[] {
    const props: string[] = [];
    
    // Interface props
    const interfaceMatch = content.match(/interface\s+\w*Props\s*\{([^}]+)\}/);
    if (interfaceMatch) {
      const propMatches = interfaceMatch[1].match(/(\w+)\s*[?:]?/g);
      if (propMatches) {
        props.push(...propMatches.map(p => p.replace(/[?:]/g, '').trim()));
      }
    }
    
    // Destructured props in function params
    const destructuredMatch = content.match(/\(\s*\{\s*([^}]+)\s*\}/);
    if (destructuredMatch) {
      const propNames = destructuredMatch[1].split(',').map(p => p.trim().split(':')[0].trim());
      props.push(...propNames);
    }
    
    return [...new Set(props.filter(Boolean))];
  }

  private calculateComplexity(content: string): number {
    // Simple complexity calculation based on:
    // - Number of functions/hooks
    // - Conditional statements
    // - JSX elements
    // - State management
    
    let complexity = 0;
    
    // Functions and hooks
    complexity += (content.match(/use\w+\(/g) || []).length * 2;
    complexity += (content.match(/function\s+\w+/g) || []).length * 3;
    complexity += (content.match(/const\s+\w+\s*=\s*\(/g) || []).length * 2;
    
    // Control structures
    complexity += (content.match(/if\s*\(/g) || []).length * 2;
    complexity += (content.match(/\?.*\s*:/g) || []).length; // Ternary
    complexity += (content.match(/\.map\(/g) || []).length;
    complexity += (content.match(/\.filter\(/g) || []).length;
    
    // JSX complexity
    complexity += (content.match(/<\w+/g) || []).length * 0.5;
    
    // State management
    complexity += (content.match(/useState/g) || []).length * 2;
    complexity += (content.match(/useEffect/g) || []).length * 3;
    complexity += (content.match(/useContext/g) || []).length * 2;
    
    return Math.round(complexity);
  }

  private async findDuplications(): Promise<DuplicationAnalysis> {
    const similarComponents: ComponentInfo[][] = [];
    const redundantComponents: string[] = [];
    const consolidationPlan: ConsolidationPlan[] = [];

    // Group components by similarity
    const componentGroups = this.groupSimilarComponents();
    
    // Analyze each group
    for (const group of componentGroups) {
      if (group.length > 1) {
        similarComponents.push(group);
        
        // Generate consolidation plan for this group
        const plan = this.generateConsolidationPlan(group);
        if (plan) {
          consolidationPlan.push(plan);
        }
      }
    }

    // Find obviously redundant components
    for (const component of this.components) {
      if (this.isRedundant(component)) {
        redundantComponents.push(component.path);
      }
    }

    return { similarComponents, redundantComponents, consolidationPlan };
  }

  private groupSimilarComponents(): ComponentInfo[][] {
    const groups: ComponentInfo[][] = [];
    const processed = new Set<string>();

    for (const component of this.components) {
      if (processed.has(component.path)) continue;

      const similarGroup = [component];
      processed.add(component.path);

      for (const other of this.components) {
        if (processed.has(other.path)) continue;
        
        if (this.areSimilar(component, other)) {
          similarGroup.push(other);
          processed.add(other.path);
        }
      }

      if (similarGroup.length > 1) {
        groups.push(similarGroup);
      }
    }

    return groups;
  }

  private areSimilar(comp1: ComponentInfo, comp2: ComponentInfo): boolean {
    // Name similarity
    const nameSimilarity = this.calculateStringSimilarity(comp1.name, comp2.name);
    if (nameSimilarity > 0.7) return true;

    // Props similarity  
    const propsOverlap = this.calculateArrayOverlap(comp1.props, comp2.props);
    if (propsOverlap > 0.8 && comp1.props.length > 2) return true;

    // Import similarity (indicates similar dependencies)
    const importOverlap = this.calculateArrayOverlap(comp1.imports, comp2.imports);
    if (importOverlap > 0.6 && comp1.imports.length > 3) return true;

    // Size and complexity similarity
    const sizeDiff = Math.abs(comp1.size - comp2.size) / Math.max(comp1.size, comp2.size);
    const complexityDiff = Math.abs(comp1.complexity - comp2.complexity) / Math.max(comp1.complexity, comp2.complexity);
    
    if (sizeDiff < 0.3 && complexityDiff < 0.4 && comp1.complexity > 10) return true;

    return false;
  }

  private calculateStringSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const editDistance = this.levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array.from({ length: str2.length + 1 }, (_, i) => [i]);
    matrix[0] = Array.from({ length: str1.length + 1 }, (_, i) => i);

    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2[i - 1] === str1[j - 1]) {
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

  private calculateArrayOverlap(arr1: string[], arr2: string[]): number {
    if (arr1.length === 0 && arr2.length === 0) return 1;
    if (arr1.length === 0 || arr2.length === 0) return 0;
    
    const set1 = new Set(arr1);
    const set2 = new Set(arr2);
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  }

  private generateConsolidationPlan(group: ComponentInfo[]): ConsolidationPlan | null {
    if (group.length < 2) return null;

    // Determine the best component to keep (most complex, most used)
    const sorted = group.sort((a, b) => b.complexity - a.complexity);
    const primary = sorted[0];
    const others = sorted.slice(1);

    // Check if these are pricing-related components
    const isPricingRelated = group.some(c => 
      c.name.toLowerCase().includes('pricing') || 
      c.name.toLowerCase().includes('tier') ||
      c.path.includes('pricing')
    );

    if (isPricingRelated) {
      return {
        action: 'merge',
        components: group.map(c => c.path),
        targetPath: 'src/components/ui/unified-pricing-card.tsx',
        reason: 'Multiple pricing components detected',
        benefits: [
          'Single source of truth for pricing',
          'Consistent pricing UX',
          'Easier maintenance',
          'Better performance'
        ],
        risks: [
          'Potential feature loss during merge',
          'Need to update all imports',
          'Testing required for all pricing flows'
        ]
      };
    }

    // Check if these are auth-related components
    const isAuthRelated = group.some(c =>
      c.name.toLowerCase().includes('auth') ||
      c.name.toLowerCase().includes('login') ||
      c.name.toLowerCase().includes('signup')
    );

    if (isAuthRelated) {
      return {
        action: 'merge',
        components: group.map(c => c.path),
        targetPath: 'src/components/auth/unified-auth-modal.tsx',
        reason: 'Multiple auth components detected',
        benefits: [
          'Unified auth experience',
          'Modal-based auth flows',
          'Consistent validation',
          'Better mobile UX'
        ],
        risks: [
          'Need to migrate existing auth logic',
          'Update all auth entry points',
          'Test all authentication flows'
        ]
      };
    }

    // Generic component consolidation
    return {
      action: 'refactor',
      components: group.map(c => c.path),
      targetPath: primary.path,
      reason: `Similar components with ${Math.round(this.calculateArrayOverlap(group[0].props, group[1].props) * 100)}% prop overlap`,
      benefits: [
        'Reduced code duplication',
        'Easier maintenance',
        'Consistent behavior',
        'Smaller bundle size'
      ],
      risks: [
        'Need careful prop mapping',
        'Update import statements',
        'Comprehensive testing required'
      ]
    };
  }

  private isRedundant(component: ComponentInfo): boolean {
    // Check if component is in demo-pages (should be archived)
    if (component.path.startsWith('demo-pages/')) return true;

    // Check if component is very simple and potentially unnecessary
    if (component.complexity < 5 && component.exports.length === 1 && component.size < 1000) {
      return true;
    }

    // Check if component is a simple re-export
    const content = `// Content check needed for ${component.path}`;
    if (content.includes('export { default }') && component.size < 500) {
      return true;
    }

    return false;
  }

  async generateReport(): Promise<string> {
    const analysis = await this.analyzeComponents();

    let report = `# 🧩 Component Consolidation Report

## Summary
- **Total Components:** ${this.components.length}
- **Similar Component Groups:** ${analysis.similarComponents.length}
- **Redundant Components:** ${analysis.redundantComponents.length}
- **Consolidation Plans:** ${analysis.consolidationPlan.length}

## Consolidation Plans

`;

    for (const plan of analysis.consolidationPlan) {
      report += `### ${plan.action.toUpperCase()}: ${plan.targetPath}

**Reason:** ${plan.reason}

**Components to consolidate:**
${plan.components.map(c => `- ${c}`).join('\n')}

**Benefits:**
${plan.benefits.map(b => `- ${b}`).join('\n')}

**Risks:**
${plan.risks.map(r => `- ⚠️ ${r}`).join('\n')}

---

`;
    }

    report += `## Redundant Components (Recommend Archive)

${analysis.redundantComponents.map(c => `- ${c}`).join('\n')}

## Similar Component Groups

`;

    for (const [index, group] of analysis.similarComponents.entries()) {
      report += `### Group ${index + 1}
${group.map(c => `- **${c.name}** (${c.path}) - ${c.complexity} complexity, ${c.props.length} props`).join('\n')}

`;
    }

    report += `## Component Statistics

| Component | Path | Complexity | Size | Props | Exports |
|-----------|------|------------|------|-------|---------|
${this.components
  .sort((a, b) => b.complexity - a.complexity)
  .slice(0, 20)
  .map(c => `| ${c.name} | ${c.path} | ${c.complexity} | ${c.size}B | ${c.props.length} | ${c.exports.length} |`)
  .join('\n')}

---
*Generated on ${new Date().toISOString()}
`;

    return report;
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  const consolidator = new ComponentConsolidator();

  switch (command) {
    case 'analyze':
      console.log('🔍 Analyzing component duplication...\n');
      const analysis = await consolidator.analyzeComponents();
      
      console.log('\n📋 Analysis Results:');
      console.log(`- ${analysis.similarComponents.length} similar component groups found`);
      console.log(`- ${analysis.redundantComponents.length} redundant components found`);
      console.log(`- ${analysis.consolidationPlan.length} consolidation plans generated`);
      break;

    case 'report':
      console.log('📊 Generating consolidation report...\n');
      const report = await consolidator.generateReport();
      await fs.writeFile('component-consolidation-report.md', report);
      console.log('✅ Report saved to component-consolidation-report.md');
      break;

    default:
      console.log(`
🧩 Component Consolidator

Usage:
  npm run components:analyze  # Analyze component duplication
  npm run components:report   # Generate detailed report

Examples:
  npx ts-node scripts/component-consolidator.ts analyze
  npx ts-node scripts/component-consolidator.ts report

This tool helps identify:
- Duplicate pricing components
- Similar auth components  
- Redundant utility components
- Consolidation opportunities
`);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { ComponentConsolidator };
