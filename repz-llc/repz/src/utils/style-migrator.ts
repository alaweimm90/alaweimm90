// Style migration utilities - Tools to migrate existing code to centralized design system
// Run these utilities to automatically convert hardcoded styles to design tokens

import * as fs from 'fs';
import * as path from 'path';
import { DESIGN_SYSTEM_RULES } from '../config/enhanced-design-system';

// Style mapping - converts hardcoded classes to design tokens
const STYLE_MIGRATIONS = {
  // Color migrations
  'bg-blue-500': 'bg-tier-core',
  'bg-blue-600': 'bg-tier-core',
  'bg-orange-500': 'bg-tier-adaptive', 
  'bg-orange-600': 'bg-tier-adaptive',
  'bg-purple-500': 'bg-tier-performance',
  'bg-purple-600': 'bg-tier-performance',
  'bg-yellow-500': 'bg-tier-longevity',
  'bg-yellow-600': 'bg-tier-longevity',
  
  'text-blue-500': 'text-tier-core',
  'text-blue-600': 'text-tier-core',
  'text-orange-500': 'text-tier-adaptive',
  'text-orange-600': 'text-tier-adaptive', 
  'text-purple-500': 'text-tier-performance',
  'text-purple-600': 'text-tier-performance',
  'text-yellow-500': 'text-tier-longevity',
  'text-yellow-600': 'text-tier-longevity',
  
  'border-blue-500': 'border-tier-core',
  'border-orange-500': 'border-tier-adaptive',
  'border-purple-500': 'border-tier-performance', 
  'border-yellow-500': 'border-tier-longevity',
  
  // Common UI colors to semantic tokens
  'bg-gray-900': 'bg-background',
  'bg-gray-800': 'bg-surface-elegant',
  'bg-gray-700': 'bg-surface-velvet',
  'bg-white': 'bg-background',
  'text-white': 'text-foreground',
  'text-gray-300': 'text-muted-foreground',
  'text-gray-400': 'text-muted-foreground',
  'text-gray-500': 'text-muted-foreground',
  'text-gray-600': 'text-muted-foreground',
  'text-black': 'text-foreground',
  
  // Component-specific migrations
  'px-4 py-2': 'px-md py-sm',
  'px-6 py-3': 'px-lg py-md',
  'px-8 py-4': 'px-xl py-lg',
  'rounded-lg': 'rounded-lg', // Already correct
  'rounded-md': 'rounded-md', // Already correct
} as const;

// Component pattern migrations - converts inline styles to variants
const COMPONENT_MIGRATIONS = {
  // Button patterns
  buttonPatterns: [
    {
      pattern: /className="([^"]*bg-blue-[0-9]+[^"]*)"/, 
      replacement: 'variant="tierCore"',
      component: 'Button'
    },
    {
      pattern: /className="([^"]*bg-orange-[0-9]+[^"]*)"/, 
      replacement: 'variant="tierAdaptive"',
      component: 'Button'
    },
    {
      pattern: /className="([^"]*bg-purple-[0-9]+[^"]*)"/, 
      replacement: 'variant="tierPerformance"',
      component: 'Button'
    },
    {
      pattern: /className="([^"]*bg-yellow-[0-9]+[^"]*)"/, 
      replacement: 'variant="tierLongevity"',
      component: 'Button'
    }
  ],
  
  // Card patterns
  cardPatterns: [
    {
      pattern: /className="([^"]*shadow-lg[^"]*)"/, 
      replacement: 'variant="elevated"',
      component: 'Card'
    },
    {
      pattern: /className="([^"]*backdrop-blur[^"]*)"/, 
      replacement: 'variant="glass"',
      component: 'Card'
    }
  ]
};

// File processing utilities
export class StyleMigrator {
  private processedFiles: Set<string> = new Set();
  private migrationLog: Array<{
    file: string;
    changes: Array<{ from: string; to: string; line: number }>;
  }> = [];

  // Migrate a single file
  async migrateFile(filePath: string): Promise<void> {
    if (this.processedFiles.has(filePath)) {
      return;
    }

    try {
      const content = await fs.promises.readFile(filePath, 'utf-8');
      let updatedContent = content;
      const changes: Array<{ from: string; to: string; line: number }> = [];

      // Apply style migrations
      for (const [oldStyle, newStyle] of Object.entries(STYLE_MIGRATIONS)) {
        const regex = new RegExp(oldStyle, 'g');
        if (regex.test(updatedContent)) {
          const lines = updatedContent.split('\n');
          lines.forEach((line, index) => {
            if (line.includes(oldStyle)) {
              changes.push({
                from: oldStyle,
                to: newStyle,
                line: index + 1
              });
            }
          });
          updatedContent = updatedContent.replace(regex, newStyle);
        }
      }

      // Apply component pattern migrations
      for (const pattern of COMPONENT_MIGRATIONS.buttonPatterns) {
        if (pattern.pattern.test(updatedContent)) {
          updatedContent = updatedContent.replace(pattern.pattern, pattern.replacement);
        }
      }

      for (const pattern of COMPONENT_MIGRATIONS.cardPatterns) {
        if (pattern.pattern.test(updatedContent)) {
          updatedContent = updatedContent.replace(pattern.pattern, pattern.replacement);
        }
      }

      // Write updated content if changes were made
      if (content !== updatedContent) {
        await fs.promises.writeFile(filePath, updatedContent, 'utf-8');
        this.migrationLog.push({
          file: filePath,
          changes
        });
      }

      this.processedFiles.add(filePath);
    } catch (error) {
      console.error(`Error migrating file ${filePath}:`, error);
    }
  }

  // Migrate entire directory
  async migrateDirectory(dirPath: string): Promise<void> {
    const files = await this.getReactFiles(dirPath);
    
    for (const file of files) {
      await this.migrateFile(file);
    }
  }

  // Get all React/TypeScript files in directory
  private async getReactFiles(dirPath: string): Promise<string[]> {
    const files: string[] = [];
    
    const scan = async (currentPath: string) => {
      const entries = await fs.promises.readdir(currentPath, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          await scan(fullPath);
        } else if (entry.isFile() && /\.(tsx?|jsx?)$/.test(entry.name)) {
          files.push(fullPath);
        }
      }
    };
    
    await scan(dirPath);
    return files;
  }

  // Generate migration report
  generateReport(): string {
    const totalFiles = this.migrationLog.length;
    const totalChanges = this.migrationLog.reduce((sum, file) => sum + file.changes.length, 0);
    
    let report = `# Style Migration Report\n\n`;
    report += `**Files Processed**: ${totalFiles}\n`;
    report += `**Total Changes**: ${totalChanges}\n\n`;
    
    if (totalChanges === 0) {
      report += `✅ No migrations needed - all styles already use design tokens!\n`;
      return report;
    }
    
    report += `## Files Modified\n\n`;
    
    for (const fileLog of this.migrationLog) {
      report += `### ${fileLog.file}\n`;
      report += `**Changes**: ${fileLog.changes.length}\n\n`;
      
      for (const change of fileLog.changes) {
        report += `- Line ${change.line}: \`${change.from}\` → \`${change.to}\`\n`;
      }
      
      report += `\n`;
    }
    
    return report;
  }

  // Validate files follow design system rules
  async validateFile(filePath: string): Promise<Array<{
    file: string;
    violations: Array<{ line: number; issue: string; suggestion: string }>;
  }>> {
    const content = await fs.promises.readFile(filePath, 'utf-8');
    const lines = content.split('\n');
    const violations: Array<{ line: number; issue: string; suggestion: string }> = [];

    lines.forEach((line, index) => {
      // Check for banned classes
      for (const bannedClass of DESIGN_SYSTEM_RULES.bannedClasses) {
        if (line.includes(bannedClass)) {
          const suggestion = STYLE_MIGRATIONS[bannedClass as keyof typeof STYLE_MIGRATIONS] || 'Use design system token';
          violations.push({
            line: index + 1,
            issue: `Direct color class: ${bannedClass}`,
            suggestion: `Replace with: ${suggestion}`
          });
        }
      }
      
      // Check for hardcoded hex colors
      const hexColorRegex = /#[0-9a-fA-F]{6}/g;
      const hexMatches = line.match(hexColorRegex);
      if (hexMatches) {
        violations.push({
          line: index + 1,
          issue: `Hardcoded hex color: ${hexMatches.join(', ')}`,
          suggestion: 'Use HSL color from design system'
        });
      }
    });

    return [{
      file: filePath,
      violations
    }];
  }
}

// CLI interface for running migrations
export const runStyleMigration = async (targetPath: string = 'src') => {
  const migrator = new StyleMigrator();
  
  console.log('🎨 Starting style migration to centralized design system...');
  
  await migrator.migrateDirectory(targetPath);
  
  const report = migrator.generateReport();
  console.log(report);
  
  // Save report to file
  await fs.promises.writeFile(
    `style-migration-report-${Date.now()}.md`, 
    report, 
    'utf-8'
  );
  
  console.log('✅ Style migration complete!');
};

// Validation CLI
export const validateStyleCompliance = async (targetPath: string = 'src') => {
  const migrator = new StyleMigrator();
  const files = await migrator['getReactFiles'](targetPath);
  
  console.log('🔍 Validating design system compliance...');
  
  let totalViolations = 0;
  
  for (const file of files) {
    const results = await migrator.validateFile(file);
    for (const result of results) {
      if (result.violations.length > 0) {
        console.log(`\n❌ ${result.file}:`);
        for (const violation of result.violations) {
          console.log(`  Line ${violation.line}: ${violation.issue}`);
          console.log(`  💡 ${violation.suggestion}`);
        }
        totalViolations += result.violations.length;
      }
    }
  }
  
  if (totalViolations === 0) {
    console.log('\n✅ All files comply with design system standards!');
  } else {
    console.log(`\n⚠️  Found ${totalViolations} design system violations`);
    console.log('Run style migration to fix automatically');
  }
  
  return totalViolations === 0;
};

export default StyleMigrator;