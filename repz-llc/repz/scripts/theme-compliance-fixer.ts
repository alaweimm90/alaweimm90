#!/usr/bin/env node

/**
 * 🎨 Automated Theme Compliance Fixer
 * Automatically fixes direct color usage violations
 */

import { promises as fs } from 'fs';
import { join } from 'path';

interface ThemeMapping {
  find: RegExp;
  replace: string;
  description: string;
}

class ThemeComplianceFixer {
  private themeMappings: ThemeMapping[] = [
    // Text colors
    { 
      find: /className="([^"]*\s)?text-white(\s[^"]*)?"/g, 
      replace: 'className="$1text-foreground$2"',
      description: 'Replace text-white with semantic text-foreground'
    },
    { 
      find: /className="([^"]*\s)?text-black(\s[^"]*)?"/g, 
      replace: 'className="$1text-foreground$2"',
      description: 'Replace text-black with semantic text-foreground'
    },
    
    // Background colors
    { 
      find: /className="([^"]*\s)?bg-white(\s[^"]*)?"/g, 
      replace: 'className="$1bg-background$2"',
      description: 'Replace bg-white with semantic bg-background'
    },
    { 
      find: /className="([^"]*\s)?bg-black(\s[^"]*)?"/g, 
      replace: 'className="$1bg-background$2"',
      description: 'Replace bg-black with semantic bg-background'
    },

    // Primary colors
    { 
      find: /className="([^"]*\s)?text-orange-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1text-primary$3"',
      description: 'Replace text-orange-* with semantic text-primary'
    },
    { 
      find: /className="([^"]*\s)?bg-orange-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1bg-primary$3"',
      description: 'Replace bg-orange-* with semantic bg-primary'
    },

    // Error colors
    { 
      find: /className="([^"]*\s)?text-red-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1text-destructive$3"',
      description: 'Replace text-red-* with semantic text-destructive'
    },
    { 
      find: /className="([^"]*\s)?bg-red-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1bg-destructive$3"',
      description: 'Replace bg-red-* with semantic bg-destructive'
    },

    // Success colors
    { 
      find: /className="([^"]*\s)?text-green-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1text-success$3"',
      description: 'Replace text-green-* with semantic text-success'
    },
    { 
      find: /className="([^"]*\s)?bg-green-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1bg-success$3"',
      description: 'Replace bg-green-* with semantic bg-success'
    },

    // Warning colors
    { 
      find: /className="([^"]*\s)?text-yellow-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1text-warning$3"',
      description: 'Replace text-yellow-* with semantic text-warning'
    },
    { 
      find: /className="([^"]*\s)?bg-yellow-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1bg-warning$3"',
      description: 'Replace bg-yellow-* with semantic bg-warning'
    },

    // Info colors  
    { 
      find: /className="([^"]*\s)?text-blue-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1text-info$3"',
      description: 'Replace text-blue-* with semantic text-info'
    },
    { 
      find: /className="([^"]*\s)?bg-blue-(\d+)(\s[^"]*)?"/g, 
      replace: 'className="$1bg-info$3"',
      description: 'Replace bg-blue-* with semantic bg-info'
    },

    // Border colors
    { 
      find: /className="([^"]*\s)?border-white(\s[^"]*)?"/g, 
      replace: 'className="$1border-border$2"',
      description: 'Replace border-white with semantic border-border'
    },
    { 
      find: /className="([^"]*\s)?border-black(\s[^"]*)?"/g, 
      replace: 'className="$1border-border$2"',
      description: 'Replace border-black with semantic border-border'
    }
  ];

  async fixFile(filePath: string, dryRun: boolean = false): Promise<{
    fixed: boolean;
    changes: { line: number; old: string; new: string; description: string }[];
  }> {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n');
      let newContent = content;
      const changes: { line: number; old: string; new: string; description: string }[] = [];

      for (const mapping of this.themeMappings) {
        let match;
        while ((match = mapping.find.exec(content)) !== null) {
          const oldMatch = match[0];
          const newMatch = oldMatch.replace(mapping.find, mapping.replace);
          
          if (oldMatch !== newMatch) {
            // Find line number
            const beforeMatch = content.substring(0, match.index);
            const lineNumber = beforeMatch.split('\n').length;
            
            changes.push({
              line: lineNumber,
              old: oldMatch,
              new: newMatch,
              description: mapping.description
            });

            newContent = newContent.replace(oldMatch, newMatch);
          }
        }
        
        // Reset regex
        mapping.find.lastIndex = 0;
      }

      if (changes.length > 0 && !dryRun) {
        await fs.writeFile(filePath, newContent, 'utf-8');
      }

      return {
        fixed: changes.length > 0,
        changes
      };

    } catch (error) {
      console.error(`Error processing ${filePath}:`, error);
      return { fixed: false, changes: [] };
    }
  }

  async fixProject(basePath: string = process.cwd(), dryRun: boolean = false): Promise<{
    totalFiles: number;
    fixedFiles: number;
    totalChanges: number;
    summary: Record<string, number>;
  }> {
    const extensions = ['.tsx', '.ts', '.jsx', '.js'];
    const targetDirs = ['src/components', 'src/pages', 'demo-pages'];
    
    let totalFiles = 0;
    let fixedFiles = 0;
    let totalChanges = 0;
    const summary: Record<string, number> = {};

    console.log(`🎨 ${dryRun ? 'Analyzing' : 'Fixing'} theme compliance issues...\n`);

    for (const dir of targetDirs) {
      const fullDir = join(basePath, dir);
      
      try {
        await fs.access(fullDir);
        const files = await this.getAllFiles(fullDir, extensions);
        
        for (const file of files) {
          totalFiles++;
          const result = await this.fixFile(file, dryRun);
          
          if (result.fixed) {
            fixedFiles++;
            totalChanges += result.changes.length;
            
            console.log(`${dryRun ? '📋' : '✅'} ${file.replace(basePath, '.')}`);
            
            for (const change of result.changes) {
              console.log(`   Line ${change.line}: ${change.description}`);
              if (dryRun) {
                console.log(`     - ${change.old}`);
                console.log(`     + ${change.new}`);
              }
              
              // Track fix types
              const fixType = change.description.split(' ')[1]; // e.g., "text-white"
              summary[fixType] = (summary[fixType] || 0) + 1;
            }
            console.log();
          }
        }
      } catch (error) {
        console.warn(`Warning: Could not process directory ${dir}`);
      }
    }

    console.log(`\n📊 Summary:`);
    console.log(`Files processed: ${totalFiles}`);
    console.log(`Files ${dryRun ? 'with issues' : 'fixed'}: ${fixedFiles}`);
    console.log(`Total changes ${dryRun ? 'needed' : 'made'}: ${totalChanges}`);
    
    if (Object.keys(summary).length > 0) {
      console.log(`\nFix breakdown:`);
      Object.entries(summary)
        .sort(([,a], [,b]) => b - a)
        .forEach(([type, count]) => {
          console.log(`  ${type}: ${count}`);
        });
    }

    return { totalFiles, fixedFiles, totalChanges, summary };
  }

  private async getAllFiles(dir: string, extensions: string[]): Promise<string[]> {
    const files: string[] = [];
    const items = await fs.readdir(dir, { withFileTypes: true });

    for (const item of items) {
      const fullPath = join(dir, item.name);
      
      if (item.isDirectory() && !item.name.startsWith('.') && item.name !== 'node_modules') {
        const subFiles = await this.getAllFiles(fullPath, extensions);
        files.push(...subFiles);
      } else if (extensions.some(ext => item.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }

    return files;
  }

  async generateThemeTokens(): Promise<string> {
    return `/* 🎨 Semantic Design Tokens */
:root {
  /* Enhanced semantic color tokens */
  --success: 142 76% 36%;          /* Green success states */
  --success-foreground: 356 29% 98%;
  
  --warning: 43 96% 56%;           /* Yellow warning states */
  --warning-foreground: 25 95% 53%;
  
  --info: 217 91% 60%;             /* Blue info states */
  --info-foreground: 0 0% 98%;
  
  --tier-core: 240 5% 64%;         /* Core tier accent */
  --tier-adaptive: 330 81% 60%;    /* Adaptive tier accent */
  --tier-performance: 262 83% 58%; /* Performance tier accent */
  --tier-longevity: 48 96% 53%;    /* Longevity tier accent */
}

.dark {
  --success: 142 76% 36%;
  --success-foreground: 356 29% 98%;
  
  --warning: 43 96% 56%;
  --warning-foreground: 25 95% 53%;
  
  --info: 217 91% 60%;
  --info-foreground: 0 0% 98%;
  
  --tier-core: 240 5% 64%;
  --tier-adaptive: 330 81% 60%;
  --tier-performance: 262 83% 58%;
  --tier-longevity: 48 96% 53%;
}

/* Utility classes for semantic colors */
.text-success { color: hsl(var(--success)); }
.bg-success { background-color: hsl(var(--success)); }
.text-warning { color: hsl(var(--warning)); }
.bg-warning { background-color: hsl(var(--warning)); }
.text-info { color: hsl(var(--info)); }
.bg-info { background-color: hsl(var(--info)); }

/* Tier-specific utilities */
.text-tier-core { color: hsl(var(--tier-core)); }
.bg-tier-core { background-color: hsl(var(--tier-core)); }
.text-tier-adaptive { color: hsl(var(--tier-adaptive)); }
.bg-tier-adaptive { background-color: hsl(var(--tier-adaptive)); }
.text-tier-performance { color: hsl(var(--tier-performance)); }
.bg-tier-performance { background-color: hsl(var(--tier-performance)); }
.text-tier-longevity { color: hsl(var(--tier-longevity)); }
.bg-tier-longevity { background-color: hsl(var(--tier-longevity)); }`;
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  const fixer = new ThemeComplianceFixer();

  switch (command) {
    case 'check':
      console.log('🔍 Checking theme compliance (dry run)...\n');
      await fixer.fixProject(process.cwd(), true);
      break;

    case 'fix':
      console.log('🔧 Fixing theme compliance issues...\n');
      const result = await fixer.fixProject(process.cwd(), false);
      
      if (result.fixedFiles > 0) {
        console.log('\n✅ Theme compliance fixes applied successfully!');
        console.log('💡 Run `npm run audit:theme` to verify all issues are resolved.');
      } else {
        console.log('\n🎉 No theme compliance issues found!');
      }
      break;

    case 'tokens':
      const tokens = await fixer.generateThemeTokens();
      await fs.writeFile('src/styles/theme-tokens.css', tokens);
      console.log('✅ Enhanced theme tokens generated in src/styles/theme-tokens.css');
      console.log('💡 Import this file in your main CSS to enable semantic color utilities.');
      break;

    default:
      console.log(`
🎨 Theme Compliance Fixer

Usage:
  npm run theme:check   # Analyze theme violations (dry run)
  npm run theme:fix     # Fix theme violations automatically
  npm run theme:tokens  # Generate enhanced theme tokens

Examples:
  npx ts-node scripts/theme-compliance-fixer.ts check
  npx ts-node scripts/theme-compliance-fixer.ts fix
  npx ts-node scripts/theme-compliance-fixer.ts tokens

Note: Always run 'check' first to preview changes before applying fixes.
`);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { ThemeComplianceFixer };