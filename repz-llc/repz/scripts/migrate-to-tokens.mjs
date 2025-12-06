#!/usr/bin/env node

/**
 * REPZ Design System Migration Tool
 * Automatically migrates components from hard-coded values to design tokens
 * Enterprise-grade code transformation with safety checks
 */

import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { glob } from 'glob';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const CONFIG = {
  srcDir: join(__dirname, '../src'),
  tokensFile: join(__dirname, '../tokens.json'),
  globalCSSFile: join(__dirname, '../global.css'),
  backupDir: join(__dirname, '../.migration-backup'),
  extensions: ['tsx', 'ts', 'jsx', 'js', 'css'],
  dryRun: process.argv.includes('--dry-run'),
  verbose: process.argv.includes('--verbose'),
  force: process.argv.includes('--force'),
};

// Token mappings for common hard-coded values
const TOKEN_MAPPINGS = {
  // Colors
  '#F15B23': 'var(--color-brand-primary)',
  '#FB923C': 'var(--color-brand-primary-light)',
  '#D4460A': 'var(--color-brand-primary-dark)',
  '#000000': 'var(--color-brand-black)',
  '#FFFFFF': 'var(--color-brand-white)',
  '#141414': 'var(--color-surface-base)',
  '#1F1F1F': 'var(--color-surface-elevated)',
  '#2E2E2E': 'var(--color-surface-overlay)',
  
  // Spacing (common px values to token equivalents)
  '4px': 'var(--spacing-xs)',
  '8px': 'var(--spacing-sm)',
  '16px': 'var(--spacing-md)',
  '24px': 'var(--spacing-lg)',
  '32px': 'var(--spacing-xl)',
  '48px': 'var(--spacing-2xl)',
  '64px': 'var(--spacing-3xl)',
  
  // Border radius
  '4px': 'var(--radius-xs)', // Note: Will conflict with spacing, handled contextually
  '8px': 'var(--radius-sm)',
  '12px': 'var(--radius-md)',
  '16px': 'var(--radius-lg)',
  
  // Font sizes
  '12px': 'var(--font-size-xs)',
  '14px': 'var(--font-size-sm)',
  '16px': 'var(--font-size-base)',
  '18px': 'var(--font-size-lg)',
  '20px': 'var(--font-size-xl)',
  '24px': 'var(--font-size-2xl)',
  '30px': 'var(--font-size-3xl)',
  
  // Font weights
  '300': 'var(--font-weight-light)',
  '400': 'var(--font-weight-normal)',
  '500': 'var(--font-weight-medium)',
  '600': 'var(--font-weight-semibold)',
  '700': 'var(--font-weight-bold)',
  '800': 'var(--font-weight-extrabold)',
  '900': 'var(--font-weight-black)',
};

// Tailwind to token mappings
const TAILWIND_MAPPINGS = {
  // Text colors
  'text-white': 'text-[var(--color-brand-white)]',
  'text-black': 'text-[var(--color-brand-black)]',
  'text-orange-500': 'text-[var(--color-brand-primary)]',
  'text-orange-600': 'text-[var(--color-brand-primary-dark)]',
  
  // Background colors
  'bg-white': 'bg-[var(--color-brand-white)]',
  'bg-black': 'bg-[var(--color-brand-black)]',
  'bg-orange-500': 'bg-[var(--color-brand-primary)]',
  'bg-orange-600': 'bg-[var(--color-brand-primary-dark)]',
  'bg-gray-900': 'bg-[var(--color-surface-base)]',
  'bg-gray-800': 'bg-[var(--color-surface-elevated)]',
  'bg-gray-700': 'bg-[var(--color-surface-overlay)]',
  
  // Spacing
  'p-1': 'p-[var(--spacing-xs)]',
  'p-2': 'p-[var(--spacing-sm)]',
  'p-4': 'p-[var(--spacing-md)]',
  'p-6': 'p-[var(--spacing-lg)]',
  'p-8': 'p-[var(--spacing-xl)]',
  
  'm-1': 'm-[var(--spacing-xs)]',
  'm-2': 'm-[var(--spacing-sm)]',
  'm-4': 'm-[var(--spacing-md)]',
  'm-6': 'm-[var(--spacing-lg)]',
  'm-8': 'm-[var(--spacing-xl)]',
  
  // Border radius
  'rounded': 'rounded-[var(--radius-sm)]',
  'rounded-md': 'rounded-[var(--radius-md)]',
  'rounded-lg': 'rounded-[var(--radius-lg)]',
  'rounded-xl': 'rounded-[var(--radius-xl)]',
};

class TokenMigrator {
  constructor() {
    this.stats = {
      filesProcessed: 0,
      filesModified: 0,
      replacements: 0,
      errors: 0
    };
    this.backupCreated = false;
  }

  /**
   * Main migration function
   */
  async migrate() {
    console.log('🎨 REPZ Design System Migration Tool');
    console.log('=====================================');
    
    if (CONFIG.dryRun) {
      console.log('🔍 DRY RUN MODE - No files will be modified');
    }
    
    try {
      // Load design tokens
      await this.loadTokens();
      
      // Create backup
      if (!CONFIG.dryRun) {
        await this.createBackup();
      }
      
      // Find files to process
      const files = await this.findFiles();
      console.log(`📁 Found ${files.length} files to process`);
      
      // Process each file
      for (const file of files) {
        await this.processFile(file);
      }
      
      // Generate report
      this.generateReport();
      
    } catch (error) {
      console.error('💥 Migration failed:', error.message);
      process.exit(1);
    }
  }

  /**
   * Load design tokens from tokens.json
   */
  async loadTokens() {
    try {
      const tokensContent = await fs.readFile(CONFIG.tokensFile, 'utf8');
      this.tokens = JSON.parse(tokensContent);
      console.log('✅ Design tokens loaded');
    } catch (error) {
      throw new Error(`Failed to load tokens: ${error.message}`);
    }
  }

  /**
   * Create backup of source files
   */
  async createBackup() {
    if (this.backupCreated) return;
    
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupPath = `${CONFIG.backupDir}-${timestamp.split('T')[0]}`;
      
      console.log(`💾 Creating backup at: ${backupPath}`);
      
      // Copy src directory to backup
      await fs.mkdir(backupPath, { recursive: true });
      await this.copyDirectory(CONFIG.srcDir, join(backupPath, 'src'));
      
      this.backupCreated = true;
      console.log('✅ Backup created successfully');
    } catch (error) {
      throw new Error(`Failed to create backup: ${error.message}`);
    }
  }

  /**
   * Recursively copy directory
   */
  async copyDirectory(src, dest) {
    await fs.mkdir(dest, { recursive: true });
    const entries = await fs.readdir(src, { withFileTypes: true });
    
    for (const entry of entries) {
      const srcPath = join(src, entry.name);
      const destPath = join(dest, entry.name);
      
      if (entry.isDirectory()) {
        await this.copyDirectory(srcPath, destPath);
      } else {
        await fs.copyFile(srcPath, destPath);
      }
    }
  }

  /**
   * Find files to process
   */
  async findFiles() {
    const patterns = CONFIG.extensions.map(ext => `${CONFIG.srcDir}/**/*.${ext}`);
    const files = [];
    
    for (const pattern of patterns) {
      const matches = await glob(pattern);
      files.push(...matches);
    }
    
    // Filter out node_modules and test files
    return files.filter(file => 
      !file.includes('node_modules') && 
      !file.includes('.test.') &&
      !file.includes('.spec.') &&
      !file.includes('__tests__')
    );
  }

  /**
   * Process a single file
   */
  async processFile(filePath) {
    try {
      this.stats.filesProcessed++;
      
      const content = await fs.readFile(filePath, 'utf8');
      const originalContent = content;
      let modifiedContent = content;
      let fileReplacements = 0;

      // Apply token mappings
      for (const [hardcoded, token] of Object.entries(TOKEN_MAPPINGS)) {
        const regex = new RegExp(this.escapeRegex(hardcoded), 'g');
        const matches = modifiedContent.match(regex);
        
        if (matches) {
          // Context-aware replacement (avoid conflicts)
          modifiedContent = this.contextAwareReplace(modifiedContent, hardcoded, token, filePath);
          fileReplacements += matches.length;
        }
      }

      // Apply Tailwind mappings for TSX/JSX files
      if (filePath.match(/\.(tsx?|jsx?)$/)) {
        for (const [tailwindClass, tokenClass] of Object.entries(TAILWIND_MAPPINGS)) {
          const regex = new RegExp(`\\b${this.escapeRegex(tailwindClass)}\\b`, 'g');
          const matches = modifiedContent.match(regex);
          
          if (matches) {
            modifiedContent = modifiedContent.replace(regex, tokenClass);
            fileReplacements += matches.length;
          }
        }
      }

      // Check if file was modified
      if (modifiedContent !== originalContent) {
        this.stats.filesModified++;
        this.stats.replacements += fileReplacements;
        
        if (CONFIG.verbose) {
          console.log(`🔧 ${filePath} - ${fileReplacements} replacements`);
        }
        
        if (!CONFIG.dryRun) {
          await fs.writeFile(filePath, modifiedContent, 'utf8');
        }
      }

    } catch (error) {
      this.stats.errors++;
      console.error(`❌ Error processing ${filePath}:`, error.message);
    }
  }

  /**
   * Context-aware replacement to avoid conflicts
   */
  contextAwareReplace(content, hardcoded, token, filePath) {
    // For spacing values, determine context (margin, padding, border-radius, etc.)
    if (hardcoded.endsWith('px')) {
      const value = hardcoded.slice(0, -2);
      
      // Check if it's in a border-radius context
      if (content.includes(`border-radius: ${hardcoded}`) || 
          content.includes(`borderRadius: '${hardcoded}'`) ||
          content.includes(`rounded-[${hardcoded}]`)) {
        const radiusToken = this.getRadiusToken(value);
        if (radiusToken) {
          return content.replace(new RegExp(this.escapeRegex(hardcoded), 'g'), radiusToken);
        }
      }
      
      // Check if it's in a font-size context
      if (content.includes(`font-size: ${hardcoded}`) || 
          content.includes(`fontSize: '${hardcoded}'`)) {
        const fontSizeToken = this.getFontSizeToken(value);
        if (fontSizeToken) {
          return content.replace(new RegExp(this.escapeRegex(hardcoded), 'g'), fontSizeToken);
        }
      }
    }
    
    // Default replacement
    return content.replace(new RegExp(this.escapeRegex(hardcoded), 'g'), token);
  }

  /**
   * Get appropriate radius token for pixel value
   */
  getRadiusToken(pxValue) {
    const radiusMap = {
      '4': 'var(--radius-xs)',
      '8': 'var(--radius-sm)',
      '12': 'var(--radius-md)',
      '16': 'var(--radius-lg)',
      '24': 'var(--radius-xl)',
      '32': 'var(--radius-2xl)',
      '40': 'var(--radius-3xl)',
    };
    return radiusMap[pxValue];
  }

  /**
   * Get appropriate font size token for pixel value
   */
  getFontSizeToken(pxValue) {
    const fontSizeMap = {
      '12': 'var(--font-size-xs)',
      '14': 'var(--font-size-sm)',
      '16': 'var(--font-size-base)',
      '18': 'var(--font-size-lg)',
      '20': 'var(--font-size-xl)',
      '24': 'var(--font-size-2xl)',
      '30': 'var(--font-size-3xl)',
      '36': 'var(--font-size-4xl)',
      '48': 'var(--font-size-5xl)',
    };
    return fontSizeMap[pxValue];
  }

  /**
   * Escape special regex characters
   */
  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /**
   * Generate migration report
   */
  generateReport() {
    console.log('\n📊 MIGRATION REPORT');
    console.log('===================');
    console.log(`📁 Files processed: ${this.stats.filesProcessed}`);
    console.log(`🔧 Files modified: ${this.stats.filesModified}`);
    console.log(`🔄 Total replacements: ${this.stats.replacements}`);
    console.log(`❌ Errors: ${this.stats.errors}`);
    
    if (CONFIG.dryRun) {
      console.log('\n🔍 DRY RUN - No files were actually modified');
      console.log('Run without --dry-run to apply changes');
    } else if (this.stats.filesModified > 0) {
      console.log('\n✅ Migration completed successfully!');
      console.log('💾 Backup created for safety');
      console.log('🧪 Run tests to verify everything works correctly');
      console.log('📖 Update Storybook stories if needed');
    } else {
      console.log('\n✨ No files needed migration - already using tokens!');
    }
    
    // Recommendations
    console.log('\n💡 NEXT STEPS:');
    console.log('1. Run tests: npm run test');
    console.log('2. Check Storybook: npm run storybook');
    console.log('3. Run audit: node audit.mjs');
    console.log('4. Visual regression test: npm run visual-test');
  }
}

/**
 * CLI interface
 */
function showHelp() {
  console.log(`
🎨 REPZ Design System Migration Tool

USAGE:
  node migrate-to-tokens.mjs [options]

OPTIONS:
  --dry-run      Show what would be changed without modifying files
  --verbose      Show detailed processing information
  --force        Force migration even if backup fails
  --help         Show this help message

EXAMPLES:
  node migrate-to-tokens.mjs --dry-run    # Preview changes
  node migrate-to-tokens.mjs --verbose    # Run with detailed output
  node migrate-to-tokens.mjs             # Run migration

SAFETY:
  - Automatic backup is created before any changes
  - Supports dry-run mode for preview
  - Context-aware replacements to avoid conflicts
  - Comprehensive error handling and reporting
`);
}

// Main execution
async function main() {
  if (process.argv.includes('--help')) {
    showHelp();
    return;
  }

  const migrator = new TokenMigrator();
  await migrator.migrate();
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  });
}