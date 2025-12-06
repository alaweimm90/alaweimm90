#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🏗️  Fixing root directory structure...\n');

// Define target structure
const targetDirs = {
  scripts: 'scripts',
  docs: 'docs',
  branding: 'branding',
  storybook: '.storybook'
};

// Create target directories
Object.values(targetDirs).forEach(dir => {
  const dirPath = path.join(rootDir, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`📁 Created: ${dir}/`);
  }
});

// File mapping rules
const moveRules = [
  // Scripts and tools
  { pattern: /\.mjs$/, dest: 'scripts', exclude: ['vite.config.mjs'] },
  { pattern: /^(audit|cleanup|verify|check|test-.*)\.(js|ts)$/, dest: 'scripts' },
  
  // Documentation (but preserve key files in root)
  { pattern: /\.md$/, dest: 'docs', exclude: ['README.md', 'CLAUDE.md', 'CONTRIBUTING.md', 'SECURITY.md'] },
  
  // Branding assets
  { pattern: /^logo/, dest: 'branding' },
  { pattern: /\.(svg|png|jpg|jpeg|gif|ico)$/, dest: 'branding', exclude: ['public', 'src'] },
  { pattern: /tokens\.json$/, dest: 'branding' },
  
  // Storybook configs
  { pattern: /\.stories\.(tsx?|jsx?)$/, dest: '.storybook/stories' },
  { pattern: /storybook/, dest: '.storybook', exclude: ['node_modules'] }
];

// Statistics
const stats = {
  moved: 0,
  skipped: 0,
  errors: 0
};

// Function to check if file should be excluded
function shouldExclude(file, excludeList = []) {
  return excludeList.some(exclude => {
    if (typeof exclude === 'string') {
      return file.includes(exclude);
    }
    return false;
  });
}

// Function to move file using git mv or fs
function moveFile(source, destination) {
  try {
    const destDir = path.dirname(destination);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    // Try git mv first
    try {
      execSync(`git mv "${source}" "${destination}"`, { 
        stdio: 'pipe',
        cwd: rootDir 
      });
    } catch {
      // Fallback to regular move
      fs.renameSync(source, destination);
    }
    
    console.log(`✅ Moved: ${path.relative(rootDir, source)} → ${path.relative(rootDir, destination)}`);
    stats.moved++;
    return true;
  } catch (error) {
    console.error(`❌ Error moving ${path.relative(rootDir, source)}:`, error.message);
    stats.errors++;
    return false;
  }
}

// Scan root directory
console.log('🔍 Scanning root directory...\n');

const rootFiles = fs.readdirSync(rootDir);

rootFiles.forEach(file => {
  const filePath = path.join(rootDir, file);
  const stat = fs.statSync(filePath);
  
  // Skip directories (except for specific cases)
  if (stat.isDirectory()) {
    // Skip already organized directories
    if (['src', 'public', 'node_modules', 'dist', 'build', '.git', 'scripts', 'docs', 'branding', '_graveyard'].includes(file)) {
      return;
    }
    
    // Handle logo directory specially
    if (file === 'logo') {
      const logoFiles = fs.readdirSync(filePath);
      logoFiles.forEach(logoFile => {
        const logoPath = path.join(filePath, logoFile);
        const destPath = path.join(rootDir, 'branding', 'logo', logoFile);
        moveFile(logoPath, destPath);
      });
      
      // Remove empty logo directory
      try {
        fs.rmdirSync(filePath);
        console.log(`📁 Removed empty directory: logo/`);
      } catch {}
      return;
    }
  }
  
  // Check file against move rules
  let moved = false;
  for (const rule of moveRules) {
    if (rule.pattern.test(file)) {
      if (rule.exclude && shouldExclude(file, rule.exclude)) {
        console.log(`⏩ Skipped (excluded): ${file}`);
        stats.skipped++;
        break;
      }
      
      const destPath = path.join(rootDir, rule.dest, file);
      if (moveFile(filePath, destPath)) {
        moved = true;
        break;
      }
    }
  }
  
  if (!moved && stat.isFile()) {
    // Log files that weren't moved
    const ext = path.extname(file);
    if (!['.ts', '.tsx', '.js', '.jsx', '.json', '.css', '.html'].includes(ext) && 
        !file.startsWith('.') &&
        !['package.json', 'package-lock.json', 'tsconfig.json', 'vite.config.ts', 'tailwind.config.ts', 'postcss.config.js'].includes(file)) {
      console.log(`❓ Unhandled file: ${file}`);
    }
  }
});

// Special handling for scattered files
console.log('\n🔍 Looking for scattered files in subdirectories...\n');

// Move any .md files from src/ to docs/
function findAndMoveMdFiles(dir) {
  if (dir.includes('node_modules') || dir.includes('_graveyard')) return;
  
  try {
    const files = fs.readdirSync(dir);
    files.forEach(file => {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        findAndMoveMdFiles(fullPath);
      } else if (file.endsWith('.md') && !dir.includes('docs')) {
        const relativePath = path.relative(rootDir, fullPath);
        const destPath = path.join(rootDir, 'docs', 'auto-moved', relativePath);
        moveFile(fullPath, destPath);
      }
    });
  } catch (error) {
    // Ignore permission errors
  }
}

findAndMoveMdFiles(path.join(rootDir, 'src'));

// Create structure documentation
const structureDoc = `# Repository Structure

Last updated: ${new Date().toISOString()}

## Directory Organization

\`\`\`
repzcoach-pro-ai/
├── src/                    # Source code (to be refactored to feature-first)
├── public/                 # Static assets
├── scripts/                # Build, audit, and maintenance scripts
├── docs/                   # Documentation
│   ├── architecture/       # System architecture docs
│   ├── development/        # Development guides
│   ├── operations/         # Deployment and operations
│   └── archive/            # Historical documentation
├── branding/               # Brand assets and design tokens
│   ├── logo/              # Logo files
│   └── tokens.json        # Design tokens
├── .storybook/            # Storybook configuration
├── .github/               # GitHub workflows and templates
└── _graveyard/            # Quarantined unused files
\`\`\`

## Key Files in Root

- \`README.md\` - Project overview and setup
- \`CLAUDE.md\` - AI assistant instructions
- \`CONTRIBUTING.md\` - Contribution guidelines
- \`SECURITY.md\` - Security policies
- \`package.json\` - Node.js dependencies
- \`tsconfig.json\` - TypeScript configuration
- \`vite.config.ts\` - Vite bundler configuration
- \`tailwind.config.ts\` - Tailwind CSS configuration
- \`postcss.config.js\` - PostCSS configuration
`;

fs.writeFileSync(path.join(rootDir, 'docs', 'STRUCTURE.md'), structureDoc);

// Summary
console.log('\n📊 Root Structure Fix Summary:');
console.log('─'.repeat(50));
console.log(`Files moved: ${stats.moved}`);
console.log(`Files skipped: ${stats.skipped}`);
console.log(`Errors: ${stats.errors}`);
console.log('─'.repeat(50));
console.log('\n✅ Root directory structure has been organized!');
console.log('📄 Structure documented in: docs/STRUCTURE.md');

process.exit(0);