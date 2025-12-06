#!/usr/bin/env node

/**
 * REPZ Monorepo Setup Script
 * Configures shared configurations across all packages
 * Sets up Turborepo, shared configs, and development environment
 */

import { execSync } from 'child_process';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

// Colors for output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function exec(command, options = {}) {
  log(`Running: ${command}`, 'cyan');
  try {
    return execSync(command, { 
      stdio: 'inherit', 
      cwd: ROOT_DIR,
      ...options 
    });
  } catch (error) {
    log(`Error executing: ${command}`, 'red');
    throw error;
  }
}

function updatePackageJson() {
  log('📦 Updating main package.json with shared configs...', 'blue');
  
  const packageJsonPath = join(ROOT_DIR, 'package.json');
  const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));

  // Add shared config dependencies
  packageJson.devDependencies = {
    ...packageJson.devDependencies,
    'turbo': '^2.0.0',
    '@repz/eslint-config': 'workspace:*',
    '@repz/prettier-config': 'workspace:*',
    '@repz/typescript-config': 'workspace:*'
  };

  // Add Turborepo scripts
  packageJson.scripts = {
    ...packageJson.scripts,
    'build:all': 'turbo run build',
    'lint:all': 'turbo run lint',
    'test:all': 'turbo run test',
    'type-check:all': 'turbo run type-check',
    'clean:all': 'turbo run clean',
    'dev:all': 'turbo run dev --parallel',
    'format': 'prettier --write .',
    'format:check': 'prettier --check .',
    'changeset': 'changeset',
    'version-packages': 'changeset version',
    'release': 'turbo run build --filter=./packages/* && changeset publish'
  };

  // Add prettier config reference
  packageJson.prettier = '@repz/prettier-config';

  writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
  log('✅ Updated main package.json', 'green');
}

function createESLintConfig() {
  log('🔧 Creating ESLint configuration...', 'blue');
  
  const eslintConfig = {
    root: true,
    extends: ['@repz/eslint-config/react', '@repz/eslint-config/typescript'],
    parserOptions: {
      project: ['./tsconfig.json']
    },
    ignorePatterns: [
      'dist',
      'build',
      'node_modules',
      'coverage',
      'storybook-static',
      '*.config.js',
      '*.config.ts',
      'public'
    ],
    overrides: [
      {
        files: ['**/*.stories.{ts,tsx}'],
        extends: ['@repz/eslint-config/react'],
        rules: {
          'react/prop-types': 'off',
          '@typescript-eslint/no-explicit-any': 'off'
        }
      }
    ]
  };

  writeFileSync(
    join(ROOT_DIR, '.eslintrc.json'),
    JSON.stringify(eslintConfig, null, 2) + '\n'
  );
  log('✅ Created .eslintrc.json', 'green');
}

function createPrettierConfig() {
  log('💄 Creating Prettier configuration...', 'blue');
  
  const prettierConfig = `module.exports = require('@repz/prettier-config');
`;

  writeFileSync(join(ROOT_DIR, 'prettier.config.js'), prettierConfig);
  log('✅ Created prettier.config.js', 'green');
}

function updateTSConfig() {
  log('📘 Updating TypeScript configuration...', 'blue');
  
  const tsConfigPath = join(ROOT_DIR, 'tsconfig.json');
  let tsConfig;
  
  if (existsSync(tsConfigPath)) {
    tsConfig = JSON.parse(readFileSync(tsConfigPath, 'utf8'));
  } else {
    tsConfig = {};
  }

  // Extend the shared React config
  tsConfig.extends = '@repz/typescript-config/react.json';
  
  // Ensure project-specific settings are preserved
  tsConfig.compilerOptions = {
    ...tsConfig.compilerOptions,
    baseUrl: '.',
    paths: {
      '@/*': ['./src/*'],
      '@/components/*': ['./src/components/*'],
      '@/pages/*': ['./src/pages/*'],
      '@/hooks/*': ['./src/hooks/*'],
      '@/contexts/*': ['./src/contexts/*'],
      '@/lib/*': ['./src/lib/*'],
      '@/utils/*': ['./src/utils/*'],
      '@/types/*': ['./src/types/*'],
      '@/constants/*': ['./src/constants/*'],
      '@/integrations/*': ['./src/integrations/*']
    }
  };

  writeFileSync(tsConfigPath, JSON.stringify(tsConfig, null, 2) + '\n');
  log('✅ Updated tsconfig.json', 'green');
}

function createChangesetConfig() {
  log('📦 Setting up Changesets for version management...', 'blue');
  
  if (!existsSync(join(ROOT_DIR, '.changeset'))) {
    mkdirSync(join(ROOT_DIR, '.changeset'), { recursive: true });
  }

  const changesetConfig = {
    "$schema": "https://unpkg.com/@changesets/config@2.3.1/schema.json",
    "changelog": "@changesets/cli/changelog",
    "commit": false,
    "fixed": [],
    "linked": [],
    "access": "restricted",
    "baseBranch": "main",
    "updateInternalDependencies": "patch",
    "ignore": []
  };

  writeFileSync(
    join(ROOT_DIR, '.changeset', 'config.json'),
    JSON.stringify(changesetConfig, null, 2) + '\n'
  );

  log('✅ Created Changesets configuration', 'green');
}

function createVSCodeSettings() {
  log('⚙️ Creating VS Code workspace settings...', 'blue');
  
  const vscodeDir = join(ROOT_DIR, '.vscode');
  if (!existsSync(vscodeDir)) {
    mkdirSync(vscodeDir, { recursive: true });
  }

  const settings = {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": "explicit",
      "source.organizeImports": "explicit"
    },
    "typescript.preferences.includePackageJsonAutoImports": "on",
    "typescript.updateImportsOnFileMove.enabled": "always",
    "emmet.includeLanguages": {
      "typescript": "html",
      "typescriptreact": "html"
    },
    "files.associations": {
      "*.css": "tailwindcss"
    },
    "tailwindCSS.experimental.classRegex": [
      ["cva\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"],
      ["cx\\(([^)]*)\\)", "(?:'|\"|`)([^']*)(?:'|\"|`)"]
    ]
  };

  const extensions = {
    "recommendations": [
      "esbenp.prettier-vscode",
      "dbaeumer.vscode-eslint",
      "bradlc.vscode-tailwindcss",
      "ms-vscode.vscode-typescript-next",
      "usernamehw.errorlens",
      "christian-kohler.path-intellisense",
      "formulahendry.auto-rename-tag",
      "ms-vscode.vscode-json"
    ]
  };

  writeFileSync(
    join(vscodeDir, 'settings.json'),
    JSON.stringify(settings, null, 2) + '\n'
  );

  writeFileSync(
    join(vscodeDir, 'extensions.json'),
    JSON.stringify(extensions, null, 2) + '\n'
  );

  log('✅ Created VS Code settings', 'green');
}

function installDependencies() {
  log('📥 Installing dependencies...', 'blue');
  
  // Install Turbo globally if not installed
  try {
    exec('turbo --version', { stdio: 'pipe' });
    log('✅ Turbo already installed', 'green');
  } catch {
    log('Installing Turbo globally...', 'yellow');
    exec('npm install -g turbo');
  }

  // Install Changesets if needed
  try {
    exec('changeset --version', { stdio: 'pipe' });
    log('✅ Changesets already installed', 'green');
  } catch {
    log('Installing Changesets...', 'yellow');
    exec('npm install -D @changesets/cli');
  }

  // Install all dependencies
  exec('npm install');
  log('✅ Dependencies installed', 'green');
}

function runInitialBuild() {
  log('🏗️ Running initial build to verify setup...', 'blue');
  
  try {
    exec('turbo run type-check');
    log('✅ Type checking passed', 'green');
  } catch (error) {
    log('⚠️ Type checking failed - this is expected on first setup', 'yellow');
  }

  try {
    exec('turbo run lint');
    log('✅ Linting passed', 'green');
  } catch (error) {
    log('⚠️ Linting failed - running fix...', 'yellow');
    try {
      exec('turbo run lint:fix');
      log('✅ Linting issues fixed', 'green');
    } catch {
      log('⚠️ Some linting issues need manual attention', 'yellow');
    }
  }
}

function printNextSteps() {
  log('\n🎉 Monorepo setup complete!', 'green');
  log('\n📋 Next steps:', 'bright');
  log('1. Run "npm run dev" to start development', 'cyan');
  log('2. Run "npm run build:all" to build all packages', 'cyan');
  log('3. Run "npm run lint:all" to lint all packages', 'cyan');
  log('4. Run "npm run test:all" to test all packages', 'cyan');
  log('5. Use "npm run changeset" to create changesets for releases', 'cyan');
  
  log('\n🔧 Available commands:', 'bright');
  log('• npm run dev:all - Start all packages in dev mode', 'blue');
  log('• npm run build:all - Build all packages', 'blue');
  log('• npm run lint:all - Lint all packages', 'blue');
  log('• npm run test:all - Test all packages', 'blue');
  log('• npm run type-check:all - Type check all packages', 'blue');
  log('• npm run clean:all - Clean all build artifacts', 'blue');
  log('• npm run format - Format all files with Prettier', 'blue');
  
  log('\n📚 Shared configurations:', 'bright');
  log('• ESLint: @repz/eslint-config', 'magenta');
  log('• Prettier: @repz/prettier-config', 'magenta');
  log('• TypeScript: @repz/typescript-config', 'magenta');
  log('• Turborepo: turbo.json', 'magenta');
}

// Main execution
async function main() {
  log('🚀 Setting up REPZ Monorepo...', 'bright');
  
  try {
    updatePackageJson();
    createESLintConfig();
    createPrettierConfig();
    updateTSConfig();
    createChangesetConfig();
    createVSCodeSettings();
    installDependencies();
    runInitialBuild();
    printNextSteps();
    
    log('\n✨ Setup completed successfully!', 'green');
  } catch (error) {
    log(`\n❌ Setup failed: ${error.message}`, 'red');
    process.exit(1);
  }
}

// Run the setup
main().catch(console.error);