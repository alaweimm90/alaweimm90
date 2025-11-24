#!/usr/bin/env node

/**
 * Governance Installation Script
 * Sets up all governance tools and integrations
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🛡️ Installing Governance Framework...\n');

// Check if running from repository root
if (!fs.existsSync('.governance')) {
  console.error('❌ Error: Must run from repository root');
  process.exit(1);
}

// Installation steps
const steps = [
  {
    name: 'Install Husky',
    command: 'npm install --save-dev husky',
    check: () => fs.existsSync('node_modules/husky'),
  },
  {
    name: 'Initialize Husky',
    command: 'npx husky install',
    check: () => fs.existsSync('.husky'),
  },
  {
    name: 'Install ESLint',
    command: 'npm install --save-dev eslint eslint-config-prettier',
    check: () => fs.existsSync('node_modules/eslint'),
  },
  {
    name: 'Install Prettier',
    command: 'npm install --save-dev prettier',
    check: () => fs.existsSync('node_modules/prettier'),
  },
  {
    name: 'Install TypeScript',
    command: 'npm install --save-dev typescript @types/node',
    check: () => fs.existsSync('node_modules/typescript'),
  },
  {
    name: 'Install documentation tools',
    command: 'npm install --save-dev jsdoc typedoc doctoc',
    check: () => fs.existsSync('node_modules/jsdoc'),
  },
  {
    name: 'Install testing tools',
    command: 'npm install --save-dev jest @types/jest',
    check: () => fs.existsSync('node_modules/jest'),
  },
];

// Run installation steps
steps.forEach(step => {
  console.log(`📦 ${step.name}...`);
  if (step.check && step.check()) {
    console.log(`  ✅ Already installed`);
  } else {
    try {
      execSync(step.command, { stdio: 'pipe' });
      console.log(`  ✅ Installed successfully`);
    } catch (error) {
      console.log(`  ⚠️ Failed: ${error.message}`);
    }
  }
});

// Create configuration files
console.log('\n📝 Creating configuration files...');

// ESLint configuration
const eslintConfig = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: ['eslint:recommended', 'prettier'],
  parserOptions: {
    ecmaVersion: 12,
    sourceType: 'module',
  },
  rules: {
    'no-unused-vars': 'error',
    'no-console': 'warn',
    'prefer-const': 'error',
    'no-var': 'error',
  },
  overrides: [
    {
      files: ['*.ts', '*.tsx'],
      parser: '@typescript-eslint/parser',
      plugins: ['@typescript-eslint'],
      extends: ['plugin:@typescript-eslint/recommended'],
    },
  ],
};

fs.writeFileSync('.eslintrc.json', JSON.stringify(eslintConfig, null, 2));
console.log('  ✅ Created .eslintrc.json');

// Prettier configuration
const prettierConfig = {
  semi: true,
  trailingComma: 'es5',
  singleQuote: true,
  printWidth: 100,
  tabWidth: 2,
  useTabs: false,
  bracketSpacing: true,
  arrowParens: 'avoid',
};

fs.writeFileSync('.prettierrc.json', JSON.stringify(prettierConfig, null, 2));
console.log('  ✅ Created .prettierrc.json');

// TypeScript configuration
const tsConfig = {
  compilerOptions: {
    target: 'ES2020',
    module: 'commonjs',
    lib: ['ES2020'],
    strict: true,
    esModuleInterop: true,
    skipLibCheck: true,
    forceConsistentCasingInFileNames: true,
    resolveJsonModule: true,
    declaration: true,
    outDir: './dist',
    rootDir: './src',
    noImplicitAny: true,
    strictNullChecks: true,
    strictFunctionTypes: true,
    noUnusedLocals: true,
    noUnusedParameters: true,
    noImplicitReturns: true,
    noFallthroughCasesInSwitch: true,
  },
  include: ['src/**/*'],
  exclude: ['node_modules', 'dist', '**/*.test.ts', '**/*.spec.ts'],
};

fs.writeFileSync('tsconfig.json', JSON.stringify(tsConfig, null, 2));
console.log('  ✅ Created tsconfig.json');

// Jest configuration
const jestConfig = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: ['**/*.test.ts', '**/*.test.js', '**/*.spec.ts', '**/*.spec.js'],
  collectCoverageFrom: [
    'src/**/*.{ts,js}',
    '!src/**/*.d.ts',
    '!src/**/*.test.{ts,js}',
    '!src/**/*.spec.{ts,js}',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  coverageReporters: ['text', 'lcov', 'html'],
  verbose: true,
};

fs.writeFileSync('jest.config.json', JSON.stringify(jestConfig, null, 2));
console.log('  ✅ Created jest.config.json');

// Update package.json scripts
console.log('\n📝 Updating package.json scripts...');
const packageJsonPath = 'package.json';
let packageJson = {};

if (fs.existsSync(packageJsonPath)) {
  packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
}

packageJson.scripts = packageJson.scripts || {};
Object.assign(packageJson.scripts, {
  prepare: 'husky install',
  lint: 'eslint . --ext .js,.jsx,.ts,.tsx',
  'lint:fix': 'eslint . --ext .js,.jsx,.ts,.tsx --fix',
  format: 'prettier --write "**/*.{js,jsx,ts,tsx,json,md,yml,yaml}"',
  'format:check': 'prettier --check "**/*.{js,jsx,ts,tsx,json,md,yml,yaml}"',
  'type-check': 'tsc --noEmit',
  test: 'jest',
  'test:watch': 'jest --watch',
  'test:coverage': 'jest --coverage',
  validate: 'node .governance/validators/repository-validator.js',
  'validate:fix': 'node .governance/validators/repository-validator.js --fix',
  cleanup: 'node .governance/scripts/automated-cleanup.js',
  track: 'node .governance/registry/file-tracking-system.js',
  'governance:install': 'node .governance/setup/install-governance.js',
  'governance:check':
    'npm run lint && npm run format:check && npm run type-check && npm run test && npm run validate',
  'governance:fix': 'npm run lint:fix && npm run format && npm run validate:fix',
});

fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
console.log('  ✅ Updated package.json scripts');

// Set up Git hooks
console.log('\n🎣 Setting up Git hooks...');
try {
  execSync('npx husky add .husky/pre-commit "npm run governance:check"', { stdio: 'pipe' });
  console.log('  ✅ Added pre-commit hook');
} catch (e) {
  // Hook might already exist
  console.log('  ℹ️ Pre-commit hook already configured');
}

// Create VS Code settings
console.log('\n📝 Creating VS Code settings...');
const vscodeDir = '.vscode';
if (!fs.existsSync(vscodeDir)) {
  fs.mkdirSync(vscodeDir);
}

const vscodeSettings = {
  'editor.formatOnSave': true,
  'editor.codeActionsOnSave': {
    'source.fixAll.eslint': true,
  },
  'eslint.validate': ['javascript', 'javascriptreact', 'typescript', 'typescriptreact'],
  'typescript.tsdk': 'node_modules/typescript/lib',
  'files.associations': {
    '*.js': 'javascript',
    '*.ts': 'typescript',
    '*.json': 'json',
  },
  'files.exclude': {
    '**/.git': true,
    '**/node_modules': true,
    '**/dist': true,
    '**/build': true,
    '**/.next': true,
  },
  'governance.enabled': true,
  'governance.autoFix': true,
  'governance.validateOnSave': true,
};

fs.writeFileSync(path.join(vscodeDir, 'settings.json'), JSON.stringify(vscodeSettings, null, 2));
console.log('  ✅ Created VS Code settings');

// Create VS Code extensions recommendations
const vscodeExtensions = {
  recommendations: [
    'dbaeumer.vscode-eslint',
    'esbenp.prettier-vscode',
    'streetsidesoftware.code-spell-checker',
    'ms-vscode.vscode-typescript-tslint-plugin',
    'eamodio.gitlens',
    'donjayamanne.githistory',
    'yzhang.markdown-all-in-one',
    'bierner.markdown-mermaid',
    'davidanson.vscode-markdownlint',
  ],
};

fs.writeFileSync(
  path.join(vscodeDir, 'extensions.json'),
  JSON.stringify(vscodeExtensions, null, 2)
);
console.log('  ✅ Created VS Code extensions recommendations');

// Run initial validation
console.log('\n🔍 Running initial validation...');
try {
  execSync('node .governance/validators/repository-validator.js', { stdio: 'inherit' });
} catch (e) {
  console.log('  ⚠️ Validation found issues - run "npm run governance:fix" to auto-fix');
}

// Create PiecesOS integration config
console.log('\n🧩 Setting up PiecesOS integration...');
const piecesConfig = {
  version: '1.0.0',
  integration: {
    enabled: true,
    autoSync: true,
    contextPreservation: true,
    snippetManagement: true,
  },
  workflows: {
    onSave: ['preserve-context'],
    onCommit: ['sync-snippets'],
    onPush: ['backup-context'],
  },
};

fs.writeFileSync('.pieces.config.json', JSON.stringify(piecesConfig, null, 2));
console.log('  ✅ Created PiecesOS configuration');

// Success message
console.log('\n' + '='.repeat(50));
console.log('✅ GOVERNANCE FRAMEWORK INSTALLED SUCCESSFULLY!');
console.log('='.repeat(50));
console.log('\nAvailable commands:');
console.log('  npm run governance:check  - Run all governance checks');
console.log('  npm run governance:fix    - Auto-fix governance issues');
console.log('  npm run validate         - Validate repository structure');
console.log('  npm run cleanup          - Run automated cleanup');
console.log('  npm run track            - Update file tracking');
console.log(
  '\nGovernance dashboard: file://' +
    path.resolve('.governance/dashboard/governance-dashboard.html')
);
console.log('\n🚀 Your repository is now governance-compliant!');
