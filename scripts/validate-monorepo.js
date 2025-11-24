#!/usr/bin/env node

/**
 * Comprehensive Monorepo Validation Script
 * Validates all aspects of the monorepo setup
 */

const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

const results = {
  passed: [],
  failed: [],
  warnings: [],
};

function pass(message) {
  results.passed.push(message);
  console.log(chalk.green('✓'), message);
}

function fail(message) {
  results.failed.push(message);
  console.log(chalk.red('✗'), message);
}

function warn(message) {
  results.warnings.push(message);
  console.log(chalk.yellow('⚠'), message);
}

function section(name) {
  console.log('\n' + chalk.bold.blue(`=== ${name} ===`) + '\n');
}

// Validation checks
section('Package Structure');

// Check root package.json
if (fs.existsSync('package.json')) {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf-8'));

  if (pkg.devDependencies['@types/jest'] === '^29.5.11') {
    pass('@types/jest version corrected (^29.5.11)');
  } else {
    fail(`@types/jest version incorrect: ${pkg.devDependencies['@types/jest']}`);
  }

  if (pkg.dependencies.uuid === '^9.0.1') {
    pass('uuid version corrected (^9.0.1)');
  } else {
    fail(`uuid version incorrect: ${pkg.dependencies.uuid}`);
  }

  if (pkg.dependencies.express === '^4.18.0') {
    pass('express version stable (^4.18.0)');
  } else {
    fail(`express version incorrect: ${pkg.dependencies.express}`);
  }
} else {
  fail('package.json not found');
}

section('Workspace Configuration');

// Check pnpm-workspace.yaml
if (fs.existsSync('pnpm-workspace.yaml')) {
  pass('pnpm-workspace.yaml exists');
  const content = fs.readFileSync('pnpm-workspace.yaml', 'utf-8');
  if (content.includes('packages/*')) {
    pass('Workspace includes packages/*');
  }
} else {
  fail('pnpm-workspace.yaml not found');
}

section('Build Configuration');

// Check turbo.json
if (fs.existsSync('turbo.json')) {
  pass('turbo.json exists');
  const turbo = JSON.parse(fs.readFileSync('turbo.json', 'utf-8'));

  if (turbo.pipeline) {
    pass('Turbo pipeline configured');

    const requiredTasks = ['build', 'test', 'lint', 'type-check'];
    requiredTasks.forEach((task) => {
      if (turbo.pipeline[task]) {
        pass(`Turbo task configured: ${task}`);
      } else {
        warn(`Turbo task missing: ${task}`);
      }
    });

    if (turbo.pipeline.build && turbo.pipeline.build.cache) {
      pass('Build caching enabled');
    }
  }
} else {
  fail('turbo.json not found');
}

section('Core Packages');

// Check core packages exist
const corePackages = [
  'mcp-core',
  'agent-core',
  'context-provider',
  'workflow-templates',
  'issue-library',
  'shared-utils',
];

corePackages.forEach((pkg) => {
  const pkgPath = path.join('packages', pkg);
  if (fs.existsSync(pkgPath)) {
    pass(`Package exists: ${pkg}`);

    const pkgJsonPath = path.join(pkgPath, 'package.json');
    if (fs.existsSync(pkgJsonPath)) {
      pass(`  package.json exists for ${pkg}`);
    } else {
      fail(`  package.json missing for ${pkg}`);
    }
  } else {
    if (pkg === 'shared-utils') {
      warn(`Package not found: ${pkg} (newly created)`);
    } else {
      fail(`Package not found: ${pkg}`);
    }
  }
});

section('Git Configuration');

// Check .gitignore
if (fs.existsSync('.gitignore')) {
  pass('.gitignore exists');
  const content = fs.readFileSync('.gitignore', 'utf-8');

  const requiredPatterns = ['.backup_*', '.cache/backups-*', '*-duplicate/'];
  requiredPatterns.forEach((pattern) => {
    if (content.includes(pattern)) {
      pass(`  .gitignore includes: ${pattern}`);
    } else {
      warn(`  .gitignore missing: ${pattern}`);
    }
  });
} else {
  fail('.gitignore not found');
}

section('Documentation');

// Check documentation files
const requiredDocs = [
  'START_HERE.md',
  'MONOREPO_ANALYSIS_SUMMARY.md',
  'P0_FIXES_COMPLETED.md',
  'DELIVERY_SUMMARY.md',
];

requiredDocs.forEach((doc) => {
  if (fs.existsSync(doc)) {
    pass(`Documentation exists: ${doc}`);
  } else {
    warn(`Documentation missing: ${doc}`);
  }
});

section('TypeScript Configuration');

// Check tsconfig.json
if (fs.existsSync('tsconfig.json')) {
  pass('tsconfig.json exists');
  const tsconfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf-8'));

  if (tsconfig.compilerOptions && tsconfig.compilerOptions.strict) {
    pass('Strict mode enabled');
  } else {
    warn('Strict mode not enabled');
  }
} else {
  fail('tsconfig.json not found');
}

// Summary
section('Summary');

console.log(`${chalk.green('Passed:')} ${results.passed.length}`);
console.log(`${chalk.red('Failed:')} ${results.failed.length}`);
console.log(`${chalk.yellow('Warnings:')} ${results.warnings.length}`);

const total = results.passed.length + results.failed.length + results.warnings.length;
const successRate = ((results.passed.length / total) * 100).toFixed(1);

console.log(`\n${chalk.bold('Success Rate:')} ${successRate}%\n`);

if (results.failed.length === 0) {
  console.log(chalk.green.bold('✅ VALIDATION PASSED!\n'));
  process.exit(0);
} else {
  console.log(chalk.red.bold('❌ VALIDATION FAILED\n'));
  console.log(chalk.yellow('Fix the issues above and re-run validation.\n'));
  process.exit(1);
}
