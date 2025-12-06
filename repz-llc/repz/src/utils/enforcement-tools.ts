// Pre-commit hooks and linting configuration for design system enforcement
// This file contains automated tools to prevent design system violations

export const eslintDesignSystemRules = {
  // ESLint plugin configuration for design system enforcement
  '@typescript-eslint/design-system': {
    // Ban direct Tailwind color classes
    'no-direct-colors': 'error',
    'require-design-tokens': 'error',
    'no-hardcoded-styles': 'warn',
    'component-variant-required': 'error',
  },
  
  // Custom rules
  rules: {
    // Prevent direct color usage
    'no-restricted-syntax': [
      'error',
      {
        selector: 'Literal[value=/bg-(blue|red|green|yellow|purple)-[0-9]+/]',
        message: 'Use design system tier colors instead of direct Tailwind colors'
      },
      {
        selector: 'Literal[value=/text-(blue|red|green|yellow|purple)-[0-9]+/]',
        message: 'Use design system tier colors instead of direct Tailwind colors'
      },
      {
        selector: 'Literal[value=/#[0-9a-fA-F]{6}/]',
        message: 'Use HSL colors from design system instead of hex colors'
      }
    ],
    
    // Require imports from centralized locations
    'import/no-restricted-paths': [
      'error',
      {
        zones: [
          {
            target: './src/**/*',
            from: './src/**/*',
            except: ['./src/constants/**', './src/config/**', './src/types/**'],
            message: 'Import types and constants from centralized locations only'
          }
        ]
      }
    ]
  }
};

export const stylelintConfig = {
  extends: ['stylelint-config-standard'],
  rules: {
    // Require CSS custom properties for colors
    'declaration-property-value-disallowed-list': {
      '/^(background|color|border).*$/': ['/^#[0-9a-fA-F]+$/'],
      message: 'Use CSS custom properties instead of hex colors'
    },
    
    // Enforce design token usage
    'custom-property-pattern': '^(primary|secondary|accent|tier|surface|repz)-.*$',
    
    // Prevent hardcoded values
    'declaration-property-value-no-unknown': true,
  }
};

export const preCommitHooks = {
  // Git hooks configuration
  'pre-commit': `#!/bin/sh
# Design system enforcement pre-commit hook

echo "🎨 Checking design system compliance..."

# Run style validation
npm run validate:styles
STYLE_EXIT_CODE=$?

# Run type checking
npm run type-check  
TYPE_EXIT_CODE=$?

# Run linting
npm run lint
LINT_EXIT_CODE=$?

# Check for direct color usage
echo "🔍 Scanning for hardcoded colors..."
HARDCODED_COLORS=$(git diff --cached --name-only | xargs grep -l "bg-blue-\\|bg-red-\\|bg-green-\\|bg-yellow-\\|bg-purple-\\|#[0-9a-fA-F]\\{6\\}" || true)

if [ ! -z "$HARDCODED_COLORS" ]; then
  echo "❌ Hardcoded colors found in:"
  echo "$HARDCODED_COLORS"
  echo ""
  echo "Please use design system tokens instead:"
  echo "  bg-blue-500 → bg-tier-core"
  echo "  bg-orange-500 → bg-tier-adaptive" 
  echo "  bg-purple-500 → bg-tier-performance"
  echo "  bg-yellow-500 → bg-tier-longevity"
  echo ""
  echo "Run 'npm run migrate:styles' to fix automatically"
  exit 1
fi

# Check for route violations
echo "🧭 Checking navigation compliance..."
HARDCODED_ROUTES=$(git diff --cached --name-only | xargs grep -l "to=\\"/" || true)

if [ ! -z "$HARDCODED_ROUTES" ]; then
  echo "⚠️  Potential hardcoded routes found. Ensure you're using ROUTES constants."
fi

# Exit with error if any checks failed
if [ $STYLE_EXIT_CODE -ne 0 ] || [ $TYPE_EXIT_CODE -ne 0 ] || [ $LINT_EXIT_CODE -ne 0 ]; then
  echo "❌ Pre-commit checks failed. Please fix issues before committing."
  exit 1
fi

echo "✅ All design system checks passed!"
exit 0`,

  'commit-msg': `#!/bin/sh
# Commit message linting

# Check for conventional commit format
commit_regex="^(feat|fix|docs|style|refactor|test|chore)(\\(.+\\))?: .{1,50}"

if ! grep -qE "$commit_regex" "$1"; then
  echo "❌ Invalid commit message format!"
  echo ""
  echo "Use conventional commit format:"
  echo "  feat: add new feature"
  echo "  fix: resolve bug"
  echo "  style: update design system"
  echo "  refactor: centralize navigation"
  echo ""
  exit 1
fi`,

  'pre-push': `#!/bin/sh
# Run comprehensive audit before push

echo "🔍 Running comprehensive codebase audit..."

# Run full audit
npm run audit:full
AUDIT_EXIT_CODE=$?

if [ $AUDIT_EXIT_CODE -ne 0 ]; then
  echo "❌ Audit found critical issues. Push blocked."
  echo "Run 'npm run audit:fix' to resolve issues"
  exit 1
fi

echo "✅ Audit passed - pushing to remote"
exit 0`
};

export const packageJsonScripts = {
  // Additional scripts for design system enforcement
  "validate:styles": "node -e \"require('./src/utils/style-migrator.js').validateStyleCompliance()\"",
  "migrate:styles": "node -e \"require('./src/utils/style-migrator.js').runStyleMigration()\"",
  "audit:styles": "stylelint 'src/**/*.{css,scss}' --config stylelint.config.js",
  "audit:design-system": "node -e \"require('./src/utils/comprehensive-audit.js').auditDesignSystemCompliance()\"",
  "audit:full": "node src/utils/audit-cli.js full",
  "audit:quick": "node src/utils/audit-cli.js quick",
  "audit:pre-commit": "node src/utils/audit-cli.js pre-commit",
  "setup:hooks": "node -e \"require('./src/utils/enforcement-tools.js').setupGitHooks()\"",
  "type-check": "tsc --noEmit",
  "lint:design-system": "eslint src --ext .ts,.tsx --config .eslintrc.design-system.js"
};

// Git hooks installation utility
export const setupGitHooks = async () => {
  const fs = require('fs');
  const path = require('path');
  
  const gitHooksDir = '.git/hooks';
  
  // Ensure hooks directory exists
  if (!fs.existsSync(gitHooksDir)) {
    console.log('❌ .git/hooks directory not found. Run from git repository root.');
    return;
  }
  
  // Install pre-commit hook
  const preCommitPath = path.join(gitHooksDir, 'pre-commit');
  fs.writeFileSync(preCommitPath, preCommitHooks['pre-commit']);
  fs.chmodSync(preCommitPath, '755');
  
  // Install commit-msg hook
  const commitMsgPath = path.join(gitHooksDir, 'commit-msg');
  fs.writeFileSync(commitMsgPath, preCommitHooks['commit-msg']);
  fs.chmodSync(commitMsgPath, '755');
  
  // Install pre-push hook
  const prePushPath = path.join(gitHooksDir, 'pre-push');
  fs.writeFileSync(prePushPath, preCommitHooks['pre-push']);
  fs.chmodSync(prePushPath, '755');
  
  console.log('✅ Git hooks installed successfully!');
  console.log('');
  console.log('Installed hooks:');
  console.log('  - pre-commit: Design system compliance checks');
  console.log('  - commit-msg: Conventional commit format validation');
  console.log('  - pre-push: Full codebase audit');
  console.log('');
  console.log('To test hooks manually:');
  console.log('  npm run audit:pre-commit');
  console.log('  npm run validate:styles');
};

// ESLint configuration for design system
export const eslintConfigDesignSystem = {
  extends: [
    '@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended'
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'react', 'react-hooks', 'import'],
  rules: {
    ...eslintDesignSystemRules.rules,
    
    // Additional design system specific rules
    'react/prop-types': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    
    // Enforce centralized imports
    'import/no-relative-parent-imports': 'error',
    'import/order': [
      'error',
      {
        groups: [
          'builtin',
          'external', 
          'internal',
          'parent',
          'sibling',
          'index'
        ],
        'newlines-between': 'always',
        alphabetize: { order: 'asc' }
      }
    ]
  },
  settings: {
    react: {
      version: 'detect'
    }
  }
};

// CI/CD integration for automated enforcement
export const githubActionsWorkflow = `
name: Design System Enforcement

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  design-system-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Type check
      run: npm run type-check
    
    - name: Lint design system
      run: npm run lint:design-system
    
    - name: Validate styles
      run: npm run validate:styles
    
    - name: Run comprehensive audit
      run: npm run audit:full
    
    - name: Check for hardcoded colors
      run: |
        if grep -r "bg-blue-[0-9]\\|bg-red-[0-9]\\|bg-green-[0-9]\\|bg-yellow-[0-9]\\|bg-purple-[0-9]\\|#[0-9a-fA-F]\\{6\\}" src/; then
          echo "❌ Hardcoded colors found! Use design system tokens."
          exit 1
        fi
    
    - name: Generate audit report
      run: npm run audit:full -- --output audit-report.json
    
    - name: Upload audit report
      uses: actions/upload-artifact@v3
      with:
        name: audit-report
        path: audit-report.json
        
    - name: Comment PR with audit results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = JSON.parse(fs.readFileSync('audit-report.json', 'utf8'));
          
          const comment = \`## 🔍 Design System Audit Results
          
          **Health Score:** \${report.healthScore}/100
          
          **Issues Found:** \${report.issues.length}
          
          \${report.issues.length > 0 ? '❌ Issues found that need attention' : '✅ All checks passed!'}
          
          [View full audit report](https://github.com/\${context.repo.owner}/\${context.repo.repo}/actions/runs/\${context.runId})
          \`;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
`;

export default {
  eslintDesignSystemRules,
  stylelintConfig,
  preCommitHooks,
  packageJsonScripts,
  setupGitHooks,
  eslintConfigDesignSystem,
  githubActionsWorkflow
};