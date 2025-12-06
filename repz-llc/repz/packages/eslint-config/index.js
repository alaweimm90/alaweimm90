/**
 * REPZ Shared ESLint Configuration - Base
 * Enterprise-grade code quality standards
 * Used across all REPZ platform packages
 */

module.exports = {
  extends: [
    'eslint:recommended',
    'prettier'
  ],
  plugins: [
    'prettier'
  ],
  env: {
    browser: true,
    es2022: true,
    node: true
  },
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    // Code Quality
    'no-unused-vars': ['error', { 
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_',
      ignoreRestSiblings: true 
    }],
    'no-console': ['warn', { 
      allow: ['warn', 'error', 'info'] 
    }],
    'no-debugger': 'error',
    'no-alert': 'error',
    
    // Best Practices
    'eqeqeq': ['error', 'always'],
    'curly': ['error', 'all'],
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    'no-return-assign': 'error',
    'no-sequences': 'error',
    'no-throw-literal': 'error',
    'no-unmodified-loop-condition': 'error',
    'no-unused-expressions': 'error',
    'no-useless-call': 'error',
    'no-useless-concat': 'error',
    'no-useless-return': 'error',
    'prefer-promise-reject-errors': 'error',
    'radix': 'error',
    
    // ES6+ Features
    'prefer-const': 'error',
    'prefer-arrow-callback': 'error',
    'prefer-template': 'error',
    'prefer-destructuring': ['error', {
      array: false,
      object: true
    }],
    'prefer-rest-params': 'error',
    'prefer-spread': 'error',
    'no-var': 'error',
    'object-shorthand': 'error',
    'arrow-body-style': ['error', 'as-needed'],
    
    // Import Organization
    'sort-imports': ['error', {
      ignoreCase: true,
      ignoreDeclarationSort: true,
      ignoreMemberSort: false,
      memberSyntaxSortOrder: ['none', 'all', 'multiple', 'single'],
      allowSeparatedGroups: true
    }],
    
    // Formatting (handled by Prettier)
    'prettier/prettier': 'error',
    
    // Performance
    'no-loop-func': 'error',
    'no-new-object': 'error',
    'no-new-wrappers': 'error',
    
    // Security
    'no-new-func': 'error',
    'no-script-url': 'error'
  },
  
  // Override for specific file patterns
  overrides: [
    {
      files: ['**/*.test.{js,ts,tsx}', '**/*.spec.{js,ts,tsx}'],
      env: {
        jest: true,
        'vitest-globals/env': true
      },
      rules: {
        'no-console': 'off'
      }
    },
    {
      files: ['**/scripts/**', '**/tools/**'],
      rules: {
        'no-console': 'off'
      }
    }
  ]
};