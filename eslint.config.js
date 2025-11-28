import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.node,
      },
    },
  },
  // Strict rules for tools and tests
  {
    files: ['tools/**/*.ts', 'tests/**/*.ts'],
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-explicit-any': 'warn',
      'no-console': 'off', // CLI tools need console
      'prefer-const': 'error',
      'no-var': 'error',
    },
  },
  // Relaxed rules for config files
  {
    files: ['*.config.js', '*.config.ts'],
    rules: {
      '@typescript-eslint/no-require-imports': 'off',
    },
  },
  // Ignore patterns
  {
    ignores: ['node_modules/**', 'dist/**', '.metaHub/**', 'organizations/**', 'coverage/**'],
  }
);
