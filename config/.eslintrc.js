module.exports = {
  root: true,
  extends: ['../.metaHub/dev-tools/linters/.eslintrc.js'],
  env: { jest: true, node: true },
  globals: { test: 'readonly', expect: 'readonly', describe: 'readonly' },
  settings: {
    'import/resolver': {
      typescript: { project: './tsconfig.json' },
      node: { extensions: ['.js', '.jsx', '.ts', '.tsx'] },
    },
  },
  overrides: [
    {
      files: ['tests/**/*.test.js', '**/__tests__/**/*.js'],
      env: { jest: true },
    },
  ],
  rules: {
    'import/extensions': [
      'error',
      'ignorePackages',
      { js: 'never', jsx: 'never', ts: 'never', tsx: 'never' },
    ],
    'no-else-return': 'off',
    'jsdoc/require-jsdoc': 'warn',
    'jsdoc/require-param': 'warn',
    'jsdoc/require-returns': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
  },
  ignorePatterns: [
    'node_modules',
    'dist',
    'build',
    'coverage',
    '.next',
    '.archive',
    '.archives',
    '.automation',
    '.config/archives/**',
    // Ignore workspace packages - they have their own configs
    '.organizations/**',
    'alaweimm90/**',
    '.metaHub/**',
  ],
};
