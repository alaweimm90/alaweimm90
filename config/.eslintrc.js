module.exports = {
  root: true,
  extends: ['../.metaHub/dev-tools/linters/.eslintrc.js'],
  settings: {
    'import/resolver': {
      typescript: { project: './tsconfig.json' },
      node: { extensions: ['.js', '.jsx', '.ts', '.tsx'] },
    },
  },
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
