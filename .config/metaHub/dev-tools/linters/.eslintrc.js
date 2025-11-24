module.exports = {
  root: true,
  env: { es2022: true, node: true, browser: true, jest: true },
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'import', 'jsdoc'],
  extends: [
    'airbnb-base',
    'plugin:@typescript-eslint/recommended',
    'plugin:jsdoc/recommended',

    'prettier',
  ],
  rules: {
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/explicit-function-return-type': 'warn',
    'import/no-extraneous-dependencies': ['error', { devDependencies: true }],
    'jsdoc/require-jsdoc': ['warn', { publicOnly: true }],
  },
  overrides: [
    {
      files: ['**/*.ts', '**/*.tsx'],
      rules: {
        '@typescript-eslint/no-require-imports': 'error',
      },
    },
    {
      files: ['**/*.js'],
      rules: {
        '@typescript-eslint/no-require-imports': 'off',
        '@typescript-eslint/explicit-function-return-type': 'off',
        '@typescript-eslint/no-unused-vars': 'off',
        'class-methods-use-this': 'off',
        'no-restricted-syntax': 'off',
        'no-use-before-define': 'off',
        'no-await-in-loop': 'off',
        'no-plusplus': 'off',
        'no-continue': 'off',
        'no-return-await': 'off',
        'default-case': 'off',
        'no-param-reassign': 'off',
        'import/order': 'off',
        'max-classes-per-file': 'off',
        'new-cap': 'off',
        'no-case-declarations': 'off',
        'global-require': 'off',
        'import/no-dynamic-require': 'off',
        'no-console': 'warn',
        radix: 'off',
      },
    },
  ],
};
