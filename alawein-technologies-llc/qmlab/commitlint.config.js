module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // New feature
        'fix',      // Bug fix
        'docs',     // Documentation only
        'style',    // Code style (formatting, missing semi colons, etc)
        'refactor', // Code refactoring
        'perf',     // Performance improvements
        'test',     // Adding or updating tests
        'build',    // Build system or dependencies
        'ci',       // CI configuration
        'chore',    // Other changes that don't modify src or test files
        'revert',   // Revert a previous commit
        'a11y',     // Accessibility improvements
      ],
    ],
    'subject-case': [2, 'always', 'sentence-case'],
  },
};
