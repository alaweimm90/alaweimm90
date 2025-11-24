module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js', '**/apps/**/*.test.js'],
  testPathIgnorePatterns: [
    '<rootDir>/.config/',
    '<rootDir>/.cache/',
    '<rootDir>/templates/',
    '<rootDir>/.tools/automation/',
  ],
  modulePathIgnorePatterns: ['<rootDir>/.config/', '<rootDir>/.cache/'],
  collectCoverage: true,
  collectCoverageFrom: [
    'scripts/standards-validator.js',
    'scripts/standards-lib.js',
    'apps/ai-agent-demo/**/*.js',
  ],
  coveragePathIgnorePatterns: [
    '<rootDir>/alaweimm90/',
    '<rootDir>/packages/',
    '<rootDir>/.organizations/',
    '<rootDir>/.config/',
    '<rootDir>/.tools/automation/',
    '<rootDir>/apps/ai-agent-demo/server.js',
    '<rootDir>/scripts/standards-lib.js',
    '<rootDir>/scripts/standards-validator.js',
  ],
  coverageThreshold: {
    global: {
      branches: 23,
      functions: 40,
      lines: 39,
      statements: 35,
    },
  },
  coverageDirectory: 'coverage',
  restoreMocks: true,
};
