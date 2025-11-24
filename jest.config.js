module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js'],
  testPathIgnorePatterns: [
    '<rootDir>/.config/',
    '<rootDir>/.cache/',
    '<rootDir>/templates/',
    '<rootDir>/.tools/automation/',
  ],
  modulePathIgnorePatterns: ['<rootDir>/.config/', '<rootDir>/.cache/'],
  collectCoverage: true,
  collectCoverageFrom: ['scripts/standards-validator.js'],
  restoreMocks: true,
};
