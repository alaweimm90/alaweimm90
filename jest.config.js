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
  collectCoverage: true,  coverageThreshold: {    global: {      branches: 80,      functions: 80,      lines: 80,      statements: 80    }  },
  collectCoverageFrom: ['scripts/standards-validator.js', 'packages/**/*.js', 'packages/**/*.ts', 'alaweimm90/**/*.js', 'alaweimm90/**/*.ts'],  coverageDirectory: 'coverage',
  restoreMocks: true,
};
