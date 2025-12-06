module.exports = {
  // Test environment
  testEnvironment: 'jsdom',

  // Test file patterns
  testMatch: [
    '**/__tests__/**/*.(ts|tsx|js|jsx)',
    '**/*.(test|spec).(ts|tsx|js|jsx)',
    '**/tests/**/*.(ts|tsx|js|jsx)'
  ],

  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.(ts|tsx|js|jsx)',
    '!src/**/*.d.ts',
    '!src/**/*.stories.(ts|tsx|js|jsx)',
    '!src/**/*.config.(js|ts)',
    '!src/**/index.(ts|tsx|js|jsx)',
    '!src/**/*.mock.(ts|tsx|js|jsx)',
    '!src/**/*.test.(ts|tsx|js|jsx)',
    '!src/**/*.spec.(ts|tsx|js|jsx)',
    '!src/**/node_modules/**',
    '!src/**/vendor/**'
  ],

  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    },
    // Category-specific thresholds
    './src/llcs/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    './src/research/': {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    },
    './src/personal/': {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },

  // Coverage reporting
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json',
    'clover'
  ],

  // Coverage directory
  coverageDirectory: 'coverage',

  // Module file extensions
  moduleFileExtensions: [
    'ts',
    'tsx',
    'js',
    'jsx',
    'json',
    'node'
  ],

  // Transform configuration
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      useESM: true
    }],
    '^.+\\.(js|jsx)$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', { targets: { node: 'current' } }],
        '@babel/preset-react',
        '@babel/preset-typescript'
      ]
    }]
  },

  // Module name mapping
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@llcs/(.*)$': '<rootDir>/src/llcs/$1',
    '^@research/(.*)$': '<rootDir>/src/research/$1',
    '^@personal/(.*)$': '<rootDir>/src/personal/$1',
    '^@shared/(.*)$': '<rootDir>/src/shared/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(gif|ttf|eot|svg|png|jpg|jpeg)$': '<rootDir>/tests/mocks/fileMock.js'
  },

  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup/jest.setup.js'
  ],

  // Global setup
  globalSetup: '<rootDir>/tests/setup/global.setup.js',

  // Global teardown
  globalTeardown: '<rootDir>/tests/setup/global.teardown.js',

  // Mock patterns
  modulePathIgnorePatterns: [
    '<rootDir>/dist/',
    '<rootDir>/build/',
    '<rootDir>/node_modules/'
  ],

  // Test timeout
  testTimeout: 10000,

  // Verbose output
  verbose: true,

  // Clear mocks between tests
  clearMocks: true,

  // Restore mocks after each test
  restoreMocks: true,

  // Error handling
  errorOnDeprecated: true,

  // Watch plugins
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname',
    'jest-watch-select-projects',
    'jest-watch-run-all'
  ],

  // Projects configuration for different categories
  projects: [
    {
      displayName: 'LLC Tests',
      testMatch: ['<rootDir>/src/llcs/**/*.test.(ts|tsx|js|jsx)'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup/llc.setup.js']
    },
    {
      displayName: 'Research Tests',
      testMatch: ['<rootDir>/src/research/**/*.test.(ts|tsx|js|jsx)'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup/research.setup.js']
    },
    {
      displayName: 'Personal Tests',
      testMatch: ['<rootDir>/src/personal/**/*.test.(ts|tsx|js|jsx)'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup/personal.setup.js']
    },
    {
      displayName: 'Shared Tests',
      testMatch: ['<rootDir>/src/shared/**/*.test.(ts|tsx|js|jsx)'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup/shared.setup.js']
    },
    {
      displayName: 'Integration Tests',
      testMatch: ['<rootDir>/tests/integration/**/*.test.(ts|tsx|js|jsx)'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup/integration.setup.js']
    }
  ],

  // Reporter configuration
  reporters: [
    'default',
    [
      'jest-html-reporters',
      {
        publicPath: './coverage/html-report',
        filename: 'report.html',
        expand: true,
        hideIcon: false,
        pageTitle: 'Test Report',
        logoImgPath: undefined,
        inlineSource: false
      }
    ],
    [
      'jest-junit',
      {
        outputDirectory: './coverage',
        outputName: 'junit.xml',
        classNameTemplate: '{classname}',
        titleTemplate: '{title}',
        ancestorSeparator: ' › ',
        usePathForSuiteName: true
      }
    ]
  ],

  // Performance testing
  maxWorkers: '50%',

  // Cache configuration
  cache: true,
  cacheDirectory: '<rootDir>/.jest-cache',

  // Test results processor
  testResultsProcessor: undefined,

  // Snapshot configuration
  snapshotSerializers: [],

  // Test runner
  testRunner: 'jest-circus/runner',

  // Transform ignore patterns
  transformIgnorePatterns: [
    'node_modules/(?!(.*\\.mjs$))'
  ],

  // Watchman
  watchman: true,

  // Force exit
  forceExit: false,

  // Detect open handles
  detectOpenHandles: false,

  // Detect leaks
  detectLeaks: false,

  // Notify
  notify: false,
  notifyMode: 'failure-change',

  // Bail mode
  bail: false,

  // Max concurrency
  maxConcurrency: 5,

  // Randomize tests
  randomize: false
};
