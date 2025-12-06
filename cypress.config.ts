import { defineConfig } from 'cypress';

export default defineConfig({
  // Project configuration
  projectId: 'github-repository-ecosystem',

  // Configuration for different environments
  e2e: {
    // Base URL for tests
    baseUrl: 'http://localhost:3000',

    // Test files pattern
    specPattern: [
      'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
      'tests/e2e/**/*.cy.{js,jsx,ts,tsx}'
    ],

    // Support files
    supportFile: 'cypress/support/e2e.ts',

    // Fixtures directory
    fixturesFolder: 'cypress/fixtures',

    // Screenshots
    screenshotsFolder: 'cypress/screenshots',

    // Videos
    videosFolder: 'cypress/videos',

    // Video recording
    video: true,
    videoCompression: 32,

    // Screenshot on failure
    screenshotOnRunFailure: true,

    // Viewport configuration
    viewportWidth: 1280,
    viewportHeight: 720,

    // Default command timeout
    defaultCommandTimeout: 10000,

    // Request timeout
    requestTimeout: 10000,

    // Response timeout
    responseTimeout: 10000,

    // Task timeout
    taskTimeout: 10000,

    // Execution speed
    execTimeout: 10000,

    // Page load timeout
    pageLoadTimeout: 30000,

    // Test configuration
    setupNodeEvents(on, config) {
      // Plugins configuration
      on('task', {
        // Custom tasks for testing
        log(message) {
          console.log(message);
          return null;
        },

        // Database operations
        queryDatabase(query) {
          // Database query implementation
          return null;
        },

        // File system operations
        readFile(filename) {
          // File reading implementation
          return null;
        }
      });

      return config;
    },

    // Environment variables
    env: {
      // API endpoints
      API_BASE_URL: 'http://localhost:3001/api',

      // Authentication
      AUTH_USERNAME: 'testuser',
      AUTH_PASSWORD: 'testpass',

      // Database
      DATABASE_URL: 'postgresql://test:test@localhost:5432/test',

      // Feature flags
      FEATURE_FLAG_NEW_UI: true,
      FEATURE_FLAG_EXPERIMENTAL: false,

      // Testing configuration
      TEST_ENVIRONMENT: 'development',
      MOCK_EXTERNAL_APIS: true,

      // Category-specific configuration
      LLC_ENVIRONMENT: 'production',
      RESEARCH_ENVIRONMENT: 'development',
      PERSONAL_ENVIRONMENT: 'staging'
    },

    // Browser configuration
    chromeWebSecurity: false,

    // retries configuration
    retryOnNetworkFailure: true,
    retries: {
      runMode: 2,
      openMode: 0
    },

    // Experimental features
    experimentalStudio: true,
    experimentalWebKitSupport: true
  },

  // Component testing configuration
  component: {
    // Test files pattern
    specPattern: [
      'cypress/component/**/*.cy.{js,jsx,ts,tsx}',
      'src/**/*.cy.{js,jsx,ts,tsx}',
      'tests/component/**/*.cy.{js,jsx,ts,tsx}'
    ],

    // Support files
    supportFile: 'cypress/support/component.ts',

    // Fixtures directory
    fixturesFolder: 'cypress/fixtures',

    // Index HTML file
    indexHtmlFile: 'cypress/support/component-index.html',

    // Development server
    devServer: {
      framework: 'react',
      bundler: 'webpack',
      webpackConfig: {
        mode: 'development',
        devtool: false,
        resolve: {
          extensions: ['.ts', '.tsx', '.js', '.jsx', '.json']
        },
        module: {
          rules: [
            {
              test: /\.(ts|tsx)$/,
              use: 'ts-loader',
              exclude: /node_modules/
            },
            {
              test: /\.(js|jsx)$/,
              use: 'babel-loader',
              exclude: /node_modules/
            },
            {
              test: /\.css$/,
              use: ['style-loader', 'css-loader']
            }
          ]
        }
      }
    },

    // Environment variables
    env: {
      // Component testing specific
      COMPONENT_TESTING: true,
      MOCK_APIS: true,
      FAST_TESTS: true
    }
  },

  // Configuration for different categories
  // LLC Projects - Production-level testing
  llc: {
    e2e: {
      baseUrl: 'http://localhost:3001',
      specPattern: 'tests/e2e/llc/**/*.cy.{js,jsx,ts,tsx}',
      env: {
        CATEGORY: 'llc',
        SECURITY_LEVEL: 'maximum',
        TESTING_LEVEL: 'production'
      },
      retries: {
        runMode: 3,
        openMode: 1
      }
    }
  },

  // Research Projects - Development testing
  research: {
    e2e: {
      baseUrl: 'http://localhost:3004',
      specPattern: 'tests/e2e/research/**/*.cy.{js,jsx,ts,tsx}',
      env: {
        CATEGORY: 'research',
        SECURITY_LEVEL: 'basic',
        TESTING_LEVEL: 'development'
      },
      retries: {
        runMode: 1,
        openMode: 0
      }
    }
  },

  // Personal Platforms - Flexible testing
  personal: {
    e2e: {
      baseUrl: 'http://localhost:3000',
      specPattern: 'tests/e2e/personal/**/*.cy.{js,jsx,ts,tsx}',
      env: {
        CATEGORY: 'personal',
        SECURITY_LEVEL: 'basic',
        TESTING_LEVEL: 'flexible'
      },
      retries: {
        runMode: 1,
        openMode: 0
      }
    }
  },

  // Global configuration
  reporter: 'cypress-multi-reporters',
  reporterOptions: {
    configFile: 'cypress-reporter-config.json'
  },

  // Parallel execution
  parallel: true,
  blockHosts: ['*.google-analytics.com'],

  // Performance configuration
  numTestsKeptInMemory: 50,

  // Trash assets before runs
  trashAssetsBeforeRuns: true,

  // Watch for file changes
  watchForFileChanges: false,

  // Default browser
  browser: 'chrome',

  // Browsers configuration
  browsers: ['chrome', 'firefox', 'edge'],

  // User agent
  userAgent: null,

  // Include shadow DOM
  includeShadowDom: true,

  // NSFW
  nsfw: false,

  // Block URLs
  blockUrls: null,

  // Slow test threshold
  slowTestThreshold: 10000,

  // Animation distance threshold
  animationDistanceThreshold: 5,

  // File server folder
  fileServerFolder: '.',

  // Exclude spec pattern
  excludeSpecPattern: [],

  // Integration folder
  integrationFolder: 'cypress/e2e',

  // Downloads folder
  downloadsFolder: 'cypress/downloads',

  // Modify obstructive code
  modifyObstructiveCode: false
});
