import { defineConfig, devices } from '@playwright/test';

/**
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './tests/e2e',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/results.xml' }],
    ['github']
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.PLAYWRIGHT_TEST_BASE_URL || 'http://localhost:8081',

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    
    /* Take screenshot on failure */
    screenshot: 'only-on-failure',
    
    /* Record video on failure */
    video: 'retain-on-failure',
    
    /* Global timeout for each action */
    actionTimeout: 30000,
    
    /* Global timeout for navigation */
    navigationTimeout: 30000,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'setup',
      testMatch: /.*\.setup\.ts/,
    },
    
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Use prepared auth state
        storageState: 'tests/fixtures/auth/client-auth.json',
      },
      dependencies: ['setup'],
    },

    {
      name: 'firefox',
      use: { 
        ...devices['Desktop Firefox'],
        storageState: 'tests/fixtures/auth/client-auth.json',
      },
      dependencies: ['setup'],
    },

    {
      name: 'webkit',
      use: { 
        ...devices['Desktop Safari'],
        storageState: 'tests/fixtures/auth/client-auth.json',
      },
      dependencies: ['setup'],
    },

    /* Test against mobile viewports. */
    {
      name: 'Mobile Chrome',
      use: { 
        ...devices['Pixel 5'],
        storageState: 'tests/fixtures/auth/client-auth.json',
      },
      dependencies: ['setup'],
    },
    {
      name: 'Mobile Safari',
      use: { 
        ...devices['iPhone 12'],
        storageState: 'tests/fixtures/auth/client-auth.json',
      },
      dependencies: ['setup'],
    },
  ],

  /* Run your local dev server before starting the tests */
  webServer: {
    command: 'npm run dev',
    url: process.env.PLAYWRIGHT_TEST_BASE_URL || 'http://localhost:8081',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },

  /* Global setup and teardown */
  globalSetup: './tests/fixtures/global-setup.ts',
  globalTeardown: './tests/fixtures/global-teardown.ts',

  /* Test timeout */
  timeout: 60 * 1000,

  /* Expect timeout */
  expect: {
    timeout: 10 * 1000,
  },

  /* Output directories */
  outputDir: 'test-results/',
  
  /* Test metadata */
  metadata: {
    'test-environment': process.env.NODE_ENV || 'test',
    'base-url': process.env.PLAYWRIGHT_TEST_BASE_URL || 'http://localhost:8081',
    'browser-versions': 'latest',
  },

  /* Test tags for filtering */
  grep: process.env.PLAYWRIGHT_GREP,
  grepInvert: process.env.PLAYWRIGHT_GREP_INVERT,
});