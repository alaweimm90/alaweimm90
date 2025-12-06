import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
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
    ['junit', { outputFile: 'test-results/junit.xml' }]
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.BASE_URL || 'http://localhost:4173',
    
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    
    /* Screenshot on failure */
    screenshot: 'only-on-failure',
    
    /* Video recording */
    video: 'retain-on-failure',
    
    /* Global test timeout */
    actionTimeout: 10000,
    navigationTimeout: 30000,
  },

  /* Configure projects for major browsers */
  projects: [
    /* Desktop browsers */
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Enable accessibility tree snapshots
        contextOptions: {
          reducedMotion: 'reduce', // Test with reduced motion
        },
      },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* Mobile browsers */
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },

    /* Accessibility testing */
    {
      name: 'accessibility',
      use: {
        ...devices['Desktop Chrome'],
        contextOptions: {
          reducedMotion: 'reduce',
          forcedColors: 'active', // Test high contrast mode
        },
      },
      testMatch: /accessibility\.spec\.ts/,
    },

    /* Performance testing */
    {
      name: 'performance',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--enable-precise-memory-info'],
        },
      },
      testMatch: /performance\.spec\.ts/,
    },

    /* Visual regression testing */
    {
      name: 'visual-regression',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
      },
      testMatch: /visual\.spec\.ts/,
    },
  ],

  /* Global setup and teardown */
  globalSetup: require.resolve('./e2e/global-setup.ts'),
  globalTeardown: require.resolve('./e2e/global-teardown.ts'),

  /* Run your local dev server before starting the tests */
  webServer: {
    command: 'npm run preview',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000, // 2 minutes
  },

  /* Test configuration */
  timeout: 30000,
  expect: {
    timeout: 10000,
    // Visual comparison threshold
    threshold: 0.2,
  },

  /* Output directories */
  outputDir: 'test-results/',

  /* Test metadata */
  metadata: {
    'test-environment': process.env.NODE_ENV || 'test',
    'app-version': process.env.npm_package_version || '1.0.0',
  },
});