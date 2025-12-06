import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/accessibility',
  timeout: 30 * 1000,
  expect: {
    timeout: 5000,
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['junit', { outputFile: 'test-results/accessibility-results.xml' }],
    ['json', { outputFile: 'test-results/accessibility-results.json' }],
  ],
  use: {
    baseURL: 'http://localhost:4175',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium-a11y',
      use: { 
        ...devices['Desktop Chrome'],
        contextOptions: {
          // Enable accessibility features
          reducedMotion: 'no-preference',
          colorScheme: 'light',
        }
      },
    },
    {
      name: 'chromium-reduced-motion',
      use: { 
        ...devices['Desktop Chrome'],
        contextOptions: {
          reducedMotion: 'reduce',
          colorScheme: 'light',
        }
      },
    },
    {
      name: 'chromium-dark-mode',
      use: { 
        ...devices['Desktop Chrome'],
        contextOptions: {
          colorScheme: 'dark',
          reducedMotion: 'no-preference',
        }
      },
    },
    {
      name: 'chromium-high-contrast',
      use: { 
        ...devices['Desktop Chrome'],
        contextOptions: {
          forcedColors: 'active',
          colorScheme: 'light',
        }
      },
    },
    {
      name: 'no-webgl',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--disable-webgl', '--disable-accelerated-2d-canvas']
        }
      },
    },
    {
      name: 'mobile-a11y',
      use: { 
        ...devices['Pixel 5'],
        contextOptions: {
          reducedMotion: 'no-preference',
        }
      },
    },
    {
      name: 'tablet-a11y',
      use: { 
        ...devices['iPad Pro'],
        contextOptions: {
          reducedMotion: 'no-preference',
        }
      },
    },
  ],

  webServer: {
    command: 'npm run preview',
    port: 4175,
    reuseExistingServer: true,
  },
});