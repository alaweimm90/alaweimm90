import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e/specs',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  globalSetup: './tests/e2e/global-setup.ts',
  reporter: [
    ['html'],
    ['list'],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    storageState: 'tests/e2e/storage/user.json',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], storageState: 'tests/e2e/storage/user.json' },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'], storageState: 'tests/e2e/storage/user.json' },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'], storageState: 'tests/e2e/storage/user.json' },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'], storageState: 'tests/e2e/storage/user.json' },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'], storageState: 'tests/e2e/storage/user.json' },
    },
    {
      name: 'admin (chromium)',
      use: { ...devices['Desktop Chrome'], storageState: 'tests/e2e/storage/admin.json' },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
