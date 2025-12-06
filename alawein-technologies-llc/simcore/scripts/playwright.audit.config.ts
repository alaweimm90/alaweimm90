import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './scripts/tests',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on',
    screenshot: 'on',
    video: 'retain-on-failure'
  },
  reporter: [['list'], ['html', { outputFolder: '.audit/playwright' }]],
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
});

