/** @type {import('@playwright/test').PlaywrightTestConfig} */
module.exports = {
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on',
    screenshot: 'on',
    video: 'retain-on-failure'
  },
  reporter: [['list'], ['html', { outputFolder: '.audit/playwright' }]]
};

