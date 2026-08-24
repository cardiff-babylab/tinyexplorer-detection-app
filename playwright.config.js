// @ts-check
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 120 * 1000, // 2 minutes for test timeout
  expect: {
    timeout: 30000 // 30 seconds for assertions
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    // Use headed mode when DISPLAY is set (remote X11 forwarding)
    headless: !process.env.DISPLAY,
  },

  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Additional Chrome args for X11 forwarding compatibility
        launchOptions: {
          ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
            ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
            : {}),
          args: process.env.DISPLAY ? [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu-sandbox',
            '--remote-debugging-port=9222'
          ] : []
        }
      },
    },
  ],

  webServer: {
    command: 'npm run react-start',
    port: 3000,
    reuseExistingServer: !process.env.CI,
    env: {
      NODE_OPTIONS: '--openssl-legacy-provider'
    }
  },
});
