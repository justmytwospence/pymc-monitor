module.exports = {
  testDir: './tests/browser',
  timeout: 30000,
  workers: 1,
  use: { browserName: 'chromium', headless: true },
};
