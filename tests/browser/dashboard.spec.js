const { test, expect } = require('@playwright/test');
const { spawn } = require('node:child_process');

test('authenticated local dashboard renders safely and stops', async ({ page, request }) => {
  const server = spawn('.venv/bin/python', ['scripts/dashboard_server.py'], {cwd: process.cwd()});
  try {
    const url = await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.stdout.once('data', chunk => resolve(chunk.toString().trim()));
      server.once('exit', code => reject(Error(`server exited: ${code}`)));
    });
    expect(new URL(url).hostname).toBe('127.0.0.1');
    const token = new URL(url).hash.slice(1);
    expect((await request.get(new URL('/state', url).toString())).status()).toBe(403);
    await page.goto(url);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await expect(page.locator('#phases')).toContainText('working <img');
    await expect(page.locator('#convergence')).toContainText('<script>window.pwned=1</script>');
    await expect(page.locator('#checks')).toContainText('ess_bulk: 80; target 100 (fail)');
    await expect(page.locator('#traces svg polyline')).toHaveCount(2);
    expect(await page.evaluate(() => window.pwned)).toBeUndefined();
    await expect(page.locator('#phases img')).toHaveCount(0);
    await expect(page.locator('#status')).toContainText('Sampling in progress');
    expect(await page.evaluate(() => location.hash)).toBe('');
    const response = await request.get(new URL('/state', url).toString(), {
      headers: {Authorization: `Bearer ${token}`}
    });
    expect(response.status()).toBe(200);
    expect((await request.post(new URL('/stop', url).toString(), {
      headers: {Authorization: `Bearer ${token}`, Origin: 'https://evil.test'}
    })).status()).toBe(403);
    expect((await request.get(new URL(`/state?token=${token}`, url).toString())).status()).toBe(403);
    await page.getByRole('button', {name: 'Stop sampling'}).click();
    await expect(page.getByRole('button', {name: 'Stop sampling'})).toBeDisabled();
    await expect(page.locator('#status')).toContainText('Run complete');
    expect(errors).toEqual([]);
    await page.reload();
    await expect(page.locator('#status')).toContainText('Missing dashboard token');
  } finally { server.kill(); }
});
