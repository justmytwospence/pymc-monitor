const { test, expect } = require('@playwright/test');
const { spawn } = require('node:child_process');

async function fixture(script) {
  const server = spawn('.venv/bin/python', [script], {cwd: process.cwd()});
  const urls = await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.stdout.once('data', chunk => resolve(JSON.parse(chunk.toString().trim())));
    server.once('exit', code => reject(Error(`server exited: ${code}`)));
  });
  return {server, urls};
}

test('authenticated local dashboard redirects, isolates runs and renders untrusted text', async ({ page, request }) => {
  const {server, urls: [url, other]} = await fixture('scripts/dashboard_server.py');
  try {
    const token = new URL(url).pathname.split('/')[2];
    const runId = new URL(url).pathname.split('/')[3];
    const stateUrl = new URL(`/runs/${runId}/state`, url).toString();
    const stopUrl = new URL(`/runs/${runId}/stop`, url).toString();
    expect((await request.get(stateUrl)).status()).toBe(403);
    expect((await request.get(new URL('/assets/dashboard.js', url).toString())).status()).toBe(403);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await expect(page.locator('#detail')).toContainText('working <img');
    await expect(page.locator('#detail')).toContainText('<script>window.pwned=1</script>');
    await expect(page.locator('#detail')).toContainText('ess_bulk: 80; target 100 (fail)');
    await expect(page.locator('figure')).toHaveCount(2);
    await expect(page.locator('.uplot')).toHaveCount(4);
    expect(await page.evaluate(() => window.pwned)).toBeUndefined();
    await expect(page.locator('#detail img')).toHaveCount(0);
    await expect(page.locator('#status')).toContainText('Sampling in progress');
    expect(new URL(page.url()).pathname).toBe(`/r/${runId}/`);
    expect(page.url()).not.toContain(token);
    await page.reload();
    await expect(page.locator('#status')).toContainText('Sampling in progress');
    expect((await request.get(stateUrl, {headers: {Authorization: `Bearer ${token}`}})).status()).toBe(200);
    expect((await request.post(stopUrl, {headers: {Authorization: `Bearer ${token}`, Origin: 'https://evil.test'}})).status()).toBe(403);
    expect((await request.get(`${stateUrl}?token=${token}`)).status()).toBe(403);
    page.on('dialog', dialog => dialog.accept());
    await page.getByRole('button', {name: 'Stop sampling'}).click();
    await expect(page.getByRole('button', {name: 'Stopping…'})).toBeDisabled();
    await expect(page.locator('#detail')).toContainText('Run stopped');
    expect((await request.post(stopUrl, {headers: {Authorization: `Bearer ${token}`}})).status()).toBe(409);
    await page.goto(other);
    await expect(page.locator('#status')).toContainText('Sampling in progress');
    expect(errors).toEqual([]);
  } finally { server.kill(); }
});

test('flat page and read-only page use only embedded assets and retry failed stops', async ({page, request}) => {
  const {server, urls: [dashboard, api]} = await fixture('scripts/dashboard_contract_server.py');
  try {
    page.on('dialog', dialog => dialog.accept());
    await page.goto(dashboard + '/flat');
    await expect(page.locator('#detail')).toContainText('Flat run');
    await page.getByRole('button', {name: 'Stop sampling'}).click();
    await expect(page.locator('#status')).toContainText('Stop failed: HTTP 403');
    await expect(page.getByRole('button', {name: 'Stop sampling'})).toBeEnabled();
    await page.getByRole('button', {name: 'Stop sampling'}).click();
    await expect(page.getByRole('button', {name: 'Stopping…'})).toBeDisabled();
    const log = await (await request.get(api + '/test/log')).json();
    expect(log.options).toBe(0);
    expect(log.posts).toHaveLength(2);
    for (const post of log.posts) {
      const url = new URL(post.url, api);
      expect(url.searchParams.get('model')).toBeNull();
      expect(url.searchParams.get('mode')).toBe('flat');
      expect(url.searchParams.get('token')).toBe('secret?&');
    }
    await page.goto(dashboard + '/nostop');
    await expect(page.locator('#detail')).toContainText('Flat run');
    await expect(page.locator('#stop-all')).toBeDisabled();
    await page.goto(dashboard + '/readonly');
    await expect(page.locator('#overview .model-card')).toHaveCount(2);
    await expect(page.locator('#stop-all')).toBeHidden();
    await page.locator('.model-card').first().click();
    await expect(page.locator('.model-stop')).toHaveCount(0);
  } finally { server.kill(); }
});

test('self-contained page cross-origin manifest, simple stop calls, failures and completion', async ({ page, request }) => {
  const {server, urls: [dashboard, api]} = await fixture('scripts/dashboard_contract_server.py');
  const label = 'model / <img src=x onerror=window.pwned=1> &?';
  try {
    const errors = [], violations = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && /Content Security Policy|Refused to/.test(message.text()))
        violations.push(message.text());
    });
    page.on('dialog', dialog => dialog.accept());
    await page.goto(dashboard);
    await expect(page.locator('.model-card')).toHaveCount(2);
    await expect(page.locator('#overview')).toContainText('Waiting for sampling to start');
    expect(await page.evaluate(() => window.pwned)).toBeUndefined();
    const first = page.locator('.model-card').first();
    await first.click();
    expect(page.url()).toContain(`#model=${encodeURIComponent(label)}`);
    await expect(page.locator('#detail')).toContainText('Working');
    await expect(page.locator('figure')).toHaveCount(1);
    await expect(page.locator('.uplot')).toHaveCount(2);
    await page.getByRole('button', {name: 'Stop', exact: true}).click();
    await expect(page.locator('#status')).toContainText('Stop failed: HTTP 200');
    await expect(page.getByRole('button', {name: 'Stop', exact: true})).toBeEnabled();
    await page.getByRole('button', {name: 'Stop', exact: true}).click();
    await expect(page.locator('#status')).toContainText('Stop failed: HTTP 500');
    await page.getByRole('button', {name: 'Stop', exact: true}).click();
    await expect(page.getByRole('button', {name: 'Stopping…'})).toBeDisabled();
    await page.locator('#back').click();
    await expect(page.locator('#overview')).toContainText('Cached result');
    await expect(page.locator('#stop-all')).toBeDisabled();
    const log = await (await request.get(api + '/test/log')).json();
    expect(log.options).toBe(0);
    expect(log.reads).toBeGreaterThanOrEqual(8);
    await page.waitForTimeout(1200);
    expect((await (await request.get(api + '/test/log')).json()).reads).toBe(log.reads);
    expect(log.posts).toHaveLength(3);
    for (const entry of log.posts) {
      const url = new URL(entry.url, api);
      expect(url.searchParams.getAll('token')).toEqual(['secret?&']);
      expect(url.searchParams.get('existing')).toBe('1');
      expect(url.searchParams.get('model')).toBe(label);
      expect(entry.authorization).toBeNull();
      expect(entry.contentType).toBeNull();
      expect(entry.body).toBe('');
    }
    expect(errors).toEqual([]);
    expect(violations).toEqual([]);
  } finally { server.kill(); }
});
