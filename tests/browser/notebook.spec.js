const { test, expect } = require('@playwright/test');
const { execFileSync } = require('node:child_process');

test('notebook widget renders escaped diagnostics and sends cooperative stop', async ({ page }) => {
  const payload = JSON.parse(execFileSync('.venv/bin/python', ['-c', `
import json
from pymc_monitor.display import _PROGRESS_ESM
from pymc_monitor.events import ConvergenceUpdate, ParamConvergence
from pymc_monitor.rendering import convergence_html
update = ConvergenceUpdate({'<img src=x onerror=window.pwned=1>': ParamConvergence(None, 12, 10)},
    draws=25, verdict={'decision': 'continue', 'hint': '<script>window.pwned=1</script>'})
print(json.dumps({'esm': _PROGRESS_ESM, 'html': convergence_html(update)}))
  `], { encoding: 'utf8' }));
  await page.setContent('<main id="widget"></main>');
  await page.evaluate(async ({ esm, html }) => {
    const module = await import('data:text/javascript;base64,' + btoa(esm));
    const listeners = new Map();
    const state = { html, stop_enabled: true, stop_url: '', stop_token: '' };
    window.messages = [];
    const model = {
      get: key => state[key],
      on: (event, fn) => listeners.set(event, fn),
      off: (event, fn) => { if (listeners.get(event) === fn) listeners.delete(event); },
      send: message => window.messages.push(message),
    };
    window.hideStop = () => { state.stop_enabled = false; listeners.get('change:stop_enabled')(); };
    window.listenerCount = () => listeners.size;
    window.cleanup = module.default.render({model, el: document.querySelector('#widget')});
  }, payload);
  await expect(page.locator('#widget')).toContainText('<img src=x onerror=window.pwned=1>');
  await expect(page.locator('#widget')).toContainText('unavailable');
  expect(await page.evaluate(() => window.pwned)).toBeUndefined();
  await expect(page.locator('#widget img')).toHaveCount(0);
  await page.getByRole('button', {name: 'Stop sampling'}).click();
  expect(await page.evaluate(() => window.messages)).toEqual([{type: 'stop'}]);
  await expect(page.getByRole('button', {name: 'Stopping...'})).toBeDisabled();
  await page.evaluate(() => window.hideStop());
  await expect(page.locator('button')).toBeHidden();
  await page.evaluate(() => window.cleanup());
  expect(await page.evaluate(() => window.listenerCount())).toBe(0);
});
