import { chromium } from 'playwright';
import { AxeBuilder } from '@axe-core/playwright';
import fs from 'fs';
import path from 'path';

const OUT = '.audit/axe';
fs.mkdirSync(OUT, { recursive: true });

const urls = [
  '/',
  '/modules/graphene-band-structure',
  '/modules/ising-model',
  '/documentation',
  '/simulation-dashboard'
];

const run = async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext();
  const page = await ctx.newPage();

  const results = [];
  for (const u of urls) {
    const url = `http://127.0.0.1:4173${u}`;
    await page.goto(url, { waitUntil: 'networkidle' });
    const axe = await new AxeBuilder({ page }).withTags(['wcag2a','wcag2aa']).analyze();
    results.push({ url, violations: axe.violations });
  }

  fs.writeFileSync(path.join(OUT, 'axe.json'), JSON.stringify(results, null, 2));
  console.log('axe results written to', path.join(OUT, 'axe.json'));
  await browser.close();
};

run();

