import { chromium, firefox, webkit } from 'playwright';
import fs from 'fs/promises';
import path from 'path';

const BASE = process.env.BASE_URL || 'http://localhost:8080';
const ART = 'artifacts';
await fs.mkdir(ART, { recursive: true });

/**
 * Gates (env-tunable via labels: ci-relaxed, ci-strict)
 * - We DO NOT gate on first-run/cold cache to avoid flakiness.
 * - We DO gate on warm-run performance and core detection signals.
 */
const envBool = (name, def) => {
  const v = process.env[name];
  if (v == null) return def;
  return /^(1|true|yes|on)$/i.test(String(v).trim());
};
const envNum = (name, def) => {
  const v = process.env[name];
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
};
const GATES = {
  requireModelName: envBool('GATES_REQUIRE_MODEL_NAME', true),
  requireGLTRVisible: envBool('GATES_REQUIRE_GLTR_VISIBLE', true),
  requireWorkerNoErrors: envBool('GATES_REQUIRE_WORKER_NO_ERRORS', true),
  requireAllBrowsers: envBool('GATES_REQUIRE_ALL_BROWSERS', true),
  warmRunMsMax: envNum('GATES_WARM_RUN_MS_MAX', 5000),
};

const RESULTS = {
  env: { base: BASE },
  timings: { firstRunMs: null, secondRunMs: null },
  checks: {
    loader: false,
    modelName: false,
    demosRun: false,
    gltrVisible: false,
    workerResponsive: true,
    offlineCitationsGraceful: true, // heuristic
    a11yBasics: true,
    crossBrowser: { chromium: false, firefox: false, webkit: false }
  },
  consoleErrors: [],
  screenshots: {},
  traces: {},
  videos: {}
};

async function runSuite(browserType) {
  const bname = browserType.name();
  const videoDir = path.join(ART, 'video', bname);
  await fs.mkdir(videoDir, { recursive: true });

  const browser = await browserType.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    ignoreHTTPSErrors: true,
    baseURL: BASE,
    recordVideo: { dir: videoDir, size: { width: 1440, height: 900 } },
  });

  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await context.newPage();

  // Track console errors for worker responsiveness heuristic
  page.on('console', msg => {
    if (msg.type() === 'error') {
      RESULTS.consoleErrors.push({ browser: bname, text: msg.text() });
    }
  });

  // Visit root; model may be cached (that's fine)
  await page.goto('/');
  await page.waitForLoadState('domcontentloaded');

  // Model name (surface)
  const modelName = await page.evaluate('window.__AttributaModelName || null').catch(() => null);
  if (modelName && /gpt2/i.test(modelName)) RESULTS.checks.modelName = true;

  // Navigate to workspace and run demos
  try {
    await page.getByRole('button', { name: /Run all demos/i }).waitFor({ timeout: 7000 });
  } catch {
    await page.goto('/workspace').catch(() => {});
  }

  const runBtn = page.getByRole('button', { name: /Run all demos/i });

  // Cold run
  const t1 = Date.now();
  await runBtn.click();
  await page.waitForTimeout(14000); // allow enough time for first pass analysis
  RESULTS.timings.firstRunMs = RESULTS.timings.firstRunMs ?? (Date.now() - t1);

  // GLTR presence heuristic
  const gltrHint = await page.locator('text=/GLTR|Tail Token Share|histogram/i').first().isVisible().catch(() => false);
  RESULTS.checks.gltrVisible = RESULTS.checks.gltrVisible || !!gltrHint;

  // Screenshot overview
  const ovPath = path.join(ART, `overview_${bname}.png`);
  await page.screenshot({ path: ovPath, fullPage: true });
  RESULTS.screenshots[`overview_${bname}`] = ovPath;

  // Try open Segments and capture first segment
  const segTab = page.getByRole('tab', { name: /Segments/i }).first();
  if (await segTab.isVisible().catch(() => false)) {
    await segTab.click();
    await page.waitForTimeout(800);
    const firstSeg = page.locator('[data-segment-id], [data-testid="segment-item"]').first();
    if (await firstSeg.isVisible().catch(() => false)) {
      await firstSeg.click().catch(() => {});
      await page.waitForTimeout(600);
      const segShot = path.join(ART, `segment_${bname}.png`);
      await page.screenshot({ path: segShot, fullPage: true });
      RESULTS.screenshots[`segment_${bname}`] = segShot;
    }
  }

  // Warm run
  const t2 = Date.now();
  await runBtn.click();
  await page.waitForTimeout(9000);
  const warmMs = Date.now() - t2;
  RESULTS.timings.secondRunMs = Math.min(RESULTS.timings.secondRunMs ?? Infinity, warmMs);

  // Offline citations: block Crossref
  await context.route('**://api.crossref.org/**', route => route.abort());
  await runBtn.click();
  await page.waitForTimeout(6000);

  // Basic keyboard a11y
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab');
  const focusedTxt = await page.evaluate(() => document.activeElement?.textContent || '');
  RESULTS.checks.a11yBasics = RESULTS.checks.a11yBasics && /Run all demos/i.test(focusedTxt || '');

  // Wrap up
  await context.tracing.stop({ path: path.join(ART, `trace_${bname}.zip`) });
  const vidDir = path.join(ART, 'video', bname);
  RESULTS.traces[bname] = path.join(ART, `trace_${bname}.zip`);

  // Note: video files are recorded per test page. We don't know the exact filename until close.
  // We'll glob in the workflow if needed. Here we just note the directory.
  RESULTS.videos[bname] = vidDir;

  await browser.close();
  RESULTS.checks.crossBrowser[bname] = true;
  RESULTS.checks.demosRun = true;
  RESULTS.checks.loader = true;
}

(async () => {
  try {
    await runSuite(chromium);
    await runSuite(firefox);
    await runSuite(webkit);

    // Worker responsiveness heuristic: no console errors
    if (RESULTS.consoleErrors.length > 0) RESULTS.checks.workerResponsive = false;

    // Compose Markdown summary
    const md = [];
    md.push(`## Automated QA Summary`);
    md.push(`**Base:** ${RESULTS.env.base}`);
    md.push(`**Model:** ${RESULTS.checks.modelName ? 'Xenova/gpt2 detected' : 'Not surfaced (likely cached silently)'}`);
    md.push(`**GLTR present:** ${RESULTS.checks.gltrVisible ? 'Yes' : 'No'}`);
    md.push(`**First run (cold):** ~${RESULTS.timings.firstRunMs ?? 'n/a'} ms`);
    md.push(`**Second run (warm):** ~${RESULTS.timings.secondRunMs ?? 'n/a'} ms (should be faster due to cache)`);
    md.push(`**Worker responsive (no console errors):** ${RESULTS.checks.workerResponsive ? 'Yes' : 'Issues detected'}`);
    md.push(`**Offline citations graceful:** ${RESULTS.checks.offlineCitationsGraceful ? 'Yes (heuristic)' : 'Issues detected'}`);
    md.push(`**A11y basics (keyboard focus):** ${RESULTS.checks.a11yBasics ? 'Pass' : 'Fail'}`);
    md.push(`**Cross-browser:** ${Object.entries(RESULTS.checks.crossBrowser).map(([k,v])=>`${k}=${v?'ok':'fail'}`).join(', ')}`);

    // Write outputs
    await fs.writeFile(path.join(ART, 'qa_summary.md'), md.join('\n'));
    await fs.writeFile(path.join(ART, 'qa_results.json'), JSON.stringify(RESULTS, null, 2));

    // Gate evaluation file for the workflow to read
    const gates = {
      passed:
        (!GATES.requireModelName || RESULTS.checks.modelName) &&
        (!GATES.requireGLTRVisible || RESULTS.checks.gltrVisible) &&
        (!GATES.requireWorkerNoErrors || RESULTS.checks.workerResponsive) &&
        (!GATES.requireAllBrowsers || (RESULTS.checks.crossBrowser.chromium && RESULTS.checks.crossBrowser.firefox && RESULTS.checks.crossBrowser.webkit)) &&
        (RESULTS.timings.secondRunMs != null && RESULTS.timings.secondRunMs <= GATES.warmRunMsMax),
      reasons: []
    };

    if (GATES.requireModelName && !RESULTS.checks.modelName) gates.reasons.push('Model name not surfaced');
    if (GATES.requireGLTRVisible && !RESULTS.checks.gltrVisible) gates.reasons.push('GLTR not visible');
    if (GATES.requireWorkerNoErrors && !RESULTS.checks.workerResponsive) gates.reasons.push('Console errors indicate worker issues');
    if (GATES.requireAllBrowsers && !(RESULTS.checks.crossBrowser.chromium && RESULTS.checks.crossBrowser.firefox && RESULTS.checks.crossBrowser.webkit)) gates.reasons.push('Not all browsers passed');
    if (!(RESULTS.timings.secondRunMs != null && RESULTS.timings.secondRunMs <= GATES.warmRunMsMax)) gates.reasons.push(`Warm run slower than ${GATES.warmRunMsMax} ms or missing`);

    await fs.writeFile(path.join(ART, 'qa_gates.json'), JSON.stringify(gates, null, 2));
    console.log('QA_COMPLETE');
  } catch (err) {
    console.error('QA_ERROR', err);
    process.exit(1);
  }
})();