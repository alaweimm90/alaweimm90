#!/usr/bin/env tsx
/**
 * Accessibility audit using axe-core concepts
 * Usage: npm run a11y:audit [url]
 */
import { execSync } from 'child_process';

const DEFAULT_URLS = [
  'http://localhost:3000',  // REPZ
  'http://localhost:3001',  // LiveItIconic
];

interface A11yIssue {
  impact: 'critical' | 'serious' | 'moderate' | 'minor';
  description: string;
  selector?: string;
}

function runLighthouseA11y(url: string): void {
  console.log(`\n🔍 Auditing: ${url}`);
  try {
    // Use lighthouse CLI if available
    execSync(`npx lighthouse ${url} --only-categories=accessibility --output=json --quiet`, {
      stdio: 'pipe'
    });
    console.log('  ✅ Lighthouse audit complete');
  } catch {
    console.log('  ⚠️  Lighthouse not available or URL not reachable');
    console.log('  💡 Install: npm i -g lighthouse');
  }
}

function printChecklist(): void {
  console.log('\n♿ WCAG 2.1 AA Checklist\n');
  console.log('='.repeat(50));

  const checks = [
    '[ ] All images have alt text',
    '[ ] Color contrast ratio ≥ 4.5:1 for text',
    '[ ] All form inputs have labels',
    '[ ] Keyboard navigation works (Tab, Enter, Escape)',
    '[ ] Focus indicators visible',
    '[ ] Skip-to-content link present',
    '[ ] Headings in logical order (h1 → h2 → h3)',
    '[ ] ARIA labels on interactive elements',
    '[ ] Error messages linked to inputs',
    '[ ] Touch targets ≥ 44x44px on mobile',
  ];

  checks.forEach(c => console.log(`  ${c}`));
  console.log('='.repeat(50));
}

const [,, cmd, url] = process.argv;
if (cmd === 'audit' && url) {
  runLighthouseA11y(url);
} else if (cmd === 'checklist' || !cmd) {
  printChecklist();
  console.log('\n💡 Run with URL: npm run a11y:audit http://localhost:3000');
} else {
  console.log('Usage: npm run a11y:audit <url> OR npm run a11y:check');
}
