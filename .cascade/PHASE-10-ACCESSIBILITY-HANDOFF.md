# ♿ PHASE 10: ACCESSIBILITY COMPLIANCE - CASCADE HANDOFF

## Mission
Ensure WCAG 2.1 AA compliance for all web applications. AI-accelerated: 20-30 minutes.

## Context
- Consumer apps: REPZ, LiveItIconic, Attributa
- Need accessibility before Q1 2025 launch
- Focus on automated checks + guidance

---

## Tasks (Execute in Order)

### 1. Add axe-core Testing (10 min)
Create `tools/accessibility/audit.ts`:
```typescript
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

const [,, url] = process.argv;
if (url) {
  runLighthouseA11y(url);
} else {
  printChecklist();
  console.log('\n💡 Run with URL: npm run a11y:audit http://localhost:3000');
}
```

### 2. Create Accessibility Config (5 min)
Create `.config/accessibility/wcag.yaml`:
```yaml
version: "1.0"
accessibility:
  standard: "WCAG 2.1 AA"
  
  requirements:
    perceivable:
      - text_alternatives: "All images have alt text"
      - captions: "Videos have captions"
      - contrast: "4.5:1 minimum for normal text"
      
    operable:
      - keyboard: "All functionality keyboard accessible"
      - timing: "Users can extend time limits"
      - seizures: "No content flashes > 3 times/second"
      
    understandable:
      - readable: "Language defined in HTML"
      - predictable: "Navigation consistent across pages"
      - input_assistance: "Error prevention and recovery"
      
    robust:
      - compatible: "Valid HTML, ARIA correctly used"
  
  testing:
    automated:
      - tool: "axe-core"
        integration: "jest-axe for unit tests"
      - tool: "lighthouse"
        threshold: 90
    
    manual:
      - "Screen reader testing (NVDA/VoiceOver)"
      - "Keyboard-only navigation"
      - "Zoom to 200%"
```

### 3. Add React A11y ESLint Rules (5 min)
Update `.eslintrc.js` or `eslint.config.js` to include:
```javascript
// Add to extends or plugins
{
  extends: [
    'plugin:jsx-a11y/recommended'
  ],
  plugins: ['jsx-a11y'],
  rules: {
    'jsx-a11y/alt-text': 'error',
    'jsx-a11y/anchor-has-content': 'error',
    'jsx-a11y/click-events-have-key-events': 'warn',
    'jsx-a11y/no-noninteractive-element-interactions': 'warn',
  }
}
```

Run: `npm install -D eslint-plugin-jsx-a11y`

### 4. Create Skip-to-Content Component Template (5 min)
Create `docs/templates/SkipToContent.tsx`:
```tsx
/**
 * Skip-to-content link for keyboard users
 * Add to the top of your layout component
 */
export function SkipToContent() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded"
    >
      Skip to main content
    </a>
  );
}

// Usage in layout:
// <body>
//   <SkipToContent />
//   <nav>...</nav>
//   <main id="main-content">...</main>
// </body>
```

### 5. Add npm Scripts (2 min)
Add to `package.json`:
```json
"a11y:audit": "tsx tools/accessibility/audit.ts",
"a11y:check": "tsx tools/accessibility/audit.ts checklist"
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `tools/accessibility/audit.ts` | Create |
| `.config/accessibility/wcag.yaml` | Create |
| `.eslintrc.js` or `eslint.config.js` | Add jsx-a11y |
| `docs/templates/SkipToContent.tsx` | Create |
| `package.json` | Add scripts + eslint-plugin-jsx-a11y |

---

## Success Criteria

- [ ] `npm run a11y:audit` runs (or shows checklist)
- [ ] jsx-a11y ESLint plugin installed and configured
- [ ] WCAG config documents requirements
- [ ] Skip-to-content template available

---

## Commit
`feat(a11y): Complete Phase 10 accessibility compliance`

