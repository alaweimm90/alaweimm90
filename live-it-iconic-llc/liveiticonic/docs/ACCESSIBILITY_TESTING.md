# Accessibility Testing Guide
## Live It Iconic - WCAG 2.2 Level AA Compliance Testing

---

## Quick Start Testing Checklist

### Daily Development Testing (5 minutes)
- [ ] Tab through all interactive elements
- [ ] Verify focus indicators are visible
- [ ] Test Escape key in modals
- [ ] Check contrast on text changes
- [ ] Run browser DevTools Lighthouse

### Before Pull Request (15 minutes)
- [ ] Run ESLint JSA11y plugin
- [ ] Run Axe accessibility scanner
- [ ] Test with keyboard only
- [ ] Test with screen reader
- [ ] Review ARIA labels

### Before Release (30 minutes)
- [ ] Full keyboard navigation test
- [ ] Screen reader test (all pages)
- [ ] Color contrast audit
- [ ] Mobile accessibility test
- [ ] Lighthouse audit

---

## Manual Testing Methods

### 1. Keyboard Navigation Testing

**Steps:**
1. Disconnect your mouse
2. Use only Tab, Shift+Tab, Enter, Space, and Escape
3. Navigate through the entire page

**What to Check:**
- All interactive elements are reachable
- Tab order makes sense
- Focus is always visible
- No elements are "trapped"
- Enter activates buttons/links
- Space toggles checkboxes
- Escape closes modals

**Common Issues:**
- Focus gets stuck on an element
- Tab order jumps around randomly
- Focus invisible (not visible when tabbing)
- Cannot reach element with keyboard

### 2. Color Contrast Testing

**Using WebAIM Contrast Checker:**
1. Go to https://webaim.org/resources/contrastchecker/
2. Use browser's color picker to get exact colors
3. Enter background and foreground colors
4. Check ratio against WCAG AA standards:
   - Normal text: 4.5:1
   - Large text (18pt+): 3:1
   - Focus indicators: 3:1

**Using Chrome DevTools:**
1. Inspect element
2. Click the color in the Styles panel
3. Check the contrast ratio shown

### 3. Screen Reader Testing

#### Windows - NVDA (Free)
Installation:
```bash
# Download from https://www.nvaccess.org/
# Run installer
```

Basic Commands:
| Key | Action |
|-----|--------|
| Insert+Down | Read current element |
| Insert+Right | Read next element |
| Insert+F7 | Elements list |
| Insert+H | Toggle heading navigation |
| H | Jump to next heading |

#### macOS - VoiceOver (Built-in)
```bash
# Enable: Cmd + F5
# List headings: VO + U (opens rotor)
# Navigate: VO + Right Arrow
# Activate: VO + Space
```

#### Quick Test Script
```javascript
// Run in browser console to test screen reader readability
const testElements = document.querySelectorAll('button, a, input, [role="button"]');
testElements.forEach(el => {
  const label = el.getAttribute('aria-label') ||
                el.textContent.trim() ||
                el.getAttribute('title');
  console.log(`${el.tagName}: ${label}`);
});
```

### 4. Zoom and Text Enlargement

**Test at different zoom levels:**
- 100% (default)
- 200% (Cmd/Ctrl + +)
- 400% (Cmd/Ctrl + +++)

**Check:**
- No text is cut off
- Layout remains usable
- No horizontal scrolling at 200%
- Focus indicators remain visible

### 5. Mobile Accessibility Testing

**Android (TalkBack):**
```
Enable: Settings → Accessibility → TalkBack → On
Swipe right/left: Navigate elements
Swipe down twice: Activate
Swipe up twice: Go back
```

**iOS (VoiceOver):**
```
Enable: Settings → Accessibility → VoiceOver → On
Swipe right: Next element
Swipe left: Previous element
Double-tap: Activate
Swipe up: Read from top
```

---

## Automated Testing Tools

### 1. Axe DevTools (Recommended)

**Installation:**
```bash
npm install --save-dev @axe-core/react
```

**Using in Tests:**
```typescript
import { axe, toHaveNoViolations } from '@axe-core/react';

expect.extend(toHaveNoViolations);

describe('Accessibility', () => {
  it('should have no accessibility violations', async () => {
    const { container } = render(<YourComponent />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
```

**Chrome Extension:**
1. Install "axe DevTools" from Chrome Web Store
2. Open DevTools
3. Click "Scan ALL of my page"
4. Review violations

### 2. WAVE Browser Extension

1. Install from https://wave.webaim.org/extension/
2. Click WAVE icon in toolbar
3. Review errors, warnings, and contrast issues

### 3. Lighthouse (Built into Chrome DevTools)

1. Open Chrome DevTools
2. Go to "Lighthouse" tab
3. Select "Accessibility"
4. Click "Analyze page load"
5. Review results

**Target Score: 90+**

### 4. ESLint JSA11y Plugin

**Installation:**
```bash
npm install --save-dev eslint-plugin-jsx-a11y
```

**Configuration:**
```javascript
// eslint.config.js
import jsxA11y from 'eslint-plugin-jsx-a11y';

export default [
  {
    plugins: { 'jsx-a11y': jsxA11y },
    rules: {
      'jsx-a11y/alt-text': 'error',
      'jsx-a11y/anchor-is-valid': 'error',
      'jsx-a11y/aria-props': 'error',
      'jsx-a11y/aria-role': 'error',
      'jsx-a11y/heading-has-content': 'error',
      'jsx-a11y/label-has-associated-control': 'error',
      'jsx-a11y/no-static-element-interactions': 'error',
      'jsx-a11y/click-events-have-key-events': 'error',
      // More rules...
    },
  },
];
```

**Run:**
```bash
npm run lint
```

### 5. Pa11y (Command Line Tool)

**Installation:**
```bash
npm install --save-dev pa11y pa11y-ci
```

**Usage:**
```bash
# Test single page
pa11y http://localhost:3000/

# Test with configuration
pa11y --config .pa11yrc.json http://localhost:3000/
```

---

## Test Cases by Component

### Navigation Component

**Keyboard:**
- [ ] Tab reaches all navigation links
- [ ] Focus indicator visible on each link
- [ ] Escape closes mobile menu
- [ ] Mobile menu button announces state

**Screen Reader:**
- [ ] Navigation landmark is announced
- [ ] Links have descriptive text
- [ ] Mobile menu button state is announced
- [ ] Current page indication (aria-current)

**Visual:**
- [ ] Focus outline visible against background
- [ ] Outline doesn't obscure text
- [ ] Color contrast > 4.5:1

### ProductCard Component

**Keyboard:**
- [ ] Product button is reachable by Tab
- [ ] Favorite button is reachable
- [ ] Both buttons are activatable with Enter

**Screen Reader:**
- [ ] Product name is announced
- [ ] Category is announced
- [ ] Price is announced
- [ ] Favorite button has context label

**Mobile:**
- [ ] Product button is at least 44x44px
- [ ] Favorite button is at least 44x44px

### CartDrawer Component

**Keyboard:**
- [ ] Tab navigates through all controls
- [ ] Escape closes drawer
- [ ] Quantity +/- buttons work with keyboard
- [ ] Focus returns to cart icon after close

**Screen Reader:**
- [ ] Dialog title is announced
- [ ] Item count is announced
- [ ] Quantity changes announced
- [ ] Removal announced
- [ ] Total announced

**Touch:**
- [ ] All buttons are 44x44px minimum
- [ ] Touch targets are clearly spaced

### Forms (When Implemented)

**Keyboard:**
- [ ] All inputs are keyboard accessible
- [ ] Tab moves between fields logically
- [ ] Required indicator visible
- [ ] Error on blur is announced

**Screen Reader:**
- [ ] Labels associated with inputs
- [ ] Required/optional announced
- [ ] Error messages associated
- [ ] Fieldset legend announced

**Visual:**
- [ ] Labels visible and adjacent to inputs
- [ ] Error messages in red text + icon
- [ ] Focus indicator on input fields

---

## Performance Optimization Verification

### Lighthouse Scores Target
- **Accessibility:** 90+
- **Performance:** 80+
- **Best Practices:** 85+

### Core Web Vitals
- **Largest Contentful Paint (LCP):** < 2.5s
- **Cumulative Layout Shift (CLS):** < 0.1
- **First Input Delay (FID):** < 100ms

### Bundle Size
- Monitor CSS/JS accessibility code doesn't increase bundle more than 10KB

---

## Browser Compatibility Testing

### Keyboard Navigation
- [ ] Chrome/Edge (Windows)
- [ ] Firefox (Windows)
- [ ] Safari (macOS)
- [ ] Chrome (macOS)

### Screen Readers
- [ ] NVDA (Windows)
- [ ] JAWS (Windows, paid)
- [ ] VoiceOver (macOS, iOS)
- [ ] TalkBack (Android)

### Mobile Testing
- [ ] iOS (Safari, VoiceOver)
- [ ] Android (Chrome, TalkBack)
- [ ] iOS (Chrome)
- [ ] Android (Samsung Internet)

---

## Test Report Template

### Component Test Report
```markdown
# Component: [Name]
Date: [Date]
Tester: [Name]

## Keyboard Navigation
- [ ] All interactive elements reachable by Tab
- [ ] Tab order is logical
- [ ] Focus always visible
- [ ] No keyboard traps
- [x] Notes: [Any issues found]

## Screen Reader Testing
- [ ] Tested with: NVDA / JAWS / VoiceOver
- [ ] All elements properly announced
- [ ] Reading order is logical
- [ ] ARIA labels are descriptive
- [ ] Notes: [Any issues found]

## Color Contrast
- [ ] All text meets 4.5:1 for normal or 3:1 for large
- [ ] Focus indicators meet 3:1
- [ ] All checks passed
- [ ] Issues: [Any violations]

## Mobile/Touch
- [ ] Touch targets are 44x44px minimum
- [ ] Zoom to 200% works properly
- [ ] No horizontal scroll at 200%
- [ ] Issues: [Any violations]

## Overall Score
- Lighthouse Accessibility: [Score]/100
- Axe Violations: [Count]
- WAVE Errors: [Count]

## Sign-off
Tested by: [Name]
Approved: [Name]
Date: [Date]
```

---

## Common Issues & Fixes

### Issue: Focus indicator not visible
**Fix:**
```css
/* Ensure sufficient contrast */
focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2
```

### Issue: Missing alt text on image
**Fix:**
```tsx
<img
  src="photo.jpg"
  alt="Athlete in black hoodie with supercar"
/>
```

### Issue: Button doesn't work with keyboard
**Fix:**
```tsx
// Use semantic button element
<button onClick={handleClick}>Click</button>

// Or add keyboard handler to div
<div
  role="button"
  tabIndex={0}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick();
    }
  }}
>
  Click
</div>
```

### Issue: Form label not associated
**Fix:**
```tsx
<label htmlFor="email">Email</label>
<input id="email" type="email" />
```

### Issue: No visible focus on input
**Fix:**
```tsx
<input
  className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold"
/>
```

---

## Resources for Testers

### Documentation
- [ACCESSIBILITY.md](/docs/ACCESSIBILITY.md) - Comprehensive guide
- [ACCESSIBILITY_AUDIT.md](/ACCESSIBILITY_AUDIT.md) - Audit report

### Tools
- [Axe DevTools](https://www.deque.com/axe/devtools/) - Chrome/Firefox
- [WAVE](https://wave.webaim.org/) - Web-based and browser extension
- [Lighthouse](https://developers.google.com/web/tools/lighthouse) - Built in Chrome
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/) - Web-based
- [NVDA](https://www.nvaccess.org/) - Free screen reader

### Learning
- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [Web Accessibility by Google](https://www.udacity.com/course/web-accessibility--ud891)
- [A11ycasts](https://www.youtube.com/playlist?list=PLNYkxOF6rcICWx0C9Xc-RgEzwLvePng7V)

---

## Continuous Testing Integration

### GitHub Actions Workflow
```yaml
name: Accessibility Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - run: npm install
      - run: npm run lint
      - run: npm run test:a11y
      - run: npm run build
      - name: Run Lighthouse CI
        uses: treosh/lighthouse-ci-action@v9
```

---

## Schedule

### Daily
- Run ESLint (`npm run lint`)
- Manual keyboard navigation test

### Weekly
- Run Lighthouse audit
- Test with screen reader
- Review WCAG compliance

### Before Release
- Full accessibility audit
- User testing with assistive tech
- Third-party review

---

**Last Updated:** November 12, 2025
**Maintained By:** Accessibility Team
