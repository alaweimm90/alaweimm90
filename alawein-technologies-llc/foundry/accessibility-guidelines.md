# Accessibility Guidelines

## WCAG 2.1 AA Compliance Checklist

This document outlines the accessibility standards implemented in the Foundry Design System to ensure WCAG 2.1 AA compliance.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Color Contrast](#color-contrast)
3. [Keyboard Navigation](#keyboard-navigation)
4. [Screen Reader Support](#screen-reader-support)
5. [Focus Management](#focus-management)
6. [Form Accessibility](#form-accessibility)
7. [Component-Specific Guidelines](#component-specific-guidelines)
8. [Testing Checklist](#testing-checklist)
9. [Tools & Resources](#tools--resources)

---

## Core Principles

### 1. Perceivable
- Information and UI components must be presentable in ways users can perceive
- Provide text alternatives for non-text content
- Create content that can be presented without losing meaning
- Make it easier for users to see and hear content

### 2. Operable
- UI components and navigation must be operable
- Make all functionality available from keyboard
- Give users enough time to read and use content
- Don't design content that causes seizures

### 3. Understandable
- Information and UI operation must be understandable
- Make text content readable and understandable
- Make web pages appear and operate predictably
- Help users avoid and correct mistakes

### 4. Robust
- Content must be robust enough for interpretation by wide variety of user agents
- Maximize compatibility with assistive technologies

---

## Color Contrast

### Minimum Contrast Ratios

#### Normal Text
- **Minimum**: 4.5:1 against background
- **Enhanced**: 7:1 for AAA compliance

#### Large Text (18pt+ or 14pt+ bold)
- **Minimum**: 3:1 against background
- **Enhanced**: 4.5:1 for AAA compliance

#### UI Components & Graphics
- **Minimum**: 3:1 for essential visual information

### Implementation

```css
/* Light Mode */
--text-primary: #111827;     /* Against white: 15.3:1 ✓ */
--text-secondary: #4b5563;   /* Against white: 7.5:1 ✓ */
--text-muted: #9ca3af;       /* Against white: 3.3:1 (large text only) */

/* Dark Mode */
--text-primary-dark: #f9fafb;    /* Against black: 20.1:1 ✓ */
--text-secondary-dark: #d1d5db;  /* Against black: 13.1:1 ✓ */
--text-muted-dark: #6b7280;      /* Against black: 5.2:1 ✓ */
```

### Testing Tools
- Chrome DevTools Lighthouse
- WAVE (WebAIM)
- axe DevTools
- Stark (Figma plugin)

---

## Keyboard Navigation

### Required Support

#### Tab Navigation
- All interactive elements must be reachable via Tab key
- Tab order must follow logical reading order
- Shift+Tab must navigate backwards

#### Keyboard Shortcuts
```javascript
// Common keyboard patterns
Enter/Space - Activate buttons, links
Arrow Keys  - Navigate menus, lists, tabs
Escape      - Close modals, dropdowns
Home/End    - Jump to first/last item
```

### Implementation Example

```tsx
// Button component with proper keyboard support
<button
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }}
  tabIndex={disabled ? -1 : 0}
  aria-disabled={disabled}
>
  Click me
</button>
```

### Focus Trap for Modals

```tsx
useEffect(() => {
  if (!isOpen) return;

  const focusableElements = modalRef.current?.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );

  const firstElement = focusableElements?.[0] as HTMLElement;
  const lastElement = focusableElements?.[focusableElements.length - 1] as HTMLElement;

  // Focus first element when modal opens
  firstElement?.focus();

  const handleTab = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey && document.activeElement === firstElement) {
      e.preventDefault();
      lastElement?.focus();
    } else if (!e.shiftKey && document.activeElement === lastElement) {
      e.preventDefault();
      firstElement?.focus();
    }
  };

  document.addEventListener('keydown', handleTab);
  return () => document.removeEventListener('keydown', handleTab);
}, [isOpen]);
```

---

## Screen Reader Support

### ARIA Attributes

#### Required ARIA Labels

```tsx
// Button with icon only
<button aria-label="Delete item">
  <TrashIcon />
</button>

// Form input
<input
  aria-label="Email address"
  aria-describedby="email-error"
  aria-invalid={hasError}
  aria-required={required}
/>

// Loading state
<div aria-live="polite" aria-busy={loading}>
  {loading ? 'Loading...' : content}
</div>
```

#### ARIA Roles

```tsx
// Navigation
<nav role="navigation" aria-label="Main navigation">

// Search
<form role="search" aria-label="Site search">

// Alert
<div role="alert" aria-live="assertive">
  Error: Invalid email address
</div>

// Status
<div role="status" aria-live="polite">
  3 items added to cart
</div>
```

### Semantic HTML

Always prefer semantic HTML over ARIA:

```tsx
// ✅ Good - Semantic HTML
<nav>
  <ul>
    <li><a href="/home">Home</a></li>
    <li><a href="/about">About</a></li>
  </ul>
</nav>

// ❌ Avoid - Non-semantic with ARIA
<div role="navigation">
  <div role="list">
    <div role="listitem"><div role="link">Home</div></div>
  </div>
</div>
```

---

## Focus Management

### Visual Focus Indicators

```css
/* Default focus style for all interactive elements */
:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

/* Custom focus styles */
.button:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
}

/* Remove focus for mouse users */
.button:focus:not(:focus-visible) {
  outline: none;
}
```

### Skip Links

```tsx
// Skip to main content link
<a href="#main-content" className="skip-link">
  Skip to main content
</a>

<style>
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--color-primary-500);
  color: white;
  padding: 8px;
  text-decoration: none;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
</style>
```

---

## Form Accessibility

### Label Association

```tsx
// Explicit label
<label htmlFor="email">Email Address</label>
<input id="email" type="email" />

// Label wrapping
<label>
  <span>Email Address</span>
  <input type="email" />
</label>

// Hidden label (screen reader only)
<label htmlFor="search" className="sr-only">Search</label>
<input id="search" type="search" placeholder="Search..." />
```

### Error Handling

```tsx
<div className="form-field">
  <label htmlFor="email">
    Email Address
    <span aria-label="required">*</span>
  </label>

  <input
    id="email"
    type="email"
    aria-describedby="email-error email-hint"
    aria-invalid={hasError}
    aria-required="true"
  />

  <span id="email-hint" className="hint">
    Enter your work email address
  </span>

  {hasError && (
    <span id="email-error" role="alert" className="error">
      Please enter a valid email address
    </span>
  )}
</div>
```

### Fieldset & Legend

```tsx
<fieldset>
  <legend>Notification Preferences</legend>

  <label>
    <input type="checkbox" name="notifications" value="email" />
    Email notifications
  </label>

  <label>
    <input type="checkbox" name="notifications" value="sms" />
    SMS notifications
  </label>
</fieldset>
```

---

## Component-Specific Guidelines

### Button
- Must have accessible name (text content or aria-label)
- Disabled buttons should use aria-disabled
- Loading state should be announced
- Icon-only buttons require aria-label

### Modal
- Focus trapped within modal
- Focus returns to trigger element on close
- Escape key closes modal
- Background content marked with aria-hidden
- Role="dialog" with aria-modal="true"

### Table
- Use proper table markup (<table>, <thead>, <tbody>, etc.)
- Column headers with scope attribute
- Caption or aria-label for table description
- Sortable columns announced to screen readers

### Navigation
- Use semantic <nav> element
- Current page indicated with aria-current="page"
- Submenus properly associated with triggers
- Keyboard navigation with arrow keys

### Toast/Alert
- Use appropriate ARIA live regions
- Error: role="alert" aria-live="assertive"
- Success: role="status" aria-live="polite"
- Auto-dismiss timing considerations (minimum 5 seconds)

---

## Testing Checklist

### Manual Testing

#### Keyboard Testing
- [ ] Tab through entire page
- [ ] All interactive elements reachable
- [ ] Focus indicators visible
- [ ] No keyboard traps
- [ ] Logical tab order
- [ ] Escape closes overlays

#### Screen Reader Testing
- [ ] All content readable
- [ ] Images have alt text
- [ ] Form labels announced
- [ ] Error messages announced
- [ ] Dynamic content updates announced
- [ ] Headings properly nested

#### Visual Testing
- [ ] Zoom to 200% without horizontal scroll
- [ ] Text spacing adjustable
- [ ] Color not sole indicator
- [ ] Sufficient contrast ratios
- [ ] Focus indicators visible

### Automated Testing

```javascript
// Example test with jest-axe
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('Button should be accessible', async () => {
  const { container } = render(<Button>Click me</Button>);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### Browser Testing
- [ ] Chrome + NVDA
- [ ] Firefox + JAWS
- [ ] Safari + VoiceOver
- [ ] Mobile Safari + VoiceOver
- [ ] Android Chrome + TalkBack

---

## Tools & Resources

### Development Tools
- **axe DevTools** - Browser extension for accessibility testing
- **WAVE** - Web accessibility evaluation tool
- **Lighthouse** - Chrome DevTools auditing
- **Pa11y** - Command line accessibility testing
- **React Testing Library** - Encourages accessible queries

### Screen Readers
- **NVDA** (Windows) - Free, open source
- **JAWS** (Windows) - Commercial, widely used
- **VoiceOver** (macOS/iOS) - Built-in Apple screen reader
- **TalkBack** (Android) - Built-in Android screen reader
- **ORCA** (Linux) - Free, open source

### Color Contrast Tools
- **Stark** - Figma/Sketch plugin
- **Contrast** - macOS app
- **WebAIM Contrast Checker** - Online tool
- **Chrome DevTools** - Built-in contrast checker

### Documentation
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [WAI-ARIA Authoring Practices](https://www.w3.org/TR/wai-aria-practices-1.1/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [WebAIM Resources](https://webaim.org/resources/)
- [A11y Project](https://www.a11yproject.com/)

---

## Component Accessibility Status

| Component | Keyboard | Screen Reader | Focus | ARIA | Color Contrast | Status |
|-----------|----------|---------------|-------|------|----------------|---------|
| Button | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Input | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Select | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Modal | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Table | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Card | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Badge | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Toast | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Spinner | N/A | ✅ | N/A | ✅ | ✅ | Complete |

---

## Accessibility Statement

The Foundry Design System is committed to ensuring digital accessibility for people with disabilities. We are continually improving the user experience for everyone and applying the relevant accessibility standards.

### Conformance Status

The Web Content Accessibility Guidelines (WCAG) defines requirements for designers and developers to improve accessibility for people with disabilities. It defines three levels of conformance: Level A, Level AA, and Level AAA. The Foundry Design System is partially conformant with WCAG 2.1 level AA.

### Feedback

We welcome your feedback on the accessibility of the Foundry Design System. Please let us know if you encounter accessibility barriers.

---

*Last Updated: November 2024*