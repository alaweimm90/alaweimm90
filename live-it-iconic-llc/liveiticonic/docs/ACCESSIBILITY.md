# Accessibility Standards & Guidelines
## Live It Iconic - WCAG 2.2 Level AA Compliance

This document provides standards and patterns for implementing accessible features in the Live It Iconic e-commerce platform.

---

## Table of Contents

1. [Color Contrast](#color-contrast)
2. [Focus Management](#focus-management)
3. [ARIA Labels & Attributes](#aria-labels--attributes)
4. [Semantic HTML](#semantic-html)
5. [Keyboard Navigation](#keyboard-navigation)
6. [Images & Media](#images--media)
7. [Forms](#forms)
8. [Common Patterns](#common-patterns)
9. [Testing Procedures](#testing-procedures)
10. [Keyboard Shortcuts Reference](#keyboard-shortcuts-reference)

---

## Color Contrast

### Standard Colors & Contrast Ratios

**Live It Iconic Brand Colors:**
```
Background (lii-bg):   #0B0B0C - 0% brightness
Ink (lii-ink):         #14161A - 9% brightness
Cloud (lii-cloud):     #E6E9EF - 93% brightness  ← Use for text
Gold (lii-gold):       #C1A060 - 57% brightness  ← Use for accents
Ash (lii-ash):         #8C93A3 - 57% brightness  ← Avoid for body text
```

### Contrast Requirements (WCAG 2.2 AA)

| Text Type | Requirement | Acceptable Color Combinations |
|-----------|-------------|------------------------------|
| Normal text (< 18pt) | 4.5:1 | Cloud on Background (15.8:1) |
| Large text (≥ 18pt) | 3:1 | Gold on Background (7.2:1) |
| Focus indicator | 3:1 | Gold outline on any bg (7.2:1) |

### Implementation

✅ **DO:**
```tsx
// Use cloud for body text
<p className="text-lii-cloud">Readable text</p>

// Use gold for interactive elements and accents
<a href="#" className="text-lii-gold hover:text-lii-cloud">Link</a>

// Large headings can use gradient with cloud
<h1 className="bg-gradient-to-r from-lii-cloud to-lii-gold bg-clip-text text-transparent">
  Title
</h1>
```

❌ **DON'T:**
```tsx
// Avoid ash for body text - insufficient contrast
<p className="text-lii-ash">Poor contrast text</p>

// Avoid light text on light backgrounds
<span className="text-lii-cloud bg-lii-cloud">Invisible</span>
```

---

## Focus Management

### Focus Indicator Standard

All interactive elements must have visible focus indicators:

```css
focus-visible:outline
focus-visible:outline-2
focus-visible:outline-lii-gold
focus-visible:outline-offset-2
```

**Specifications:**
- Width: 2px solid
- Color: #C1A060 (lii-gold)
- Offset: 2px
- Contrast: 7.2:1 (exceeds 3:1 requirement)

### Implementation

```tsx
// Buttons
<button className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2">
  Click me
</button>

// Links
<a href="#" className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 rounded-lg">
  Link
</a>

// Custom interactive elements
<div
  role="button"
  tabIndex={0}
  className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2"
>
  Interactive
</div>
```

### Focus Order

Tab order must match visual order. Use `tabindex` only when necessary:

```tsx
// Natural tab order (recommended)
<button>First</button>
<button>Second</button>
<button>Third</button>

// If reordering needed
<button tabIndex={1}>First</button>
<button tabIndex={3}>Third</button>
<button tabIndex={2}>Second</button>

// Never use tabindex > 0 (bad practice)
// ❌ <button tabIndex={5}>Bad</button>
```

### Focus Restoration

After closing modals/dialogs, restore focus to trigger element:

```tsx
const [isOpen, setIsOpen] = useState(false);
const triggerRef = useRef<HTMLButtonElement>(null);

const handleClose = () => {
  setIsOpen(false);
  triggerRef.current?.focus();
};

return (
  <>
    <button ref={triggerRef} onClick={() => setIsOpen(true)}>
      Open Modal
    </button>

    <Modal isOpen={isOpen} onClose={handleClose} />
  </>
);
```

---

## ARIA Labels & Attributes

### Common ARIA Attributes

```tsx
// aria-label - Provides accessible name when no visible label
<button aria-label="Close menu">✕</button>

// aria-labelledby - Links element to another element that labels it
<h2 id="modal-title">Confirm Action</h2>
<div role="dialog" aria-labelledby="modal-title">...</div>

// aria-describedby - Provides additional description
<input type="password" aria-describedby="pwd-hint" />
<small id="pwd-hint">Must be at least 8 characters</small>

// aria-live - Announces dynamic content changes
<div aria-live="polite" aria-atomic="true">
  Item added to cart
</div>

// aria-current - Marks current page in navigation
<a href="/about" aria-current="page">About</a>

// aria-expanded - Indicates if collapsible element is open
<button aria-expanded={isOpen} aria-controls="menu">Menu</button>

// aria-hidden - Hides decorative elements from screen readers
<div aria-hidden="true" className="decorative-glow"></div>

// aria-modal - Indicates modal dialog
<div role="dialog" aria-modal="true">...</div>
```

### Naming Pattern for Interactive Elements

Use descriptive aria-labels that include context:

```tsx
// ❌ Generic
<button aria-label="Submit">Submit</button>

// ✅ Specific
<button aria-label="Submit contact form">Submit</button>

// ❌ Vague
<button aria-label="Click">+</button>

// ✅ Descriptive
<button aria-label="Add to cart">+</button>

// ❌ Redundant
<a href="/shop" aria-label="Shop link">Shop</a>

// ✅ Contextual
<a href="/shop" aria-label="Shop our premium collection">Shop</a>
```

---

## Semantic HTML

### HTML Elements to Use

```tsx
// Navigation
<nav aria-label="Main navigation">
  <a href="/">Home</a>
  <a href="/shop">Shop</a>
</nav>

// Main content area
<main id="main-content">
  <h1>Page Title</h1>
  <article>
    <h2>Article Title</h2>
    <p>Content...</p>
  </article>
</main>

// Sections
<section>
  <h2>Section Title</h2>
  <p>Content...</p>
</section>

// Sidebar/complementary content
<aside>
  <h2>Related Links</h2>
  <ul>...</ul>
</aside>

// Forms
<form>
  <fieldset>
    <legend>Shipping Information</legend>
    <label htmlFor="address">Address</label>
    <input id="address" type="text" />
  </fieldset>
</form>

// Lists
<ul>
  <li>Item 1</li>
  <li>Item 2</li>
</ul>

// Definition lists
<dl>
  <dt>Product</dt>
  <dd>Premium apparel</dd>
</dl>
```

### Heading Hierarchy

Maintain logical h1 → h2 → h3 progression:

```tsx
✅ Correct:
<h1>Page Title</h1>
<h2>Section</h2>
<h3>Subsection</h3>
<h3>Another Subsection</h3>
<h2>Another Section</h2>

❌ Incorrect:
<h1>Page Title</h1>
<h3>Skipped h2!</h3>  ← Jumps from h1 to h3
<h2>Goes back to h2</h2>
```

---

## Keyboard Navigation

### Keyboard Support Requirements

All interactive elements must support:
- **Tab/Shift+Tab** - Navigate between elements
- **Enter/Space** - Activate buttons and links
- **Escape** - Close dialogs/modals
- **Arrow keys** - Navigate within components (menus, tabs, sliders)

### Implementation Checklist

```tsx
// ✅ Keyboard accessible button
<button onClick={handleClick}>
  Save Changes
</button>

// ❌ Not keyboard accessible
<div onClick={handleClick} className="cursor-pointer">
  Save Changes
</div>

// ✅ Link with visible focus
<a href="/page" className="focus-visible:outline...">
  Go to page
</a>

// ✅ Custom button with proper attributes
<div
  role="button"
  tabIndex={0}
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }}
  className="focus-visible:outline..."
>
  Custom Button
</div>
```

### Minimum Touch Target Size

All interactive elements should be at least 44x44px:

```tsx
// ✅ Meets 44x44px minimum
<button className="min-w-[44px] min-h-[44px]">
  <Icon />
</button>

// ✅ Text button is large enough
<button className="px-6 py-3 text-base">
  Click me
</button>

// ❌ Too small
<button className="w-8 h-8">x</button>
```

---

## Images & Media

### Alt Text Guidelines

Provide descriptive alt text (10-14 words) for all meaningful images:

```tsx
// ✅ Descriptive
<img
  src="athlete.jpg"
  alt="Athlete in black Live It Iconic hoodie posing with luxury supercar"
/>

// ✅ Decorative - empty alt
<img
  src="background-pattern.png"
  alt=""
/>

// ✅ Icon with text alternative
<img
  src="cart-icon.svg"
  alt="Shopping cart"
/>

// ❌ Generic
<img src="photo.jpg" alt="photo" />

// ❌ Redundant
<img src="arrow.svg" alt="arrow icon" />
<span>Next</span>  ← Already says "arrow"
```

### Video & Audio

```tsx
// ✅ Video with captions
<video controls>
  <source src="video.mp4" type="video/mp4" />
  <track kind="captions" src="captions.vtt" />
</video>

// ✅ Audio with transcript
<audio controls>
  <source src="podcast.mp3" type="audio/mpeg" />
</audio>
<a href="#transcript">Jump to transcript</a>

// Audio transcript
<div id="transcript">
  <h2>Transcript</h2>
  <p>Full text of audio content...</p>
</div>
```

---

## Forms

### Form Accessibility Pattern

```tsx
<form onSubmit={handleSubmit}>
  <fieldset>
    <legend>Contact Information</legend>

    {/* Required field */}
    <div className="form-group">
      <label htmlFor="email">
        Email Address
        <span aria-label="required">*</span>
      </label>
      <input
        id="email"
        type="email"
        required
        aria-required="true"
        aria-describedby="email-error"
      />
      {errors.email && (
        <div id="email-error" role="alert" className="error-message">
          {errors.email}
        </div>
      )}
    </div>

    {/* Optional field */}
    <div className="form-group">
      <label htmlFor="phone">
        Phone Number
        <span className="optional">(optional)</span>
      </label>
      <input
        id="phone"
        type="tel"
        aria-describedby="phone-format"
      />
      <small id="phone-format">Format: (123) 456-7890</small>
    </div>

    {/* Checkbox group */}
    <fieldset>
      <legend>Preferences</legend>
      <div>
        <input id="newsletter" type="checkbox" />
        <label htmlFor="newsletter">
          Subscribe to our newsletter
        </label>
      </div>
    </fieldset>

    <button type="submit">Submit</button>
  </fieldset>
</form>
```

### Form Error Handling

```tsx
const [errors, setErrors] = useState({});

const handleSubmit = (e) => {
  e.preventDefault();
  const newErrors = validateForm(data);

  if (Object.keys(newErrors).length > 0) {
    setErrors(newErrors);
    // Announce error to screen readers
    announceToScreenReader(
      `Form submission failed. ${Object.keys(newErrors).length} errors found.`,
      'assertive'
    );
    // Focus first error
    document.getElementById(Object.keys(newErrors)[0]).focus();
  } else {
    submitForm(data);
  }
};
```

---

## Common Patterns

### Skip Navigation Link

```tsx
import SkipNavigation from '@/components/SkipNavigation';

<SkipNavigation />
<nav>...</nav>
<main id="main-content">
  {/* Page content */}
</main>
```

### Accessible Modal

```tsx
import AccessibleModal from '@/components/AccessibleModal';

<AccessibleModal
  isOpen={isOpen}
  onClose={handleClose}
  title="Confirm Action"
>
  Are you sure?
</AccessibleModal>
```

### Screen Reader Announcements

```tsx
import { announceToScreenReader } from '@/lib/a11y';

// Polite announcement (waits for pause)
announceToScreenReader('Item added to cart', 'polite');

// Assertive announcement (interrupts)
announceToScreenReader('Error: Please fill in required fields', 'assertive');
```

### Focus Trap (in modals)

```tsx
// Already implemented in AccessibleModal component
// Use for custom implementations:

import { useFocusTrap } from '@/lib/a11y';

const CustomModal = () => {
  const containerRef = useRef(null);
  useFocusTrap(containerRef, isOpen);

  return <div ref={containerRef}>{/* Modal content */}</div>;
};
```

---

## Testing Procedures

### Manual Testing Checklist

#### Keyboard Navigation
- [ ] All interactive elements are reachable by Tab
- [ ] Tab order matches visual order
- [ ] Focus indicator is visible on all elements
- [ ] No keyboard traps
- [ ] Escape closes modals
- [ ] Enter activates buttons
- [ ] Space activates buttons/checkboxes

#### Screen Reader Testing
- [ ] Skip link is announced
- [ ] Page structure is announced correctly
- [ ] Form labels are associated
- [ ] Error messages are announced
- [ ] Dynamic content changes are announced
- [ ] Button purposes are clear
- [ ] Image alt text is accurate

#### Color Contrast
- [ ] All text meets 4.5:1 ratio (normal) or 3:1 ratio (large)
- [ ] Focus indicators have sufficient contrast
- [ ] No color-only information (always use text/icon)

#### Mobile/Touch
- [ ] Touch targets are at least 44x44px
- [ ] Forms are zoomed properly on mobile
- [ ] Focus is visible on touch devices

### Automated Testing

#### Using Axe DevTools
```bash
npm install @axe-core/react
```

```tsx
import { axe } from '@axe-core/react';

const results = await axe(container);
console.log(results);
```

#### Using ESLint Plugin
```bash
npm install eslint-plugin-jsx-a11y
```

Configure in `eslint.config.js` to catch common issues.

#### Lighthouse Audit
1. Open Chrome DevTools
2. Go to Lighthouse tab
3. Run accessibility audit
4. Review issues and recommendations

---

## Keyboard Shortcuts Reference

### Browser Built-ins
| Key | Action |
|-----|--------|
| Tab | Move to next interactive element |
| Shift+Tab | Move to previous interactive element |
| Enter | Activate button/link |
| Space | Toggle checkbox/button |
| Escape | Close modal/menu |
| Alt+Left | Go back |
| Alt+Right | Go forward |

### Live It Iconic App Shortcuts
| Key | Action |
|-----|--------|
| Shift+? | Show shortcuts (coming soon) |
| / | Focus search (coming soon) |
| g h | Go to home (coming soon) |
| g s | Go to shop (coming soon) |

---

## Resources

### WCAG 2.2
- [Official Standard](https://www.w3.org/WAI/WCAG22/quickref/)
- [Understanding WCAG](https://www.w3.org/WAI/WCAG22/Understanding/)

### ARIA
- [Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [ARIA in HTML](https://www.w3.org/TR/html-aria/)

### Testing Tools
- [Axe DevTools](https://www.deque.com/axe/devtools/)
- [WAVE Browser Extension](https://wave.webaim.org/extension/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)

### Screen Readers
- [NVDA (Free)](https://www.nvaccess.org/) - Windows
- [JAWS](https://www.freedomscientific.com/products/software/jaws/) - Windows
- [VoiceOver](https://www.apple.com/accessibility/voiceover/) - macOS/iOS
- [TalkBack](https://support.google.com/accessibility/android/answer/6283677) - Android

---

## Getting Help

For accessibility questions or issues:
1. Check this documentation
2. Review WCAG 2.2 guidelines
3. Test with automated tools
4. Consult with accessibility specialists

---

**Last Updated:** November 12, 2025
**Maintained By:** Accessibility Team
