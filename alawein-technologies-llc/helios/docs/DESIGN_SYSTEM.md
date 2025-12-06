# HELIOS Design System v1.0

**Professional, accessible, modern design system for enterprise-grade research platform.**

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Color System](#color-system)
3. [Typography](#typography)
4. [Spacing & Layout](#spacing--layout)
5. [Components](#components)
6. [Interactions & Motion](#interactions--motion)
7. [Accessibility](#accessibility)
8. [Responsive Design](#responsive-design)
9. [Dark Mode](#dark-mode)
10. [Usage Guidelines](#usage-guidelines)

---

## Design Principles

HELIOS design is guided by five core principles:

### 🎯 **Clarity**
Every element serves a purpose. We eliminate ambiguity through clear visual hierarchy, descriptive labels, and intuitive interactions.

### 🤝 **Inclusivity**
Design for everyone. We support keyboard navigation, screen readers, high contrast modes, and reduced motion preferences.

### ⚡ **Efficiency**
Minimize friction. Users should accomplish their goals with the fewest clicks, lowest cognitive load, and fastest feedback.

### 🎨 **Beauty**
Professional aesthetics matter. Consistent spacing, thoughtful colors, and premium details build trust and confidence.

### 🔄 **Consistency**
One system, one language. All interfaces follow the same patterns, making them predictable and learnable.

---

## Color System

### Primary Color Palette (Light Mode)

| Color | Hex | Usage |
|-------|-----|-------|
| **Primary** | `#2563eb` | Primary actions, links, focus states |
| **Primary Dark** | `#1e40af` | Hover states, dark variant |
| **Primary Light** | `#3b82f6` | Secondary actions |
| **Primary Lighter** | `#eff6ff` | Backgrounds, accents |
| **Secondary** | `#8b5cf6` | Accents, highlights, gradients |
| **Secondary Light** | `#a78bfa` | Light variant |

### Semantic Colors

| Color | Hex | Usage |
|-------|-----|-------|
| **Success** | `#10b981` | Positive actions, confirmations, success states |
| **Warning** | `#f59e0b` | Warnings, cautions, alerts |
| **Error** | `#ef4444` | Errors, deletions, danger |
| **Info** | `#06b6d4` | Information, notifications |

### Neutral Palette (Light Mode)

| Color | Hex | Usage |
|-------|-----|-------|
| **Background** | `#ffffff` | Main backgrounds |
| **BG Secondary** | `#f9fafb` | Subtle backgrounds, sections |
| **BG Tertiary** | `#f3f4f6` | Cards, panels, hover states |
| **Text Primary** | `#111827` | Main text, headings |
| **Text Secondary** | `#6b7280` | Secondary text, descriptions |
| **Text Tertiary** | `#9ca3af` | Disabled, placeholder, subtle text |
| **Border** | `#e5e7eb` | Dividers, borders, strokes |
| **Border Light** | `#f3f4f6` | Subtle borders |

### Dark Mode Palette

Dark mode uses complementary colors for contrast and readability:
- Background: `#0f172a` (slate-900)
- Secondary BG: `#1e293b` (slate-800)
- Text Primary: `#f1f5f9` (slate-100)
- All other colors maintain their hue for consistency

---

## Typography

### Font Stack

```css
/* Sans Serif (UI) */
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;

/* Monospace (Code) */
font-family: 'JetBrains Mono', 'Courier New', monospace;
```

### Type Scale

| Size | Token | Usage |
|------|-------|-------|
| 12px | `--text-xs` | Labels, badges, small text |
| 14px | `--text-sm` | Secondary text, descriptions |
| 16px | `--text-base` | Body text (default) |
| 18px | `--text-lg` | Subheadings, input labels |
| 20px | `--text-xl` | Smaller headings |
| 24px | `--text-2xl` | Section titles |
| 30px | `--text-3xl` | Page titles |
| 36px | `--text-4xl` | Hero titles |

### Font Weights

| Weight | Usage |
|--------|-------|
| **300** | Light, subtle emphasis |
| **400** | Regular body text, default |
| **500** | Medium, slightly emphasized |
| **600** | Semi-bold, labels, headings |
| **700** | Bold, primary headings |

### Line Height

| Token | Value | Usage |
|-------|-------|-------|
| Tight | 1.2 | Headings, dense text |
| Normal | 1.5 | Body text, default |
| Relaxed | 1.75 | Long-form text, descriptions |

---

## Spacing & Layout

### Spacing Scale

All spacing uses a consistent 4px grid:

```
xs  = 0.25rem (4px)
sm  = 0.5rem  (8px)
md  = 1rem    (16px)
lg  = 1.5rem  (24px)
xl  = 2rem    (32px)
2xl = 3rem    (48px)
3xl = 4rem    (64px)
```

### Layout Grid

- **Container Width**: 1400px (max-width)
- **Gutter**: 24px (responsive: 16px on tablets, 8px on mobile)
- **Column System**: 12-column flexible grid
- **Breakpoints**:
  - Desktop: 1200px+
  - Tablet: 768px - 1199px
  - Mobile: < 768px

### Common Spacing Patterns

```css
/* Margin bottom for spacing sections */
h1, h2, h3 { margin-bottom: 1.5rem; }
p { margin-bottom: 1rem; }

/* Padding for containers */
.container { padding: 3rem 1.5rem; }

/* Gap between grid items */
.grid { gap: 2rem; }
```

---

## Components

### Buttons

**Primary Button**
- Background: Primary color
- Color: White
- Padding: 8px 24px
- Border Radius: 12px
- Box Shadow: Shadow MD
- Hover: Darker background, lifted shadow
- Active: No lift, pressed feel

**Secondary Button**
- Background: Tertiary background
- Color: Primary text
- Border: 1px solid border color
- Padding: 8px 24px
- Hover: Border color changes, background shifts

**Small Button**
- Padding: 4px 12px
- Font Size: 14px

### Input Fields

**Select Input**
- Padding: 8px 12px
- Border: 1px solid border color
- Border Radius: 6px
- Focus: Blue border, shadow ring
- Disabled: Opacity 0.5

**Text Area**
- Padding: 12px
- Font Family: Monospace
- Resize: Vertical only
- Min Height: 100px

### Cards

**Standard Card**
- Background: Secondary background
- Border: 1px solid border
- Padding: 24px
- Border Radius: 16px
- Box Shadow: Shadow SM

**Card Hover**
- Background: Tertiary background
- Border: Primary color
- Box Shadow: Shadow LG
- Transform: translateY(-4px)

### Badges

- Padding: 4px 8px
- Font Size: 12px
- Font Weight: 600
- Border Radius: 4px
- Background: Primary lighter
- Color: Primary dark

### Tables

- Header Background: Tertiary background
- Row Hover: Tertiary background
- Border: 1px solid border
- Cell Padding: 16px
- Font Size: 14px (header), 13px (body)

---

## Interactions & Motion

### Transitions

| Token | Duration | Easing | Usage |
|-------|----------|--------|-------|
| Fast | 150ms | cubic-bezier(0.4, 0, 0.2, 1) | Hover states, color changes |
| Base | 250ms | cubic-bezier(0.4, 0, 0.2, 1) | General interactions |
| Slow | 350ms | cubic-bezier(0.4, 0, 0.2, 1) | Page transitions |

### Animations

**Fade In**
```css
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
```
Duration: 250ms | Easing: Base

**Slide In**
```css
@keyframes slideIn {
    from { width: 0; }
    to { width: 100%; }
}
```
Duration: 350ms | Easing: Base

### Micro-interactions

- **Button Press**: Press effect on click (scale: 0.98)
- **Hover Lift**: Cards move up 4px on hover
- **Focus Ring**: 2px solid blue outline with 2px offset
- **Loading**: Animated progress bar with gradient
- **Disabled**: 50% opacity, no interaction feedback

### Reduced Motion

Respects `prefers-reduced-motion: reduce`:
```css
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

## Accessibility

### Color Contrast

All text meets WCAG AAA standards:
- Normal text: 7:1 contrast ratio
- Large text: 4.5:1 contrast ratio

### Keyboard Navigation

- **Tab**: Navigate forward
- **Shift+Tab**: Navigate backward
- **Enter/Space**: Activate buttons
- **Escape**: Close modals/dropdowns
- **Arrow Keys**: Navigate within components

### Focus Management

- Focus visible on all interactive elements
- Focus outline: 2px solid primary color
- Focus offset: 2px
- Focus trap in modals

### Screen Reader Support

- All interactive elements have `aria-label` or label text
- Form fields have associated labels
- Live regions announce updates (`role="status"`, `aria-live="polite"`)
- Tables use proper header elements (`<th scope="col">`)
- Icons have text alternatives

### Semantic HTML

- Proper heading hierarchy (h1 > h2 > h3)
- Forms use `<label>` elements
- Buttons use `<button>` (not divs)
- Navigation uses `<nav>` with `role="navigation"`
- Main content in `<main>` with `id="main-content"`

---

## Responsive Design

### Breakpoints

```css
/* Desktop (1200px and up) */
@media (min-width: 1200px) { }

/* Tablet (768px to 1199px) */
@media (max-width: 1199px) { }

/* Mobile (below 768px) */
@media (max-width: 768px) { }

/* Small Mobile (below 480px) */
@media (max-width: 480px) { }
```

### Responsive Patterns

- **Grid**: Auto-fit with minmax
  ```css
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  ```

- **Flexible Layouts**: Use flexbox for wrapping
  ```css
  flex-wrap: wrap;
  ```

- **Padding/Margin**: Scale with viewport
  ```css
  padding: var(--space-2xl) var(--space-lg); /* 48px 24px */
  ```

- **Font Size**: Maintain readability
  ```css
  font-size: clamp(1.5rem, 5vw, 2.25rem);
  ```

---

## Dark Mode

### Implementation

Uses CSS custom properties that update via media query:

```css
@media (prefers-color-scheme: dark) {
    :root {
        --color-bg: #0f172a;
        --color-text-primary: #f1f5f9;
        /* ... all other colors ... */
    }
}
```

### Color Adjustments

- Text colors invert for contrast
- Backgrounds use darker neutrals
- Shadows increase opacity
- Borders use lighter neutrals

### Testing Dark Mode

- Firefox DevTools: `Ctrl+Shift+M` (open responsive design mode, then click system preference indicator)
- Chrome DevTools: Rendering tab → Emulate CSS media feature prefers-color-scheme
- Manual: System settings → Display → Dark mode

---

## Usage Guidelines

### When to Use Each Component

| Component | Best For |
|-----------|----------|
| Primary Button | Main actions, form submission |
| Secondary Button | Alternative actions, cancellation |
| Link | Navigation, within text |
| Badge | Status, category, tag |
| Card | Related content grouping |
| Table | Structured data comparison |
| Form Input | Data entry |
| Select | Choosing from list |
| Textarea | Long-form input |

### Color Usage

- **Primary**: Main actions, important information
- **Secondary**: Highlights, accents, gradients
- **Success**: Confirmations, positive results
- **Warning**: Cautions, alerts, attention needed
- **Error**: Errors, destructive actions, danger
- **Neutral**: Content, backgrounds, structure

### Spacing Rules

- Never use arbitrary spacing—always use scale values
- For spacing between elements: use margin on bottom element
- For internal padding: use padding on container
- For gaps between grid items: use CSS grid gap

### Typography Rules

- Use semantic HTML (`<h1>`, `<h2>`, etc.)
- Never skip heading levels
- Use 14px or larger for body text
- Line-height ≥ 1.5 for readability
- Max line-width ~75 characters for long text

### Motion Guidelines

- Use fast (150ms) for simple state changes
- Use base (250ms) for general interactions
- Use slow (350ms) for complex transitions
- Always respect prefers-reduced-motion
- Avoid motion that distracts from content

---

## File Structure

```
helios/web/
├── index.html          # HTML structure
├── styles.css          # All styling & design tokens
├── app.js              # Interactive functionality
├── components/         # Reusable components (future)
│   ├── button.css
│   ├── card.css
│   └── ...
└── assets/
    ├── logo.svg
    ├── icons/
    └── fonts/
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-19 | Initial design system, complete component library |

---

## Questions?

For design questions, refer to this guide. For implementation questions, check `IMPLEMENTATION_GUIDE.md`.

**Made with ❤️ for the HELIOS project.**
