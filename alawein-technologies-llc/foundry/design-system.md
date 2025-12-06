# Foundry Design System

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Color System](#color-system)
3. [Typography](#typography)
4. [Spacing Scale](#spacing-scale)
5. [Elevation & Shadows](#elevation--shadows)
6. [Motion & Animation](#motion--animation)
7. [Accessibility Standards](#accessibility-standards)
8. [Component Specifications](#component-specifications)
9. [Responsive Design](#responsive-design)
10. [Dark Mode Strategy](#dark-mode-strategy)

---

## Design Philosophy

### Core Principles

1. **Clarity First**: Every element should have a clear purpose and hierarchy
2. **Consistency**: Unified experience across all applications
3. **Accessibility**: WCAG 2.1 AA compliant, inclusive design for all users
4. **Performance**: Lightweight, optimized components with minimal overhead
5. **Flexibility**: Composable components that adapt to various contexts
6. **Delight**: Subtle animations and interactions that feel natural

### Design Values

- **Minimalist**: Clean interfaces that prioritize content
- **Modern**: Contemporary design patterns that feel fresh
- **Professional**: Trustworthy and reliable aesthetic
- **Innovative**: Forward-thinking without sacrificing usability
- **Responsive**: Seamless experience across all devices

---

## Color System

### Light Mode Palette

```css
/* Primary Colors */
--color-primary-50: #eef2ff;
--color-primary-100: #e0e7ff;
--color-primary-200: #c7d2fe;
--color-primary-300: #a5b4fc;
--color-primary-400: #818cf8;
--color-primary-500: #6366f1;
--color-primary-600: #4f46e5;
--color-primary-700: #4338ca;
--color-primary-800: #3730a3;
--color-primary-900: #312e81;
--color-primary-950: #1e1b4b;

/* Secondary Colors */
--color-secondary-50: #faf5ff;
--color-secondary-100: #f3e8ff;
--color-secondary-200: #e9d5ff;
--color-secondary-300: #d8b4fe;
--color-secondary-400: #c084fc;
--color-secondary-500: #a855f7;
--color-secondary-600: #9333ea;
--color-secondary-700: #7e22ce;
--color-secondary-800: #6b21a8;
--color-secondary-900: #581c87;
--color-secondary-950: #3b0764;

/* Semantic Colors */
--color-success-50: #f0fdf4;
--color-success-100: #dcfce7;
--color-success-200: #bbf7d0;
--color-success-300: #86efac;
--color-success-400: #4ade80;
--color-success-500: #22c55e;
--color-success-600: #16a34a;
--color-success-700: #15803d;
--color-success-800: #166534;
--color-success-900: #14532d;

--color-warning-50: #fffbeb;
--color-warning-100: #fef3c7;
--color-warning-200: #fde68a;
--color-warning-300: #fcd34d;
--color-warning-400: #fbbf24;
--color-warning-500: #f59e0b;
--color-warning-600: #d97706;
--color-warning-700: #b45309;
--color-warning-800: #92400e;
--color-warning-900: #78350f;

--color-danger-50: #fef2f2;
--color-danger-100: #fee2e2;
--color-danger-200: #fecaca;
--color-danger-300: #fca5a5;
--color-danger-400: #f87171;
--color-danger-500: #ef4444;
--color-danger-600: #dc2626;
--color-danger-700: #b91c1c;
--color-danger-800: #991b1b;
--color-danger-900: #7f1d1d;

/* Neutral Colors */
--color-gray-50: #f9fafb;
--color-gray-100: #f3f4f6;
--color-gray-200: #e5e7eb;
--color-gray-300: #d1d5db;
--color-gray-400: #9ca3af;
--color-gray-500: #6b7280;
--color-gray-600: #4b5563;
--color-gray-700: #374151;
--color-gray-800: #1f2937;
--color-gray-900: #111827;
--color-gray-950: #030712;
```

### Dark Mode Palette

```css
/* Dark mode uses inverted gray scale and adjusted colors for better contrast */
--color-background: #0f172a;
--color-surface: #1e293b;
--color-surface-hover: #334155;
--color-border: #334155;
--color-text-primary: #f1f5f9;
--color-text-secondary: #cbd5e1;
--color-text-muted: #94a3b8;

/* Adjusted semantic colors for dark mode */
--color-primary-dark: #818cf8;
--color-secondary-dark: #c084fc;
--color-success-dark: #4ade80;
--color-warning-dark: #fbbf24;
--color-danger-dark: #f87171;
```

---

## Typography

### Type Scale

```css
/* Font Families */
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
--font-mono: 'Fira Code', 'Consolas', 'Monaco', 'Andale Mono', monospace;

/* Font Sizes */
--text-xs: 0.75rem;     /* 12px */
--text-sm: 0.875rem;    /* 14px */
--text-base: 1rem;      /* 16px */
--text-lg: 1.125rem;    /* 18px */
--text-xl: 1.25rem;     /* 20px */
--text-2xl: 1.5rem;     /* 24px */
--text-3xl: 1.875rem;   /* 30px */
--text-4xl: 2.25rem;    /* 36px */
--text-5xl: 3rem;       /* 48px */
--text-6xl: 3.75rem;    /* 60px */
--text-7xl: 4.5rem;     /* 72px */
--text-8xl: 6rem;       /* 96px */
--text-9xl: 8rem;       /* 128px */

/* Line Heights */
--leading-none: 1;
--leading-tight: 1.25;
--leading-snug: 1.375;
--leading-normal: 1.5;
--leading-relaxed: 1.625;
--leading-loose: 2;

/* Font Weights */
--font-thin: 100;
--font-extralight: 200;
--font-light: 300;
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
--font-extrabold: 800;
--font-black: 900;

/* Letter Spacing */
--tracking-tighter: -0.05em;
--tracking-tight: -0.025em;
--tracking-normal: 0em;
--tracking-wide: 0.025em;
--tracking-wider: 0.05em;
--tracking-widest: 0.1em;
```

### Typography Components

```css
/* Headings */
.h1 {
  font-size: var(--text-5xl);
  font-weight: var(--font-bold);
  line-height: var(--leading-tight);
  letter-spacing: var(--tracking-tight);
}

.h2 {
  font-size: var(--text-4xl);
  font-weight: var(--font-bold);
  line-height: var(--leading-tight);
  letter-spacing: var(--tracking-tight);
}

.h3 {
  font-size: var(--text-3xl);
  font-weight: var(--font-semibold);
  line-height: var(--leading-snug);
}

.h4 {
  font-size: var(--text-2xl);
  font-weight: var(--font-semibold);
  line-height: var(--leading-snug);
}

.h5 {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  line-height: var(--leading-normal);
}

.h6 {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  line-height: var(--leading-normal);
}

/* Body Text */
.body-large {
  font-size: var(--text-lg);
  line-height: var(--leading-relaxed);
}

.body {
  font-size: var(--text-base);
  line-height: var(--leading-normal);
}

.body-small {
  font-size: var(--text-sm);
  line-height: var(--leading-normal);
}

/* Special Text */
.caption {
  font-size: var(--text-xs);
  line-height: var(--leading-normal);
}

.overline {
  font-size: var(--text-xs);
  font-weight: var(--font-semibold);
  text-transform: uppercase;
  letter-spacing: var(--tracking-widest);
}

.code {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
}
```

---

## Spacing Scale

### Base Unit: 4px

```css
/* Spacing Values */
--space-0: 0;
--space-px: 1px;
--space-0.5: 0.125rem;  /* 2px */
--space-1: 0.25rem;     /* 4px */
--space-1.5: 0.375rem;  /* 6px */
--space-2: 0.5rem;      /* 8px */
--space-2.5: 0.625rem;  /* 10px */
--space-3: 0.75rem;     /* 12px */
--space-3.5: 0.875rem;  /* 14px */
--space-4: 1rem;        /* 16px */
--space-5: 1.25rem;     /* 20px */
--space-6: 1.5rem;      /* 24px */
--space-7: 1.75rem;     /* 28px */
--space-8: 2rem;        /* 32px */
--space-9: 2.25rem;     /* 36px */
--space-10: 2.5rem;     /* 40px */
--space-12: 3rem;       /* 48px */
--space-14: 3.5rem;     /* 56px */
--space-16: 4rem;       /* 64px */
--space-20: 5rem;       /* 80px */
--space-24: 6rem;       /* 96px */
--space-28: 7rem;       /* 112px */
--space-32: 8rem;       /* 128px */
--space-36: 9rem;       /* 144px */
--space-40: 10rem;      /* 160px */
--space-44: 11rem;      /* 176px */
--space-48: 12rem;      /* 192px */
--space-52: 13rem;      /* 208px */
--space-56: 14rem;      /* 224px */
--space-60: 15rem;      /* 240px */
--space-64: 16rem;      /* 256px */
--space-72: 18rem;      /* 288px */
--space-80: 20rem;      /* 320px */
--space-96: 24rem;      /* 384px */
```

### Component Spacing Guidelines

- **Extra Small (xs)**: 4px - Used for icon spacing, small gaps
- **Small (sm)**: 8px - Used for tight component padding
- **Medium (md)**: 16px - Default spacing for most components
- **Large (lg)**: 24px - Used for section spacing
- **Extra Large (xl)**: 32px - Used for major section breaks
- **2X Large (2xl)**: 48px - Used for page sections
- **3X Large (3xl)**: 64px - Used for hero sections
- **4X Large (4xl)**: 80px - Used for major page divisions

---

## Elevation & Shadows

### Shadow Scale

```css
/* Shadow Values */
--shadow-none: none;
--shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
--shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
--shadow-3xl: 0 35px 60px -15px rgb(0 0 0 / 0.3);
--shadow-inner: inset 0 2px 4px 0 rgb(0 0 0 / 0.05);

/* Colored Shadows */
--shadow-primary: 0 10px 25px -5px rgb(99 102 241 / 0.3);
--shadow-secondary: 0 10px 25px -5px rgb(168 85 247 / 0.3);
--shadow-success: 0 10px 25px -5px rgb(34 197 94 / 0.3);
--shadow-warning: 0 10px 25px -5px rgb(245 158 11 / 0.3);
--shadow-danger: 0 10px 25px -5px rgb(239 68 68 / 0.3);
```

### Elevation Levels

1. **Level 0**: No shadow - Base level
2. **Level 1**: xs shadow - Cards, tiles
3. **Level 2**: sm shadow - Raised cards, hover states
4. **Level 3**: md shadow - Dropdowns, tooltips
5. **Level 4**: lg shadow - Modals, popovers
6. **Level 5**: xl shadow - Floating action buttons
7. **Level 6**: 2xl shadow - Dialogs, overlays

---

## Motion & Animation

### Duration Scale

```css
--duration-75: 75ms;
--duration-100: 100ms;
--duration-150: 150ms;
--duration-200: 200ms;
--duration-300: 300ms;
--duration-500: 500ms;
--duration-700: 700ms;
--duration-1000: 1000ms;
```

### Easing Functions

```css
--ease-linear: linear;
--ease-in: cubic-bezier(0.4, 0, 1, 1);
--ease-out: cubic-bezier(0, 0, 0.2, 1);
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
```

### Animation Presets

```css
/* Fade */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes fadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

/* Scale */
@keyframes scaleUp {
  from { transform: scale(0.95); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

@keyframes scaleDown {
  from { transform: scale(1.05); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

/* Slide */
@keyframes slideInUp {
  from { transform: translateY(100%); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes slideInDown {
  from { transform: translateY(-100%); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes slideInLeft {
  from { transform: translateX(-100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

@keyframes slideInRight {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Spin */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Pulse */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Bounce */
@keyframes bounce {
  0%, 100% {
    transform: translateY(-25%);
    animation-timing-function: cubic-bezier(0.8, 0, 1, 1);
  }
  50% {
    transform: translateY(0);
    animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
  }
}

/* Skeleton Loading */
@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
```

### Transition Guidelines

- **Micro-interactions**: 75-150ms (hover, focus)
- **Small transitions**: 200-300ms (dropdowns, tooltips)
- **Medium transitions**: 300-500ms (modals, page transitions)
- **Large transitions**: 500-700ms (complex animations)
- **Extra large**: 700-1000ms (staggered animations)

---

## Accessibility Standards

### WCAG 2.1 AA Compliance

#### Color Contrast Ratios
- **Normal text**: Minimum 4.5:1
- **Large text** (18pt+): Minimum 3:1
- **UI components**: Minimum 3:1
- **Non-text contrast**: Minimum 3:1

#### Keyboard Navigation
- All interactive elements must be keyboard accessible
- Tab order must be logical and predictable
- Focus indicators must be clearly visible (2px minimum)
- Skip links for main navigation
- Escape key closes modals/dropdowns

#### Screen Reader Support
- Semantic HTML elements
- Proper ARIA labels and descriptions
- Live regions for dynamic content
- Meaningful link text
- Image alt texts

#### Interactive Elements
- Minimum touch target: 44x44px
- Clear hover and focus states
- Error messages associated with form fields
- Loading states announced to screen readers

---

## Component Specifications

### Component Structure

Each component follows this structure:

```typescript
interface ComponentProps {
  // Required props
  children?: React.ReactNode;
  className?: string;

  // Variant props
  variant?: 'primary' | 'secondary' | 'danger' | 'success' | 'warning';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  // State props
  disabled?: boolean;
  loading?: boolean;

  // Accessibility props
  ariaLabel?: string;
  ariaDescribedBy?: string;

  // Event handlers
  onClick?: () => void;
  onChange?: () => void;
}
```

### Component States

1. **Default**: Base state
2. **Hover**: Mouse over interaction
3. **Focus**: Keyboard navigation
4. **Active**: Being clicked/pressed
5. **Disabled**: Non-interactive
6. **Loading**: Async operation
7. **Error**: Validation failure
8. **Success**: Positive feedback

### Component Sizes

- **Extra Small (xs)**: Compact, minimal padding
- **Small (sm)**: Reduced size for dense UIs
- **Medium (md)**: Default size
- **Large (lg)**: Increased size for emphasis
- **Extra Large (xl)**: Maximum size for CTAs

---

## Responsive Design

### Breakpoints

```css
/* Mobile First Approach */
--screen-xs: 320px;   /* Extra small devices */
--screen-sm: 640px;   /* Small devices */
--screen-md: 768px;   /* Medium devices */
--screen-lg: 1024px;  /* Large devices */
--screen-xl: 1280px;  /* Extra large devices */
--screen-2xl: 1536px; /* 2X large devices */
```

### Container Widths

```css
.container {
  width: 100%;
  margin: 0 auto;
  padding: 0 1rem;
}

@media (min-width: 640px) {
  .container { max-width: 640px; }
}

@media (min-width: 768px) {
  .container { max-width: 768px; }
}

@media (min-width: 1024px) {
  .container { max-width: 1024px; }
}

@media (min-width: 1280px) {
  .container { max-width: 1280px; }
}

@media (min-width: 1536px) {
  .container { max-width: 1536px; }
}
```

### Grid System

```css
.grid {
  display: grid;
  gap: var(--space-4);
}

/* Column system */
.grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
.grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid-cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.grid-cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.grid-cols-12 { grid-template-columns: repeat(12, minmax(0, 1fr)); }
```

---

## Dark Mode Strategy

### Implementation Approach

1. **CSS Variables**: All colors defined as CSS custom properties
2. **Class-based switching**: `.dark` class on root element
3. **System preference detection**: `prefers-color-scheme` media query
4. **Local storage persistence**: Remember user preference
5. **Smooth transitions**: Fade between themes

### Dark Mode Color Adjustments

```css
:root.dark {
  /* Backgrounds */
  --color-background: var(--color-gray-950);
  --color-surface: var(--color-gray-900);
  --color-surface-hover: var(--color-gray-800);

  /* Text */
  --color-text-primary: var(--color-gray-50);
  --color-text-secondary: var(--color-gray-200);
  --color-text-muted: var(--color-gray-400);

  /* Borders */
  --color-border: var(--color-gray-800);
  --color-border-hover: var(--color-gray-700);

  /* Adjusted colors for better contrast */
  --color-primary: var(--color-primary-400);
  --color-secondary: var(--color-secondary-400);
  --color-success: var(--color-success-400);
  --color-warning: var(--color-warning-400);
  --color-danger: var(--color-danger-400);
}
```

### Dark Mode Best Practices

1. **Reduce brightness**: Lower overall luminance
2. **Increase contrast**: Ensure text remains readable
3. **Desaturate colors**: Reduce saturation for comfort
4. **Avoid pure black**: Use dark grays for better depth
5. **Test thoroughly**: Verify all components in both modes

---

## Component Library Overview

### Layout Components
- Container
- Grid
- Flex
- Spacer
- Divider

### Navigation Components
- Navbar
- Sidebar
- Breadcrumb
- Tabs
- Pagination
- Link

### Form Components
- Input
- Textarea
- Select
- Checkbox
- Radio
- Switch
- DatePicker
- FileUpload
- Form
- FormField
- FormValidation

### Data Display Components
- Table
- Card
- List
- Avatar
- Badge
- Tag
- Stat
- EmptyState
- Skeleton

### Feedback Components
- Button
- Alert
- Toast
- Modal
- Drawer
- Popover
- Tooltip
- Progress
- Spinner
- LoadingDots

### Advanced Components
- Chart
- DataGrid
- SearchAutocomplete
- FilterPanel
- Timeline
- Stepper
- Accordion
- Carousel

---

## Usage Guidelines

### Getting Started

1. **Install dependencies**:
```bash
npm install tailwindcss framer-motion lucide-react
```

2. **Import design tokens**:
```javascript
import { theme } from './theme';
import { colors, spacing, typography } from './design-tokens';
```

3. **Use components**:
```jsx
import { Button, Card, Input } from './components';

function App() {
  return (
    <Card>
      <Input placeholder="Enter text" />
      <Button variant="primary">Submit</Button>
    </Card>
  );
}
```

### Best Practices

1. **Consistent spacing**: Use spacing scale for all margins/padding
2. **Semantic colors**: Use color meanings (success, danger, etc.)
3. **Responsive first**: Design for mobile, enhance for desktop
4. **Accessibility always**: Never compromise on a11y
5. **Performance matters**: Optimize bundle size and render performance

---

## Version History

- **v1.0.0**: Initial design system release
- Components: 30+ production-ready components
- Full TypeScript support
- WCAG 2.1 AA compliant
- Dark mode support
- Responsive design system
- Comprehensive documentation