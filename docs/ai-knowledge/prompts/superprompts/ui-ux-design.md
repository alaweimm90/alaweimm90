---
name: 'UI/UX Design Superprompt'
version: '1.0'
category: 'project'
tags: ['ui', 'ux', 'design', 'accessibility', 'prototyping', 'user-research']
created: '2024-11-30'
---

# UI/UX Design Superprompt

## Purpose

Comprehensive framework for user interface design, user experience optimization, accessibility standards, and design system implementation.

---

## System Prompt

```
You are a Senior UI/UX Designer and Design Systems Architect with expertise in:
- User-centered design and design thinking
- Accessibility standards (WCAG 2.1 AA/AAA)
- Design systems and component libraries
- Prototyping and user testing
- Responsive and adaptive design
- Motion design and micro-interactions
- Design tokens and theming

Your mission is to create designs that:
1. Prioritize user needs and accessibility
2. Maintain consistency through design systems
3. Enable rapid prototyping and iteration
4. Support responsive, cross-platform experiences
5. Integrate seamlessly with development workflows
```

---

## Design System Architecture

### Design Token Structure

```yaml
# design-tokens/tokens.yaml
tokens:
  colors:
    primitive:
      blue:
        50: '#eff6ff'
        100: '#dbeafe'
        500: '#3b82f6'
        600: '#2563eb'
        900: '#1e3a8a'
      gray:
        50: '#f9fafb'
        100: '#f3f4f6'
        500: '#6b7280'
        900: '#111827'

    semantic:
      primary:
        default: '{colors.blue.600}'
        hover: '{colors.blue.700}'
        active: '{colors.blue.800}'
        disabled: '{colors.blue.300}'
      background:
        default: '{colors.gray.50}'
        surface: '#ffffff'
        elevated: '#ffffff'
      text:
        primary: '{colors.gray.900}'
        secondary: '{colors.gray.600}'
        disabled: '{colors.gray.400}'
      border:
        default: '{colors.gray.200}'
        focus: '{colors.blue.500}'
      status:
        success: '#10b981'
        warning: '#f59e0b'
        error: '#ef4444'
        info: '#3b82f6'

  spacing:
    0: '0'
    1: '0.25rem' # 4px
    2: '0.5rem' # 8px
    3: '0.75rem' # 12px
    4: '1rem' # 16px
    5: '1.25rem' # 20px
    6: '1.5rem' # 24px
    8: '2rem' # 32px
    10: '2.5rem' # 40px
    12: '3rem' # 48px
    16: '4rem' # 64px

  typography:
    fontFamily:
      sans: 'Inter, system-ui, sans-serif'
      mono: 'JetBrains Mono, monospace'
    fontSize:
      xs: '0.75rem' # 12px
      sm: '0.875rem' # 14px
      base: '1rem' # 16px
      lg: '1.125rem' # 18px
      xl: '1.25rem' # 20px
      2xl: '1.5rem' # 24px
      3xl: '1.875rem' # 30px
      4xl: '2.25rem' # 36px
    fontWeight:
      normal: 400
      medium: 500
      semibold: 600
      bold: 700
    lineHeight:
      tight: 1.25
      normal: 1.5
      relaxed: 1.75

  borderRadius:
    none: '0'
    sm: '0.125rem' # 2px
    default: '0.25rem' # 4px
    md: '0.375rem' # 6px
    lg: '0.5rem' # 8px
    xl: '0.75rem' # 12px
    2xl: '1rem' # 16px
    full: '9999px'

  shadows:
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)'
    default: '0 1px 3px 0 rgb(0 0 0 / 0.1)'
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)'

  animation:
    duration:
      fast: '150ms'
      normal: '300ms'
      slow: '500ms'
    easing:
      default: 'cubic-bezier(0.4, 0, 0.2, 1)'
      in: 'cubic-bezier(0.4, 0, 1, 1)'
      out: 'cubic-bezier(0, 0, 0.2, 1)'
      inOut: 'cubic-bezier(0.4, 0, 0.2, 1)'
```

### Component Library Structure

```
design-system/
├── tokens/
│   ├── colors.json
│   ├── typography.json
│   ├── spacing.json
│   └── index.ts
├── components/
│   ├── primitives/
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.stories.tsx
│   │   │   ├── Button.test.tsx
│   │   │   └── Button.module.css
│   │   ├── Input/
│   │   ├── Select/
│   │   └── Checkbox/
│   ├── layout/
│   │   ├── Container/
│   │   ├── Grid/
│   │   ├── Stack/
│   │   └── Flex/
│   ├── feedback/
│   │   ├── Alert/
│   │   ├── Toast/
│   │   ├── Modal/
│   │   └── Spinner/
│   └── navigation/
│       ├── Navbar/
│       ├── Sidebar/
│       ├── Tabs/
│       └── Breadcrumb/
├── patterns/
│   ├── forms/
│   ├── tables/
│   └── cards/
├── icons/
├── docs/
└── storybook/
```

---

## Component Implementation

### Button Component

```tsx
// components/Button/Button.tsx
import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  // Base styles
  `inline-flex items-center justify-center rounded-md text-sm font-medium
   transition-colors focus-visible:outline-none focus-visible:ring-2
   focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50`,
  {
    variants: {
      variant: {
        primary: `bg-primary text-primary-foreground hover:bg-primary/90
                  focus-visible:ring-primary`,
        secondary: `bg-secondary text-secondary-foreground hover:bg-secondary/80
                    focus-visible:ring-secondary`,
        outline: `border border-input bg-background hover:bg-accent
                  hover:text-accent-foreground focus-visible:ring-accent`,
        ghost: `hover:bg-accent hover:text-accent-foreground`,
        destructive: `bg-destructive text-destructive-foreground
                      hover:bg-destructive/90 focus-visible:ring-destructive`,
        link: `text-primary underline-offset-4 hover:underline`,
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 py-2',
        lg: 'h-12 px-6 text-base',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, isLoading, leftIcon, rightIcon, children, disabled, ...props },
    ref
  ) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || isLoading}
        aria-busy={isLoading}
        {...props}
      >
        {isLoading ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
        ) : leftIcon ? (
          <span className="mr-2" aria-hidden="true">
            {leftIcon}
          </span>
        ) : null}
        {children}
        {rightIcon && !isLoading && (
          <span className="ml-2" aria-hidden="true">
            {rightIcon}
          </span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';
```

### Form Input Component

```tsx
// components/Input/Input.tsx
import React from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  leftAddon?: React.ReactNode;
  rightAddon?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = 'text', label, error, hint, leftAddon, rightAddon, id, ...props }, ref) => {
    const inputId = id || `input-${React.useId()}`;
    const errorId = `${inputId}-error`;
    const hintId = `${inputId}-hint`;

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-1">
            {label}
            {props.required && (
              <span className="text-red-500 ml-1" aria-hidden="true">
                *
              </span>
            )}
          </label>
        )}

        <div className="relative">
          {leftAddon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              {leftAddon}
            </div>
          )}

          <input
            type={type}
            id={inputId}
            ref={ref}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={cn(error && errorId, hint && hintId)}
            className={cn(
              `block w-full rounded-md border-gray-300 shadow-sm
               focus:border-primary focus:ring-primary sm:text-sm
               disabled:bg-gray-50 disabled:text-gray-500`,
              leftAddon && 'pl-10',
              rightAddon && 'pr-10',
              error && 'border-red-500 focus:border-red-500 focus:ring-red-500',
              className
            )}
            {...props}
          />

          {rightAddon && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center">{rightAddon}</div>
          )}
        </div>

        {error && (
          <p id={errorId} className="mt-1 text-sm text-red-600" role="alert">
            {error}
          </p>
        )}

        {hint && !error && (
          <p id={hintId} className="mt-1 text-sm text-gray-500">
            {hint}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
```

---

## Accessibility Standards

### WCAG 2.1 Checklist

```yaml
accessibility_checklist:
  perceivable:
    text_alternatives:
      - All images have alt text
      - Decorative images have empty alt=""
      - Complex images have long descriptions
      - Icons have accessible labels

    time_based_media:
      - Videos have captions
      - Audio has transcripts
      - No auto-playing media

    adaptable:
      - Semantic HTML structure
      - Proper heading hierarchy
      - Landmarks used correctly
      - Reading order is logical

    distinguishable:
      - Color contrast ratio >= 4.5:1 (text)
      - Color contrast ratio >= 3:1 (large text)
      - Color not sole indicator
      - Text resizable to 200%
      - No horizontal scrolling at 320px

  operable:
    keyboard:
      - All functionality keyboard accessible
      - No keyboard traps
      - Focus visible
      - Skip links provided
      - Logical tab order

    timing:
      - Adjustable time limits
      - Pause/stop/hide for moving content
      - No content flashing > 3 times/second

    navigation:
      - Multiple ways to find pages
      - Clear page titles
      - Focus order meaningful
      - Link purpose clear

  understandable:
    readable:
      - Language of page identified
      - Language of parts identified
      - Unusual words explained

    predictable:
      - Consistent navigation
      - Consistent identification
      - No unexpected context changes

    input_assistance:
      - Error identification
      - Labels or instructions
      - Error suggestion
      - Error prevention for legal/financial

  robust:
    compatible:
      - Valid HTML
      - Name, role, value for custom components
      - Status messages programmatically determined
```

### Accessibility Testing

```typescript
// tests/accessibility.test.ts
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility Tests', () => {
  test('homepage should have no accessibility violations', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('forms should be accessible', async ({ page }) => {
    await page.goto('/contact');

    // Check form labels
    const inputs = await page.locator('input:not([type="hidden"])').all();
    for (const input of inputs) {
      const id = await input.getAttribute('id');
      const label = await page.locator(`label[for="${id}"]`);
      expect(await label.count()).toBeGreaterThan(0);
    }

    // Check error messages
    await page.click('button[type="submit"]');
    const errors = await page.locator('[role="alert"]').all();
    for (const error of errors) {
      const id = await error.getAttribute('id');
      const input = await page.locator(`[aria-describedby*="${id}"]`);
      expect(await input.count()).toBeGreaterThan(0);
    }
  });

  test('keyboard navigation should work', async ({ page }) => {
    await page.goto('/');

    // Tab through interactive elements
    await page.keyboard.press('Tab');
    const firstFocused = await page.locator(':focus').first();
    expect(await firstFocused.isVisible()).toBe(true);

    // Check skip link
    await page.keyboard.press('Tab');
    const skipLink = await page.locator('a[href="#main-content"]');
    if ((await skipLink.count()) > 0) {
      await page.keyboard.press('Enter');
      const mainContent = await page.locator('#main-content');
      expect(await mainContent.isVisible()).toBe(true);
    }
  });

  test('color contrast should meet WCAG AA', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['cat.color'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });
});
```

---

## Responsive Design

### Breakpoint System

```css
/* styles/breakpoints.css */
:root {
  /* Mobile first breakpoints */
  --breakpoint-sm: 640px; /* Small devices */
  --breakpoint-md: 768px; /* Tablets */
  --breakpoint-lg: 1024px; /* Laptops */
  --breakpoint-xl: 1280px; /* Desktops */
  --breakpoint-2xl: 1536px; /* Large screens */
}

/* Container widths */
.container {
  width: 100%;
  margin-left: auto;
  margin-right: auto;
  padding-left: 1rem;
  padding-right: 1rem;
}

@media (min-width: 640px) {
  .container {
    max-width: 640px;
  }
}

@media (min-width: 768px) {
  .container {
    max-width: 768px;
  }
}

@media (min-width: 1024px) {
  .container {
    max-width: 1024px;
  }
}

@media (min-width: 1280px) {
  .container {
    max-width: 1280px;
  }
}

@media (min-width: 1536px) {
  .container {
    max-width: 1536px;
  }
}
```

### Responsive Component

```tsx
// components/ResponsiveGrid/ResponsiveGrid.tsx
import React from 'react';
import { cn } from '@/lib/utils';

interface ResponsiveGridProps {
  children: React.ReactNode;
  cols?: {
    default?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
  };
  gap?: number;
  className?: string;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  cols = { default: 1, sm: 2, md: 3, lg: 4 },
  gap = 4,
  className,
}) => {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4',
    5: 'grid-cols-5',
    6: 'grid-cols-6',
  };

  return (
    <div
      className={cn(
        'grid',
        `gap-${gap}`,
        cols.default && gridCols[cols.default as keyof typeof gridCols],
        cols.sm && `sm:${gridCols[cols.sm as keyof typeof gridCols]}`,
        cols.md && `md:${gridCols[cols.md as keyof typeof gridCols]}`,
        cols.lg && `lg:${gridCols[cols.lg as keyof typeof gridCols]}`,
        cols.xl && `xl:${gridCols[cols.xl as keyof typeof gridCols]}`,
        className
      )}
    >
      {children}
    </div>
  );
};
```

---

## User Feedback Loops

### Feedback Collection System

```yaml
feedback_system:
  collection_methods:
    in_app_feedback:
      - Rating widgets
      - Comment forms
      - Feature requests
      - Bug reports

    user_research:
      - Usability testing
      - A/B testing
      - Surveys
      - Interviews

    analytics:
      - Click tracking
      - Scroll depth
      - Time on page
      - Conversion funnels

  feedback_components:
    rating_widget:
      trigger: 'After key actions'
      questions:
        - 'How easy was this to complete?'
        - 'Did you find what you were looking for?'
      scale: '1-5 stars or emoji'

    nps_survey:
      trigger: 'Monthly for active users'
      question: 'How likely are you to recommend us?'
      follow_up: "What's the main reason for your score?"

    exit_survey:
      trigger: 'On account cancellation'
      questions:
        - 'Why are you leaving?'
        - 'What could we have done better?'
```

---

## Execution Phases

### Phase 1: Design Foundation

- [ ] Define design tokens
- [ ] Create color palette
- [ ] Establish typography scale
- [ ] Set up spacing system

### Phase 2: Component Library

- [ ] Build primitive components
- [ ] Create layout components
- [ ] Implement feedback components
- [ ] Add navigation components

### Phase 3: Accessibility

- [ ] Implement WCAG compliance
- [ ] Add keyboard navigation
- [ ] Create screen reader support
- [ ] Set up accessibility testing

### Phase 4: Documentation

- [ ] Set up Storybook
- [ ] Document components
- [ ] Create usage guidelines
- [ ] Build pattern library

---

_Last updated: 2024-11-30_
