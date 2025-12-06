# Live It Iconic - Design Tokens Documentation

## Overview

The Live It Iconic Design Tokens System is a comprehensive, modular design system that provides consistent, reusable design values across all platforms. Built with TypeScript and optimized for both web and cross-platform use.

**Latest Version**: 1.0.0
**Location**: `/src/design-tokens/`
**Status**: Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Token Categories](#token-categories)
3. [Usage Examples](#usage-examples)
4. [Architecture](#architecture)
5. [TypeScript Support](#typescript-support)
6. [Tailwind Integration](#tailwind-integration)
7. [CSS Variables](#css-variables)
8. [Migration Guide](#migration-guide)
9. [Best Practices](#best-practices)
10. [Contributing](#contributing)

---

## Quick Start

### Installation

The design tokens are built into the project at `/src/design-tokens/`.

### Basic Usage

```typescript
// Import entire token system
import { tokens } from '@/design-tokens';

// Access specific tokens
const primaryColor = tokens.colors.gold[300]; // '#C1A060'
const spacing16 = tokens.spacing[4]; // '1rem'
const fontSizeBase = tokens.typography.fontSizes.base; // '1rem'
```

### React Components

```typescript
import { colors, typography, spacing } from '@/design-tokens';

const MyComponent = () => {
  return (
    <div style={{
      color: colors.cloud[200],
      fontSize: typography.fontSizes.base,
      padding: spacing[4],
    }}>
      Hello World
    </div>
  );
};
```

### Tailwind CSS

```jsx
// Automatically integrated in tailwind.config.ts
<div className="text-lii-gold-300 p-4 rounded-xl shadow-md">
  Styled with design tokens
</div>
```

### CSS Custom Properties

```css
/* Auto-generated CSS variables */
:root {
  --lii-color-gold-300: #C1A060;
  --lii-spacing-4: 1rem;
  --lii-font-size-base: 1rem;
}

.element {
  color: var(--lii-color-gold-300);
  padding: var(--lii-spacing-4);
  border-radius: var(--lii-radius-xl);
}
```

---

## Token Categories

### 1. Colors (`colors.ts`)

Premium luxury brand color palette with semantic variants.

#### Primary Colors
- **Gold** (Luxury accent): Shades 50-900
  - Primary: `gold[300]` (#C1A060)
  - Used for: CTAs, highlights, luxury accents

- **Charcoal** (Primary background): Shades 50-500
  - Primary: `charcoal[300]` (#0B0B0C)
  - Used for: Primary background, text on light

- **Cloud** (Light neutral): Shades 50-400
  - Primary: `cloud[200]` (#E6E9EF)
  - Used for: Headings, light backgrounds, accents

#### Semantic Colors
- **Success**: Green palette (primary: #10B981)
- **Warning**: Orange palette (primary: #F59E0B)
- **Error**: Red palette (primary: #E03A2F)
- **Info**: Blue palette (primary: #3B82F6)

#### Usage

```typescript
import { colors } from '@/design-tokens';

// Access color
const buttonColor = colors.gold[300];

// Access semantic color
const successColor = colors.success[500];

// Create semi-transparent color
const withOpacity = colors.gold[300] + '80'; // 50% opacity
```

#### Color Variants

Each color palette includes multiple shades:
- **50-100**: Lightest (backgrounds, very subtle)
- **200-300**: Light-medium (UI elements, secondary)
- **400-500**: Medium (primary actions)
- **600-700**: Dark-medium (text, hover states)
- **800-900**: Darkest (strong contrast)

---

### 2. Typography (`typography.ts`)

Font families, sizes, weights, and spacing for text.

#### Font Families

```typescript
import { typography } from '@/design-tokens';

const fontFamilies = {
  display: "'Playfair Display', serif",      // Headlines, display text
  body: "'Inter Variable', sans-serif",       // Body text, UI
  mono: "'JetBrains Mono', monospace",       // Code, data
};
```

#### Font Sizes (Mobile-First)

| Scale | Size | Use Case |
|-------|------|----------|
| `xs` | 12px | Captions, labels |
| `sm` | 14px | Small text |
| `base` | 16px | Body text (default) |
| `lg` | 18px | Large body |
| `xl` | 20px | Small headings |
| `2xl` | 24px | Component headings |
| `3xl` | 30px | Section headings |
| `4xl` | 36px | Page headings |
| `5xl` | 48px | Hero headings |
| `6xl-9xl` | 60px-128px | Display/massive |

#### Typography Presets

Pre-built typography configurations:

```typescript
import { typographyPresets } from '@/design-tokens/typography';

// Display heading
const displayStyle = typographyPresets.displayLarge;

// Body text
const bodyStyle = typographyPresets.bodyRegular;

// Button text
const buttonStyle = typographyPresets.button;
```

#### Font Weights

| Weight | Value | Use |
|--------|-------|-----|
| light | 300 | Decorative |
| normal | 400 | Body text |
| medium | 500 | Labels, UI |
| semibold | 600 | Subheadings |
| bold | 700 | Headings |
| extrabold | 800 | Display |
| black | 900 | Massive display |

---

### 3. Spacing (`spacing.ts`)

Consistent spacing scale based on 8px base unit.

#### Spacing Scale

```typescript
import { spacing, grid } from '@/design-tokens';

// Base scale (multiples of 8px)
spacing[0];   // 0
spacing[1];   // 4px
spacing[2];   // 8px (base unit)
spacing[4];   // 16px (common)
spacing[6];   // 24px
spacing[8];   // 32px
spacing[12];  // 48px
spacing[16];  // 64px
spacing[24];  // 96px
spacing[32];  // 128px
```

#### Container Sizes

```typescript
import { sizes } from '@/design-tokens';

sizes.xs;    // 320px - Mobile
sizes.sm;    // 384px - Tablet
sizes.md;    // 448px
sizes.lg;    // 512px
sizes.xl;    // 576px
sizes['2xl']; // 672px
sizes['3xl']; // 768px
sizes['4xl']; // 896px
sizes['5xl']; // 1024px - Desktop
sizes['6xl']; // 1152px - Large
```

#### Grid System

```typescript
import { grid } from '@/design-tokens';

// Container widths
grid.containers.sm;   // 576px
grid.containers.md;   // 768px
grid.containers.lg;   // 1024px
grid.containers.xl;   // 1280px
grid.containers['2xl']; // 1536px

// Common gaps
grid.gaps.md;  // 16px
grid.gaps.lg;  // 24px

// Common padding
grid.padding.md;  // 24px
grid.padding.lg;  // 32px
```

#### Responsive Helpers

```typescript
import { responsivePadding, responsiveGap } from '@/design-tokens/spacing';

// Generate responsive class strings
const padding = responsivePadding(4, 6, 8);
// Result: "p-4 sm:p-6 lg:p-8"

const gap = responsiveGap(2, 4, 6);
// Result: "gap-2 sm:gap-4 lg:gap-6"
```

---

### 4. Shadows (`shadows.ts`)

Elevation system with gold-tinted luxury shadows.

#### Shadow Scales

```typescript
import { shadows } from '@/design-tokens';

shadows.none;   // none
shadows.xs;     // Subtle (1px)
shadows.sm;     // Light (3px blur)
shadows.md;     // Medium (6px blur) - Default
shadows.lg;     // Large (15px blur)
shadows.xl;     // Extra large (25px blur)
shadows['2xl'];  // Massive (50px blur)
```

#### Elevation Levels

```typescript
import { getElevationShadow } from '@/design-tokens/shadows';

// Map elevation level to shadow
const shadow1 = getElevationShadow(1); // xs
const shadow3 = getElevationShadow(3); // md
const shadow6 = getElevationShadow(6); // 2xl
```

#### Component-Specific Shadows

```typescript
import { shadowPresets } from '@/design-tokens/shadows';

shadowPresets.card;         // md
shadowPresets.cardHover;    // lg
shadowPresets.modal;        // 2xl
shadowPresets.tooltip;      // sm
shadowPresets.button;       // sm
shadowPresets.buttonHover;  // md
```

#### Dynamic Shadow Creation

```typescript
import { createShadow, createGlow } from '@/design-tokens/shadows';

// Custom shadow
const customShadow = createShadow(8, 12, 32, 0, 0.16);
// Result: "8px 12px 32px 0px rgba(193, 160, 96, 0.16)"

// Glow effect
const glow = createGlow('#C1A060', 20, 0.5);
```

---

### 5. Borders (`borders.ts`)

Border radius, widths, and focus ring system.

#### Border Radius

```typescript
import { borders } from '@/design-tokens';

// Canonical radius values
borders.radii.none;      // 0
borders.radii.sm;        // 2px
borders.radii.md;        // 6px
borders.radii.lg;        // 8px - Default for controls
borders.radii.xl;        // 12px - Primary buttons
borders.radii['2xl'];     // 16px - Card radius
borders.radii['3xl'];     // 24px - Large containers
borders.radii.full;      // 9999px - Fully rounded
```

#### Component-Specific Radius

```typescript
import { radiusPresets } from '@/design-tokens/borders';

radiusPresets.button;      // 12px
radiusPresets.card;        // 16px
radiusPresets.input;       // 8px
radiusPresets.modal;       // 16px
radiusPresets.tooltip;     // 6px
radiusPresets.avatar;      // 8px
radiusPresets.badge;       // 9999px (fully rounded)
```

#### Border Widths

```typescript
borders.widths[1];  // 1px - Default
borders.widths[2];  // 2px - Prominent
borders.widths[4];  // 4px
borders.widths[8];  // 8px - Heavy
```

#### Focus Ring

```typescript
import { focusRingStyles, focusRingCSS } from '@/design-tokens/borders';

// CSS object for focus ring
const styles = {
  ...focusRingStyles,  // Includes focus ring
};

// Manual focus ring
const customFocus = {
  outline: focusRingCSS.outline,
  outlineOffset: focusRingCSS.outlineOffset,
};
```

---

### 6. Motion (`motion.ts`)

Animation durations, easing functions, and keyframes.

#### Durations

```typescript
import { motion } from '@/design-tokens';

motion.durations.instant;  // 0ms - Immediate
motion.durations.fast;     // 120ms - Quick interactions
motion.durations.normal;   // 160ms - Standard
motion.durations.slow;     // 240ms - Entrance animations
motion.durations.slower;   // 320ms - Slower entrance
motion.durations.slowest;  // 500ms - Page transitions
```

#### Easing Functions

```typescript
// Standard easing
motion.easings.linear;     // linear
motion.easings.in;         // ease-in
motion.easings.out;        // ease-out
motion.easings.inOut;      // ease-in-out

// Brand easing (premium)
motion.easings.brand;      // cubic-bezier(0.16, 1, 0.3, 1)

// Advanced easings
motion.easings.easeOutCubic;    // Smooth exit
motion.easings.easeInCubic;     // Smooth entrance
motion.easings.bounce;          // Bouncy effect
motion.easings.easeOutBack;     // Overshoot effect
```

#### Motion Presets

```typescript
import { motionPresets } from '@/design-tokens/motion';

const enterAnimation = motionPresets.enterNormal;
// { duration: '160ms', easing: 'cubic-bezier(0.215, 0.61, 0.355, 1)' }

const exitAnimation = motionPresets.exitQuick;
// { duration: '120ms', easing: 'cubic-bezier(0.645, 0.045, 0.355, 1)' }

const hover = motionPresets.hover;
// { duration: '120ms', easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)' }
```

#### Keyframe Animations

```typescript
import { motion } from '@/design-tokens/motion';

// Available keyframes
motion.keyframes.fadeIn;      // Fade in from 0% to 100%
motion.keyframes.slideInUp;   // Slide up with fade
motion.keyframes.scaleIn;     // Scale from 95% to 100%
motion.keyframes.spin;        // 360 degree rotation
motion.keyframes.pulse;       // Opacity pulse
motion.keyframes.bounce;      // Bouncing animation
motion.keyframes.shimmer;     // Loading shimmer effect
```

#### Create Transitions

```typescript
import { createTransition, createAnimation } from '@/design-tokens/motion';

// CSS transition string
const colorTransition = createTransition('color', 'normal', 'brand');
// Result: "color 160ms cubic-bezier(0.16, 1, 0.3, 1)"

// CSS animation string
const fadeIn = createAnimation('fadeIn', 'normal', 'easeOutCubic', 1);
// Result: "fadeIn 160ms cubic-bezier(0.215, 0.61, 0.355, 1) 1"
```

---

### 7. Breakpoints (`breakpoints.ts`)

Mobile-first responsive design breakpoints.

#### Breakpoint Sizes

```typescript
import { breakpoints } from '@/design-tokens';

breakpoints.xs;   // 320px - Mobile
breakpoints.sm;   // 640px - Large mobile
breakpoints.md;   // 768px - Tablet
breakpoints.lg;   // 1024px - Large tablet
breakpoints.xl;   // 1280px - Desktop
breakpoints['2xl']; // 1536px - Large desktop
```

#### Breakpoint Names & Details

```typescript
import { breakpointDefinitions } from '@/design-tokens/breakpoints';

const mdBreakpoint = breakpointDefinitions.md;
// {
//   name: 'Medium',
//   size: '768px',
//   width: 768,
//   description: 'Tablets (iPad, Android tablets)',
//   devices: ['iPad (10.2")', 'iPad Air', ...]
// }
```

#### Media Queries

```typescript
import { mediaQueries } from '@/design-tokens/breakpoints';

// Min-width queries
mediaQueries.minMd;    // '(min-width: 768px)'
mediaQueries.minLg;    // '(min-width: 1024px)'

// Max-width queries
mediaQueries.maxSm;    // '(max-width: 639px)'
mediaQueries.maxMd;    // '(max-width: 767px)'

// Orientation
mediaQueries.portrait;   // '(orientation: portrait)'
mediaQueries.landscape;  // '(orientation: landscape)'

// Accessibility
mediaQueries.prefersReducedMotion;   // For motion sensitivity
mediaQueries.canHover;               // For hover-capable devices

// Color scheme
mediaQueries.prefersDark;   // Dark mode preference
mediaQueries.prefersLight;  // Light mode preference

// Retina displays
mediaQueries.retina;  // High DPI devices
```

#### Responsive Utilities

```typescript
import {
  getBreakpointName,
  isSmallScreen,
  isMediumScreen,
  isLargeScreen,
  createResponsiveStyle
} from '@/design-tokens/breakpoints';

// Get breakpoint name for width
const name = getBreakpointName(800); // 'md'

// Screen size checks
if (isSmallScreen(320)) { /* ... */ }

// CSS-in-JS responsive styles
const responsive = createResponsiveStyle(
  { fontSize: '14px' },        // Mobile
  { fontSize: '16px' },        // sm+
  { fontSize: '18px' },        // md+
  undefined,                   // lg+ (skip)
  { fontSize: '20px' }         // xl+
);
```

---

### 8. Z-Index (`z-index.ts`)

Stacking context management for layered components.

#### Z-Index Layers

```typescript
import { zIndex } from '@/design-tokens';

zIndex.hide;      // -1 (hidden elements)
zIndex.base;      // 0 (default)
zIndex.docked;    // 10 (sticky elements)
zIndex.dropdown;  // 1000 (dropdown menus)
zIndex.sticky;    // 1020 (sticky headers)
zIndex.banner;    // 1030 (banners)
zIndex.overlay;   // 1040 (backdrop)
zIndex.modal;     // 1050 (modals)
zIndex.popover;   // 1060 (popovers)
zIndex.toast;     // 1080 (toast notifications)
zIndex.tooltip;   // 1090 (tooltips, topmost)
```

#### Z-Index Presets by Component

```typescript
import { zIndexPresets } from '@/design-tokens/z-index';

// Modal stacking
zIndexPresets.modal.backdrop;     // 1040
zIndexPresets.modal.content;      // 1050
zIndexPresets.modal.closeButton;  // 1051

// Notification stacking
zIndexPresets.notification.toast;    // 1080
zIndexPresets.notification.banner;   // 1030

// Dropdown/menu stacking
zIndexPresets.dropdown.backdrop;  // 990
zIndexPresets.dropdown.menu;      // 1000
zIndexPresets.dropdown.submenu;   // 1001
```

#### Z-Index Utilities

```typescript
import {
  getComponentZIndex,
  createLayeredZIndex,
  validateZIndexStack
} from '@/design-tokens/z-index';

// Get z-index for component
const modalZ = getComponentZIndex('modal'); // 1050

// Create layered z-indexes
const layers = createLayeredZIndex('dropdown', 1, 5);
// [1000, 1001, 1002, 1003, 1004]

// Validate no conflicts
const isValid = validateZIndexStack({
  header: 1020,
  modal: 1050,
  tooltip: 1090,
});
```

---

## Usage Examples

### React with Styled Components

```typescript
import styled from 'styled-components';
import { colors, spacing, typography, motion } from '@/design-tokens';

const Button = styled.button`
  background-color: ${colors.gold[300]};
  color: ${colors.charcoal[300]};
  padding: ${spacing[4]} ${spacing[6]};
  border-radius: ${borders.radii.xl};
  font-size: ${typography.fontSizes.base};
  font-weight: ${typography.fontWeights.semibold};
  border: none;
  cursor: pointer;
  transition: ${motion.transitions.all};

  &:hover {
    background-color: ${colors.gold[400]};
    box-shadow: ${shadows.lg};
  }

  &:active {
    transform: scale(0.98);
  }
`;
```

### React with CSS Modules

```typescript
import { tokens } from '@/design-tokens';
import styles from './Button.module.css';

export const Button = () => {
  return (
    <button
      style={{
        backgroundColor: tokens.colors.gold[300],
        padding: `${tokens.spacing[4]} ${tokens.spacing[6]}`,
        borderRadius: tokens.borders.radii.xl,
      }}
      className={styles.button}
    >
      Click me
    </button>
  );
};
```

### Next.js with Tailwind CSS

```jsx
// Automatically available through tailwind.config.ts integration
export default function Card() {
  return (
    <div className="bg-lii-charcoal-300 p-6 rounded-2xl shadow-lg">
      <h2 className="text-lii-cloud-200 text-2xl font-bold">
        Styled with Design Tokens
      </h2>
      <p className="text-lii-ash mt-4">
        Uses Tailwind classes powered by design tokens
      </p>
    </div>
  );
}
```

### Vue with Scoped Styles

```vue
<template>
  <button class="premium-button">
    {{ label }}
  </button>
</template>

<script setup lang="ts">
import { colors, spacing, borders } from '@/design-tokens';

const label = 'Click Me';
</script>

<style scoped>
.premium-button {
  background-color: v-bind('colors.gold[300]');
  padding: v-bind('spacing[4]');
  border-radius: v-bind('borders.radii.xl');
  color: v-bind('colors.charcoal[300]');
  border: none;
  cursor: pointer;
}
</style>
```

### CSS-in-JS with Emotion

```typescript
import { css } from '@emotion/react';
import { colors, spacing, typography } from '@/design-tokens';

const buttonStyles = css`
  background-color: ${colors.gold[300]};
  padding: ${spacing[4]} ${spacing[6]};
  font-size: ${typography.fontSizes.base};

  &:hover {
    background-color: ${colors.gold[400]};
  }
`;

export const Button = () => (
  <button css={buttonStyles}>Click me</button>
);
```

---

## Architecture

### File Structure

```
src/design-tokens/
├── colors.ts              # Color palette & semantic colors
├── typography.ts          # Fonts, sizes, weights, spacing
├── spacing.ts             # Spacing scale & grid system
├── shadows.ts             # Elevation system & shadows
├── borders.ts             # Radius & border width
├── motion.ts              # Durations, easings, keyframes
├── breakpoints.ts         # Responsive breakpoints
├── z-index.ts             # Layer management
├── generate-css.ts        # CSS variable generator
└── index.ts               # Main export & aggregation
```

### Module Dependencies

```
index.ts (main export)
├── colors.ts (independent)
├── typography.ts (independent)
├── spacing.ts (independent)
├── shadows.ts (references colors)
├── borders.ts (independent)
├── motion.ts (independent)
├── breakpoints.ts (independent)
├── z-index.ts (independent)
└── generate-css.ts (aggregates all tokens)
```

### Export Structure

```typescript
// Individual exports
export { colors } from './colors';
export { typography } from './typography';
// ... more

// Aggregated object
export const tokens = {
  colors,
  typography,
  spacing,
  sizes,
  shadows,
  borders,
  motion,
  breakpoints,
  zIndex,
} as const;

// Type exports
export type DesignTokens = typeof tokens;
```

---

## TypeScript Support

### Full Type Safety

```typescript
import type {
  Colors,
  Typography,
  Spacing,
  SpacingKey,
  FontSize,
  DesignTokens,
} from '@/design-tokens';

// Type-safe token access
const color: string = colors.gold[300];
const fontSize: FontSize = 'base';
const spacing: SpacingKey = 4;

// Type inference
const tokens: DesignTokens = { /* ... */ };
```

### Custom Type Guards

```typescript
import type { ColorToken } from '@/design-tokens/colors';

function isValidColor(value: any): value is ColorToken {
  return typeof value === 'string' && value.startsWith('#');
}

if (isValidColor(colors.gold[300])) {
  // Safe to use as color
}
```

---

## Tailwind Integration

### Automatic Integration

The design tokens are automatically integrated into Tailwind CSS via `tailwind.config.ts`:

```typescript
// All these are available:
<div className="text-lii-gold-300 p-4 rounded-xl shadow-md">
  Tailwind classes using design tokens
</div>
```

### Available Tailwind Classes

#### Colors
- `text-lii-gold-{50-900}`
- `bg-lii-gold-{50-900}`
- `border-lii-gold-{50-900}`
- `text-lii-charcoal-{50-500}`
- `text-lii-success-{50-900}`
- `text-lii-error-{50-900}`

#### Spacing
- `p-0` through `p-32`
- `m-0` through `m-32`
- `gap-0` through `gap-32`

#### Border Radius
- `rounded-none`, `rounded-sm`, `rounded-md`, `rounded-lg`
- `rounded-xl`, `rounded-2xl`, `rounded-3xl`, `rounded-full`

#### Shadows
- `shadow-none`, `shadow-xs`, `shadow-sm`, `shadow-md`
- `shadow-lg`, `shadow-xl`, `shadow-2xl`

#### Motion
- `duration-instant` through `duration-slowest`
- `ease-linear`, `ease-in`, `ease-out`, `ease-in-out`, `ease-brand`

---

## CSS Variables

### Auto-Generated CSS Variables

All design tokens are automatically converted to CSS custom properties:

```css
/* Colors */
--lii-color-gold-300: #C1A060;
--lii-color-charcoal-300: #0B0B0C;
--lii-color-success-500: #10B981;

/* Typography */
--lii-font-family-display: 'Playfair Display', serif;
--lii-font-size-base: 1rem;
--lii-font-weight-bold: 700;

/* Spacing */
--lii-spacing-4: 1rem;

/* Shadows */
--lii-shadow-md: 0 4px 6px -1px rgba(193, 160, 96, 0.1), ...;

/* Breakpoints */
--lii-breakpoint-md: 768px;

/* Z-Index */
--lii-z-modal: 1050;
```

### Usage in CSS

```css
.card {
  background-color: var(--lii-color-charcoal-300);
  padding: var(--lii-spacing-6);
  border-radius: var(--lii-radius-2xl);
  box-shadow: var(--lii-shadow-lg);
  color: var(--lii-color-cloud-200);
}
```

### Generating CSS Files

```bash
# Generate all CSS variable files
npx ts-node src/design-tokens/generate-css.ts

# Outputs:
# - tokens.css (all variables)
# - tokens-media.css (responsive overrides)
# - tokens-dark.css (dark mode)
# - utilities.css (utility classes)
```

---

## Migration Guide

### From Legacy Theme System

#### Old (src/theme/tokens.ts)
```typescript
import { tokens } from '@/theme/tokens';

const color = tokens.colors.gold; // String
const spacing = tokens.spacing[4]; // String
```

#### New (src/design-tokens)
```typescript
import { tokens, colors, spacing } from '@/design-tokens';

// Modular access
const color = colors.gold[300];
const sp = spacing[4];

// Same functionality with better organization
```

### Parallel Migration Strategy

Both systems work simultaneously:

```typescript
// Can use both during transition
import { tokens as legacyTokens } from '@/theme/tokens';
import { colors, spacing } from '@/design-tokens';

// Migrate gradually, component by component
```

### Breaking Changes

None! The new design tokens system is fully backward compatible with Tailwind configuration.

---

## Best Practices

### 1. Always Use Token Values

```typescript
// GOOD - Uses design token
const padding = spacing[4];

// AVOID - Magic number
const padding = '16px';
```

### 2. Leverage Type Safety

```typescript
import type { SpacingKey } from '@/design-tokens/spacing';

function getPadding(size: SpacingKey): string {
  return spacing[size];
}

// Type-safe function usage
getPadding(4); // OK
getPadding(999); // TypeScript error
```

### 3. Use Semantic Colors

```typescript
// GOOD - Semantic meaning
const errorColor = colors.error[500];

// AVOID - Just picking a color
const errorColor = colors.gold[300];
```

### 4. Respect Component Hierarchy

```typescript
// GOOD - Component specific z-index
const modalZ = zIndex.modal;

// AVOID - Arbitrary z-index
const modalZ = 9999;
```

### 5. Use Motion Presets

```typescript
// GOOD - Consistent timing
const style = {
  transition: motion.transitions.all,
};

// AVOID - Custom timing
const style = {
  transition: 'all 0.25s ease-out',
};
```

### 6. Responsive Design Patterns

```typescript
// GOOD - Mobile-first
className="p-4 md:p-6 lg:p-8"

// AVOID - Desktop-first
className="p-8 max-md:p-4"
```

### 7. Document Token Usage

```typescript
/**
 * Card component using design tokens
 * - Color: lii-gold for primary actions
 * - Shadow: elevation 3 (md) for depth
 * - Radius: 2xl for luxury aesthetic
 */
const Card = () => { /* ... */ };
```

---

## Contributing

### Adding New Tokens

1. Identify the token category (color, spacing, etc.)
2. Add to appropriate file in `src/design-tokens/`
3. Update type exports
4. Add to documentation
5. Test Tailwind integration

### Token Review Checklist

- [ ] Token follows naming conventions
- [ ] Value is mathematically consistent with scale
- [ ] TypeScript types are correct
- [ ] Documentation is updated
- [ ] Works in Tailwind CSS
- [ ] Backward compatible

### Naming Conventions

```typescript
// Colors
colors.[palette].[shade]  // colors.gold[300]

// Spacing
spacing.[scale]           // spacing[4]

// Border Radius
borders.radii.[size]      // borders.radii.xl

// Motion
motion.durations.[speed]  // motion.durations.normal
motion.easings.[type]     // motion.easings.brand

// Z-Index
zIndex.[layer]            // zIndex.modal
```

---

## FAQ

### Q: How do I update a token value globally?

Edit the specific token file in `/src/design-tokens/`, and it automatically updates everywhere the token is imported.

### Q: Can I use tokens with Figma/design tools?

Yes! Export tokens as JSON using `exportTokensAsJSON()` and import into design tool plugins.

### Q: How do I add dark mode variants?

Extend the theme in `tailwind.config.ts` or use CSS variables with `prefers-color-scheme` media queries.

### Q: Performance impact of modular tokens?

Negligible. Tree-shaking removes unused tokens, and CSS variables are native browser features with zero runtime cost.

### Q: Can I override tokens in a specific context?

Yes, use CSS variables or Tailwind's arbitrary value syntax:

```jsx
<div style={{ '--lii-color-primary': 'red' }}>
  Custom color
</div>

{/* or */}
<div className="[--lii-color-primary:red]">
  Custom color
</div>
```

---

## Support & Resources

- **Documentation**: This file
- **Source Code**: `/src/design-tokens/`
- **Configuration**: `/tailwind.config.ts`
- **Type Definitions**: Auto-exported from each token file
- **Design System**: Live It Iconic Premium Lifestyle Brand

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainer**: Live It Iconic Design System Team
