# Design Tokens - Usage Examples

Quick reference guide with practical examples for using Live It Iconic design tokens across different frameworks and patterns.

---

## Table of Contents

1. [Basic Import Patterns](#basic-import-patterns)
2. [React Examples](#react-examples)
3. [Vue Examples](#vue-examples)
4. [Tailwind CSS Examples](#tailwind-css-examples)
5. [CSS Modules Examples](#css-modules-examples)
6. [Styled Components Examples](#styled-components-examples)
7. [Emotion Examples](#emotion-examples)
8. [TypeScript Patterns](#typescript-patterns)
9. [Advanced Patterns](#advanced-patterns)

---

## Basic Import Patterns

### Import Everything

```typescript
import { tokens } from '@/design-tokens';

// Access nested tokens
tokens.colors.gold[300];        // '#C1A060'
tokens.spacing[4];              // '1rem'
tokens.typography.fontSizes.base; // '1rem'
```

### Import Specific Modules

```typescript
import { colors, spacing, typography } from '@/design-tokens';

// Cleaner imports for common usage
const color = colors.gold[300];
const space = spacing[4];
const font = typography.fontSizes.base;
```

### Import Type Utilities

```typescript
import type {
  Colors,
  Spacing,
  SpacingKey,
  DesignTokens,
} from '@/design-tokens';

const value: SpacingKey = 4;
```

---

## React Examples

### Functional Component with Inline Styles

```typescript
import { colors, spacing, typography } from '@/design-tokens';

export const Button = ({ children }: { children: React.ReactNode }) => {
  return (
    <button
      style={{
        backgroundColor: colors.gold[300],
        color: colors.charcoal[300],
        padding: `${spacing[3]} ${spacing[6]}`,
        fontSize: typography.fontSizes.base,
        fontWeight: typography.fontWeights.semibold,
        border: 'none',
        borderRadius: borders.radii.xl,
        cursor: 'pointer',
        transition: motion.transitions.all,
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.backgroundColor = colors.gold[400];
        (e.currentTarget as HTMLButtonElement).style.boxShadow = shadows.lg;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.backgroundColor = colors.gold[300];
        (e.currentTarget as HTMLButtonElement).style.boxShadow = shadows.sm;
      }}
    >
      {children}
    </button>
  );
};
```

### React with Context for Theme

```typescript
import { createContext, useContext } from 'react';
import { tokens } from '@/design-tokens';

type Theme = typeof tokens;

const ThemeContext = createContext<Theme>(tokens);

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => (
  <ThemeContext.Provider value={tokens}>
    {children}
  </ThemeContext.Provider>
);

export const useTheme = (): Theme => {
  const theme = useContext(ThemeContext);
  if (!theme) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return theme;
};

// Usage
const Card = () => {
  const theme = useTheme();
  return (
    <div style={{
      padding: theme.spacing[6],
      backgroundColor: theme.colors.charcoal[400],
      borderRadius: theme.borders.radii['2xl'],
    }}>
      Content
    </div>
  );
};
```

### React Hook for Responsive Values

```typescript
import { useEffect, useState } from 'react';
import { breakpoints, getBreakpointName } from '@/design-tokens';

export const useResponsiveValue = <T,>(
  mobileValue: T,
  tabletValue?: T,
  desktopValue?: T
): T => {
  const [value, setValue] = useState(mobileValue);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const breakpointName = getBreakpointName(width);

      if (breakpointName === 'md' || breakpointName === 'lg') {
        setValue(tabletValue ?? mobileValue);
      } else if (breakpointName === 'xl' || breakpointName === '2xl') {
        setValue(desktopValue ?? mobileValue);
      } else {
        setValue(mobileValue);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [mobileValue, tabletValue, desktopValue]);

  return value;
};

// Usage
const PaddingExample = () => {
  const padding = useResponsiveValue(spacing[4], spacing[6], spacing[8]);
  return <div style={{ padding }}>Responsive padding</div>;
};
```

### React Component Library Pattern

```typescript
import { colors, spacing, borders } from '@/design-tokens';

interface CardProps {
  variant?: 'default' | 'elevated' | 'outlined';
  children: React.ReactNode;
}

const cardVariants = {
  default: {
    backgroundColor: colors.charcoal[400],
    border: `1px solid ${colors.gold[300]}40`,
    boxShadow: shadows.md,
  },
  elevated: {
    backgroundColor: colors.charcoal[300],
    border: 'none',
    boxShadow: shadows.xl,
  },
  outlined: {
    backgroundColor: 'transparent',
    border: `2px solid ${colors.gold[300]}`,
    boxShadow: 'none',
  },
};

export const Card = ({ variant = 'default', children }: CardProps) => {
  return (
    <div
      style={{
        padding: spacing[6],
        borderRadius: borders.radii['2xl'],
        ...cardVariants[variant],
      }}
    >
      {children}
    </div>
  );
};
```

---

## Vue Examples

### Vue 3 Composition API

```vue
<template>
  <button
    :style="buttonStyles"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
  >
    {{ label }}
  </button>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { colors, spacing, typography, borders, motion } from '@/design-tokens';

const label = 'Click me';
const isHovered = ref(false);

const buttonStyles = computed(() => ({
  backgroundColor: isHovered.value ? colors.gold[400] : colors.gold[300],
  color: colors.charcoal[300],
  padding: `${spacing[3]} ${spacing[6]}`,
  fontSize: typography.fontSizes.base,
  fontWeight: typography.fontWeights.semibold,
  border: 'none',
  borderRadius: borders.radii.xl,
  cursor: 'pointer',
  transition: motion.transitions.all,
}));
</script>
```

### Vue with Custom CSS Variables

```vue
<template>
  <div class="themed-container">
    <h1>Styled with CSS Variables</h1>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { colors, spacing } from '@/design-tokens';

onMounted(() => {
  // Set CSS variables dynamically
  document.documentElement.style.setProperty('--theme-color', colors.gold[300]);
  document.documentElement.style.setProperty('--theme-padding', spacing[6]);
});
</script>

<style scoped>
.themed-container {
  background-color: var(--theme-color);
  padding: var(--theme-padding);
  border-radius: 1rem;
}
</style>
```

---

## Tailwind CSS Examples

### Basic Class Usage

```jsx
// Colors
<div className="text-lii-gold-300">Gold text</div>
<div className="bg-lii-charcoal-300">Dark background</div>
<div className="border-lii-cloud-200">Cloud border</div>

// Spacing
<div className="p-6 m-4 gap-4">Padded, margined, with gap</div>

// Border radius
<div className="rounded-xl">Rounded corners</div>
<div className="rounded-full">Fully rounded</div>

// Shadows
<div className="shadow-lg">Elevated shadow</div>
<div className="shadow-2xl">High elevation</div>

// Typography
<h1 className="text-5xl font-bold text-lii-cloud-200">Heading</h1>
<p className="text-base font-normal text-lii-ash">Body text</p>

// Motion
<div className="duration-normal ease-brand transition-all">Smooth transition</div>
```

### Responsive Classes

```jsx
<div className="
  p-4 gap-2                          // Mobile
  sm:p-5 sm:gap-3                    // Small screens
  md:p-6 md:gap-4                    // Medium tablets
  lg:p-8 lg:gap-6                    // Large screens
  xl:p-12 xl:gap-8                   // Desktops
">
  Responsive spacing
</div>
```

### Complex Component

```jsx
export const PremiumCard = () => {
  return (
    <div className="
      bg-lii-charcoal-400
      border border-lii-gold-300 border-opacity-20
      rounded-2xl
      p-8
      shadow-lg
      hover:shadow-2xl
      transition-all
      duration-300
      ease-brand
    ">
      <h2 className="
        text-3xl
        font-bold
        text-lii-cloud-200
        mb-4
      ">
        Premium Content
      </h2>
      <p className="
        text-base
        text-lii-ash
        leading-relaxed
      ">
        Styled with design tokens through Tailwind CSS
      </p>
      <button className="
        mt-6
        px-6 py-3
        bg-lii-gold-300
        text-lii-charcoal-300
        rounded-xl
        font-semibold
        hover:bg-lii-gold-400
        transition-colors
        duration-200
      ">
        Action
      </button>
    </div>
  );
};
```

---

## CSS Modules Examples

### TypeScript with CSS Modules

```typescript
// Button.tsx
import { colors, spacing, typography, borders } from '@/design-tokens';
import styles from './Button.module.css';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
}

export const Button = ({ children, variant = 'primary' }: ButtonProps) => {
  const baseStyle = {
    padding: `${spacing[3]} ${spacing[6]}`,
    fontSize: typography.fontSizes.base,
    fontWeight: typography.fontWeights.semibold,
    border: 'none',
    borderRadius: borders.radii.xl,
    cursor: 'pointer',
    transition: 'all 0.3s ease-out',
  };

  const variantStyles = {
    primary: {
      backgroundColor: colors.gold[300],
      color: colors.charcoal[300],
    },
    secondary: {
      backgroundColor: 'transparent',
      color: colors.gold[300],
      border: `1.5px solid ${colors.gold[300]}`,
    },
  };

  return (
    <button
      style={{ ...baseStyle, ...variantStyles[variant] }}
      className={styles.button}
    >
      {children}
    </button>
  );
};
```

```css
/* Button.module.css */
.button:hover {
  transform: translateY(-2px);
}

.button:active {
  transform: translateY(0);
}

@media (max-width: 768px) {
  .button {
    width: 100%;
  }
}
```

---

## Styled Components Examples

```typescript
import styled from 'styled-components';
import { colors, spacing, typography, borders, shadows, motion } from '@/design-tokens';

// Basic styled component
export const StyledButton = styled.button`
  background-color: ${colors.gold[300]};
  color: ${colors.charcoal[300]};
  padding: ${spacing[3]} ${spacing[6]};
  font-size: ${typography.fontSizes.base};
  font-weight: ${typography.fontWeights.semibold};
  border: none;
  border-radius: ${borders.radii.xl};
  cursor: pointer;
  box-shadow: ${shadows.sm};
  transition: ${motion.transitions.all};

  &:hover {
    background-color: ${colors.gold[400]};
    box-shadow: ${shadows.lg};
    transform: translateY(-2px);
  }

  &:active {
    transform: translateY(0);
  }
`;

// Component with props
interface CardProps {
  $elevation?: 'sm' | 'md' | 'lg';
}

export const StyledCard = styled.div<CardProps>`
  background-color: ${colors.charcoal[400]};
  padding: ${spacing[6]};
  border-radius: ${borders.radii['2xl']};
  box-shadow: ${(props) => {
    const shadowMap = {
      sm: shadows.sm,
      md: shadows.md,
      lg: shadows.lg,
    };
    return shadowMap[props.$elevation || 'md'];
  }};
  border: 1px solid ${colors.gold[300]}40;

  &:hover {
    box-shadow: ${shadows.xl};
    transition: ${motion.transitions.all};
  }
`;

// Global styled components
export const GlobalContainer = styled.div`
  max-width: ${sizes.xl};
  margin: 0 auto;
  padding: ${spacing[6]};
`;
```

---

## Emotion Examples

```typescript
import { css } from '@emotion/react';
import styled from '@emotion/styled';
import { colors, spacing, typography, borders } from '@/design-tokens';

// CSS with emotion
const buttonStyles = css`
  background-color: ${colors.gold[300]};
  padding: ${spacing[3]} ${spacing[6]};
  border-radius: ${borders.radii.xl};
  border: none;
  cursor: pointer;

  &:hover {
    background-color: ${colors.gold[400]};
  }
`;

export const Button = () => (
  <button css={buttonStyles}>Click me</button>
);

// Styled component with emotion
const Card = styled.div`
  background-color: ${colors.charcoal[400]};
  padding: ${spacing[6]};
  border-radius: ${borders.radii['2xl']};
`;

export const CardComponent = () => (
  <Card>
    <h2 css={{ color: colors.cloud[200] }}>Title</h2>
  </Card>
);
```

---

## TypeScript Patterns

### Type-Safe Token Access

```typescript
import type {
  SpacingKey,
  ColorToken,
  DesignTokens,
} from '@/design-tokens';
import { colors, spacing } from '@/design-tokens';

// Type-safe spacing
const getPadding = (size: SpacingKey): string => {
  return spacing[size];
};

// Valid
getPadding(4);     // ✓
getPadding(8);     // ✓

// Type Error
getPadding(999);   // ✗ TypeScript error

// Type-safe color
type BrandColor = 'gold' | 'charcoal' | 'cloud';

const getBrandColor = (name: BrandColor, shade: keyof typeof colors.gold): string => {
  return colors[name][shade] as string;
};

// Usage
const color = getBrandColor('gold', 300);  // ✓ Fully type-safe
```

### Generic Component with Tokens

```typescript
import { colors, spacing } from '@/design-tokens';
import type { SpacingKey } from '@/design-tokens/spacing';
import type { Colors } from '@/design-tokens/colors';

interface StyledElementProps {
  padding?: SpacingKey;
  color?: keyof typeof colors;
  shade?: number;
}

const StyledElement = ({
  padding = 4,
  color = 'cloud',
  shade = 200,
}: StyledElementProps) => {
  // Type-safe access
  const colorValue = (colors[color as keyof Colors] as any)?.[shade];
  const paddingValue = spacing[padding];

  return (
    <div style={{ padding: paddingValue, color: colorValue }}>
      Content
    </div>
  );
};
```

---

## Advanced Patterns

### Token Theme Manager

```typescript
import { tokens } from '@/design-tokens';

class ThemeManager {
  private theme = tokens;
  private overrides: Partial<typeof tokens> = {};

  getColor(path: string): string {
    const [category, ...rest] = path.split('.');
    let value: any = this.theme[category as keyof typeof tokens];

    for (const key of rest) {
      value = value?.[key];
    }

    return value || '#000000';
  }

  setOverride(path: string, value: any): void {
    const [category, ...rest] = path.split('.');
    if (!this.overrides[category as keyof typeof tokens]) {
      this.overrides[category as keyof typeof tokens] = {};
    }
    // ... set nested value
  }

  reset(): void {
    this.overrides = {};
  }
}

// Usage
const themeManager = new ThemeManager();
const goldColor = themeManager.getColor('colors.gold.300');
```

### Responsive Styles Factory

```typescript
import { breakpoints, spacing } from '@/design-tokens';

interface ResponsiveValue<T> {
  xs?: T;
  sm?: T;
  md?: T;
  lg?: T;
  xl?: T;
  '2xl'?: T;
}

const createResponsiveStyle = <T>(
  property: string,
  values: ResponsiveValue<T>
): Record<string, any> => {
  const result: Record<string, any> = {};

  if (values.xs) result[property] = values.xs;

  if (values.sm) {
    result[`@media (min-width: ${breakpoints.sm})`] = {
      [property]: values.sm,
    };
  }

  if (values.md) {
    result[`@media (min-width: ${breakpoints.md})`] = {
      [property]: values.md,
    };
  }

  // ... continue for other breakpoints

  return result;
};

// Usage
const responsivePadding = createResponsiveStyle('padding', {
  xs: spacing[2],
  md: spacing[4],
  lg: spacing[6],
  xl: spacing[8],
});
```

### Token Validation Hook

```typescript
import { useEffect } from 'react';
import { tokens } from '@/design-tokens';

export const useTokenValidation = () => {
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      // Validate all tokens are accessible
      try {
        Object.keys(tokens).forEach((category) => {
          const categoryTokens = tokens[category as keyof typeof tokens];
          if (!categoryTokens) {
            console.warn(`Missing token category: ${category}`);
          }
        });
        console.log('✓ All design tokens validated');
      } catch (error) {
        console.error('Design token validation failed:', error);
      }
    }
  }, []);
};
```

---

## Performance Tips

### Memoize Token Calculations

```typescript
import { useMemo } from 'react';
import { colors, spacing } from '@/design-tokens';

const Button = ({ variant }: { variant: 'primary' | 'secondary' }) => {
  const styles = useMemo(
    () => ({
      primary: {
        bg: colors.gold[300],
        padding: spacing[4],
      },
      secondary: {
        bg: colors.gold[200],
        padding: spacing[3],
      },
    }),
    []
  );

  return <button style={styles[variant]}>Button</button>;
};
```

### Cache Token References

```typescript
import { colors, spacing, typography } from '@/design-tokens';

// Extract common values
const COMMON_TOKENS = {
  primaryColor: colors.gold[300],
  defaultPadding: spacing[4],
  baseFont: typography.fontSizes.base,
} as const;

// Reuse throughout app
export const Button = () => (
  <button style={{ color: COMMON_TOKENS.primaryColor }}>
    Button
  </button>
);
```

---

## Testing with Tokens

```typescript
import { render, screen } from '@testing-library/react';
import { colors, spacing } from '@/design-tokens';
import { Button } from './Button';

describe('Button component', () => {
  it('applies gold color from design tokens', () => {
    const { container } = render(<Button>Click me</Button>);
    const button = container.querySelector('button');

    expect(button).toHaveStyle({
      backgroundColor: colors.gold[300],
    });
  });

  it('applies correct spacing', () => {
    const { container } = render(<Button>Click me</Button>);
    const button = container.querySelector('button');

    expect(button).toHaveStyle({
      padding: `${spacing[3]} ${spacing[6]}`,
    });
  });
});
```

---

## Need More Examples?

See the full documentation in `/docs/DESIGN_TOKENS.md` or explore the token files directly in `/src/design-tokens/`.

