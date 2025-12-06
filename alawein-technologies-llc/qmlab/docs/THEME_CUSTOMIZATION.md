# Theme Customization Guide

## Overview

QMLab's theme system is centralized and highly customizable. All theme settings can be modified in one place to change the entire application's appearance.

## Quick Start

To customize the theme, edit `src/config/theme.config.ts`:

```typescript
export const defaultTheme: ThemeConfig = {
  name: 'My Custom Theme',
  colors: {
    primary: '#3b82f6',      // Change primary color
    secondary: '#a855f7',    // Change secondary color
    // ... other colors
  },
  // ... other theme settings
};
```

## Theme Structure

### Colors

Located in `src/styles/design-system/colors.css` and `src/config/theme.config.ts`:

- **Primary**: Main brand color (blue by default)
- **Secondary**: Accent color (purple by default)
- **Semantic**: Success, warning, danger, info
- **Surface**: Background layers
- **Text**: Primary, secondary, muted, disabled

All colors use HSL format for easy manipulation:
```css
--quantum-blue-500: 217 91 60; /* HSL values */
```

### Typography

Configure fonts in `src/config/theme.config.ts`:

```typescript
typography: {
  fontFamily: {
    sans: 'Inter, system-ui, sans-serif',
    mono: 'Fira Code, monospace',
  },
  fontSize: {
    base: '16px',
    scale: 1.25, // Modular scale ratio
  },
}
```

To add custom fonts, update `index.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=YourFont:wght@400;700&display=swap" rel="stylesheet">
```

### Spacing

Based on a 4px unit system:

```typescript
spacing: {
  base: 4,
  scale: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24],
}
```

Multiplied by base: `--space-4` = 16px

### Border Radius

Consistent rounding system:

```typescript
borderRadius: {
  base: '0.5rem',
  scale: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    full: '9999px',
  },
}
```

### Animations

Configure timing and easing:

```typescript
animation: {
  duration: {
    fast: '150ms',
    normal: '250ms',
    slow: '350ms',
  },
  easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
}
```

## Creating a New Theme

1. **Copy the default theme** in `src/config/theme.config.ts`:

```typescript
export const darkTheme: ThemeConfig = {
  name: 'Dark Mode',
  colors: {
    primary: '#60a5fa',
    // ... customize
  },
  // ... other settings
};
```

2. **Update color tokens** in `src/styles/design-system/colors.css`:

```css
:root[data-theme="dark"] {
  --quantum-blue-500: 96 165 250;
  /* ... other colors */
}
```

3. **Switch active theme**:

```typescript
export const activeTheme = darkTheme;
```

## Dark Mode

Dark mode is built-in. Toggle via:

```tsx
<html data-theme="dark">
```

Or programmatically:

```typescript
document.documentElement.setAttribute('data-theme', 'dark');
```

## Advanced Customization

### Glass Morphism

Adjust glass effects in `src/styles/design-system/effects.css`:

```css
--glass-bg: rgba(15, 23, 42, 0.8);
--glass-blur: 12px;
```

### Shadows

Customize shadow system:

```css
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
```

### Gradients

Modify quantum gradients:

```css
--gradient-primary: linear-gradient(135deg, 
  hsl(var(--quantum-blue-500)), 
  hsl(var(--quantum-purple-500))
);
```

## Best Practices

1. **Use semantic tokens**: Prefer `var(--color-primary)` over direct hex values
2. **Test contrast**: Ensure WCAG AA compliance (4.5:1 for text)
3. **Test dark mode**: Verify theme in both light and dark modes
4. **Use HSL**: Easier to manipulate (lightness, saturation)
5. **Document changes**: Note custom theme modifications

## Troubleshooting

### Colors not updating?
- Check CSS variable names match
- Verify HSL format (no `hsl()` wrapper in variables)
- Clear browser cache

### Font not loading?
- Check font URL in `index.html`
- Verify font name in theme config
- Check network tab for 404s

### Dark mode issues?
- Ensure `data-theme` attribute is set
- Check color contrast in dark mode
- Test with system preferences

## Examples

### Brand Color Change

```typescript
// In theme.config.ts
colors: {
  primary: '#f59e0b', // Orange
  primaryHover: '#d97706',
  primaryActive: '#b45309',
}
```

### Typography Change

```typescript
// In theme.config.ts
typography: {
  fontFamily: {
    sans: 'Roboto, sans-serif',
    mono: 'Source Code Pro, monospace',
  },
}
```

### Spacing Adjustment

```typescript
// In theme.config.ts
spacing: {
  base: 8, // 8px instead of 4px
  scale: [0, 1, 2, 3, 4, 6, 8, 12],
}
```
