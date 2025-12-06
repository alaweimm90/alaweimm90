# Enhanced Quantum Theme System

## Overview

QMLab's enhanced theme system provides sophisticated visual effects, advanced animations, and rich micro-interactions for a premium quantum computing interface experience.

## New Features

### 1. Enhanced Color System (`enhanced-colors.css`)

**Extended Color Palettes**
- 11-step primary palette (50-950)
- 11-step secondary palette (50-950)
- Accent colors: cyan, emerald, amber, rose, violet

**Quantum Gradients**
```css
--gradient-quantum-blue      /* Blue gradient */
--gradient-quantum-purple    /* Purple gradient */
--gradient-quantum-rainbow   /* Multi-color spectrum */
--gradient-quantum-aurora    /* Aurora effect */
--gradient-quantum-sunrise   /* Warm sunrise */
--gradient-quantum-ocean     /* Ocean depth */
--gradient-quantum-forest    /* Forest green */
```

**Usage**
```tsx
<div className="bg-gradient-quantum-aurora">
  Quantum Aurora Background
</div>

<h1 className="text-glow-strong">
  Glowing Quantum Text
</h1>
```

**Quantum Shadows**
```css
--shadow-quantum-sm          /* Subtle shadow */
--shadow-quantum-md          /* Medium shadow */
--shadow-quantum-lg          /* Large shadow */
--shadow-quantum-xl          /* Extra large */
--shadow-quantum-glow        /* Glow effect */
--shadow-quantum-glow-strong /* Strong glow */
```

### 2. Advanced Animations (`animations.css`)

**New Keyframes**
- `quantum-shimmer` - Flowing shimmer effect
- `quantum-wave` - Gentle wave motion
- `quantum-particle` - Particle movement
- `quantum-ripple` - Ripple expansion
- `quantum-breathe` - Breathing animation
- `quantum-slide-up/down` - Smooth entrance

**Animation Classes**
```tsx
<div className="animate-quantum-shimmer">
  Shimmering element
</div>

<div className="animate-quantum-float">
  Floating element
</div>

<div className="animate-quantum-breathe">
  Breathing element
</div>
```

**Hover Effects**
```tsx
<button className="hover-lift">
  Lifts on hover
</button>

<div className="hover-glow">
  Glows on hover
</div>

<img className="hover-scale">
  Scales on hover
</img>
```

### 3. Micro-Interactions (`micro-interactions.css`)

**Button Interactions**
```tsx
<button className="btn-quantum">
  Quantum Button with ripple effect
</button>

<button className="ripple-quantum">
  Click for ripple
</button>
```

**Card Interactions**
```tsx
<div className="card-quantum">
  Card with rainbow border on hover
</div>

<div className="glow-on-hover">
  Glows when hovered
</div>
```

**Input Interactions**
```tsx
<input className="input-quantum" />
{/* Auto shimmer on focus */}
```

**Link Interactions**
```tsx
<a className="link-quantum">
  Animated underline
</a>
```

**Tooltip System**
```tsx
<span className="tooltip-quantum" data-tooltip="Helpful info">
  Hover me
</span>
```

**Badge System**
```tsx
<span className="badge-quantum">
  Status: Active
</span>
```

**Progress Bars**
```tsx
<div className="progress-quantum">
  <div 
    className="progress-quantum-bar" 
    style={{ width: '60%' }}
  />
</div>
```

**Skeleton Loading**
```tsx
<div className="skeleton-quantum" style={{ height: 120 }} />
```

### 4. Enhanced Theme Config

**New Configuration Options**
```typescript
effects: {
  blur: {
    subtle: '8px',
    medium: '12px',
    strong: '20px',
  },
  glow: {
    subtle: '0 0 20px hsl(217 91% 60% / 0.2)',
    medium: '0 0 40px hsl(217 91% 60% / 0.3)',
    strong: '0 0 60px hsl(217 91% 60% / 0.4)',
  },
}
```

## Design Patterns

### Quantum Cards

```tsx
<div className="card-quantum p-6 rounded-xl hover-lift">
  <h3 className="text-glow-subtle mb-4">Quantum Feature</h3>
  <p>Content with elegant hover effects</p>
</div>
```

### Quantum Buttons

```tsx
<button className="btn-quantum px-6 py-3 rounded-lg bg-gradient-quantum-blue">
  Execute Quantum Circuit
</button>
```

### Quantum Inputs

```tsx
<input 
  type="text"
  className="input-quantum px-4 py-2 rounded-lg"
  placeholder="Enter quantum state..."
/>
```

### Quantum Badges

```tsx
<span className="badge-quantum">
  Entangled State
</span>
```

### Progress Indicators

```tsx
<div className="progress-quantum h-2 rounded-full">
  <div 
    className="progress-quantum-bar"
    style={{ width: `${progress}%` }}
  />
</div>
```

## Accessibility

All interactions respect `prefers-reduced-motion`:

```css
@media (prefers-reduced-motion: reduce) {
  /* Animations disabled */
  /* Transforms neutralized */
}
```

Users with motion sensitivity will experience:
- No animations
- Instant state changes
- Static hover effects

## Performance

**GPU Acceleration**
All animations use GPU-accelerated properties:
- `transform`
- `opacity`
- `filter`

**Optimizations**
- `will-change` used sparingly
- Animations on composite layers
- Minimal repaints/reflows

## Best Practices

### 1. Layer Visual Effects

```tsx
<div className="card-quantum">
  <div className="hover-lift">
    <div className="glow-on-hover">
      Layered effects
    </div>
  </div>
</div>
```

### 2. Combine Gradients with Animations

```tsx
<div className="bg-gradient-quantum-aurora animate-quantum-breathe">
  Animated gradient background
</div>
```

### 3. Use Semantic Glow

```tsx
{/* Success state */}
<div className="shadow-quantum-glow border-emerald-500">
  Success!
</div>

{/* Active state */}
<div className="shadow-quantum-glow-strong">
  Active quantum state
</div>
```

### 4. Progressive Enhancement

```tsx
{/* Base styles work without enhanced theme */}
<button className="btn-quantum bg-blue-600 hover:bg-blue-700">
  Works with or without enhancements
</button>
```

## Migration Guide

### From Basic to Enhanced

**Before:**
```tsx
<button className="bg-blue-600 hover:bg-blue-700">
  Button
</button>
```

**After:**
```tsx
<button className="btn-quantum bg-gradient-quantum-blue">
  Button
</button>
```

**Before:**
```tsx
<div className="shadow-lg">
  Card
</div>
```

**After:**
```tsx
<div className="card-quantum shadow-quantum-glow">
  Card
</div>
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

All modern browsers with CSS Grid, CSS Custom Properties, and backdrop-filter support.

## Examples

### Quantum Dashboard Card

```tsx
<div className="card-quantum p-6 rounded-xl bg-gradient-quantum-ocean/10">
  <div className="flex items-center gap-4">
    <div className="badge-quantum">Active</div>
    <h3 className="text-glow-subtle">Quantum Circuit</h3>
  </div>
  
  <div className="mt-4 progress-quantum">
    <div className="progress-quantum-bar" style={{ width: '75%' }} />
  </div>
  
  <button className="btn-quantum mt-4 w-full py-2 rounded-lg">
    Execute
  </button>
</div>
```

### Quantum Form

```tsx
<form className="space-y-4">
  <input 
    className="input-quantum w-full px-4 py-3 rounded-lg"
    placeholder="Circuit name..."
  />
  
  <button 
    type="submit"
    className="btn-quantum w-full py-3 rounded-lg bg-gradient-quantum-blue"
  >
    Create Circuit
  </button>
</form>
```

## Resources

- [Animations Documentation](./animations.md)
- [Color System Documentation](./colors.md)
- [Component Library](./components.md)
- [Accessibility Guidelines](./accessibility.md)
