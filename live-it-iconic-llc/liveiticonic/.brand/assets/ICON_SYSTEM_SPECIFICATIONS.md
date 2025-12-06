# Live It Iconic - Icon System Specifications

**Version:** 1.0
**Date:** November 2024
**Purpose:** Complete icon library for website, app, and digital applications

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Icon Grid System](#icon-grid-system)
3. [Icon Categories](#icon-categories)
4. [Technical Specifications](#technical-specifications)
5. [Usage Guidelines](#usage-guidelines)
6. [Implementation](#implementation)

---

## Design Principles

### Visual Language

**Style:** Minimalist line icons with optional fill variants

**Characteristics:**
- Clean, geometric forms
- Consistent stroke weight (2px at 24×24px)
- Rounded corners (2px radius)
- Optical balance over mathematical precision
- 24×24px base size with 2px padding (20×20px live area)

**Aesthetic:**
- Elegant and refined
- Modern but timeless
- Automotive-inspired where appropriate
- Matches brand voice (precise, premium, authentic)

---

## Icon Grid System

### Base Grid

**Grid:** 24×24px with 2px padding
**Live Area:** 20×20px (safe area for visual elements)
**Stroke Weight:** 2px (at base size)
**Corner Radius:** 2px (for rounded elements)

```
┌─────────────────────────┐
│ Padding (2px)           │
│  ┌───────────────────┐  │
│  │                   │  │
│  │   Live Area       │  │
│  │   (20×20px)       │  │
│  │                   │  │
│  └───────────────────┘  │
│                         │
└─────────────────────────┘
    24×24px total
```

### Scaling

Icons should scale proportionally:

| Size | Use Case | Stroke Weight |
|------|----------|---------------|
| 16×16px | Small UI elements, tight spaces | 1.5px |
| 24×24px | Default (buttons, navigation) | 2px |
| 32×32px | Larger touch targets | 2.5px |
| 48×48px | Hero icons, feature highlights | 3px |
| 64×64px | Large feature icons | 4px |

---

## Icon Categories

### 1. Navigation Icons

**Purpose:** Primary navigation, menu items, directional

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `home` | House outline | Simple peaked roof, centered |
| `search` | Magnifying glass | Circle + angled handle |
| `menu` | Hamburger (3 lines) | Evenly spaced horizontal lines |
| `close` | X mark | Two diagonal lines crossing |
| `arrow-right` | Right-pointing arrow | Simple arrow, 45° head angle |
| `arrow-left` | Left-pointing arrow | Mirror of arrow-right |
| `arrow-up` | Up-pointing arrow | Vertical arrow |
| `arrow-down` | Down-pointing arrow | Vertical arrow pointing down |
| `chevron-right` | Angled bracket > | Single line, 90° angle |
| `chevron-left` | Angled bracket < | Mirror of chevron-right |
| `chevron-up` | Angled bracket ^ | Pointing up |
| `chevron-down` | Angled bracket ∨ | Pointing down |

**Design Example (arrow-right):**
```
        →
  ───────────→
       (2px stroke, rounded cap)
```

---

### 2. Social Media Icons

**Purpose:** Social media links, share buttons

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `youtube` | YouTube logo | Play button in rounded rectangle |
| `instagram` | Instagram logo | Camera outline with circle center |
| `tiktok` | TikTok logo | Musical note OR stylized "T" |
| `twitter` | Twitter/X logo | Bird silhouette OR "X" |
| `facebook` | Facebook logo | Lowercase "f" |
| `email` | Envelope | Simple envelope outline |
| `link` | Chain link | Two connected ovals |
| `share` | Share icon | Three connected dots OR arrow from box |

**Note:** For social media icons, use official brand icons where possible. Create custom versions that match our line weight if needed.

---

### 3. E-Commerce Icons

**Purpose:** Shopping cart, checkout, account

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `cart` | Shopping cart | Classic cart with wheels |
| `bag` | Shopping bag | Simplified bag with handles |
| `heart` | Heart (wishlist) | Outline heart, symmetrical |
| `heart-fill` | Filled heart | Solid heart |
| `user` | User profile | Circle head + semicircle body |
| `login` | Login | Arrow entering rectangle |
| `logout` | Logout | Arrow exiting rectangle |
| `credit-card` | Payment | Rectangle with stripe |
| `package` | Shipping | Box outline |
| `check` | Success/confirm | Checkmark |
| `tag` | Price tag | Tag shape with hole |

---

### 4. Content Icons

**Purpose:** Video player, content types, actions

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `play` | Play button | Right-pointing triangle |
| `pause` | Pause button | Two vertical rectangles |
| `video` | Video icon | Play button in rectangle |
| `image` | Image icon | Mountain + sun outline |
| `text` | Text document | Document with lines |
| `download` | Download | Down arrow to horizontal line |
| `upload` | Upload | Up arrow from horizontal line |
| `edit` | Edit | Pencil at 45° angle |
| `delete` | Delete | Trash can outline |
| `star` | Star (rating) | 5-point star outline |
| `star-fill` | Filled star | Solid 5-point star |
| `flag` | Flag | Banner on pole |

---

### 5. Automotive Icons (Custom)

**Purpose:** Brand-specific, automotive context

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `car` | Car silhouette | Side view, generic sedan |
| `steering-wheel` | Steering wheel | Circle with spokes |
| `speedometer` | Gauge | Half circle with needle |
| `key` | Car key | Traditional key shape |
| `garage` | Garage | House with wide door |
| `road` | Road | Two parallel lines with perspective |
| `engine` | Engine | Simplified engine block outline |
| `wheel` | Wheel/tire | Circle with spokes or tread |
| `wrench` | Wrench | Simple wrench tool |
| `gas-pump` | Fuel pump | Classic pump silhouette |

**Design Example (steering-wheel):**
```
     ╱───╲
    │  ┼  │
     ╲───╱
  (Circle with center cross)
```

---

### 6. Utility Icons

**Purpose:** General UI elements, states, actions

| Icon Name | Description | Design Notes |
|-----------|-------------|--------------|
| `info` | Information | Circle with "i" |
| `warning` | Warning | Triangle with "!" |
| `error` | Error | Circle with "X" |
| `success` | Success | Circle with checkmark |
| `question` | Help | Circle with "?" |
| `settings` | Settings | Gear icon |
| `filter` | Filter | Funnel shape |
| `calendar` | Calendar | Grid with binding |
| `clock` | Time | Circle with clock hands |
| `location` | Location | Map pin |
| `phone` | Phone | Handset or smartphone |
| `eye` | View/visible | Eye outline |
| `eye-off` | Hidden | Eye with slash |

---

## Technical Specifications

### File Formats

**SVG (Primary):**
- Vector format, infinitely scalable
- Code-optimized (clean paths, no unnecessary metadata)
- ViewBox: `0 0 24 24`
- Fill: `currentColor` (inherits text color)
- Stroke: `currentColor` for outline icons

**PNG (Fallback):**
- Export at 1x, 2x, 3x for retina displays
- Sizes: 24px, 48px, 72px
- Transparent background
- Format: PNG-24

**Icon Font (Optional):**
- Useful for web applications
- Tools: IcoMoon, Fontello
- Format: WOFF2, WOFF, TTF

### SVG Code Example

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <line x1="5" y1="12" x2="19" y2="12"></line>
  <polyline points="12 5 19 12 12 19"></polyline>
</svg>
```

**Key Attributes:**
- `fill="none"` for outline icons
- `stroke="currentColor"` to inherit text color
- `stroke-width="2"` at 24×24px base size
- `stroke-linecap="round"` for rounded ends
- `stroke-linejoin="round"` for rounded corners

---

### Color Specifications

**Default:**
- Stroke/Fill: `currentColor` (inherits from parent element)

**Specific Colors (when needed):**

| Context | Color | Hex |
|---------|-------|-----|
| Default (dark mode) | Iconic Black | #0A0A0A |
| Default (light mode) | Pure White | #FFFFFF |
| Accent | Iconic Gold | #D4AF37 |
| Secondary | Slate Gray | #2D3436 |
| Disabled | Warm Gray | #636E72 |
| Error | Red | #E74C3C |
| Success | Green | #27AE60 |
| Warning | Yellow/Gold | #F39C12 |

---

### States

**Default:**
- Normal state, standard stroke/fill

**Hover:**
- Opacity: 80% OR
- Color shift to #D4AF37 (gold) for interactive icons

**Active/Selected:**
- Fill icon (if outline version) OR
- Color: #D4AF37 (gold)

**Disabled:**
- Opacity: 40% OR
- Color: #636E72 (warm gray)

---

## Usage Guidelines

### Do's

✅ **Use consistent stroke weight** across all icons in a view
✅ **Align icons to grid** for pixel-perfect rendering
✅ **Use appropriate size** for context (larger for primary actions)
✅ **Provide text labels** for ambiguous icons (accessibility)
✅ **Use filled variants** sparingly (for active/selected states)
✅ **Maintain padding** around icons (min 8px clickable area)

### Don'ts

❌ **Don't mix styles** (outline and filled in same context)
❌ **Don't rotate icons** arbitrarily (breaks visual consistency)
❌ **Don't use colors outside brand palette**
❌ **Don't scale non-proportionally** (will distort stroke weight)
❌ **Don't use low-resolution PNG** (use SVG whenever possible)
❌ **Don't create one-off custom icons** (maintain consistency)

---

## Implementation

### Web (HTML/CSS)

**Option 1: Inline SVG**
```html
<button>
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <!-- Icon paths here -->
  </svg>
  <span>Button Text</span>
</button>
```

**Option 2: SVG Sprite**
```html
<svg class="icon icon-arrow-right">
  <use xlink:href="#icon-arrow-right"></use>
</svg>
```

**Option 3: Icon Font**
```html
<i class="lii-icon lii-icon-arrow-right"></i>
```

### CSS Styling

```css
.icon {
  width: 24px;
  height: 24px;
  stroke: currentColor; /* Inherits text color */
  transition: all 0.2s ease;
}

.icon:hover {
  stroke: #D4AF37; /* Gold on hover */
}

.icon-sm {
  width: 16px;
  height: 16px;
}

.icon-lg {
  width: 32px;
  height: 32px;
}
```

### React Component Example

```jsx
import React from 'react';

export const Icon = ({ name, size = 24, color = 'currentColor', ...props }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <use href={`#icon-${name}`} />
    </svg>
  );
};

// Usage:
<Icon name="arrow-right" size={24} />
```

---

## Icon Library Structure

### File Organization

```
.brand/
└── assets/
    └── icons/
        ├── svg/
        │   ├── navigation/
        │   │   ├── home.svg
        │   │   ├── search.svg
        │   │   ├── menu.svg
        │   │   └── ...
        │   ├── social/
        │   │   ├── youtube.svg
        │   │   ├── instagram.svg
        │   │   └── ...
        │   ├── ecommerce/
        │   ├── content/
        │   ├── automotive/
        │   └── utility/
        ├── png/
        │   ├── 1x/ (24px)
        │   ├── 2x/ (48px)
        │   └── 3x/ (72px)
        └── sprite/
            └── icons-sprite.svg (combined sprite sheet)
```

### Naming Convention

**File Names:**
- Lowercase, hyphen-separated
- Descriptive, not cryptic
- Include variant if applicable

**Examples:**
```
arrow-right.svg
heart-outline.svg
heart-filled.svg
car-side-view.svg
user-circle.svg
```

---

## Design Tools

### Recommended Software

**For Creating Icons:**
- **Figma** (collaborative, free tier, web-based)
- **Adobe Illustrator** (professional standard)
- **Sketch** (Mac only, popular for UI design)
- **Affinity Designer** (one-time purchase alternative)

**For Optimization:**
- **SVGO** (command-line SVG optimizer)
- **SVGOMG** (web-based SVGO interface)
- **ImageOptim** (PNG optimization, Mac)
- **TinyPNG** (PNG compression, web-based)

### Creating Icons in Figma

**Process:**

1. **Create 24×24px frame** (base artboard)
2. **Add 2px padding** (guidelines at 2px from edges)
3. **Draw icon within 20×20px live area**
4. **Use 2px stroke weight**
5. **Apply 2px corner radius** to rounded elements
6. **Outline strokes** (convert to vector paths)
7. **Export as SVG** (optimize code)

**Export Settings:**
- SVG
- Outline text: Yes
- Simplify stroke: Yes
- Include "id" attribute: No

---

## Accessibility Considerations

### Semantic HTML

**Always provide accessible labels:**

```html
<!-- Good: Icon with text label -->
<button>
  <svg aria-hidden="true">...</svg>
  <span>Close</span>
</button>

<!-- Good: Icon with aria-label -->
<button aria-label="Close">
  <svg aria-hidden="true">...</svg>
</button>

<!-- Bad: Icon without label -->
<button>
  <svg>...</svg>
</button>
```

### ARIA Attributes

- Use `aria-hidden="true"` on decorative icons
- Use `aria-label` or visible text for functional icons
- Use `role="img"` and `aria-labelledby` for informational icons

### Color Contrast

- Ensure icon color has 3:1 contrast ratio with background (minimum)
- 4.5:1 contrast for small icons (under 24px)

---

## Quality Checklist

Before finalizing any icon, verify:

- [ ] **Grid alignment:** Icon centered in 24×24px artboard with 2px padding
- [ ] **Stroke weight:** Consistent 2px (or scaled proportionally)
- [ ] **Corner radius:** 2px on rounded elements
- [ ] **Optical balance:** Visually centered (not just mathematically)
- [ ] **Simplicity:** No unnecessary detail (clean and minimal)
- [ ] **Consistency:** Matches style of other icons in set
- [ ] **Scalability:** Readable at 16px, clear at 64px
- [ ] **File size:** SVG under 2KB (optimized)
- [ ] **Code cleanliness:** No unnecessary paths or metadata

---

**This icon system ensures visual consistency across all digital touchpoints for Live It Iconic.**

**Last Updated:** November 2024
**Next Review:** February 2025
